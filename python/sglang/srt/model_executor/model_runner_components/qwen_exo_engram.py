"""QWEN-EXO Engram runtime for one target model.

Owns the host-memory table, the reader weights and the per-request token ring
(see ``qwen_exo_booster.engram`` for the method). Built after the memory pools
and before CUDA-graph capture: ``forward_inputs`` prepares per-batch tensors in
``ForwardBatch.init_new`` and ``compute_addition`` runs inside the model's layer
loop, where it is captured into the decode / target-verify graphs.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
import torch

from qwen_exo_booster.engram import (
    EngramExtendInputs,
    EngramHashTensors,
    engram_knowledge_requested,
    engram_requested,
    extend_inputs_host,
    hash_rows,
    is_injectable_token,
    reader_delta,
    ring_update_and_read,
    ring_width_for,
)
from qwen_exo_booster.engram_artifact import (
    EngramManifest,
    HostEngramTable,
    SparseEngramTable,
    gather_rows_reference,
    load_reader_weights,
)
from sglang.jit_kernel.qwen_exo_engram import engram_gather_dequant
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_EXTEND_CHUNK_TOKENS = 4096


def drop_page_cache(paths: Sequence[str]) -> None:
    """Evict files (e.g. already-loaded model shards) from the page cache so the
    pinned table fits under the container memory limit."""
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


class QwenExoEngram:
    def __init__(
        self,
        *,
        engram_path: str,
        model: torch.nn.Module,
        model_path: str,
        num_req_slots: int,
        tokens_per_req: int,
        hidden_size: int,
        num_layers: int,
        device: str,
    ):
        from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

        text_model = model.model
        if not isinstance(text_model, Qwen3_5ForCausalLM):
            raise ValueError(f"--qwen-exo-engram-path supports Qwen3.5 text models, got {type(text_model).__name__}")
        if envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
            raise ValueError("--qwen-exo-engram-path needs a uniform verify width (SGLANG_RAGGED_VERIFY_MODE=static)")
        manifest = EngramManifest.load(engram_path)
        manifest.validate(hidden_size=hidden_size, num_layers=num_layers)
        drop_page_cache(glob.glob(os.path.join(model_path, "*.safetensors")))

        self.layer_index = manifest.reader.layer
        self.num_heads = manifest.hash.num_heads
        self.embed_dim = manifest.reader.embed_dim
        self.vocab_size = manifest.hash.vocab_size
        self.eos_id = manifest.hash.eos_id
        self.ring_width = ring_width_for(tokens_per_req)
        self.sparse = manifest.table.kind == "sparse_trained"
        if self.sparse:
            self.table = SparseEngramTable.load(manifest, device=device)
        else:
            self.table = HostEngramTable.load(manifest, pin=True)
        self.reader = load_reader_weights(manifest, device=device)
        self.knowledge_table = None
        self.knowledge_reader = None
        if manifest.knowledge_path:
            knowledge = EngramManifest.load(manifest.resolve(manifest.knowledge_path))
            knowledge.validate(hidden_size=hidden_size, num_layers=num_layers)
            if knowledge.knowledge_path or knowledge.table.kind != "sparse_trained":
                raise ValueError("Additional Engram knowledge must be one independently trained sparse table")
            if knowledge.hash != manifest.hash or knowledge.reader.layer != self.layer_index:
                raise ValueError("Additional Engram knowledge must share base hash and injection layer")
            self.knowledge_table = SparseEngramTable.load(knowledge, device=device)
            self.knowledge_reader = load_reader_weights(knowledge, device=device)
            logger.info("Engram knowledge %s loaded alongside base %s", knowledge.name, manifest.name)
        self.hash = EngramHashTensors(manifest.hash, device)
        self.ring = torch.full((num_req_slots, self.ring_width), self.eos_id, dtype=torch.int64, device=device)
        self.suppress = self._build_suppress_mask(model_path, device)
        self._self_test(manifest, device)
        text_model.attach_qwen_exo_engram(self)
        logger.info(
            "Engram %s attached at layer %d (ring %dx%d, %d heads x %d)",
            manifest.name,
            self.layer_index,
            num_req_slots,
            self.ring_width,
            self.num_heads,
            manifest.table.head_dim,
        )

    def _build_suppress_mask(self, model_path: str, device: str) -> torch.Tensor:
        """Per-vocab flag: inject the Engram delta only when the current token is
        a natural-language (letter/whitespace) token; suppress on digits,
        punctuation and markup, where it would only flip a prediction the model
        is already confident about (IP octets, versions, <tool_call>, code)."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for token_id in range(self.vocab_size):
            try:
                text = tokenizer.decode([token_id])
            except Exception:
                text = ""
            if not is_injectable_token(text):
                mask[token_id] = True
        logger.info("Engram: injecting on %d / %d letter tokens, suppressing the other %d",
                    self.vocab_size - int(mask.sum()), self.vocab_size, int(mask.sum()))
        return mask.to(device)

    def _self_test(self, manifest: EngramManifest, device: str) -> None:
        """Validate hash/gather and compile the host gather before graph capture."""
        if not manifest.check_tokens:
            raise ValueError("Engram manifest has no hash check rows")
        tokens = torch.tensor(manifest.check_tokens, dtype=torch.int64, device=device)
        rows = hash_rows(tokens[:, 0], tokens[:, 1], tokens[:, 2], self.hash)
        expected = torch.tensor(manifest.check_rows, dtype=torch.int64, device=device)
        if not torch.equal(rows, expected):
            raise RuntimeError("Engram hash self-check failed: rows differ from the manifest")
        gathered = self._gather(rows)
        if self.sparse:
            cpu_rows = rows.cpu()
            unique_rows = self.table.uniq.cpu()
            idx = torch.searchsorted(unique_rows, cpu_rows).clamp(max=unique_rows.numel() - 1)
            hit = (unique_rows[idx] == cpu_rows).unsqueeze(-1)
            reference = torch.where(hit, self.table.weight[idx], self.table.weight.new_zeros(())).flatten(-2)
        else:
            reference = gather_rows_reference(self.table.data, self.table.scale, rows).reshape(gathered.shape)
        if not torch.equal(gathered.cpu(), reference):
            raise RuntimeError("Engram gather self-check failed: host kernel differs from reference")
        if self.knowledge_table is not None:
            gathered = self.knowledge_table.gather(rows)
            reference = self.knowledge_table.gather(rows.cpu())
            if not torch.equal(gathered.cpu(), reference):
                raise RuntimeError("Engram knowledge gather differs from host reference")

    def _gather(self, rows: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            return self.table.gather(rows.reshape(-1, self.num_heads))
        out = torch.empty((rows.shape[0], self.embed_dim), dtype=torch.bfloat16, device=rows.device)
        engram_gather_dequant(
            table_data=self.table.data,
            table_scale=self.table.scale,
            rows=rows.reshape(-1),
            out=out.view(-1, self.embed_dim // self.num_heads),
            num_heads=self.num_heads,
        )
        return out

    def forward_inputs(
        self,
        *,
        reqs: Sequence[Req],
        forward_mode: ForwardMode,
        prefix_lens: Optional[Sequence[int]],
        extend_lens: Optional[Sequence[int]],
        device: torch.device,
    ) -> tuple[torch.Tensor, Optional[EngramExtendInputs]]:
        """Per-request on/off mask (always set while Engram is on, so CUDA-graph
        replays never see a stale mask) and, for EXTEND, exact token triples."""
        on = [engram_requested(req.sampling_params.custom_params) for req in reqs]
        knowledge_on = [
            self.knowledge_table is not None and engram_knowledge_requested(req.sampling_params.custom_params)
            for req in reqs
        ]
        mask = torch.tensor(list(zip(on, knowledge_on)), dtype=torch.bool).reshape(-1, 2).to(device, non_blocking=True)
        if forward_mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            return mask, None
        if forward_mode != ForwardMode.EXTEND:
            raise RuntimeError(f"Engram does not support forward mode {forward_mode.name}")
        host = extend_inputs_host(
            fill_ids=[req.full_untruncated_fill_ids for req in reqs],
            req_pool_indices=[req.req_pool_idx for req in reqs],
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            request_mask=on,
            ring_width=self.ring_width,
            vocab_size=self.vocab_size,
            eos_id=self.eos_id,
        )

        def to_device(array):
            return torch.from_numpy(array).pin_memory().to(device, non_blocking=True)

        return mask, EngramExtendInputs(
            tokens=to_device(host.tokens),
            token_mask=to_device(np.stack((host.token_mask, np.repeat(knowledge_on, extend_lens)), axis=1)),
            tail_rows=to_device(host.tail_rows),
            tail_cols=to_device(host.tail_cols),
            tail_tokens=to_device(host.tail_tokens),
        )

    def compute_addition(
        self,
        *,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[torch.Tensor]:
        """Gated Engram delta for the residual stream at ``layer_index``."""
        mask = forward_batch.qwen_exo_engram_mask
        if mask is None or hidden_states.shape[0] == 0:
            return None  # warmup / dummy forwards
        mode = forward_batch.forward_mode
        if mode.is_target_verify() or mode.is_decode():
            x0, x1, x2 = ring_update_and_read(
                ring=self.ring,
                req_pool_indices=forward_batch.req_pool_indices,
                positions=forward_batch.positions,
                input_ids=forward_batch.input_ids,
                eos_id=self.eos_id,
            )
            token_mask = mask.repeat_interleave(hidden_states.shape[0] // mask.shape[0], dim=0)
        else:
            extend = forward_batch.qwen_exo_engram_extend
            self.ring[extend.tail_rows, extend.tail_cols] = extend.tail_tokens
            x0, x1, x2 = extend.tokens
            token_mask = extend.token_mask
        h = hidden_states + residual
        deltas = []
        for start in range(0, h.shape[0], _EXTEND_CHUNK_TOKENS):
            end = start + _EXTEND_CHUNK_TOKENS
            rows = hash_rows(x0[start:end], x1[start:end], x2[start:end], self.hash)
            chunk = reader_delta(h[start:end], self._gather(rows), self.reader)
            chunk = chunk * token_mask[start:end, 0].unsqueeze(-1).to(chunk.dtype)
            if self.knowledge_table is not None:
                addition = reader_delta(
                    h[start:end], self.knowledge_table.gather(rows), self.knowledge_reader
                )
                chunk = chunk + addition * token_mask[start:end, 1].unsqueeze(-1).to(addition.dtype)
            deltas.append(chunk)
        delta = deltas[0] if len(deltas) == 1 else torch.cat(deltas)
        # Suppress injection where the current token is numeric / separator, so
        # an Engram nudge cannot flip an already-confident IP / version / hash.
        keep = ~self.suppress[x0]
        return delta * keep.unsqueeze(-1).to(delta.dtype)


__all__ = ["QwenExoEngram", "drop_page_cache"]
