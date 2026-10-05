"""Differentiable, frozen native Flash-Next text reference with sparse PLE deltas.

Main routed weights are decoded NVFP4 with BF16/F32 activations (W4A16).
This is a QLoRA-style reference, NOT bit-exact W4A4 serving. HF's native
GR/GDN/QSA/PLE remain intact; QSA's discrete index selection has its native
piecewise-constant gradient semantics. No SGLang, CUDA initialization, or
checkpoint/profile writes occur on import. Only the independent delta learns.
"""
from __future__ import annotations

import contextvars
import json
import math
import mmap
import struct
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity, SparseNativePLEDelta

_WINDOW = contextvars.ContextVar("native_ple_window", default=None)


@contextmanager
def _window_context(state):
    token = _WINDOW.set(state)
    try:
        yield
    finally:
        _WINDOW.reset(token)


def _checkpoint_contexts():
    # Non-reentrant checkpoints must replay the original forward's history/mode,
    # not whichever window the caller most recently selected.
    state = _WINDOW.get()
    return _window_context(state), _window_context(state)


def _hf_native():
    try:
        from transformers.models.qwen4_exp import modeling_qwen4_exp as native
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    except ImportError as exc:
        raise RuntimeError("Native PLE reference requires installed Transformers Qwen4Exp (verified 5.17)") from exc
    return native, Qwen4ExpTextConfig


class DiskNativePLERows(nn.Module):
    """Read only requested effective base rows; never materialize the full table."""

    def __init__(self, profile: str | Path, identity: NativePLEIdentity, *, dtype=torch.bfloat16):
        super().__init__()
        profile = Path(profile).resolve()
        marker = json.loads((profile / "native-ple.json").read_text(encoding="utf-8"))
        if marker.get("global_scale") != 1.0 or marker.get("schema") != 1:
            raise ValueError("External native PLE rows must be effective unscaled values")
        formats = {"fp8_e4m3_rowscale": ("F8_E4M3", 1, torch.float8_e4m3fn),
                   "bf16": ("BF16", 2, torch.bfloat16), "bfloat16": ("BF16", 2, torch.bfloat16),
                   "f16": ("F16", 2, torch.float16), "float16": ("F16", 2, torch.float16)}
        if marker.get("storage") not in formats:
            raise ValueError("Unsupported external native PLE row encoding")
        self.encoding, self.itemsize, self.source_dtype = formats[marker["storage"]]
        self.dtype = dtype
        self.num_embeddings = identity.num_embeddings
        self.embedding_dim = identity.head_dim
        self.rows_per_shard = int(marker["rows_per_shard"])
        table_root = Path(marker["root"])
        if not table_root.is_absolute():
            raise ValueError("Native PLE table root must be absolute")
        table_root = table_root.resolve()
        descriptors = marker["shards"]
        if (self.rows_per_shard <= 0 or marker["num_shards"] != len(descriptors)
                or marker["num_embeddings"] != self.num_embeddings
                or marker["embedding_dim"] != self.embedding_dim):
            raise ValueError("Native PLE manifest geometry mismatch")
        self.shards = []
        total = 0
        for i, desc in enumerate(descriptors):
            path = (table_root / desc["file"]).resolve()
            if not path.is_relative_to(table_root):
                raise ValueError("Native PLE shard escapes table root")
            if path.stat().st_size != desc["size"]:
                raise ValueError("Native PLE shard size differs from bound manifest")
            with path.open("rb") as source:
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated native PLE shard")
                size = struct.unpack("<Q", prefix)[0]
                if not 0 < size <= 128 * 1024**2:
                    raise ValueError("Invalid native PLE shard header")
                encoded = source.read(size)
                if len(encoded) != size:
                    raise ValueError("Truncated native PLE shard header")
                header = json.loads(encoded)
            rows = int(desc["rows"])
            if rows <= 0 or rows > self.rows_per_shard or (i < len(descriptors) - 1 and rows != self.rows_per_shard):
                raise ValueError("Native PLE shard row count mismatch")
            data = header[desc["data_tensor"]]
            start, end = data["data_offsets"]
            if (data["dtype"] != self.encoding or data["shape"] != [rows, self.embedding_dim]
                    or not 0 <= start < end or end - start != rows * self.embedding_dim * self.itemsize
                    or 8 + size + end > desc["size"]):
                raise ValueError("Native PLE data tensor differs from bound layout")
            scale_offset = None
            if self.encoding == "F8_E4M3":
                scale = header[desc["scale_tensor"]]
                ss, se = scale["data_offsets"]
                if (scale["dtype"] != "F32" or scale["shape"] not in ([rows], [rows, 1])
                        or not 0 <= ss < se or se - ss != rows * 4 or 8 + size + se > desc["size"]
                        or max(start, ss) < min(end, se)):
                    raise ValueError("Native PLE FP8 row-scale layout mismatch")
                scale_offset = 8 + size + ss
            self.shards.append((path, rows, 8 + size + start, scale_offset))
            total += rows
        if total != self.num_embeddings:
            raise ValueError("Native PLE shards do not cover the table")
        # Native HF probes weight.device. A zero-storage meta placeholder keeps
        # it from relocating this read-only external table to the accelerator.
        self.register_buffer("weight", torch.empty((self.num_embeddings, self.embedding_dim), device="meta", dtype=dtype),
                             persistent=False)

    def forward(self, row_ids: torch.Tensor) -> torch.Tensor:
        ids = row_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if bool(((ids < 0) | (ids >= self.num_embeddings)).any()):
            raise ValueError("Native PLE row address is outside the base table")
        unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        values = torch.empty((unique.numel(), self.embedding_dim), dtype=self.dtype)
        shard_ids = unique // self.rows_per_shard
        for shard_index in torch.unique(shard_ids).tolist():
            positions = torch.where(shard_ids == shard_index)[0]
            local = (unique[positions] - shard_index * self.rows_per_shard).numpy()
            path, rows, offset, scale_offset = self.shards[shard_index]
            with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                raw = np.ndarray((rows, self.embedding_dim * self.itemsize), dtype=np.uint8, buffer=mapping, offset=offset)
                scales = None if scale_offset is None else np.ndarray((rows,), dtype="<f4", buffer=mapping, offset=scale_offset)
                # Bound temporary conversion memory independently of source-table size.
                for begin in range(0, positions.numel(), 4096):
                    stop = begin + 4096
                    selected = local[begin:stop]
                    decoded = torch.from_numpy(raw[selected].copy()).view(self.source_dtype).float()
                    if scales is not None:
                        row_scale = torch.from_numpy(scales[selected].copy()).unsqueeze(-1)
                        if not bool(torch.isfinite(row_scale).all()):
                            raise ValueError("Native PLE row scale is nonfinite")
                        decoded = decoded * row_scale
                    if not bool(torch.isfinite(decoded).all()):
                        raise ValueError("Native PLE base row is nonfinite")
                    values[positions[begin:stop]] = decoded.to(self.dtype)
                del raw, scales
        return values[inverse].reshape(*row_ids.shape, self.embedding_dim).to(row_ids.device).detach()

    def close(self):
        # Read mappings/descriptors are context-managed per shard lookup.
        return None


class _DeltaRows(nn.Module):
    def __init__(self, base: nn.Module, delta: SparseNativePLEDelta):
        super().__init__()
        self.base = base
        self.delta = delta

    @property
    def weight(self):
        return self.base.weight

    def forward(self, row_ids):
        state = _WINDOW.get()
        mode = "real" if state is None else state[1]
        base = self.base(row_ids).detach()
        if row_ids.device != self.delta.weight.device:
            raise ValueError("Native PLE rows and sparse delta must execute on the same device")
        if mode == "shuffled" and state is not None:
            # Freeze the permutation per forward too: selecting another shuffle
            # seed before backward must not alter checkpoint recomputation.
            keys = row_ids.long().clone()
            indices = torch.searchsorted(self.delta.row_ids, keys).clamp(max=self.delta.row_ids.numel() - 1)
            hit = self.delta.row_ids[indices] == keys
            keys[hit] = self.delta.row_ids[state[2][indices[hit]]]
            return base + self.delta(keys, mode="real").to(base.dtype)
        return base + self.delta(row_ids.long(), mode=mode).to(base.dtype)


def _install_seeded_ngram(ngram, native):
    class SeededNativeNGram(type(ngram)):
        def forward(self, input_ids, past_key_values=None):
            if past_key_values is not None:
                raise ValueError("Native PLE training reference never retains a cache")
            state = _WINDOW.get()
            history = None if state is None else state[0]
            if history is None:
                return super().forward(input_ids, None)
            history = history.to(device=input_ids.device, dtype=torch.int64)
            if history.ndim == 1:
                history = history.unsqueeze(0)
            if history.shape != (input_ids.shape[0], self.context_len):
                raise ValueError("PLE window history must contain exactly two tokens per batch item")
            if bool(((history < 0) | (history >= self.unigram_vocab_size)).any()):
                raise ValueError("PLE history is outside the exact tokenizer vocabulary")
            full = torch.cat((history, input_ids.long()), dim=1)
            return super().forward(full, None)[:, -input_ids.shape[1]:]
    ngram.__class__ = SeededNativeNGram


class _ChunkedFrozenHeadCE(torch.autograd.Function):
    """Retain hidden states, never a sequence-by-vocabulary activation graph."""
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_size):
        if (hidden.ndim != 3 or labels.shape != hidden.shape[:2] or labels.dtype != torch.int64
                or weight.ndim != 2 or weight.shape[1] != hidden.shape[2]
                or weight.requires_grad or hidden.dtype != weight.dtype
                or hidden.device != weight.device or labels.device != hidden.device
                or hidden.dtype not in (torch.float32, torch.float64, torch.bfloat16)
                or not isinstance(chunk_size, int) or chunk_size < 1):
            raise ValueError("chunked_ce_contract_invalid")
        shifted = labels[:, 1:]
        if bool(((shifted != -100) & ((shifted < 0) | (shifted >= weight.shape[0]))).any()):
            raise ValueError("chunked_ce_label_invalid")
        positions = (shifted.reshape(-1) != -100).nonzero().flatten()
        if not positions.numel():
            raise ValueError("chunked_ce_no_targets")
        # Mapping flattened causal positions into the original unshifted hidden tensor.
        length = hidden.shape[1]
        source = positions // (length - 1) * length + positions % (length - 1)
        target = shifted.reshape(-1)[positions]
        flat = hidden.reshape(-1, hidden.shape[-1])
        accumulation = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        total = torch.zeros((), device=hidden.device, dtype=accumulation)
        for begin in range(0, source.numel(), chunk_size):
            index = source[begin:begin + chunk_size]
            logits = torch.nn.functional.linear(flat[index], weight).to(accumulation)
            total += torch.nn.functional.cross_entropy(logits, target[begin:begin + chunk_size], reduction="sum")
        ctx.save_for_backward(hidden, weight, source, target)
        ctx.chunk_size = chunk_size
        ctx.accumulation = accumulation
        return total / source.numel()

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, source, target = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        grad_hidden = torch.zeros_like(flat)
        for begin in range(0, source.numel(), ctx.chunk_size):
            index = source[begin:begin + ctx.chunk_size]
            logits = torch.nn.functional.linear(flat[index], weight).to(ctx.accumulation)
            derivative = logits.softmax(dim=-1)
            derivative[torch.arange(index.numel(), device=index.device), target[begin:begin + ctx.chunk_size]] -= 1
            derivative *= grad_output.to(ctx.accumulation) / source.numel()
            # Match native linear backward: CE's F32 derivative casts to the
            # activation dtype before multiplying the frozen head.
            grad_hidden[index] = derivative.to(hidden.dtype) @ weight
        return grad_hidden.reshape_as(hidden), None, None, None


def chunked_causal_cross_entropy(hidden, frozen_head_weight, labels, chunk_size=128):
    """Mean teacher-forced causal CE, ignoring -100, with bounded head workspace."""
    return _ChunkedFrozenHeadCE.apply(hidden, frozen_head_weight, labels, chunk_size)


def _reference_class(native):
    class NativePLETrainingModel(native.Qwen4ExpForCausalLM):
        def set_ple_window(self, history=None, *, mode="real", shuffle_seed=0):
            if mode not in {"off", "real", "shuffled"}:
                raise ValueError("PLE mode must be off, real or shuffled")
            if mode == "shuffled":
                self.ple_delta.set_shuffle_seed(shuffle_seed)
            self._ple_window = (None if history is None else history.detach().clone(), mode)

        @property
        def ple_delta(self):
            return self.model.layers[self.ple_identity.layer_id].ple.ple_embedding.ngram_embedding.delta

        def forward(self, *args, ple_history=None, ple_mode=None, **kwargs):
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache"):
                raise ValueError("Native PLE training reference requires use_cache=False and fresh state")
            kwargs["use_cache"] = False
            default_history, default_mode = self._ple_window
            mode = default_mode if ple_mode is None else ple_mode
            if mode not in {"off", "real", "shuffled"}:
                raise ValueError("PLE mode must be off, real or shuffled")
            history = default_history if ple_history is None else ple_history.detach().clone()
            permutation = self.ple_delta.shuffle_indices.detach().clone() if mode == "shuffled" else None
            with _window_context((history, mode, permutation)):
                return super().forward(*args, **kwargs)

        def forward_hidden(self, input_ids, *, ple_history=None, ple_mode=None):
            """Exact native final HC mixer output, without materializing logits."""
            default_history, default_mode = self._ple_window
            history = default_history if ple_history is None else ple_history.detach().clone()
            mode = default_mode if ple_mode is None else ple_mode
            if mode not in {"off", "real", "shuffled"}:
                raise ValueError("PLE mode must be off, real or shuffled")
            permutation = self.ple_delta.shuffle_indices.detach().clone() if mode == "shuffled" else None
            with _window_context((history, mode, permutation)):
                return self.model(input_ids=input_ids, use_cache=False,
                                  output_router_logits=False).last_hidden_state

        def close(self):
            if hasattr(self, "native_checkpoint"):
                self.native_checkpoint.close()
            self.model.layers[self.ple_identity.layer_id].ple.ple_embedding.ngram_embedding.base.close()
    return NativePLETrainingModel


def _freeze_reference(model, delta, gradient_checkpointing):
    model.requires_grad_(False)
    delta.weight.requires_grad_(True)
    model.config.use_cache = False
    model._ple_window = (None, "real")
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "context_fn": _checkpoint_contexts})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) != 1 or trainable[0] is not delta.weight:
        raise ValueError("Native PLE reference must train only the separate sparse delta")


def load_native_ple_training_model(profile, delta, device="cpu", dtype=torch.bfloat16,
                                   gradient_checkpointing=True, metadata_only=False):
    """Strictly load the exact native text checkpoint, with bounded frozen experts.

    Vision and MTP are explicitly excluded. Ordinary checkpoint dtypes/shapes
    are validated before placement; all text keys must be consumed. No random
    missing-weight fallback and no generic HF quantizer bypass are used.
    """
    from qwen_exo_booster.native_ple_checkpoint import IndexedCheckpoint, FrozenCheckpointExperts
    native, config_type = _hf_native()
    root = Path(profile).resolve()
    identity = NativePLEIdentity.from_profile(root)
    if delta.identity != identity:
        raise ValueError("Sparse delta does not match the exact profile/reader/base table")
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Native reference activations must be BF16 or F32")
    if delta.weight.device != torch.device(device):
        raise ValueError("Place sparse delta on the requested reference device first")
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config = config_type(**raw["text_config"])
    config._attn_implementation = "sdpa" if torch.device(device).type == "cuda" else "eager"
    config.use_cache = False
    checkpoint = IndexedCheckpoint(root)
    try:
        with torch.device("meta"):
            model = _reference_class(native)(config)
        model.ple_identity = identity
        consumed = set()
        names = set(checkpoint.tensor_names())
        for i, layer in enumerate(model.model.layers):
            prefix = f"model.language_model.layers.{i}.mlp.experts"
            layer.mlp.experts = FrozenCheckpointExperts(checkpoint, prefix, config.num_experts,
                                                        config.hidden_size, config.moe_intermediate_size,
                                                        activation=config.hidden_act)
            expected = {f"{prefix}.{expert}.{projection}.{suffix}"
                        for expert in range(config.num_experts)
                        for projection in ("gate_proj", "up_proj", "down_proj")
                        for suffix in ("weight", "weight_scale", "weight_scale_2", "input_scale")}
            actual = {name for name in names if name.startswith(prefix + ".")}
            if actual != expected:
                raise ValueError(f"Frozen expert tensor inventory mismatch in layer {i}")
            consumed.update(expected)
        ngram = model.model.layers[identity.layer_id].ple.ple_embedding
        ngram.ngram_embedding = _DeltaRows(DiskNativePLERows(root, identity, dtype=dtype), delta)
        _install_seeded_ngram(ngram, native)
        # Only these two nonpersistent buffers are generated from config. Never
        # replace persistent checkpoint tensors with initializer values.
        model.model.rotary_emb = native.Qwen4ExpTextRotaryEmbedding(config).to(device)
        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        delta_prefix = f"model.layers.{identity.layer_id}.ple.ple_embedding.ngram_embedding.delta."
        generated = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}
        placeholder = f"model.layers.{identity.layer_id}.ple.ple_embedding.ngram_embedding.base.weight"
        for target, value in {**parameters, **buffers}.items():
            if target.startswith(delta_prefix) or target in generated or target == placeholder:
                continue
            source = "model.language_model." + target[6:] if target.startswith("model.") else target
            if source not in names:
                raise ValueError(f"Missing native text tensor: {source}")
            info = checkpoint.metadata(source)
            if tuple(info["shape"]) != tuple(value.shape):
                raise ValueError(f"Native text tensor shape mismatch: {source}")
            if info["dtype"] not in {"BF16", "F16", "F32", "I64"}:
                raise ValueError(f"Unsupported ordinary native text tensor dtype: {source}")
            if value.is_floating_point() != (info["dtype"] != "I64"):
                raise ValueError(f"Native text tensor dtype category mismatch: {source}")
            if metadata_only and not (".ple.ple_embedding." in target and value.dtype == torch.int64):
                consumed.add(source)
                continue
            tensor = checkpoint.read(source)
            if not tensor.is_floating_point() and tensor.dtype != torch.int64:
                raise ValueError(f"Native integer tensor dtype mismatch: {source}")
            # PLE hash buffers are authoritative and must match the identity,
            # not merely have the expected integer dtype and dimensions.
            hash_expected = {"layer_multipliers": identity.hash.multipliers,
                             "ngram_heads_offsets": identity.hash.head_offsets,
                             "ngram_heads_vocab_sizes": identity.hash.head_sizes}
            if target.rsplit(".", 1)[-1] in hash_expected and ".ple.ple_embedding." in target:
                expected_values = hash_expected[target.rsplit(".", 1)[-1]]
                if tensor.tolist() != list(expected_values):
                    raise ValueError(f"Native PLE hash buffer differs from identity: {source}")
            if metadata_only:
                consumed.add(source)
                continue
            tensor = tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
            module_name, leaf = target.rsplit(".", 1)
            module = model.get_submodule(module_name)
            if target in parameters:
                module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            else:
                module._buffers[leaf] = tensor
            consumed.add(source)
        excluded = {name for name in names if name.startswith(("model.visual.", "mtp."))}
        unexpected = names - consumed - excluded
        if unexpected:
            raise ValueError(f"Unexpected/unconsumed native checkpoint tensors: {sorted(unexpected)[:8]}")
        meta = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
                if tensor.device.type == "meta" and name != placeholder]
        if meta and not metadata_only:
            raise ValueError(f"Unmaterialized native reference tensors: {meta[:8]}")
        model.native_checkpoint = checkpoint
        model.native_load_report = {"ordinary_tensors": len(consumed) - config.num_hidden_layers * config.num_experts * 12,
                                    "excluded_vision_mtp_tensors": len(excluded),
                                    "mapping": "model.language_model.* -> model.*; lm_head unchanged",
                                    "weight_reference": "frozen NVFP4 decoded W4A16, not serving W4A4",
                                    "hf_source": native.__file__}
        if metadata_only:
            report = {**model.native_load_report, "metadata_only": True,
                      "ordinary_weight_bytes": sum(math.prod(checkpoint.metadata(name)["shape"])
                          * {"BF16": 2, "F16": 2, "F32": 4, "I64": 8}[checkpoint.metadata(name)["dtype"]]
                          for name in consumed if ".mlp.experts." not in name),
                      "backbone_loaded": False, "training_started": False}
            ngram.ngram_embedding.base.close()
            checkpoint.close()
            return report
        _freeze_reference(model, delta, gradient_checkpointing)
        return model
    except BaseException:
        checkpoint.close()
        raise


def verify_tiny_native_ple_backward():
    """CPU-only packed-checkpoint -> native model -> sparse delta backward gate.

    Synthetic fixture only: creates tiny ModelOpt-layout NVFP4 experts and a
    separate bound BF16 PLE table, then exercises the actual strict loader.
    No optimizer steps, downloads, GPU calls, or production files are involved.
    Parent invokes once after integration.
    """
    import hashlib
    import tempfile
    import msgspec
    from safetensors.torch import save_file
    from qwen_exo_booster.engram import EngramHashSpec
    from qwen_exo_booster.native_ple_checkpoint import FrozenCheckpointExperts
    from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys, sha256_file
    native, config_type = _hf_native()
    config = config_type(vocab_size=64, hidden_size=16, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=3,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=3, num_experts_per_tok=2, hc_count=4, hc_lowrank=8,
        layer_types=["linear_attention", "linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[2], ple_embed_dim=16, heads_per_ngram=2,
        ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=1,
        ple_conv_kernel_size=2, eos_token_id=63, pad_token_id=None,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
        output_gate_type="sigmoid", use_cache=False)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(193)
        source_model = native.Qwen4ExpForCausalLM(config).float().cpu()
    ngram = source_model.model.layers[1].ple.ple_embedding
    spec = EngramHashSpec(vocab_size=config.vocab_size, eos_id=config.eos_token_id,
        ngram_size=3, heads_per_order=2, multipliers=tuple(ngram.layer_multipliers.tolist()),
        head_sizes=tuple(ngram.head_vocab_sizes), head_offsets=tuple(ngram.head_offsets))

    def encode_expert(weight):
        # ModelOpt NVFP4 byte layout: even input column is low nibble, odd is
        # high; each 16-column block has E4M3 scale plus an independent F32
        # global scale. Small fixture quantization is deliberate, not training.
        weight = weight.detach().float().contiguous()
        blocks = weight.reshape(weight.shape[0], -1, 16)
        global_scale = torch.tensor(0.5, dtype=torch.float32)
        scales = (blocks.abs().amax(-1) / (6 * global_scale)).clamp_min(2**-8).to(torch.float8_e4m3fn)
        normalized = blocks / (scales.float().unsqueeze(-1) * global_scale)
        levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                              -0., -.5, -1., -1.5, -2., -3., -4., -6.])
        codes = (normalized.unsqueeze(-1) - levels).abs().argmin(-1).reshape_as(weight).to(torch.uint8)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return {"weight": packed.contiguous(), "weight_scale": scales.contiguous(),
                "weight_scale_2": global_scale, "input_scale": torch.tensor(0.25, dtype=torch.float32)}

    with tempfile.TemporaryDirectory(prefix="native-ple-packed-cpu-") as temporary:
        root = Path(temporary)
        table_root = root / "base-ple"
        table_root.mkdir()
        table_path = table_root / "shard_0.safetensors"
        base_rows = ngram.ngram_embedding.weight.detach().to(torch.bfloat16).contiguous()
        save_file({"data": base_rows}, str(table_path))
        manifest = {"schema": 1, "global_scale": 1.0, "storage": "bf16",
            "root": str(table_root.resolve()), "num_embeddings": base_rows.shape[0],
            "embedding_dim": base_rows.shape[1], "num_shards": 1,
            "rows_per_shard": base_rows.shape[0],
            "hash": msgspec.to_builtins(spec),
            "shards": [{"file": table_path.name, "data_tensor": "data", "rows": base_rows.shape[0],
                        "size": table_path.stat().st_size, "sha256": sha256_file(table_path)}]}
        marker = root / "native-ple.json"
        marker.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        raw_config = {"architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp", "text_config": config.to_dict(),
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            "qwen_exo_native_ple": {"schema": 1, "manifest": marker.name,
                                    "manifest_sha256": sha256_file(marker)}}
        (root / "config.json").write_text(json.dumps(raw_config, sort_keys=True), encoding="utf-8")
        tensors = {}
        for name, tensor in source_model.state_dict().items():
            if ".mlp.experts." in name or name.endswith(".ngram_embedding.weight"):
                continue
            source_name = "model.language_model." + name[6:] if name.startswith("model.") else name
            tensors[source_name] = tensor.detach().contiguous()
        for layer_index, layer in enumerate(source_model.model.layers):
            experts = layer.mlp.experts
            for expert_index in range(config.num_experts):
                gate, up = experts.gate_up_proj[expert_index].chunk(2, dim=0)
                for projection, weight in (("gate_proj", gate), ("up_proj", up),
                                           ("down_proj", experts.down_proj[expert_index])):
                    prefix = f"model.language_model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}"
                    for suffix, tensor in encode_expert(weight).items():
                        tensors[f"{prefix}.{suffix}"] = tensor
        shard_path = root / "model.safetensors"
        save_file(tensors, str(shard_path))
        (root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: shard_path.name for name in tensors}}, sort_keys=True), encoding="utf-8")
        del source_model, experts, gate, up, weight, tensors, base_rows, ngram
        identity = NativePLEIdentity.from_profile(root)
        ids = torch.tensor([[1, 4, 7, 63, 2, 5, 8, 9]], dtype=torch.int64)
        history = torch.tensor([[11, 12]], dtype=torch.int64)
        rows = native_ple_row_keys(ids[0], identity, history[0])
        delta = SparseNativePLEDelta(torch.unique(rows), identity)
        model = load_native_ple_training_model(root, delta, device="cpu", dtype=torch.float32,
                                               gradient_checkpointing=True)
        try:
            if not all(isinstance(layer.mlp.experts, FrozenCheckpointExperts) for layer in model.model.layers):
                raise AssertionError("Tiny loader retained dense original experts")
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if len(trainable) != 1 or trainable[0] is not delta.weight:
                raise AssertionError("Mixed native loader did not freeze every original weight")

            def artifact_digest():
                return {str(path.relative_to(root)): sha256_file(path)
                        for path in root.rglob("*") if path.is_file()}

            def frozen_digest():
                digest = hashlib.sha256()
                for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                    if ".ngram_embedding.delta." in name or tensor.device.type == "meta":
                        continue
                    digest.update(name.encode())
                    digest.update(tensor.detach().cpu().numpy().tobytes())
                return digest.hexdigest()

            original_artifacts = artifact_digest()
            original_tensors = frozen_digest()
            with torch.no_grad():
                baseline = model(ids, ple_history=history, ple_mode="off").logits.detach()
                # Make off parity meaningful: a nonzero fixture control delta
                # must remain disabled, but real mode must affect the reference.
                delta.weight.copy_(torch.arange(delta.weight.numel(), dtype=torch.float32).reshape_as(delta.weight).sin() * .01)
                off = model(ids, ple_history=history, ple_mode="off").logits.detach()
                active = model(ids, ple_history=history, ple_mode="real").logits.detach()
                active_difference = float((active - baseline).abs().max())
                delta.weight.zero_()
            if not torch.equal(off, baseline):
                raise AssertionError("Disabled delta changed packed native-reference outputs")
            if active_difference <= 0 or not bool(torch.isfinite(active).all()):
                raise AssertionError("Nonzero delta did not affect the packed native reference")
            result = model(ids, labels=ids.clone(), ple_history=history, ple_mode="real")
            # A different window selection before backward tests checkpoint
            # replay capture, without another forward or retained native cache.
            model.set_ple_window(torch.tensor([[20, 21]]), mode="off")
            result.loss.backward()
            grad = delta.weight.grad
            if grad is None or not grad.is_sparse:
                raise AssertionError("Packed native causal loss did not reach sparse PLE delta")
            values = grad.coalesce().values()
            if not bool(torch.isfinite(result.loss)) or not bool(torch.isfinite(values).all()) or not bool((values != 0).any()):
                raise AssertionError("Packed native causal loss->delta gradient is zero/nonfinite")
            expected_gradient = grad.coalesce().to_dense()
            delta.weight.grad = None
            hidden = model.forward_hidden(ids, ple_history=history, ple_mode="real")
            chunked = chunked_causal_cross_entropy(hidden, model.lm_head.weight, ids.clone(), chunk_size=3)
            model.set_ple_window(torch.tensor([[30, 31]]), mode="off")
            chunked.backward()
            actual_gradient = delta.weight.grad.coalesce().to_dense()
            torch.testing.assert_close(chunked.detach(), result.loss.detach(), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=1e-7)
            chunked_gradient_error = float((actual_gradient - expected_gradient).abs().max())
            if any(p.grad is not None for p in model.parameters() if p is not delta.weight):
                raise AssertionError("Original packed-reference weight acquired a gradient")
            if frozen_digest() != original_tensors or artifact_digest() != original_artifacts:
                raise AssertionError("Original native tensors or checkpoint/base-table artifacts changed")
            if not torch.equal(delta.weight.detach(), torch.zeros_like(delta.weight)):
                raise AssertionError("Backward-only gate mutated the independent delta")
            return {"loss": float(result.loss.detach()), "delta_gradient_abs_max": float(values.abs().max()),
                    "gradient_rows": int(grad.coalesce().indices().shape[1]), "off_parity_exact": True,
                    "chunked_ce_loss_difference": float((chunked.detach() - result.loss.detach()).abs()),
                    "chunked_ce_delta_gradient_max_difference": chunked_gradient_error,
                    "nonzero_delta_control_max_difference": active_difference,
                    "original_gradients": 0, "original_weights_unchanged": True,
                    "original_artifact_bytes_unchanged": True,
                    "streamed_expert_layers": len(model.model.layers),
                    "fixture_loader": "load_native_ple_training_model",
                    "fixture_weight_encoding": "ModelOpt NVFP4 packed E2M1/E4M3+F32 scales",
                    "optimizer_steps": 0, "device": "cpu", "native_layers": ["GR4", "GDN", "QSA", "PLE"],
                    "hf_source": native.__file__}
        finally:
            model.close()
