"""MLX-specific TpModelWorker subclass for Apple Silicon.

Routes forward passes through the MLX model runner, bypassing PyTorch
MPS.  A lightweight stub provides scheduler bookkeeping; the actual
attention KV data lives in MlxAttentionKVPool.

The worker also exposes an async (lazy-eval) surface used by the MLX
overlap scheduler: ``async_forward_batch_generation_mlx`` launches a
batch without blocking on the GPU, ``async_chained_decode_mlx`` builds
the next decode step on top of a still-lazy previous decode, and
``finalize_mlx_result`` blocks on the lazy outputs and produces a
normal ``GenerationBatchResult``.
"""

import logging
from typing import Optional, Union

import mlx.core as mx
import torch

from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxPendingDecode,
    MlxPendingExtend,
    MlxPendingPrefill,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)

logger = logging.getLogger(__name__)


class MlxTpModelWorker(TpModelWorker):
    """A tensor parallel model worker that routes inference through MLX.

    Inherits from TpModelWorker for scheduler integration, but replaces
    the standard ModelRunner with MlxModelRunnerStub (no PyTorch weights,
    zero-memory KV cache) and delegates all forward passes to a native
    MlxModelRunner.
    """

    def _init_model_runner(self):
        """Create MLX runner first (auto-sizes pool), then stub with matching size."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        init_kwargs = dict(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            disable_radix_cache=self.server_args.disable_radix_cache,
            mem_fraction_static=self.server_args.mem_fraction_static,
            quantization=self.server_args.quantization,
        )
        if self.server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = self.server_args.max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            mlx_pool_size=self._mlx_runner.pool_size,
        )

        self._mlx_active_rids: set[str] = set()
        self._mlx_pool_initialized = False

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

    def _ensure_mlx_pool_initialized(self):
        """Lazily initialize MLX cache pools after the stub pools are ready."""
        if not self._mlx_pool_initialized:
            self._mlx_runner.init_cache_pools(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
            capture_hidden_mode=capture_hidden_mode,
        )

    def _cleanup_stale_rids(self, forward_mode, current_rids: set[str]) -> None:
        """Remove MLX state for decode-mode requests that dropped out of the batch."""
        if forward_mode.is_decode():
            stale_rids = self._mlx_active_rids - current_rids
            for rid in stale_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids = current_rids
        else:
            self._mlx_active_rids |= current_rids

    def prepare_for_kv_cache_release(self, req) -> None:
        """Snapshot MLX auxiliary state at the scheduler's radix insert point."""
        if self._mlx_runner.has_request(req.rid):
            self._mlx_runner.store_auxiliary_state_for_request(req.rid)
            # Prefer the just-snapshotted live auxiliary state for the final
            # insert. Any older tracked slot is released during component cleanup.
            req.mamba_last_track_seqlen = None

    def _route_extend_request(self, rid: str, decoding_rids: set[str]) -> str:
        """Classify a request within an extend / mixed batch.

        Shared by the sync (:meth:`_forward_batch_generation_mlx`) and async
        (:meth:`_async_extend_batch`) paths so both route identically.

        Returns one of:

        * ``"prefill"``      -- not seen before; start a fresh prefill.
        * ``"decode"``       -- a genuine single-token decode step mixed into
          this batch (present in ``batch.decoding_reqs``).
        * ``"continuation"`` -- a chunked-prefill continuation.  Routing keys on
          request state, **not** ``seq_len``: a final continuation chunk can be
          exactly one token, which must still extend.  Routing it as a decode
          would drop the real token and feed the model its own previous-chunk
          prediction, silently corrupting the output.
        """
        if not self._mlx_runner.has_request(rid):
            return "prefill"
        if rid in decoding_rids:
            return "decode"
        return "continuation"

    def _forward_batch_generation_mlx(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Run one MLX batch through the shared lazy launch/finalize path."""

        _lazy, prefills, extends, decode, mode = (
            self.async_forward_batch_generation_mlx(batch)
        )
        return self.finalize_mlx_result(
            prefills,
            extends,
            decode,
            mode,
            list(batch.reqs),
        )

    def async_forward_batch_generation_mlx(
        self, batch: ScheduleBatch
    ) -> tuple[
        Union[mx.array, None],
        list[MlxPendingPrefill],
        list[MlxPendingExtend],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Start an async (lazy) forward pass through the MLX model runner.

        Returns ``(lazy_result, prefills, extends, decode, mode)``:

        * ``lazy_result`` — an ``mx.array`` that, when evaluated, forces
          materialisation of the whole batch's outputs.  ``None`` for
          idle batches.
        * ``prefills`` — list of :class:`MlxPendingPrefill` for new
          requests in an extend batch.
        * ``extends`` — list of :class:`MlxPendingExtend` for chunked
          prefill continuations in an extend batch.
        * ``decode`` — :class:`MlxPendingDecode` for the decode
          sub-batch (covers full decode mode AND mixed decodes inside
          an extend batch).
        * ``mode`` — one of ``"idle"``, ``"decode"``, ``"extend"``.

        The caller must make sure the returned pendings are fed into a
        subsequent ``mx.async_eval`` or ``.item()`` / ``.tolist()`` call
        — :meth:`finalize_mlx_result` does that.
        """
        self._ensure_mlx_pool_initialized()

        forward_mode = batch.forward_mode
        reqs = batch.reqs

        if forward_mode.is_idle():
            return None, [], [], None, "idle"

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        if forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            pending_decode = self._mlx_runner.decode_batch_start(
                req_ids,
                reqs=list(reqs),
            )
            mx.async_eval(
                pending_decode.lazy_tokens,
                *self._mlx_runner.sampling_arrays(pending_decode.sampling),
                *(pending_decode.customized_info or {}).values(),
            )
            return pending_decode.lazy_tokens, [], [], pending_decode, "decode"

        if forward_mode.is_extend():
            # TODO (changminbark): Implement per-batch flushing using prefix_slot_ids
            # Ensure the pool is up-to-date before pool-backed attention
            # reads it for prefix-cached prefills. Mirror the sync path.
            self._mlx_runner.flush_all_decode_kv()
            return self._async_extend_batch(batch)

        raise ValueError(
            f"MLX async runner does not support forward mode: {forward_mode}"
        )

    def _async_extend_batch(
        self, batch: ScheduleBatch
    ) -> tuple[
        Union[mx.array, None],
        list[MlxPendingPrefill],
        list[MlxPendingExtend],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Launch each request in an EXTEND batch lazily and kick GPU work."""
        reqs = batch.reqs
        input_ids_cpu = batch.input_ids.cpu().tolist()
        out_cache_loc_cpu = batch.out_cache_loc.cpu().tolist()
        extend_seq_lens = batch.extend_lens
        logprob_starts = batch.extend_logprob_start_lens or list(extend_seq_lens)
        final_prefill_flags = batch.qwen_exo_final_prefill or [True] * len(reqs)

        offset = 0
        slot_offset = 0
        pending_prefills: list[MlxPendingPrefill] = []
        pending_extends: list[MlxPendingExtend] = []
        mixed_decode_rids: list[str] = []
        mixed_decode_reqs: list = []
        # Genuine decode steps mixed into this extend batch; see
        # _route_extend_request.
        decoding_rids = {r.rid for r in (batch.decoding_reqs or [])}

        for i, req in enumerate(reqs):
            seq_len = extend_seq_lens[i]
            req_token_ids = input_ids_cpu[offset : offset + seq_len]
            req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
            offset += seq_len
            slot_offset += seq_len

            route = self._route_extend_request(req.rid, decoding_rids)
            if route == "continuation":
                pending_extends.append(
                    self._mlx_runner.extend_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        new_slot_ids=req_new_slots,
                        req=req,
                        logprob_start_len=int(logprob_starts[i]),
                        final_prefill=bool(final_prefill_flags[i]),
                    )
                )
            elif route == "decode":
                mixed_decode_rids.append(req.rid)
                mixed_decode_reqs.append(req)
            else:  # "prefill"
                prefix_slot_ids = req.prefix_indices.tolist()
                full_token_ids = list(req.get_fill_ids())
                pending_prefills.append(
                    self._mlx_runner.prefill_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        req=req,
                        logprob_start_len=int(logprob_starts[i]),
                        final_prefill=bool(final_prefill_flags[i]),
                    )
                )

        pending_mixed_decode: Optional[MlxPendingDecode] = None
        if mixed_decode_rids:
            pending_mixed_decode = self._mlx_runner.decode_batch_start(
                mixed_decode_rids,
                reqs=mixed_decode_reqs,
            )

        # Stack lazy tokens so the caller has a single handle to evaluate
        # after CPU scheduling work.  We also hand every cache buffer
        # (and the decode cache arrays) to mx.async_eval so the GPU
        # kernel-launch stream sees everything the next step depends on
        # before we actually block on anything.
        prefill_ext_tokens: list[mx.array] = [p.lazy_token for p in pending_prefills]
        prefill_ext_tokens.extend(e.lazy_token for e in pending_extends)

        async_args: list[mx.array] = []
        if prefill_ext_tokens:
            lazy_stacked = mx.stack(prefill_ext_tokens, axis=0)
            async_args.append(lazy_stacked)
        else:
            lazy_stacked = None

        for p in pending_prefills:
            async_args.extend(self._cache_state(p.cache))
            async_args.extend(self._mlx_runner.sampling_arrays(p.sampling))
            if p.input_token_logprobs is not None:
                async_args.append(p.input_token_logprobs)
            if p.customized_info is not None:
                async_args.extend(p.customized_info.values())
        for e in pending_extends:
            async_args.extend(self._cache_state(self._mlx_runner._req_caches[e.req_id]))
            async_args.extend(self._mlx_runner.sampling_arrays(e.sampling))
            if e.input_token_logprobs is not None:
                async_args.append(e.input_token_logprobs)
            if e.customized_info is not None:
                async_args.extend(e.customized_info.values())
        if pending_mixed_decode is not None:
            async_args.append(pending_mixed_decode.lazy_tokens)
            async_args.extend(
                self._mlx_runner.sampling_arrays(pending_mixed_decode.sampling)
            )
            for c_list in pending_mixed_decode.caches:
                async_args.extend(self._cache_state(c_list))
            if pending_mixed_decode.customized_info is not None:
                async_args.extend(pending_mixed_decode.customized_info.values())

        if async_args:
            mx.async_eval(*async_args)

        return (
            lazy_stacked,
            pending_prefills,
            pending_extends,
            pending_mixed_decode,
            "extend",
        )

    @staticmethod
    def _cache_state(cache_list) -> list[mx.array]:
        """Flatten a per-layer cache list to its ``state`` arrays."""
        arrays: list[mx.array] = []

        def collect(value):
            if isinstance(value, mx.array):
                arrays.append(value)
            elif value is None:
                return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for cache in cache_list:
            collect(getattr(cache, "state", ()))
        return arrays

    def async_chained_decode_mlx(
        self,
        prev_pending: MlxPendingDecode,
    ) -> tuple[mx.array, list, list, MlxPendingDecode, str]:
        """Launch a decode step that chains off a still-lazy previous decode.

        This is the "no idle gap" pipelining primitive: build the next
        decode's compute graph using ``prev_pending.lazy_tokens`` (still
        unevaluated) as its input ids, hand the combined graph to
        ``mx.async_eval``, and return.  The GPU runs the new step
        immediately after ``prev_pending`` with no scheduling gap, while
        the caller is free to block on ``prev_pending`` and run CPU-side
        bookkeeping.

        Preconditions (caller must ensure):

        * ``prev_pending`` was produced by a previous decode start
          (either :meth:`async_forward_batch_generation_mlx` in decode
          mode or a previous :meth:`async_chained_decode_mlx`).
        * The batch composition for this step is identical to
          ``prev_pending`` — same requests, same order.  Composition
          changes (finished reqs, new prefills) must break the chain.
        * ``prev_pending`` should be finalised BEFORE the returned
          pending, so per-request token lists are appended in order.

        Returns a 5-tuple matching
        :meth:`async_forward_batch_generation_mlx` for the decode case:
        ``(lazy_tokens, [], [], pending_decode, "decode")``.  The empty
        prefill/extend lists are always absent for chained decodes.
        """
        pending = self._mlx_runner.decode_batch_start_chained(prev_pending)
        mx.async_eval(
            pending.lazy_tokens,
            *self._mlx_runner.sampling_arrays(pending.sampling),
            *(pending.customized_info or {}).values(),
        )
        return pending.lazy_tokens, [], [], pending, "decode"

    @staticmethod
    def _mlx_logits_output(
        reqs: list,
        prefills: list[MlxPendingPrefill],
        extends: list[MlxPendingExtend],
        decode: Optional[MlxPendingDecode],
    ):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        rows: dict[str, tuple[object, int]] = {}
        for pending in prefills:
            rows[pending.req_id] = (pending, 0)
        for pending in extends:
            rows[pending.req_id] = (pending, 0)
        if decode is not None:
            for row, rid in enumerate(decode.req_ids):
                rows[rid] = (decode, row)

        if not any(
            getattr(pending, "sampling", None) is not None
            for pending, _ in rows.values()
        ):
            return LogitsProcessorOutput(next_token_logits=None)

        next_logprobs: list[float] = []
        input_logprobs: list[float] = []
        top_values: list[torch.Tensor] = []
        top_indices: list[torch.Tensor] = []
        requested_values: list[torch.Tensor] = []
        requested_indices: list[list[int]] = []
        any_top = False
        customized_info: dict[str, torch.Tensor] = {}
        customized_keys = {
            key
            for pending, _ in rows.values()
            for key in (getattr(pending, "customized_info", None) or {})
        }
        for key in customized_keys:
            row_tensors: list[torch.Tensor | None] = []
            max_size = 1
            scalar = True
            for req in reqs:
                pending, row = rows[req.rid]
                value = (getattr(pending, "customized_info", None) or {}).get(key)
                if value is None or int(value.shape[0]) <= row:
                    row_tensors.append(None)
                    continue
                tensor = torch.tensor(value[row].tolist(), dtype=torch.float32)
                scalar = scalar and tensor.ndim == 0
                max_size = max(max_size, tensor.numel())
                row_tensors.append(tensor.reshape(-1))
            if scalar and all(
                tensor is None or tensor.numel() == 1 for tensor in row_tensors
            ):
                customized_info[key] = torch.tensor(
                    [
                        float("nan") if tensor is None else float(tensor.item())
                        for tensor in row_tensors
                    ],
                    dtype=torch.float32,
                )
            else:
                padded = []
                for tensor in row_tensors:
                    if tensor is None:
                        padded.append(torch.full((max_size,), float("nan")))
                    elif tensor.numel() < max_size:
                        padded.append(
                            torch.cat(
                                (
                                    tensor,
                                    torch.full(
                                        (max_size - tensor.numel(),), float("nan")
                                    ),
                                )
                            )
                        )
                    else:
                        padded.append(tensor)
                customized_info[key] = torch.stack(padded)
        any_requested = False

        for req in reqs:
            pending, row = rows[req.rid]
            sampling = pending.sampling
            if sampling is None:
                next_logprobs.append(float("nan"))
                top_values.append(torch.empty(0, dtype=torch.float32))
                top_indices.append(torch.empty(0, dtype=torch.int64))
                requested_values.append(torch.empty(0, dtype=torch.float32))
                requested_indices.append([])
                continue
            next_logprobs.append(float(sampling.token_logprobs[row].item()))
            input_values = getattr(pending, "input_token_logprobs", None)
            if input_values is not None:
                raw_input = input_values.tolist()
                input_logprobs.extend(
                    float(value)
                    for value in (
                        raw_input if isinstance(raw_input, list) else [raw_input]
                    )
                )

            values = sampling.top_logprobs_val[row]
            indices = sampling.top_logprobs_idx[row]
            any_top = any_top or values is not None
            top_values.append(
                torch.tensor(values.tolist(), dtype=torch.float32)
                if values is not None
                else torch.empty(0, dtype=torch.float32)
            )
            top_indices.append(
                torch.tensor(indices.tolist(), dtype=torch.int64)
                if indices is not None
                else torch.empty(0, dtype=torch.int64)
            )

            requested = sampling.token_ids_logprobs_val[row]
            requested_ids = sampling.token_ids_logprobs_idx[row]
            any_requested = any_requested or requested is not None
            requested_values.append(
                torch.tensor(requested.tolist(), dtype=torch.float32)
                if requested is not None
                else torch.empty(0, dtype=torch.float32)
            )
            requested_indices.append(list(requested_ids or ()))

        return LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.tensor(next_logprobs, dtype=torch.float32),
            next_token_top_logprobs_val=top_values if any_top else None,
            next_token_top_logprobs_idx=top_indices if any_top else None,
            next_token_token_ids_logprobs_val=(
                requested_values if any_requested else None
            ),
            next_token_token_ids_logprobs_idx=(
                requested_indices if any_requested else None
            ),
            input_token_logprobs=torch.tensor(input_logprobs, dtype=torch.float32),
            customized_info=customized_info or None,
        )

    def finalize_mlx_result(
        self,
        prefills: list[MlxPendingPrefill],
        extends: list[MlxPendingExtend],
        decode: Optional[MlxPendingDecode],
        mode: str,
        reqs: list,
    ) -> GenerationBatchResult:
        """Materialise a lazy MLX result into a :class:`GenerationBatchResult`.

        The blocking wait happens inside ``decode_batch_finalize`` /
        ``prefill_finalize`` / ``extend_finalize`` via ``.tolist()`` /
        ``.item()`` on the specific lazy outputs.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if mode == "idle":
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        if mode == "decode":
            assert decode is not None
            next_tokens_list = self._mlx_runner.decode_batch_finalize(decode)

        elif mode == "extend":
            prefill_map: dict[str, int] = {}
            for pending_p in prefills:
                prefill_map[pending_p.req_id] = self._mlx_runner.prefill_finalize(
                    pending_p
                )

            extend_map: dict[str, int] = {}
            for pending_e in extends:
                extend_map[pending_e.req_id] = self._mlx_runner.extend_finalize(
                    pending_e
                )

            decode_map: dict[str, int] = {}
            if decode is not None:
                mixed_tokens = self._mlx_runner.decode_batch_finalize(decode)
                decode_map = {
                    rid: tok for rid, tok in zip(decode.req_ids, mixed_tokens)
                }

            next_tokens_list = []
            for req in reqs:
                if req.rid in decode_map:
                    next_tokens_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_tokens_list.append(extend_map[req.rid])
                else:
                    next_tokens_list.append(prefill_map[req.rid])

        else:
            raise ValueError(f"Unknown MLX async mode: {mode}")

        next_token_ids = torch.tensor(next_tokens_list, dtype=torch.long, device="cpu")
        logits_output = self._mlx_logits_output(reqs, prefills, extends, decode)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
