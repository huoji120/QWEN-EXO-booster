from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

from qwen_exo_booster.attention_signals import inverse_qwen35_rope
from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.hybrid_state import qwen_exo_model_state_directory

_SCHEMA = "qwen-exo-native-state-bank-v1"
_FP8_MAX = 448.0
_SAFE_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class NativeStateBankError(RuntimeError):
    """A native Bank artifact or scheduler restore violated its contract."""


def _quantize_fp8(
    value: torch.Tensor, *, reduce_dims: tuple[int, ...]
) -> dict[str, Any]:
    source = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if source.numel() == 0:
        raise NativeStateBankError("cannot quantize an empty native-state tensor")
    maximum = source.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (maximum / _FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    encoded = (source / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return {
        "data": encoded.view(torch.uint8).contiguous(),
        "scale": scale.contiguous(),
        "shape": tuple(int(item) for item in source.shape),
    }


def _dequantize_fp8(
    payload: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = tuple(int(item) for item in payload["shape"])
    encoded = payload["data"].view(torch.float8_e4m3fn).reshape(shape)
    scale = payload["scale"]
    if indices is not None:
        cpu_indices = indices.detach().to(device="cpu", dtype=torch.long)
        encoded = encoded.index_select(0, cpu_indices)
        if scale.shape[0] == shape[0]:
            scale = scale.index_select(0, cpu_indices)
    return (encoded.float() * scale.float()).to(device=device, dtype=dtype)


def _page_path(root: Path, source_digest: str, page_id: int, rank: int) -> Path:
    if not _SAFE_DIGEST.fullmatch(str(source_digest)):
        raise NativeStateBankError("native Bank source digest must be 64 lowercase hex")
    if int(page_id) < 0 or int(rank) < 0:
        raise NativeStateBankError("native Bank page and rank must be non-negative")
    return root / source_digest / f"page-{int(page_id):08d}-rank-{int(rank):04d}.pt"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_page_payload(
    root: Path,
    *,
    source_digest: str,
    page_id: int,
    rank: int,
) -> dict[str, Any]:
    path = _page_path(root, source_digest, page_id, rank)
    try:
        payload = torch.load(
            str(path), map_location="cpu", weights_only=True, mmap=True
        )
    except FileNotFoundError as exc:
        raise NativeStateBankError(
            f"native Bank rank artifact is missing: {path}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise NativeStateBankError(
            f"native Bank rank artifact is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        raise NativeStateBankError(
            f"native Bank rank artifact has an invalid schema: {path}"
        )
    expected = (str(source_digest), int(page_id), int(rank))
    observed = (
        str(payload.get("source_digest")),
        int(payload.get("page_id", -1)),
        int(payload.get("rank", -1)),
    )
    if observed != expected:
        raise NativeStateBankError(
            f"native Bank rank artifact identity mismatch: expected={expected}, observed={observed}"
        )
    return payload


def validate_page_artifacts(
    root: str | Path,
    *,
    source_digest: str,
    page_id: int,
    world_size: int,
    model_fingerprint: str,
    prefix_identity: str,
    token_count: int,
) -> None:
    for rank in range(int(world_size)):
        payload = _load_page_payload(
            Path(root), source_digest=source_digest, page_id=page_id, rank=rank
        )
        if (
            int(payload.get("world_size", -1)) != int(world_size)
            or str(payload.get("model_fingerprint")) != str(model_fingerprint)
            or str(payload.get("prefix_identity")) != str(prefix_identity)
            or int(payload.get("capture_count", -1)) != int(token_count)
            or len(tuple(payload.get("token_ids") or ())) != int(token_count)
            or not payload.get("full_attention")
        ):
            raise NativeStateBankError(
                "native Bank rank artifact header is stale or incomplete"
            )
        section_delta = payload.get("section_delta") or {}
        has_cuda_delta = bool(section_delta.get("conv")) and bool(
            section_delta.get("temporal")
        )
        has_mlx_delta = bool(section_delta.get("mlx_auxiliary_state"))
        if not (has_cuda_delta or has_mlx_delta):
            raise NativeStateBankError(
                "native Bank rank artifact lacks complete CUDA or MLX GDN state"
            )


def load_page_key_heads(
    root: str | Path,
    *,
    source_digest: str,
    page_id: int,
    world_size: int,
    model_fingerprint: str | None = None,
    prefix_identity: str | None = None,
    token_count: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load final-layer raw K heads from every TP rank without projection."""

    rank_keys: list[torch.Tensor] = []
    expected_tokens: int | None = None
    expected_key_shape: tuple[int, int] | None = None
    for rank in range(int(world_size)):
        payload = _load_page_payload(
            Path(root), source_digest=source_digest, page_id=page_id, rank=rank
        )
        if model_fingerprint is not None and str(
            payload.get("model_fingerprint")
        ) != str(model_fingerprint):
            raise NativeStateBankError(
                "native Bank TP artifact model fingerprint is stale"
            )
        if int(payload.get("world_size", -1)) != int(world_size):
            raise NativeStateBankError("native Bank TP artifact world size is stale")
        if prefix_identity is not None and str(payload.get("prefix_identity")) != str(
            prefix_identity
        ):
            raise NativeStateBankError("native Bank TP artifact page identity is stale")
        if token_count is not None and (
            int(payload.get("capture_count", -1)) != int(token_count)
            or len(tuple(payload.get("token_ids") or ())) != int(token_count)
        ):
            raise NativeStateBankError("native Bank TP artifact token map is stale")
        layers = payload.get("full_attention") or {}
        layer_ids = tuple(int(item) for item in payload.get("full_layer_ids") or ())
        if not layer_ids:
            raise NativeStateBankError(
                "native Bank artifact has no Full-Attention layers"
            )
        final_layer = layers.get(str(layer_ids[-1]))
        if not isinstance(final_layer, dict) or "key" not in final_layer:
            raise NativeStateBankError(
                "native Bank artifact lacks its final-layer raw K"
            )
        keys = _dequantize_fp8(final_layer["key"], dtype=dtype)
        if keys.ndim != 3 or not bool(torch.isfinite(keys.float()).all()):
            raise NativeStateBankError(
                "native Bank raw K must be finite [tokens, heads, head_dim]"
            )
        key_shape = (int(keys.shape[1]), int(keys.shape[2]))
        if expected_tokens is None:
            expected_tokens = int(keys.shape[0])
            expected_key_shape = key_shape
        elif int(keys.shape[0]) != expected_tokens or key_shape != expected_key_shape:
            raise NativeStateBankError("native Bank TP ranks disagree on raw-K shape")
        rank_keys.append(keys.contiguous())
    if not rank_keys:
        raise NativeStateBankError("native Bank has no TP rank artifacts")
    return torch.cat(rank_keys, dim=1).contiguous()


def _inverse_rotary_key(
    key: torch.Tensor,
    *,
    positions: torch.Tensor,
    rotary: Any,
) -> torch.Tensor:
    return inverse_qwen35_rope(
        key.float(),
        positions,
        rotary=rotary,
        head_dim=int(key.shape[-1]),
    ).to(dtype=key.dtype)


def _apply_rotary_key(
    key: torch.Tensor,
    *,
    positions: torch.Tensor,
    rotary: Any,
) -> torch.Tensor:
    rotary_dim = int(getattr(rotary, "rotary_dim", key.shape[-1]))
    cache = rotary.cos_sin_cache.index_select(
        0, positions.to(device=rotary.cos_sin_cache.device, dtype=torch.long)
    )
    cos, sin = cache.chunk(2, dim=-1)
    cos = cos.to(device=key.device, dtype=torch.float32).unsqueeze(1)
    sin = sin.to(device=key.device, dtype=torch.float32).unsqueeze(1)
    raw = key[..., :rotary_dim].float()
    if bool(getattr(rotary, "is_neox_style", True)):
        first, second = torch.chunk(raw, 2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
    else:
        first = raw[..., ::2]
        second = raw[..., 1::2]
        rotated = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    if rotary_dim == key.shape[-1]:
        return rotated.to(dtype=key.dtype)
    return torch.cat((rotated.to(dtype=key.dtype), key[..., rotary_dim:]), dim=-1)


def _language_layers(model: Any) -> list[Any]:
    candidates = (
        getattr(
            getattr(getattr(model, "model", None), "language_model", None),
            "layers",
            None,
        ),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return list(layers)
    raise NativeStateBankError("cannot locate Qwen3.5 decoder layers")


def _full_layer_ids(model_config: Any) -> tuple[int, ...]:
    hf_config = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", model_config
    )
    block_types = tuple(getattr(hf_config, "layer_types", ()) or ())
    if not block_types:
        block_types = tuple(getattr(hf_config, "layers_block_type", ()) or ())
    ids = tuple(
        index
        for index, block_type in enumerate(block_types)
        if str(block_type).lower() in {"full_attention", "attention", "full"}
    )
    if not ids:
        raise NativeStateBankError("Qwen3.5 config exposes no Full-Attention layer IDs")
    return ids


def _model_fingerprint(model_config: Any) -> str:
    model_path = str(getattr(model_config, "model_path", ""))
    if model_path:
        try:
            from qwen_exo_booster.fingerprint import ModelIdentity

            return ModelIdentity.from_path(model_path).fingerprint
        except (FileNotFoundError, OSError, ValueError):
            pass
    hf_config = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", model_config
    )
    return stable_digest(
        model_path,
        getattr(hf_config, "model_type", ""),
        getattr(hf_config, "num_hidden_layers", ""),
        getattr(hf_config, "hidden_size", ""),
        getattr(hf_config, "num_attention_heads", ""),
        getattr(hf_config, "num_key_value_heads", ""),
    )


def _custom_params(req: Any) -> dict[str, Any]:
    return dict(getattr(req.sampling_params, "custom_params", None) or {})


def _node_mamba_value(node: Any) -> Any:
    component_data = getattr(node, "component_data", None)
    if component_data is not None:
        return component_data[2].value
    return getattr(node, "mamba_value", None)


@dataclass(slots=True)
class NativeStateBankManager:
    root: Path
    model_config: Any
    model: Any
    kv_pool: Any
    kv_allocator: Any
    req_pool: Any
    tree_cache: Any
    rank: int
    world_size: int
    page_size: int
    consensus: Callable[[bool], bool]
    insert_params_factory: Callable[..., Any] | None = None
    full_layer_ids: tuple[int, ...] = field(init=False)
    layers: list[Any] = field(init=False)
    model_fingerprint: str = field(init=False)
    hits: int = field(init=False, default=0)
    misses: int = field(init=False, default=0)
    loads: int = field(init=False, default=0)
    exports: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.full_layer_ids = _full_layer_ids(self.model_config)
        self.layers = _language_layers(self.model)
        self.model_fingerprint = _model_fingerprint(self.model_config)
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.exports = 0

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> NativeStateBankManager:
        runner = scheduler.tp_worker.model_runner
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        return cls(
            root=qwen_exo_model_state_directory(scheduler.server_args) / "native-bank",
            model_config=scheduler.model_config,
            model=runner.model,
            kv_pool=runner.token_to_kv_pool,
            kv_allocator=scheduler.token_to_kv_pool_allocator,
            req_pool=scheduler.req_to_token_pool,
            tree_cache=scheduler.tree_cache,
            rank=int(scheduler.ps.tp_rank),
            world_size=int(scheduler.tp_group.world_size),
            page_size=int(scheduler.page_size),
            consensus=scheduler._qwen_exo_admission_consensus,
            insert_params_factory=InsertParams,
        )

    def maybe_export(self, req: Any) -> bool:
        params = _custom_params(req)
        export = params.get("qwen_exo_native_bank_export")
        if params.get("qwen_exo_job_type") != "bank_index" or not isinstance(
            export, dict
        ):
            return False
        req.qwen_exo_native_bank_no_cache = True
        self._export(req, export)
        req.qwen_exo_bank_export_status = "exported"
        self.exports += 1
        return True

    def _export(self, req: Any, export: dict[str, Any]) -> None:
        source_digest = str(export.get("source_digest") or "")
        page_id = int(export.get("page_id", -1))
        capture_start = int(export.get("capture_start", -1))
        capture_count = int(export.get("capture_count", 0))
        token_start = int(export.get("token_start", 0))
        prefix_identity = str(export.get("prefix_identity") or "")
        if not _SAFE_DIGEST.fullmatch(source_digest):
            raise NativeStateBankError("bank-index export has an invalid source digest")
        if page_id < 0 or capture_start < 0 or capture_count <= 0:
            raise NativeStateBankError("bank-index export has an invalid capture span")
        prompt_tokens = len(req.origin_input_ids)
        if capture_start + capture_count > prompt_tokens:
            raise NativeStateBankError(
                "bank-index export capture span exceeds its prompt"
            )
        if req.req_pool_idx is None:
            raise NativeStateBankError(
                "bank-index export request has no request-pool row"
            )
        mapping = self.req_pool.req_to_token[
            req.req_pool_idx, capture_start : capture_start + capture_count
        ].to(dtype=torch.long)
        if mapping.numel() != capture_count or bool((mapping <= 0).any().item()):
            raise NativeStateBankError("bank-index export has an incomplete KV mapping")
        source_positions = torch.arange(
            capture_start,
            capture_start + capture_count,
            device=mapping.device,
            dtype=torch.long,
        )
        full_attention: dict[str, dict[str, Any]] = {}
        for layer_id in self.full_layer_ids:
            layer = self.layers[layer_id]
            rotary = getattr(layer, "rotary_emb", None)
            if rotary is None:
                raise NativeStateBankError(
                    f"Full-Attention layer {layer_id} exposes no rotary embedding"
                )
            key = self.kv_pool.get_key_buffer(layer_id).index_select(0, mapping)
            value = self.kv_pool.get_value_buffer(layer_id).index_select(0, mapping)
            raw_key = _inverse_rotary_key(
                key, positions=source_positions, rotary=rotary
            )
            full_attention[str(layer_id)] = {
                "key": _quantize_fp8(raw_key, reduce_dims=(0, 2)),
                "value": _quantize_fp8(value, reduce_dims=(0, 2)),
            }
        mamba_pool = getattr(self.req_pool, "mamba_pool", None)
        if mamba_pool is None or req.mamba_pool_idx is None:
            raise NativeStateBankError("bank-index export has no active GDN state")
        physical_mamba = self.req_pool.translate_mamba_indices(
            req.mamba_pool_idx.reshape(1)
        )
        conv_states, temporal_states = mamba_pool.get_cpu_copy(physical_mamba)
        section_delta = {
            "conv": tuple(
                _quantize_fp8(value, reduce_dims=(value.ndim - 1,))
                for value in conv_states
            ),
            "temporal": _quantize_fp8(
                temporal_states,
                reduce_dims=(temporal_states.ndim - 2, temporal_states.ndim - 1),
            ),
        }
        payload = {
            "schema": _SCHEMA,
            "source_digest": source_digest,
            "page_id": page_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "model_fingerprint": self.model_fingerprint,
            "prefix_identity": prefix_identity,
            "token_start": token_start,
            "token_end": token_start + capture_count,
            "capture_count": capture_count,
            "token_ids": tuple(
                int(token)
                for token in req.origin_input_ids[
                    capture_start : capture_start + capture_count
                ]
            ),
            "full_layer_ids": self.full_layer_ids,
            "full_attention": full_attention,
            "section_delta": section_delta,
        }
        _atomic_torch_save(
            payload, _page_path(self.root, source_digest, page_id, self.rank)
        )

    def ensure_prefix(self, req: Any) -> bool:
        selection = _custom_params(req).get("qwen_exo_native_bank_selection")
        if not isinstance(selection, dict):
            return False
        source_digest = str(selection.get("source_digest") or "")
        page_id = int(selection.get("page_id", -1))
        local_positions = tuple(
            int(item) for item in selection.get("local_positions") or ()
        )
        prefix_identity = str(selection.get("prefix_identity") or "")
        prefix_count = len(local_positions)
        if (
            not _SAFE_DIGEST.fullmatch(source_digest)
            or page_id < 0
            or prefix_count == 0
            or prefix_count % self.page_size != 0
            or len(set(local_positions)) != prefix_count
            or any(position < 0 for position in local_positions)
        ):
            raise NativeStateBankError(
                "native Bank selection has an invalid aligned plan"
            )
        if len(req.origin_input_ids) < prefix_count:
            raise NativeStateBankError(
                "native Bank selection exceeds the request prompt"
            )
        observed_identity = stable_digest(
            source_digest,
            page_id,
            *local_positions,
            *req.origin_input_ids[:prefix_count],
        )
        if observed_identity != prefix_identity:
            raise NativeStateBankError(
                "native Bank selection identity does not bind its tokens"
            )
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        key = RadixKey(req.origin_input_ids[:prefix_count], req.extra_key)
        existing = self.tree_cache.match_prefix(
            MatchPrefixParams(key=key, cow_mamba=False, req=None)
        )
        local_hit = (
            len(existing.device_indices) == prefix_count
            and _node_mamba_value(existing.last_device_node) is not None
        )
        all_hit = self.consensus(local_hit)
        if all_hit:
            self.hits += 1
            req.qwen_exo_bank_cache_status = "hit"
            return True
        all_miss = self.consensus(not local_hit)
        if not all_miss:
            raise NativeStateBankError(
                "native Bank radix residency diverged across TP ranks"
            )
        payload: dict[str, Any] | None = None
        locally_ready = True
        try:
            payload = _load_page_payload(
                self.root,
                source_digest=source_digest,
                page_id=page_id,
                rank=self.rank,
            )
            self._validate_restore_payload(
                payload,
                local_positions=local_positions,
                prefix_token_ids=tuple(req.origin_input_ids[:prefix_count]),
            )
        except NativeStateBankError:
            locally_ready = False
        if not self.consensus(locally_ready):
            raise NativeStateBankError(
                "native Bank selection is unavailable on one or more TP ranks"
            )
        assert payload is not None
        self.misses += 1
        self._restore_prefix(
            req,
            payload=payload,
            key=key,
            local_positions=local_positions,
            memory_key=f"qwen-exo-native:{prefix_identity}",
        )
        self.loads += 1
        req.qwen_exo_bank_cache_status = "loaded"
        return True

    def _validate_restore_payload(
        self,
        payload: dict[str, Any],
        *,
        local_positions: tuple[int, ...],
        prefix_token_ids: tuple[int, ...],
    ) -> None:
        if int(payload.get("world_size", -1)) != self.world_size:
            raise NativeStateBankError("native Bank artifact TP world size is stale")
        if str(payload.get("model_fingerprint")) != self.model_fingerprint:
            raise NativeStateBankError(
                "native Bank artifact model fingerprint is stale"
            )
        if (
            tuple(int(item) for item in payload.get("full_layer_ids") or ())
            != self.full_layer_ids
        ):
            raise NativeStateBankError(
                "native Bank artifact Full-Attention layout is stale"
            )
        token_count = int(payload.get("capture_count", 0))
        if not local_positions or max(local_positions) >= token_count:
            raise NativeStateBankError(
                "native Bank selection references a missing source token"
            )
        artifact_token_ids = tuple(int(item) for item in payload.get("token_ids") or ())
        if len(artifact_token_ids) != token_count:
            raise NativeStateBankError("native Bank artifact token map is incomplete")
        selected_token_ids = tuple(
            artifact_token_ids[position] for position in local_positions
        )
        if selected_token_ids != prefix_token_ids:
            raise NativeStateBankError(
                "native Bank selection tokens do not match the source artifact"
            )
        section_delta = payload.get("section_delta") or {}
        if not section_delta.get("conv") or not section_delta.get("temporal"):
            raise NativeStateBankError(
                "native Bank artifact lacks its complete document GDN state"
            )

    def _restore_prefix(
        self,
        req: Any,
        *,
        payload: dict[str, Any],
        key: Any,
        local_positions: tuple[int, ...],
        memory_key: str | None = None,
    ) -> None:
        count = len(local_positions)
        kv_indices = self.kv_allocator.alloc(count)
        mamba_allocator = getattr(self.req_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            if kv_indices is not None:
                self.kv_allocator.free(kv_indices)
            raise NativeStateBankError("native Bank restore requires a GDN allocator")
        mamba_index = mamba_allocator.alloc(1)
        locally_allocated = kv_indices is not None and mamba_index is not None
        if not locally_allocated:
            if kv_indices is not None:
                self.kv_allocator.free(kv_indices)
            if mamba_index is not None:
                mamba_allocator.free(mamba_index)
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            self.tree_cache.evict(EvictParams(num_tokens=count, mamba_num=1))
            kv_indices = self.kv_allocator.alloc(count)
            mamba_index = mamba_allocator.alloc(1)
            locally_allocated = kv_indices is not None and mamba_index is not None
        if not self.consensus(locally_allocated):
            if kv_indices is not None:
                self.kv_allocator.free(kv_indices)
            if mamba_index is not None:
                mamba_allocator.free(mamba_index)
            raise NativeStateBankError(
                "native Bank allocation failed atomically across TP ranks"
            )
        assert kv_indices is not None and mamba_index is not None
        try:
            selected = torch.tensor(local_positions, dtype=torch.long)
            virtual_positions = torch.arange(
                count, device=kv_indices.device, dtype=torch.long
            )
            full_attention = payload["full_attention"]
            for layer_id in self.full_layer_ids:
                layer_payload = full_attention[str(layer_id)]
                key_buffer = self.kv_pool.get_key_buffer(layer_id)
                value_buffer = self.kv_pool.get_value_buffer(layer_id)
                # Native pages are stored independently from the live KV cache
                # dtype. Restore BF16 activations through the pool API so FP8
                # caches apply their quantization scale and use supported CUDA
                # store kernels instead of torch.index_copy on Float8 tensors.
                raw_key = _dequantize_fp8(
                    layer_payload["key"],
                    device=key_buffer.device,
                    dtype=torch.bfloat16,
                    indices=selected,
                )
                if memory_key is not None and layer_id == self.full_layer_ids[-1]:
                    tracker = getattr(
                        self.layers[layer_id], "qwen_exo_signal_tracker", None
                    )
                    if tracker is not None:
                        tracker.register_memory_keys(memory_key, raw_key)
                value = _dequantize_fp8(
                    layer_payload["value"],
                    device=value_buffer.device,
                    dtype=torch.bfloat16,
                    indices=selected,
                )
                layer = self.layers[layer_id]
                rotated_key = _apply_rotary_key(
                    raw_key,
                    positions=virtual_positions.to(raw_key.device),
                    rotary=getattr(layer, "rotary_emb"),
                )
                attention = getattr(layer, "attn", None)
                if attention is None:
                    raise NativeStateBankError(
                        f"Full-Attention layer {layer_id} exposes no KV cache writer"
                    )
                self.kv_pool.set_kv_buffer(
                    attention,
                    kv_indices,
                    rotated_key,
                    value,
                    k_scale=getattr(attention, "k_scale", None),
                    v_scale=getattr(attention, "v_scale", None),
                )
            section_delta = payload["section_delta"]
            conv = tuple(
                _dequantize_fp8(item, dtype=torch.bfloat16)
                for item in section_delta["conv"]
            )
            temporal = _dequantize_fp8(section_delta["temporal"], dtype=torch.bfloat16)
            physical_mamba = self.req_pool.translate_mamba_indices(mamba_index)
            self.req_pool.mamba_pool.load_cpu_copy((conv, temporal), physical_mamba)
            insert_params_factory = self.insert_params_factory
            if insert_params_factory is None:
                from sglang.srt.mem_cache.base_prefix_cache import InsertParams

                insert_params_factory = InsertParams
            insert_result = self.tree_cache.insert(
                insert_params_factory(
                    key=key,
                    value=kv_indices,
                    mamba_value=mamba_index,
                )
            )
            if insert_result.mamba_exist:
                mamba_allocator.free(mamba_index)
        except Exception:
            self.kv_allocator.free(kv_indices)
            mamba_allocator.free(mamba_index)
            raise

    def stats(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "loads": int(self.loads),
            "exports": int(self.exports),
        }


__all__ = [
    "NativeStateBankError",
    "NativeStateBankManager",
    "load_page_key_heads",
    "validate_page_artifacts",
]
