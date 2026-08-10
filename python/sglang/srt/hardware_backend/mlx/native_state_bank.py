from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import torch

from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.hybrid_state import qwen_exo_model_state_directory
from qwen_exo_booster.native_state_bank import (
    NativeStateBankError,
    _SAFE_DIGEST,
    _SCHEMA,
    _atomic_torch_save,
    _custom_params,
    _dequantize_fp8,
    _load_page_payload,
    _model_fingerprint,
    _node_mamba_value,
    _page_path,
    _quantize_fp8,
)
from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx

_MLX_AUXILIARY_FORMAT = "mlx-arrays-cache-v1"


def _mlx_to_owned_torch(value: mx.array) -> torch.Tensor:
    return mlx_to_torch(value, device="cpu").clone()


def _encode_tree(value: Any) -> Any:
    if isinstance(value, mx.array):
        tensor = _mlx_to_owned_torch(value)
        if tensor.is_floating_point() and tensor.ndim > 0:
            return {
                "kind": "fp8",
                "dtype": str(tensor.dtype),
                "value": _quantize_fp8(
                    tensor,
                    reduce_dims=(tensor.ndim - 1,),
                ),
            }
        return {"kind": "tensor", "value": tensor.contiguous()}
    if isinstance(value, list):
        return {"kind": "list", "value": [_encode_tree(item) for item in value]}
    if isinstance(value, tuple):
        return {"kind": "tuple", "value": [_encode_tree(item) for item in value]}
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "value": {str(key): _encode_tree(item) for key, item in value.items()},
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise NativeStateBankError(
        f"MLX native Bank cannot serialize {type(value).__name__} auxiliary state"
    )


def _decode_tree(payload: Any, *, dtype: mx.Dtype) -> Any:
    if not isinstance(payload, dict):
        raise NativeStateBankError("MLX native Bank auxiliary state is malformed")
    kind = payload.get("kind")
    value = payload.get("value")
    if kind == "fp8":
        dtype_map = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        target_dtype = dtype_map.get(str(payload.get("dtype")), torch.float32)
        tensor = _dequantize_fp8(value, dtype=target_dtype)
        return torch_to_mlx(tensor)
    if kind == "tensor":
        if not isinstance(value, torch.Tensor):
            raise NativeStateBankError("MLX native Bank tensor state is malformed")
        return torch_to_mlx(value)
    if kind == "list":
        return [_decode_tree(item, dtype=dtype) for item in value]
    if kind == "tuple":
        return tuple(_decode_tree(item, dtype=dtype) for item in value)
    if kind == "dict":
        return {key: _decode_tree(item, dtype=dtype) for key, item in value.items()}
    if kind == "scalar":
        return value
    raise NativeStateBankError(f"unknown MLX native Bank state kind: {kind!r}")


def _rope_key(attention: Any, key: mx.array, positions: mx.array) -> mx.array:
    """Apply RoPE token-wise by treating each token as a batch row."""

    rope = getattr(attention, "rope", None)
    if rope is None:
        raise NativeStateBankError("MLX Full-Attention layer exposes no RoPE")
    return rope(key[:, :, None, :], offset=positions)[:, :, 0, :]


def _cache_attributes(cache: Any) -> dict[str, Any]:
    return {
        name: _encode_tree(getattr(cache, name))
        for name in ("lengths", "left_padding")
        if hasattr(cache, name)
    }


@dataclass(slots=True)
class MlxNativeStateBankManager:
    root: Path
    model_config: Any
    runner: Any
    kv_allocator: Any
    req_pool: Any
    tree_cache: Any
    page_size: int
    consensus: Callable[[bool], bool]
    insert_params_factory: Callable[..., Any]
    model_fingerprint: str = field(init=False)
    full_layer_ids: tuple[int, ...] = field(init=False)
    auxiliary_layer_ids: tuple[int, ...] = field(init=False)
    hits: int = field(init=False, default=0)
    misses: int = field(init=False, default=0)
    loads: int = field(init=False, default=0)
    exports: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.model_fingerprint = _model_fingerprint(self.model_config)
        self.full_layer_ids = tuple(self.runner._cache_layout.attention_layer_indices)
        self.auxiliary_layer_ids = tuple(
            self.runner._cache_layout.auxiliary_layer_indices
        )
        if not self.full_layer_ids or not self.auxiliary_layer_ids:
            raise NativeStateBankError(
                "QWEN-EXO MLX native Bank requires Full-Attention and GDN layers"
            )

    @classmethod
    def from_scheduler(cls, scheduler: Any) -> MlxNativeStateBankManager:
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        runner = scheduler.tp_worker._mlx_runner
        return cls(
            root=qwen_exo_model_state_directory(scheduler.server_args) / "native-bank",
            model_config=scheduler.model_config,
            runner=runner,
            kv_allocator=scheduler.token_to_kv_pool_allocator,
            req_pool=scheduler.req_to_token_pool,
            tree_cache=scheduler.tree_cache,
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
        if capture_start + capture_count > len(req.origin_input_ids):
            raise NativeStateBankError(
                "bank-index export capture span exceeds its prompt"
            )
        caches = self.runner._req_caches.get(str(req.rid))
        if caches is None:
            raise NativeStateBankError("bank-index export has no MLX request cache")

        positions = mx.arange(
            capture_start,
            capture_start + capture_count,
            dtype=mx.int32,
        )
        full_attention: dict[str, dict[str, Any]] = {}
        for layer_id in self.full_layer_ids:
            cache = caches[layer_id]
            keys = getattr(cache, "keys", None)
            values = getattr(cache, "values", None)
            if (
                keys is None
                or values is None
                or int(getattr(cache, "offset", 0)) < capture_start + capture_count
            ):
                raise NativeStateBankError(
                    f"MLX Full-Attention layer {layer_id} has incomplete KV state"
                )
            rotated_key = keys[
                0, :, capture_start : capture_start + capture_count, :
            ].transpose(1, 0, 2)
            value = values[
                0, :, capture_start : capture_start + capture_count, :
            ].transpose(1, 0, 2)
            attention = self.runner._attention_module_for_layer(layer_id)
            raw_key = _rope_key(attention, rotated_key, -positions)
            full_attention[str(layer_id)] = {
                "key": _quantize_fp8(_mlx_to_owned_torch(raw_key), reduce_dims=(0, 2)),
                "value": _quantize_fp8(_mlx_to_owned_torch(value), reduce_dims=(0, 2)),
            }

        auxiliary_layers: dict[str, Any] = {}
        for layer_id in self.auxiliary_layer_ids:
            cache = caches[layer_id]
            auxiliary_layers[str(layer_id)] = {
                "state": _encode_tree(cache.state),
                "attributes": _cache_attributes(cache),
            }
        payload = {
            "schema": _SCHEMA,
            "source_digest": source_digest,
            "page_id": page_id,
            "rank": 0,
            "world_size": 1,
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
            "section_delta": {
                "mlx_auxiliary_state": {
                    "format": _MLX_AUXILIARY_FORMAT,
                    "layers": auxiliary_layers,
                }
            },
        }
        _atomic_torch_save(
            payload,
            _page_path(self.root, source_digest, page_id, 0),
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
        if self.consensus(local_hit):
            self.hits += 1
            req.qwen_exo_bank_cache_status = "hit"
            return True
        if not self.consensus(not local_hit):
            raise NativeStateBankError(
                "native Bank radix residency diverged across ranks"
            )

        payload: dict[str, Any] | None = None
        locally_ready = True
        try:
            payload = _load_page_payload(
                self.root,
                source_digest=source_digest,
                page_id=page_id,
                rank=0,
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
                "native Bank selection is unavailable on one or more ranks"
            )
        assert payload is not None
        self.misses += 1
        self._restore_prefix(
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
        if int(payload.get("world_size", -1)) != 1:
            raise NativeStateBankError("MLX native Bank artifact world size is stale")
        if str(payload.get("model_fingerprint")) != self.model_fingerprint:
            raise NativeStateBankError("MLX native Bank model fingerprint is stale")
        if tuple(int(item) for item in payload.get("full_layer_ids") or ()) != (
            self.full_layer_ids
        ):
            raise NativeStateBankError("MLX native Bank Full-Attention layout is stale")
        token_count = int(payload.get("capture_count", 0))
        if not local_positions or max(local_positions) >= token_count:
            raise NativeStateBankError(
                "MLX native Bank selection references a missing source token"
            )
        artifact_token_ids = tuple(int(item) for item in payload.get("token_ids") or ())
        if len(artifact_token_ids) != token_count:
            raise NativeStateBankError(
                "MLX native Bank artifact token map is incomplete"
            )
        selected_token_ids = tuple(
            artifact_token_ids[position] for position in local_positions
        )
        if selected_token_ids != prefix_token_ids:
            raise NativeStateBankError(
                "MLX native Bank selection tokens do not match the artifact"
            )
        auxiliary = (payload.get("section_delta") or {}).get("mlx_auxiliary_state")
        if (
            not isinstance(auxiliary, dict)
            or auxiliary.get("format") != _MLX_AUXILIARY_FORMAT
            or set((auxiliary.get("layers") or {}).keys())
            != {str(layer_id) for layer_id in self.auxiliary_layer_ids}
        ):
            raise NativeStateBankError(
                "MLX native Bank artifact lacks complete GDN state"
            )

    def _allocate(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv_indices = self.kv_allocator.alloc(count)
        auxiliary_pool = self.req_pool.auxiliary_state_pool
        auxiliary_index = auxiliary_pool.alloc(1)
        locally_allocated = kv_indices is not None and auxiliary_index is not None
        if not locally_allocated:
            if kv_indices is not None:
                self.kv_allocator.free(kv_indices)
            if auxiliary_index is not None:
                auxiliary_pool.free(auxiliary_index)
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            self.tree_cache.evict(EvictParams(num_tokens=count, mamba_num=1))
            kv_indices = self.kv_allocator.alloc(count)
            auxiliary_index = auxiliary_pool.alloc(1)
            locally_allocated = kv_indices is not None and auxiliary_index is not None
        if not self.consensus(locally_allocated):
            if kv_indices is not None:
                self.kv_allocator.free(kv_indices)
            if auxiliary_index is not None:
                auxiliary_pool.free(auxiliary_index)
            raise NativeStateBankError(
                "MLX native Bank allocation failed atomically across ranks"
            )
        assert kv_indices is not None and auxiliary_index is not None
        return kv_indices, auxiliary_index

    def _restore_prefix(
        self,
        *,
        payload: dict[str, Any],
        key: Any,
        local_positions: tuple[int, ...],
        memory_key: str,
    ) -> None:
        count = len(local_positions)
        kv_indices, auxiliary_index = self._allocate(count)
        auxiliary_pool = self.req_pool.auxiliary_state_pool
        try:
            selected = torch.tensor(local_positions, dtype=torch.long)
            slots = mx.array(kv_indices.detach().cpu().tolist(), dtype=mx.int32)
            virtual_positions = mx.arange(count, dtype=mx.int32)
            full_attention = payload["full_attention"]
            for layer_id in self.full_layer_ids:
                layer_payload = full_attention[str(layer_id)]
                raw_key = torch_to_mlx(
                    _dequantize_fp8(
                        layer_payload["key"],
                        dtype=torch.float32,
                        indices=selected,
                    )
                ).astype(self.runner._attention_kv_pool.dtype)
                value = torch_to_mlx(
                    _dequantize_fp8(
                        layer_payload["value"],
                        dtype=torch.float32,
                        indices=selected,
                    )
                ).astype(self.runner._attention_kv_pool.dtype)
                attention = self.runner._attention_module_for_layer(layer_id)
                rotated_key = _rope_key(attention, raw_key, virtual_positions)
                pool_index = self.runner._cache_layout.attention_pool_index_by_layer[
                    layer_id
                ]
                self.runner._attention_kv_pool.set_kv(
                    pool_index,
                    slots,
                    rotated_key,
                    value,
                )
                if layer_id == self.full_layer_ids[-1] and self.runner._qwen_exo:
                    self.runner._qwen_exo.register_memory_keys(memory_key, raw_key)

            cache = self.runner._new_cache_skeleton()
            auxiliary = payload["section_delta"]["mlx_auxiliary_state"]
            for layer_id in self.auxiliary_layer_ids:
                encoded = auxiliary["layers"][str(layer_id)]
                target = cache[layer_id]
                target.state = _decode_tree(
                    encoded["state"],
                    dtype=self.runner._attention_kv_pool.dtype,
                )
                for name, value in (encoded.get("attributes") or {}).items():
                    setattr(
                        target,
                        name,
                        _decode_tree(
                            value,
                            dtype=self.runner._attention_kv_pool.dtype,
                        ),
                    )
            auxiliary_pool.store_cache(
                auxiliary_index,
                cache,
                self.auxiliary_layer_ids,
            )
            mx.eval(*self.runner._attention_kv_pool.all_buffers())
            insert_result = self.tree_cache.insert(
                self.insert_params_factory(
                    key=key,
                    value=kv_indices,
                    mamba_value=auxiliary_index,
                )
            )
            if insert_result.mamba_exist:
                auxiliary_pool.free(auxiliary_index)
        except Exception:
            self.kv_allocator.free(kv_indices)
            auxiliary_pool.free(auxiliary_index)
            raise

    def stats(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "loads": int(self.loads),
            "exports": int(self.exports),
        }


__all__ = ["MlxNativeStateBankManager"]
