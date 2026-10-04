from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from qwen_exo_booster.attention_signals import inverse_qwen35_rope
from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.hybrid_state import qwen_exo_model_state_directory

_SCHEMA = "qwen-exo-native-state-bank-v1"
_SESSION_INITIAL_GDN_SCHEMA = "qwen-exo-session-initial-gdn-v1"
_QWEN4_STATE_SCHEMA = "qwen-exo-qwen4-qsa-ple-v1"
_SESSION_GDN_MAX_SOURCES = 2
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


def _session_initial_gdn_path(
    root: Path, source_digest: str, state_identity: str, rank: int
) -> Path:
    if not _SAFE_DIGEST.fullmatch(str(source_digest)) or not _SAFE_DIGEST.fullmatch(
        str(state_identity)
    ):
        raise NativeStateBankError(
            "session-initial GDN source and state identities must be 64 lowercase hex"
        )
    if int(rank) < 0:
        raise NativeStateBankError("session-initial GDN rank must be non-negative")
    return (
        root
        / "session-initial-gdn"
        / source_digest
        / state_identity
        / f"rank-{int(rank):04d}.pt"
    )


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_page_file(
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
    if not isinstance(payload, dict) or payload.get("schema") not in {
        _SCHEMA,
        "qwen-exo-native-state-reference-v1",
    }:
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


def _load_page_payload(
    root: Path, *, source_digest: str, page_id: int, rank: int
) -> dict[str, Any]:
    payload = _load_page_file(
        root, source_digest=source_digest, page_id=page_id, rank=rank
    )
    if payload["schema"] == _SCHEMA:
        return payload
    reference = payload.get("reused_from")
    if not isinstance(reference, dict):
        raise NativeStateBankError("native Bank page reference is missing its source")
    base = _load_page_file(
        root,
        source_digest=str(reference.get("source_digest") or ""),
        page_id=int(reference.get("page_id", -1)),
        rank=rank,
    )
    if (
        base["schema"] != _SCHEMA
        or base.get("prefix_identity") != reference.get("prefix_identity")
        or any(
            base.get(key) != payload.get(key)
            for key in ("model_fingerprint", "world_size", "capture_count")
        )
    ):
        raise NativeStateBankError("native Bank page reference is stale or chained")
    return {
        **base,
        "source_digest": source_digest,
        "page_id": page_id,
        "prefix_identity": payload["prefix_identity"],
        "reused_from": reference,
    }


def load_page_compression_ratio(
    root: str | Path, *, source_digest: str, page_id: int
) -> int:
    """Read the native artifact's selection granularity without loading tensors."""
    payload = _load_page_payload(Path(root), source_digest=source_digest, page_id=page_id, rank=0)
    state = payload.get("qwen4_state")
    if state is None:
        return 1
    if not isinstance(state, dict) or state.get("schema") != _QWEN4_STATE_SCHEMA:
        raise NativeStateBankError("native Qwen4 state schema is stale")
    ratio = int(state.get("compress_ratio", 0))
    if ratio < 2 or 64 % ratio:
        raise NativeStateBankError("native Qwen4 compression layout is invalid")
    return ratio


def reuse_page_artifacts(
    root: Path,
    *,
    source_digest: str,
    page_id: int,
    prefix_identity: str,
    target_digest: str,
    target_page_id: int,
    target_prefix_identity: str,
    world_size: int,
    model_fingerprint: str,
    token_ids: tuple[int, ...],
) -> None:
    """Publish tiny references to immutable, exactly matching TP document states.

    References always point at original tensor files, never at another reference.
    Old generations remain live dependencies; age alone cannot authorize deletion.
    """
    if target_digest == source_digest:
        raise NativeStateBankError("native Bank reuse cannot overwrite its source")
    validate_page_artifacts(
        root,
        source_digest=source_digest,
        page_id=page_id,
        world_size=world_size,
        model_fingerprint=model_fingerprint,
        prefix_identity=prefix_identity,
        token_count=len(token_ids),
    )
    if all(
        _page_path(root, target_digest, target_page_id, rank).is_file()
        for rank in range(world_size)
    ):
        validate_page_artifacts(
            root,
            source_digest=target_digest,
            page_id=target_page_id,
            world_size=world_size,
            model_fingerprint=model_fingerprint,
            prefix_identity=target_prefix_identity,
            token_count=len(token_ids),
        )
        for rank in range(world_size):
            existing = _load_page_payload(
                root, source_digest=target_digest, page_id=target_page_id, rank=rank
            )
            if tuple(existing["token_ids"]) != token_ids:
                raise NativeStateBankError("native Bank reuse target tokens changed")
        return
    manifests = []
    for rank in range(world_size):
        payload = _load_page_payload(
            root, source_digest=source_digest, page_id=page_id, rank=rank
        )
        if tuple(payload["token_ids"]) != token_ids:
            raise NativeStateBankError("native Bank reuse document tokens changed")
        reference = payload.get("reused_from") or {
            "source_digest": source_digest,
            "page_id": page_id,
            "prefix_identity": prefix_identity,
        }
        manifests.append(
            {
                "schema": "qwen-exo-native-state-reference-v1",
                "source_digest": target_digest,
                "page_id": target_page_id,
                "rank": rank,
                "world_size": world_size,
                "model_fingerprint": model_fingerprint,
                "prefix_identity": target_prefix_identity,
                "capture_count": len(token_ids),
                "reused_from": reference,
            }
        )
    for rank, manifest in enumerate(manifests):
        _atomic_torch_save(
            manifest, _page_path(root, target_digest, target_page_id, rank)
        )


def _load_session_initial_gdn_payload(
    root: Path,
    *,
    source_digest: str,
    state_identity: str,
    rank: int,
) -> dict[str, Any]:
    path = _session_initial_gdn_path(
        Path(root),
        source_digest=source_digest,
        state_identity=state_identity,
        rank=rank,
    )
    try:
        payload = torch.load(
            str(path), map_location="cpu", weights_only=True, mmap=True
        )
    except FileNotFoundError as exc:
        raise NativeStateBankError(
            f"session-initial GDN rank artifact is missing: {path}"
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise NativeStateBankError(
            f"session-initial GDN rank artifact is unreadable: {path}"
        ) from exc
    expected = (str(source_digest), str(state_identity), int(rank))
    observed = (
        str(payload.get("source_digest")) if isinstance(payload, dict) else "",
        str(payload.get("state_identity")) if isinstance(payload, dict) else "",
        int(payload.get("rank", -1)) if isinstance(payload, dict) else -1,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _SESSION_INITIAL_GDN_SCHEMA
        or observed != expected
    ):
        raise NativeStateBankError(
            "session-initial GDN rank artifact identity or schema is invalid"
        )
    return payload


def _validate_session_initial_gdn_state(payload: dict[str, Any]) -> None:
    """Validate the single-slot Dense, ReplaySSM, and PLE host layouts.

    Pool-specific geometry is checked again by the manager before adoption.
    """
    state = payload.get("mamba_state")
    if (
        not isinstance(state, (tuple, list))
        or len(state) not in {2, 3, 4}
        or not isinstance(state[0], (tuple, list))
        or not state[0]
    ):
        raise NativeStateBankError(
            "session-initial GDN artifact recurrent payload is incomplete"
        )
    components = payload.get("mamba_state_components")
    if "mamba_state_components" in payload:
        if (
            not isinstance(components, (tuple, list))
            or tuple(components) not in (
                ("conv", "temporal"),
                ("conv", "temporal", "replayssm_cursors"),
                ("conv", "temporal", "slot_siblings"),
                ("conv", "temporal", "replayssm_cursors", "slot_siblings"),
            )
            or len(components) != len(state)
        ):
            raise NativeStateBankError(
                "session-initial GDN artifact declared component layout is incomplete or invalid"
            )
    conv, temporal = state[:2]
    state_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    for source in (*conv, temporal):
        if (
            not isinstance(source, torch.Tensor)
            or source.ndim < 3
            or source.shape[1] != 1
            or any(size < 1 for size in source.shape)
            or source.dtype not in state_dtypes
            or not bool(torch.isfinite(source).all().item())
        ):
            raise NativeStateBankError(
                "session-initial GDN artifact recurrent tensor shape/dtype/value is invalid"
            )
    if any(source.shape[0] != temporal.shape[0] for source in conv):
        raise NativeStateBankError(
            "session-initial GDN artifact convolution layer count is stale"
        )

    if len(state) == 2:
        return
    # Historical three-part states have no declaration; distinguish their
    # cursor triple from the differently shaped PLE tensors.
    third = state[2]
    has_cursors = (
        "replayssm_cursors" in components
        if components is not None
        else len(state) == 4 or (
            isinstance(third, (tuple, list)) and len(third) == 3
        )
    )
    if has_cursors:
        if not isinstance(third, (tuple, list)) or len(third) != 3:
            raise NativeStateBankError(
                "session-initial GDN artifact ReplaySSM cursors are invalid"
            )
        for cursor, dtype in zip(third, (torch.int32, torch.int32, torch.int8)):
            if (
                not isinstance(cursor, torch.Tensor)
                or tuple(cursor.shape) != (1,)
                or cursor.dtype != dtype
            ):
                raise NativeStateBankError(
                    "session-initial GDN artifact ReplaySSM cursor shape/dtype is invalid"
                )
            if bool((cursor != 0).any().item()):
                raise NativeStateBankError(
                    "session-initial GDN artifact contains an unflushed ReplaySSM ring"
                )
        if len(state) == 3:
            return

    siblings = state[-1]
    if not isinstance(siblings, (tuple, list)) or len(siblings) not in {1, 2}:
        raise NativeStateBankError(
            "session-initial PLE side states are incomplete"
        )
    for index, source in enumerate(siblings):
        # Disabled side pools do not register a sibling. A one-part bundle
        # may therefore hold either short-conv or token context.
        is_short_conv = isinstance(source, torch.Tensor) and source.ndim == 4
        if is_short_conv:
            valid = (
                (len(siblings) == 1 or index == 0)
                and source.shape[1] == 1
                and all(size > 0 for size in source.shape)
                and source.dtype in state_dtypes
                and bool(torch.isfinite(source).all().item())
            )
        else:
            valid = (
                (len(siblings) == 1 or index == 1)
                and isinstance(source, torch.Tensor)
                and source.ndim == 2
                and source.shape[0] == 1
                and source.shape[1] > 0
                and source.dtype == torch.int64
            )
        if not valid:
            raise NativeStateBankError(
                "session-initial PLE state shape/dtype/value is invalid"
            )


def validate_session_initial_gdn_artifacts(
    root: str | Path,
    *,
    source_digest: str,
    state_identity: str,
    world_size: int,
    model_fingerprint: str,
) -> dict[str, int]:
    if int(world_size) < 1:
        raise NativeStateBankError(
            "session-initial GDN validation requires a positive TP world size"
        )
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    for rank in range(int(world_size)):
        payload = _load_session_initial_gdn_payload(
            Path(root),
            source_digest=source_digest,
            state_identity=state_identity,
            rank=rank,
        )
        if int(payload.get("world_size", -1)) != int(world_size):
            raise NativeStateBankError(
                "session-initial GDN artifact TP world size is stale"
            )
        if str(payload.get("model_fingerprint")) != str(model_fingerprint):
            raise NativeStateBankError(
                "session-initial GDN artifact model fingerprint is stale"
            )
        _validate_session_initial_gdn_state(payload)
        rank_prompt_tokens = int(payload.get("prompt_tokens", -1))
        rank_completion_tokens = int(payload.get("completion_tokens", -1))
        if rank_prompt_tokens < 1 or rank_completion_tokens < 0:
            raise NativeStateBankError(
                "session-initial GDN artifact token counts are invalid"
            )
        if prompt_tokens is None:
            prompt_tokens = rank_prompt_tokens
            completion_tokens = rank_completion_tokens
        elif (
            rank_prompt_tokens != prompt_tokens
            or rank_completion_tokens != completion_tokens
        ):
            raise NativeStateBankError(
                "session-initial GDN TP ranks disagree on token counts"
            )
    assert prompt_tokens is not None and completion_tokens is not None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


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
    layer_id: int | None = None,
) -> torch.Tensor:
    """Load one full-attention layer's raw K heads from every TP rank.

    Every full-attention layer's K is exported per page, so the recall layer
    is a load-time choice. ``layer_id`` defaults to the final layer.
    """

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
        selected_layer = int(layer_ids[-1]) if layer_id is None else int(layer_id)
        if selected_layer not in layer_ids:
            raise NativeStateBankError(
                f"native Bank artifact has no Full-Attention layer {selected_layer}; "
                f"available layers: {list(layer_ids)}"
            )
        final_layer = layers.get(str(selected_layer))
        if not isinstance(final_layer, dict) or "key" not in final_layer:
            raise NativeStateBankError(
                f"native Bank artifact lacks raw K for layer {selected_layer}"
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
    if str(getattr(hf_config, "model_type", "")) in {"qwen4_exp", "qwen4_exp_text"}:
        # Fallback identities still bind all hybrid/indexer/PLE/RoPE geometry.
        import json
        config_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)
        return stable_digest(model_path, json.dumps(config_dict, sort_keys=True, default=str))
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
    # Reserved recurrent-state slots holding decoded session-initial GDN
    # artifacts, keyed by state identity in least-recently-used order. Two
    # entries let requests queued under the previous identity still bind
    # while a refreshed identity starts arriving.
    session_gdn_sources: OrderedDict[str, torch.Tensor] = field(
        init=False, default_factory=OrderedDict, repr=False
    )
    session_gdn_loads: int = field(init=False, default=0)
    session_gdn_binds: int = field(init=False, default=0)
    session_gdn_lock: threading.RLock = field(
        init=False, default_factory=threading.RLock, repr=False
    )

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.full_layer_ids = _full_layer_ids(self.model_config)
        self.layers = _language_layers(self.model)
        self.model_fingerprint = _model_fingerprint(self.model_config)
        self.hits = 0
        self.misses = 0
        self.loads = 0
        self.exports = 0
        self.session_gdn_sources = OrderedDict()
        self.session_gdn_loads = 0
        self.session_gdn_binds = 0

    def _qwen4_layout(self) -> dict[str, Any] | None:
        hf_config = getattr(self.model_config, "hf_config", self.model_config)
        config = getattr(self.model_config, "hf_text_config", None) or hf_config
        if str(getattr(config, "model_type", "")) not in {"qwen4_exp", "qwen4_exp_text"}:
            architectures = tuple(getattr(hf_config, "architectures", ()) or ())
            if "Qwen4ExpForConditionalGeneration" not in architectures:
                return None
        pool = self.kv_pool
        if not hasattr(pool, "qsa_compress_ratio"):
            raise NativeStateBankError("Qwen4 native Bank requires its compressed QSA pool")
        short = getattr(self.req_pool, "short_conv_pool", None)
        ngram = getattr(self.req_pool, "ngram_pool", None)
        if short is None or ngram is None or not short.enabled or not ngram.enabled:
            raise NativeStateBankError("Qwen4 native Bank requires complete PLE state pools")
        return {
            "compress_ratio": int(pool.qsa_compress_ratio),
            "index_heads": int(pool.qsa_index_kv_heads),
            "index_head_dim": int(pool.qsa_index_head_dim),
            "token_topk": int(pool.qsa_token_topk),
            "short_conv_layers": tuple(short.layer_map),
            "short_conv_shape": tuple(short.conv_state.shape[2:]),
            "ngram_shape": tuple(ngram.context.shape[1:]),
            "ngram_eos_token_id": int(ngram.eos_token_id),
        }

    def _export_qwen4_state(
        self, req: Any, mapping: torch.Tensor, capture_start: int,
        capture_count: int, physical_mamba: torch.Tensor,
    ) -> dict[str, Any] | None:
        layout = self._qwen4_layout()
        if layout is None:
            return None
        ratio = layout["compress_ratio"]
        write_pos = getattr(self.req_pool.mamba_pool, "replayssm_write_pos", None)
        if write_pos is not None and bool((write_pos[physical_mamba.long()] != 0).any().item()):
            raise NativeStateBankError("Qwen4 native export requires fully flushed GDN ReplaySSM")
        if capture_start % ratio or capture_count % ratio:
            raise NativeStateBankError("Qwen4 native export must contain complete compression groups")
        if capture_start + capture_count != len(req.origin_input_ids):
            raise NativeStateBankError("Qwen4 native export must end at its complete document state")
        groups = mapping.reshape(-1, ratio)
        if bool((groups // ratio != groups[:, :1] // ratio).any().item()):
            raise NativeStateBankError("Qwen4 KV mapping splits a compression group")
        compressed_slots = groups[:, 0] // ratio
        source_positions = torch.arange(
            capture_start, capture_start + capture_count, ratio,
            device=mapping.device, dtype=torch.long,
        )
        ring_slots = torch.arange(ratio, device=mapping.device) + int(req.req_pool_idx) * ratio
        layers = {}
        for layer_id in self.full_layer_ids:
            compressed = self.kv_pool.get_qsa_compressed_k_buffer(layer_id)[compressed_slots]
            layers[str(layer_id)] = {
                # Compression keys are already normalized, but carry source RoPE.
                "compressed_key": compressed.cpu().clone(),
                "pending_key": self.kv_pool.get_qsa_key_state_buffer(layer_id)[ring_slots].cpu().clone(),
            }
        return {
            "schema": _QWEN4_STATE_SCHEMA, **layout,
            "capture_start": capture_start,
            "source_group_positions": source_positions.cpu(),
            "layers": layers,
            "pending_rope": self.kv_pool.get_qsa_rope_position_buffer(ring_slots).cpu().clone(),
            "pending_count": len(req.origin_input_ids) % ratio,
            "short_conv": self.req_pool.short_conv_pool.conv_state[:, physical_mamba].cpu().clone(),
            "ngram": self.req_pool.ngram_pool.context[physical_mamba.long()].cpu().clone(),
        }

    def _validate_qwen4_state(
        self, payload: dict[str, Any], local_positions: tuple[int, ...],
    ) -> None:
        layout = self._qwen4_layout()
        state = payload.get("qwen4_state")
        if layout is None:
            if state is not None:
                raise NativeStateBankError("Qwen4 state cannot restore into a non-Qwen4 pool")
            return
        if not isinstance(state, dict) or state.get("schema") != _QWEN4_STATE_SCHEMA:
            raise NativeStateBankError("Qwen4 artifact lacks complete QSA and PLE state; rebuild the Bank")
        if any(state.get(key) != value for key, value in layout.items()):
            raise NativeStateBankError("Qwen4 QSA/PLE layout is stale")
        ratio = layout["compress_ratio"]
        if len(local_positions) % ratio:
            raise NativeStateBankError("Qwen4 selection splits compression groups")
        for offset in range(0, len(local_positions), ratio):
            start = local_positions[offset]
            if start % ratio or local_positions[offset:offset + ratio] != tuple(range(start, start + ratio)):
                raise NativeStateBankError("Qwen4 selection must preserve complete ordered source groups")
        count = int(payload["capture_count"])
        if count <= 0 or count % ratio:
            raise NativeStateBankError("Qwen4 artifact contains an incomplete compression group")
        expected_groups = count // ratio
        layers = state.get("layers")
        if not isinstance(layers, dict) or set(layers) != {str(i) for i in self.full_layer_ids}:
            raise NativeStateBankError("Qwen4 QSA layer map is incomplete")
        for value in layers.values():
            if not isinstance(value, dict):
                raise NativeStateBankError("Qwen4 QSA layer payload is invalid")
            for name, shape in (
                ("compressed_key", (expected_groups, layout["index_heads"], layout["index_head_dim"])),
                ("pending_key", (ratio, layout["index_heads"], layout["index_head_dim"])),
            ):
                tensor = value.get(name)
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                    raise NativeStateBankError(f"Qwen4 {name} shape/dtype is stale")
        for name, shape, dtype in (
            ("short_conv", (len(layout["short_conv_layers"]), 1, *layout["short_conv_shape"]), self.req_pool.short_conv_pool.conv_state.dtype),
            ("ngram", (1, *layout["ngram_shape"]), self.req_pool.ngram_pool.context.dtype),
            ("pending_rope", (ratio, 3), torch.int64),
            ("source_group_positions", (expected_groups,), torch.int64),
        ):
            tensor = state.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise NativeStateBankError(f"Qwen4 {name} shape/dtype is stale")
        capture_start = int(state.get("capture_start", -1))
        if capture_start < 0 or capture_start % ratio or not torch.equal(
            state["source_group_positions"], torch.arange(capture_start, capture_start + count, ratio)
        ):
            raise NativeStateBankError("Qwen4 group RoPE provenance is stale")
        if int(state.get("pending_count", -1)) != 0:
            raise NativeStateBankError("Qwen4 native artifact ends inside a pending compression group")

    def _restore_qwen4_state(
        self, payload: dict[str, Any], local_positions: tuple[int, ...],
        kv_indices: torch.Tensor, physical_mamba: torch.Tensor,
    ) -> None:
        state = payload.get("qwen4_state")
        if state is None:
            return
        ratio = int(state["compress_ratio"])
        groups = kv_indices.long().reshape(-1, ratio)
        if bool((groups // ratio != groups[:, :1] // ratio).any().item()):
            raise NativeStateBankError("Qwen4 restored KV allocation splits compression groups")
        selected_groups = torch.tensor(local_positions[::ratio], dtype=torch.long) // ratio
        compressed_slots = groups[:, 0] // ratio
        positions = torch.arange(0, len(local_positions), ratio, device=kv_indices.device)
        for layer_id in self.full_layer_ids:
            compressed_source = state["layers"][str(layer_id)]["compressed_key"][selected_groups].to(kv_indices.device)
            source_positions = state["source_group_positions"][selected_groups].to(kv_indices.device)
            raw = _inverse_rotary_key(
                compressed_source.float(), positions=source_positions, rotary=self.layers[layer_id].rotary_emb,
            )
            compressed = _apply_rotary_key(raw, positions=positions, rotary=self.layers[layer_id].rotary_emb)
            self.kv_pool.set_qsa_compressed_k_buffer(layer_id, compressed_slots, compressed)
        # Prefixes are page/group aligned: there are no live pending ring members.
        # Stored ring rows retain export provenance but must not be treated as
        # pending members of the selected (possibly sparse) virtual prefix.
        self.req_pool.short_conv_pool.load_cpu_slots(state["short_conv"], physical_mamba)
        self.req_pool.ngram_pool.load_cpu_slots(state["ngram"], physical_mamba)

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
        session_export = params.get("qwen_exo_session_initial_gdn_export")
        if isinstance(session_export, dict):
            req.qwen_exo_native_bank_no_cache = True
            local_error: Exception | None = None
            try:
                self._export_session_initial_gdn(req, session_export)
            except Exception as exc:
                local_error = exc
            if not self.consensus(local_error is None):
                raise local_error or NativeStateBankError(
                    "session-initial GDN export failed on a peer TP rank"
                )
            req.qwen_exo_session_initial_gdn_status = "exported"
            self.exports += 1
            return True

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

    def _export_session_initial_gdn(self, req: Any, export: dict[str, Any]) -> None:
        source_digest = str(export.get("source_digest") or "")
        state_identity = str(export.get("state_identity") or "")
        if not _SAFE_DIGEST.fullmatch(source_digest) or not _SAFE_DIGEST.fullmatch(
            state_identity
        ):
            raise NativeStateBankError(
                "session-initial GDN export has an invalid identity"
            )
        mamba_pool = getattr(self.req_pool, "mamba_pool", None)
        if mamba_pool is None or req.mamba_pool_idx is None:
            raise NativeStateBankError(
                "session-initial GDN export has no active recurrent state"
            )
        physical_mamba = self.req_pool.translate_mamba_indices(
            req.mamba_pool_idx.reshape(1)
        )
        write_pos = getattr(mamba_pool, "replayssm_write_pos", None)
        if write_pos is not None and bool(
            (write_pos[physical_mamba.to(dtype=torch.long)] != 0).any().item()
        ):
            raise NativeStateBankError(
                "session-initial GDN export requires a fully flushed ReplaySSM state"
            )
        components = ["conv", "temporal"]
        if getattr(mamba_pool, "replayssm_cache_base", None) is not None:
            components.append("replayssm_cursors")
        if getattr(mamba_pool, "_slot_siblings", ()):
            components.append("slot_siblings")
        payload = {
            "schema": _SESSION_INITIAL_GDN_SCHEMA,
            "source_digest": source_digest,
            "state_identity": state_identity,
            "rank": self.rank,
            "world_size": self.world_size,
            "model_fingerprint": self.model_fingerprint,
            "prompt_tokens": len(req.origin_input_ids),
            "completion_tokens": len(req.output_ids_through_stop),
            "mamba_state": mamba_pool.get_cpu_copy(physical_mamba),
            "mamba_state_components": tuple(components),
        }
        _atomic_torch_save(
            payload,
            _session_initial_gdn_path(
                self.root,
                source_digest=source_digest,
                state_identity=state_identity,
                rank=self.rank,
            ),
        )

    @staticmethod
    def _session_initial_gdn_selection(req: Any) -> dict[str, Any] | None:
        selection = _custom_params(req).get("qwen_exo_session_initial_gdn")
        return selection if isinstance(selection, dict) else None

    def _validate_session_initial_gdn_payload(self, payload: dict[str, Any]) -> None:
        if int(payload.get("world_size", -1)) != self.world_size:
            raise NativeStateBankError(
                "session-initial GDN artifact TP world size is stale"
            )
        if str(payload.get("model_fingerprint")) != self.model_fingerprint:
            raise NativeStateBankError(
                "session-initial GDN artifact model fingerprint is stale"
            )
        mamba_pool = getattr(self.req_pool, "mamba_pool", None)
        state = payload.get("mamba_state")
        if (
            mamba_pool is None
            or not isinstance(state, (tuple, list))
            or len(state) not in {2, 3, 4}
        ):
            raise NativeStateBankError(
                "session-initial GDN artifact has an invalid recurrent payload"
            )
        _validate_session_initial_gdn_state(payload)
        siblings = tuple(getattr(mamba_pool, "_slot_siblings", ()))
        has_spec_cursors = getattr(mamba_pool, "replayssm_cache_base", None) is not None
        expected_length = 2 + int(has_spec_cursors) + int(bool(siblings))
        if len(state) != expected_length:
            raise NativeStateBankError("session-initial GDN/Ple state layout is stale")
        if siblings:
            sources = state[-1]
            if not isinstance(sources, (tuple, list)) or len(sources) != len(siblings):
                raise NativeStateBankError("session-initial PLE side states are incomplete")
            for sibling, source in zip(siblings, sources):
                target = getattr(sibling, "conv_state", None)
                if target is not None:
                    shape = (target.shape[0], 1, *target.shape[2:])
                else:
                    target = sibling.context
                    shape = (1, *target.shape[1:])
                if not isinstance(source, torch.Tensor) or tuple(source.shape) != shape or source.dtype != target.dtype:
                    raise NativeStateBankError("session-initial PLE state shape/dtype is stale")
        conv, temporal = state[:2]
        current_conv = tuple(mamba_pool.mamba_cache.conv)
        if not isinstance(conv, (tuple, list)) or len(conv) != len(current_conv):
            raise NativeStateBankError(
                "session-initial GDN artifact convolution layout is stale"
            )
        for source, target in zip(conv, current_conv):
            if (
                not isinstance(source, torch.Tensor)
                or source.ndim != target.ndim
                or source.dtype != target.dtype
                or source.shape[0] != target.shape[0]
                or source.shape[1] != 1
                or tuple(source.shape[2:]) != tuple(target.shape[2:])
            ):
                raise NativeStateBankError(
                    "session-initial GDN artifact convolution shape is stale"
                )
        target_temporal = mamba_pool.mamba_cache.temporal
        if (
            not isinstance(temporal, torch.Tensor)
            or temporal.ndim != target_temporal.ndim
            or temporal.dtype != target_temporal.dtype
            or temporal.shape[0] != target_temporal.shape[0]
            or temporal.shape[1] != 1
            or tuple(temporal.shape[2:]) != tuple(target_temporal.shape[2:])
        ):
            raise NativeStateBankError(
                "session-initial GDN artifact temporal shape is stale"
            )
        if len(state) - int(bool(siblings)) != 2 + int(has_spec_cursors):
            raise NativeStateBankError(
                "session-initial GDN artifact ReplaySSM layout is stale"
            )
        if has_spec_cursors:
            cursors = state[2]
            if not isinstance(cursors, (tuple, list)) or len(cursors) != 3:
                raise NativeStateBankError(
                    "session-initial GDN artifact ReplaySSM cursors are invalid"
                )
            if any(
                not isinstance(cursor, torch.Tensor)
                or cursor.numel() != 1
                or bool((cursor != 0).any().item())
                for cursor in cursors
            ):
                raise NativeStateBankError(
                    "session-initial GDN artifact contains an unflushed ReplaySSM ring"
                )

    def ensure_session_initial_gdn_source(self, req: Any) -> bool:
        with self.session_gdn_lock:
            return self._ensure_session_initial_gdn_source(req)

    @staticmethod
    def _session_initial_gdn_identity(selection: dict[str, Any]) -> tuple[str, str]:
        source_digest = str(selection.get("source_digest") or "")
        state_identity = str(selection.get("state_identity") or "")
        if not _SAFE_DIGEST.fullmatch(source_digest) or not _SAFE_DIGEST.fullmatch(
            state_identity
        ):
            raise NativeStateBankError(
                "session-initial GDN selection has an invalid identity"
            )
        return source_digest, state_identity

    def _ensure_session_initial_gdn_source(self, req: Any) -> bool:
        selection = self._session_initial_gdn_selection(req)
        if selection is None:
            return False
        source_digest, state_identity = self._session_initial_gdn_identity(selection)
        if state_identity in self.session_gdn_sources:
            self.session_gdn_sources.move_to_end(state_identity)
            return True

        payload: dict[str, Any] | None = None
        local_error: Exception | None = None
        try:
            payload = _load_session_initial_gdn_payload(
                self.root,
                source_digest=source_digest,
                state_identity=state_identity,
                rank=self.rank,
            )
            self._validate_session_initial_gdn_payload(payload)
        except Exception as exc:
            local_error = exc
        if not self.consensus(local_error is None):
            raise local_error or NativeStateBankError(
                "session-initial GDN artifact is unavailable on a peer TP rank"
            )
        assert payload is not None

        mamba_allocator = getattr(self.req_pool, "mamba_allocator", None)
        if mamba_allocator is None:
            raise NativeStateBankError(
                "session-initial GDN restore requires a recurrent-state allocator"
            )
        source_index = self._reserve_session_initial_gdn_slot(mamba_allocator)

        load_error: Exception | None = None
        try:
            physical_source = self.req_pool.translate_mamba_indices(source_index)
            self.req_pool.mamba_pool.load_cpu_copy(
                payload["mamba_state"], physical_source
            )
        except Exception as exc:
            load_error = exc
        if not self.consensus(load_error is None):
            mamba_allocator.free(source_index)
            raise load_error or NativeStateBankError(
                "session-initial GDN source load failed on a peer TP rank"
            )
        self.session_gdn_sources[state_identity] = source_index
        self.session_gdn_loads += 1
        return True

    def _reserve_session_initial_gdn_slot(self, mamba_allocator: Any) -> torch.Tensor:
        """Return a recurrent-state slot for a new identity.

        The least recently used reserved slot is recycled once the small
        reserve is full; otherwise one slot is allocated, evicting cached
        recurrent state when the pool is exhausted. All TP ranks must agree.
        """
        if len(self.session_gdn_sources) >= _SESSION_GDN_MAX_SOURCES:
            _stale_identity, recycled = self.session_gdn_sources.popitem(last=False)
            return recycled
        source_index = mamba_allocator.alloc(1)
        if source_index is None:
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams

            self.tree_cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            source_index = mamba_allocator.alloc(1)
        if not self.consensus(source_index is not None):
            if source_index is not None:
                mamba_allocator.free(source_index)
            raise NativeStateBankError(
                "session-initial GDN source allocation failed across TP ranks"
            )
        assert source_index is not None
        return source_index

    def bind_session_initial_gdn(self, req: Any) -> bool:
        with self.session_gdn_lock:
            return self._bind_session_initial_gdn(req)

    def _bind_session_initial_gdn(self, req: Any) -> bool:
        selection = self._session_initial_gdn_selection(req)
        if selection is None:
            return False
        self._ensure_session_initial_gdn_source(req)
        _source_digest, state_identity = self._session_initial_gdn_identity(selection)
        source_index = self.session_gdn_sources.get(state_identity)
        if source_index is None:
            raise NativeStateBankError(
                "session-initial GDN source was not prepared before scheduling"
            )
        cache_namespace = str(selection.get("cache_namespace") or "")
        if cache_namespace and not str(req.extra_key or "").startswith(cache_namespace):
            raise NativeStateBankError(
                "session-initial GDN selection is outside its session cache namespace"
            )
        if req.mamba_cow_src_index is not None:
            req.qwen_exo_session_initial_gdn_status = "session_cache_hit"
            return True
        if len(req.prefix_indices) != 0:
            raise NativeStateBankError(
                "session-initial GDN cannot replace an unmatched cached prefix state"
            )
        req.mamba_cow_src_index = source_index
        req.mamba_needs_clear = False
        req.qwen_exo_session_initial_gdn_status = "bound"
        self.session_gdn_binds += 1
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
            # Read through the pool so packed FP4 caches are dequantized.
            key, value = self.kv_pool.get_kv_tokens(layer_id, mapping, torch.bfloat16)
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
        conv_states, temporal_states = mamba_pool.get_cpu_copy(physical_mamba)[:2]
        qwen4_state = self._export_qwen4_state(
            req, mapping, capture_start, capture_count, physical_mamba,
        )
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
        if qwen4_state is not None:
            payload["qwen4_state"] = qwen4_state
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
        self._validate_qwen4_state(payload, local_positions)

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
            state = (conv, temporal)
            if payload.get("qwen4_state") is not None:
                # Complete PLE slot siblings accompany the GDN slot lifecycle.
                if getattr(self.req_pool.mamba_pool, "replayssm_cache_base", None) is not None:
                    state = (*state, tuple(torch.zeros(1, dtype=torch.int64) for _ in range(3)))
                state = (*state, [payload["qwen4_state"]["short_conv"], payload["qwen4_state"]["ngram"]])
            self.req_pool.mamba_pool.load_cpu_copy(state, physical_mamba)
            self._restore_qwen4_state(payload, local_positions, kv_indices, physical_mamba)
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

    def reserved_mamba_slots(self) -> int:
        with self.session_gdn_lock:
            return len(self.session_gdn_sources)

    def stats(self) -> dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "loads": int(self.loads),
            "exports": int(self.exports),
            "session_gdn_loads": int(self.session_gdn_loads),
            "session_gdn_binds": int(self.session_gdn_binds),
        }


__all__ = [
    "NativeStateBankError",
    "NativeStateBankManager",
    "load_page_key_heads",
    "load_page_compression_ratio",
    "validate_page_artifacts",
    "validate_session_initial_gdn_artifacts",
]
