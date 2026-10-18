from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.hybrid_state import qwen_exo_model_state_directory

_SESSION_INITIAL_GDN_SCHEMA = "qwen-exo-session-initial-gdn-v1"
_SESSION_GDN_MAX_SOURCES = 2
_SAFE_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class NativeStateBankError(RuntimeError):
    """A native Bank artifact or scheduler restore violated its contract."""


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
        self.exports = 0
        self.session_gdn_sources = OrderedDict()
        self.session_gdn_loads = 0
        self.session_gdn_binds = 0

        # Prefixes are page/group aligned: there are no live pending ring members.
        # Stored ring rows retain export provenance but must not be treated as
        # pending members of the selected (possibly sparse) virtual prefix.

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

        return False

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
        self._validate_session_initial_gdn_payload(payload)
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

    def reserved_mamba_slots(self) -> int:
        with self.session_gdn_lock:
            return len(self.session_gdn_sources)

    def stats(self) -> dict[str, int]:
        return {
            "exports": int(self.exports),
            "session_gdn_loads": int(self.session_gdn_loads),
            "session_gdn_binds": int(self.session_gdn_binds),
        }


__all__ = [
    "NativeStateBankError",
    "NativeStateBankManager",
    "validate_session_initial_gdn_artifacts",
]
