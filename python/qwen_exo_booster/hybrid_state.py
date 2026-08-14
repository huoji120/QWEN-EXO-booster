from __future__ import annotations

import os
from array import array
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from qwen_exo_booster.contracts import (
    ContractViolation,
    HybridLifecycleState,
    HybridStateHandle,
    HybridStateNamespace,
    stable_digest,
)

_QWEN_EXO_LOGPROB_CHUNK_SIZE = 512
_QWEN_EXO_MIN_WORKSPACE_SAFETY_RESERVE_MIB = 512


def resolve_qwen_exo_backend(server_args: Any) -> str:
    """Return the model-execution backend that owns QWEN-EXO native state."""

    explicit = str(getattr(server_args, "qwen_exo_backend", "") or "").lower()
    if explicit:
        if explicit not in {"cuda", "mlx"}:
            raise ValueError("QWEN-EXO backend must be cuda or mlx")
        return explicit
    use_mlx = str(os.getenv("SGLANG_USE_MLX", "")).strip().lower()
    if use_mlx in {"1", "true", "yes", "on"}:
        return "mlx"
    return "cuda"


def _topology_component(value: object) -> str:
    normalized = "".join(
        character if character.isalnum() else "-"
        for character in str(value or "none").strip().lower()
    ).strip("-")
    return normalized or "none"


def qwen_exo_topology_key(
    *,
    backend: str,
    tp_size: int,
    dtype: str,
    quantization: str | None = None,
    kv_cache_dtype: str | None = None,
) -> str:
    """Stable namespace for model-native artifacts that cannot cross topologies."""

    return "-".join(
        (
            _topology_component(backend),
            f"tp{int(tp_size)}",
            _topology_component(dtype),
            _topology_component(quantization),
            _topology_component(kv_cache_dtype),
        )
    )


def qwen_exo_model_state_directory(server_args: Any) -> Path:
    topology = qwen_exo_topology_key(
        backend=resolve_qwen_exo_backend(server_args),
        tp_size=int(server_args.tp_size),
        dtype=str(getattr(server_args, "dtype", None) or "auto"),
        quantization=str(getattr(server_args, "quantization", None) or "none"),
        kv_cache_dtype=str(getattr(server_args, "kv_cache_dtype", None) or "auto"),
    )
    return Path(server_args.qwen_exo_state_dir).expanduser() / "model-native" / topology


class HybridRequestPhase(str, Enum):
    ADMITTED = "admitted"
    PREFILL = "prefill"
    DECODE = "decode"
    CACHED = "cached"
    EVICTED = "evicted"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class HybridRequestState:
    """Logical request state bound to native SGLang-owned allocations."""

    handle: HybridStateHandle
    phase: HybridRequestPhase
    native_req_pool_idx: int | None = None

    def __post_init__(self) -> None:
        if self.native_req_pool_idx is not None and self.native_req_pool_idx < 0:
            raise ContractViolation("native_req_pool_idx cannot be negative")
        if self.phase is HybridRequestPhase.ADMITTED:
            if self.handle.lifecycle is not HybridLifecycleState.NEW:
                raise ContractViolation("Admitted hybrid state must have a new handle")
            if self.native_req_pool_idx is not None:
                raise ContractViolation(
                    "Admitted hybrid state cannot claim native allocation"
                )
        elif self.phase is HybridRequestPhase.RELEASED:
            if self.handle.lifecycle is not HybridLifecycleState.RELEASED:
                raise ContractViolation(
                    "Released hybrid state must have a released handle"
                )
            if self.native_req_pool_idx is not None:
                raise ContractViolation(
                    "Released hybrid state cannot retain native allocation"
                )
        elif self.phase is HybridRequestPhase.EVICTED:
            if self.handle.lifecycle is not HybridLifecycleState.EVICTED:
                raise ContractViolation(
                    "Evicted hybrid state must have an evicted handle"
                )
            if self.native_req_pool_idx is not None:
                raise ContractViolation(
                    "Evicted hybrid state cannot retain native allocation"
                )
        elif self.phase is HybridRequestPhase.CACHED:
            if self.handle.lifecycle is not HybridLifecycleState.CACHED:
                raise ContractViolation("Cached hybrid state must have a cached handle")
            if self.native_req_pool_idx is not None:
                raise ContractViolation(
                    "Cached hybrid state cannot retain request-pool binding"
                )
        elif self.handle.lifecycle is not HybridLifecycleState.ACTIVE:
            raise ContractViolation(
                "Prefill/decode hybrid state requires an active handle"
            )
        elif self.native_req_pool_idx is None:
            raise ContractViolation(
                "Active prefill/decode state requires a native request-pool binding"
            )


@dataclass(frozen=True, slots=True)
class HybridRuntimePolicy:
    tp_size: int
    dtype: str
    page_size: int
    mamba_strategy: str
    mamba_state_dtype: str
    mamba_conv_dtype: str = "bfloat16"

    logprob_chunk_size: int = _QWEN_EXO_LOGPROB_CHUNK_SIZE
    workspace_safety_reserve_bytes: int = (
        _QWEN_EXO_MIN_WORKSPACE_SAFETY_RESERVE_MIB << 20
    )
    backend: str = "cuda"
    quantization: str = "none"
    kv_cache_dtype: str = "auto"

    @classmethod
    def from_server_args(cls, server_args: Any) -> HybridRuntimePolicy:
        chunk_value = os.environ.setdefault(
            "SGLANG_LOGPROB_CHUNK_SIZE", str(_QWEN_EXO_LOGPROB_CHUNK_SIZE)
        )
        reserve_mib_value = os.getenv(
            "SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB",
            str(_QWEN_EXO_MIN_WORKSPACE_SAFETY_RESERVE_MIB),
        )
        try:
            chunk_size = int(chunk_value)
            reserve_mib = int(reserve_mib_value)
        except ValueError as exc:
            raise ValueError(
                "QWEN-EXO workspace environment values must be integers"
            ) from exc
        backend = resolve_qwen_exo_backend(server_args)
        policy = cls(
            tp_size=int(server_args.tp_size),
            dtype=str(server_args.dtype),
            page_size=int(server_args.page_size),
            mamba_strategy=str(server_args.mamba_radix_cache_strategy),
            mamba_state_dtype=os.getenv("SGLANG_MAMBA_SSM_DTYPE", "").lower(),
            mamba_conv_dtype=os.getenv(
                "SGLANG_MAMBA_CONV_DTYPE",
                os.getenv("SGLANG_MAMBA_SSM_DTYPE", ""),
            ).lower(),
            backend=backend,
            quantization=str(getattr(server_args, "quantization", None) or "none"),
            kv_cache_dtype=str(getattr(server_args, "kv_cache_dtype", None) or "auto"),
            logprob_chunk_size=chunk_size,
            workspace_safety_reserve_bytes=reserve_mib << 20,
        )
        policy.validate(server_args)
        return policy

    def validate(self, server_args: Any) -> None:
        if not 1 <= self.logprob_chunk_size <= _QWEN_EXO_LOGPROB_CHUNK_SIZE:
            raise ValueError("QWEN-EXO requires SGLANG_LOGPROB_CHUNK_SIZE in [1, 512]")
        if self.workspace_safety_reserve_bytes < (
            _QWEN_EXO_MIN_WORKSPACE_SAFETY_RESERVE_MIB << 20
        ):
            raise ValueError(
                "QWEN-EXO requires workspace safety reserve of at least 512 MiB"
            )
        if self.backend == "mlx":
            if self.tp_size != 1:
                raise ValueError("QWEN-EXO MLX requires tp_size=1")
            if self.dtype not in {"auto", "float16", "bfloat16", "half"}:
                raise ValueError(
                    "QWEN-EXO MLX requires auto, float16, or bfloat16 weights"
                )
            if self.mamba_strategy not in {"auto", "no_buffer"}:
                raise ValueError(
                    "QWEN-EXO MLX requires the no-buffer auxiliary-state radix cache"
                )
            if self.page_size < 1:
                raise ValueError("QWEN-EXO MLX page_size must be positive")
        else:
            if self.tp_size not in {1, 2}:
                raise ValueError("QWEN-EXO CUDA supports tp_size=1 or tp_size=2")
            if self.quantization in {"gptq", "gptq_marlin"}:
                if self.dtype not in {"float16", "half"}:
                    raise ValueError("QWEN-EXO CUDA GPTQ requires float16 activations")
            elif self.dtype != "bfloat16":
                raise ValueError(
                    "QWEN-EXO CUDA correctness baseline requires bfloat16 weights"
                )

            expected_mamba_state_dtype = (
                "float16"
                if self.quantization in {"gptq", "gptq_marlin"}
                else "bfloat16"
            )
            if self.mamba_state_dtype != expected_mamba_state_dtype:
                raise ValueError(
                    "QWEN-EXO CUDA requires SGLANG_MAMBA_SSM_DTYPE="
                    f"{expected_mamba_state_dtype} for quantization={self.quantization}"
                )
            if self.mamba_conv_dtype != expected_mamba_state_dtype:
                raise ValueError(
                    "QWEN-EXO CUDA requires SGLANG_MAMBA_CONV_DTYPE="
                    f"{expected_mamba_state_dtype} for quantization={self.quantization}"
                )

            if self.mamba_strategy not in {"extra_buffer", "extra_buffer_lazy"}:
                raise ValueError(
                    "QWEN-EXO CUDA hybrid state requires an extra-buffer Mamba radix cache"
                )
            if self.page_size != 64:
                raise ValueError(
                    "QWEN-EXO CUDA extra-buffer baseline requires page_size=64"
                )
        disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
        disaggregation_mode = getattr(disaggregation_mode, "value", disaggregation_mode)
        if disaggregation_mode not in {None, "null"}:
            raise ValueError(
                "QWEN-EXO requires disaggregation_mode=null; "
                f"received {disaggregation_mode!r}"
            )
        if bool(getattr(server_args, "enable_hierarchical_cache", False)):
            raise ValueError(
                "QWEN-EXO hybrid-prefix lifecycle does not support hierarchical cache"
            )
        if bool(getattr(server_args, "enable_session_radix_cache", False)):
            raise ValueError(
                "QWEN-EXO hybrid-prefix lifecycle does not support session radix cache"
            )

    @property
    def topology_key(self) -> str:
        return qwen_exo_topology_key(
            backend=self.backend,
            tp_size=self.tp_size,
            dtype=self.dtype,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    @staticmethod
    def namespace_key(namespace: HybridStateNamespace, logical_identity: str) -> str:
        if not logical_identity.strip():
            raise ValueError("Hybrid cache namespace identity cannot be empty")
        return (
            f"qwen-exo:v1:{namespace.value}:" f"{stable_digest(logical_identity)[:24]}"
        )

    @staticmethod
    def boundary_fingerprint(
        *,
        token_ids: Iterable[int],
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        namespace_key: str,
    ) -> str:
        ids = (
            token_ids
            if isinstance(token_ids, array) and token_ids.typecode == "q"
            else array("q", (int(token_id) for token_id in token_ids))
        )
        return stable_digest(
            model_fingerprint,
            tokenizer_fingerprint,
            namespace_key,
            ids.tobytes().hex(),
        )

    def new_handle(
        self,
        *,
        handle_id: str,
        request_id: str,
        prefix_identity: str,
        boundary_fingerprint: str,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        tp_rank: int,
        sequence_length: int,
        namespace: HybridStateNamespace,
    ) -> HybridStateHandle:
        return HybridStateHandle(
            handle_id=handle_id,
            request_id=request_id,
            prefix_identity=prefix_identity,
            boundary_fingerprint=boundary_fingerprint,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tp_world_size=self.tp_size,
            tp_rank=tp_rank,
            sequence_length=sequence_length,
            namespace=namespace,
        )

    def new_request_state(
        self,
        *,
        request_id: str,
        token_ids: Iterable[int],
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        tp_rank: int,
        generation: int = 0,
        namespace: HybridStateNamespace = HybridStateNamespace.REQUEST_PREFIX,
    ) -> HybridRequestState:
        token_ids = (
            token_ids
            if isinstance(token_ids, array) and token_ids.typecode == "q"
            else array("q", (int(token_id) for token_id in token_ids))
        )
        sequence_length = len(token_ids)
        layout_fingerprint = stable_digest(
            model_fingerprint,
            self.dtype,
            self.tp_size,
            self.page_size,
            self.mamba_strategy,
            self.mamba_state_dtype,
            self.mamba_conv_dtype,
        )
        logical_identity = stable_digest(
            layout_fingerprint,
            tokenizer_fingerprint,
            token_ids.tobytes().hex(),
        )
        prefix_identity = self.namespace_key(namespace, logical_identity)
        boundary_fingerprint = self.boundary_fingerprint(
            token_ids=token_ids,
            model_fingerprint=layout_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            namespace_key=prefix_identity,
        )
        handle = self.new_handle(
            handle_id=stable_digest(
                "qwen-exo-hybrid",
                request_id,
                namespace.value,
                int(generation),
                boundary_fingerprint,
            ),
            request_id=request_id,
            prefix_identity=prefix_identity,
            boundary_fingerprint=boundary_fingerprint,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tp_rank=tp_rank,
            sequence_length=sequence_length,
            namespace=namespace,
        )
        return HybridRequestState(handle=handle, phase=HybridRequestPhase.ADMITTED)

    @staticmethod
    def assert_request_binding(
        state: HybridRequestState,
        *,
        request_id: str,
        namespace: HybridStateNamespace,
    ) -> None:
        mismatched = []
        if state.handle.request_id != request_id:
            mismatched.append("request_id")
        if state.handle.namespace is not namespace:
            mismatched.append("namespace")
        if mismatched:
            raise ContractViolation(
                f"Hybrid request binding mismatch: {sorted(set(mismatched))}"
            )

    def bind_prefill(
        self,
        state: HybridRequestState,
        *,
        request_id: str,
        namespace: HybridStateNamespace,
        native_req_pool_idx: int,
        full_kv_blocks: Iterable[int],
        recurrent_state_slots: Iterable[int],
        conv_state_slots: Iterable[int],
    ) -> HybridRequestState:
        self.assert_request_binding(state, request_id=request_id, namespace=namespace)
        full_kv_blocks = tuple(full_kv_blocks)
        recurrent_state_slots = tuple(recurrent_state_slots)
        conv_state_slots = tuple(conv_state_slots)
        if state.phase not in {
            HybridRequestPhase.ADMITTED,
            HybridRequestPhase.PREFILL,
            HybridRequestPhase.CACHED,
            HybridRequestPhase.EVICTED,
        }:
            raise ContractViolation(
                f"Cannot bind prefill while request is {state.phase.value}"
            )

        handle = state.handle
        if handle.lifecycle in {HybridLifecycleState.NEW, HybridLifecycleState.EVICTED}:
            handle = handle.bind_components(
                full_kv_blocks=full_kv_blocks,
                recurrent_state_slots=recurrent_state_slots,
                conv_state_slots=conv_state_slots,
            )
        elif handle.lifecycle is HybridLifecycleState.CACHED:
            handle = handle.transition(HybridLifecycleState.ACTIVE)
            handle = replace(
                handle,
                full_kv_blocks=full_kv_blocks,
                recurrent_state_slots=recurrent_state_slots,
                conv_state_slots=conv_state_slots,
            )
        elif handle.lifecycle is HybridLifecycleState.ACTIVE:
            handle = replace(
                handle,
                full_kv_blocks=full_kv_blocks,
                recurrent_state_slots=recurrent_state_slots,
                conv_state_slots=conv_state_slots,
            )
        else:
            raise ContractViolation(
                f"Cannot bind released/suspended hybrid handle: {handle.lifecycle.value}"
            )
        return HybridRequestState(
            handle=handle,
            phase=HybridRequestPhase.PREFILL,
            native_req_pool_idx=int(native_req_pool_idx),
        )

    def cache_prefill(
        self,
        state: HybridRequestState,
        *,
        request_id: str,
        namespace: HybridStateNamespace,
    ) -> HybridRequestState:
        self.assert_request_binding(state, request_id=request_id, namespace=namespace)
        if state.phase is not HybridRequestPhase.PREFILL:
            raise ContractViolation(
                f"Cannot cache prefill while request is {state.phase.value}"
            )
        handle = state.handle
        if handle.lifecycle is HybridLifecycleState.ACTIVE:
            handle = handle.transition(HybridLifecycleState.CACHED)
        elif handle.lifecycle is not HybridLifecycleState.CACHED:
            raise ContractViolation(
                f"Cannot cache hybrid handle while {handle.lifecycle.value}"
            )
        return replace(
            state,
            handle=handle,
            phase=HybridRequestPhase.CACHED,
            native_req_pool_idx=None,
        )

    def new_cached_prefix_state(
        self,
        *,
        request_id: str,
        token_ids: Iterable[int],
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        tp_rank: int,
        generation: int,
        full_kv_blocks: Iterable[int],
        recurrent_state_slots: Iterable[int],
        conv_state_slots: Iterable[int],
        namespace: HybridStateNamespace = HybridStateNamespace.REQUEST_PREFIX,
    ) -> HybridRequestState:
        state = self.new_request_state(
            request_id=request_id,
            token_ids=token_ids,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tp_rank=tp_rank,
            generation=generation,
            namespace=namespace,
        )
        handle = state.handle.bind_components(
            full_kv_blocks=full_kv_blocks,
            recurrent_state_slots=recurrent_state_slots,
            conv_state_slots=conv_state_slots,
        ).transition(HybridLifecycleState.CACHED)
        return HybridRequestState(handle=handle, phase=HybridRequestPhase.CACHED)

    def new_evicted_prefix_state(
        self,
        *,
        request_id: str,
        token_ids: Iterable[int],
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        tp_rank: int,
        generation: int,
        namespace: HybridStateNamespace = HybridStateNamespace.REQUEST_PREFIX,
    ) -> HybridRequestState:
        state = self.new_request_state(
            request_id=request_id,
            token_ids=token_ids,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            tp_rank=tp_rank,
            generation=generation,
            namespace=namespace,
        )
        return HybridRequestState(
            handle=replace(state.handle, lifecycle=HybridLifecycleState.EVICTED),
            phase=HybridRequestPhase.EVICTED,
        )

    @staticmethod
    def assert_prefix_identity(
        left: HybridRequestState, right: HybridRequestState
    ) -> None:
        fields = (
            "prefix_identity",
            "boundary_fingerprint",
            "model_fingerprint",
            "tokenizer_fingerprint",
            "tp_world_size",
            "sequence_length",
            "namespace",
        )
        mismatched = [
            field
            for field in fields
            if getattr(left.handle, field) != getattr(right.handle, field)
        ]
        if mismatched:
            raise ContractViolation(
                f"Hybrid prefix reuse fingerprint mismatch: {mismatched}"
            )

    @staticmethod
    def assert_cached_reusable(
        cached: HybridRequestState, candidate: HybridRequestState
    ) -> None:
        if cached.phase is not HybridRequestPhase.CACHED:
            raise ContractViolation(
                f"Hybrid prefix is not resident: {cached.phase.value}"
            )
        if candidate.phase is not HybridRequestPhase.CACHED:
            raise ContractViolation("Hybrid reuse candidate must be cached")
        cached.handle.assert_reusable_with(candidate.handle)
        component_fields = (
            "full_kv_blocks",
            "recurrent_state_slots",
            "conv_state_slots",
        )
        mismatched = [
            field
            for field in component_fields
            if getattr(cached.handle, field) != getattr(candidate.handle, field)
        ]
        if mismatched:
            raise ContractViolation(
                f"Hybrid prefix native component mismatch: {mismatched}"
            )

    @staticmethod
    def evict_cached_state(state: HybridRequestState) -> HybridRequestState:
        if state.phase is HybridRequestPhase.EVICTED:
            return state
        if state.phase is not HybridRequestPhase.CACHED:
            raise ContractViolation(
                f"Cannot evict hybrid prefix while {state.phase.value}"
            )
        return HybridRequestState(
            handle=state.handle.transition(HybridLifecycleState.EVICTED),
            phase=HybridRequestPhase.EVICTED,
        )

    def begin_decode(
        self,
        state: HybridRequestState,
        *,
        request_id: str,
        namespace: HybridStateNamespace,
    ) -> HybridRequestState:
        self.assert_request_binding(state, request_id=request_id, namespace=namespace)
        if state.phase is HybridRequestPhase.DECODE:
            return state
        if state.phase is not HybridRequestPhase.PREFILL:
            raise ContractViolation(
                f"Cannot begin decode while request is {state.phase.value}"
            )
        handle = state.handle
        if handle.lifecycle is HybridLifecycleState.ACTIVE:
            handle = handle.transition(HybridLifecycleState.CACHED)
        if handle.lifecycle is not HybridLifecycleState.CACHED:
            raise ContractViolation(
                f"Cannot begin decode while handle is {handle.lifecycle.value}"
            )
        handle = handle.transition(HybridLifecycleState.ACTIVE)
        return replace(state, handle=handle, phase=HybridRequestPhase.DECODE)

    def release_request_state(
        self,
        state: HybridRequestState,
        *,
        request_id: str,
        namespace: HybridStateNamespace,
    ) -> HybridRequestState:
        self.assert_request_binding(state, request_id=request_id, namespace=namespace)
        if state.phase is HybridRequestPhase.RELEASED:
            return state
        handle = state.handle.transition(HybridLifecycleState.RELEASED)
        return HybridRequestState(
            handle=handle,
            phase=HybridRequestPhase.RELEASED,
            native_req_pool_idx=None,
        )
