import os
from types import SimpleNamespace

import pytest
from qwen_exo_booster.contracts import (
    ContractViolation,
    HybridLifecycleState,
    HybridStateNamespace,
)
from qwen_exo_booster.hybrid_state import (
    HybridRequestPhase,
    HybridRuntimePolicy,
)


def args(**overrides):
    values = {
        "tp_size": 2,
        "dtype": "bfloat16",
        "page_size": 64,
        "mamba_radix_cache_strategy": "extra_buffer",
        "disable_radix_cache": False,
        "disaggregation_mode": "null",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hybrid_policy_accepts_single_gpu_cuda(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    policy = HybridRuntimePolicy.from_server_args(args(tp_size=1))

    assert policy.tp_size == 1
    assert policy.backend == "cuda"


def test_hybrid_policy_rejects_unsupported_cuda_tp(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    with pytest.raises(ValueError, match="tp_size=1 or tp_size=2"):
        HybridRuntimePolicy.from_server_args(args(tp_size=3))


def test_hybrid_policy_accepts_single_gpu_mlx(monkeypatch):
    monkeypatch.delenv("SGLANG_MAMBA_SSM_DTYPE", raising=False)
    policy = HybridRuntimePolicy.from_server_args(
        args(
            qwen_exo_backend="mlx",
            tp_size=1,
            dtype="float16",
            page_size=1,
            mamba_radix_cache_strategy="no_buffer",
            quantization="mlx_q4",
            kv_cache_dtype="auto",
        )
    )

    assert policy.backend == "mlx"
    assert policy.topology_key == "mlx-tp1-float16-mlx-q4-auto"


def test_hybrid_policy_rejects_mlx_tp(monkeypatch):
    monkeypatch.delenv("SGLANG_MAMBA_SSM_DTYPE", raising=False)
    with pytest.raises(ValueError, match="MLX requires tp_size=1"):
        HybridRuntimePolicy.from_server_args(
            args(
                qwen_exo_backend="mlx",
                tp_size=2,
                dtype="float16",
                page_size=1,
                mamba_radix_cache_strategy="no_buffer",
            )
        )


def test_hybrid_policy_rejects_non_atomic_mamba_cache(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    with pytest.raises(ValueError, match="extra-buffer"):
        HybridRuntimePolicy.from_server_args(
            args(mamba_radix_cache_strategy="no_buffer", page_size=1)
        )


def test_hybrid_policy_bounds_logprob_chunk_and_workspace_reserve(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    monkeypatch.delenv("SGLANG_LOGPROB_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB", raising=False)

    policy = HybridRuntimePolicy.from_server_args(args())

    assert policy.logprob_chunk_size == 512
    assert policy.workspace_safety_reserve_bytes == 512 << 20
    assert os.environ["SGLANG_LOGPROB_CHUNK_SIZE"] == "512"


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("SGLANG_LOGPROB_CHUNK_SIZE", "513", "in \\[1, 512\\]"),
        (
            "SGLANG_LOGPROB_CHUNK_SIZE",
            "not-an-int",
            "must be integers",
        ),
        (
            "SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB",
            "511",
            "at least 512 MiB",
        ),
    ],
)
def test_hybrid_policy_rejects_unsafe_workspace_environment(
    monkeypatch, name, value, error
):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    monkeypatch.setenv("SGLANG_LOGPROB_CHUNK_SIZE", "512")
    monkeypatch.setenv("SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB", "512")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=error):
        HybridRuntimePolicy.from_server_args(args())


def test_external_memory_and_request_prefixes_use_distinct_namespaces():
    logical_identity = "same-prefix"
    request_key = HybridRuntimePolicy.namespace_key(
        HybridStateNamespace.REQUEST_PREFIX, logical_identity
    )
    memory_key = HybridRuntimePolicy.namespace_key(
        HybridStateNamespace.EXTERNAL_MEMORY, logical_identity
    )

    assert request_key != memory_key
    assert request_key == HybridRuntimePolicy.namespace_key(
        HybridStateNamespace.REQUEST_PREFIX, logical_identity
    )


def test_boundary_fingerprint_covers_token_order():
    fields = {
        "model_fingerprint": "model",
        "tokenizer_fingerprint": "tokenizer",
        "namespace_key": "namespace",
    }
    forward = HybridRuntimePolicy.boundary_fingerprint(token_ids=[1, 2, 3], **fields)
    reversed_tokens = HybridRuntimePolicy.boundary_fingerprint(
        token_ids=[3, 2, 1], **fields
    )

    assert forward != reversed_tokens


def test_hybrid_policy_rejects_disaggregation(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")

    with pytest.raises(ValueError, match="disaggregation_mode=null"):
        HybridRuntimePolicy.from_server_args(args(disaggregation_mode="decode"))


@pytest.mark.parametrize(
    "unsupported",
    ["enable_hierarchical_cache", "enable_session_radix_cache"],
)
def test_hybrid_policy_rejects_unhooked_cache_lifecycles(monkeypatch, unsupported):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")

    with pytest.raises(ValueError, match="does not support"):
        HybridRuntimePolicy.from_server_args(args(**{unsupported: True}))


def test_request_state_binds_native_evidence_and_transitions_atomically(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    policy = HybridRuntimePolicy.from_server_args(args())
    state = policy.new_request_state(
        request_id="request-1",
        token_ids=[1, 2, 3],
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
    )

    assert state.phase is HybridRequestPhase.ADMITTED
    assert state.handle.lifecycle is HybridLifecycleState.NEW
    assert not state.handle.has_any_component

    state = policy.bind_prefill(
        state,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
        native_req_pool_idx=4,
        full_kv_blocks=(11,),
        recurrent_state_slots=(7,),
        conv_state_slots=(7,),
    )
    assert state.phase is HybridRequestPhase.PREFILL
    assert state.native_req_pool_idx == 4
    assert state.handle.full_kv_blocks == (11,)
    assert state.handle.recurrent_state_slots == (7,)

    state = policy.cache_prefill(
        state,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
    )
    assert state.handle.lifecycle is HybridLifecycleState.CACHED
    state = policy.bind_prefill(
        state,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
        native_req_pool_idx=5,
        full_kv_blocks=(21,),
        recurrent_state_slots=(8,),
        conv_state_slots=(8,),
    )
    state = policy.begin_decode(
        state,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
    )
    assert state.phase is HybridRequestPhase.DECODE
    assert state.handle.lifecycle is HybridLifecycleState.ACTIVE
    assert state.handle.full_kv_blocks == (21,)

    state = policy.release_request_state(
        state,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
    )
    assert state.phase is HybridRequestPhase.RELEASED
    assert state.handle.lifecycle is HybridLifecycleState.RELEASED
    assert not state.handle.has_any_component
    assert state.native_req_pool_idx is None


def test_request_state_rejects_namespace_reuse(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    policy = HybridRuntimePolicy.from_server_args(args())
    state = policy.new_request_state(
        request_id="request-1",
        token_ids=[1],
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        namespace=HybridStateNamespace.EXTERNAL_MEMORY,
    )

    with pytest.raises(ContractViolation, match="namespace"):
        policy.bind_prefill(
            state,
            request_id="request-1",
            namespace=HybridStateNamespace.REQUEST_PREFIX,
            native_req_pool_idx=1,
            full_kv_blocks=(2,),
            recurrent_state_slots=(3,),
            conv_state_slots=(3,),
        )


def test_cached_prefix_survives_request_and_reuses_across_request_ids(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    policy = HybridRuntimePolicy.from_server_args(args())
    common = {
        "token_ids": [1, 2, 3, 4],
        "model_fingerprint": "model",
        "tokenizer_fingerprint": "tokenizer",
        "tp_rank": 0,
        "generation": 0,
        "full_kv_blocks": (10, 11),
        "recurrent_state_slots": (20, 21, 22),
        "conv_state_slots": (20, 21, 22),
    }
    cached = policy.new_cached_prefix_state(request_id="request-1", **common)
    candidate = policy.new_cached_prefix_state(request_id="request-2", **common)

    released_request = policy.release_request_state(
        cached,
        request_id="request-1",
        namespace=HybridStateNamespace.REQUEST_PREFIX,
    )
    assert released_request.phase is HybridRequestPhase.RELEASED
    assert cached.phase is HybridRequestPhase.CACHED

    policy.assert_cached_reusable(cached, candidate)
    assert cached.handle.request_id == "request-1"
    assert candidate.handle.request_id == "request-2"
    assert cached.handle.prefix_identity == candidate.handle.prefix_identity
    assert cached.handle.full_kv_blocks == (10, 11)
    external = policy.new_cached_prefix_state(
        request_id="request-3",
        namespace=HybridStateNamespace.EXTERNAL_MEMORY,
        **common,
    )
    with pytest.raises(ContractViolation, match="namespace"):
        policy.assert_cached_reusable(cached, external)
    corrupted_common = dict(common)
    corrupted_common["recurrent_state_slots"] = (20, 21, 99)
    corrupted = policy.new_cached_prefix_state(
        request_id="request-4", **corrupted_common
    )
    with pytest.raises(ContractViolation, match="native component mismatch"):
        policy.assert_cached_reusable(cached, corrupted)


def test_cached_prefix_eviction_clears_all_native_evidence(monkeypatch):
    monkeypatch.setenv("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")
    policy = HybridRuntimePolicy.from_server_args(args())
    cached = policy.new_cached_prefix_state(
        request_id="request-1",
        token_ids=[1, 2],
        model_fingerprint="model",
        tokenizer_fingerprint="tokenizer",
        tp_rank=0,
        generation=0,
        full_kv_blocks=(10,),
        recurrent_state_slots=(20,),
        conv_state_slots=(20,),
    )

    evicted = policy.evict_cached_state(cached)

    assert evicted.phase is HybridRequestPhase.EVICTED
    assert evicted.handle.lifecycle is HybridLifecycleState.EVICTED
    assert not evicted.handle.has_any_component
    with pytest.raises(ContractViolation, match="not resident"):
        policy.assert_cached_reusable(evicted, cached)
