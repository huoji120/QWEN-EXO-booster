import time

import pytest
from qwen_exo_booster.contracts import (
    CancellationToken,
    ContractViolation,
    EligibilityDecision,
    EligibilityStatus,
    HybridLifecycleState,
    HybridStateHandle,
    InternalJob,
    InternalJobType,
    ResourceEstimate,
)


def new_handle(**overrides):
    values = {
        "handle_id": "handle-1",
        "request_id": "request-1",
        "prefix_identity": "prefix-1",
        "boundary_fingerprint": "boundary-1",
        "model_fingerprint": "model-1",
        "tokenizer_fingerprint": "tokenizer-1",
        "tp_world_size": 2,
        "tp_rank": 0,
        "sequence_length": 128,
    }
    values.update(overrides)
    return HybridStateHandle(**values)


def test_hybrid_state_requires_all_components_when_resident():
    with pytest.raises(ContractViolation, match="requires Full-Attention"):
        new_handle(
            lifecycle=HybridLifecycleState.ACTIVE,
            full_kv_blocks=(1,),
            recurrent_state_slots=(2,),
        )


def test_hybrid_state_evicts_all_components_atomically():
    active = new_handle().bind_components(
        full_kv_blocks=(1, 2),
        recurrent_state_slots=(3,),
        conv_state_slots=(4,),
    )
    cached = active.transition(HybridLifecycleState.CACHED)
    evicted = cached.transition(HybridLifecycleState.EVICTED)

    assert evicted.lifecycle is HybridLifecycleState.EVICTED
    assert not evicted.has_any_component


def test_hybrid_prefix_reuse_rejects_boundary_mismatch():
    left = new_handle().bind_components(
        full_kv_blocks=(1,), recurrent_state_slots=(2,), conv_state_slots=(3,)
    )
    right = new_handle(
        handle_id="handle-2", boundary_fingerprint="different"
    ).bind_components(
        full_kv_blocks=(4,), recurrent_state_slots=(5,), conv_state_slots=(6,)
    )

    with pytest.raises(ContractViolation, match="boundary_fingerprint"):
        left.assert_reusable_with(right)


def test_semantic_eligibility_fails_closed():
    decision = EligibilityDecision.create(
        candidate_id="candidate-1",
        parent_request_id="request-1",
        question="question",
        reference="reference",
        status=EligibilityStatus.INVALID,
        judge_method="strict_binary",
        judge_model_fingerprint="model-1",
        decision_margin=None,
    )

    assert not decision.eligible
    with pytest.raises(ContractViolation, match="not semantically eligible"):
        decision.require_eligible()


def test_internal_job_rejects_recursion_and_honors_cancellation():
    fields = {
        "parent_request_id": "request-1",
        "turn_id": "turn-1",
        "job_id": "job-1",
        "job_type": InternalJobType.REFERENCE_JUDGE,
        "priority": -1,
        "shared_prefix_key": "prefix-1",
        "token_budget": 16,
        "state_budget_bytes": 1024,
        "deadline_monotonic": time.monotonic() + 10,
        "cancellation_token": CancellationToken("cancel-1"),
        "telemetry_correlation_id": "trace-1",
        "max_fanout": 4,
    }
    with pytest.raises(ContractViolation, match="recursively"):
        InternalJob(**fields, recursion_depth=1)

    job = InternalJob(**fields)
    assert not job.is_cancelled_or_expired()
    cancelled = InternalJob(
        **{**fields, "cancellation_token": fields["cancellation_token"].cancel()}
    )
    assert cancelled.is_cancelled_or_expired()


def test_resource_estimate_preserves_parent_reserve():
    estimate = ResourceEstimate(
        full_attention_kv_bytes=40,
        recurrent_state_bytes=10,
        conv_state_bytes=5,
        internal_branch_bytes=10,
        scoring_workspace_bytes=5,
        output_workspace_bytes=10,
        safety_reserve_bytes=20,
    )

    assert estimate.total_bytes == 100
    assert estimate.fits(120, parent_reserved_bytes=20)
    assert not estimate.fits(119, parent_reserved_bytes=20)
