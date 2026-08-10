from dataclasses import replace

import pytest
from qwen_exo_booster.causal_replay import CausalReplayService
from qwen_exo_booster.contracts import (
    EligibilityDecision,
    EligibilityStatus,
    stable_digest,
)
from qwen_exo_booster.internal_jobs import InternalScoreResult
from qwen_exo_booster.knowledge import KnowledgeCandidate, NativePrefixSelection
from qwen_exo_booster.observer import MidThinkEvent
from qwen_exo_booster.telemetry import TelemetryStore


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in str(text)]


class _UnexpectedRunner:
    async def run_score_batch(self, jobs, input_ids, label_starts, sampling_params):
        raise AssertionError("rejected replay must not be scored")


class _EmptyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return []


class _FailingRunner:
    async def run_score_batch(self, jobs, input_ids, label_starts, sampling_params):
        raise RuntimeError("score failed")


class _SuccessfulRunner:
    async def run_score_batch(self, jobs, input_ids, label_starts, sampling_params):
        return tuple(
            InternalScoreResult(
                job=job,
                token_logprobs=tuple((-2.0 if index == 0 else -1.0) for _ in range(8)),
                mean_nll=2.0 if index == 0 else 1.0,
                prompt_tokens=len(input_ids[index]),
                finish_reason={"type": "stop"},
                latency_seconds=0.01,
            )
            for index, job in enumerate(jobs)
        )


class _NativeRunner:
    def __init__(self):
        self.call = None

    async def run_score_batch(
        self,
        jobs,
        input_ids,
        label_starts,
        sampling_params,
        *,
        custom_params_per_job,
        extra_keys,
    ):
        self.call = (
            tuple(tuple(item) for item in input_ids),
            tuple(label_starts),
            tuple(custom_params_per_job),
            tuple(extra_keys),
        )
        return await _SuccessfulRunner().run_score_batch(
            jobs, input_ids, label_starts, sampling_params
        )


def _event(
    request_id: str, event_id: str, *, generation_token_index: int = 2
) -> MidThinkEvent:
    return MidThinkEvent(
        event_id=event_id,
        request_id=request_id,
        token_index=2,
        trigger_reasons=("selected_token_surprisal_window",),
        current_surprisal=7.0,
        window_mean=6.0,
        history_mean=1.0,
        ema_surprisal=4.0,
        recovery_window_mean=6.0,
        uncertainty_state="persistent_uncertainty",
        pre_q_sketches=(),
        post_q_sketches=(),
        generation_token_index=generation_token_index,
    )


def _candidate_and_decision(request_id: str, event_id: str):
    reference = "Use the verified constant."
    candidate = KnowledgeCandidate(
        candidate_id=f"candidate-{event_id}",
        document_id="document-1",
        relative_path="reference.md",
        score=1.0,
        lexical_score=1.0,
        quality_prior=1.0,
        canonical=True,
        reference_digest=stable_digest(reference),
        reference_content=reference,
        normalized_reference_content=reference,
    )
    decision = EligibilityDecision.create(
        candidate_id=candidate.candidate_id,
        parent_request_id=request_id,
        question="Which constant is required?",
        reference=reference,
        status=EligibilityStatus.ELIGIBLE,
        judge_method="strict-test",
        judge_model_fingerprint="model",
        decision_margin=0.0,
    )
    return candidate, decision


def _completed_events(telemetry: TelemetryStore, request_id: str):
    return tuple(
        event
        for event in telemetry.events(request_id)
        if event.event_type == "causal_replay.completed"
    )


def _assert_terminal_event(telemetry, request_id, result):
    completed = _completed_events(telemetry, request_id)
    assert len(completed) == 1
    assert completed[0].payload == result.public_dict()
    assert completed[0].payload["replay_decision"]
    assert completed[0].payload["latency_seconds"] is not None
    assert completed[0].payload["latency_seconds"] >= 0.0


@pytest.mark.asyncio
async def test_insufficient_future_observation_emits_one_completed_event(tmp_path):
    request_id = "request-short-observation"
    telemetry = TelemetryStore(tmp_path / "short-observation.jsonl")
    service = CausalReplayService(_UnexpectedRunner(), _Tokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, "event-short"),
        prompt_ids=(1, 2, 3),
        output_ids=(10, 11, 12),
        candidates=(),
        decisions=(),
    )

    assert result.decision == "insufficient_future_observation"
    assert result.scheduled_next_turn is False
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_judge_mismatch_emits_one_completed_event(tmp_path):
    request_id = "request-judge-mismatch"
    event_id = "event-judge-mismatch"
    candidate, mismatched_decision = _candidate_and_decision(
        "different-request", event_id
    )
    telemetry = TelemetryStore(tmp_path / "judge-mismatch.jsonl")
    service = CausalReplayService(_UnexpectedRunner(), _Tokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id),
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(10)),
        candidates=(candidate,),
        decisions=(mismatched_decision,),
    )

    assert result.decision == "reject_no_semantic_candidate"
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_empty_prefix_emits_one_completed_event(tmp_path):
    request_id = "request-empty-prefix"
    event_id = "event-empty-prefix"
    candidate, decision = _candidate_and_decision(request_id, event_id)
    telemetry = TelemetryStore(tmp_path / "empty-prefix.jsonl")
    service = CausalReplayService(_UnexpectedRunner(), _Tokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id, generation_token_index=0),
        prompt_ids=(),
        output_ids=tuple(range(8)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "reject_empty_prefix"
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_empty_candidate_state_emits_one_completed_event(tmp_path):
    request_id = "request-empty-candidate"
    event_id = "event-empty-candidate"
    candidate, decision = _candidate_and_decision(request_id, event_id)
    telemetry = TelemetryStore(tmp_path / "empty-candidate.jsonl")
    service = CausalReplayService(_UnexpectedRunner(), _EmptyTokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id),
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(10)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "reject_empty_candidate_state"
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_scored_replay_emits_one_completed_event_with_winner_fields(tmp_path):
    request_id = "request-scored"
    event_id = "event-scored"
    candidate, decision = _candidate_and_decision(request_id, event_id)
    telemetry = TelemetryStore(tmp_path / "scored.jsonl")
    service = CausalReplayService(_SuccessfulRunner(), _Tokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id),
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(10)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "shadow_would_switch"
    assert result.maybe_decision == "admit_maybe"
    assert result.winner_candidate_id == candidate.candidate_id
    assert result.winner_gain == 1.0
    assert result.winner_kl is not None
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_score_exception_emits_one_failed_closed_completed_event(tmp_path):
    request_id = "request-score-failure"
    event_id = "event-score-failure"
    candidate, decision = _candidate_and_decision(request_id, event_id)
    telemetry = TelemetryStore(tmp_path / "score-failure.jsonl")
    service = CausalReplayService(_FailingRunner(), _Tokenizer(), telemetry)

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id),
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(10)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "failed_closed:RuntimeError"
    assert result.maybe_decision == "not_compiled"
    assert result.scheduled_next_turn is False
    assert [event.event_type for event in telemetry.events(request_id)].count(
        "causal_replay.failed_closed"
    ) == 1
    _assert_terminal_event(telemetry, request_id, result)


@pytest.mark.asyncio
async def test_native_replay_loads_bank_prefix_instead_of_reprefilling_reference(
    tmp_path,
):
    request_id = "request-native"
    event_id = "event-native"
    candidate, decision = _candidate_and_decision(request_id, event_id)
    native = NativePrefixSelection(
        source_digest="a" * 64,
        page_id=4,
        document_id=candidate.document_id,
        local_positions=tuple(range(64)),
        source_positions=tuple(range(512, 576)),
        token_ids=tuple(range(1000, 1064)),
        prefix_identity="prefix-identity",
        radix_namespace="qwen-exo:v1:tensor-bank-native:test",
    )
    candidate = replace(candidate, native_prefix=native)
    runner = _NativeRunner()
    service = CausalReplayService(
        runner, _Tokenizer(), TelemetryStore(tmp_path / "native.jsonl")
    )

    result = await service.evaluate(
        parent_request_id=request_id,
        event=_event(request_id, event_id),
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(10)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "shadow_would_switch"
    branch_ids, label_starts, custom_params, extra_keys = runner.call
    assert branch_ids[1][:64] == native.token_ids
    assert label_starts[1] == 64 + 5
    assert custom_params[0] == {}
    assert custom_params[1] == {
        "qwen_exo_native_bank_selection": native.scheduler_payload()
    }
    assert extra_keys[1] == native.radix_namespace
