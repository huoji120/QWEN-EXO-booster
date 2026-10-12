import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import SimpleNamespace

from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.memory_refresh import AdaptiveMemoryRefreshGate
from qwen_exo_booster.observer import InFlightObserver, ObserverResult
from qwen_exo_booster.pipeline import MemoryPipeline, MemoryPreparationState
from qwen_exo_booster.policy_data import PolicyDataAttachment
from qwen_exo_booster.telemetry import TelemetryStore


def gate(**kwargs):
    return AdaptiveMemoryRefreshGate(
        max_conversations=2, surprisal_threshold=6.0,
        surprisal_margin=0.2, window_tokens=4, **kwargs,
    )


def memory(request_id="initial", *, attachment="synthetic reference"):
    policy = PolicyDataAttachment(
        "policy-source", "policy-attachment", ("policy",), ("policy-digest",),
        3, "Synthetic policy instructions",
    )
    return MemoryPreparationState(
        request_id=request_id, previous_response_id=None,
        question_digest="old-question", retrieval_question_digest="old-question",
        source_digest="source", policy_source_digest=policy.source_digest,
        policy_document_ids=policy.document_ids,
        policy_document_digests=policy.document_digests,
        policy_attachment_digest=policy.attachment_digest,
        policy_attached_tokens=policy.attached_tokens,
        candidates=(), decisions=(), selected_document_ids=("document",),
        selected_reference_digests=("reference-digest",),
        attachment_digest="attachment", cache_namespace="old-key",
        attached_tokens=2 if attachment else 0, private_attachment=attachment,
        policy_attachment=policy, policy_instructions="old client and policy",
        policy_cache_namespace="old-key", original_instructions="old client",
        original_extra_key="old-key", created_at=0.0,
        retrieval_latency_seconds=10.0, judge_latency_seconds=5.0,
        judge_executed_count=3, query_probe_status="ready",
        query_probe_prompt_tokens=100, qk_shortlist_size=8,
    )


def observation(request_id, samples, *, start=0):
    last = samples[-1] if samples else (0.0, 0.0, 0.0)
    return ObserverResult(
        request_id=request_id, token_count=start + len(samples), new_tokens=len(samples),
        current_surprisal=last[0], local_window_mean=last[1], history_mean=last[2],
        ema_surprisal=last[0], max_surprisal=max((row[0] for row in samples), default=0),
        latest_q_drift=None, latest_memory_energy=None, triggered=False,
        trigger_reasons=(), events=(), finished=False, refresh_signals=tuple(samples),
    )


def test_three_spikes_must_be_consecutive_and_ignore_duplicate_observations():
    async def scenario():
        g = gate(cooldown_turns=0)
        async with g.conversation("conversation") as state:
            g.decide(state, request_id="initial", revision="v1")
            g.commit(state, memory(), revision="v1")
            high, low = (8.0, 2.0, 1.0), (0.0, 2.0, 1.0)
            first = observation("initial", [high, high, low, high, high])
            g.observe("conversation", first)
            g.observe("conversation", first)
            assert state.pending_reason is None
            g.observe("conversation", observation("initial", [high], start=5))
            assert state.pending_reason == "sustained_high_surprisal"
            decision = g.decide(state, request_id="next", revision="v1")
            assert decision.retrieve
            assert decision.high_surprisal_streak == 3
    asyncio.run(scenario())


def test_high_average_requires_two_complete_nonoverlapping_windows():
    async def scenario():
        g = gate(cooldown_turns=0)
        async with g.conversation("conversation") as state:
            g.decide(state, request_id="initial", revision="v1")
            g.commit(state, memory(), revision="v1")
            # Below the relative spike threshold, but absolutely high uncertainty.
            high = (7.0, 7.0, 7.0)
            g.observe("conversation", observation("initial", [high] * 7))
            assert state.high_average_streak == 1
            assert state.pending_reason is None
            g.observe("conversation", observation("initial", [high], start=7))
            assert state.pending_reason == "sustained_high_average"
    asyncio.run(scenario())


def test_refresh_waits_three_turns_then_resets_evidence():
    async def scenario():
        g = gate()
        async with g.conversation("conversation") as state:
            assert g.decide(state, request_id="initial", revision="v1").retrieve
            g.commit(state, memory(), revision="v1")
            g.observe("conversation", observation("initial", [(8, 2, 1)] * 3))
            for turn in range(3):
                decision = g.decide(state, request_id=f"turn-{turn}", revision="v1")
                assert not decision.retrieve
                assert decision.reason == "cooldown"
                assert decision.cooldown_remaining == 3 - turn
            decision = g.decide(state, request_id="refresh", revision="v1")
            assert decision.retrieve
            assert decision.reason == "sustained_high_surprisal"
            g.commit(state, memory("refresh"), revision="v1")
            assert state.pending_reason is None
            for turn in range(4):
                decision = g.decide(state, request_id=f"calm-{turn}", revision="v1")
                assert not decision.retrieve
            assert decision.reason == "no_sustained_uncertainty"
    asyncio.run(scenario())


def test_missing_signals_and_stale_requests_do_not_force_refresh():
    async def scenario():
        g = gate(cooldown_turns=0)
        async with g.conversation("conversation") as state:
            g.decide(state, request_id="initial", revision="v1")
            g.commit(state, memory(attachment=None), revision="v1")
            g.decide(state, request_id="current", revision="v1")
            g.observe("conversation", observation("initial", [(8, 2, 1)] * 3))
            g.observe("conversation", observation("current", []))
            assert not g.decide(state, request_id="next", revision="v1").retrieve
            # Empty admission is cached too, rather than re-running Judge forever.
            assert state.memory.private_attachment is None
    asyncio.run(scenario())


def test_changed_sources_invalidate_cache_even_during_cooldown():
    async def scenario():
        g = gate()
        async with g.conversation("conversation") as state:
            g.decide(state, request_id="initial", revision="v1")
            g.commit(state, memory(), revision="v1")
            decision = g.decide(state, request_id="next", revision="v2")
            assert decision.retrieve and decision.reason == "source_changed"
            assert state.memory is None
    asyncio.run(scenario())


def test_cache_isolation_eviction_and_failed_preparation_retry():
    async def scenario():
        g = gate()
        async with g.conversation("a") as state:
            g.decide(state, request_id="a1", revision="v1")
            g.commit(state, memory("a1"), revision="v1")
        async with g.conversation("b") as state:
            assert state.memory is None
            assert g.decide(state, request_id="b1", revision="v1").retrieve
            # Simulate an exception before commit: the next turn must retry.
            assert g.decide(state, request_id="b2", revision="v1").retrieve
        async with g.conversation("c"):
            pass
        async with g.conversation("a") as state:
            assert state.memory is None
            assert g.decide(state, request_id="a2", revision="v1").retrieve
    asyncio.run(scenario())


def test_real_observer_gate_is_independent_of_stream_chunking(tmp_path):
    async def scenario(cumulative):
        g = gate(cooldown_turns=0)
        observer = InFlightObserver(
            TelemetryStore(tmp_path / f"trace-{cumulative}.jsonl"), mode="off",
            surprisal_threshold=6.0, surprisal_window=4,
        )
        values = [1.0] * 8 + [8.0] * 3
        async with g.conversation("conversation") as state:
            g.decide(state, request_id="initial", revision="v1")
            g.commit(state, memory(), revision="v1")
            ends = range(1, len(values) + 1) if cumulative else [len(values)]
            for end in ends:
                result = observer.observe_generation_result("initial", {
                    "meta_info": {"output_token_logprobs": [(-x, i) for i, x in enumerate(values[:end])]},
                })
                g.observe("conversation", result)
            return g.decide(state, request_id="next", revision="v1")
    batched = asyncio.run(scenario(False))
    streamed = asyncio.run(scenario(True))
    assert batched == streamed
    assert batched.retrieve and batched.reason == "sustained_high_surprisal"


@dataclass(frozen=True)
class Request:
    request_id: str
    instructions: str
    extra_key: str
    input: str = "current input"
    qwen_exo_memory_attachment: str | None = None

    def model_copy(self, update):
        return replace(self, **update)


def test_reused_attachment_preserves_new_client_fields_and_clears_stale_metrics():
    async def scenario():
        pipeline = MemoryPipeline.__new__(MemoryPipeline)
        pipeline._lock = asyncio.Lock()
        pipeline._states = OrderedDict()
        pipeline.max_states = 4
        cached = memory()
        current = Request("current", "new client instructions", "new client key")
        prepared, state = await pipeline.reuse_responses_request(
            current, cached, retrieval_question="current question",
            memory_previous_response_id="effective-parent",
            published_previous_response_id="api-parent",
            decision_context=SimpleNamespace(digest="current-context", token_count=42),
        )
        assert prepared.input == current.input
        assert prepared.instructions == "new client instructions\n\nSynthetic policy instructions"
        assert prepared.qwen_exo_memory_attachment == cached.private_attachment
        assert state.original_instructions == current.instructions
        assert state.original_extra_key == current.extra_key
        assert state.previous_response_id == "api-parent"
        assert state.effective_memory_previous_response_id == "effective-parent"
        assert state.request_id == "current"
        assert state.question_digest == stable_digest("current question")
        assert state.decision_context_digest == "current-context"
        assert state.decision_context_tokens == 42
        assert state.memory_reused_from_request_id == "initial"
        assert state.selected_document_ids == cached.selected_document_ids
        assert state.retrieval_latency_seconds == state.judge_latency_seconds == 0
        assert state.judge_executed_count == state.qk_shortlist_size == 0
        assert state.query_probe_prompt_tokens == 0
        assert state.qk_rank_audit["status"] == "not_run"
        assert state.query_heads == state.query_states == ()
        assert await pipeline.get_state("current") == state
        assert cached.original_instructions == "old client"
    asyncio.run(scenario())
