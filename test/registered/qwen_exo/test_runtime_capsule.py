import asyncio
import json
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.capsule import CapsuleUpdateInput
from qwen_exo_booster.compaction import CompactionSummary
from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags
from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.hybrid_state import HybridRuntimePolicy
from qwen_exo_booster.observer import MidThinkEvent, ObserverResult
from qwen_exo_booster.query_probe import QueryStateSpan
from qwen_exo_booster.runtime import (
    PendingReflectionMemory,
    QwenExoCapacityConflict,
    QwenExoRuntime,
    QwenExoRuntimeState,
)


@dataclass(frozen=True)
class FakeRequest:
    request_id: str
    input: object
    previous_response_id: str | None = None
    instructions: str | None = None
    extra_key: str | None = None
    background: bool = False
    user: str | None = None
    session_id: str | None = None
    prompt_cache_key: str | None = None

    def model_copy(self, update):
        return replace(self, **update)


def config(tmp_path):
    return QwenExoConfig(
        state_directory=tmp_path / "state",
        knowledge_directory=tmp_path / "knowledge",
        max_internal_fanout=8,
        max_internal_tokens=1024,
        max_candidates=8,
        max_memory_tokens=256,
        observer_mode="shadow",
        feature_flags=QwenExoFeatureFlags(
            hybrid_prefix=True,
            external_memory=False,
            reference_judge=False,
            capsule=True,
            observer=True,
            adaptive_refresh=False,
        ),
        model_path="model",
        tp_size=2,
    )


def runtime(tmp_path):
    value = QwenExoRuntime(
        config(tmp_path),
        SimpleNamespace(),
        HybridRuntimePolicy(
            tp_size=2,
            dtype="bfloat16",
            page_size=64,
            mamba_strategy="extra_buffer_lazy",
            mamba_state_dtype="bfloat16",
        ),
    )
    value.state = QwenExoRuntimeState.READY
    return value


def capture_retrieval_questions(value):
    captures = {"query_probe": [], "memory_pipeline": [], "role_plans": []}

    class QueryProbe:
        async def probe(self, request_id, plan):
            captures["role_plans"].append((request_id, plan))
            state = QueryStateSpan("original_task", 0, 1, 0, 1)
            return SimpleNamespace(
                query_heads=(((0.1, 0.2),),),
                query_states=(state,),
                role_plan_digest=plan.identity,
                status="ready",
                prompt_tokens=2,
            )

    class Pipeline:
        def __init__(self):
            self.states = {}

        async def prepare_responses_request(self, request, **kwargs):
            question = kwargs["retrieval_question"]
            captures["memory_pipeline"].append((request.request_id, question))
            captures["query_probe"].append((request.request_id, question))
            state = SimpleNamespace(
                public_dict=lambda: {
                    "question_digest": stable_digest(question),
                    "retrieval_question_digest": stable_digest(question),
                    "next_native_attractor": {"status": "ready"},
                },
                question_digest=stable_digest(question),
                retrieval_question_digest=stable_digest(question),
                policy_attachment=None,
                radix_prefix_identity=None,
                previous_response_id=kwargs.get("published_previous_response_id"),
                effective_memory_previous_response_id=kwargs.get(
                    "memory_previous_response_id"
                ),
                policy_attached_tokens=0,
                attached_tokens=0,
                policy_document_ids=(),
                radix_prefix_page_id=None,
                radix_prefix_token_ids=(),
                restoration_status="not_requested",
            )
            self.states[request.request_id] = state
            return request, state

        async def capture_native_attractor(self, request_id):
            return self.states.get(request_id)

        async def get_state(self, request_id):
            return self.states.get(request_id)

    value.query_probe = QueryProbe()
    value.memory_pipeline = Pipeline()
    return captures


def test_reflection_memory_runs_after_own_request_becomes_idle(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags, external_memory=True),
        reflection_memory_mode="active",
        max_internal_tokens=12288,
    )
    calls = []

    class ReflectionMemoryService:
        async def reflect(self, **kwargs):
            calls.append(kwargs)

    async def immediate_sleep(_seconds):
        return None

    value.reflection_memory_service = ReflectionMemoryService()
    value._request_conversation_keys["resp-1"] = "conversation-1"
    value._reflection_memory_last_activity["conversation-1"] = 1.0
    monkeypatch.setattr("qwen_exo_booster.runtime.asyncio.sleep", immediate_sleep)

    asyncio.run(
        value._run_reflection_memory_after_idle(
            conversation_key="conversation-1",
            activity_at=1.0,
            trajectory_id="resp-1",
            original_task="Inspect the WFP wrapper",
            tool_ledger=({"observation": "WFP layer verified"},),
            trajectory_history=(
                {
                    "kind": "tool_observation",
                    "content": "WFP layer verified",
                    "source_digest": "row-1",
                },
            ),
            capsule_history=(),
            source_token_count=512,
            source_digest="reflection-source",
        )
    )

    assert calls[0]["trajectory_id"] == "resp-1"
    assert calls[0]["trajectory_history"][0]["source_digest"] == "row-1"
    assert calls[0]["source_token_count"] == 512
    assert value._reflection_memory_sources["conversation-1"] == "reflection-source"


@pytest.mark.asyncio
async def test_pending_reflection_can_be_listed_and_started_immediately(tmp_path):
    value = runtime(tmp_path)
    calls = []

    class ReflectionMemoryService:
        async def reflect(self, **kwargs):
            calls.append(kwargs)

    value.reflection_memory_service = ReflectionMemoryService()
    activity_at = time.monotonic()
    now = time.time()
    work = PendingReflectionMemory(
        conversation_key="conversation-pending",
        trajectory_id="resp-pending",
        original_task="inspect pending reflection",
        tool_ledger=({"observation": "verified"},),
        trajectory_history=({"kind": "tool_observation", "content": "verified"},),
        capsule_history=(),
        source_token_count=512,
        source_digest="pending-source",
        activity_at=activity_at,
        last_activity_at=now,
        scheduled_at=now,
        due_at=now + 300,
    )
    value._reflection_memory_last_activity[work.conversation_key] = activity_at
    value._pending_reflection_memories[work.conversation_key] = work
    sleeper = asyncio.create_task(asyncio.sleep(3600))
    value._reflection_memory_tasks[work.conversation_key] = sleeper

    listed = value.pending_reflection_memories()
    accepted = value.start_pending_reflections([work.conversation_key])
    reflection_task = value._reflection_memory_tasks[work.conversation_key]
    await asyncio.gather(sleeper, return_exceptions=True)
    await reflection_task

    assert listed[0]["status"] == "waiting"
    assert 0 < listed[0]["timeout_remaining_seconds"] <= 300
    assert listed[0]["last_activity_at"] == now
    assert accepted["started"] == [work.conversation_key]
    assert calls[0]["trajectory_id"] == "resp-pending"
    assert value.pending_reflection_memories() == []


@pytest.mark.asyncio
async def test_pending_reflection_can_be_cancelled(tmp_path):
    value = runtime(tmp_path)
    now = time.time()
    work = PendingReflectionMemory(
        conversation_key="conversation-cancel",
        trajectory_id="resp-cancel",
        original_task="cancel this reflection",
        tool_ledger=({"observation": "verified"},),
        trajectory_history=(),
        capsule_history=(),
        source_token_count=512,
        source_digest="cancel-source",
        activity_at=time.monotonic(),
        last_activity_at=now,
        scheduled_at=now,
        due_at=now + 300,
    )
    value._pending_reflection_memories[work.conversation_key] = work
    sleeper = asyncio.create_task(asyncio.sleep(3600))
    value._reflection_memory_tasks[work.conversation_key] = sleeper

    result = value.cancel_pending_reflections([work.conversation_key])
    await asyncio.gather(sleeper, return_exceptions=True)

    assert result["cancelled"] == [work.conversation_key]
    assert value.pending_reflection_memories() == []
    assert work.conversation_key not in value._reflection_memory_tasks


def test_reflection_memory_input_rows_keep_user_reasoning_and_tool_history():
    rows = QwenExoRuntime._reflection_memory_input_rows(
        [
            {"type": "message", "role": "user", "content": "inspect wrapper"},
            {"type": "reasoning", "summary": "need direct source evidence"},
            {
                "type": "function_call",
                "name": "read_repository_file",
                "call_id": "call-1",
                "arguments": '{"path":"policy.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "WFP_LAYER_ALE_AUTH_CONNECT_V4",
            },
        ]
    )

    assert [row["kind"] for row in rows] == [
        "user_context",
        "assistant_trajectory",
        "tool_action",
        "tool_observation",
    ]
    assert rows[-1]["tool_name"] == "read_repository_file"
    assert rows[-1]["content"] == "WFP_LAYER_ALE_AUTH_CONNECT_V4"


def test_activation_editor_identity_isolated_in_radix_cache(tmp_path):
    value = runtime(tmp_path)
    root = value.config.state_directory / "activation-editors"
    root.mkdir(parents=True)
    artifact = root / "combined-trajectories.editor.pt"
    artifact.write_bytes(b"first")
    (root / "active.json").write_text(
        json.dumps({"editor": "combined-trajectories", "strength": 1.0}),
        encoding="utf-8",
    )

    first = value.activation_editor_request()
    artifact.write_bytes(b"second-artifact")
    second = value.activation_editor_request()
    (root / "active.json").unlink()
    disabled = value.activation_editor_request()

    assert first["spec"]["editor"] == "combined-trajectories"
    assert second["spec"] == first["spec"]
    assert second["cache_identity"] != first["cache_identity"]
    assert disabled["spec"] is None
    assert disabled["cache_identity"] not in {
        first["cache_identity"],
        second["cache_identity"],
    }


def valid_capsule():
    return {
        "summary": "Connection filtering task is in progress.",
        "phase": "implementation",
        "established": ["WFP layer selected"],
        "unresolved": ["Run verification"],
        "next_action": "Execute the focused test.",
        "event": "PROGRESS",
        "state_change": "YES",
        "verification": "NO",
        "repetition": "NO",
    }


def observer_event(state, *, event_id):
    return MidThinkEvent(
        event_id=event_id,
        request_id="resp-observer",
        token_index=10,
        trigger_reasons=("surprisal",),
        current_surprisal=7.0,
        window_mean=7.0,
        history_mean=1.0,
        ema_surprisal=2.0,
        recovery_window_mean=1.0,
        uncertainty_state=state,
        pre_q_sketches=(),
        post_q_sketches=(),
    )


def observer_result(*events):
    return ObserverResult(
        request_id="resp-observer",
        token_count=10,
        new_tokens=1,
        current_surprisal=7.0,
        local_window_mean=7.0,
        history_mean=1.0,
        ema_surprisal=2.0,
        max_surprisal=7.0,
        latest_q_drift=None,
        latest_memory_energy=None,
        triggered=bool(events),
        trigger_reasons=("surprisal",),
        events=tuple(events),
        finished=False,
    )


def test_resolved_observer_event_does_not_schedule_hidden_refresh(tmp_path):
    value = runtime(tmp_path)
    resolved = observer_event("resolved_by_continuation", event_id="resolved-event")
    value.refresh_service = SimpleNamespace()
    value.observer.observe_generation_result = lambda *args, **kwargs: (
        observer_result(resolved)
    )

    value.observe_generation_result("resp-observer", {"meta_info": {}})

    assert "resp-observer" not in value._refresh_tasks
    event_types = [
        event.event_type for event in value.telemetry.events("resp-observer")
    ]
    assert "adaptive.resolved_without_refresh" in event_types
    assert "adaptive.transition" not in event_types


def test_mixed_observer_events_selects_persistent_uncertainty(tmp_path):
    value = runtime(tmp_path)
    resolved = observer_event("resolved_by_continuation", event_id="resolved-event")
    persistent = observer_event("persistent_uncertainty", event_id="persistent-event")
    refresh_events = []

    class RefreshService:
        async def refresh(self, **kwargs):
            refresh_events.append(kwargs["event"].event_id)
            return SimpleNamespace(status="no_eligible_reference")

    value.refresh_service = RefreshService()
    value._request_questions["resp-observer"] = "question"
    value.observer.observe_generation_result = lambda *args, **kwargs: (
        observer_result(resolved, persistent)
    )

    async def exercise():
        value.observe_generation_result("resp-observer", {"meta_info": {}})
        await value._refresh_tasks["resp-observer"]

    asyncio.run(exercise())

    assert refresh_events == ["persistent-event"]


def test_previous_turn_capsule_is_privately_restored(tmp_path):
    value = runtime(tmp_path)
    original = CapsuleUpdateInput(
        parent_request_id="resp-1",
        turn_id="resp-1",
        trajectory_id="resp-1",
        event_sequence=0,
        original_task="Implement WFP filter",
        previous_capsule=None,
        assistant_reasoning="Implemented the first step.",
        assistant_tool_calls=(),
        tool_observation="",
        telemetry_correlation_id="trace-1",
    )
    value.capsule_store.commit(original, valid_capsule())
    value.capsules = SimpleNamespace()

    prepared, memory_state = asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-2",
                previous_response_id="resp-1",
                input="Continue the task",
                instructions="Answer precisely.",
            )
        )
    )

    assert memory_state is None
    assert prepared.instructions.startswith("Answer precisely.")
    assert "private execution capsule" in prepared.instructions
    assert "untrusted JSON data" in prepared.instructions
    assert "<execution_capsule_json>" in prepared.instructions
    assert "Run verification" in prepared.instructions
    assert prepared.extra_key
    assert "Run verification" not in str(value.telemetry_events("resp-2"))
    dropped = value.drop_restored_capsule_for_context(
        prepared,
        rendered_prompt_tokens=102500,
        context_length=102400,
        reserved_output_tokens=32,
        reason="context_capacity",
    )
    assert dropped.instructions == "Answer precisely."
    assert dropped.extra_key is None
    assert not value.has_restored_capsule("resp-2")


def test_immediate_child_waits_for_parent_capsule_finalization(tmp_path):
    value = runtime(tmp_path)

    async def exercise():
        update_started = asyncio.Event()
        release_update = asyncio.Event()

        class BlockingCapsuleService:
            async def update_many(self, updates):
                update_started.set()
                await release_update.wait()
                update = updates[0]
                record = value.capsule_store.commit(update, valid_capsule())
                return (
                    SimpleNamespace(
                        valid=True,
                        deduplicated=False,
                        tokens=12,
                        latency_seconds=0.01,
                        record=record,
                    ),
                )

        value.capsules = BlockingCapsuleService()
        await value.prepare_responses_request(
            FakeRequest(request_id="resp-parent", input="Implement WFP filter")
        )
        value.observe_generation_result(
            "resp-parent", {"text": "Implemented the first step.", "meta_info": {}}
        )
        parent_finalization = value.schedule_completion("resp-parent")
        assert parent_finalization is not None
        await update_started.wait()

        child_preparation = asyncio.create_task(
            value.prepare_responses_request(
                FakeRequest(
                    request_id="resp-child",
                    previous_response_id="resp-parent",
                    input="Continue the task",
                )
            )
        )
        await asyncio.sleep(0)
        assert not child_preparation.done()
        assert value.is_finalizing("resp-parent")

        release_update.set()
        prepared, memory_state = await child_preparation
        await parent_finalization

        assert memory_state is None
        assert "private execution capsule" in prepared.instructions
        assert "Run verification" in prepared.instructions
        assert value._parent_capsules["resp-child"].trajectory_id == "resp-parent"

    asyncio.run(exercise())


def test_cancelled_child_waiting_for_parent_does_not_rebuild_state(tmp_path):
    value = runtime(tmp_path)

    async def exercise():
        parent_release = asyncio.Event()

        async def finalize_parent():
            await parent_release.wait()

        parent_finalization = asyncio.create_task(finalize_parent())
        value._finalize_tasks["resp-parent"] = parent_finalization
        child_preparation = asyncio.create_task(
            value.prepare_responses_request(
                FakeRequest(
                    request_id="resp-cancelled-child",
                    previous_response_id="resp-parent",
                    input="Continue the task",
                    background=True,
                )
            )
        )
        await asyncio.sleep(0)
        assert value.is_pending_background_request("resp-cancelled-child")
        assert await value.cancel_pending_background_request("resp-cancelled-child")

        parent_release.set()
        await parent_finalization
        with pytest.raises(asyncio.CancelledError):
            await child_preparation

        assert not value.owns_request("resp-cancelled-child")
        assert "resp-cancelled-child" not in value._parent_response_ids
        assert "resp-cancelled-child" not in value._parent_capsules
        assert "resp-cancelled-child" not in value._original_tasks
        assert "resp-cancelled-child" not in value._capsule_restorations
        value._finalize_tasks.pop("resp-parent", None)
        value.acknowledge_request_cancellation("resp-cancelled-child")

    asyncio.run(exercise())


def test_duplicate_active_request_id_preserves_original_state_and_tasks(tmp_path):
    value = runtime(tmp_path)
    adaptive_begins = []
    value.adaptive_retrieval = SimpleNamespace(begin=adaptive_begins.append)

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(request_id="resp-duplicate", input="original question")
        )
        task_release = asyncio.Event()
        original_task = asyncio.create_task(task_release.wait())
        value._refresh_tasks["resp-duplicate"] = original_task

        with pytest.raises(
            ValueError,
            match="request_id 'resp-duplicate' is already active",
        ):
            await value.prepare_responses_request(
                FakeRequest(
                    request_id="resp-duplicate",
                    input="replacement question",
                    background=True,
                )
            )

        assert value._request_questions["resp-duplicate"] == "original question"
        assert value._refresh_tasks["resp-duplicate"] is original_task
        assert not original_task.done()
        assert adaptive_begins == ["resp-duplicate"]
        assert not value.is_pending_background_request("resp-duplicate")
        assert [
            event["event_type"] for event in value.telemetry_events("resp-duplicate")
        ] == ["request.started"]

        task_release.set()
        await original_task
        value.adaptive_retrieval = None
        await value.cancel_request("resp-duplicate")

    asyncio.run(exercise())


def test_concurrent_parent_capacity_fails_before_internal_preparation(tmp_path):
    value = runtime(tmp_path)
    value.config = replace(value.config, max_running_requests=1)

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(request_id="resp-first", input="first question")
        )

        with pytest.raises(QwenExoCapacityConflict, match="capacity is exhausted"):
            await value.prepare_responses_request(
                FakeRequest(request_id="resp-overload", input="second question")
            )

        assert "resp-overload" not in value._request_questions
        assert not value.is_pending_background_request("resp-overload")
        assert value.telemetry_events("resp-overload") == []
        await value.cancel_request("resp-first")

    asyncio.run(exercise())


class FakeCapsuleService:
    def __init__(self):
        self.updates = []

    async def update_many(self, updates):
        self.updates.extend(updates)
        return (
            SimpleNamespace(
                valid=True,
                deduplicated=False,
                tokens=12,
                latency_seconds=0.01,
                record=None,
            ),
        )


def test_completed_response_updates_capsule_once_and_releases_state(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeCapsuleService()
    value.capsules = capsules
    value._request_questions["resp-1"] = "Implement WFP filter"
    value._original_tasks["resp-1"] = "Implement WFP filter"
    attractor_calls = []

    class FakeMemoryPipeline:
        async def capture_native_attractor(self, request_id):
            attractor_calls.append(request_id)
            return SimpleNamespace(
                public_dict=lambda: {
                    "next_native_attractor": {
                        "status": "ready",
                        "lane": "knowledge",
                        "document_id": "threejs",
                        "page_id": 7,
                    }
                }
            )

        async def get_state(self, _request_id):
            return None

    value.memory_pipeline = FakeMemoryPipeline()

    async def exercise():
        async def generate():
            value.observe_generation_result(
                "resp-1",
                {
                    "text": "Implemented and verified the WFP filter.",
                    "meta_info": {
                        "output_token_logprobs": [(-1.0, 1)],
                        "finish_reason": {"type": "stop"},
                        "qwen_exo_q_sketch": [[0.1, 0.2, 0.3]],
                    },
                },
            )
            yield "chunk"

        chunks = []
        async for chunk in value.track_generation("resp-1", generate()):
            chunks.append(chunk)
            assert capsules.updates == []
        assert chunks == ["chunk"]
        assert value.owns_request("resp-1")
        assert value.is_finalizing("resp-1")
        await value.complete_request("resp-1")
        assert not value.owns_request("resp-1")
        assert value.observer.state("resp-1") is None

    asyncio.run(exercise())

    assert len(capsules.updates) == 1
    update = capsules.updates[0]
    assert update.trajectory_id == "resp-1"
    assert update.original_task == "Implement WFP filter"
    assert "verified the WFP filter" in update.assistant_reasoning
    assert attractor_calls == ["resp-1"]
    attractor_event = next(
        event
        for event in value.telemetry_events("resp-1")
        if event["event_type"] == "native_attractor.completed"
    )
    assert attractor_event["payload"]["status"] == "ready"
    stage_summary = next(
        event
        for event in value.telemetry_events("resp-1")
        if event["event_type"] == "request.stage_summary"
    )["payload"]
    assert stage_summary["schema"] == "qwen-exo-stage-summary-v1"
    assert stage_summary["stages"] == [
        "prefill_retrieval",
        "semantic_judge",
        "mid_think_observer",
        "self_ask_answer",
        "causal_replay",
        "maybe_gate",
        "next_turn_restoration",
        "post_tool_recall",
        "execution_capsule",
    ]
    assert stage_summary["capsule"]["valid"] is True


def test_capsule_output_preserves_incremental_repetition_and_tool_turns(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeCapsuleService()
    value.capsules = capsules
    value._request_questions["resp-tools"] = "Run a tool"
    value._original_tasks["resp-tools"] = "Run a tool"

    async def exercise():
        value.observe_generation_result(
            "resp-tools",
            {"text": "ha", "meta_info": {"output_token_logprobs": [(-1.0, 1)]}},
            incremental_logprobs=True,
            generation_index=0,
        )
        value.observe_generation_result(
            "resp-tools",
            {"text": "ha", "meta_info": {"output_token_logprobs": [(-1.0, 2)]}},
            incremental_logprobs=True,
            generation_index=0,
        )
        value.observe_generation_result(
            "resp-tools",
            {"text": "answer", "meta_info": {"output_token_logprobs": []}},
            generation_index=1,
        )
        value.observe_generation_result(
            "resp-tools",
            {"text": "answer done", "meta_info": {"output_token_logprobs": []}},
            generation_index=1,
        )
        value.record_tool_event(
            "resp-tools",
            "tool returned 42",
            tool_call={"recipient": "python", "arguments": "6 * 7"},
        )
        await value.complete_request("resp-tools")

    asyncio.run(exercise())

    assert capsules.updates[0].assistant_reasoning == "hahaanswer done"
    assert capsules.updates[0].assistant_tool_calls == (
        {"recipient": "python", "arguments": "6 * 7"},
    )
    assert capsules.updates[0].tool_observation == "tool returned 42"


def test_response_tool_events_preserve_function_name():
    events = QwenExoRuntime._response_tool_events(
        [
            {
                "type": "function_call",
                "call_id": "call-7",
                "name": "read_repository_file",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call-7",
                "output": "repository contents",
            },
        ]
    )

    assert events == (
        (
            {
                "type": "function_call_output",
                "call_id": "call-7",
                "name": "read_repository_file",
            },
            "repository contents",
        ),
    )


def test_function_call_output_queues_native_think_context_before_generation(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []

    class RefreshService:
        async def refresh(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="ready_for_safe_replay",
                question="Q?",
                answer="verified",
                selected_document_ids=("document-1",),
                decision_ids=("decision-1",),
                maybe_decision="admit_post_tool",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()
    prepared, _ = asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-tool-output",
                input=[
                    {"role": "user", "content": "Continue the task"},
                    {
                        "role": "assistant",
                        "content": "The current hypothesis may miss a wrapper callsite.",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call-7",
                        "output": "direct result",
                    },
                ],
                instructions="Keep tool calls structured.",
            )
        )
    )

    assert calls[0]["latest_tool_observation"] == "direct result"
    assert "direct result" in calls[0]["partial_output"]
    assert "current hypothesis may miss a wrapper" in calls[0]["partial_output"]
    assert "QWEN-EXO private post-tool reference" not in prepared.instructions
    injection = asyncio.run(value.await_think_context("resp-tool-output"))
    assert injection is not None
    assert injection.question == "Q?"
    assert injection.answer == "verified"
    assert injection.text == ("\n\nSelf-question: Q?\n" "Self-answer: verified\n")
    assert value._request_tool_calls["resp-tool-output"] == [
        {"type": "function_call_output", "call_id": "call-7"}
    ]


@pytest.mark.asyncio
async def test_reasoning_budget_discards_pending_self_ask_and_refresh(tmp_path):
    value = runtime(tmp_path)
    request_id = "resp-reasoning-budget"
    turn_id = f"{request_id}:post_tool:0"
    value._pending_think_contexts[request_id] = SimpleNamespace(turn_id=turn_id)

    async def blocked():
        await asyncio.Event().wait()

    refresh_task = asyncio.create_task(blocked())
    replay_task = asyncio.create_task(blocked())
    value._refresh_tasks[request_id] = refresh_task
    value._replay_tasks[request_id] = replay_task
    await asyncio.sleep(0)

    await value.discard_think_context_for_reasoning_budget(
        request_id,
        observed_tokens=3072,
        generation_index=2,
    )

    assert request_id not in value._pending_think_contexts
    assert turn_id in value._consumed_think_contexts
    assert refresh_task.cancelled()
    assert replay_task.cancelled()
    events = value.telemetry.events(request_id, limit=10)
    assert [event.event_type for event in events] == [
        "reasoning.budget_forced",
        "self_ask.think_context_skipped",
    ]
    assert events[0].payload == {
        "max_reasoning_tokens": 3072,
        "observed_tokens": 3072,
        "generation_index": 2,
        "self_ask_skipped": True,
        "had_pending_context": True,
        "cancelled_refresh_tasks": 2,
    }
    assert events[1].payload["reason"] == "reasoning_budget_forced"


def test_response_trajectory_context_keeps_recent_model_and_tool_evidence(tmp_path):
    value = runtime(tmp_path)
    context = value._response_trajectory_context(
        [
            {"role": "user", "content": "Original private task text"},
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Inspect wrappers next."}],
            },
            {
                "type": "function_call",
                "name": "shell",
                "arguments": '{"command":"focused check"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "focused check failed at the public boundary",
            },
        ],
        max_tokens=1024,
    )

    assert "Original private task text" not in context
    assert "Inspect wrappers next." in context
    assert "focused check" in context
    assert "focused check failed at the public boundary" in context


def test_function_call_output_input_is_captured_for_capsule(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeCapsuleService()
    value.capsules = capsules

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-function-output",
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "focused test passed",
                    },
                    {"role": "user", "content": "Continue"},
                ],
            )
        )
        value.observe_generation_result(
            "resp-function-output", {"text": "Done", "meta_info": {}}
        )
        await value.complete_request("resp-function-output")

    asyncio.run(exercise())

    update = capsules.updates[0]
    assert update.assistant_tool_calls == (
        {"type": "function_call_output", "call_id": "call-1"},
    )
    assert update.tool_observation == "focused test passed"


def test_full_history_recalls_only_unseen_tool_outputs_and_keeps_eligible_result(
    tmp_path,
):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []

    class RefreshService:
        async def refresh(self, **kwargs):
            observation = kwargs["latest_tool_observation"]
            calls.append(observation)
            admitted = observation != "rejected"
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status=(
                    "ready_for_safe_replay" if admitted else "no_eligible_reference"
                ),
                question=("Q?" if admitted else None),
                answer=(observation if admitted else None),
                selected_document_ids=(("document-1",) if admitted else ()),
                decision_ids=(("decision-1",) if admitted else ()),
                maybe_decision=("admit_post_tool" if admitted else "not_compiled"),
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()

    async def exercise():
        first_input = [
            {"role": "user", "content": "Implement the requested repository change"},
            {
                "role": "user",
                "content": "eligible",
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "eligible",
            },
            {
                "role": "user",
                "content": "rejected",
                "type": "function_call_output",
                "call_id": "call-2",
                "output": "rejected",
            },
        ]
        first, _ = await value.prepare_responses_request(
            FakeRequest(request_id="resp-history-1", input=first_input)
        )
        second, _ = await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-history-2",
                previous_response_id="resp-history-1",
                input=[
                    *first_input,
                    {
                        "role": "user",
                        "content": "fresh",
                        "type": "function_call_output",
                        "call_id": "call-3",
                        "output": "fresh",
                    },
                ],
            )
        )
        first_injection = await value.await_think_context("resp-history-1")
        second_injection = await value.await_think_context("resp-history-2")
        return first, second, first_injection, second_injection

    first, second, first_injection, second_injection = asyncio.run(exercise())

    assert calls == ["eligible", "rejected", "fresh"]
    assert value._request_questions["resp-history-2"] == (
        "Implement the requested repository change"
    )
    assert "Self-answer:" not in str(first.instructions)
    assert "Self-answer:" not in str(second.instructions)
    assert first_injection is not None
    assert first_injection.answer == "eligible"
    assert second_injection is not None
    assert second_injection.answer == "fresh"
    deduplicated = next(
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.history_deduplicated"
    )
    assert deduplicated == {
        "discovered_count": 3,
        "new_count": 1,
        "skipped_count": 2,
    }


def test_stateless_full_history_recalls_each_tool_output_once(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []

    class RefreshService:
        async def refresh(self, **kwargs):
            observation = kwargs["latest_tool_observation"]
            calls.append(observation)
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="no_eligible_reference",
                question=None,
                answer=None,
                selected_document_ids=(),
                decision_ids=(),
                maybe_decision="not_compiled",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()
    task = {"role": "user", "content": "Create the requested Three.js scenes"}
    first_output = {
        "type": "function_call_output",
        "call_id": "call-first",
        "output": "wrote pinball.html",
    }
    second_output = {
        "type": "function_call_output",
        "call_id": "call-second",
        "output": "wrote factory.html",
    }

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-stateless-1",
                input=[task, first_output],
            )
        )
        first_key = value._request_conversation_keys["resp-stateless-1"]
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-stateless-2",
                input=[task, first_output, second_output],
            )
        )
        return first_key, value._request_conversation_keys["resp-stateless-2"]

    first_key, second_key = asyncio.run(exercise())

    assert first_key == second_key
    assert calls == ["wrote pinball.html", "wrote factory.html"]
    deduplicated = next(
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.history_deduplicated"
    )
    assert deduplicated == {
        "discovered_count": 2,
        "new_count": 1,
        "skipped_count": 1,
    }


def test_query_probe_precedes_stateless_post_tool_recall(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    order = []
    pipeline_kwargs = {}

    class QueryProbe:
        async def probe(self, request_id, plan):
            del request_id
            order.append("query_probe")
            return SimpleNamespace(
                query_heads=(((0.1, 0.2),),),
                query_states=(QueryStateSpan("current_user", 0, 1, 0, 1),),
                role_plan_digest=plan.identity,
                status="ready",
                prompt_tokens=2,
            )

    class RefreshService:
        async def refresh(self, **kwargs):
            order.append("post_tool_recall")
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="no_eligible_reference",
                question=None,
                answer=None,
                selected_document_ids=(),
                decision_ids=(),
                maybe_decision="not_compiled",
                reflection_kind="none",
            )

    class Pipeline:
        async def prepare_responses_request(self, request, **kwargs):
            order.append("memory_pipeline")
            pipeline_kwargs.update(kwargs)
            state = SimpleNamespace(
                public_dict=lambda: {},
                policy_attachment=None,
                radix_prefix_identity=None,
                previous_response_id=None,
                policy_attached_tokens=0,
                attached_tokens=0,
                policy_document_ids=(),
                radix_prefix_page_id=None,
                radix_prefix_token_ids=(),
                restoration_status="not_requested",
            )
            return request, state

    value.query_probe = QueryProbe()
    value.refresh_service = RefreshService()
    value.memory_pipeline = Pipeline()

    asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-probe-first",
                input=[
                    {"role": "user", "content": "Create a Three.js scene"},
                    {
                        "type": "function_call_output",
                        "call_id": "call-write",
                        "output": "wrote scene.html",
                    },
                ],
            )
        )
    )

    assert order == ["query_probe", "post_tool_recall", "memory_pipeline"]
    assert pipeline_kwargs["query_heads"] == (((0.1, 0.2),),)
    assert pipeline_kwargs["query_probe_status"] == "ready"
    assert pipeline_kwargs["query_probe_prompt_tokens"] == 2


def test_parent_root_drives_tool_only_query_probe_without_changing_started_input(
    tmp_path,
):
    value = runtime(tmp_path)
    captures = capture_retrieval_questions(value)
    root = "Implement the WFP request-start lineage fix"
    observation = "Script written successfully"
    value._original_tasks["resp-parent"] = root

    _prepared, state = asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-child",
                previous_response_id="resp-parent",
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": "call-write",
                        "output": observation,
                    }
                ],
            )
        )
    )

    question = captures["query_probe"][0][1]
    assert captures["memory_pipeline"] == [("resp-child", question)]
    assert question.count("ORIGINAL TASK:\n") == 1
    assert question.count(root) == 1
    assert question.count("RECENT EXECUTION TRAJECTORY:\n") == 1
    assert question.count(observation) == 1
    assert value._request_questions["resp-child"] == root
    assert state.question_digest == stable_digest(question)
    assert state.retrieval_question_digest == stable_digest(question)
    started = next(
        event
        for event in value.telemetry.events("resp-child")
        if event.event_type == "request.started"
    )
    expected_started_input = (
        "RECENT EXECUTION TRAJECTORY:\n" f"TOOL OBSERVATION:\n{observation}"
    )
    assert started.payload["input"] == {
        "redacted": True,
        "sha256": stable_digest(expected_started_input),
        "bytes": len(expected_started_input.encode("utf-8")),
    }
    assert started.payload["input"]["sha256"] != stable_digest(root)
    assert started.payload["retrieval_query_digest"] == stable_digest(question)


def test_stateless_full_history_drives_root_current_and_trajectory_once(tmp_path):
    value = runtime(tmp_path)
    captures = capture_retrieval_questions(value)
    root = "Implement the stateless QK fix"
    current = "Run the focused verification"
    observation = "Source edit completed"

    asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-stateless-capture",
                input=[
                    {"role": "user", "content": root},
                    {"role": "assistant", "content": "Editing the runtime."},
                    {
                        "type": "function_call_output",
                        "call_id": "call-stateless",
                        "output": observation,
                    },
                    {"role": "user", "content": current},
                ],
            )
        )
    )

    question = captures["query_probe"][0][1]
    assert captures["memory_pipeline"] == [("resp-stateless-capture", question)]
    assert question.count(root) == 1
    assert question.count(current) == 1
    assert question.count(observation) == 1
    assert question.count("ORIGINAL TASK:\n") == 1
    assert question.count("CURRENT USER REQUEST:\n") == 1
    assert question.count("RECENT EXECUTION TRAJECTORY:\n") == 1
    assert value._request_questions["resp-stateless-capture"] == root


def test_stateless_call_association_requires_learned_isolated_root(tmp_path):
    value = runtime(tmp_path)
    captures = capture_retrieval_questions(value)
    root = "Audit the authoritative caller lineage"
    seed = FakeRequest(
        request_id="resp-seed",
        user="caller-a",
        session_id="session-a",
        instructions="shared system instructions",
        input=[
            {"role": "user", "content": root},
            {
                "type": "function_call_output",
                "call_id": "call-known",
                "output": "seed observation",
            },
        ],
    )
    known = FakeRequest(
        request_id="resp-known",
        user="caller-a",
        session_id="session-a",
        instructions="shared system instructions",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-known",
                "output": "known observation",
            }
        ],
    )
    unknown = replace(
        known,
        request_id="resp-unknown",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-unknown",
                "output": "unknown observation",
            }
        ],
    )
    isolated = replace(
        known,
        request_id="resp-isolated",
        user="caller-b",
        session_id="session-b",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-known",
                "output": "isolated observation",
            }
        ],
    )

    async def exercise():
        for request in (seed, known, unknown, isolated):
            await value.prepare_responses_request(request)

    asyncio.run(exercise())

    questions = dict(captures["query_probe"])
    assert questions["resp-known"].count(root) == 1
    assert "known observation" in questions["resp-known"]
    assert root not in questions["resp-unknown"]
    assert questions["resp-unknown"].startswith("RECENT EXECUTION TRAJECTORY:\n")
    assert root not in questions["resp-isolated"]
    assert questions["resp-isolated"].startswith("RECENT EXECUTION TRAJECTORY:\n")


def test_compaction_query_uses_root_and_summary_once_and_rejects_parent_mismatch(
    tmp_path,
):
    value = runtime(tmp_path)
    captures = capture_retrieval_questions(value)
    response_id = "resp_compact_parent"
    source_digest = "compaction-source"
    root = "Implement the compaction lineage fix"
    summary_text = "Runtime ordering was corrected; focused verification remains."
    current = "Run the compaction regression"
    observation = "Pipeline question captured"
    summary = CompactionSummary(
        summary=summary_text,
        input_tokens=10,
        output_tokens=8,
        reasoning_tokens=1,
        source_digest=source_digest,
    )
    encoded = summary.encrypted_content(response_id=response_id, memory={})
    value._compaction_summaries[response_id] = {
        "summary": summary_text,
        "source_digest": source_digest,
        "model_fingerprint": None,
    }
    value._original_tasks[response_id] = root
    input_items = [
        {"type": "compaction", "encrypted_content": encoded},
        {
            "type": "function_call_output",
            "call_id": "call-compact",
            "output": observation,
        },
        {"type": "message", "role": "user", "content": current},
    ]

    asyncio.run(
        value.prepare_responses_request(
            FakeRequest(request_id="resp-after-compact", input=input_items)
        )
    )

    question = captures["query_probe"][0][1]
    assert question.count(summary_text) == 1
    assert question.count(root) == 1
    assert question.count(current) == 1
    assert question.count(observation) == 1
    assert "<context_compaction>" not in question
    assert value._request_questions["resp-after-compact"] == root

    with pytest.raises(ValueError, match="does not match verified compaction lineage"):
        asyncio.run(
            value.prepare_responses_request(
                FakeRequest(
                    request_id="resp-mismatched-compact",
                    previous_response_id="resp-other-parent",
                    input=input_items,
                )
            )
        )


def test_semantically_repeated_self_question_is_not_reinjected(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    questions = {
        "first": (
            "Does predict actually load feature_schema.joblib or recompute the "
            "schema from input?"
        ),
        "second": (
            "Does predict actually deserialize feature_schema.joblib or recompute "
            "the schema from input?"
        ),
        "third": (
            "Does evaluate actually deserialize feature_schema.joblib or recompute "
            "the schema from input?"
        ),
    }
    answers = {
        "first": "verified predict",
        "second": "verified predict",
        "third": "verified evaluate",
    }

    class RefreshService:
        async def refresh(self, **kwargs):
            observation = kwargs["partial_output"].rsplit("TOOL OBSERVATION:\n", 1)[-1]
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="ready_for_safe_replay",
                question=questions[observation],
                answer=answers[observation],
                selected_document_ids=("document-1",),
                decision_ids=("decision-1",),
                maybe_decision="admit_post_tool",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()

    async def exercise():
        request_id = "resp-repeat"
        value._request_questions[request_id] = (
            "Implement the requested repository change"
        )
        value._request_conversation_keys[request_id] = "conversation-repeat"
        injections = []
        for index, observation in enumerate(("first", "second", "third"), 1):
            injections.append(
                await value.recall_after_tool(
                    request_id,
                    observation,
                    generation_index=-index,
                    trajectory_context=f"TOOL OBSERVATION:\n{observation}",
                )
            )
        return injections

    first, repeated, distinct = asyncio.run(exercise())

    assert first is not None
    assert repeated is None
    assert distinct is not None
    assert distinct.question.startswith("Does evaluate")
    skipped = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "self_ask.think_context_skipped"
    ]
    assert [payload["reason"] for payload in skipped] == ["repeated_self_question"]
    completed = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.completed"
    ]
    assert [payload["repeat_suppressed"] for payload in completed] == [
        False,
        True,
        False,
    ]


def test_unanswered_self_question_deduplicates_only_same_evidence(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"

    class RefreshService:
        async def refresh(self, **kwargs):
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="no_eligible_reference",
                question="Which generated wrapper currently fails?",
                answer=None,
                selected_document_ids=(),
                selected_reference_digests=(),
                context_source_digests=("same-evidence",),
                candidate_ids=(),
                decision_ids=(),
                maybe_decision="not_compiled",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()

    async def exercise():
        history = [{"role": "user", "content": "Repair the generated wrapper"}]
        for index in range(2):
            history = [
                *history,
                {
                    "type": "function_call_output",
                    "call_id": f"call-{index}",
                    "output": "same failing wrapper evidence",
                },
            ]
            await value.prepare_responses_request(
                FakeRequest(request_id=f"resp-no-answer-{index}", input=history)
            )

    asyncio.run(exercise())

    suppressed = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "self_ask.repeat_suppressed"
    ]
    assert len(suppressed) == 1
    assert suppressed[0]["reason"] == "repeated_unanswered_question_same_evidence"


def test_context_correction_is_restored_once_on_next_turn(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    record = SimpleNamespace(
        status="context_evidence_ready",
        context_source_digests=("context-digest",),
        question="Which wrapper is stale?",
        answer="The generated async wrapper is stale.",
        turn_id="resp-parent:post_tool:0",
        event_id=None,
        purpose="post_tool",
    )

    class RefreshService:
        @staticmethod
        def record(previous_response_id):
            return record if previous_response_id == "resp-parent" else None

    value.refresh_service = RefreshService()

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-child",
                previous_response_id="resp-parent",
                input=[{"role": "user", "content": "Continue the repair"}],
            )
        )
        first = await value.await_think_context("resp-child")
        second = await value.await_think_context("resp-child")
        return first, second

    first, second = asyncio.run(exercise())

    assert first is not None
    assert first.purpose == "next_turn_correction"
    assert first.answer == "The generated async wrapper is stale."
    assert second is None
    restored = [
        event.payload
        for event in value.telemetry.events("resp-child")
        if event.event_type == "self_ask.next_turn_context_restored"
    ]
    assert restored[-1]["previous_response_id"] == "resp-parent"


def test_request_completed_is_terminal_and_counts_all_generations(tmp_path):
    value = runtime(tmp_path)
    value._request_questions["resp-terminal"] = "Finish the task"
    value.observe_generation_result(
        "resp-terminal",
        {"text": "first", "output_ids": [1, 2], "meta_info": {}},
        incremental_logprobs=True,
        generation_index=0,
    )
    value.observe_generation_result(
        "resp-terminal",
        {"text": "second", "output_ids": [3, 4, 5], "meta_info": {}},
        incremental_logprobs=True,
        generation_index=1,
    )
    assert not [
        event
        for event in value.telemetry.events("resp-terminal")
        if event.event_type == "request.completed"
    ]

    value._emit_request_completed("resp-terminal")
    value._emit_request_completed("resp-terminal")

    completed = [
        event
        for event in value.telemetry.events("resp-terminal")
        if event.event_type == "request.completed"
    ]
    assert len(completed) == 1
    assert completed[0].payload["terminal"] is True
    assert completed[0].payload["output_tokens"] == 5
    assert completed[0].payload["generation_count"] == 2


def test_post_tool_refresh_budget_bounds_long_conversations(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []

    class RefreshService:
        async def refresh(self, **kwargs):
            observation = kwargs["latest_tool_observation"]
            calls.append(observation)
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="ready_for_safe_replay",
                question=f"What does observation {observation} establish?",
                answer=f"Observation {observation} is resolved.",
                selected_document_ids=("document-1",),
                decision_ids=("decision-1",),
                maybe_decision="admit_post_tool",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()

    async def exercise():
        injections = []
        for index in range(9):
            request_id = f"resp-budget-{index}"
            value._request_questions[request_id] = "Implement the repository change"
            value._request_conversation_keys[request_id] = "conversation-1"
            injections.append(
                await value.recall_after_tool(
                    request_id,
                    f"observation-{index}",
                    generation_index=index,
                )
            )
        return injections

    injections = asyncio.run(exercise())

    assert calls == [f"observation-{index}" for index in range(8)]
    assert all(injection is not None for injection in injections[:8])
    assert injections[8] is None
    skipped = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.skipped"
    ]
    assert skipped == [
        {
            "reason": "conversation_refresh_budget_exhausted",
            "conversation_key": "conversation-1",
            "refresh_count": 8,
            "refresh_limit": 8,
            "generation_index": 8,
        }
    ]


def _no_eligible_refresh_service(calls):
    class RefreshService:
        async def refresh(self, **kwargs):
            calls.append(kwargs["turn_id"])
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="no_eligible_reference",
                reference_status="no_eligible_reference",
                question=None,
                answer=None,
                selected_document_ids=(),
                decision_ids=(),
                maybe_decision="reject",
                reflection_kind="none",
            )

    return RefreshService()


def _prepare_post_tool_request(value, request_id, conversation_key="conversation-1"):
    value._request_questions[request_id] = "Implement the repository change"
    value._request_conversation_keys[request_id] = conversation_key


def test_post_tool_recall_cools_down_after_consecutive_no_eligible(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []
    value.refresh_service = _no_eligible_refresh_service(calls)

    async def exercise():
        for index in range(7):
            request_id = f"resp-cooldown-{index}"
            _prepare_post_tool_request(value, request_id)
            await value.recall_after_tool(
                request_id,
                f"observation-{index}",
                generation_index=index,
            )

    asyncio.run(exercise())

    assert calls == [
        "resp-cooldown-0:post_tool:0",
        "resp-cooldown-1:post_tool:1",
        "resp-cooldown-6:post_tool:6",
    ]
    skipped = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.skipped"
    ]
    assert skipped == [
        {
            "reason": "no_eligible_reference_cooldown",
            "conversation_key": "conversation-1",
            "refresh_count": 2,
            "refresh_limit": 8,
            "cooldown_remaining": remaining,
            "generation_index": index,
        }
        for remaining, index in zip((3, 2, 1, 0), range(2, 6))
    ]


def test_post_tool_recall_cooldown_resets_on_eligible_result(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    calls = []

    class RefreshService:
        eligible_next = False

        async def refresh(self, **kwargs):
            calls.append(kwargs["turn_id"])
            if RefreshService.eligible_next:
                RefreshService.eligible_next = False
                return SimpleNamespace(
                    turn_id=kwargs["turn_id"],
                    event_id=None,
                    purpose=kwargs["purpose"],
                    status="ready_for_safe_replay",
                    question="What changed?",
                    answer="The wrapper is stale.",
                    selected_document_ids=("document-1",),
                    decision_ids=("decision-1",),
                    maybe_decision="admit_post_tool",
                    reflection_kind="none",
                )
            return SimpleNamespace(
                turn_id=kwargs["turn_id"],
                event_id=None,
                purpose=kwargs["purpose"],
                status="no_eligible_reference",
                reference_status="no_eligible_reference",
                question=None,
                answer=None,
                selected_document_ids=(),
                decision_ids=(),
                maybe_decision="reject",
                reflection_kind="none",
            )

    value.refresh_service = RefreshService()

    async def exercise():
        injections = []
        for index in range(4):
            request_id = f"resp-reset-{index}"
            _prepare_post_tool_request(value, request_id)
            if index == 1:
                RefreshService.eligible_next = True
            injections.append(
                await value.recall_after_tool(
                    request_id,
                    f"observation-{index}",
                    generation_index=index,
                )
            )
        return injections

    injections = asyncio.run(exercise())

    assert len(calls) == 4
    assert injections[1] is not None
    assert not [
        event
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.skipped"
    ]
    # Without the reset on the eligible turn-1 result, the streak would be 3
    # after turn 3; the reset makes turns 2-3 a fresh streak of 2, which only
    # now reaches the arming threshold.
    assert value._post_tool_no_eligible_streaks["conversation-1"] == 2
    assert value._post_tool_no_eligible_cooldowns["conversation-1"] == 4


def test_post_tool_recall_hard_budget_wins_over_cooldown(tmp_path):
    value = runtime(tmp_path)
    value.observer.mode = "active"
    value._max_post_tool_refreshes_per_conversation = 3
    value._post_tool_no_eligible_cooldown_turns = 1
    calls = []
    value.refresh_service = _no_eligible_refresh_service(calls)

    async def exercise():
        for index in range(6):
            request_id = f"resp-hard-{index}"
            _prepare_post_tool_request(value, request_id)
            await value.recall_after_tool(
                request_id,
                f"observation-{index}",
                generation_index=index,
            )

    asyncio.run(exercise())

    assert calls == [
        "resp-hard-0:post_tool:0",
        "resp-hard-1:post_tool:1",
        "resp-hard-3:post_tool:3",
    ]
    skipped = [
        event.payload
        for event in value.telemetry.events()
        if event.event_type == "post_tool_recall.skipped"
    ]
    assert skipped == [
        {
            "reason": "no_eligible_reference_cooldown",
            "conversation_key": "conversation-1",
            "refresh_count": 2,
            "refresh_limit": 3,
            "cooldown_remaining": 0,
            "generation_index": 2,
        },
        *(
            {
                "reason": "conversation_refresh_budget_exhausted",
                "conversation_key": "conversation-1",
                "refresh_count": 3,
                "refresh_limit": 3,
                "generation_index": index,
            }
            for index in range(4, 6)
        ),
    ]


def test_response_conversation_keys_use_prompt_cache_then_crc_fallback(
    tmp_path, monkeypatch
):
    value = runtime(tmp_path)
    shared_input = [
        {"role": "system", "content": "system\r\nrules"},
        {"role": "user", "content": "implement the task"},
    ]
    prompt_a = FakeRequest(
        request_id="resp-prompt-a",
        prompt_cache_key="session-a",
        input=shared_input,
    )
    prompt_a_next = replace(prompt_a, request_id="resp-prompt-a-next")
    prompt_b = replace(
        prompt_a, request_id="resp-prompt-b", prompt_cache_key="session-b"
    )

    def resolve(request):
        identity = value._canonical_response_identity(request)
        return value._response_conversation_key(
            request_id=request.request_id,
            previous_response_id=None,
            request=request,
            canonical_identity=identity,
        )

    assert resolve(prompt_a) == resolve(prompt_a_next)
    assert resolve(prompt_a) != resolve(prompt_b)

    crc_a = replace(prompt_a, request_id="resp-crc-a", prompt_cache_key=None)
    crc_b = replace(crc_a, request_id="resp-crc-b")
    assert resolve(crc_a) == resolve(crc_b)
    assert resolve(crc_a).startswith("responses-crc32:")

    role_changed = replace(
        crc_a,
        request_id="resp-role-changed",
        input=[
            {"role": "developer", "content": "system\r\nrules"},
            {"role": "user", "content": "implement the task"},
        ],
    )
    assert resolve(crc_a) != resolve(role_changed)

    monkeypatch.setattr("qwen_exo_booster.runtime.zlib.crc32", lambda _payload: 7)
    collision_a = replace(crc_a, request_id="resp-collision-a")
    collision_b = replace(
        crc_a,
        request_id="resp-collision-b",
        input=[{"role": "user", "content": "a different task"}],
    )
    key_a = resolve(collision_a)
    key_b = resolve(collision_b)
    assert key_a.startswith("responses-crc32:00000007:")
    assert key_b.startswith("responses-crc32:00000007:")
    assert key_a != key_b


def test_learned_call_alias_points_to_prompt_cache_conversation(tmp_path):
    value = runtime(tmp_path)
    seed = FakeRequest(
        request_id="resp-seed",
        prompt_cache_key="session-call",
        input=[{"role": "user", "content": "root task"}],
    )
    seed_identity = value._canonical_response_identity(seed)
    seed_key = value._response_conversation_key(
        request_id=seed.request_id,
        previous_response_id=None,
        request=seed,
        canonical_identity=seed_identity,
        call_ids=("call-known",),
    )
    fallback = replace(
        seed,
        request_id="resp-fallback",
        prompt_cache_key=None,
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-known",
                "output": "done",
            }
        ],
    )
    unknown = replace(
        fallback,
        request_id="resp-unknown",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call-unknown",
                "output": "done",
            }
        ],
    )
    assert (
        value._response_conversation_key(
            request_id=fallback.request_id,
            previous_response_id=None,
            request=fallback,
            call_ids=("call-known",),
        )
        == seed_key
    )
    assert (
        value._response_conversation_key(
            request_id=unknown.request_id,
            previous_response_id=None,
            request=unknown,
            call_ids=("call-unknown",),
        )
        != seed_key
    )


@pytest.mark.asyncio
async def test_prompt_cache_restores_only_finalized_internal_memory_parent(tmp_path):
    value = runtime(tmp_path)
    capture_retrieval_questions(value)

    async def skip_stage_summary(_request_id):
        return None

    value._emit_stage_summary = skip_stage_summary
    first = FakeRequest(
        request_id="resp-memory-first",
        prompt_cache_key="memory-session",
        instructions="system rules",
        input=[{"role": "user", "content": "implement the task"}],
    )
    _prepared, first_state = await value.prepare_responses_request(first)
    assert first_state.previous_response_id is None
    assert first_state.effective_memory_previous_response_id is None
    await value.complete_request(first.request_id)

    second = replace(first, request_id="resp-memory-second")
    _prepared, second_state = await value.prepare_responses_request(second)
    assert second_state.previous_response_id is None
    assert second_state.effective_memory_previous_response_id == first.request_id
    assert second.request_id not in value._parent_response_ids
    assert second.request_id not in value._parent_capsules
    started = next(
        event
        for event in value.telemetry.events(second.request_id)
        if event.event_type == "request.started"
    )
    assert started.payload["parent_response_id"] is None


@pytest.mark.asyncio
async def test_prompt_cache_keeps_reflection_rows_and_pending_work_cumulative(tmp_path):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags, external_memory=True),
        reflection_memory_mode="active",
        reflection_memory_min_events=2,
        reflection_memory_min_tokens=0,
        max_internal_tokens=12288,
    )
    value.reflection_memory_service = SimpleNamespace(reflect=lambda **_kwargs: None)
    root = {"role": "user", "content": "inspect the runtime"}
    first_event = {
        "type": "function_call_output",
        "call_id": "call-one",
        "output": "first observation",
    }
    second_event = {
        "type": "function_call_output",
        "call_id": "call-two",
        "output": "second observation",
    }
    first = FakeRequest(
        request_id="resp-reflection-one",
        prompt_cache_key="reflection-session",
        input=[root, first_event],
    )
    second = replace(
        first,
        request_id="resp-reflection-two",
        input=[root, first_event, second_event],
    )

    await value.prepare_responses_request(first)
    await value.prepare_responses_request(second)
    first_key = value._request_conversation_keys[first.request_id]
    second_key = value._request_conversation_keys[second.request_id]
    assert first_key == second_key
    assert len(value._context_integrity_ledgers[first_key]) == 2
    rows = value._reflection_memory_trajectories[first_key]
    assert {row["call_id"] for row in rows if row["call_id"]} == {
        "call-one",
        "call-two",
    }

    value._request_outputs[second.request_id] = "verification complete"
    value._schedule_reflection_memory(second.request_id)
    assert list(value._pending_reflection_memories) == [first_key]
    pending = value._pending_reflection_memories[first_key]
    assert len(pending.tool_ledger) == 2
    assert pending.original_task == "inspect the runtime"
    task = value._reflection_memory_tasks[first_key]
    value._cancel_reflection_memory_task(first_key)
    await asyncio.gather(task, return_exceptions=True)


def test_replayed_full_history_skips_redundant_capsule_update(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeCapsuleService()
    value.capsules = capsules
    first_input = [
        {"role": "user", "content": "Implement the requested repository change"},
        {
            "role": "user",
            "content": "first observation",
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "first observation",
        },
    ]

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(request_id="resp-capsule-history-1", input=first_input)
        )
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-capsule-history-2",
                input=[
                    *first_input,
                    {
                        "role": "user",
                        "content": "second observation",
                        "type": "function_call_output",
                        "call_id": "call-2",
                        "output": "second observation",
                    },
                ],
            )
        )
        value.observe_generation_result(
            "resp-capsule-history-2", {"text": "Done", "meta_info": {}}
        )
        await value.complete_request("resp-capsule-history-2")

    asyncio.run(exercise())

    assert capsules.updates == []
    assert any(
        event.event_type == "capsule.skipped"
        and event.payload == {"reason": "stateless_full_history"}
        for event in value.telemetry.events()
    )


def test_trajectory_lineage_and_original_task_survive_runtime_restart(tmp_path):
    first = runtime(tmp_path)
    parent_update = CapsuleUpdateInput(
        parent_request_id="resp-root",
        turn_id="resp-root",
        trajectory_id="resp-root",
        event_sequence=0,
        original_task="Persistent root task",
        previous_capsule=None,
        assistant_reasoning="root result",
        assistant_tool_calls=(),
        tool_observation="",
        telemetry_correlation_id="root",
    )
    first.capsule_store.commit(parent_update, valid_capsule())

    restarted = runtime(tmp_path)
    capsules = FakeCapsuleService()
    restarted.capsules = capsules

    async def exercise():
        await restarted.prepare_responses_request(
            FakeRequest(
                request_id="resp-child",
                previous_response_id="resp-root",
                input="Continue after restart",
            )
        )
        restarted.observe_generation_result(
            "resp-child", {"text": "child result", "meta_info": {}}
        )
        await restarted.complete_request("resp-child")

    asyncio.run(exercise())

    child_update = capsules.updates[0]
    assert child_update.original_task == "Persistent root task"
    assert child_update.parent_trajectory_id == "resp-root"
    assert child_update.event_sequence == 1
    restarted.capsule_store.commit(child_update, valid_capsule())
    lineage = restarted.trajectory_lineage("resp-child")
    assert [record["trajectory_id"] for record in lineage] == [
        "resp-root",
        "resp-child",
    ]
    assert all(record["original_task"] == "Persistent root task" for record in lineage)


def test_concurrent_capsule_branches_restore_only_their_own_child(tmp_path):
    value = runtime(tmp_path)
    value.capsules = FakeCapsuleService()
    parent_update = CapsuleUpdateInput(
        parent_request_id="resp-parent",
        turn_id="resp-parent",
        trajectory_id="resp-parent",
        event_sequence=0,
        original_task="Original task",
        previous_capsule=None,
        assistant_reasoning="parent",
        assistant_tool_calls=(),
        tool_observation="",
        telemetry_correlation_id="parent-trace",
    )
    value.capsule_store.commit(parent_update, valid_capsule())
    value._original_tasks["resp-parent"] = "Original task"

    async def exercise():
        for request_id in ("resp-a", "resp-b"):
            await value.prepare_responses_request(
                FakeRequest(
                    request_id=request_id,
                    previous_response_id="resp-parent",
                    input=f"Continue branch {request_id}",
                )
            )
            value.observe_generation_result(
                request_id,
                {"text": f"output-{request_id}", "meta_info": {}},
            )
        await asyncio.gather(
            value.complete_request("resp-a"), value.complete_request("resp-b")
        )

    asyncio.run(exercise())

    updates = {update.trajectory_id: update for update in value.capsules.updates}
    assert set(updates) == {"resp-a", "resp-b"}
    assert updates["resp-a"].event_sequence == 1
    assert updates["resp-b"].event_sequence == 1
    assert updates["resp-a"].previous_capsule == valid_capsule()
    assert updates["resp-b"].previous_capsule == valid_capsule()

    branch_a_capsule = {**valid_capsule(), "next_action": "Only branch A"}
    branch_b_capsule = {**valid_capsule(), "next_action": "Only branch B"}
    value.capsule_store.commit(updates["resp-a"], branch_a_capsule)
    value.capsule_store.commit(updates["resp-b"], branch_b_capsule)
    prepared, _ = asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-a-next",
                previous_response_id="resp-a",
                input="Continue A",
            )
        )
    )

    assert "Only branch A" in prepared.instructions
    assert "Only branch B" not in prepared.instructions


def test_pending_background_preparation_can_be_cancelled(tmp_path):
    value = runtime(tmp_path)

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        class BlockingPipeline:
            async def prepare_responses_request(
                self, request, restoration=None, **_kwargs
            ):
                started.set()
                await release.wait()
                return request, SimpleNamespace(public_dict=lambda: {})

        value.memory_pipeline = BlockingPipeline()
        task = asyncio.create_task(
            value.prepare_responses_request(
                FakeRequest(
                    request_id="resp-pending",
                    input="question",
                    background=True,
                )
            )
        )
        await started.wait()
        assert value.is_pending_background_request("resp-pending")
        assert await value.cancel_pending_background_request("resp-pending")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not await value.claim_pending_background_request("resp-pending")
        value.acknowledge_request_cancellation("resp-pending")
        assert not value.is_pending_background_request("resp-pending")

    asyncio.run(exercise())


def test_background_dispatch_claim_prevents_pending_cancel(tmp_path):
    value = runtime(tmp_path)

    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(
                request_id="resp-claimed",
                input="question",
                background=True,
            )
        )
        assert await value.claim_pending_background_request("resp-claimed")
        assert not await value.cancel_pending_background_request("resp-claimed")
        await value.cancel_request("resp-claimed")

    asyncio.run(exercise())


def test_closed_generation_cancels_owned_request(tmp_path):
    value = runtime(tmp_path)
    value._request_questions["resp-cancel"] = "Cancel this request"

    async def exercise():
        async def generate():
            yield "first"
            await asyncio.sleep(60)

        tracked = value.track_generation("resp-cancel", generate())
        assert await tracked.__anext__() == "first"
        await tracked.aclose()
        await value.cancel_request("resp-cancel")

    asyncio.run(exercise())

    assert not value.owns_request("resp-cancel")
    events = value.telemetry_events("resp-cancel")
    assert [event["event_type"] for event in events] == ["request.cancelled"]


class FakeVariableCapsuleService:
    def __init__(self, validities):
        self.updates = []
        self._validities = list(validities)

    async def update_many(self, updates):
        self.updates.extend(updates)
        valid = self._validities.pop(0) if self._validities else True
        return (
            SimpleNamespace(
                valid=valid,
                deduplicated=False,
                tokens=12,
                latency_seconds=0.01,
                record=None,
            ),
        )


def run_chained_capsule_turns(value, turn_count):
    async def exercise():
        previous = None
        for index in range(turn_count):
            request_id = f"resp-cooldown-{index}"
            await value.prepare_responses_request(
                FakeRequest(
                    request_id=request_id,
                    previous_response_id=previous,
                    input="Continue the task",
                )
            )
            value.observe_generation_result(
                request_id, {"text": "Done", "meta_info": {}}
            )
            await value.complete_request(request_id)
            previous = request_id

    asyncio.run(exercise())


def invalid_cooldown_skips(value):
    return [
        event
        for event in value.telemetry.events()
        if event.event_type == "capsule.skipped"
        and event.payload.get("reason") == "invalid_cooldown"
    ]


def test_invalid_capsule_updates_trigger_cooldown_skip(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeVariableCapsuleService([False, False])
    value.capsules = capsules

    run_chained_capsule_turns(value, 4)

    assert len(capsules.updates) == 2
    skipped = invalid_cooldown_skips(value)
    assert [event.payload["cooldown_remaining"] for event in skipped] == [3, 2]
    assert all(event.payload["previous_valid"] is False for event in skipped)
    assert [event.payload["trajectory_id"] for event in skipped] == [
        "resp-cooldown-2",
        "resp-cooldown-3",
    ]
    assert len({event.payload["conversation_key"] for event in skipped}) == 1


def test_valid_capsule_update_resets_invalid_streak(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeVariableCapsuleService([False, True, False, False])
    value.capsules = capsules

    run_chained_capsule_turns(value, 5)

    assert len(capsules.updates) == 4
    skipped = invalid_cooldown_skips(value)
    assert [event.payload["trajectory_id"] for event in skipped] == ["resp-cooldown-4"]
    assert skipped[0].payload["cooldown_remaining"] == 3


def test_capsule_invalid_cooldown_expires_and_retries(tmp_path):
    value = runtime(tmp_path)
    capsules = FakeVariableCapsuleService([False, False, False])
    value.capsules = capsules

    run_chained_capsule_turns(value, 7)

    assert len(capsules.updates) == 3
    assert capsules.updates[-1].turn_id == "resp-cooldown-6"
    skipped = invalid_cooldown_skips(value)
    assert [event.payload["cooldown_remaining"] for event in skipped] == [3, 2, 1, 0]
