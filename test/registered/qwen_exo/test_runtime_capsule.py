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
from qwen_exo_booster.reflection_memory import (
    ReflectionMemory,
    ReflectionSourceSnapshot,
)
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
        observer_mode="shadow",
        feature_flags=QwenExoFeatureFlags(
            hybrid_prefix=True,
            capsule=True,
            observer=True,
            policy_data=False,
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


def test_reflection_memory_runs_after_own_request_becomes_idle(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags),
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


def test_reflection_waits_again_after_plain_user_turn(tmp_path, monkeypatch):
    """A user turn without new tool output still counts as activity: the idle
    waiter re-arms for the newer timestamp instead of reflecting mid-chat or
    dropping the pending reflection."""
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags),
        reflection_memory_mode="active",
        max_internal_tokens=12288,
    )
    calls = []
    sleeps = []

    class ReflectionMemoryService:
        async def reflect(self, **kwargs):
            calls.append(kwargs)

    async def recording_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            value._reflection_memory_last_activity["conversation-1"] = 2.0

    value.reflection_memory_service = ReflectionMemoryService()
    value._reflection_memory_last_activity["conversation-1"] = 1.0
    value._foreground_last_activity = -3600.0
    monkeypatch.setattr("qwen_exo_booster.runtime.asyncio.sleep", recording_sleep)

    asyncio.run(
        value._run_reflection_memory_after_idle(
            conversation_key="conversation-1",
            activity_at=1.0,
            trajectory_id="resp-1",
            original_task="Inspect the WFP wrapper",
            tool_ledger=(),
            trajectory_history=({"kind": "tool_observation", "content": "ok"},),
            capsule_history=(),
            source_token_count=512,
            source_digest="reflection-source",
            due_at=0.0,
        )
    )

    assert len(sleeps) == 2
    assert len(calls) == 1
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


@pytest.mark.asyncio
async def test_reflection_regeneration_is_recoverable_and_rejects_duplicates(tmp_path):
    value = runtime(tmp_path)
    document = value.knowledge.upsert(
        "reflection-memory/associated.md",
        "---\nsource_kind: trajectory_reflection\n"
        "document_group: reflection_memory\nreflection_memory_schema: 3\n"
        "title: 关联反思\n---\n\n旧的反思规则。",
    )
    reflection = ReflectionMemory(
        trajectory_id="resp-associated",
        conversation_key="conversation-associated",
        source_digest="source-associated",
        title="关联反思",
        outcome="mixed",
        reflection="旧反思",
        evidence="旧证据",
        causal_analysis="旧因果",
        reusable_experience="旧经验",
        avoid="旧禁忌",
        next_time="旧计划",
        memory_action="insert",
        target_document_path=None,
        target_document_sha256=None,
        source_event_count=1,
        source_token_count=512,
        attempts=1,
        created_at=1.0,
        document_path=document.relative_path,
        document_sha256=document.sha256,
        native_source_digest="bank-before",
        hot_updated=True,
        publication_status="published",
    )
    value.reflection_memory_store.append(reflection)
    value.reflection_source_store.save(
        ReflectionSourceSnapshot(
            source_digest=reflection.source_digest,
            trajectory_id=reflection.trajectory_id,
            conversation_key=reflection.conversation_key,
            original_task="Repair the GraphQL stream",
            trajectory_history=(
                {"kind": "tool_observation", "content": "17 checks passed"},
            ),
            capsule_history=(),
            verifier_feedback="",
            source_event_count=1,
            source_token_count=512,
            source_audit={"source_tokens": 128},
            captured_at=1.0,
        )
    )
    started = asyncio.Event()
    release = asyncio.Event()
    captures = []

    class RegenerationService:
        async def reflect(self, **kwargs):
            captures.append(kwargs)
            kwargs["stage_callback"]("similar_retrieval")
            kwargs["stage_callback"]("model_review")
            started.set()
            await release.wait()
            kwargs["stage_callback"]("publishing")
            return replace(
                reflection,
                source_digest="source-regenerated",
                memory_action="update",
                target_document_path=document.relative_path,
                target_document_sha256=document.sha256,
                merge_document_paths=(document.relative_path,),
                merge_document_sha256s=((document.relative_path, document.sha256),),
                document_sha256="b" * 64,
                native_source_digest="bank-after",
            )

    value.reflection_memory_service = RegenerationService()
    feedback = "Hidden verifier: four F2P checks failed after timeout."

    accepted = value.start_reflection_memory_regeneration(
        reflection.source_digest,
        verifier_feedback=feedback,
        expected_document_sha256=document.sha256,
    )
    await started.wait()
    running = value.reflection_memory_regeneration_status()
    with pytest.raises(RuntimeError, match="already running"):
        value.start_reflection_memory_regeneration(
            reflection.source_digest,
            verifier_feedback=feedback,
            expected_document_sha256=document.sha256,
        )
    release.set()
    await value._reflection_memory_regeneration_task
    completed = value.reflection_memory_regeneration_status()

    assert accepted["status"] == "queued"
    assert running["status"] == "running"
    assert running["stage"] == "model_review"
    assert completed["status"] == "succeeded"
    assert completed["result"]["source_digest"] == "source-regenerated"
    assert captures[0]["verifier_feedback"] == feedback
    assert captures[0]["required_update_target"].document_path == document.relative_path
    listed = value.reflection_memories()["reflections"][0]
    assert listed["source_available"] is True
    assert listed["trajectory_source"]["trajectory_id"] == reflection.trajectory_id


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


def test_activation_editor_request_is_disabled_without_experimental_flag(tmp_path):
    value = runtime(tmp_path)
    root = value.config.state_directory / "activation-editors"
    root.mkdir(parents=True)
    (root / "combined-trajectories.editor.pt").write_bytes(b"artifact")
    (root / "active.json").write_text(
        json.dumps({"editor": "combined-trajectories", "strength": 1.0}),
        encoding="utf-8",
    )

    request = value.activation_editor_request()

    assert request["spec"] is None
    assert request["cache_identity"] == stable_digest(
        "activation-editor-v1", "experimental-disabled"
    )


def test_activation_editor_identity_isolated_in_radix_cache(tmp_path):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags, activation_training=True),
    )
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

    prepared = asyncio.run(
        value.prepare_responses_request(
            FakeRequest(
                request_id="resp-2",
                previous_response_id="resp-1",
                input="Continue the task",
                instructions="Answer precisely.",
            )
        )
    )

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
        prepared = await child_preparation
        await parent_finalization

        assert "private execution capsule" in prepared.instructions
        assert "Run verification" in prepared.instructions
        assert value._parent_capsules["resp-child"].trajectory_id == "resp-parent"

    asyncio.run(exercise())


def test_cancelled_child_waiting_for_parent_does_not_rebuild_state(tmp_path):
    value = runtime(tmp_path)
    value.capsules = SimpleNamespace()

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
    async def exercise():
        await value.prepare_responses_request(
            FakeRequest(request_id="resp-duplicate", input="original question")
        )
        task_release = asyncio.Event()
        original_task = asyncio.create_task(task_release.wait())
        value._capsule_tasks["resp-duplicate"] = original_task

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
        assert value._capsule_tasks["resp-duplicate"] is original_task
        assert not original_task.done()
        assert not value.is_pending_background_request("resp-duplicate")
        assert [
            event["event_type"] for event in value.telemetry_events("resp-duplicate")
        ] == ["request.started"]

        task_release.set()
        await original_task
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
    stage_summary = next(
        event
        for event in value.telemetry_events("resp-1")
        if event["event_type"] == "request.stage_summary"
    )["payload"]
    assert stage_summary["schema"] == "qwen-exo-stage-summary-v2"
    assert stage_summary["stages"] == ["memory_tools", "execution_capsule"]
    assert stage_summary["memory_tools"] == {"calls": 0, "failed": 0}
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


def test_forced_reasoning_boundary_is_traced_and_recorded_as_output(tmp_path):
    """The trace page reads ``reasoning.budget_forced``; capsule, reflection and
    score bias read the generation's output ids, which must include the
    synthetic stop line and </think> the server appended."""
    value = runtime(tmp_path)
    request_id = "resp-reasoning-budget"

    value.record_forced_reasoning_boundary(
        request_id,
        observed_tokens=4090,
        max_reasoning_tokens=4090,
        committed_text="\nstop\n</think>",
        token_ids=(11, 12, 999),
        generation_index=2,
    )

    events = value.telemetry.events(request_id, limit=10)
    forced = [event for event in events if event.event_type == "reasoning.budget_forced"]
    assert forced[0].payload == {
        "max_reasoning_tokens": 4090,
        "observed_tokens": 4090,
        "generation_index": 2,
    }
    assert value._request_generation_output_ids[(request_id, 2)] == (11, 12, 999)


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


def test_response_conversation_keys_use_prompt_cache_then_discriminated_fallback(tmp_path):
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
    crc_b = replace(
        crc_a,
        request_id="resp-crc-b",
        input=[
            *shared_input,
            {"role": "assistant", "content": "Inspected the parser."},
            {"role": "user", "content": "Continue with the repair."},
        ],
    )
    assert resolve(crc_a) == resolve(crc_b)

    role_changed = replace(
        crc_a,
        request_id="resp-role-changed",
        input=[
            {"role": "developer", "content": "system\r\nrules"},
            {"role": "user", "content": "implement the task"},
        ],
    )
    assert resolve(crc_a) != resolve(role_changed)

    same_label_a = replace(
        crc_a,
        request_id="resp-common-first-line-a",
        input=[{"role": "user", "content": "Inspect the task\nRepair parser A"}],
    )
    same_label_b = replace(
        same_label_a,
        request_id="resp-common-first-line-b",
        input=[{"role": "user", "content": "Inspect the task\nRepair parser B"}],
    )
    identity_a = value._canonical_response_identity(same_label_a)
    identity_b = value._canonical_response_identity(same_label_b)
    assert identity_a.first_lines_digest == identity_b.first_lines_digest
    assert identity_a.payload_digest != identity_b.payload_digest
    key_a = resolve(same_label_a)
    key_b = resolve(same_label_b)
    assert key_a != key_b


def test_first_lines_and_full_head_normalize_crlf_only(tmp_path):
    value = runtime(tmp_path)
    crlf = FakeRequest(
        request_id="resp-crlf",
        instructions="Rules\r\nPreserve boundaries",
        input=[{"role": "user", "content": "Task\r\nRepair the parser"}],
    )
    lf = replace(
        crlf,
        request_id="resp-lf",
        instructions="Rules\nPreserve boundaries",
        input=[{"role": "user", "content": "Task\nRepair the parser"}],
    )
    crlf_identity = value._canonical_response_identity(crlf)
    lf_identity = value._canonical_response_identity(lf)
    assert crlf_identity.first_lines_digest == lf_identity.first_lines_digest
    assert crlf_identity.payload_digest == lf_identity.payload_digest
    assert crlf_identity.conversation_key == lf_identity.conversation_key
    changed_rules = value._canonical_response_identity(
        replace(lf, instructions="Rules\nDo not repair the parser")
    )
    assert changed_rules.first_lines_digest == lf_identity.first_lines_digest
    assert changed_rules.conversation_key != lf_identity.conversation_key


@pytest.mark.asyncio
async def test_response_cache_scope_keeps_explicit_continuity_and_model_isolation(tmp_path):
    value = runtime(tmp_path)
    value.model_identity = SimpleNamespace(fingerprint="model-a")
    first = FakeRequest(
        request_id="resp-cache-first",
        prompt_cache_key="client-conversation-a",
        input=[{"role": "user", "content": "Repair the parser"}],
    )
    await value.prepare_responses_request(first)
    first_namespace = value.response_cache_namespace(first.request_id)
    continued = replace(
        first,
        request_id="resp-cache-continued",
        input=[
            *first.input,
            {"role": "assistant", "content": "I found the boundary error."},
            {"role": "user", "content": "Apply the fix."},
        ],
    )
    await value.prepare_responses_request(continued)
    assert value.response_cache_namespace(continued.request_id) == first_namespace
    isolated = replace(
        continued,
        request_id="resp-cache-isolated",
        prompt_cache_key="client-conversation-b",
    )
    await value.prepare_responses_request(isolated)
    assert value.response_cache_namespace(isolated.request_id) != first_namespace
    lineage = replace(
        continued,
        request_id="resp-cache-lineage",
        previous_response_id=first.request_id,
        prompt_cache_key="different-client-key",
    )
    await value.prepare_responses_request(lineage)
    assert value.response_cache_namespace(lineage.request_id) == first_namespace
    value.model_identity = SimpleNamespace(fingerprint="model-b")
    assert value.response_cache_namespace(first.request_id) != first_namespace
    value.model_identity = None
    assert value.response_cache_namespace(first.request_id) != value.response_cache_namespace(
        continued.request_id
    )


@pytest.mark.asyncio
async def test_parallel_same_conversation_keeps_request_local_mutable_state(tmp_path):
    value = runtime(tmp_path)
    first = FakeRequest(
        request_id="resp-parallel-a",
        prompt_cache_key="shared-conversation",
        input=[{"role": "user", "content": "Repair the parser"}],
    )
    second = replace(first, request_id="resp-parallel-b")
    await asyncio.gather(
        value.prepare_responses_request(first),
        value.prepare_responses_request(second),
    )
    assert value._request_conversation_keys[first.request_id] == value._request_conversation_keys[
        second.request_id
    ]
    assert value._active_foreground_requests() == 2
    value.record_tool_event(first.request_id, "branch A found an overflow")
    value.record_tool_event(second.request_id, "branch B found a null pointer")
    value.record_tool_event(first.request_id, "branch A repaired its overflow")
    assert value._request_tool_observations[first.request_id] == [
        "branch A found an overflow",
        "branch A repaired its overflow",
    ]
    assert value._request_tool_observations[second.request_id] == [
        "branch B found a null pointer"
    ]
    assert first.request_id not in value._parent_response_ids
    assert second.request_id not in value._parent_response_ids


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
    conflicting = replace(
        seed, request_id="resp-conflicting-seed", prompt_cache_key="another-session"
    )
    conflicting_key = value._response_conversation_key(
        request_id=conflicting.request_id,
        previous_response_id=None,
        request=conflicting,
        canonical_identity=value._canonical_response_identity(conflicting),
        call_ids=("call-known",),
    )
    ambiguous = replace(fallback, request_id="resp-ambiguous")
    ambiguous_key = value._response_conversation_key(
        request_id=ambiguous.request_id,
        previous_response_id=None,
        request=ambiguous,
        call_ids=("call-known",),
    )
    assert ambiguous_key not in {seed_key, conflicting_key}
    ambiguous_next = replace(ambiguous, request_id="resp-ambiguous-next")
    assert value._response_conversation_key(
        request_id=ambiguous_next.request_id,
        previous_response_id=None,
        request=ambiguous_next,
        call_ids=("call-known",),
    ) != ambiguous_key


@pytest.mark.asyncio
async def test_prompt_cache_keeps_reflection_rows_and_pending_work_cumulative(tmp_path):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        feature_flags=replace(value.config.feature_flags),
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


def reflection_lifecycle_runtime(tmp_path, monkeypatch):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        reflection_memory_mode="active",
        reflection_memory_min_events=10,
        reflection_memory_min_tokens=0,
        max_internal_tokens=12288,
    )
    value.tokenizer_manager.tokenizer = SimpleNamespace(
        encode=lambda text, **_kwargs: text.split()
    )

    async def skip_stage_summary(_request_id):
        return None

    monkeypatch.setattr(value, "_emit_stage_summary", skip_stage_summary)
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_role,with_tools", [("system", True), ("user", False)])
async def test_client_compaction_reflection_survives_next_request(
    tmp_path, monkeypatch, changed_role, with_tools
):
    value = reflection_lifecycle_runtime(tmp_path, monkeypatch)
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    captured = []
    published = []

    class PausedReflectionService:
        async def reflect(self, **kwargs):
            captured.append(kwargs)
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            if not kwargs["tool_ledger"] and not kwargs.get(
                "allow_without_tool_events"
            ):
                return None
            published.append(kwargs)
            return SimpleNamespace(document_path="reflection-memory/checkpoint.md")

    value.reflection_memory_service = PausedReflectionService()
    original_task = "Repair the synthetic streaming decoder"
    old_assistant = "The original decoder dropped the final buffered byte."
    old_tool = "Regression evidence: expected final byte 7, observed no byte."
    first = FakeRequest(
        request_id="resp-before-client-compact",
        prompt_cache_key="omp:compaction-lifecycle",
        input=[
            {"role": "system", "content": "Inspect the decoder before editing."},
            {"role": "user", "content": original_task},
            {"role": "assistant", "content": old_assistant},
            *(
                [
                    {
                        "type": "function_call_output",
                        "call_id": "decoder-check",
                        "output": old_tool,
                    }
                ]
                if with_tools
                else []
            ),
        ],
    )
    compacted_text = "CLIENT COMPACTED SUMMARY: continue from the shortened history."
    compacted = replace(
        first,
        request_id="resp-client-compact",
        input=[
            {
                "role": "system",
                "content": (
                    compacted_text
                    if changed_role == "system"
                    else first.input[0]["content"]
                ),
            },
            {
                "role": "user",
                "content": compacted_text if changed_role == "user" else original_task,
            },
        ],
    )
    try:
        await value.prepare_responses_request(first)
        await value.complete_request(first.request_id)
        unchanged = replace(first, request_id="resp-before-compact-unchanged")
        await value.prepare_responses_request(unchanged)
        await value.complete_request(unchanged.request_id)
        assert not started.is_set()
        assert any(
            event.event_type == "reflection_memory.skipped"
            and event.payload["reason"] == "minimum_external_tool_events_not_met"
            for event in value.telemetry.events(first.request_id)
        )

        await value.prepare_responses_request(compacted)
        await value.complete_request(compacted.request_id)
        await asyncio.wait_for(started.wait(), timeout=2)
        following = replace(
            compacted,
            request_id="resp-after-client-compact",
            input=[
                *compacted.input,
                {"role": "assistant", "content": "NEW POST-COMPACTION WORK"},
            ],
        )
        await value.prepare_responses_request(following)
        await value.complete_request(following.request_id)
        release.set()
        await asyncio.wait_for(value._compaction_reflection_queue.join(), timeout=2)

        assert not cancelled.is_set()
        completed = [
            event
            for event in value.telemetry.events()
            if event.event_type == "reflection_memory.compaction_checkpoint_completed"
        ]
        assert len(completed) == 1
        assert completed[0].payload["status"] == "published"
        assert (
            completed[0].payload["document_path"] == "reflection-memory/checkpoint.md"
        )
        assert len(captured) == len(published) == 1
        snapshot = published[0]
        assert snapshot["original_task"] == original_task
        evidence = json.dumps(snapshot["trajectory_history"])
        assert old_assistant in evidence
        assert original_task in evidence
        assert compacted_text not in evidence
        assert "NEW POST-COMPACTION WORK" not in evidence
        if with_tools:
            assert old_tool in evidence
            assert old_tool in json.dumps(snapshot["tool_ledger"])
        else:
            assert not snapshot["tool_ledger"]
        queued = [
            event
            for event in value.telemetry.events()
            if event.event_type == "reflection_memory.compaction_checkpoint_queued"
        ]
        assert len(queued) == 1
        assert (
            queued[0].payload["checkpoint_id"] == completed[0].payload["checkpoint_id"]
        )
    finally:
        release.set()
        await value.close()


@pytest.mark.asyncio
async def test_client_compaction_does_not_reflect_when_disabled(tmp_path, monkeypatch):
    value = reflection_lifecycle_runtime(tmp_path, monkeypatch)
    value.config = replace(value.config, reflection_memory_mode="off")
    calls = []

    class ReflectionService:
        async def reflect(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(document_path="reflection-memory/unexpected.md")

    value.reflection_memory_service = ReflectionService()
    first = FakeRequest(
        request_id="resp-disabled-before",
        prompt_cache_key="omp:disabled-compaction",
        input=[{"role": "user", "content": "Original synthetic task"}],
    )
    try:
        await value.prepare_responses_request(first)
        await value.complete_request(first.request_id)
        compacted = replace(
            first,
            request_id="resp-disabled-after",
            input=[{"role": "user", "content": "Compacted synthetic task"}],
        )
        await value.prepare_responses_request(compacted)
        await value.complete_request(compacted.request_id)
        await asyncio.wait_for(value._compaction_reflection_queue.join(), timeout=2)
        assert calls == []
        assert not any(
            event.event_type.startswith("reflection_memory.compaction_checkpoint_")
            for event in value.telemetry.events()
        )
    finally:
        await value.close()


@pytest.mark.asyncio
async def test_same_conversation_request_still_cancels_ordinary_idle_reflection(
    tmp_path, monkeypatch
):
    value = reflection_lifecycle_runtime(tmp_path, monkeypatch)
    value.config = replace(value.config, reflection_memory_min_events=2)
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    published = []

    class PausedReflectionService:
        async def reflect(self, **kwargs):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            published.append(kwargs)

    value.reflection_memory_service = PausedReflectionService()
    first = FakeRequest(
        request_id="resp-ordinary-idle",
        prompt_cache_key="omp:ordinary-idle",
        input=[
            {"role": "user", "content": "Inspect the synthetic decoder"},
            {"type": "function_call", "call_id": "idle-check", "name": "inspect"},
            {
                "type": "function_call_output",
                "call_id": "idle-check",
                "output": "Found a dropped byte",
            },
            {"type": "function_call", "call_id": "idle-check-2", "name": "inspect"},
            {
                "type": "function_call_output",
                "call_id": "idle-check-2",
                "output": "Confirmed the same dropped byte",
            },
        ],
    )
    try:
        await value.prepare_responses_request(first)
        conversation_key = value._request_conversation_keys[first.request_id]
        value._context_integrity_ledgers[conversation_key] = [
            {"call_id": "idle-check", "observation": "Found a dropped byte"},
            {
                "call_id": "idle-check-2",
                "observation": "Confirmed the same dropped byte",
            },
        ]
        await value.complete_request(first.request_id)
        pending = value.pending_reflection_memories()
        assert len(pending) == 1
        value.start_pending_reflections([pending[0]["conversation_key"]])
        await asyncio.wait_for(started.wait(), timeout=2)
        await value.prepare_responses_request(
            replace(first, request_id="resp-ordinary-next")
        )
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        release.set()
        assert published == []
        assert value.pending_reflection_memories() == []
        assert not any(
            event.event_type == "reflection_memory.compaction_checkpoint_queued"
            for event in value.telemetry.events()
        )
    finally:
        release.set()
        await value.close()


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
    prepared = asyncio.run(
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


def test_reflection_checkpoint_keeps_evidence_evicted_from_hot_window(tmp_path):
    value = runtime(tmp_path)
    value.config = replace(
        value.config,
        reflection_memory_mode="active",
        reflection_memory_max_history_tokens=1024,
        feature_flags=replace(value.config.feature_flags),
        max_internal_tokens=16384,
    )
    early = "early decisive evidence " + "x" * 5000
    value._record_reflection_memory_rows(
        "evidence-conversation",
        "resp-early",
        ({"kind": "tool_observation", "call_id": "early", "content": early},),
    )
    value._record_reflection_memory_rows(
        "evidence-conversation",
        "resp-late",
        ({"kind": "tool_observation", "call_id": "late", "content": "y" * 3000},),
    )
    history = value._reflection_memory_history("evidence-conversation")
    assert [row["call_id"] for row in history] == ["early", "late"]
    assert history[0]["content"] == early
    recovered = value.reflection_evidence_store.get_event(history[0]["event_id"])
    assert recovered["content"] == early


@pytest.mark.asyncio
async def test_memory_finalizes_and_slot_frees_before_slow_capsule(tmp_path):
    """Agent loops send the next turn right after a tool call. The previous
    turn's capsule is a low-priority generation that can wait tens of seconds
    behind other agents; neither the client slot nor the next turn's memory
    parent may wait for it."""
    value = runtime(tmp_path)
    release = asyncio.Event()

    async def slow_capsule():
        await release.wait()

    value._request_questions["resp-1"] = "task"
    value._request_conversation_keys["resp-1"] = "conversation-1"
    value._capsule_tasks["resp-1"] = asyncio.create_task(slow_capsule())
    value._memory_finalized["resp-1"] = asyncio.Event()
    finishing = asyncio.create_task(value._finish_request("resp-1"))
    value._finalize_tasks["resp-1"] = finishing

    await asyncio.wait_for(value._memory_finalized["resp-1"].wait(), 5)
    assert not finishing.done()
    assert value._active_foreground_requests() == 0

    release.set()
    await finishing


def test_capacity_counts_only_generating_requests(tmp_path):
    value = runtime(tmp_path)
    value._request_questions = {"generating": "a", "finalizing": "b"}
    value._finalize_tasks = {"finalizing": object()}
    assert value._active_foreground_requests() == 1


@pytest.mark.asyncio
async def test_compaction_checkpoint_never_blocks_a_user_request(tmp_path):
    """Regression: the compaction reflection worker waits for user requests to
    go quiet, while user requests that detected client compaction awaited a
    blocking put() into its bounded queue. Once the queue filled, every such
    request held its client slot forever and the server answered 429."""
    value = runtime(tmp_path)
    release = asyncio.Event()

    class ReflectionMemoryService:
        async def reflect(self, **kwargs):
            await release.wait()

    value.reflection_memory_service = ReflectionMemoryService()
    checkpoint = SimpleNamespace(
        response_id="resp", checkpoint_id="cp", conversation_key="conversation",
        trajectory_history=(), source_token_count=1, original_task="task",
        tool_ledger=(), capsule_history=(),
    )
    capacity = value._compaction_reflection_queue.maxsize
    results = []
    for _ in range(capacity + 3):
        results.append(
            await asyncio.wait_for(
                value._enqueue_compaction_reflection_checkpoint(checkpoint), 1
            )
        )
        await asyncio.sleep(0)
    assert results[0] is True
    assert results[-1] is False
    release.set()
    value._compaction_reflection_worker.cancel()
    await asyncio.gather(value._compaction_reflection_worker, return_exceptions=True)


def test_memory_block_is_offered_every_turn_even_after_a_search(tmp_path):
    value = runtime(tmp_path)

    def turn(request_id, blocked_reason=None):
        value._request_conversation_keys[request_id] = "conv"
        return asyncio.run(
            value.memory_block_for_turn(
                request_id, blocked_reason=blocked_reason, query_text="deploy"
            )
        )

    assert all(turn(f"a{index}") for index in range(10))
    asyncio.run(
        value.record_memory_tool_round(
            "a9",
            0,
            [
                {
                    "id": "call-1",
                    "name": "qwen_exo_memory_search",
                    "arguments": "{}",
                    "ok": True,
                    "output": "no matches",
                    "duration_ms": 1.0,
                }
            ],
        )
    )
    assert turn("b0") is not None
    assert turn("c0", blocked_reason="tool_choice") is None
    stats = value.status()["memory_tools"]
    assert (stats["blocks_shown"], stats["searches_after_block"]) == (11, 1)
    assert (stats["offered"], stats["not_offered:tool_choice"]) == (11, 1)
