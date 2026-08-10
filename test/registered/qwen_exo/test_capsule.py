import asyncio
import json

from qwen_exo_booster.capsule import (
    CapsuleUpdateInput,
    ExecutionCapsuleService,
    ExecutionCapsuleStore,
    parse_execution_capsule,
)
from qwen_exo_booster.internal_jobs import InternalJobResult

VALID_CAPSULE = {
    "summary": "Implemented the cache guard.",
    "phase": "verification",
    "established": ["The unit test passes."],
    "unresolved": ["GPU behavior is not verified."],
    "next_action": "Run the GPU smoke test.",
    "event": "PROGRESS",
    "state_change": "YES",
    "verification": "YES",
    "repetition": "NO",
}


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"] + messages[1]["content"]


class FakeRunner:
    def __init__(self, text=None, finish_reason=None):
        self.text = text or json.dumps(VALID_CAPSULE)
        self.finish_reason = finish_reason or {"type": "stop"}
        self.calls = 0
        self.max_fanout = 8

    async def run_batch(self, jobs, prompts, sampling_params):
        self.calls += 1
        job = tuple(jobs)[0]
        assert sampling_params["json_schema"]
        return (
            InternalJobResult(
                job=job,
                text=self.text,
                prompt_tokens=100,
                completion_tokens=32,
                finish_reason=self.finish_reason,
                latency_seconds=0.1,
            ),
        )


def update(sequence=1, **overrides):
    values = {
        "parent_request_id": "parent-1",
        "turn_id": f"turn-{sequence}",
        "trajectory_id": "trajectory-1",
        "event_sequence": sequence,
        "original_task": "Fix the cache",
        "previous_capsule": None,
        "assistant_reasoning": "The cache guard is implemented.",
        "assistant_tool_calls": (),
        "tool_observation": "Tests passed.",
        "telemetry_correlation_id": "trace-1",
    }
    values.update(overrides)
    return CapsuleUpdateInput(**values)


def test_capsule_parser_is_strict():
    assert parse_execution_capsule(json.dumps(VALID_CAPSULE)) == VALID_CAPSULE
    assert parse_execution_capsule('{"summary":"partial"}') is None
    duplicate = json.dumps(VALID_CAPSULE)[:-1] + ',"summary":"duplicate"}'
    assert parse_execution_capsule(duplicate) is None
    assert parse_execution_capsule(json.dumps({**VALID_CAPSULE, "extra": True})) is None
    assert (
        parse_execution_capsule(json.dumps({**VALID_CAPSULE, "event": "DONE"})) is None
    )
    assert (
        parse_execution_capsule(json.dumps({**VALID_CAPSULE, "established": [1]}))
        is None
    )


def test_valid_capsule_is_persisted_and_restored(tmp_path):
    path = tmp_path / "capsules.json"
    runner = FakeRunner()
    service = ExecutionCapsuleService(
        runner, ExecutionCapsuleStore(path), FakeTokenizer()
    )

    result = asyncio.run(service.update_many([update()]))[0]

    assert result.valid
    assert result.record.capsule == VALID_CAPSULE
    assert path.is_file()
    restored = ExecutionCapsuleStore(path).get("trajectory-1")
    assert restored is not None
    assert restored.capsule == VALID_CAPSULE


def test_invalid_capsule_does_not_overwrite_previous_state(tmp_path):
    store = ExecutionCapsuleStore(tmp_path / "capsules.json")
    valid_service = ExecutionCapsuleService(FakeRunner(), store, FakeTokenizer())
    first = asyncio.run(valid_service.update_many([update()]))[0]
    malformed_service = ExecutionCapsuleService(
        FakeRunner('{"summary":"partial"}'), store, FakeTokenizer()
    )

    second = asyncio.run(
        malformed_service.update_many(
            [
                update(
                    2,
                    previous_capsule=VALID_CAPSULE,
                    assistant_reasoning="new but invalid",
                )
            ]
        )
    )[0]

    assert not second.valid
    assert store.get("trajectory-1") == first.record


def test_duplicate_event_does_not_schedule_another_job(tmp_path):
    store = ExecutionCapsuleStore(tmp_path / "capsules.json")
    runner = FakeRunner()
    service = ExecutionCapsuleService(runner, store, FakeTokenizer())
    item = update()

    first = asyncio.run(service.update_many([item]))[0]
    second = asyncio.run(service.update_many([item]))[0]

    assert first.valid
    assert second.deduplicated
    assert runner.calls == 1


def test_out_of_order_event_cannot_replace_newer_capsule(tmp_path):
    store = ExecutionCapsuleStore(tmp_path / "capsules.json")
    runner = FakeRunner()
    service = ExecutionCapsuleService(runner, store, FakeTokenizer())
    newest = update(3)
    older = update(2, assistant_reasoning="late old event")

    latest_result = asyncio.run(service.update_many([newest]))[0]
    old_result = asyncio.run(service.update_many([older]))[0]

    assert old_result.deduplicated
    assert old_result.record == latest_result.record
    assert runner.calls == 1


def test_equal_sequence_cannot_overwrite_or_schedule_again(tmp_path):
    store = ExecutionCapsuleStore(tmp_path / "capsules.json")
    runner = FakeRunner()
    service = ExecutionCapsuleService(runner, store, FakeTokenizer())

    first = asyncio.run(service.update_many([update(1)]))[0]
    equal = asyncio.run(
        service.update_many([update(1, assistant_reasoning="conflicting branch")])
    )[0]

    assert equal.deduplicated
    assert equal.record == first.record
    assert runner.calls == 1


def test_capsule_store_retention_is_bounded_across_restart(tmp_path):
    path = tmp_path / "capsules.json"
    store = ExecutionCapsuleStore(path, max_records=2)
    for index in range(3):
        item = update(
            index,
            trajectory_id=f"response-{index}",
            turn_id=f"response-{index}",
        )
        assert store.commit(item, VALID_CAPSULE) is not None

    assert store.get("response-0") is None
    assert store.get("response-1") is not None
    assert store.get("response-2") is not None

    restored = ExecutionCapsuleStore(path, max_records=2)
    assert restored.get("response-0") is None
    assert restored.get("response-1") is not None
    assert restored.get("response-2") is not None


def test_invalid_previous_capsule_never_reaches_scheduler(tmp_path):
    store = ExecutionCapsuleStore(tmp_path / "capsules.json")
    runner = FakeRunner()
    service = ExecutionCapsuleService(runner, store, FakeTokenizer())

    result = asyncio.run(
        service.update_many([update(previous_capsule={"summary": "partial"})])
    )[0]

    assert not result.valid
    assert result.record is None
    assert runner.calls == 0


def test_non_stop_capsule_output_cannot_commit(tmp_path):
    path = tmp_path / "capsules.json"
    runner = FakeRunner(finish_reason={"type": "length"})
    service = ExecutionCapsuleService(
        runner, ExecutionCapsuleStore(path), FakeTokenizer()
    )

    result = asyncio.run(service.update_many([update()]))[0]

    assert not result.valid
    assert result.record is None
    assert not path.exists()


def test_corrupt_or_wrong_schema_store_fails_closed(tmp_path):
    path = tmp_path / "capsules.json"
    path.write_text("{not-json", encoding="utf-8")
    assert ExecutionCapsuleStore(path).get("trajectory-1") is None

    path.write_text(json.dumps({"schema": 99, "records": []}), encoding="utf-8")
    assert ExecutionCapsuleStore(path).get("trajectory-1") is None
