import json

import pytest
from qwen_exo_booster.observer import InFlightObserver
from qwen_exo_booster.telemetry import TelemetryStore




def test_telemetry_redacts_sensitive_text(tmp_path):
    store = TelemetryStore(tmp_path / "trace.jsonl")
    event = store.emit(
        "request-1",
        "judge.completed",
        {
            "candidate_id": "candidate-1",
            "reference_text": "secret reference",
            "tool_arguments": {"value": 1},
            "private_attachment": "private memory",
            "original_task": "customer task",
            "nested": {"normalized_content": "normalized secret"},
        },
    )

    assert event.payload["reference_text"]["redacted"] is True
    assert event.payload["reference_text"]["bytes"] > 0
    assert event.payload["private_attachment"]["redacted"] is True
    assert event.payload["original_task"]["redacted"] is True
    assert event.payload["nested"]["normalized_content"]["redacted"] is True
    assert event.payload["candidate_id"] == "candidate-1"
    persisted = json.loads((tmp_path / "trace.jsonl").read_text(encoding="utf-8"))
    encoded = json.dumps(persisted)
    assert "secret reference" not in encoded
    assert "private memory" not in encoded
    assert "customer task" not in encoded
    assert "normalized secret" not in encoded


def test_telemetry_normalizes_non_finite_floats_to_strict_json(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = TelemetryStore(path)

    event = store.emit(
        "request-non-finite",
        "observer.decode_summary",
        {
            "positive": float("inf"),
            "negative": float("-inf"),
            "unknown": float("nan"),
            "nested": [1.0, float("inf")],
        },
    )

    assert event.payload == {
        "positive": "Infinity",
        "negative": "-Infinity",
        "unknown": "NaN",
        "nested": [1.0, "Infinity"],
    }
    persisted = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: pytest.fail(f"non-finite JSON constant: {value}"),
    )
    assert persisted["payload"] == event.payload


def test_telemetry_sequences_are_per_request(tmp_path):
    store = TelemetryStore(tmp_path / "trace.jsonl")

    assert store.emit("a", "one", {}).sequence == 0
    assert store.emit("a", "two", {}).sequence == 1
    assert store.emit("b", "one", {}).sequence == 0
    assert [event.event_type for event in store.events("a")] == ["one", "two"]


def test_telemetry_continues_ids_and_sequences_after_restart(tmp_path):
    path = tmp_path / "trace.jsonl"
    first = TelemetryStore(path, max_events=4)
    assert first.emit("request-1", "one", {}).event_id == 0
    assert first.emit("request-1", "two", {}).sequence == 1

    restarted = TelemetryStore(path, max_events=4)
    event = restarted.emit("request-1", "three", {})

    assert event.event_id == 2
    assert event.sequence == 2
    assert [item.event_id for item in restarted.events()] == [0, 1, 2]


def test_restart_redacts_trace_written_by_prior_raw_text_mode(tmp_path):
    path = tmp_path / "trace.jsonl"
    raw = TelemetryStore(path, include_text=True)
    raw.emit("request-raw", "raw", {"prompt": "prior secret prompt"})
    assert "prior secret prompt" in path.read_text(encoding="utf-8")

    redacted = TelemetryStore(path, include_text=False)

    assert redacted.events()[0].payload["prompt"]["redacted"] is True
    assert "prior secret prompt" not in path.read_text(encoding="utf-8")


def test_telemetry_disk_retention_is_bounded(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = TelemetryStore(path, max_events=2, max_file_bytes=1024 * 1024)

    for index in range(7):
        store.emit(f"request-{index}", "event", {"index": index})

    persisted = path.read_text(encoding="utf-8").splitlines()
    assert len(store.events()) == 2
    assert len(persisted) <= 3
    assert store.persistence_status()["disk_events"] <= 3


def test_telemetry_write_failure_does_not_break_inference(tmp_path, monkeypatch):
    store = TelemetryStore(tmp_path / "trace.jsonl")

    def fail_write(event):
        raise OSError("disk full")

    monkeypatch.setattr(store, "_append_locked", fail_write)
    event = store.emit("request-1", "observer.decode_summary", {"token_count": 1})

    assert event.event_id == 0
    assert store.events() == (event,)
    assert store.persistence_status()["ok"] is False
    assert "disk full" in store.persistence_status()["last_error"]


def test_observer_does_not_double_count_cumulative_stream_chunks(tmp_path):
    store = TelemetryStore(tmp_path / "trace.jsonl")
    observer = InFlightObserver(store, mode="shadow")
    first = observer.observe_generation_result(
        "request-1",
        {
            "meta_info": {
                "output_token_logprobs": [
                    (-1.0, 1),
                    (-1.0, 2),
                    (-7.0, 3),
                    (-7.0, 4),
                ],
                "finish_reason": None,
                "qwen_exo_q_drift": [0.0, 0.0, 0.0, 0.42],
            }
        },
    )
    second = observer.observe_generation_result(
        "request-1",
        {
            "meta_info": {
                "output_token_logprobs": [
                    (-1.0, 1),
                    (-1.0, 2),
                    (-7.0, 3),
                    (-7.0, 4),
                    (-2.0, 5),
                    (-2.0, 6),
                    (-8.0, 7),
                    (-8.0, 8),
                ],
                "qwen_exo_q_drift": [0.0, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.42],
                "finish_reason": {"type": "stop"},
            }
        },
    )

    assert first.new_tokens == 4
    assert second.new_tokens == 4
    assert second.token_count == 8
    assert observer.surprisal_history("request-1") == (1.0, 1.0, 7.0, 7.0, 2.0, 2.0, 8.0, 8.0)


def test_observer_counts_each_incremental_stream_chunk(tmp_path):
    observer = InFlightObserver(TelemetryStore(tmp_path / "trace.jsonl"), mode="shadow")

    first = observer.observe_generation_result(
        "request-stream",
        {"meta_info": {"output_token_logprobs": [(-1.0, 1), (-2.0, 2)]}},
        incremental_logprobs=True,
    )
    second = observer.observe_generation_result(
        "request-stream",
        {"meta_info": {"output_token_logprobs": [(-3.0, 3), (-4.0, 4)]}},
        incremental_logprobs=True,
    )

    assert first.new_tokens == 2
    assert second.new_tokens == 2
    assert second.token_count == 4


def test_observer_resets_logprob_cursor_for_next_tool_generation(tmp_path):
    observer = InFlightObserver(TelemetryStore(tmp_path / "trace.jsonl"), mode="shadow")

    observer.observe_generation_result(
        "request-tools",
        {"meta_info": {"output_token_logprobs": [(-1.0, 1), (-2.0, 2)]}},
        generation_index=0,
    )
    second_generation = observer.observe_generation_result(
        "request-tools",
        {"meta_info": {"output_token_logprobs": [(-3.0, 3)]}},
        generation_index=1,
    )

    assert second_generation.new_tokens == 1
    assert second_generation.token_count == 3


def test_observer_state_is_bound_to_request_id(tmp_path):
    observer = InFlightObserver(TelemetryStore(tmp_path / "trace.jsonl"), mode="shadow")
    observer.observe_generation_result(
        "a", {"meta_info": {"output_token_logprobs": [(-1.0, 1)]}}
    )
    observer.observe_generation_result(
        "b", {"meta_info": {"output_token_logprobs": [(-2.0, 2)]}}
    )

    assert observer.state("a").token_count == 1
    assert observer.state("b").token_count == 1
    assert observer.state("a").ema_surprisal != observer.state("b").ema_surprisal




def test_attention_q_drift_and_memory_energy_are_reported(tmp_path):
    store = TelemetryStore(tmp_path / "trace.jsonl")
    observer = InFlightObserver(store, mode="shadow")

    result = observer.observe_generation_result(
        "request-q",
        {
            "meta_info": {
                "output_token_logprobs": [
                    (-1.0, 1),
                    (-1.0, 2),
                    (-4.0, 3),
                    (-4.0, 4),
                ],
                "qwen_exo_q_norm": [1.0, 1.0, 1.1, 1.2],
                "qwen_exo_q_drift": [float("nan"), 0.0, 0.0, 0.42],
                "qwen_exo_memory_energy": [0.4, 0.45, 0.55, 0.61],
                "finish_reason": {"type": "stop"},
            }
        },
    )

    assert result.latest_q_drift == 0.42
    assert result.latest_memory_energy == 0.61
    event = store.events("request-q")[-1]
    assert event.event_type == "observer.decode_summary"
    assert event.payload["memory_attention_energy_proxy"] == 0.61
    assert event.payload["max_q_drift"] == 0.42






def test_telemetry_edited_mode_records_bounded_text_only_for_edited_requests(tmp_path):
    store = TelemetryStore(tmp_path / "trace.jsonl", text_mode="edited")
    store.text_scope = lambda request_id: request_id == "edited-request"
    long_text = "x" * 2000

    plain = store.emit("plain-request", "request.completed", {"output_text": long_text})
    assert plain.payload["output_text"]["redacted"] is True

    edited = store.emit("edited-request", "request.completed", {"output_text": long_text})
    recorded = edited.payload["output_text"]
    assert isinstance(recorded, str)
    assert len(recorded) < len(long_text)
    assert "截断" in recorded


def test_telemetry_clear_truncates_disk_and_memory(tmp_path):
    path = tmp_path / "trace.jsonl"
    store = TelemetryStore(path)
    store.emit("request-1", "one", {"value": 1})
    assert store.events()
    assert path.stat().st_size > 0

    store.clear()

    assert store.events() == ()
    status = store.persistence_status()
    assert status["disk_events"] == 0
    assert status["ok"] is True










