import tempfile
from pathlib import Path

import pytest
from qwen_exo_booster.reflection_evidence import (
    ReflectionEvidenceStore,
    merge_causal_entries,
)


def test_events_recover_and_replay_without_collapsing_calls():
    with tempfile.TemporaryDirectory() as d:
        store = ReflectionEvidenceStore(Path(d) / "evidence.sqlite")
        rows = [
            {"kind": "tool_call", "tool_name": "read", "call_id": "a", "content": "x"},
            {"kind": "tool_call", "tool_name": "read", "call_id": "b", "content": "x"},
        ]
        first = store.append_rows("conversation", "request-1", rows)
        assert len(first) == 2 and first[0]["event_id"] != first[1]["event_id"]
        replayed = store.append_rows("conversation", "request-2", rows)
        assert [row["event_id"] for row in replayed] == [
            row["event_id"] for row in first
        ]
        recovered = ReflectionEvidenceStore(Path(d) / "evidence.sqlite")
        assert [x["call_id"] for x in recovered.rows("conversation")] == ["a", "b"]
        assert recovered.get_event(first[0]["event_id"])["content"] == "x"


def test_analysis_collision_and_causal_stale_preservation():
    with tempfile.TemporaryDirectory() as d:
        store = ReflectionEvidenceStore(Path(d) / "evidence.sqlite")
        store.save_analysis("segment", {"status": "ok"})
        with pytest.raises(ValueError):
            store.save_analysis("segment", {"status": "changed"})
    old = {"entry_id": "e1", "version": 2, "title": "old", "admission_status": "active"}
    result = merge_causal_entries(
        (old,),
        ({"operation": "add", "reason": "new", "entry": {"title": "new"}},),
        source_digest="s",
    )
    assert result[0] == old
    with pytest.raises(ValueError):
        merge_causal_entries(
            result,
            (
                {
                    "operation": "revise",
                    "entry_id": "e1",
                    "expected_version": 1,
                    "reason": "bad",
                    "entry": {"title": "bad"},
                },
            ),
            source_digest="s",
        )
    revised = merge_causal_entries(
        result,
        (
            {
                "operation": "revise",
                "entry_id": "e1",
                "expected_version": 2,
                "reason": "new evidence",
                "entry": {"title": "next"},
            },
        ),
        source_digest="s",
    )
    assert revised[0]["version"] == 3 and revised[0]["versions"][0]["title"] == "old"


def test_cumulative_overlap_does_not_erase_a_later_equal_action(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "evidence.sqlite")
    a = {"kind": "assistant_trajectory", "content": "inspect state"}
    b = {"kind": "user_request", "content": "continue"}
    first = store.append_rows("c", "r1", (a, b))
    replay = store.append_rows("c", "r2", (a, b, a), replay=True)
    assert [row["event_id"] for row in replay[:2]] == [row["event_id"] for row in first]
    assert replay[2]["event_id"] != first[0]["event_id"]
    later = store.append_rows("c", "r3", (a,))
    assert later[0]["event_id"] != replay[2]["event_id"]
    assert [row["content"] for row in store.rows("c")] == [
        "inspect state",
        "continue",
        "inspect state",
        "inspect state",
    ]


def test_retention_keeps_current_evidence_and_reports_purged_scope(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "evidence.sqlite", max_bytes=1024)
    old = store.append_rows("old", "r1", ({"content": "a" * 800},))[0]
    current = store.append_rows("current", "r2", ({"content": "b" * 800},))[0]
    assert store.get_event(old["event_id"]) is None
    assert store.get_event(current["event_id"])["content"] == "b" * 800
    assert store.retention_metadata("old")["gap"] is True
    assert store.retention_metadata("current")["gap"] is False
    assert store.retention_metadata()["purged_bytes"] == 800
    with pytest.raises(ValueError, match="capacity exceeded"):
        store.append_rows("current", "r3", ({"content": "c" * 1100},))
    assert [row["event_id"] for row in store.rows("current")] == [current["event_id"]]


def test_foreign_event_id_cannot_overwrite_or_read_another_scope(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "evidence.sqlite")
    original = store.append_rows(
        "owner", "r1", ({"kind": "tool_observation", "content": "exact"},)
    )[0]
    with pytest.raises(ValueError, match="collision"):
        store.append_rows("other", "r2", ({**original, "content": "forged"},))
    assert store.rows("other", (original["event_id"],)) == ()
    assert store.get_event(original["event_id"])["content"] == "exact"


def test_private_analysis_retention_does_not_evict_source_evidence(tmp_path):
    store = ReflectionEvidenceStore(tmp_path / "evidence.sqlite", max_bytes=1024)
    source = store.append_rows("c", "r", ({"content": "原始工具证据"},))[0]
    store.save_analysis("old", {"text": "a" * 600})
    store.save_analysis("current", {"text": "b" * 600})
    recovered = ReflectionEvidenceStore(store.path, max_bytes=1024)
    assert recovered.get_analysis("old") is None
    assert recovered.get_analysis("current")["text"] == "b" * 600
    assert recovered.get_event(source["event_id"])["content"] == "原始工具证据"
    with pytest.raises(ValueError, match="capacity exceeded"):
        store.save_analysis("oversized", {"text": "x" * 1024})
    assert recovered.get_analysis("oversized") is None
    assert recovered.get_analysis("current") is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
