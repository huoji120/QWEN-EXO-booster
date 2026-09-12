import asyncio
import threading
import sqlite3
import pytest
import json
from collections import OrderedDict
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwen_exo_booster.attention_diagnostic_conversations import (
    AttentionDiagnosticConversations,
)
from qwen_exo_booster.reflection_evidence import ReflectionEvidenceStore
from qwen_exo_booster.reflection_memory import (
    ReflectionMemoryStore,
    ReflectionSourceSnapshot,
    ReflectionSourceStore,
)
from qwen_exo_booster.router import router
from qwen_exo_booster.runtime import QwenExoRuntime


def session_runtime(path):
    runtime = object.__new__(QwenExoRuntime)
    runtime.reflection_evidence_store = ReflectionEvidenceStore(
        path / "evidence.sqlite3"
    )
    runtime.reflection_source_store = ReflectionSourceStore(path / "sources.sqlite3")
    runtime._reflection_memory_trajectories = OrderedDict()
    runtime._pending_reflection_memories = OrderedDict()
    runtime._reflection_memory_sources = OrderedDict()
    runtime._reflection_memory_last_activity = OrderedDict()
    runtime._reflection_memory_tasks = {}
    runtime._request_conversation_keys = {}
    runtime._finalize_tasks = {}
    return runtime


def retain(runtime, key):
    rows = [{"kind": "user_context", "content": f"original {key}"}]
    runtime.reflection_evidence_store.append_rows(key, f"req-{key}", rows)
    runtime._reflection_memory_trajectories[key] = rows
    for suffix in ("old", "new"):
        runtime.reflection_source_store.save(
            ReflectionSourceSnapshot(
                source_digest=f"{key}-{suffix}",
                trajectory_id=f"req-{key}",
                conversation_key=key,
                original_task="retained source",
                trajectory_history=tuple(rows),
                capsule_history=(),
                verifier_feedback="",
                source_event_count=1,
                source_token_count=4,
                source_audit={},
                captured_at=10.0,
            )
        )


def api_for(runtime):
    app = FastAPI()
    app.state.qwen_exo_runtime = runtime
    app.include_router(router)
    return TestClient(app)


def test_delete_session_removes_all_import_sources_and_allows_fresh_history(tmp_path):
    runtime = session_runtime(tmp_path)
    retain(runtime, "selected")
    retain(runtime, "kept")
    published = {
        "source_digest": "lesson",
        "conversation_key": "selected",
        "source_snapshot_digest": "selected-new",
        "publication_status": "published",
        "document_path": "reflection-memory/lesson.md",
    }
    record_path = tmp_path / "reflections.json"
    record_path.write_text(json.dumps([published]), encoding="utf-8")
    runtime.reflection_memory_store = ReflectionMemoryStore(record_path)
    importer = AttentionDiagnosticConversations(runtime)
    selected_id = next(
        row["id"]
        for row in importer.list()["conversations"]
        if row["title"] == "selected"
    )
    with api_for(runtime) as api:
        result = api.post(
            "/qwen-exo/server-sessions/delete", json={"conversation_keys": ["selected"]}
        )
        assert result.status_code == 200, result.text
        assert result.json()["deleted"] == ["selected"]
        assert {
            row["conversation_key"]
            for row in api.get("/qwen-exo/server-sessions").json()["sessions"]
        } == {"kept"}
    assert importer.detail(selected_id) is None
    assert runtime.reflection_source_store.get("selected-old") is None
    assert runtime.reflection_source_store.get("selected-new") is None
    assert runtime.reflection_evidence_store.rows("selected") == ()
    assert importer.list()["total"] == 1
    assert runtime.reflection_memory_store.get("lesson") == published
    assert json.loads(record_path.read_text(encoding="utf-8")) == [published]
    runtime.reflection_evidence_store.append_rows(
        "selected", "new-request", [{"kind": "user_context", "content": "fresh only"}]
    )
    rows = runtime.reflection_evidence_store.rows("selected")
    assert [row["content"] for row in rows] == ["fresh only"]


def test_clear_all_is_not_limited_to_page_and_skips_active(tmp_path):
    runtime = session_runtime(tmp_path)
    for key in ("one", "two", "active"):
        retain(runtime, key)
    runtime._request_conversation_keys["live-request"] = "active"
    with api_for(runtime) as api:
        page = api.get(
            "/qwen-exo/server-sessions", params={"limit": 1, "q": "one"}
        ).json()
        assert [row["conversation_key"] for row in page["sessions"]] == ["one"]
        result = api.post("/qwen-exo/server-sessions/delete", json={"all": True})
        assert result.status_code == 200, result.text
        assert set(result.json()["deleted"]) == {"one", "two"}
        assert result.json()["skipped_active"] == ["active"]
        assert runtime.reflection_evidence_store.rows("active")
        runtime._request_conversation_keys.clear()
        result = api.post("/qwen-exo/server-sessions/delete", json={"all": True})
        assert result.status_code == 200
        assert api.get("/qwen-exo/server-sessions").json()["total"] == 0


def test_invalid_selection_never_clears_retained_sessions(tmp_path):
    runtime = session_runtime(tmp_path)
    retain(runtime, "kept")
    with api_for(runtime) as api:
        for payload in (
            {},
            {"all": False},
            {"all": True, "conversation_keys": ["kept"]},
            {"conversation_keys": []},
            {"conversation_keys": [""]},
            {"conversation_keys": "kept"},
            {"all": "true"},
        ):
            response = api.post("/qwen-exo/server-sessions/delete", json=payload)
            assert response.status_code == 422, (payload, response.text)
        assert api.get("/qwen-exo/server-sessions").json()["total"] == 1


def test_source_delete_rolls_back_journal_when_snapshot_delete_fails(tmp_path):
    from qwen_exo_booster.server_sessions import ServerSessionStore

    runtime = session_runtime(tmp_path)
    retain(runtime, "kept")
    with runtime.reflection_source_store._connect() as db:
        db.execute(
            "CREATE TRIGGER reject_delete BEFORE DELETE ON reflection_sources BEGIN SELECT RAISE(ABORT, 'storage failure'); END"
        )
    with pytest.raises(sqlite3.DatabaseError):
        asyncio.run(ServerSessionStore(runtime).delete(conversation_keys=["kept"]))
    assert runtime.reflection_evidence_store.rows("kept")
    assert runtime.reflection_source_store.get("kept-new") is not None
    assert "kept" in runtime._reflection_memory_trajectories


def test_cancelled_delete_waits_for_commit_before_allowing_new_source(
    tmp_path, monkeypatch
):
    from qwen_exo_booster.server_sessions import (
        ServerSessionStore,
        retained_source_producer,
    )

    runtime = session_runtime(tmp_path)
    retain(runtime, "old")
    entered, release = threading.Event(), threading.Event()
    store = ServerSessionStore(runtime)
    erase = store._erase_databases

    def blocked_erase(keys):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("test did not release transaction")
        erase(keys)

    monkeypatch.setattr(store, "_erase_databases", blocked_erase)

    @retained_source_producer
    async def new_source(value):
        value.reflection_evidence_store.append_rows(
            "old", "fresh", [{"kind": "user_context", "content": "new request"}]
        )

    async def scenario():
        deletion = asyncio.create_task(store.delete(conversation_keys=["old"]))
        assert await asyncio.to_thread(entered.wait, 5)
        deletion.cancel()
        producer = asyncio.create_task(new_source(runtime))
        try:
            await asyncio.sleep(0)
            assert not producer.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await deletion
        await producer

    asyncio.run(scenario())
    assert [
        row["content"] for row in runtime.reflection_evidence_store.rows("old")
    ] == ["new request"]


def test_pending_source_deletion_preserves_serving_lineage(tmp_path):
    runtime = session_runtime(tmp_path)
    retain(runtime, "waiting")
    runtime._pending_reflection_memories["waiting"] = SimpleNamespace(status="waiting")
    runtime._conversation_keys_by_response_id = {"previous-response": "waiting"}
    runtime._memory_parents_by_conversation = {"waiting": "previous-response"}
    with api_for(runtime) as api:
        row = api.get("/qwen-exo/server-sessions").json()["sessions"][0]
        assert row["pending_reflection"] and not row["active"]
        assert row["source_count"] == 2 and row["event_count"] == 1
        result = api.post("/qwen-exo/server-sessions/delete", json={"all": True})
        assert result.json()["deleted"] == ["waiting"]
        assert api.get("/qwen-exo/server-sessions").json()["total"] == 0
    assert runtime._conversation_keys_by_response_id == {"previous-response": "waiting"}
    assert runtime._memory_parents_by_conversation == {"waiting": "previous-response"}
