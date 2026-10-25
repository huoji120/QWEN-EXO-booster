"""Real-filesystem HTTP coverage for explicit admin private-memory selection."""

import base64
from contextlib import asynccontextmanager
import io
from types import SimpleNamespace
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import qwen_exo_booster.router as router_module
from qwen_exo_booster.api_keys import ApiKeyStore
from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
from qwen_exo_booster.memory_tools import MemoryToolExecutor
from qwen_exo_booster.scoped_memory import get_private_memory_index
from qwen_exo_booster.work_memory import WorkMemoryTrash


@pytest.fixture(params=("private", "private_only", "super"))
def admin(tmp_path, monkeypatch, request):
    shared = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    store = ApiKeyStore(tmp_path / "keys.json")
    keys = [store.create(label, request.param)["id"] for label in ("Owner A", "Owner B")]
    monkeypatch.setattr(router_module, "_api_key_store", lambda: store)
    @asynccontextmanager
    async def transfer_guard(values):
        runtime.transfer_events.append(("before", values))
        try:
            yield
        finally:
            runtime.transfer_events.append(("after", values))

    runtime = SimpleNamespace(
        file_knowledge=shared,
        memory_transfer_guard=transfer_guard,
        transfer_events=[],
        work_memory_trash=lambda: WorkMemoryTrash(shared).entries(),
        restore_work_memory=lambda batch, path: WorkMemoryTrash(shared).restore(batch, path),
        memory_summary_status=lambda: {"available": True, "shared": True},
        memory_dream_status=lambda: {"available": True, "shared": True},
    )
    app = FastAPI()
    app.include_router(router_module.router)
    app.state.qwen_exo_runtime = runtime
    with TestClient(app) as client:
        yield client, shared, store, keys


def scope(key):
    return {"memory_key_id": key} if key else {}


def upload(client, key, path, text):
    response = client.post("/qwen-exo/knowledge/files", params=scope(key), json={
        "relative_path": path,
        "content_base64": base64.b64encode(text.encode()).decode(),
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_owner_listing_is_read_only_and_retains_all_owners(admin):
    client, shared, store, (a, b) = admin
    base = shared.state_directory / "private-memory"
    empty = client.get("/qwen-exo/knowledge/memory-spaces")
    assert empty.status_code == 200
    assert {row["key_id"] for row in empty.json()["spaces"]} == {a, b}
    assert all(row["files"] == row["bytes"] == 0 for row in empty.json()["spaces"])
    assert not base.exists()
    upload(client, a, "work-memory/a.md", "private A")
    upload(client, b, "work-memory/b.md", "private B")
    store.update_memory_scope(a, "shared")
    store.revoke(a)
    store.delete([b])
    orphan = "qxk_retained-legacy-owner"
    (base / orphan / "knowledge").mkdir(parents=True)
    (base / orphan / "knowledge" / "note.txt").write_text("retained", encoding="utf-8")
    (base / "invalid owner").mkdir()
    rows = {row["key_id"]: row for row in client.get("/qwen-exo/knowledge/memory-spaces").json()["spaces"]}
    assert set(rows) == {a, b, orphan}
    assert rows[a] == {"key_id": a, "label": "Owner A", "memory_scope": "shared", "revoked": True, "deleted": False, "files": 1, "bytes": 9}
    assert rows[b]["deleted"] and rows[b]["label"] == b
    assert rows[orphan]["deleted"] and rows[orphan]["bytes"] == 8
    assert not (base / orphan / "state").exists()
    for key in (a, b):
        assert client.get("/qwen-exo/knowledge/files", params=scope(key)).json()["total"] == 1
    assert client.get("/qwen-exo/knowledge/read", params={**scope(orphan), "path": "note.txt"}).json()["text"] == "retained"


def test_owner_listing_includes_every_nonshared_mode_without_creating_storage(admin):
    client, shared, store, (a, b) = admin
    created = {
        mode: store.create(mode, mode)["id"]
        for mode in ("shared", "private", "private_only", "super")
    }
    store.revoke(created["super"])
    rows = {
        row["key_id"]: row
        for row in client.get("/qwen-exo/knowledge/memory-spaces").json()["spaces"]
    }
    assert set(rows) == {a, b, created["private"], created["private_only"], created["super"]}
    for mode in ("private", "private_only", "super"):
        assert rows[created[mode]]["memory_scope"] == mode
        assert rows[created[mode]]["files"] == rows[created[mode]]["bytes"] == 0
    assert rows[created["super"]]["revoked"] is True
    assert not (shared.state_directory / "private-memory").exists()


def test_key_admin_http_accepts_all_modes_and_rejects_unknown_modes(admin):
    client, _shared, store, _keys = admin
    created = client.post("/qwen-exo/api-keys", json={"label": "Default mode"})
    assert created.status_code == 201, created.text
    assert created.json()["memory_scope"] == "shared"
    for mode in ("shared", "private", "private_only", "super"):
        response = client.post("/qwen-exo/api-keys", json={"label": mode, "memory_scope": mode})
        assert response.status_code == 201, response.text
        key = response.json()
        assert key["memory_scope"] == mode
        for updated_mode in ("super", "private_only", "private", "shared"):
            updated = client.patch(f"/qwen-exo/api-keys/{key['id']}", json={"memory_scope": updated_mode})
            assert updated.status_code == 200, updated.text
            assert updated.json()["memory_scope"] == updated_mode
            assert store.resolve_authorization(f"Bearer {key['token']}")["memory_scope"] == updated_mode
    original = store.path.read_bytes()
    for invalid in ("unknown", "SUPER", "private-only", "", None, [], {}):
        assert client.post("/qwen-exo/api-keys", json={"label": "invalid", "memory_scope": invalid}).status_code == 422
        assert client.patch(f"/qwen-exo/api-keys/{key['id']}", json={"memory_scope": invalid}).status_code == 422
    assert store.path.read_bytes() == original


def test_admin_scoped_browsing_mutations_and_public_tool_isolation(admin):
    client, shared, _store, (a, b) = admin
    for key, text in ((None, "marker shared"), (a, "marker ownerA"), (b, "marker ownerB")):
        upload(client, key, "work-memory/same.md", text)
        listing = client.get("/qwen-exo/knowledge/files", params={**scope(key), "area": "work"})
        assert listing.status_code == 200 and listing.json()["total"] == 1
        tree = client.get("/qwen-exo/knowledge/tree", params=scope(key)).json()
        assert tree["areas"]["work"]["files"] == 1
        assert "work-memory" in tree["directories"]
        read = client.get("/qwen-exo/knowledge/read", params={**scope(key), "path": "work-memory/same.md"})
        assert read.json()["text"] == text
        search = client.get("/qwen-exo/knowledge/search", params={**scope(key), "pattern": "marker"})
        assert search.status_code == 200 and text in search.text
        assert client.get("/qwen-exo/knowledge/index", params=scope(key)).json()["file_count"] == 1
        assert client.put("/qwen-exo/knowledge/summary", params=scope(key), json={"path": "work-memory/same.md", "summary": text + " summary"}).status_code == 200
        assert client.get("/qwen-exo/knowledge/summary", params={**scope(key), "path": "work-memory/same.md"}).json()["summary"] == text + " summary"
    for key, text in ((a, "marker ownerA edited"), (b, "marker ownerB edited")):
        upload(client, key, "work-memory/same.md", text)
        assert client.post("/qwen-exo/knowledge/folders", params=scope(key), json={"path": "work-memory/folder"}).status_code == 200
        moved = client.post("/qwen-exo/knowledge/move", params=scope(key), json={"source": "work-memory/same.md", "destination": "work-memory/folder/moved.md"})
        assert moved.status_code == 200
        assert client.get("/qwen-exo/knowledge/read", params={**scope(key), "path": "work-memory/folder/moved.md"}).json()["text"] == text
        assert client.delete("/qwen-exo/knowledge/summary", params={**scope(key), "path": "work-memory/folder/moved.md"}).status_code == 200
        assert client.delete("/qwen-exo/knowledge/files/work-memory/folder/moved.md", params=scope(key)).status_code == 200
        assert client.delete("/qwen-exo/knowledge/folders/work-memory/folder", params=scope(key)).status_code == 200
    assert shared.read("work-memory/same.md")["text"] == "marker shared"
    upload(client, a, "work-memory/private.md", "unique_owner_A")
    upload(client, b, "work-memory/private.md", "unique_owner_B")
    public = MemoryToolExecutor(shared)
    private_a = get_private_memory_index(shared.root, shared.state_directory, a)
    owner = MemoryToolExecutor(private_a, shared_index=shared, key_id=a, memory_scope="private")
    assert public.read_index.grep("unique_owner")["file_count"] == 0
    assert owner.read_index.grep("unique_owner_A")["file_count"] == 1
    assert owner.read_index.grep("unique_owner_B")["file_count"] == 0
    assert client.get("/qwen-exo/knowledge/files").json()["total"] == 1


def test_raw_and_archive_uploads_stay_in_selected_space(admin):
    client, shared, _store, (a, b) = admin
    assert client.post("/qwen-exo/knowledge/upload", params={**scope(a), "path": "work-memory/raw.md"}, content=b"raw owner A").status_code == 200
    assert client.post("/qwen-exo/knowledge/upload", params={**scope(a), "path": "work-memory/raw.md"}, content=b"edited owner A").status_code == 200
    assert client.get("/qwen-exo/knowledge/read", params={**scope(a), "path": "work-memory/raw.md"}).json()["text"] == "edited owner A"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("archived.md", "archive owner")
    for endpoint, key in (("archives", a), ("upload-archive", b)):
        if endpoint == "archives":
            response = client.post("/qwen-exo/knowledge/" + endpoint, params=scope(key), json={"content_base64": base64.b64encode(archive.getvalue()).decode(), "target_directory": "work-memory"})
        else:
            response = client.post("/qwen-exo/knowledge/" + endpoint, params={**scope(key), "target_directory": "work-memory"}, content=archive.getvalue())
        assert response.status_code == 200, response.text
        assert len(response.json()["files"]) == 1
    assert client.get("/qwen-exo/knowledge/files", params=scope(a)).json()["total"] == 2
    assert client.get("/qwen-exo/knowledge/files", params=scope(b)).json()["total"] == 1
    assert shared.metadata() == []
    assert not list(shared.state_directory.glob("*.upload"))


def test_private_trash_and_automation_are_scope_safe(admin):
    client, shared, _store, (a, b) = admin
    for key, text in ((None, "shared"), (a, "private A"), (b, "private B")):
        index = get_private_memory_index(shared.root, shared.state_directory, key) if key else shared
        index.upsert("work-memory/deleted.md", text.encode())
        WorkMemoryTrash(index).discard("deleted.md", batch="same-batch")
        rows = client.get("/qwen-exo/memory/work-trash", params=scope(key)).json()["entries"]
        assert rows == [{"batch": "same-batch", "path": "work-memory/deleted.md", "bytes": len(text)}]
        response = client.post("/qwen-exo/memory/work-trash/restore", params=scope(key), json={"batch": "same-batch", "path": "work-memory/deleted.md"})
        assert response.status_code == 200 and index.read("work-memory/deleted.md")["text"] == text
    for endpoint in ("knowledge/summaries", "memory/dream"):
        assert client.get("/qwen-exo/" + endpoint, params=scope(a)).json()["available"] is False
        assert client.get("/qwen-exo/" + endpoint).json()["shared"] is True
    assert client.post("/qwen-exo/knowledge/summaries/generate", params=scope(a), json={}).status_code == 409
    assert client.post("/qwen-exo/memory/dream/run", params=scope(a)).status_code == 409
    assert client.get("/qwen-exo/memory/dream/runs/1", params=scope(a)).status_code == 409
    assert client.post("/qwen-exo/memory/work-trash/restore", params=scope(a), json={"batch": "..", "path": "x.md"}).status_code == 400


@pytest.mark.parametrize("key,status", [("", 400), ("../escape", 400), ("a/b", 400), ("qxk_unknown-owner", 404)])
def test_invalid_and_unknown_scope_never_creates_storage(admin, key, status):
    client, shared, _store, _keys = admin
    for endpoint in ("knowledge/files", "knowledge/tree", "knowledge/index", "knowledge/summaries", "memory/dream", "memory/work-trash"):
        response = client.get("/qwen-exo/" + endpoint, params=scope(key) if key else {"memory_key_id": ""})
        assert response.status_code == status, response.text
    response = client.post("/qwen-exo/knowledge/files", params={"memory_key_id": key}, json={"relative_path": "work-memory/x.md", "content_base64": "eA=="})
    assert response.status_code == status
    assert not (shared.state_directory / "private-memory").exists()
    assert shared.metadata() == []


def test_duplicate_scope_and_symlink_owner_are_rejected(admin, tmp_path):
    client, shared, _store, (a, b) = admin
    duplicate = client.get("/qwen-exo/knowledge/files", params=[("memory_key_id", a), ("memory_key_id", b)])
    assert duplicate.status_code == 400
    base = shared.state_directory / "private-memory"
    base.mkdir()
    try:
        (base / a).symlink_to(shared.root, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    spaces = client.get("/qwen-exo/knowledge/memory-spaces").json()["spaces"]
    assert {row["key_id"] for row in spaces} == {b}
    assert client.get("/qwen-exo/knowledge/files", params=scope(a)).status_code == 400
    assert not (shared.root / "state").exists()


def test_transfer_http_uses_explicit_owners_not_browsing_scope(admin):
    client, shared, _store, (a, b) = admin
    upload(client, a, "note.md", "owner A data")
    upload(client, b, "note.md", "owner B unchanged")
    payload = {"source_key_id": a, "destination_key_id": None,
               "source": "note.md", "destination": "work-memory/published.md"}
    refused = client.post("/qwen-exo/knowledge/transfer", params=scope(b), json=payload)
    assert refused.status_code == 409
    assert not shared.path_of("work-memory/published.md").exists()
    assert client.app.state.qwen_exo_runtime.transfer_events == []
    payload["confirm_publish"] = True
    published = client.post("/qwen-exo/knowledge/transfer", params=scope(b), json=payload)
    assert published.status_code == 200, published.text
    assert published.json() == {"from": "note.md", "to": "work-memory/published.md", "files": 1,
                                "source_key_id": a, "destination_key_id": None}
    assert shared.read_text("work-memory/published.md")[1] == "owner A data"
    private_b = get_private_memory_index(shared.root, shared.state_directory, b)
    assert private_b.read_text("note.md")[1] == "owner B unchanged"
    conflict = client.post("/qwen-exo/knowledge/transfer", json={
        "source_key_id": None, "destination_key_id": b,
        "source": "work-memory/published.md", "destination": "note.md",
    })
    assert conflict.status_code == 409
    moved = client.post("/qwen-exo/knowledge/transfer", json={
        "source_key_id": None, "destination_key_id": b,
        "source": "work-memory/published.md", "destination": "imported.md",
    })
    assert moved.status_code == 200, moved.text
    assert private_b.read_text("imported.md")[1] == "owner A data"
    assert shared.summary("work-memory/published.md") is None
    assert not shared.path_of("work-memory/published.md").exists()
    events = client.app.state.qwen_exo_runtime.transfer_events
    assert [phase for phase, _values in events] == ["before", "after"] * 3


@pytest.mark.parametrize("changes,status", [
    ({"source": "../escape"}, 400),
    ({"destination": "../escape"}, 400),
    ({"destination": ""}, 422),
    ({"source_key_id": "qxk_unknown-owner"}, 404),
    ({"source_key_id": "../escape"}, 400),
    ({"source_key_id": None, "source": ""}, 400),
    ({"confirm_publish": "true"}, 422),
    ({"overwrite": True}, 422),
])
def test_transfer_http_rejects_unsafe_or_ambiguous_requests(admin, changes, status):
    client, shared, _store, (a, b) = admin
    upload(client, a, "note.md", "unchanged")
    response = client.post("/qwen-exo/knowledge/transfer", json={
        "source_key_id": a, "destination_key_id": b,
        "source": "note.md", "destination": "moved.md", **changes,
    })
    assert response.status_code == status, response.text
    assert client.app.state.qwen_exo_runtime.transfer_events == []
    assert not (shared.state_directory / "private-memory" / "qxk_unknown-owner").exists()
    private = get_private_memory_index(shared.root, shared.state_directory, a)
    assert private.read_text("note.md")[1] == "unchanged"


@pytest.mark.parametrize("field", ["source_key_id", "destination_key_id"])
def test_transfer_unknown_owner_never_enters_creating_runtime_guard(admin, field):
    client, shared, _store, (a, b) = admin
    unknown = "qxk_unknown-owner"
    upload(client, a, "note.md", "unchanged")
    runtime = client.app.state.qwen_exo_runtime

    @asynccontextmanager
    async def creating_guard(values):
        runtime.transfer_events.append(("entered", values))
        get_private_memory_index(shared.root, shared.state_directory, unknown)
        yield

    runtime.memory_transfer_guard = creating_guard
    response = client.post("/qwen-exo/knowledge/transfer", json={
        "source_key_id": a, "destination_key_id": b,
        "source": "note.md", "destination": "moved.md", field: unknown,
    })
    assert response.status_code == 404, response.text
    assert runtime.transfer_events == []
    assert not (shared.state_directory / "private-memory" / unknown).exists()
