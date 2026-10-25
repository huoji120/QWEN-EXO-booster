from __future__ import annotations

import hashlib
import json
import os

import pytest

from qwen_exo_booster.api_keys import (
    ApiKeyStore,
    ApiKeyStoreError,
    is_valid_api_key_id,
)


def test_issue_validate_revoke_without_storing_plaintext(tmp_path):
    path = tmp_path / "api-keys.json"
    store = ApiKeyStore(path)

    created = store.create("OpenCode workstation")
    token = created["token"]

    assert token.startswith(created["id"] + "_")
    assert store.authorize_request("POST", "/v1/responses", f"Bearer {token}")
    assert store.authorize_request("GET", "/v1/models", None) is False
    assert store.authorize_request("GET", "/qwen-exo/status", None) is None
    assert store.authorize_request("GET", "/qwen-exo/console/v1/models", None) is None
    assert (
        store.authorize_request("POST", "/qwen-exo/console/v1/responses", None)
        is None
    )
    assert store.authorize_request("OPTIONS", "/v1/responses", None) is True

    persisted = path.read_text(encoding="utf-8")
    assert token not in persisted
    assert json.loads(persisted)["keys"][0]["digest"]
    if os.name == "posix":
        assert os.stat(path).st_mode & 0o777 == 0o600

    revoked = store.revoke(created["id"])
    assert revoked["revoked_at"] is not None
    assert not store.authorize_request(
        "POST", "/v1/responses/compact", f"Bearer {token}"
    )


def test_multiple_keys_and_idempotent_revoke(tmp_path):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    first = store.create("first")
    second = store.create("second")

    assert store.validate_token(first["token"])
    assert store.validate_token(second["token"])
    store.revoke(first["id"])
    store.revoke(first["id"])

    assert not store.validate_token(first["token"])
    assert store.validate_token(second["token"])
    listing = store.listing()
    assert len(listing["keys"]) == 2
    assert all("token" not in record and "digest" not in record for record in listing["keys"])


def test_delete_removes_records_and_access_in_one_batch(tmp_path):
    """Deletion drops the record and the token together for every listed id.

    ``revoke`` keeps records forever, so the registry only grew. Deleting an
    active key must not leave a valid digest behind, unknown ids are reported
    rather than failing the whole batch, and an all-unknown batch is a 404.
    """
    store = ApiKeyStore(tmp_path / "api-keys.json")
    active = store.create("active")
    stale = store.create("stale")
    keep = store.create("keep")
    store.revoke(stale["id"])

    result = store.delete([active["id"], stale["id"], "qxk_missing", active["id"]])

    assert [record["id"] for record in result["deleted"]] == [
        active["id"],
        stale["id"],
    ]
    assert result["missing"] == ["qxk_missing"]
    assert not store.validate_token(active["token"])
    assert store.validate_token(keep["token"])
    assert [record["id"] for record in store.listing()["keys"]] == [keep["id"]]
    with pytest.raises(ApiKeyStoreError) as excinfo:
        store.delete(["qxk_missing"])
    assert excinfo.value.code == "key_not_found"


def test_corrupt_registry_fails_closed(tmp_path):
    path = tmp_path / "api-keys.json"
    path.write_text('{"schema":1,"revision":"wrong","keys":[]}', encoding="utf-8")
    store = ApiKeyStore(path)

    with pytest.raises(ApiKeyStoreError, match="revision"):
        store.listing()
    with pytest.raises(ApiKeyStoreError, match="revision"):
        store.authorize_request("GET", "/v1/models", "Bearer anything")


def test_invalid_labels_and_missing_key_fail(tmp_path):
    store = ApiKeyStore(tmp_path / "api-keys.json")

    with pytest.raises(ApiKeyStoreError, match="1 到 80"):
        store.create(" ")
    with pytest.raises(ApiKeyStoreError, match="不存在"):
        store.revoke("qxk_missing")


def _replace_registry_records(path, records):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["keys"] = records
    encoded = json.dumps(
        records, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["revision"] = hashlib.sha256(encoded).hexdigest()[:16]
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_memory_scope_defaults_and_all_mode_resolution(tmp_path):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    shared = store.create("shared")
    scopes = ("shared", "private", "private_only", "super")
    keys = [shared, *(store.create(scope, scope) for scope in scopes[1:])]

    assert shared["memory_scope"] == "shared"
    for key in keys:
        resolved = store.resolve_authorization(f"Bearer {key['token']}")
        assert resolved == {
            field: key[field]
            for field in ("id", "label", "created_at", "revoked_at", "memory_scope")
        }
        assert "digest" not in resolved and "token" not in resolved
        resolved["memory_scope"] = "tampered"
        assert store.resolve_authorization(f"Bearer {key['token']}")[
            "memory_scope"
        ] == key["memory_scope"]
    assert [record["memory_scope"] for record in store.listing()["keys"]] == list(scopes)


def test_legacy_registry_scope_defaults_without_changing_revision(tmp_path):
    path = tmp_path / "api-keys.json"
    store = ApiKeyStore(path)
    key = store.create("legacy")
    records = json.loads(path.read_text(encoding="utf-8"))["keys"]
    del records[0]["memory_scope"]
    records[0]["id"] = "qxk_legacy"
    _replace_registry_records(path, records)
    original = path.read_bytes()

    assert store.listing()["keys"][0]["memory_scope"] == "shared"
    assert store.resolve_authorization(f"Bearer {key['token']}")[
        "memory_scope"
    ] == "shared"
    assert path.read_bytes() == original
    for memory_scope in ("private", "private_only", "super", "shared", "private"):
        store.update_memory_scope("qxk_legacy", memory_scope)
        resolved = store.resolve_authorization(f"Bearer {key['token']}")
        assert resolved["id"] == "qxk_legacy"
        assert resolved["memory_scope"] == memory_scope
        record = json.loads(path.read_text(encoding="utf-8"))["keys"][0]
        assert record["digest"] == records[0]["digest"]
    from qwen_exo_booster.file_knowledge import FileKnowledgeIndex
    from qwen_exo_booster.scoped_memory import get_private_memory_index

    shared = FileKnowledgeIndex(tmp_path / "knowledge", tmp_path / "state")
    private = get_private_memory_index(shared.root, shared.state_directory, resolved["id"])
    private.upsert("work-memory/legacy.md", b"named owner private content")
    assert private.read_bytes("work-memory/legacy.md")["text"] == "named owner private content"
    assert shared.grep("named owner")["file_count"] == 0


def test_scope_updates_refresh_cached_identity_and_do_not_reissue_tokens(tmp_path):
    path = tmp_path / "api-keys.json"
    writer = ApiKeyStore(path)
    reader = ApiKeyStore(path)
    key = writer.create("stable identity")
    header = f"Bearer {key['token']}"
    original_digest = json.loads(path.read_text(encoding="utf-8"))["keys"][0]["digest"]
    assert reader.resolve_authorization(header)["memory_scope"] == "shared"

    for scope in ("private", "private_only", "super", "shared", "super", "private_only", "private"):
        updated = writer.update_memory_scope(key["id"], scope)
        assert updated["id"] == key["id"]
        assert updated["created_at"] == key["created_at"]
        assert updated["label"] == key["label"]
        assert "token" not in updated and "digest" not in updated
        assert writer.resolve_authorization(header)["memory_scope"] == scope
        assert reader.resolve_authorization(header)["memory_scope"] == scope
        assert reader.authorize_request("POST", "/v1/responses", header)
        record = json.loads(path.read_text(encoding="utf-8"))["keys"][0]
        assert record["digest"] == original_digest

    writer.revoke(key["id"])
    assert reader.resolve_authorization(header) is None
    writer.update_memory_scope(key["id"], "shared")
    assert reader.resolve_authorization(header) is None
    writer.delete([key["id"]])
    assert reader.resolve_authorization(header) is None


@pytest.mark.parametrize("scope", [None, "", "PRIVATE", "private-only", "SUPER", "other", 1, [], {}])
def test_invalid_memory_scope_cannot_create_or_update(tmp_path, scope):
    path = tmp_path / "api-keys.json"
    store = ApiKeyStore(path)
    key = store.create("valid")
    original = path.read_bytes()
    for operation in (
        lambda: store.create("invalid", scope),
        lambda: store.update_memory_scope(key["id"], scope),
    ):
        with pytest.raises(ApiKeyStoreError) as excinfo:
            operation()
        assert excinfo.value.code == "invalid_memory_scope"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "key_id",
    ["qxk_../escape", "qxk_0000000000000000/..", "qxk_0000000000000000\\..",
     "qxk_0000000000000000\n", "qxk_000000000000000G", "qxk_", None],
)
def test_unsafe_key_ids_fail_closed(tmp_path, key_id):
    path = tmp_path / "api-keys.json"
    store = ApiKeyStore(path)
    key = store.create("valid")
    assert is_valid_api_key_id(key["id"])
    assert not is_valid_api_key_id(key_id)
    with pytest.raises(ApiKeyStoreError) as excinfo:
        store.update_memory_scope(key_id, "private")
    assert excinfo.value.code == "invalid_key_id"
    records = json.loads(path.read_text(encoding="utf-8"))["keys"]
    records[0]["id"] = key_id
    _replace_registry_records(path, records)
    with pytest.raises(ApiKeyStoreError) as excinfo:
        store.resolve_authorization(f"Bearer {key['token']}")
    assert excinfo.value.code == "key_store_invalid"


@pytest.mark.parametrize("scope", ["unknown", "SUPER", "private-only", None, [], {}])
def test_unknown_persisted_scope_fails_closed(tmp_path, scope):
    path = tmp_path / "api-keys.json"
    store = ApiKeyStore(path)
    key = store.create("valid")
    records = json.loads(path.read_text(encoding="utf-8"))["keys"]
    records[0]["memory_scope"] = scope
    _replace_registry_records(path, records)
    with pytest.raises(ApiKeyStoreError) as excinfo:
        store.resolve_authorization(f"Bearer {key['token']}")
    assert excinfo.value.code == "invalid_memory_scope"


@pytest.mark.parametrize("header", [None, "", "Basic invalid", "Bearer", "Bearer ", "Bearer invalid"])
def test_resolve_authorization_rejects_invalid_credentials(tmp_path, header):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    store.create("valid", "private")
    assert store.resolve_authorization(header) is None


def test_response_record_routes_require_managed_authorization(tmp_path):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    key = store.create("private", "private")
    header = f"Bearer {key['token']}"
    for method, path in (
        ("GET", "/v1/responses/resp_example"),
        ("DELETE", "/v1/responses/resp_example"),
        ("POST", "/v1/responses/resp_example/cancel"),
    ):
        assert store.authorize_request(method, path, None) is False
        assert store.authorize_request(method, path, header) is True
        assert store.authorize_request("OPTIONS", path, None) is True
    for path in (
        "/v1/responses/resp_example/other", "/v1/responses/resp_example/cancel/extra",
        "/qwen-exo/console/v1/responses/resp_example",
    ):
        assert store.authorize_request("GET", path, None) is None


def test_update_missing_key_does_not_create_record(tmp_path):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    with pytest.raises(ApiKeyStoreError) as excinfo:
        store.update_memory_scope("qxk_0000000000000000", "private")
    assert excinfo.value.code == "key_not_found"
    assert store.listing()["keys"] == []


@pytest.mark.parametrize("memory_scope", ["shared", "private", "private_only", "super"])
def test_alternate_inference_routes_require_managed_keys_without_changing_admin_routes(tmp_path, memory_scope):
    store = ApiKeyStore(tmp_path / "api-keys.json")
    key = store.create("inference", memory_scope)
    header = f"Bearer {key['token']}"
    paths = (
        "/v1/chat/completions", "/v1/completions",
        "/v1/messages", "/v1/messages/count_tokens",
    )
    for path in paths:
        assert store.authorize_request("POST", path, None) is False
        assert store.authorize_request("POST", path, "Bearer invalid") is False
        assert store.authorize_request("POST", path, header) is True
        assert store.authorize_request("OPTIONS", path, None) is True
    assert store.resolve_authorization(header)["memory_scope"] == memory_scope
    # This gate does not authorize admin APIs or alter their independent policy.
    for path in ("/qwen-exo/api-keys", "/qwen-exo/knowledge/memory-spaces", "/qwen-exo/status"):
        assert store.authorize_request("GET", path, None) is None
        assert store.authorize_request("GET", path, header) is None
    store.revoke(key["id"])
    for path in paths:
        assert store.authorize_request("POST", path, header) is False
