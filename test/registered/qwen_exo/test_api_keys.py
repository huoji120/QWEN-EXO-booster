from __future__ import annotations

import json
import os

import pytest

from qwen_exo_booster.api_keys import ApiKeyStore, ApiKeyStoreError


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
