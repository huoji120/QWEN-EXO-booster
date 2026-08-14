from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_API_KEY_SCHEMA = 1
_TOKEN_PREFIX = "qxk"
_PROTECTED_ENDPOINTS = frozenset(
    {
        "/v1/models",
        "/v1/responses",
        "/v1/responses/compact",
    }
)


class ApiKeyStoreError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def public_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _revision(records: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        records, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_bearer(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    scheme, separator, token = authorization_header.partition(" ")
    if not separator or scheme.casefold() != "bearer" or not token:
        return None
    return token


def _validate_label(value: object) -> str:
    label = str(value or "").strip()
    if not 1 <= len(label) <= 80:
        raise ApiKeyStoreError("invalid_label", "密钥名称长度必须为 1 到 80 个字符")
    if any(ord(character) < 32 for character in label):
        raise ApiKeyStoreError("invalid_label", "密钥名称不能包含控制字符")
    return label


class ApiKeyStore:
    """Atomic, hashed bearer-key registry shared by QWEN-EXO and SGLang auth."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.RLock()
        self._cached_signature: tuple[int, int] | None = None
        self._cached_records: tuple[dict[str, Any], ...] = ()

    def _empty_document(self) -> dict[str, Any]:
        return {
            "schema": _API_KEY_SCHEMA,
            "revision": _revision([]),
            "updated_at": None,
            "keys": [],
        }

    def _read_document(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return self._empty_document()
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiKeyStoreError(
                "key_store_unreadable", f"无法读取 API 密钥存储：{exc}"
            ) from exc
        if payload.get("schema") != _API_KEY_SCHEMA or not isinstance(
            payload.get("keys"), list
        ):
            raise ApiKeyStoreError(
                "key_store_schema_mismatch", "API 密钥存储 schema 不受支持"
            )
        records = payload["keys"]
        ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ApiKeyStoreError("key_store_invalid", "API 密钥存储包含无效记录")
            key_id = str(record.get("id") or "")
            digest = str(record.get("digest") or "")
            if (
                not key_id.startswith(f"{_TOKEN_PREFIX}_")
                or len(digest) != 64
                or key_id in ids
            ):
                raise ApiKeyStoreError(
                    "key_store_invalid", "API 密钥存储包含无效或重复记录"
                )
            ids.add(key_id)
        expected_revision = _revision(records)
        if payload.get("revision") != expected_revision:
            raise ApiKeyStoreError(
                "key_store_revision_mismatch", "API 密钥存储 revision 校验失败"
            )
        return payload

    def _write_document(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        document = {
            "schema": _API_KEY_SCHEMA,
            "revision": _revision(records),
            "updated_at": _utc_now(),
            "keys": records,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(handle, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    document, stream, ensure_ascii=False, sort_keys=True, indent=2
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            temporary.unlink(missing_ok=True)
        self._cached_signature = None
        self._cached_records = ()
        return document

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: record.get(key) for key in ("id", "label", "created_at", "revoked_at")
        }

    def listing(self) -> dict[str, Any]:
        with self._lock:
            document = self._read_document()
            return {
                "schema": document["schema"],
                "revision": document["revision"],
                "updated_at": document.get("updated_at"),
                "keys": [self._public_record(record) for record in document["keys"]],
            }

    def create(self, label: object) -> dict[str, Any]:
        normalized_label = _validate_label(label)
        with self._lock:
            document = self._read_document()
            records = list(document["keys"])
            key_id = f"{_TOKEN_PREFIX}_{secrets.token_hex(8)}"
            token = f"{key_id}_{secrets.token_urlsafe(32)}"
            record = {
                "id": key_id,
                "label": normalized_label,
                "digest": _token_digest(token),
                "created_at": _utc_now(),
                "revoked_at": None,
            }
            records.append(record)
            updated = self._write_document(records)
            return {
                **self._public_record(record),
                "token": token,
                "revision": updated["revision"],
            }

    def revoke(self, key_id: str) -> dict[str, Any]:
        with self._lock:
            document = self._read_document()
            records = list(document["keys"])
            matched = None
            for record in records:
                if secrets.compare_digest(str(record.get("id") or ""), key_id):
                    matched = record
                    break
            if matched is None:
                raise ApiKeyStoreError("key_not_found", "API 密钥不存在")
            if matched.get("revoked_at") is None:
                matched["revoked_at"] = _utc_now()
                updated = self._write_document(records)
            else:
                updated = document
            return {
                **self._public_record(matched),
                "revision": updated["revision"],
            }

    def _active_records(self) -> tuple[dict[str, Any], ...]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._cached_signature = None
            self._cached_records = ()
            return ()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature != self._cached_signature:
            document = self._read_document()
            self._cached_records = tuple(
                record
                for record in document["keys"]
                if record.get("revoked_at") is None
            )
            self._cached_signature = signature
        return self._cached_records

    def validate_token(self, token: str) -> bool:
        supplied_digest = _token_digest(token)
        with self._lock:
            return any(
                secrets.compare_digest(supplied_digest, str(record["digest"]))
                for record in self._active_records()
            )

    def authorize_request(
        self, method: str, path: str, authorization_header: str | None
    ) -> bool | None:
        if path not in _PROTECTED_ENDPOINTS:
            return None
        if method.upper() == "OPTIONS":
            return True
        token = _parse_bearer(authorization_header)
        return token is not None and self.validate_token(token)
