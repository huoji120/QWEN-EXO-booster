"""Read-only views of retained source events, not generated reflection lessons."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from qwen_exo_booster.attention_diagnostic import (
    MAX_MESSAGES,
    MAX_UPLOAD_BYTES,
    AttentionDiagnosticError,
)

_ROLES = {
    "system_context": "system",
    "developer_context": "developer",
    "user_context": "user",
    "assistant_trajectory": "assistant",
    "tool_action": "assistant",
    "tool_observation": "tool",
}
_PARTIAL_WARNING = (
    "Partial retained source history, not an exact request replay: capture may omit "
    "instructions, tool schemas, multimodal content and empty events; assistant "
    "reasoning/output can be combined and some source text was bounded before storage. "
    "No missing messages are fabricated. Import does not invoke the model or tools."
)


def _identity(source: str, key: str) -> str:
    return hashlib.sha256((source + "\0" + key).encode("utf-8")).hexdigest()


@contextmanager
def _database(store: Any):
    # Do not use store._connect(): its WAL/schema setup can mutate persistence.
    uri = Path(store.path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        yield db


def _check_size(size: int) -> None:
    if size > MAX_UPLOAD_BYTES:
        raise AttentionDiagnosticError(
            "Retained conversation exceeds 2MiB; nothing was truncated", 413
        )


def _messages(rows: Any, warnings: list[str]) -> list[dict[str, Any]]:
    messages = []
    omitted = 0
    for row in rows:
        kind = str(row.get("kind") or "")
        role = _ROLES.get(kind)
        if role is None:
            omitted += 1
            continue
        content = row.get("content", "")
        if not isinstance(content, str):
            raise AttentionDiagnosticError("Retained source has non-text event content")
        item: dict[str, Any] = {"role": role, "content": content}
        if kind == "tool_action":
            # This is an inert projection, not a new executable function call.
            item["content"] = (
                "<tool_call>"
                + json.dumps(
                    {
                        "name": row.get("tool_name", ""),
                        "call_id": row.get("call_id", ""),
                        "arguments": content,
                    },
                    ensure_ascii=False,
                )
                + "</tool_call>"
            )
        if row.get("tool_name"):
            item["name"] = str(row["tool_name"])
        if row.get("call_id"):
            item["tool_call_id"] = str(row["call_id"])
        messages.append(item)
    if omitted:
        warnings.append(
            f"{omitted} retained events have unknown roles and were not invented as messages."
        )
    if len(messages) > MAX_MESSAGES:
        raise AttentionDiagnosticError(
            f"Retained history exceeds {MAX_MESSAGES} messages; nothing was truncated",
            413,
        )
    return messages


class AttentionDiagnosticConversations:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    def _sources(self) -> list[dict[str, Any]]:
        sources = []
        seen = set()
        journal = getattr(self.runtime, "reflection_evidence_store", None)
        if journal is not None:
            with _database(journal) as db:
                rows = db.execute(
                    "SELECT conversation_key,last_seen,event_count FROM conversations "
                    "WHERE event_count>0 ORDER BY last_seen DESC,conversation_key"
                ).fetchall()
            for row in rows:
                key = row["conversation_key"]
                seen.add(key)
                sources.append(
                    self._summary(
                        "server_journal", key, key, row["last_seen"], row["event_count"]
                    )
                )
        snapshots = getattr(self.runtime, "reflection_source_store", None)
        if snapshots is not None:
            # Metadata only: never deserialize every archived multi-megabyte payload.
            with _database(snapshots) as db:
                rows = db.execute(
                    "SELECT source_digest,conversation_key,captured_at,trajectory_row_count "
                    "FROM reflection_sources ORDER BY captured_at DESC,source_digest"
                ).fetchall()
            for row in rows:
                key = row["conversation_key"]
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    self._summary(
                        "server_snapshot",
                        row["source_digest"],
                        key,
                        row["captured_at"],
                        row["trajectory_row_count"],
                    )
                )
        retained = getattr(self.runtime, "_reflection_memory_trajectories", {})
        for key, rows in list(retained.items()):
            if key in seen or not rows:
                continue
            # The in-memory store has monotonic activity, not a wall-clock timestamp.
            sources.append(self._summary("server_memory", key, key, 0, len(rows)))
        return sorted(sources, key=lambda row: (-row["updated_at"], row["id"]))

    @staticmethod
    def _summary(
        source: str, key: str, title: str, updated_at: float, count: int
    ) -> dict[str, Any]:
        return {
            "id": _identity(source, key),
            "title": title,
            "updated_at": float(updated_at),
            "message_count": int(count),
            "source": source,
            "partial": True,
            "_key": key,
        }

    def list(self, limit: int = 25, offset: int = 0) -> dict[str, Any]:
        sources = self._sources()
        return {
            "conversations": [
                {k: v for k, v in row.items() if k != "_key"}
                for row in sources[offset : offset + limit]
            ],
            "total": len(sources),
            "limit": limit,
            "offset": offset,
        }

    def _journal_rows(
        self, key: str, warnings: list[str], event_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        store = getattr(self.runtime, "reflection_evidence_store", None)
        if store is None:
            warnings.append(
                "The event journal is unavailable; referenced events cannot be recovered."
            )
            return []
        with _database(store) as db:
            meta = db.execute(
                "SELECT raw_bytes,event_count FROM conversations WHERE conversation_key=?",
                (key,),
            ).fetchone()
            if meta is None or not meta["event_count"]:
                warnings.append("No events remain in this conversation's journal.")
                return []
            _check_size(meta["raw_bytes"])
            gap = db.execute(
                "SELECT purged_events FROM retention_gaps WHERE conversation_key=?",
                (key,),
            ).fetchone()
            if gap and gap[0]:
                warnings.append(
                    f"Journal retention purged {gap[0]} events from this conversation."
                )
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT kind,content,tool_name,call_id,event_id FROM events WHERE conversation_key=? ORDER BY ordinal",
                    (key,),
                )
            ]
        if event_ids is not None:
            expected = set(event_ids)
            rows = [row for row in rows if row["event_id"] in expected]
            missing = len(expected - {row["event_id"] for row in rows})
            if missing:
                warnings.append(
                    f"{missing} referenced journal events are missing; only retained events are shown."
                )
        return rows

    def detail(self, conversation_id: str) -> dict[str, Any] | None:
        source = next(
            (row for row in self._sources() if row["id"] == conversation_id), None
        )
        if source is None:
            return None
        warnings = [_PARTIAL_WARNING]
        key = source["_key"]
        if source["source"] == "server_journal":
            rows = self._journal_rows(key, warnings)
        elif source["source"] == "server_snapshot":
            with _database(self.runtime.reflection_source_store) as db:
                size = db.execute(
                    "SELECT length(CAST(payload_json AS BLOB)) FROM reflection_sources WHERE source_digest=?",
                    (key,),
                ).fetchone()
                if size is None:
                    return None
                _check_size(size[0])
                payload = json.loads(
                    db.execute(
                        "SELECT payload_json FROM reflection_sources WHERE source_digest=?",
                        (key,),
                    ).fetchone()[0]
                )
            audit = payload.get("source_audit") or {}
            if audit.get("capture") == "journal_references":
                rows = self._journal_rows(
                    payload["conversation_key"],
                    warnings,
                    audit.get("raw_event_ids") or [],
                )
            else:
                rows = payload.get("trajectory_history") or []
                warnings.append(
                    "Archived source snapshot only; later events and cropped source text cannot be recovered here."
                )
            # original_task is a separately derived summary, not an ordered source message.
            # capsule_history/verifier_feedback/generated lessons are never injected.
        else:
            rows = tuple(
                getattr(self.runtime, "_reflection_memory_trajectories", {}).get(
                    key, ()
                )
            )
            _check_size(
                sum(len(str(row.get("content", "")).encode("utf-8")) for row in rows)
            )
            warnings.append(
                "In-memory history is capped and text-bounded; its update timestamp is unavailable."
            )
        messages = _messages(rows, warnings)
        if not messages:
            raise AttentionDiagnosticError(
                "No importable role events remain in this retained source", 404
            )
        content = json.dumps(
            {"messages": messages}, ensure_ascii=False, separators=(",", ":")
        )
        _check_size(len(content.encode("utf-8")))
        return {
            "content": content,
            "filename": f"conversation-{conversation_id}.json",
            "warnings": warnings,
        }
