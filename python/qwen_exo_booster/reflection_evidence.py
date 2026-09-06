"""Private durable evidence journal and causal-entry merge helpers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _json(value)


def normalize_rows(
    conversation_key: str, rows: Iterable[Mapping[str, Any]]
) -> tuple[dict, ...]:
    out = []
    for row in rows:
        r = dict(row)
        item = {
            "conversation_key": str(conversation_key),
            "kind": str(r.get("kind", "")),
            "request_id": str(r.get("request_id", "")),
            "tool_name": str(r.get("tool_name", "")),
            "call_id": str(r.get("call_id", "")),
            "content": _text(r.get("content", "")),
        }
        if r.get("event_id"):
            item["event_id"] = str(r["event_id"])
        out.append(item)
    return tuple(out)


class _Connection(sqlite3.Connection):
    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


class ReflectionEvidenceStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = 64 * 1024 * 1024,
        max_conversations: int = 256,
    ):
        self.path = str(path)
        self.max_bytes = max(1024, int(max_bytes))
        self.max_conversations = max(1, int(max_conversations))
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS events(
                event_id TEXT PRIMARY KEY, conversation_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL, kind TEXT NOT NULL, request_id TEXT NOT NULL,
                tool_name TEXT NOT NULL, call_id TEXT NOT NULL, content TEXT NOT NULL,
                content_sha256 TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
              CREATE INDEX IF NOT EXISTS events_conv ON events(conversation_key, ordinal);
              CREATE TABLE IF NOT EXISTS conversations(
                conversation_key TEXT PRIMARY KEY, last_seen REAL NOT NULL,
                raw_bytes INTEGER NOT NULL, event_count INTEGER NOT NULL,
                next_ordinal INTEGER NOT NULL);
              INSERT OR IGNORE INTO conversations
                SELECT conversation_key,MAX(CAST(strftime('%s',created_at) AS REAL)),
                       SUM(length(CAST(content AS BLOB))),COUNT(*),MAX(ordinal)+1
                FROM events GROUP BY conversation_key;
              CREATE TABLE IF NOT EXISTS analyses(
                segment_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
              CREATE TABLE IF NOT EXISTS retention(
                id INTEGER PRIMARY KEY CHECK(id=1), purged_events INTEGER NOT NULL DEFAULT 0,
                purged_bytes INTEGER NOT NULL DEFAULT 0, last_purge_reason TEXT NOT NULL DEFAULT '');
              CREATE TABLE IF NOT EXISTS retention_gaps(
                conversation_key TEXT PRIMARY KEY, purged_events INTEGER NOT NULL DEFAULT 0,
                first_missing_ordinal INTEGER, last_missing_ordinal INTEGER);
              INSERT OR IGNORE INTO retention(id) VALUES(1);
            """)

    @staticmethod
    def _id(r: Mapping[str, Any], occurrence: int = 0) -> str:
        if r.get("event_id"):
            return str(r["event_id"])
        identity = {
            k: r.get(k, "")
            for k in ("conversation_key", "kind", "tool_name", "call_id", "content")
        }
        if not r.get("call_id"):
            identity["occurrence"] = occurrence
        return hashlib.sha256(_json(identity).encode()).hexdigest()

    def _connect(self):
        db = sqlite3.connect(
            self.path, timeout=30, isolation_level=None, factory=_Connection
        )
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    normalize_rows = staticmethod(normalize_rows)

    def append_rows(
        self,
        conversation_key: str,
        request_id: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        replay: bool = False,
    ) -> tuple[dict, ...]:
        normalized = normalize_rows(conversation_key, rows)
        if not normalized:
            return ()
        signatures = [
            (
                r["kind"],
                r["tool_name"],
                r["call_id"],
                hashlib.sha256(r["content"].encode()).hexdigest(),
            )
            for r in normalized
        ]
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT OR IGNORE INTO conversations VALUES(?,?,0,0,1)",
                (conversation_key, time.time()),
            )
            overlap_rows = []
            if replay:
                # Match a cumulative history only at its chronological boundary.
                # Matching equal text anywhere would erase distinct later actions.
                existing = db.execute(
                    "SELECT event_id,kind,tool_name,call_id,content_sha256 FROM events "
                    "WHERE conversation_key=? ORDER BY ordinal",
                    (conversation_key,),
                ).fetchall()
                prefix = [0] * len(signatures)
                matched = 0
                for index in range(1, len(signatures)):
                    while matched and signatures[index] != signatures[matched]:
                        matched = prefix[matched - 1]
                    if signatures[index] == signatures[matched]:
                        matched += 1
                    prefix[index] = matched
                matched = 0
                for row in existing:
                    signature = tuple(
                        row[key]
                        for key in ("kind", "tool_name", "call_id", "content_sha256")
                    )
                    while matched and (
                        matched == len(signatures) or signature != signatures[matched]
                    ):
                        matched = prefix[matched - 1]
                    if signature == signatures[matched]:
                        matched += 1
                overlap_rows = existing[-matched:] if matched else []
            ordinal = db.execute(
                "SELECT next_ordinal FROM conversations WHERE conversation_key=?",
                (conversation_key,),
            ).fetchone()[0]
            result, added_bytes, added_count = [], 0, 0
            for pos, row in enumerate(normalized):
                row["request_id"] = str(row["request_id"] or request_id)
                if row.get("event_id"):
                    eid = row["event_id"]
                elif pos < len(overlap_rows):
                    eid = overlap_rows[pos]["event_id"]
                elif row["call_id"]:
                    eid = self._id(row)
                else:
                    eid = hashlib.sha256(
                        _json(
                            (conversation_key, row["request_id"], pos, signatures[pos])
                        ).encode()
                    ).hexdigest()
                old = db.execute(
                    "SELECT * FROM events WHERE event_id=?", (eid,)
                ).fetchone()
                if old:
                    if any(
                        str(old[key]) != str(row[key])
                        for key in (
                            "conversation_key",
                            "kind",
                            "tool_name",
                            "call_id",
                            "content",
                        )
                    ):
                        raise ValueError("event ID collision")
                    result.append(dict(old))
                    continue
                db.execute(
                    "INSERT INTO events(event_id,conversation_key,ordinal,kind,request_id,tool_name,call_id,content,content_sha256) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        eid,
                        conversation_key,
                        ordinal,
                        row["kind"],
                        row["request_id"],
                        row["tool_name"],
                        row["call_id"],
                        row["content"],
                        signatures[pos][-1],
                    ),
                )
                result.append(
                    dict(
                        db.execute(
                            "SELECT * FROM events WHERE event_id=?", (eid,)
                        ).fetchone()
                    )
                )
                added_bytes += len(row["content"].encode())
                added_count += 1
                ordinal += 1
            db.execute(
                "UPDATE conversations SET last_seen=?,raw_bytes=raw_bytes+?,event_count=event_count+?,next_ordinal=? WHERE conversation_key=?",
                (time.time(), added_bytes, added_count, ordinal, conversation_key),
            )
            self._purge_locked(db, conversation_key)
            db.commit()
            return tuple(result)

    def rows(
        self, conversation_key: str, event_ids: Iterable[str] | None = None
    ) -> tuple[dict, ...]:
        with self._lock, self._connect() as db:
            if event_ids is None:
                q = db.execute(
                    "SELECT * FROM events WHERE conversation_key=? ORDER BY ordinal",
                    (conversation_key,),
                )
            else:
                ids = tuple(event_ids)
                if not ids:
                    return ()
                q = db.execute(
                    "SELECT * FROM events WHERE conversation_key=? AND event_id IN (%s) ORDER BY ordinal"
                    % ",".join("?" * len(ids)),
                    (conversation_key, *ids),
                )
            return tuple(dict(x) for x in q.fetchall())

    def get_event(self, event_id: str) -> dict | None:
        with self._lock, self._connect() as db:
            x = db.execute(
                "SELECT * FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            return dict(x) if x else None

    def get_analysis(self, segment_id: str) -> dict | None:
        with self._lock, self._connect() as db:
            x = db.execute(
                "SELECT payload FROM analyses WHERE segment_id=?", (segment_id,)
            ).fetchone()
            return json.loads(x[0]) if x else None

    def save_analysis(self, segment_id: str, payload: Mapping[str, Any]) -> None:
        body = _json(payload)
        encoded = body.encode()
        if len(encoded) > self.max_bytes:
            raise ValueError("analysis retention capacity exceeded")
        digest = hashlib.sha256(encoded).hexdigest()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT payload_sha256 FROM analyses WHERE segment_id=?", (segment_id,)
            ).fetchone()
            if old and old[0] != digest:
                raise ValueError("analysis segment ID collision")
            db.execute(
                "INSERT OR IGNORE INTO analyses(segment_id,payload,payload_sha256) VALUES(?,?,?)",
                (segment_id, body, digest),
            )
            total = db.execute(
                "SELECT COALESCE(SUM(length(CAST(payload AS BLOB))),0) FROM analyses"
            ).fetchone()[0]
            while total > self.max_bytes:
                oldest = db.execute(
                    "SELECT segment_id,length(CAST(payload AS BLOB)) FROM analyses "
                    "WHERE segment_id<>? ORDER BY rowid LIMIT 1",
                    (segment_id,),
                ).fetchone()
                db.execute("DELETE FROM analyses WHERE segment_id=?", (oldest[0],))
                total -= oldest[1]
            db.commit()

    def retention_metadata(self, conversation_key: str | None = None) -> dict:
        with self._lock, self._connect() as db:
            if conversation_key is None:
                x = db.execute(
                    "SELECT purged_events,purged_bytes,last_purge_reason FROM retention WHERE id=1"
                ).fetchone()
                return (
                    {
                        "purged_events": x[0],
                        "purged_bytes": x[1],
                        "last_purge_reason": x[2],
                    }
                    if x
                    else {}
                )
            x = db.execute(
                "SELECT purged_events,first_missing_ordinal,last_missing_ordinal FROM retention_gaps WHERE conversation_key=?",
                (conversation_key,),
            ).fetchone()
            return (
                {
                    "conversation_key": conversation_key,
                    "purged_events": x[0],
                    "first_missing_ordinal": x[1],
                    "last_missing_ordinal": x[2],
                    "gap": bool(x),
                }
                if x
                else {
                    "conversation_key": conversation_key,
                    "purged_events": 0,
                    "gap": False,
                }
            )

    def _purge_locked(self, db, protected_conversation):
        total, count = db.execute(
            "SELECT COALESCE(SUM(raw_bytes),0),COUNT(*) FROM conversations WHERE event_count>0"
        ).fetchone()
        purged = purged_bytes = 0
        while total > self.max_bytes or count > self.max_conversations:
            row = db.execute(
                "SELECT conversation_key,raw_bytes,event_count FROM conversations WHERE event_count>0 AND conversation_key<>? ORDER BY last_seen LIMIT 1",
                (protected_conversation,),
            ).fetchone()
            if row is None:
                raise ValueError("evidence retention capacity exceeded")
            key, size, events = row
            bounds = db.execute(
                "SELECT MIN(ordinal),MAX(ordinal) FROM events WHERE conversation_key=?",
                (key,),
            ).fetchone()
            db.execute("DELETE FROM events WHERE conversation_key=?", (key,))
            db.execute(
                "UPDATE conversations SET raw_bytes=0,event_count=0 WHERE conversation_key=?",
                (key,),
            )
            db.execute(
                "INSERT INTO retention_gaps(conversation_key,purged_events,first_missing_ordinal,last_missing_ordinal) VALUES(?,?,?,?) ON CONFLICT(conversation_key) DO UPDATE SET purged_events=purged_events+excluded.purged_events,first_missing_ordinal=COALESCE(retention_gaps.first_missing_ordinal,excluded.first_missing_ordinal),last_missing_ordinal=excluded.last_missing_ordinal",
                (key, events, bounds[0], bounds[1]),
            )
            total -= size
            count -= 1
            purged += events
            purged_bytes += size
        if purged:
            db.execute(
                "UPDATE retention SET purged_events=purged_events+?,purged_bytes=purged_bytes+?,last_purge_reason='whole_conversation_retention' WHERE id=1",
                (purged, purged_bytes),
            )


def merge_causal_entries(
    existing: Iterable[Mapping[str, Any]],
    changes: Iterable[Mapping[str, Any]],
    *,
    source_digest: str,
) -> tuple[dict, ...]:
    current = [dict(x) for x in existing]
    byid = {x.get("entry_id"): i for i, x in enumerate(current)}
    if len(byid) != len(current) or None in byid:
        raise ValueError("duplicate or missing existing entry ID")
    touched = set()
    for change in changes:
        op, eid, reason = (
            change.get("operation"),
            change.get("entry_id"),
            change.get("reason", ""),
        )
        if not reason:
            raise ValueError("reason required")
        if eid in touched:
            raise ValueError("duplicate target")
        if op == "add":
            entry = dict(change.get("entry") or {})
            if any(k in entry for k in ("entry_id", "version", "versions")):
                raise ValueError("forged add identity")
            eid = (
                eid
                or hashlib.sha256(
                    _json({"source_digest": source_digest, "entry": entry}).encode()
                ).hexdigest()[:24]
            )
            if eid in byid:
                raise ValueError("duplicate target")
            entry.update(entry_id=eid, version=1, versions=[])
            current.append(entry)
            byid[eid] = len(current) - 1
        elif op in ("revise", "retire"):
            if eid not in byid:
                raise ValueError("unknown target")
            old = current[byid[eid]]
            if change.get("expected_version") != old.get("version"):
                raise ValueError("stale revision")
            snap = dict(old)
            snap.pop("versions", None)
            new = dict(old) if op == "retire" else dict(change.get("entry") or {})
            if op == "revise" and any(
                k in new for k in ("entry_id", "version", "versions")
            ):
                raise ValueError("forged revision identity")
            new.update(
                entry_id=eid,
                version=int(old["version"]) + 1,
                versions=list(old.get("versions", ())) + [snap],
            )
            if op == "retire":
                new.update(admission_status="retired", reason=reason)
            current[byid[eid]] = new
        else:
            raise ValueError("invalid operation")
        touched.add(eid)
    return tuple(current)
