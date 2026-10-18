"""Full-text journal of server-executed memory tool calls for the console.

Telemetry redacts text when ``telemetry_text_mode`` is off, so the recall
trace keeps call metadata in telemetry and the complete arguments and outputs
here, in a bounded SQLite file under the state directory.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

MAX_LOG_ROWS = 20_000
MAX_LOG_BYTES = 256 * 1024 * 1024
# Retention is enforced every this many inserted calls, not on every write.
_PRUNE_EVERY = 64


class MemoryToolLog:
    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = MAX_LOG_ROWS,
        max_bytes: int = MAX_LOG_BYTES,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._since_prune = 0
        connection = self._connect()
        # sqlite3's context manager commits but does not close the connection.
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    call_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    output TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    created_at REAL NOT NULL,
                    size INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memory_tool_calls_request "
                "ON memory_tool_calls (request_id, id)"
            )
        connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def record(
        self, request_id: str, round_index: int, calls: Iterable[Mapping[str, Any]]
    ) -> None:
        now = time.time()
        rows = [
            (
                str(request_id),
                int(round_index),
                str(call["id"]),
                str(call["name"]),
                str(call.get("arguments") or ""),
                1 if call.get("ok") else 0,
                str(call.get("output") or ""),
                float(call.get("duration_ms") or 0.0),
                now,
                len(str(call.get("arguments") or "")) + len(str(call.get("output") or "")),
            )
            for call in calls
        ]
        if not rows:
            return
        connection = self._connect()
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO memory_tool_calls (request_id, round, call_id, name, "
                    "arguments, ok, output, duration_ms, created_at, size) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            self._since_prune += len(rows)
            if self._since_prune >= _PRUNE_EVERY:
                self._since_prune = 0
                self._prune(connection)
        finally:
            connection.close()

    def _prune(self, connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                "DELETE FROM memory_tool_calls WHERE id <= "
                "(SELECT id FROM memory_tool_calls ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (self._max_rows,),
            )
            # Drop the oldest rows until the remaining text fits the budget.
            total = connection.execute(
                "SELECT COALESCE(SUM(size), 0) FROM memory_tool_calls"
            ).fetchone()[0]
            if total <= self._max_bytes:
                return
            excess = total - self._max_bytes
            cutoff = connection.execute(
                "SELECT id FROM (SELECT id, SUM(size) OVER (ORDER BY id) AS running "
                "FROM memory_tool_calls) WHERE running >= ? ORDER BY id LIMIT 1",
                (excess,),
            ).fetchone()
            if cutoff is not None:
                connection.execute(
                    "DELETE FROM memory_tool_calls WHERE id <= ?", (cutoff[0],)
                )

    def for_request(self, request_id: str) -> list[dict[str, Any]]:
        """Every logged call of one request, oldest first, with full text."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM memory_tool_calls WHERE request_id = ? ORDER BY id",
                (str(request_id),),
            ).fetchall()
        finally:
            connection.close()
        return [self._row(row, full=True) for row in rows]

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Newest calls first, without output text."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM memory_tool_calls ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 1000)),),
            ).fetchall()
        finally:
            connection.close()
        return [self._row(row, full=False) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row, *, full: bool) -> dict[str, Any]:
        record = {
            "request_id": row["request_id"],
            "round": row["round"],
            "call_id": row["call_id"],
            "name": row["name"],
            "arguments": row["arguments"],
            "ok": bool(row["ok"]),
            "output_chars": len(row["output"]),
            "duration_ms": row["duration_ms"],
            "created_at": row["created_at"],
        }
        if full:
            record["output"] = row["output"]
        return record


__all__ = ["MAX_LOG_BYTES", "MAX_LOG_ROWS", "MemoryToolLog"]
