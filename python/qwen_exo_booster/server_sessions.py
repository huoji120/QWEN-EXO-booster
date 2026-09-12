"""Console-only erasure of retained diagnostic/re-reflection sources.

Published lessons and serving lineage are deliberately not a privacy-purge target.
Admission and erasure reservations are event-loop local; SQLite work stays off-loop.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import ExitStack, closing
from functools import wraps
from typing import Any

from qwen_exo_booster.attention_diagnostic_conversations import _database


async def wait_source_admission(runtime):
    while (erasure := getattr(runtime, "_server_session_erasure", None)) is not None:
        await asyncio.shield(erasure)


def retained_source_producer(method):
    """Protect admission/compaction, including their pre-conversation awaits."""

    @wraps(method)
    async def protected(runtime, *args, **kwargs):
        await wait_source_admission(runtime)
        producers = getattr(runtime, "_server_session_producers", None)
        if producers is None:
            producers = runtime._server_session_producers = set()
        task = asyncio.current_task()
        producers.add(task)
        try:
            return await method(runtime, *args, **kwargs)
        finally:
            producers.discard(task)

    return protected


def require_source_admission(runtime):
    if getattr(runtime, "_server_session_erasure", None) is not None:
        raise RuntimeError(
            "Retained session deletion is in progress; retry after completion"
        )


class ServerSessionStore:
    def __init__(self, runtime: Any):
        self.runtime = runtime

    def _metadata(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        journal_keys = set()
        journal = getattr(self.runtime, "reflection_evidence_store", None)
        if journal is not None:
            with _database(journal) as db:
                for row in db.execute(
                    "SELECT conversation_key,last_seen,event_count,raw_bytes FROM conversations WHERE event_count > 0"
                ):
                    key = str(row[0])
                    journal_keys.add(key)
                    result[key] = {
                        "updated_at": float(row[1] or 0),
                        "event_count": int(row[2]),
                        "raw_bytes": int(row[3] or 0),
                        "source_count": 0,
                    }
        snapshots = getattr(self.runtime, "reflection_source_store", None)
        if snapshots is not None:
            with _database(snapshots) as db:
                rows = db.execute(
                    "SELECT conversation_key,captured_at,trajectory_row_count,payload_bytes "
                    "FROM reflection_sources ORDER BY captured_at DESC,source_digest"
                ).fetchall()
            for key, captured_at, count, raw_bytes in rows:
                row = result.setdefault(
                    key,
                    {
                        "updated_at": 0.0,
                        "event_count": 0,
                        "raw_bytes": 0,
                        "source_count": 0,
                    },
                )
                row["updated_at"] = max(row["updated_at"], float(captured_at or 0))
                if key not in journal_keys and row["source_count"] == 0:
                    row["event_count"] = int(count)
                row["raw_bytes"] += int(raw_bytes or 0)
                row["source_count"] += 1
        return result

    async def _retained_metadata(self):
        # Snapshot references on the loop; count large text payloads in a worker.
        memory = tuple(
            (key, tuple(rows))
            for key, rows in getattr(
                self.runtime, "_reflection_memory_trajectories", {}
            ).items()
        )
        metadata = await asyncio.to_thread(self._merge_memory, memory)
        for key in getattr(self.runtime, "_pending_reflection_memories", {}):
            metadata.setdefault(
                key,
                {
                    "updated_at": 0.0,
                    "event_count": 0,
                    "raw_bytes": 0,
                    "source_count": 0,
                },
            )
        return metadata

    def _merge_memory(self, memory):
        metadata = self._metadata()
        for key, rows in memory:
            if not rows:
                continue
            item = metadata.setdefault(
                key,
                {
                    "updated_at": 0.0,
                    "event_count": len(rows),
                    "raw_bytes": 0,
                    "source_count": 0,
                },
            )
            # Estimate retained text bytes, not Python overhead or physical disk size.
            item["raw_bytes"] += sum(
                len(str(row.get("content", "")).encode("utf-8")) for row in rows
            )
            if not item["event_count"]:
                item["event_count"] = len(rows)
        return metadata

    def _globally_busy(self):
        runtime = self.runtime
        if any(
            not task.done()
            for task in getattr(runtime, "_server_session_producers", ())
        ):
            return True
        for name in (
            "_reflection_memory_regeneration_task",
            "_reflection_memory_organization_task",
        ):
            task = getattr(runtime, name, None)
            if task is not None and not task.done():
                return True
        if any(
            not task.done()
            for task in getattr(runtime, "_server_session_reflections", ())
        ):
            return True
        queue = getattr(runtime, "_compaction_reflection_queue", None)
        # Includes the checkpoint held by the worker and pending queue puts.
        return bool(queue is not None and getattr(queue, "_unfinished_tasks", 0))

    def _active(self, key: str) -> bool:
        runtime = self.runtime
        if self._globally_busy():
            return True
        if key in getattr(runtime, "_request_conversation_keys", {}).values():
            return True
        associations = getattr(runtime, "_conversation_keys_by_response_id", {})
        for name in (
            "_refresh_tasks",
            "_replay_tasks",
            "_capsule_tasks",
            "_finalize_tasks",
        ):
            for request_id, task in getattr(runtime, name, {}).items():
                if not task.done() and associations.get(request_id) == key:
                    return True
        pending = getattr(runtime, "_pending_reflection_memories", {}).get(key)
        task = getattr(runtime, "_reflection_memory_tasks", {}).get(key)
        return bool(
            task is not None
            and not task.done()
            and (pending is None or getattr(pending, "status", "") != "waiting")
        )

    async def list(
        self, limit: int = 25, offset: int = 0, q: str = ""
    ) -> dict[str, Any]:
        metadata = await self._retained_metadata()
        query = q.casefold()
        keys = sorted(
            (key for key in metadata if query in key.casefold()),
            key=lambda key: (-metadata[key]["updated_at"], key),
        )
        return {
            "sessions": [
                {
                    "conversation_key": key,
                    **metadata[key],
                    "active": self._active(key),
                    "pending_reflection": key
                    in getattr(self.runtime, "_pending_reflection_memories", {}),
                }
                for key in keys[offset : offset + limit]
            ],
            "total": len(keys),
            "limit": limit,
            "offset": offset,
        }

    def _erase_databases(self, keys):
        journal = getattr(self.runtime, "reflection_evidence_store", None)
        snapshots = getattr(self.runtime, "reflection_source_store", None)
        with ExitStack() as stack:
            for store in (journal, snapshots):
                if store is not None:
                    stack.enter_context(store._lock)
            db = stack.enter_context(closing(sqlite3.connect(":memory:", timeout=5)))
            if journal is not None:
                db.execute("ATTACH DATABASE ? AS evidence", (str(journal.path),))
            if snapshots is not None:
                db.execute("ATTACH DATABASE ? AS sources", (str(snapshots.path),))
            try:
                db.execute("BEGIN IMMEDIATE")
                # Bound SQL variables even for clear-all; one transaction for all sources.
                for start in range(0, len(keys), 500):
                    batch = keys[start : start + 500]
                    placeholders = ",".join("?" for _ in batch)
                    if journal is not None:
                        for table in ("events", "conversations", "retention_gaps"):
                            db.execute(
                                f"DELETE FROM evidence.{table} WHERE conversation_key IN ({placeholders})",
                                batch,
                            )
                    if snapshots is not None:
                        db.execute(
                            f"DELETE FROM sources.reflection_sources WHERE conversation_key IN ({placeholders})",
                            batch,
                        )
                db.commit()
            except BaseException:
                db.rollback()
                raise

    async def delete(
        self, *, conversation_keys: list[str] | None = None, all: bool = False
    ) -> dict[str, Any]:
        work = asyncio.create_task(
            self._delete(conversation_keys=conversation_keys, all=all)
        )
        cancelled = False
        while not work.done():
            try:
                await asyncio.shield(work)
            except asyncio.CancelledError:
                cancelled = True
        result = work.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _delete(
        self, *, conversation_keys: list[str] | None, all: bool
    ) -> dict[str, Any]:
        require_source_admission(self.runtime)
        # Reserve before metadata IO. Incoming producers wait; existing work is skipped.
        reservation = asyncio.get_running_loop().create_future()
        self.runtime._server_session_erasure = reservation
        try:
            known = await self._retained_metadata()
            requested = sorted(known if all else dict.fromkeys(conversation_keys or ()))
            missing = [key for key in requested if key not in known]
            skipped = [key for key in requested if key in known and self._active(key)]
            targets = [key for key in requested if key in known and key not in skipped]
            if targets:
                await asyncio.to_thread(self._erase_databases, targets)
            # Only cancel dormant timers after commit. On rollback, queued work
            # remains intact and resumes when the admission reservation releases.
            timers = []
            for key in targets:
                task = getattr(self.runtime, "_reflection_memory_tasks", {}).get(key)
                if task is not None and not task.done():
                    task.cancel()
                    timers.append(task)
            if timers:
                await asyncio.gather(*timers, return_exceptions=True)
            if targets:
                for name in (
                    "_reflection_memory_trajectories",
                    "_reflection_memory_sources",
                    "_reflection_memory_last_activity",
                    "_pending_reflection_memories",
                    "_reflection_memory_tasks",
                ):
                    mapping = getattr(self.runtime, name, {})
                    for key in targets:
                        mapping.pop(key, None)
            return {
                "deleted": targets,
                "skipped_active": skipped,
                "missing": missing,
                "deleted_count": len(targets),
            }
        finally:
            self.runtime._server_session_erasure = None
            reservation.set_result(None)
