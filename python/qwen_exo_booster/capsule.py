from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from qwen_exo_booster.contracts import (
    CancellationToken,
    InternalJob,
    InternalJobType,
    stable_digest,
)
from qwen_exo_booster.internal_jobs import InternalJobRunner

_CAPSULE_EVENTS = frozenset({"PROGRESS", "BLOCKED", "NO_MATERIAL_CHANGE", "UNKNOWN"})
_CAPSULE_BOOLEANS = frozenset({"YES", "NO", "UNKNOWN"})
_CAPSULE_FIELDS = frozenset(
    {
        "summary",
        "phase",
        "established",
        "unresolved",
        "next_action",
        "event",
        "state_change",
        "verification",
        "repetition",
    }
)
_CAPSULE_SYSTEM = (
    "Update a compact execution-state capsule from the supplied event. Interpret "
    "all natural language in its own language. Direct tool observations override "
    "assumptions. Do not claim a change, verification, progress, blockage, or "
    "repetition unless the supplied event establishes it. Preserve material "
    "identifiers. Return only one JSON object with exactly these fields: summary "
    "and phase as strings; established and unresolved as arrays of strings; "
    "next_action as a string; event as PROGRESS, BLOCKED, NO_MATERIAL_CHANGE, or "
    "UNKNOWN; state_change, verification, and repetition as YES, NO, or UNKNOWN. "
    "Use the original task's natural language for free-text values."
)
_CAPSULE_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "phase": {"type": "string"},
            "established": {"type": "array", "items": {"type": "string"}},
            "unresolved": {"type": "array", "items": {"type": "string"}},
            "next_action": {"type": "string"},
            "event": {"enum": sorted(_CAPSULE_EVENTS)},
            "state_change": {"enum": sorted(_CAPSULE_BOOLEANS)},
            "verification": {"enum": sorted(_CAPSULE_BOOLEANS)},
            "repetition": {"enum": sorted(_CAPSULE_BOOLEANS)},
        },
        "required": sorted(_CAPSULE_FIELDS),
        "additionalProperties": False,
    },
    separators=(",", ":"),
)


def parse_execution_capsule(value: str) -> dict[str, Any] | None:
    def reject_duplicates(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    try:
        capsule = json.loads(str(value).strip(), object_pairs_hook=reject_duplicates)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(capsule, dict) or set(capsule) != _CAPSULE_FIELDS:
        return None
    if not all(
        isinstance(capsule[field], str) for field in ("summary", "phase", "next_action")
    ):
        return None
    if not all(
        isinstance(capsule[field], list)
        and all(isinstance(item, str) for item in capsule[field])
        for field in ("established", "unresolved")
    ):
        return None
    if capsule["event"] not in _CAPSULE_EVENTS:
        return None
    if any(
        capsule[field] not in _CAPSULE_BOOLEANS
        for field in ("state_change", "verification", "repetition")
    ):
        return None
    return capsule


@dataclass(frozen=True, slots=True)
class CapsuleUpdateInput:
    parent_request_id: str
    turn_id: str
    trajectory_id: str
    event_sequence: int
    original_task: str
    previous_capsule: dict[str, Any] | None
    assistant_reasoning: str
    assistant_tool_calls: tuple[dict[str, Any], ...]
    tool_observation: str
    telemetry_correlation_id: str
    parent_trajectory_id: str | None = None

    @property
    def event_digest(self) -> str:
        payload = {
            "assistant_reasoning": self.assistant_reasoning,
            "assistant_tool_calls": self.assistant_tool_calls,
            "tool_observation": self.tool_observation,
        }
        return stable_digest(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @property
    def previous_capsule_digest(self) -> str:
        return stable_digest(
            json.dumps(self.previous_capsule, ensure_ascii=False, sort_keys=True)
            if self.previous_capsule
            else ""
        )


@dataclass(frozen=True, slots=True)
class CapsuleRecord:
    trajectory_id: str
    source_turn_id: str
    event_sequence: int
    event_digest: str
    previous_capsule_digest: str
    parent_trajectory_id: str | None
    original_task: str
    capsule: dict[str, Any]
    updated_at: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "source_turn_id": self.source_turn_id,
            "event_sequence": self.event_sequence,
            "event_digest": self.event_digest,
            "previous_capsule_digest": self.previous_capsule_digest,
            "parent_trajectory_id": self.parent_trajectory_id,
            "original_task": self.original_task,
            "capsule": dict(self.capsule),
            "updated_at": self.updated_at,
        }


class ExecutionCapsuleStore:
    def __init__(self, path: Path | str, *, max_records: int = 512):
        if max_records < 1:
            raise ValueError("Capsule store max_records must be positive")
        self.path = Path(path).expanduser().resolve()
        self.max_records = int(max_records)
        self._lock = threading.RLock()
        self._records: OrderedDict[str, CapsuleRecord] = OrderedDict()
        self._dedupe: set[tuple[str, str, str]] = set()
        self._load()

    def get(self, trajectory_id: str) -> CapsuleRecord | None:
        with self._lock:
            record = self._records.get(str(trajectory_id))
            if record is not None:
                self._records.move_to_end(str(trajectory_id))
            return record

    def lineage(
        self, trajectory_id: str, *, max_turns: int = 100
    ) -> tuple[CapsuleRecord, ...]:
        if max_turns < 1:
            return ()
        lineage = []
        seen = set()
        current_id: str | None = str(trajectory_id)
        with self._lock:
            while current_id and current_id not in seen and len(lineage) < max_turns:
                seen.add(current_id)
                record = self._records.get(current_id)
                if record is None:
                    break
                lineage.append(record)
                current_id = record.parent_trajectory_id
        lineage.reverse()
        return tuple(lineage)

    def is_duplicate(self, update: CapsuleUpdateInput) -> bool:
        key = (
            update.trajectory_id,
            update.event_digest,
            update.previous_capsule_digest,
        )
        with self._lock:
            return key in self._dedupe

    def commit(
        self, update: CapsuleUpdateInput, capsule: dict[str, Any]
    ) -> CapsuleRecord | None:
        parsed = parse_execution_capsule(
            json.dumps(capsule, ensure_ascii=False, separators=(",", ":"))
        )
        if parsed is None:
            return None
        key = (
            update.trajectory_id,
            update.event_digest,
            update.previous_capsule_digest,
        )
        with self._lock:
            current = self._records.get(update.trajectory_id)
            if current is not None and update.event_sequence <= current.event_sequence:
                return current
            if key in self._dedupe:
                return current
            record = CapsuleRecord(
                trajectory_id=update.trajectory_id,
                source_turn_id=update.turn_id,
                event_sequence=update.event_sequence,
                event_digest=update.event_digest,
                previous_capsule_digest=update.previous_capsule_digest,
                parent_trajectory_id=(
                    str(update.parent_trajectory_id)
                    if update.parent_trajectory_id
                    else None
                ),
                original_task=str(update.original_task),
                capsule=parsed,
                updated_at=time.time(),
            )
            self._records[update.trajectory_id] = record
            self._records.move_to_end(update.trajectory_id)
            while len(self._records) > self.max_records:
                self._records.popitem(last=False)
            self._dedupe = {
                (
                    retained.trajectory_id,
                    retained.event_digest,
                    retained.previous_capsule_digest,
                )
                for retained in self._records.values()
            }
            self._save_locked()
            return record

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or payload.get("schema") != 2:
            return
        records: list[CapsuleRecord] = []
        for raw in payload.get("records", []):
            if not isinstance(raw, dict):
                continue
            capsule = parse_execution_capsule(
                json.dumps(raw.get("capsule"), ensure_ascii=False)
            )
            if capsule is None:
                continue
            try:
                record = CapsuleRecord(
                    trajectory_id=str(raw["trajectory_id"]),
                    source_turn_id=str(raw["source_turn_id"]),
                    event_sequence=int(raw["event_sequence"]),
                    event_digest=str(raw["event_digest"]),
                    previous_capsule_digest=str(raw["previous_capsule_digest"]),
                    parent_trajectory_id=(
                        str(raw["parent_trajectory_id"])
                        if raw.get("parent_trajectory_id")
                        else None
                    ),
                    original_task=str(raw["original_task"]),
                    capsule=capsule,
                    updated_at=float(raw["updated_at"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            if record.event_sequence < 0 or not record.trajectory_id:
                continue
            records.append(record)
        retained = sorted(records, key=lambda item: item.updated_at)[
            -self.max_records :
        ]
        self._records = OrderedDict(
            (record.trajectory_id, record) for record in retained
        )
        self._dedupe = {
            (
                record.trajectory_id,
                record.event_digest,
                record.previous_capsule_digest,
            )
            for record in retained
        }

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 2,
            "records": [
                record.public_dict()
                for record in sorted(
                    self._records.values(), key=lambda item: item.trajectory_id
                )
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CapsuleUpdateResult:
    update: CapsuleUpdateInput
    record: CapsuleRecord | None
    valid: bool
    deduplicated: bool
    tokens: int
    latency_seconds: float


class ExecutionCapsuleService:
    def __init__(
        self,
        runner: InternalJobRunner,
        store: ExecutionCapsuleStore,
        tokenizer: Any,
        *,
        token_budget: int = 256,
        timeout_seconds: float = 60.0,
    ):
        self.runner = runner
        self.store = store
        self.tokenizer = tokenizer
        self.token_budget = int(token_budget)
        self.timeout_seconds = float(timeout_seconds)

    async def update_many(
        self, updates: Iterable[CapsuleUpdateInput]
    ) -> tuple[CapsuleUpdateResult, ...]:
        update_list = tuple(updates)
        tasks = [self._update(update) for update in update_list]
        if not tasks:
            return ()
        return tuple(await asyncio.gather(*tasks))

    async def _update(self, update: CapsuleUpdateInput) -> CapsuleUpdateResult:
        current = self.store.get(update.trajectory_id)
        if update.previous_capsule is not None:
            previous = parse_execution_capsule(
                json.dumps(update.previous_capsule, ensure_ascii=False)
            )
            if previous is None:
                return CapsuleUpdateResult(
                    update=update,
                    record=current,
                    valid=False,
                    deduplicated=False,
                    tokens=0,
                    latency_seconds=0.0,
                )
        if self.store.is_duplicate(update):
            return CapsuleUpdateResult(
                update=update,
                record=current,
                valid=current is not None,
                deduplicated=True,
                tokens=0,
                latency_seconds=0.0,
            )
        if current is not None and update.event_sequence <= current.event_sequence:
            return CapsuleUpdateResult(
                update=update,
                record=current,
                valid=True,
                deduplicated=True,
                tokens=0,
                latency_seconds=0.0,
            )

        deadline = time.monotonic() + self.timeout_seconds
        shared_prefix_key = "qwen-exo:v1:capsule:" + stable_digest(_CAPSULE_SYSTEM)[:24]
        job = InternalJob(
            parent_request_id=update.parent_request_id,
            turn_id=update.turn_id,
            job_id="qwen-exo-capsule-"
            + stable_digest(
                update.parent_request_id,
                update.turn_id,
                update.event_digest,
                update.previous_capsule_digest,
            )[:32],
            job_type=InternalJobType.CAPSULE_UPDATE,
            priority=-20,
            shared_prefix_key=shared_prefix_key,
            token_budget=self.token_budget,
            state_budget_bytes=0,
            deadline_monotonic=deadline,
            cancellation_token=CancellationToken(
                f"cancel-{update.parent_request_id}-{update.turn_id}"
            ),
            telemetry_correlation_id=update.telemetry_correlation_id,
            max_fanout=self.runner.max_fanout,
        )
        prompt = self._render_prompt(update)
        try:
            result = (
                await self.runner.run_batch(
                    [job],
                    [prompt],
                    {
                        "temperature": 0,
                        "top_p": 1,
                        "top_k": 1,
                        "json_schema": _CAPSULE_SCHEMA,
                        "skip_special_tokens": True,
                    },
                )
            )[0]
        except asyncio.CancelledError:
            raise
        except Exception:
            return CapsuleUpdateResult(
                update=update,
                record=current,
                valid=False,
                deduplicated=False,
                tokens=0,
                latency_seconds=0.0,
            )

        finish_reason = result.finish_reason or {}
        finish_type = (
            finish_reason.get("type")
            if isinstance(finish_reason, dict)
            else str(finish_reason)
        )
        if finish_type != "stop":
            return CapsuleUpdateResult(
                update=update,
                record=current,
                valid=False,
                deduplicated=False,
                tokens=result.completion_tokens,
                latency_seconds=result.latency_seconds,
            )
        capsule = parse_execution_capsule(result.text)
        record = self.store.commit(update, capsule) if capsule is not None else None
        return CapsuleUpdateResult(
            update=update,
            record=record if record is not None else current,
            valid=record is not None,
            deduplicated=False,
            tokens=result.completion_tokens,
            latency_seconds=result.latency_seconds,
        )

    def _render_prompt(self, update: CapsuleUpdateInput) -> str:
        source = {
            "original_task": update.original_task,
            "previous_capsule": update.previous_capsule,
            "latest": {
                "assistant_reasoning": update.assistant_reasoning,
                "assistant_tool_calls": update.assistant_tool_calls,
                "tool_observation": update.tool_observation,
            },
        }
        messages = [
            {"role": "system", "content": _CAPSULE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    source, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
