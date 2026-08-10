from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SENSITIVE_KEYS = frozenset(
    {
        "prompt",
        "input",
        "output",
        "instructions",
        "messages",
        "question",
        "answer",
        "user_question",
        "partial_output",
        "original_task",
        "private_attachment",
        "tool_calls",
        "assistant_tool_calls",
        "reasoning",
        "arguments",
        "reference",
        "reflection",
        "reference_text",
        "content",
        "text",
        "tool_observation",
        "assistant_reasoning",
        "semantic_injection",
        "correction",
        "evidence_quote",
        "confirmed_facts",
        "invalid_claims",
        "contradictions",
        "stale_assumptions",
        "evidence_needed",
        "context_ledger",
        "capsule_history",
        "api_key",
        "secret",
    }
)


def _redacted_value(value: Any) -> dict[str, Any]:
    encoded = str(value).encode("utf-8")
    return {
        "redacted": True,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }


_BOUNDED_TEXT_CHARS = 800


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"…[截断，共 {len(value)} 字符]"


def redact_payload(
    value: Any, *, include_text: bool = False, max_chars: int = 0
) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if not include_text and (
                normalized_key in _SENSITIVE_KEYS
                or normalized_key.endswith("_text")
                or normalized_key.endswith("_arguments")
                or normalized_key.endswith("_content")
                or normalized_key.endswith("_prompt")
                or normalized_key.endswith("_reasoning")
                or normalized_key.endswith("_question")
                or normalized_key.endswith("_answer")
                or "api_key" in normalized_key
                or "secret" in normalized_key
            ):
                result[str(key)] = _redacted_value(item)
            else:
                result[str(key)] = redact_payload(
                    item, include_text=include_text, max_chars=max_chars
                )
        return result
    if isinstance(value, (list, tuple)):
        return [
            redact_payload(item, include_text=include_text, max_chars=max_chars)
            for item in value
        ]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, str) and include_text and max_chars > 0:
        return _truncate_text(value, max_chars)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    event_id: int
    request_id: str
    sequence: int
    event_type: str
    timestamp: float
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


class TelemetryStore:
    def __init__(
        self,
        path: Path | str,
        *,
        include_text: bool = False,
        text_mode: str | None = None,
        max_events: int = 4096,
        max_file_bytes: int = 64 * 1024 * 1024,
    ):
        if max_events < 1 or max_file_bytes < 1:
            raise ValueError("Telemetry retention limits must be positive")
        self.path = Path(path).expanduser().resolve()
        if text_mode is None:
            text_mode = "all" if include_text else "off"
        if text_mode not in {"off", "edited", "all"}:
            raise ValueError("telemetry text_mode must be off/edited/all")
        self.text_mode = text_mode
        self.include_text = text_mode != "off"
        self.text_scope: Any = None
        self.max_events = int(max_events)
        self.max_file_bytes = int(max_file_bytes)
        self._lock = threading.RLock()
        self._events: deque[TraceEvent] = deque(maxlen=self.max_events)
        self._retained_counts: defaultdict[str, int] = defaultdict(int)
        self._sequence: defaultdict[str, int] = defaultdict(int)
        self._event_id = 0
        self._disk_event_count = 0
        self._disk_bytes = 0
        self._persistence_failures = 0
        self._persistence_error: str | None = None
        self._persistence_suspended = False
        self._load_existing()

    def _payload_for(self, request_id: str, payload: dict[str, Any]) -> Any:
        if self.text_mode == "off":
            return redact_payload(payload, include_text=False)
        if self.text_mode == "all":
            return redact_payload(payload, include_text=True)
        scope = self.text_scope
        edited = bool(scope(request_id)) if callable(scope) else False
        return redact_payload(
            payload,
            include_text=edited,
            max_chars=_BOUNDED_TEXT_CHARS if edited else 0,
        )

    def emit(
        self, request_id: str, event_type: str, payload: dict[str, Any]
    ) -> TraceEvent:
        request_id = str(request_id)
        if not request_id or not str(event_type).strip():
            raise ValueError("Telemetry events require request and event identities")
        with self._lock:
            sequence = self._sequence[request_id]
            self._sequence[request_id] += 1
            event_id = self._event_id
            self._event_id += 1
            event = TraceEvent(
                event_id=event_id,
                request_id=request_id,
                sequence=sequence,
                event_type=str(event_type),
                timestamp=time.time(),
                payload=self._payload_for(request_id, payload),
            )
            self._retain_locked(event)
            if self._persistence_suspended:
                return event
            try:
                encoded_bytes = self._append_locked(event)
                self._disk_event_count += 1
                self._disk_bytes += encoded_bytes
                if (
                    self._disk_event_count >= self.max_events * 2
                    or self._disk_bytes >= self.max_file_bytes
                ):
                    self._compact_locked()
                self._persistence_error = None
            except OSError as exc:
                self._persistence_failures += 1
                self._persistence_error = f"{type(exc).__name__}: {exc}"
                self._persistence_suspended = True
            return event

    def events(
        self, request_id: str | None = None, *, limit: int = 256
    ) -> tuple[TraceEvent, ...]:
        with self._lock:
            selected = (
                tuple(self._events)
                if request_id is None
                else tuple(
                    event
                    for event in self._events
                    if event.request_id == str(request_id)
                )
            )
        return selected[-max(0, int(limit)) :]

    def events_after(
        self, event_id: int, *, limit: int = 256
    ) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(
                event for event in self._events if event.event_id > int(event_id)
            )[: max(0, int(limit))]

    def clear(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=self.path.parent,
                    prefix=f".{self.path.name}.",
                    delete=False,
                ) as temporary:
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, self.path)
                self._events.clear()
                self._retained_counts.clear()
                self._sequence.clear()
                self._event_id = 0
                self._disk_event_count = 0
                self._disk_bytes = 0
                self._persistence_error = None
                self._persistence_suspended = False
            except OSError as exc:
                self._persistence_failures += 1
                self._persistence_error = f"{type(exc).__name__}: {exc}"
                self._persistence_suspended = True
                raise
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def persistence_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": self._persistence_error is None,
                "failures": self._persistence_failures,
                "last_error": self._persistence_error,
                "retained_events": len(self._events),
                "suspended": self._persistence_suspended,
                "disk_events": self._disk_event_count,
                "disk_bytes": self._disk_bytes,
                "next_event_id": self._event_id,
            }

    def _retain_locked(self, event: TraceEvent) -> None:
        if len(self._events) == self.max_events:
            evicted = self._events[0]
            self._retained_counts[evicted.request_id] -= 1
            if self._retained_counts[evicted.request_id] <= 0:
                self._retained_counts.pop(evicted.request_id, None)
                if evicted.request_id != event.request_id:
                    self._sequence.pop(evicted.request_id, None)
        self._events.append(event)
        self._retained_counts[event.request_id] += 1

    def _load_existing(self) -> None:
        if not self.path.is_file():
            return
        try:
            self._disk_bytes = self.path.stat().st_size
            with self.path.open("r", encoding="utf-8") as source:
                for line in source:
                    self._disk_event_count += 1
                    try:
                        raw = json.loads(line)
                        event = TraceEvent(
                            event_id=int(raw["event_id"]),
                            request_id=str(raw["request_id"]),
                            sequence=int(raw["sequence"]),
                            event_type=str(raw["event_type"]),
                            timestamp=float(raw["timestamp"]),
                            payload=redact_payload(
                                dict(raw.get("payload") or {}),
                                include_text=self.include_text,
                            ),
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        continue
                    self._retain_locked(event)
                    self._event_id = max(self._event_id, event.event_id + 1)
                    self._sequence[event.request_id] = max(
                        self._sequence[event.request_id], event.sequence + 1
                    )
            if (
                (not self.include_text and self._disk_event_count > 0)
                or self._disk_event_count >= self.max_events * 2
                or self._disk_bytes >= self.max_file_bytes
            ):
                self._compact_locked()
        except OSError as exc:
            self._persistence_failures += 1
            self._persistence_error = f"{type(exc).__name__}: {exc}"
            self._persistence_suspended = True

    @staticmethod
    def _encode(event: TraceEvent) -> str:
        return json.dumps(
            event.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _append_locked(self, event: TraceEvent) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = self._encode(event)
        with self.path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(encoded)
            output.write("\n")
        return len(encoded.encode("utf-8")) + 1

    def _compact_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded_events = [self._encode(event) for event in self._events]
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            delete=False,
        ) as temporary:
            for encoded in encoded_events:
                temporary.write(encoded)
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._disk_event_count = len(encoded_events)
        self._disk_bytes = sum(len(item.encode("utf-8")) + 1 for item in encoded_events)
