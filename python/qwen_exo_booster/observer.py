from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qwen_exo_booster.contracts import ContractViolation
from qwen_exo_booster.telemetry import TelemetryStore
from qwen_exo_booster.score_bias import (
    SCORE_BIAS_BLOCK_SIZE,
    ScoreBiasRecord,
    block_surprise_records,
)


class AdaptiveRetrievalPhase(str, Enum):
    OBSERVING = "observing"
    RESTORED = "restored"
    TRIGGERED = "triggered"
    REFRESHING = "refreshing"
    SEMANTIC_READY = "semantic_ready"
    REPLAY_SCORING = "replay_scoring"
    NEXT_TURN_READY = "next_turn_ready"
    POST_TOOL_REFRESHING = "post_tool_refreshing"
    REJECTED = "rejected"
    FAILED_CLOSED = "failed_closed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


_ADAPTIVE_TRANSITIONS = {
    AdaptiveRetrievalPhase.OBSERVING: frozenset(
        {
            AdaptiveRetrievalPhase.RESTORED,
            AdaptiveRetrievalPhase.TRIGGERED,
            AdaptiveRetrievalPhase.POST_TOOL_REFRESHING,
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.RESTORED: frozenset(
        {
            AdaptiveRetrievalPhase.TRIGGERED,
            AdaptiveRetrievalPhase.POST_TOOL_REFRESHING,
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.TRIGGERED: frozenset(
        {
            AdaptiveRetrievalPhase.REFRESHING,
            AdaptiveRetrievalPhase.FAILED_CLOSED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.REFRESHING: frozenset(
        {
            AdaptiveRetrievalPhase.SEMANTIC_READY,
            AdaptiveRetrievalPhase.NEXT_TURN_READY,
            AdaptiveRetrievalPhase.REJECTED,
            AdaptiveRetrievalPhase.FAILED_CLOSED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.SEMANTIC_READY: frozenset(
        {
            AdaptiveRetrievalPhase.REPLAY_SCORING,
            AdaptiveRetrievalPhase.REJECTED,
            AdaptiveRetrievalPhase.FAILED_CLOSED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.REPLAY_SCORING: frozenset(
        {
            AdaptiveRetrievalPhase.NEXT_TURN_READY,
            AdaptiveRetrievalPhase.REJECTED,
            AdaptiveRetrievalPhase.FAILED_CLOSED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.NEXT_TURN_READY: frozenset(
        {
            AdaptiveRetrievalPhase.POST_TOOL_REFRESHING,
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.POST_TOOL_REFRESHING: frozenset(
        {
            AdaptiveRetrievalPhase.NEXT_TURN_READY,
            AdaptiveRetrievalPhase.REJECTED,
            AdaptiveRetrievalPhase.FAILED_CLOSED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.REJECTED: frozenset(
        {
            AdaptiveRetrievalPhase.POST_TOOL_REFRESHING,
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.FAILED_CLOSED: frozenset(
        {
            AdaptiveRetrievalPhase.POST_TOOL_REFRESHING,
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }
    ),
    AdaptiveRetrievalPhase.COMPLETED: frozenset(),
    AdaptiveRetrievalPhase.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class AdaptiveRetrievalState:
    request_id: str
    phase: AdaptiveRetrievalPhase
    sequence: int
    event_id: str | None = None
    decision: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase.value,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "decision": self.decision,
        }


class AdaptiveRetrievalStateMachine:
    def __init__(self, telemetry: TelemetryStore, *, max_states: int = 4096):
        if max_states < 1:
            raise ValueError("Adaptive retrieval max_states must be positive")
        self.telemetry = telemetry
        self.max_states = int(max_states)
        self._states: OrderedDict[str, AdaptiveRetrievalState] = OrderedDict()

    def begin(self, request_id: str) -> AdaptiveRetrievalState:
        request_id = str(request_id)
        existing = self._states.get(request_id)
        if existing is not None and existing.phase not in {
            AdaptiveRetrievalPhase.COMPLETED,
            AdaptiveRetrievalPhase.CANCELLED,
        }:
            raise ContractViolation(
                f"Adaptive retrieval request {request_id} is already active"
            )
        state = AdaptiveRetrievalState(
            request_id=request_id,
            phase=AdaptiveRetrievalPhase.OBSERVING,
            sequence=0,
        )
        self._store(state, previous=None)
        return state

    def transition(
        self,
        request_id: str,
        phase: AdaptiveRetrievalPhase,
        *,
        event_id: str | None = None,
        decision: str | None = None,
    ) -> AdaptiveRetrievalState:
        request_id = str(request_id)
        previous = self._states.get(request_id)
        if previous is None:
            raise ContractViolation(
                f"Adaptive retrieval request {request_id} was not initialized"
            )
        if phase not in _ADAPTIVE_TRANSITIONS[previous.phase]:
            raise ContractViolation(
                "Invalid adaptive retrieval transition "
                f"{previous.phase.value}->{phase.value}"
            )
        state = AdaptiveRetrievalState(
            request_id=request_id,
            phase=phase,
            sequence=previous.sequence + 1,
            event_id=event_id if event_id is not None else previous.event_id,
            decision=decision,
        )
        self._store(state, previous=previous)
        return state

    def state(self, request_id: str) -> AdaptiveRetrievalState | None:
        return self._states.get(str(request_id))

    def public_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for state in self._states.values():
            counts[state.phase.value] = counts.get(state.phase.value, 0) + 1
        return {
            "state_count": len(self._states),
            "phase_counts": counts,
        }

    def can_transition(self, request_id: str, phase: AdaptiveRetrievalPhase) -> bool:
        state = self._states.get(str(request_id))
        return state is not None and phase in _ADAPTIVE_TRANSITIONS[state.phase]

    def clear(self) -> None:
        self._states.clear()

    def _store(
        self,
        state: AdaptiveRetrievalState,
        *,
        previous: AdaptiveRetrievalState | None,
    ) -> None:
        self._states[state.request_id] = state
        self._states.move_to_end(state.request_id)
        while len(self._states) > self.max_states:
            self._states.popitem(last=False)
        self.telemetry.emit(
            state.request_id,
            "adaptive.transition",
            {
                "from": previous.phase.value if previous is not None else None,
                "to": state.phase.value,
                "sequence": state.sequence,
                "event_id": state.event_id,
                "decision": state.decision,
            },
        )


@dataclass(frozen=True, slots=True)
class MidThinkEvent:
    event_id: str
    request_id: str
    token_index: int
    trigger_reasons: tuple[str, ...]
    current_surprisal: float
    window_mean: float
    history_mean: float
    ema_surprisal: float
    recovery_window_mean: float
    uncertainty_state: str
    pre_q_sketches: tuple[tuple[float, ...], ...]
    post_q_sketches: tuple[tuple[float, ...], ...]
    generation_index: int = 0
    generation_token_index: int = 0
    attention_q_drift: float | None = None

    @property
    def local_q_window(self) -> tuple[tuple[float, ...], ...]:
        return self.pre_q_sketches + self.post_q_sketches

    def public_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "token_index": self.token_index,
            "generation_index": self.generation_index,
            "generation_token_index": self.generation_token_index,
            "trigger_reasons": list(self.trigger_reasons),
            "current_surprisal": self.current_surprisal,
            "window_mean": self.window_mean,
            "history_mean": self.history_mean,
            "ema_surprisal": self.ema_surprisal,
            "recovery_window_mean": self.recovery_window_mean,
            "uncertainty_state": self.uncertainty_state,
            "attention_q_drift": self.attention_q_drift,
            "retrieval_query_pre_tokens": len(self.pre_q_sketches),
            "retrieval_query_post_tokens": len(self.post_q_sketches),
            "retrieval_query_tokens": len(self.local_q_window),
        }


@dataclass(slots=True)
class ObserverRequestState:
    request_id: str
    generation_index: int = -1
    processed_logprobs: int = 0
    generation_token_count: int = 0
    token_count: int = 0
    current_surprisal: float = 0.0
    local_window_mean: float = 0.0
    history_mean: float = 0.0
    ema_surprisal: float = 0.0
    max_surprisal: float = 0.0
    latest_q_norm: float | None = None
    latest_q_drift: float | None = None
    max_q_drift: float = 0.0
    latest_memory_energy: float | None = None
    max_memory_energy: float = 0.0
    trigger_count: int = 0
    last_trigger_token: int = -1
    last_emitted_token: int = 0
    trigger_tokens: list[int] = field(default_factory=list)
    surprisal_values: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    # Full history is retained only for the request lifetime. Runtime snapshots
    # it into bounded block records before releasing this state.
    all_surprisals: list[float] = field(default_factory=list)
    all_output_ids: list[int] = field(default_factory=list)
    q_history: deque[tuple[float, ...]] = field(
        default_factory=lambda: deque(maxlen=32)
    )
    pending_event: dict[str, Any] | None = None
    in_reasoning: bool = True


@dataclass(frozen=True, slots=True)
class ObserverResult:
    request_id: str
    token_count: int
    new_tokens: int
    current_surprisal: float
    local_window_mean: float
    history_mean: float
    ema_surprisal: float
    max_surprisal: float
    latest_q_drift: float | None
    latest_memory_energy: float | None
    triggered: bool
    trigger_reasons: tuple[str, ...]
    events: tuple[MidThinkEvent, ...]
    finished: bool


class InFlightObserver:
    """Tracks request-stable decode signals and Demo-compatible events."""

    def __init__(
        self,
        telemetry: TelemetryStore,
        *,
        mode: str,
        surprisal_threshold: float = 0.8,
        surprisal_window: int = 8,
        surprisal_margin: float = 0.2,
        q_drift_threshold: float = 0.35,
        cooldown_tokens: int = 64,
        max_triggers: int = 1,
        q_pre_tokens: int = 8,
        q_post_tokens: int = 4,
        recovery_tokens: int = 8,
        immediate_uncertainty_retrieval: bool = False,
        ema_alpha: float = 0.2,
        summary_interval_tokens: int = 16,
    ):
        if mode not in {"off", "shadow", "active"}:
            raise ValueError("Observer mode must be off, shadow, or active")
        if (
            surprisal_threshold < 0
            or surprisal_window < 2
            or surprisal_margin < 0
            or q_drift_threshold < 0
            or cooldown_tokens < 1
            or max_triggers < 0
            or q_pre_tokens < 1
            or q_post_tokens < 1
            or recovery_tokens < 1
        ):
            raise ValueError("Invalid observer trigger limits")
        if not 0 < ema_alpha <= 1:
            raise ValueError("Observer EMA alpha must be in (0, 1]")
        if summary_interval_tokens < 1:
            raise ValueError("Observer summary interval must be positive")
        self.telemetry = telemetry
        self.mode = mode
        self.surprisal_threshold = float(surprisal_threshold)
        self.surprisal_window = int(surprisal_window)
        self.surprisal_margin = float(surprisal_margin)
        self.q_drift_threshold = float(q_drift_threshold)
        self.cooldown_tokens = int(cooldown_tokens)
        self.max_triggers = int(max_triggers)
        self.q_pre_tokens = int(q_pre_tokens)
        self.q_post_tokens = int(q_post_tokens)
        self.recovery_tokens = int(recovery_tokens)
        self.immediate_uncertainty_retrieval = bool(immediate_uncertainty_retrieval)
        self.ema_alpha = float(ema_alpha)
        self.summary_interval_tokens = int(summary_interval_tokens)
        self._states: dict[str, ObserverRequestState] = {}

    def observe_generation_result(
        self,
        request_id: str,
        result: dict[str, Any],
        *,
        incremental_logprobs: bool = False,
        generation_index: int = 0,
        reasoning_end_token_id: int | None = None,
    ) -> ObserverResult:
        request_id = str(request_id)
        state = self._states.setdefault(
            request_id, ObserverRequestState(request_id=request_id)
        )
        if state.generation_index != int(generation_index):
            state.generation_index = int(generation_index)
            state.processed_logprobs = 0
            state.generation_token_count = 0
        meta = result.get("meta_info") or {}
        raw_logprobs = meta.get("output_token_logprobs") or ()
        raw_output_ids = result.get("output_ids") or ()
        if incremental_logprobs:
            start_index = 0
            new_logprobs = raw_logprobs
            new_output_ids = raw_output_ids
            state.processed_logprobs += len(raw_logprobs)
        else:
            start_index = state.processed_logprobs
            new_logprobs = raw_logprobs[start_index:]
            new_output_ids = raw_output_ids[start_index:]
            state.processed_logprobs = len(raw_logprobs)

        new_count = 0
        triggered = False
        reasons: list[str] = []
        completed_events: list[MidThinkEvent] = []
        for offset, item in enumerate(new_logprobs):
            logprob = self._logprob_value(item)
            if logprob is None or not math.isfinite(logprob):
                continue
            signal_index = start_index + offset
            q_norm = self._signal_value(meta, "qwen_exo_q_norm", signal_index)
            q_drift = self._signal_value(meta, "qwen_exo_q_drift", signal_index)
            memory_energy = self._signal_value(
                meta, "qwen_exo_memory_energy", signal_index
            )
            q_sketch = self._signal_vector(meta, "qwen_exo_q_sketch", signal_index)
            if q_norm is not None:
                state.latest_q_norm = q_norm
            if q_drift is not None:
                state.latest_q_drift = q_drift
                state.max_q_drift = max(state.max_q_drift, q_drift)
            if memory_energy is not None:
                state.latest_memory_energy = memory_energy
                state.max_memory_energy = max(state.max_memory_energy, memory_energy)
            if q_sketch is not None:
                state.q_history.append(q_sketch)

            surprisal = max(0.0, -logprob)
            state.token_count += 1
            state.generation_token_count += 1
            state.current_surprisal = surprisal
            new_count += 1
            if state.token_count == 1:
                state.ema_surprisal = surprisal
            else:
                state.ema_surprisal = (
                    self.ema_alpha * surprisal
                    + (1 - self.ema_alpha) * state.ema_surprisal
                )
            state.max_surprisal = max(state.max_surprisal, surprisal)
            state.surprisal_values.append(surprisal)
            state.all_surprisals.append(surprisal)
            if offset < len(new_output_ids):
                state.all_output_ids.append(int(new_output_ids[offset]))
            values = tuple(state.surprisal_values)
            recent = values[-self.surprisal_window :]
            history = values[: -self.surprisal_window]
            state.local_window_mean = sum(recent) / len(recent)
            state.history_mean = (
                sum(history) / len(history) if history else state.local_window_mean
            )

            token_id = (
                int(new_output_ids[offset]) if offset < len(new_output_ids) else None
            )
            if (
                token_id is not None
                and reasoning_end_token_id is not None
                and token_id == int(reasoning_end_token_id)
            ):
                state.in_reasoning = False

            completed = self._advance_pending_event(state, q_sketch)
            if completed is not None:
                completed_events.append(completed)
                self.telemetry.emit(
                    request_id, "observer.mid_think_event", completed.public_dict()
                )

            token_reasons: list[str] = []
            window_ready = len(values) >= self.surprisal_window * 2
            relative_uncertainty = (
                window_ready
                and state.local_window_mean - state.history_mean
                >= self.surprisal_margin
            )
            if (
                relative_uncertainty
                and state.local_window_mean >= self.surprisal_threshold
            ):
                token_reasons.append("selected_token_surprisal_window")
            if (
                q_drift is not None
                and q_drift >= self.q_drift_threshold
                and relative_uncertainty
            ):
                token_reasons.append("attention_q_drift")
            cooldown_ready = (
                state.last_trigger_token < 0
                or state.token_count - state.last_trigger_token >= self.cooldown_tokens
            )
            can_trigger = (
                self.mode != "off"
                and state.in_reasoning
                and state.trigger_count < self.max_triggers
                and cooldown_ready
                and token_reasons
            )
            if can_trigger:
                state.trigger_count += 1
                state.last_trigger_token = state.token_count
                state.trigger_tokens.append(state.token_count)
                triggered = True
                reasons.extend(
                    reason for reason in token_reasons if reason not in reasons
                )
                if state.pending_event is None:
                    event_id = f"{request_id}:mid-think:{state.trigger_count}"
                    state.pending_event = {
                        "event_id": event_id,
                        "token_index": state.token_count,
                        "generation_index": state.generation_index,
                        "generation_token_index": state.generation_token_count,
                        "trigger_reasons": tuple(token_reasons),
                        "current_surprisal": surprisal,
                        "window_mean": state.local_window_mean,
                        "history_mean": state.history_mean,
                        "ema_surprisal": state.ema_surprisal,
                        "attention_q_drift": q_drift,
                        "pre_q_sketches": tuple(state.q_history)[-self.q_pre_tokens :],
                        "post_q_sketches": [],
                    }
                    self.telemetry.emit(
                        request_id,
                        "observer.mid_think_triggered",
                        {
                            "event_id": event_id,
                            "token_index": state.token_count,
                            "trigger_reasons": token_reasons,
                            "current_surprisal": surprisal,
                            "window_mean": state.local_window_mean,
                            "history_mean": state.history_mean,
                            "ema_surprisal": state.ema_surprisal,
                            "attention_q_drift": q_drift,
                            "retrieval_query_pre_tokens": len(
                                state.pending_event["pre_q_sketches"]
                            ),
                        },
                    )
                    if self.immediate_uncertainty_retrieval:
                        pending = state.pending_event
                        assert pending is not None
                        immediate_event = MidThinkEvent(
                            event_id=str(pending["event_id"]),
                            request_id=state.request_id,
                            token_index=int(pending["token_index"]),
                            trigger_reasons=tuple(pending["trigger_reasons"]),
                            current_surprisal=float(pending["current_surprisal"]),
                            window_mean=float(pending["window_mean"]),
                            history_mean=float(pending["history_mean"]),
                            ema_surprisal=float(pending["ema_surprisal"]),
                            recovery_window_mean=float(state.local_window_mean),
                            uncertainty_state="uncertainty_detected",
                            pre_q_sketches=tuple(pending["pre_q_sketches"]),
                            post_q_sketches=(),
                            generation_index=int(pending["generation_index"]),
                            generation_token_index=int(
                                pending["generation_token_index"]
                            ),
                            attention_q_drift=pending.get("attention_q_drift"),
                        )
                        completed_events.append(immediate_event)
                        self.telemetry.emit(
                            request_id,
                            "observer.mid_think_event",
                            immediate_event.public_dict(),
                        )
                        state.pending_event = None
                else:
                    self.telemetry.emit(
                        request_id,
                        "observer.mid_think_trigger_coalesced",
                        {
                            "event_id": state.pending_event["event_id"],
                            "token_index": state.token_count,
                            "trigger_reasons": token_reasons,
                        },
                    )

        finished = bool(meta.get("finish_reason"))
        tokens_since_emit = state.token_count - state.last_emitted_token
        should_emit = (
            finished
            or triggered
            or completed_events
            or tokens_since_emit >= self.summary_interval_tokens
        )
        if should_emit:
            state.last_emitted_token = state.token_count
            self.telemetry.emit(
                request_id,
                "observer.decode_summary",
                {
                    "mode": self.mode,
                    "new_tokens": tokens_since_emit,
                    "token_count": state.token_count,
                    "current_surprisal": state.current_surprisal,
                    "local_window_mean": state.local_window_mean,
                    "history_mean": state.history_mean,
                    "ema_surprisal": state.ema_surprisal,
                    "max_surprisal": state.max_surprisal,
                    "triggered": triggered,
                    "trigger_reasons": reasons,
                    "trigger_count": state.trigger_count,
                    "trigger_tokens": state.trigger_tokens,
                    "q_norm": state.latest_q_norm,
                    "q_drift": state.latest_q_drift,
                    "max_q_drift": state.max_q_drift,
                    "memory_attention_energy_proxy": state.latest_memory_energy,
                    "max_memory_attention_energy_proxy": state.max_memory_energy,
                    "entropy": None,
                    "finished": finished,
                },
            )
        return ObserverResult(
            request_id=request_id,
            token_count=state.token_count,
            new_tokens=new_count,
            current_surprisal=state.current_surprisal,
            local_window_mean=state.local_window_mean,
            history_mean=state.history_mean,
            ema_surprisal=state.ema_surprisal,
            max_surprisal=state.max_surprisal,
            latest_q_drift=state.latest_q_drift,
            latest_memory_energy=state.latest_memory_energy,
            triggered=triggered,
            trigger_reasons=tuple(reasons),
            events=tuple(completed_events),
            finished=finished,
        )

    def _advance_pending_event(
        self,
        state: ObserverRequestState,
        q_sketch: tuple[float, ...] | None,
    ) -> MidThinkEvent | None:
        pending = state.pending_event
        if pending is None or state.token_count <= int(pending["token_index"]):
            return None
        if (
            q_sketch is not None
            and len(pending["post_q_sketches"]) < self.q_post_tokens
        ):
            pending["post_q_sketches"].append(q_sketch)
        if state.token_count - int(pending["token_index"]) < self.recovery_tokens:
            return None
        future = tuple(state.surprisal_values)[-self.recovery_tokens :]
        recovery_mean = sum(future) / len(future)
        relative_persistence = (
            recovery_mean - float(pending["history_mean"]) >= self.surprisal_margin
        )
        persistent = relative_persistence and (
            recovery_mean >= self.surprisal_threshold
            or "attention_q_drift" in pending["trigger_reasons"]
        )
        event = MidThinkEvent(
            event_id=str(pending["event_id"]),
            request_id=state.request_id,
            token_index=int(pending["token_index"]),
            generation_index=int(pending["generation_index"]),
            generation_token_index=int(pending["generation_token_index"]),
            trigger_reasons=tuple(pending["trigger_reasons"]),
            current_surprisal=float(pending["current_surprisal"]),
            window_mean=float(pending["window_mean"]),
            history_mean=float(pending["history_mean"]),
            ema_surprisal=float(pending["ema_surprisal"]),
            recovery_window_mean=recovery_mean,
            uncertainty_state=(
                "persistent_uncertainty" if persistent else "resolved_by_continuation"
            ),
            pre_q_sketches=tuple(pending["pre_q_sketches"]),
            post_q_sketches=tuple(pending["post_q_sketches"]),
            attention_q_drift=pending.get("attention_q_drift"),
        )
        state.pending_event = None
        return event

    @staticmethod
    def _signal_value(meta: dict[str, Any], key: str, index: int) -> float | None:
        values = meta.get(key) or ()
        if index >= len(values):
            return None
        try:
            value = float(values[index])
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _signal_vector(
        meta: dict[str, Any], key: str, index: int
    ) -> tuple[float, ...] | None:
        values = meta.get(key) or ()
        if index >= len(values):
            return None
        raw = values[index]
        if not isinstance(raw, (list, tuple)) or not raw:
            return None
        try:
            vector = tuple(float(value) for value in raw)
        except (TypeError, ValueError):
            return None
        return vector if all(math.isfinite(value) for value in vector) else None

    def mark_reasoning_end(self, request_id: str) -> None:
        state = self._states.get(str(request_id))
        if state is not None:
            state.in_reasoning = False

    def state(self, request_id: str) -> ObserverRequestState | None:
        return self._states.get(str(request_id))

    def surprise_blocks(
        self,
        request_id: str,
        *,
        step: int,
        source: str = "generated",
        block_size: int = SCORE_BIAS_BLOCK_SIZE,
        max_blocks: int = 32,
    ) -> tuple[ScoreBiasRecord, ...]:
        state = self._states.get(str(request_id))
        if state is None or not state.all_output_ids:
            return ()
        count = min(len(state.all_output_ids), len(state.all_surprisals))
        if count < 1:
            return ()
        records = block_surprise_records(
            state.all_output_ids[:count],
            state.all_surprisals[:count],
            block_size=block_size,
            step=step,
            source=source,
        )
        return records[-int(max_blocks) :]

    def surprisal_history(self, request_id: str) -> tuple[float, ...]:
        state = self._states.get(str(request_id))
        return tuple(state.all_surprisals) if state is not None else ()

    def release(self, request_id: str) -> None:
        self._states.pop(str(request_id), None)

    def clear(self) -> None:
        self._states.clear()

    @staticmethod
    def _logprob_value(item: Any) -> float | None:
        if isinstance(item, dict):
            value = item.get("logprob")
        elif isinstance(item, (list, tuple)) and item:
            value = item[0]
        else:
            value = item
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
