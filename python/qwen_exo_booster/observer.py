from __future__ import annotations

import math
from collections import deque
from typing import Any

import msgspec

from qwen_exo_booster.score_bias import (
    SCORE_BIAS_BLOCK_SIZE,
    ScoreBiasRecord,
    block_surprise_records,
)
from qwen_exo_booster.telemetry import TelemetryStore

# Tokens in the "local" surprisal window reported by observer.decode_summary.
LOCAL_WINDOW_TOKENS = 8


class ObserverRequestState(msgspec.Struct):
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
    last_emitted_token: int = 0
    surprisal_values: deque[float] = msgspec.field(
        default_factory=lambda: deque(maxlen=32)
    )
    # Full history is retained only for the request lifetime. Runtime snapshots
    # it into bounded block records before releasing this state.
    all_surprisals: list[float] = msgspec.field(default_factory=list)
    all_output_ids: list[int] = msgspec.field(default_factory=list)


class ObserverResult(msgspec.Struct, frozen=True):
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
    finished: bool


class InFlightObserver:
    """Tracks request-stable decode signals (surprisal, Q drift, memory energy)
    and reports them as ``observer.decode_summary`` telemetry."""

    def __init__(
        self,
        telemetry: TelemetryStore,
        *,
        mode: str,
        ema_alpha: float = 0.2,
        summary_interval_tokens: int = 16,
    ):
        # "active" is a legacy alias of "shadow" (it used to allow self-ask).
        if mode not in {"off", "shadow", "active"}:
            raise ValueError("Observer mode must be off, shadow, or active")
        if not 0 < ema_alpha <= 1:
            raise ValueError("Observer EMA alpha must be in (0, 1]")
        if summary_interval_tokens < 1:
            raise ValueError("Observer summary interval must be positive")
        self.telemetry = telemetry
        self.mode = mode
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
    ) -> ObserverResult:
        request_id = str(request_id)
        state = self._states.get(request_id)
        if state is None:
            state = ObserverRequestState(request_id=request_id)
            self._states[request_id] = state
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
        for offset, item in enumerate(new_logprobs):
            logprob = self._logprob_value(item)
            if logprob is None or not math.isfinite(logprob):
                continue
            self._observe_signals(state, meta, start_index + offset)
            token_id = int(new_output_ids[offset]) if offset < len(new_output_ids) else None
            self._observe_token(state, max(0.0, -logprob), token_id)
            new_count += 1

        finished = bool(meta.get("finish_reason"))
        self._maybe_emit_summary(request_id, state, finished=finished)
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
            finished=finished,
        )

    def _observe_signals(
        self, state: ObserverRequestState, meta: dict[str, Any], index: int
    ) -> None:
        q_norm = self._signal_value(meta, "qwen_exo_q_norm", index)
        q_drift = self._signal_value(meta, "qwen_exo_q_drift", index)
        memory_energy = self._signal_value(meta, "qwen_exo_memory_energy", index)
        if q_norm is not None:
            state.latest_q_norm = q_norm
        if q_drift is not None:
            state.latest_q_drift = q_drift
            state.max_q_drift = max(state.max_q_drift, q_drift)
        if memory_energy is not None:
            state.latest_memory_energy = memory_energy
            state.max_memory_energy = max(state.max_memory_energy, memory_energy)

    def _observe_token(
        self, state: ObserverRequestState, surprisal: float, token_id: int | None
    ) -> None:
        state.token_count += 1
        state.generation_token_count += 1
        state.current_surprisal = surprisal
        if state.token_count == 1:
            state.ema_surprisal = surprisal
        else:
            state.ema_surprisal = (
                self.ema_alpha * surprisal + (1 - self.ema_alpha) * state.ema_surprisal
            )
        state.max_surprisal = max(state.max_surprisal, surprisal)
        state.surprisal_values.append(surprisal)
        state.all_surprisals.append(surprisal)
        if token_id is not None:
            state.all_output_ids.append(token_id)
        values = tuple(state.surprisal_values)
        recent = values[-LOCAL_WINDOW_TOKENS:]
        history = values[:-LOCAL_WINDOW_TOKENS]
        state.local_window_mean = sum(recent) / len(recent)
        state.history_mean = (
            sum(history) / len(history) if history else state.local_window_mean
        )

    def _maybe_emit_summary(
        self, request_id: str, state: ObserverRequestState, *, finished: bool
    ) -> None:
        tokens_since_emit = state.token_count - state.last_emitted_token
        if not finished and tokens_since_emit < self.summary_interval_tokens:
            return
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
                "q_norm": state.latest_q_norm,
                "q_drift": state.latest_q_drift,
                "max_q_drift": state.max_q_drift,
                "memory_attention_energy_proxy": state.latest_memory_energy,
                "max_memory_attention_energy_proxy": state.max_memory_energy,
                "entropy": None,
                "finished": finished,
            },
        )

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
