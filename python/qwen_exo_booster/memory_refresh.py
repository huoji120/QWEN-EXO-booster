from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qwen_exo_booster.observer import ObserverResult
    from qwen_exo_booster.pipeline import MemoryPreparationState


@dataclass(slots=True)
class ConversationMemoryRefresh:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    memory: MemoryPreparationState | None = None
    revision: str = ""
    request_id: str = ""
    observed_tokens: int = 0
    high_surprisal_streak: int = 0
    high_average_streak: int = 0
    cooldown_remaining: int = 0
    pending_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryRefreshDecision:
    retrieve: bool
    reason: str
    high_surprisal_streak: int
    high_average_streak: int
    cooldown_remaining: int

    def public_dict(self) -> dict[str, object]:
        return {
            "decision": "retrieve" if self.retrieve else "reuse",
            "reason": self.reason,
            "high_surprisal_streak": self.high_surprisal_streak,
            "high_average_streak": self.high_average_streak,
            "cooldown_remaining": self.cooldown_remaining,
        }


class AdaptiveMemoryRefreshGate:
    """Bounded conversation memory, with token-based rather than chunk-based gates.

    A successful retrieval (including no matches) starts a three-turn cooldown.
    Sustained evidence is latched until the next successful retrieval. High-average
    windows are complete, non-overlapping observer windows; token spikes must also
    exceed the observer's historical baseline by its configured margin.
    """

    def __init__(
        self,
        *,
        max_conversations: int,
        surprisal_threshold: float,
        surprisal_margin: float,
        window_tokens: int,
        sustain: int = 3,
        average_windows: int = 2,
        cooldown_turns: int = 3,
    ) -> None:
        if (
            max_conversations < 1 or window_tokens < 2 or sustain < 1
            or average_windows < 1 or cooldown_turns < 0
            or not math.isfinite(surprisal_threshold) or surprisal_threshold < 0
            or not math.isfinite(surprisal_margin) or surprisal_margin < 0
        ):
            raise ValueError("Invalid memory refresh gate limits")
        self.max_conversations = max_conversations
        self.surprisal_threshold = surprisal_threshold
        self.surprisal_margin = surprisal_margin
        self.window_tokens = window_tokens
        self.sustain = sustain
        self.average_windows = average_windows
        self.cooldown_turns = cooldown_turns
        self._states: OrderedDict[str, ConversationMemoryRefresh] = OrderedDict()

    @asynccontextmanager
    async def conversation(self, key: str):
        state = self._states.get(key)
        if state is None:
            state = ConversationMemoryRefresh()
            self._states[key] = state
        self._states.move_to_end(key)
        state.users += 1
        try:
            async with state.lock:
                yield state
        finally:
            state.users -= 1
            self._trim()

    def _trim(self) -> None:
        if len(self._states) <= self.max_conversations:
            return
        for key, state in tuple(self._states.items()):
            if not state.users:
                del self._states[key]
            if len(self._states) <= self.max_conversations:
                break

    def decide(
        self, state: ConversationMemoryRefresh, *, request_id: str, revision: str,
    ) -> MemoryRefreshDecision:
        changed = state.memory is not None and state.revision != revision
        if changed:
            state.memory = None
            state.pending_reason = None
            state.high_surprisal_streak = state.high_average_streak = 0
            state.cooldown_remaining = 0
        if state.memory is None:
            reason = "source_changed" if changed else "first_request"
            retrieve = True
        elif state.cooldown_remaining:
            reason, retrieve = "cooldown", False
        elif state.pending_reason:
            reason, retrieve = state.pending_reason, True
        else:
            reason, retrieve = "no_sustained_uncertainty", False
        decision = MemoryRefreshDecision(
            retrieve, reason, state.high_surprisal_streak,
            state.high_average_streak, state.cooldown_remaining,
        )
        # Token streaks never bridge separate requests; completed window evidence
        # and a pending trigger do persist across conversation turns.
        state.request_id = request_id
        state.observed_tokens = 0
        state.high_surprisal_streak = 0
        if not retrieve and state.cooldown_remaining:
            state.cooldown_remaining -= 1
        return decision

    def commit(
        self, state: ConversationMemoryRefresh, memory: MemoryPreparationState,
        *, revision: str,
    ) -> None:
        # The cache is for attachment reuse, not historical query tensors/results.
        state.memory = replace(memory, query_heads=(), query_states=())
        state.revision = revision
        state.cooldown_remaining = self.cooldown_turns
        state.high_surprisal_streak = state.high_average_streak = 0
        state.pending_reason = None

    def observe(self, key: str, observation: ObserverResult) -> None:
        state = self._states.get(key)
        if state is None or observation.request_id != state.request_id:
            return
        first_token = observation.token_count - len(observation.refresh_signals) + 1
        for offset, (current, local_mean, history_mean) in enumerate(
            observation.refresh_signals
        ):
            token_index = first_token + offset
            if token_index <= state.observed_tokens:
                continue
            state.observed_tokens = token_index
            if not all(math.isfinite(value) for value in (current, local_mean, history_mean)):
                state.high_surprisal_streak = state.high_average_streak = 0
                continue
            high = current >= max(
                self.surprisal_threshold, history_mean + self.surprisal_margin
            )
            state.high_surprisal_streak = state.high_surprisal_streak + 1 if high else 0
            if token_index % self.window_tokens == 0:
                state.high_average_streak = (
                    state.high_average_streak + 1
                    if local_mean >= self.surprisal_threshold else 0
                )
            if state.high_surprisal_streak >= self.sustain:
                state.pending_reason = "sustained_high_surprisal"
            elif state.high_average_streak >= self.average_windows and not state.pending_reason:
                state.pending_reason = "sustained_high_average"

    def discard(self, key: str) -> None:
        self._states.pop(key, None)

    def clear(self) -> None:
        self._states.clear()
