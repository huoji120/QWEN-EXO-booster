from __future__ import annotations

import asyncio
from collections import OrderedDict
import math
import time
from dataclasses import dataclass
from typing import Any
import torch

from qwen_exo_booster.contracts import (
    CancellationToken,
    InternalJob,
    InternalJobType,
    stable_digest,
)
from qwen_exo_booster.internal_jobs import InternalJobRunner

_MAX_QUERY_STATES = 8
_QUERY_ROLES = frozenset({"original_task", "current_user", "trajectory_compaction"})
_ANCHOR_QUERY_ROLES = frozenset({"original_task", "current_user"})


@dataclass(frozen=True, slots=True)
class QueryRoleText:
    role: str
    text: str = ""

    def __post_init__(self) -> None:
        if self.role not in _QUERY_ROLES:
            raise ValueError(f"Unsupported query role: {self.role}")


@dataclass(frozen=True, slots=True)
class QueryProbePlan:
    segments: tuple[QueryRoleText, ...]

    def __post_init__(self) -> None:
        roles = tuple(segment.role for segment in self.segments)
        if len(roles) != len(set(roles)):
            raise ValueError("Query probe roles must be unique")

    @classmethod
    def current_user(cls, text: str) -> QueryProbePlan:
        return cls((QueryRoleText("current_user", str(text)),))

    @property
    def identity(self) -> str:
        return stable_digest(
            "query-probe-role-plan-v1",
            *((segment.role, str(segment.text).strip()) for segment in self.segments),
        )


@dataclass(frozen=True, slots=True)
class QueryStateSpan:
    role: str
    prompt_start: int
    prompt_end: int
    source_start: int
    source_end: int

    @property
    def anchor(self) -> bool:
        return self.role in _ANCHOR_QUERY_ROLES

    def public_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt_start": self.prompt_start,
            "prompt_end": self.prompt_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class QueryProbeResult:
    status: str
    prompt_tokens: int
    query_heads: tuple[tuple[tuple[float, ...], ...], ...]
    query_states: tuple[QueryStateSpan, ...]
    latency_seconds: float
    role_plan_digest: str
    cache_hit: bool = False

    def public_dict(self) -> dict[str, Any]:
        query_head_count = len(self.query_heads[0]) if self.query_heads else 0
        head_dim = (
            len(self.query_heads[0][0])
            if self.query_heads and self.query_heads[0]
            else 0
        )
        role_counts = {
            role: sum(state.role == role for state in self.query_states)
            for role in _QUERY_ROLES
            if any(state.role == role for state in self.query_states)
        }
        return {
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "query_count": len(self.query_heads),
            "query_head_count": query_head_count,
            "head_dim": head_dim,
            "role_plan_digest": self.role_plan_digest,
            "role_counts": role_counts,
            "query_states": [state.public_dict() for state in self.query_states],
            "latency_seconds": self.latency_seconds,
            "cache_hit": self.cache_hit,
        }


class QueryProbeService:
    """Extract raw final-layer user-query Attention-Q heads before prefill."""

    def __init__(
        self,
        runner: InternalJobRunner,
        tokenizer: Any,
        telemetry: Any,
        *,
        max_prompt_tokens: int,
        cognition_token_ids: tuple[int, ...] = (),
        query_head_count: int | None = None,
        head_dim: int | None = None,
        timeout_seconds: float = 30.0,
        cache_size: int = 16,
    ) -> None:
        if max_prompt_tokens < 1 or timeout_seconds <= 0 or cache_size < 1:
            raise ValueError("Query probe limits and cache size must be positive")
        if len(cognition_token_ids) >= max_prompt_tokens:
            raise ValueError("Query probe Cognition prefix leaves no query capacity")
        if (query_head_count is None) != (head_dim is None) or (
            query_head_count is not None
            and (int(query_head_count) < 1 or int(head_dim) < 1)
        ):
            raise ValueError("Query probe raw-head geometry is invalid")
        self.runner = runner
        self.tokenizer = tokenizer
        self.telemetry = telemetry
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.cognition_token_ids = tuple(int(token) for token in cognition_token_ids)
        self.query_head_count = (
            int(query_head_count) if query_head_count is not None else None
        )
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.timeout_seconds = float(timeout_seconds)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[tuple[tuple[float, ...], ...], ...]] = (
            OrderedDict()
        )
        self._cache_lock = asyncio.Lock()

    async def probe(
        self, parent_request_id: str, plan: QueryProbePlan
    ) -> QueryProbeResult:
        started = time.perf_counter()
        role_plan_digest = plan.identity
        query_capacity = self.max_prompt_tokens - len(self.cognition_token_ids)
        try:
            query_token_ids, query_states = self._plan_query_tokens(
                plan, query_capacity
            )
        except ValueError as exc:
            self.telemetry.emit(
                str(parent_request_id),
                "query_probe.failed_closed",
                {
                    "error_type": type(exc).__name__,
                    "role_plan_digest": role_plan_digest,
                },
            )
            return self._completed(
                parent_request_id,
                QueryProbeResult(
                    "failed_closed",
                    0,
                    (),
                    (),
                    time.perf_counter() - started,
                    role_plan_digest,
                ),
            )
        if not query_token_ids:
            return self._completed(
                parent_request_id,
                QueryProbeResult(
                    "empty_query",
                    0,
                    (),
                    (),
                    time.perf_counter() - started,
                    role_plan_digest,
                ),
            )
        cognition_token_count = len(self.cognition_token_ids)
        token_ids = self.cognition_token_ids + query_token_ids
        cache_key = stable_digest(
            "query-probe-raw-heads-role-plan-v1",
            tuple(token_ids),
            tuple(
                (
                    state.role,
                    state.prompt_start,
                    state.prompt_end,
                    state.source_start,
                    state.source_end,
                )
                for state in query_states
            ),
            self.query_head_count,
            self.head_dim,
        )
        async with self._cache_lock:
            cached_query_heads = self._cache.get(cache_key)
            if cached_query_heads is not None:
                self._cache.move_to_end(cache_key)
        job = InternalJob(
            parent_request_id=str(parent_request_id),
            turn_id=f"{parent_request_id}:query-probe",
            job_id=f"qwen-exo-query-probe-{stable_digest(parent_request_id)[:32]}",
            job_type=InternalJobType.QUERY_PROBE,
            priority=-20,
            shared_prefix_key=(
                "qwen-exo:v1:query-probe:"
                + stable_digest(parent_request_id, tuple(token_ids), role_plan_digest)[
                    :24
                ]
            ),
            token_budget=1,
            state_budget_bytes=0,
            deadline_monotonic=time.monotonic() + self.timeout_seconds,
            cancellation_token=CancellationToken(
                f"cancel:{parent_request_id}:query-probe"
            ),
            telemetry_correlation_id=f"{parent_request_id}:query-probe",
            max_fanout=1,
        )
        self.telemetry.emit(
            str(parent_request_id),
            "query_probe.started",
            {
                "prompt_tokens": len(token_ids),
                "cognition_tokens": cognition_token_count,
                "query_tokens": len(query_token_ids),
                "span_count": len(query_states),
                "role_plan_digest": role_plan_digest,
                "role_counts": {
                    role: sum(state.role == role for state in query_states)
                    for role in _QUERY_ROLES
                    if any(state.role == role for state in query_states)
                },
                "cache_hit": cached_query_heads is not None,
            },
        )
        if cached_query_heads is not None:
            states = query_states[: len(cached_query_heads)]
            return self._completed(
                parent_request_id,
                QueryProbeResult(
                    "ready",
                    len(token_ids),
                    cached_query_heads[: len(states)],
                    states,
                    time.perf_counter() - started,
                    role_plan_digest,
                    cache_hit=True,
                ),
            )
        try:
            result = (
                await self.runner.run_batch(
                    (job,),
                    (token_ids,),
                    {"temperature": 0, "top_p": 1, "top_k": 1},
                    custom_params_per_job=(
                        {
                            "qwen_exo_query_spans": [
                                {
                                    "start": state.prompt_start,
                                    "end": state.prompt_end,
                                    "role": state.role,
                                }
                                for state in query_states
                            ]
                        },
                    ),
                )
            )[0]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.telemetry.emit(
                str(parent_request_id),
                "query_probe.failed_closed",
                {
                    "error_type": type(exc).__name__,
                    "role_plan_digest": role_plan_digest,
                },
            )
            return self._completed(
                parent_request_id,
                QueryProbeResult(
                    "failed_closed",
                    len(token_ids),
                    (),
                    (),
                    time.perf_counter() - started,
                    role_plan_digest,
                ),
            )
        query_heads = self._query_heads(
            result.metadata, expected_states=len(query_states)
        )
        states = query_states if query_heads else ()
        status = "ready" if query_heads else "no_q_signal"
        if query_heads:
            async with self._cache_lock:
                self._cache[cache_key] = query_heads
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return self._completed(
            parent_request_id,
            QueryProbeResult(
                status,
                len(token_ids),
                query_heads,
                states,
                time.perf_counter() - started,
                role_plan_digest,
            ),
        )

    def _encode(self, question: str) -> tuple[int, ...]:
        text = str(question).strip()
        if not text:
            return ()
        try:
            raw = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            raw = self.tokenizer.encode(text)
        return tuple(int(token) for token in raw or ())

    def _plan_query_tokens(
        self, plan: QueryProbePlan, capacity: int
    ) -> tuple[tuple[int, ...], tuple[QueryStateSpan, ...]]:
        encoded = [
            (segment.role, self._encode(segment.text))
            for segment in plan.segments
            if str(segment.text).strip()
        ]
        encoded = [(role, tokens) for role, tokens in encoded if tokens]
        if not encoded:
            return (), ()
        anchors = [
            (role, tokens) for role, tokens in encoded if role in _ANCHOR_QUERY_ROLES
        ]
        if len(anchors) > capacity:
            raise ValueError("Query probe budget cannot represent every anchor role")
        trajectory = next(
            (tokens for role, tokens in encoded if role == "trajectory_compaction"), ()
        )
        trajectory_budget = (
            min(len(trajectory), max(0, capacity // 4), capacity - len(anchors))
            if anchors
            else min(len(trajectory), capacity)
        )
        anchor_budget = capacity - trajectory_budget
        role_budgets: dict[str, int] = {}
        remaining = anchor_budget
        pending = list(anchors)
        while pending and remaining:
            next_pending = []
            for role, tokens in pending:
                if remaining < 1:
                    next_pending.append((role, tokens))
                    continue
                allocated = role_budgets.get(role, 0)
                if allocated < len(tokens):
                    role_budgets[role] = allocated + 1
                    remaining -= 1
                if role_budgets.get(role, 0) < len(tokens):
                    next_pending.append((role, tokens))
            pending = next_pending
        if trajectory_budget:
            role_budgets["trajectory_compaction"] = trajectory_budget

        selected: list[tuple[str, tuple[int, ...], int]] = []
        for role, tokens in encoded:
            budget = role_budgets.get(role, 0)
            if budget:
                source_start = len(tokens) - budget
                selected.append((role, tokens[source_start:], source_start))

        anchor_roles = [
            role for role, _tokens, _start in selected if role in _ANCHOR_QUERY_ROLES
        ]
        state_counts = {role: 1 for role in anchor_roles}
        trajectory_tokens = next(
            (
                tokens
                for role, tokens, _start in selected
                if role == "trajectory_compaction"
            ),
            (),
        )
        trajectory_states = min(
            2,
            len(trajectory_tokens),
            max(0, _MAX_QUERY_STATES - len(anchor_roles)),
        )
        if trajectory_states:
            state_counts["trajectory_compaction"] = trajectory_states
        remaining_states = _MAX_QUERY_STATES - sum(state_counts.values())
        while anchor_roles and remaining_states:
            progressed = False
            for role in anchor_roles:
                token_count = next(
                    len(tokens)
                    for item_role, tokens, _start in selected
                    if item_role == role
                )
                if state_counts[role] < token_count and remaining_states:
                    state_counts[role] += 1
                    remaining_states -= 1
                    progressed = True
            if not progressed:
                break

        query_tokens: list[int] = []
        states: list[QueryStateSpan] = []
        cognition_count = len(self.cognition_token_ids)
        for role, tokens, source_start in selected:
            prompt_role_start = cognition_count + len(query_tokens)
            query_tokens.extend(tokens)
            for start, end in self._query_spans(len(tokens), state_counts.get(role, 0)):
                states.append(
                    QueryStateSpan(
                        role=role,
                        prompt_start=prompt_role_start + start,
                        prompt_end=prompt_role_start + end,
                        source_start=source_start + start,
                        source_end=source_start + end,
                    )
                )
        return tuple(query_tokens), tuple(states)

    @staticmethod
    def _query_spans(token_count: int, state_limit: int) -> tuple[tuple[int, int], ...]:
        if token_count < 1 or state_limit < 1:
            return ()
        state_count = min(int(state_limit), int(token_count))
        width = math.ceil(token_count / state_count)
        return tuple(
            (start, min(start + width, token_count))
            for start in range(0, token_count, width)
        )[:state_count]

    def _query_heads(
        self, metadata: dict[str, Any], *, expected_states: int
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        raw_values = metadata.get("qwen_exo_user_query_full_heads")
        if raw_values is None:
            return ()
        candidates: list[Any] = [raw_values]
        if isinstance(raw_values, (list, tuple)):
            candidates.extend(reversed(raw_values))
        for raw in candidates:
            if raw is None:
                continue
            if hasattr(raw, "tolist"):
                raw = raw.tolist()
            try:
                values = torch.tensor(raw, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError):
                continue
            if self.query_head_count is not None and self.head_dim is not None:
                state_width = self.query_head_count * self.head_dim
                if values.numel() % state_width:
                    continue
                values = values.reshape(-1, self.query_head_count, self.head_dim)
            elif values.ndim == 2:
                values = values.unsqueeze(0)
            elif values.ndim < 2:
                continue
            else:
                values = values.reshape(
                    -1, int(values.shape[-2]), int(values.shape[-1])
                )
            if int(values.shape[0]) != int(expected_states):
                continue
            if not bool(torch.isfinite(values).all()):
                continue
            return tuple(
                tuple(tuple(float(value) for value in head) for head in query)
                for query in values.tolist()
            )
        return ()

    def _completed(
        self, parent_request_id: str, result: QueryProbeResult
    ) -> QueryProbeResult:
        self.telemetry.emit(
            str(parent_request_id), "query_probe.completed", result.public_dict()
        )
        return result
