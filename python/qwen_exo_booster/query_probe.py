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


@dataclass(frozen=True, slots=True)
class QueryProbeResult:
    status: str
    prompt_tokens: int
    query_heads: tuple[tuple[tuple[float, ...], ...], ...]
    latency_seconds: float
    cache_hit: bool = False

    def public_dict(self) -> dict[str, Any]:
        query_head_count = len(self.query_heads[0]) if self.query_heads else 0
        head_dim = (
            len(self.query_heads[0][0])
            if self.query_heads and self.query_heads[0]
            else 0
        )
        return {
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "query_count": len(self.query_heads),
            "query_head_count": query_head_count,
            "head_dim": head_dim,
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

    async def probe(self, parent_request_id: str, question: str) -> QueryProbeResult:
        started = time.perf_counter()
        query_token_ids = self._encode(question)
        if not query_token_ids:
            return self._completed(
                parent_request_id,
                QueryProbeResult("empty_query", 0, (), time.perf_counter() - started),
            )
        query_capacity = self.max_prompt_tokens - len(self.cognition_token_ids)
        query_token_ids = query_token_ids[-query_capacity:]
        cognition_token_count = len(self.cognition_token_ids)
        token_ids = self.cognition_token_ids + query_token_ids
        spans = tuple(
            (start + cognition_token_count, end + cognition_token_count)
            for start, end in self._query_spans(len(query_token_ids))
        )
        cache_key = stable_digest(
            "query-probe-raw-heads-v2",
            tuple(token_ids),
            spans,
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
                + stable_digest(parent_request_id, tuple(token_ids))[:24]
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
                "span_count": len(spans),
                "cache_hit": cached_query_heads is not None,
            },
        )
        if cached_query_heads is not None:
            return self._completed(
                parent_request_id,
                QueryProbeResult(
                    "ready",
                    len(token_ids),
                    cached_query_heads,
                    time.perf_counter() - started,
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
                                {"start": start, "end": end} for start, end in spans
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
                {"error_type": type(exc).__name__},
            )
            return QueryProbeResult(
                "failed_closed", len(token_ids), (), time.perf_counter() - started
            )
        query_heads = self._query_heads(result.metadata)
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
                time.perf_counter() - started,
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

    @staticmethod
    def _query_spans(token_count: int) -> tuple[tuple[int, int], ...]:
        if token_count < 1:
            return ()
        state_count = min(_MAX_QUERY_STATES, int(token_count))
        width = math.ceil(token_count / state_count)
        return tuple(
            (start, min(start + width, token_count))
            for start in range(0, token_count, width)
        )[-_MAX_QUERY_STATES:]

    def _query_heads(
        self, metadata: dict[str, Any]
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
            finite = torch.isfinite(values).flatten(start_dim=1).all(dim=1)
            values = values[finite][:_MAX_QUERY_STATES]
            if not values.numel():
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
