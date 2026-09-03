"""Pointwise-mutual-information gate in front of the Semantic Judge.

Most requests have nothing to do with any stored memory, yet they still cost
one or two listwise judge rounds (all candidates rejected, expand, reject
again). Before the judge runs, this gate scores every shortlisted candidate's
rule-card head with the target model in one teacher-forced batch:

    pmi(doc) = mean_t log P(head_t | question) - mean_t log P(head_t | neutral)

A memory the question makes more predictable has positive PMI. When no
candidate clears the threshold, the request skips the judge and attaches
nothing, which is exactly what the judge would have concluded, only without
the generation rounds. The neutral-conditioned term depends only on the
document, so it is computed once per document digest and cached.

Measured on 23 labelled questions against the live bank: fusing PMI with the
existing Q/K + BM25 order raised hit@1 from 0.70 to 0.87; PMI alone matched
the baseline. This module only uses PMI as a negative gate.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable

from qwen_exo_booster.contracts import (
    CancellationToken,
    InternalJob,
    InternalJobType,
    stable_digest,
)
from qwen_exo_booster.knowledge import KnowledgeCandidate

_NEUTRAL_QUESTION = "我在做一个软件工程相关的任务，请给我一些通用建议。"
_LEAD_IN = "与此相关的一条经验记录：\n"
_FRONT_MATTER = re.compile(r"\A---.*?\n---\s*\n", re.S)
_CARD_META_LINE = re.compile(r"^(memory_schema|scope):.*$", re.M)


@dataclass(frozen=True, slots=True)
class PmiGateResult:
    status: str
    scores: dict[str, float]
    max_pmi: float | None
    threshold: float
    latency_seconds: float
    scored_count: int
    cached_neutral_count: int

    @property
    def skip_judge(self) -> bool:
        return self.status == "skipped"

    def public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "max_pmi": self.max_pmi,
            "threshold": self.threshold,
            "latency_seconds": self.latency_seconds,
            "scored_count": self.scored_count,
            "cached_neutral_count": self.cached_neutral_count,
            "scores": {
                candidate_id: round(value, 4) for candidate_id, value in self.scores.items()
            },
        }


class PmiJudgeGate:
    def __init__(
        self,
        runner: Any,
        tokenizer: Any,
        *,
        threshold: float = 0.0,
        head_tokens: int = 200,
        max_candidates: int = 16,
        neutral_cache_size: int = 512,
        timeout_seconds: float = 60.0,
    ) -> None:
        if head_tokens < 8 or max_candidates < 1 or timeout_seconds <= 0:
            raise ValueError("PMI gate limits must be positive")
        if not math.isfinite(threshold):
            raise ValueError("PMI gate threshold must be finite")
        self.runner = runner
        self.tokenizer = tokenizer
        self.threshold = float(threshold)
        self.head_tokens = int(head_tokens)
        self.max_candidates = int(max_candidates)
        self.timeout_seconds = float(timeout_seconds)
        self._neutral_prefix: tuple[int, ...] | None = None
        self._neutral_cache: OrderedDict[str, float] = OrderedDict()
        self._neutral_cache_size = int(neutral_cache_size)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ rendering
    @staticmethod
    def rule_card_head(content: str) -> str:
        body = _FRONT_MATTER.sub("", str(content or ""))
        body = _CARD_META_LINE.sub("", body)
        return re.sub(r"\n{3,}", "\n\n", body).strip()

    def _prefix_ids(self, question: str) -> tuple[int, ...]:
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(question)},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tuple(
            int(token)
            for token in self.tokenizer.encode(text + _LEAD_IN, add_special_tokens=False)
        )

    def _head_ids(self, candidate: KnowledgeCandidate) -> tuple[int, ...]:
        head = self.rule_card_head(candidate.normalized_reference_content)
        return tuple(
            int(token)
            for token in self.tokenizer.encode(head, add_special_tokens=False)[
                : self.head_tokens
            ]
        )

    # ------------------------------------------------------------ scoring
    async def evaluate(
        self,
        *,
        parent_request_id: str,
        question: str,
        candidates: Iterable[KnowledgeCandidate],
    ) -> PmiGateResult:
        started = time.perf_counter()
        candidate_list = tuple(candidates)[: self.max_candidates]
        heads = {
            candidate.candidate_id: self._head_ids(candidate)
            for candidate in candidate_list
        }
        scorable = tuple(c for c in candidate_list if heads[c.candidate_id])
        if not scorable:
            return PmiGateResult(
                "not_run", {}, None, self.threshold, 0.0, 0, len(self._neutral_cache)
            )
        if self._neutral_prefix is None:
            self._neutral_prefix = self._prefix_ids(_NEUTRAL_QUESTION)
        question_prefix = self._prefix_ids(question)

        inputs: list[tuple[int, ...]] = []
        starts: list[int] = []
        labels: list[tuple[str, str]] = []  # (candidate_id, "cond" | "neutral")
        async with self._lock:
            neutral_missing = [
                c for c in scorable if self._neutral_key(c) not in self._neutral_cache
            ]
        for candidate in scorable:
            inputs.append(question_prefix + heads[candidate.candidate_id])
            starts.append(len(question_prefix))
            labels.append((candidate.candidate_id, "cond"))
        for candidate in neutral_missing:
            inputs.append(self._neutral_prefix + heads[candidate.candidate_id])
            starts.append(len(self._neutral_prefix))
            labels.append((candidate.candidate_id, "neutral"))

        shared_prefix_key = "qwen-exo:v1:pmi-gate:" + stable_digest(
            parent_request_id, question
        )[:24]
        deadline = time.monotonic() + self.timeout_seconds
        jobs = tuple(
            InternalJob(
                parent_request_id=str(parent_request_id),
                turn_id=f"{parent_request_id}:pmi-gate",
                job_id="qwen-exo-pmi-" + stable_digest(parent_request_id, question, i)[:32],
                job_type=InternalJobType.CAUSAL_REPLAY,
                priority=-12,
                shared_prefix_key=shared_prefix_key,
                token_budget=1,
                state_budget_bytes=0,
                deadline_monotonic=deadline,
                cancellation_token=CancellationToken(
                    f"cancel:{parent_request_id}:pmi-gate:{i}"
                ),
                telemetry_correlation_id=f"{parent_request_id}:pmi-gate",
                max_fanout=len(inputs),
            )
            for i in range(len(inputs))
        )
        results = await self.runner.run_score_batch(
            jobs,
            inputs,
            starts,
            {"temperature": 0, "top_p": 1, "top_k": 1, "skip_special_tokens": True},
        )
        by_label: dict[tuple[str, str], float] = {}
        for label, result in zip(labels, results):
            logprobs = tuple(result.token_logprobs)
            if not logprobs:
                continue
            by_label[label] = sum(logprobs) / len(logprobs)

        scores: dict[str, float] = {}
        async with self._lock:
            for candidate in scorable:
                key = self._neutral_key(candidate)
                neutral = by_label.get((candidate.candidate_id, "neutral"))
                if neutral is not None:
                    self._neutral_cache[key] = neutral
                    self._neutral_cache.move_to_end(key)
                    while len(self._neutral_cache) > self._neutral_cache_size:
                        self._neutral_cache.popitem(last=False)
                neutral = self._neutral_cache.get(key)
                cond = by_label.get((candidate.candidate_id, "cond"))
                if cond is None or neutral is None:
                    continue
                scores[candidate.candidate_id] = cond - neutral
            cached = len(self._neutral_cache)
        if not scores:
            return PmiGateResult(
                "not_run",
                {},
                None,
                self.threshold,
                time.perf_counter() - started,
                0,
                cached,
            )
        max_pmi = max(scores.values())
        status = "skipped" if max_pmi <= self.threshold else "passed"
        return PmiGateResult(
            status,
            scores,
            max_pmi,
            self.threshold,
            time.perf_counter() - started,
            len(scores),
            cached,
        )

    def _neutral_key(self, candidate: KnowledgeCandidate) -> str:
        return stable_digest(
            "pmi-neutral-v1", candidate.reference_digest, str(self.head_tokens)
        )


__all__ = ["PmiGateResult", "PmiJudgeGate"]
