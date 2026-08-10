from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from qwen_exo_booster.contracts import ContractViolation, InternalJob, InternalJobType


@dataclass(frozen=True, slots=True)
class InternalJobResult:
    job: InternalJob
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: object
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InternalScoreResult:
    job: InternalJob
    token_logprobs: tuple[float, ...]
    mean_nll: float
    prompt_tokens: int
    finish_reason: object
    latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InternalJobRunner:
    """Submits hidden child work directly to SGLang's tokenizer manager.

    No HTTP request is created. Child requests therefore share the normal
    scheduler, priority, cancellation, cache, and capacity accounting paths.
    """

    def __init__(
        self,
        tokenizer_manager: Any,
        *,
        max_fanout: int,
        max_tokens_per_parent: int,
        request_factory: Callable[..., Any] | None = None,
    ):
        if max_fanout < 1 or max_tokens_per_parent < 1:
            raise ValueError("Internal job limits must be positive")
        self.tokenizer_manager = tokenizer_manager
        self.max_fanout = int(max_fanout)
        self.max_tokens_per_parent = int(max_tokens_per_parent)
        self._request_factory = request_factory
        self._active: dict[str, set[str]] = {}
        self._cancelled_parents: set[str] = set()
        self._reserved_tokens: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._capacity_changed = asyncio.Condition(self._lock)

    async def run_batch(
        self,
        jobs: Iterable[InternalJob],
        prompts: Iterable[str | Iterable[int]],
        sampling_params: dict[str, Any],
        *,
        custom_params_per_job: Iterable[dict[str, Any]] | None = None,
        extra_keys: Iterable[str] | None = None,
    ) -> tuple[InternalJobResult, ...]:
        job_list = tuple(jobs)
        raw_prompts = tuple(prompts)
        if not job_list or len(job_list) != len(raw_prompts):
            raise ContractViolation("Internal job batch requires one prompt per job")
        if all(isinstance(prompt, str) for prompt in raw_prompts):
            prompt_kwargs = {"text": [str(prompt) for prompt in raw_prompts]}
        elif all(not isinstance(prompt, str) for prompt in raw_prompts):
            prompt_kwargs = {
                "input_ids": [
                    [int(token) for token in prompt] for prompt in raw_prompts
                ]
            }
        else:
            raise ContractViolation(
                "Internal job prompts cannot mix text and token IDs"
            )
        per_job_custom = (
            tuple(dict(item) for item in custom_params_per_job)
            if custom_params_per_job is not None
            else ({},) * len(job_list)
        )
        per_job_extra_keys = (
            tuple(str(item) for item in extra_keys)
            if extra_keys is not None
            else tuple(job.shared_prefix_key for job in job_list)
        )
        if len(per_job_custom) != len(job_list) or len(per_job_extra_keys) != len(
            job_list
        ):
            raise ContractViolation("Internal job overrides require one value per job")
        parents = {job.parent_request_id for job in job_list}
        if len(parents) != 1:
            raise ContractViolation(
                "A shared-prefix internal batch must have one parent"
            )
        parent_request_id = next(iter(parents))
        if len(job_list) > self.max_fanout:
            raise ContractViolation("Internal job batch exceeds configured fanout")
        if len(job_list) > min(job.max_fanout for job in job_list):
            raise ContractViolation(
                "Internal job batch exceeds the parent fanout contract"
            )
        if sum(job.token_budget for job in job_list) > self.max_tokens_per_parent:
            raise ContractViolation(
                "Internal job batch exceeds the parent token reserve"
            )
        if len({job.job_type for job in job_list}) != 1:
            raise ContractViolation("An internal batch cannot mix job types")
        if len({job.shared_prefix_key for job in job_list}) != 1:
            raise ContractViolation("An internal batch must share one prefix key")
        self._validate_cache_namespace(job_list[0].shared_prefix_key)
        for extra_key in per_job_extra_keys:
            self._validate_cache_namespace(extra_key)
        if any(job.is_cancelled_or_expired() for job in job_list):
            raise asyncio.CancelledError(
                "Internal job was cancelled or expired before admission"
            )

        await self._reserve(parent_request_id, job_list)
        started = time.perf_counter()
        try:
            request = self._make_request(
                rid=[job.job_id for job in job_list],
                **prompt_kwargs,
                sampling_params=[
                    {
                        **sampling_params,
                        "max_new_tokens": job.token_budget,
                        "custom_params": {
                            **(sampling_params.get("custom_params") or {}),
                            **per_job_custom[index],
                            "qwen_exo_kind": "internal",
                            "qwen_exo_job_type": job.job_type.value,
                            "qwen_exo_parent_request_id": job.parent_request_id,
                            "qwen_exo_state_budget_bytes": job.state_budget_bytes,
                        },
                    }
                    for index, job in enumerate(job_list)
                ],
                return_logprob=False,
                stream=False,
                priority=min(job.priority for job in job_list),
                extra_key=list(per_job_extra_keys),
                no_logs=True,
                custom_labels={
                    "qwen_exo_visibility": "internal",
                    "qwen_exo_job_type": job_list[0].job_type.value,
                },
            )
            timeout = min(job.deadline_monotonic for job in job_list) - time.monotonic()
            if timeout <= 0:
                raise asyncio.TimeoutError(
                    "Internal job deadline elapsed before dispatch"
                )
            outputs = await asyncio.wait_for(self._collect(request), timeout=timeout)
            if len(outputs) != len(job_list):
                raise RuntimeError(
                    f"Internal batch returned {len(outputs)} results for {len(job_list)} jobs"
                )
            elapsed = time.perf_counter() - started
            return tuple(
                self._result(job, output, elapsed)
                for job, output in zip(job_list, outputs)
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self._abort_jobs(job_list)
            raise
        except Exception:
            self._abort_jobs(job_list)
            raise
        finally:
            await self._release(parent_request_id, job_list)

    async def run_score_batch(
        self,
        jobs: Iterable[InternalJob],
        input_ids: Iterable[Iterable[int]],
        label_starts: Iterable[int],
        sampling_params: dict[str, Any] | None = None,
        *,
        custom_params_per_job: Iterable[dict[str, Any]] | None = None,
        extra_keys: Iterable[str] | None = None,
    ) -> tuple[InternalScoreResult, ...]:
        job_list = tuple(jobs)
        input_list = tuple(tuple(int(token) for token in item) for item in input_ids)
        start_list = tuple(int(value) for value in label_starts)
        per_job_custom = (
            tuple(dict(item) for item in custom_params_per_job)
            if custom_params_per_job is not None
            else ({},) * len(job_list)
        )
        per_job_extra_keys = (
            tuple(str(item) for item in extra_keys)
            if extra_keys is not None
            else tuple(job.shared_prefix_key for job in job_list)
        )
        if (
            not job_list
            or len(job_list) != len(input_list)
            or len(job_list) != len(start_list)
            or len(job_list) != len(per_job_custom)
            or len(job_list) != len(per_job_extra_keys)
        ):
            raise ContractViolation(
                "Replay score batch requires one input and label start per job"
            )
        if any(
            not tokens or start < 1 or start >= len(tokens)
            for tokens, start in zip(input_list, start_list)
        ):
            raise ContractViolation(
                "Replay label ranges must be non-empty prompt suffixes"
            )
        parents = {job.parent_request_id for job in job_list}
        if len(parents) != 1:
            raise ContractViolation("A replay score batch must have one parent")
        parent_request_id = next(iter(parents))
        if len(job_list) > self.max_fanout or len(job_list) > min(
            job.max_fanout for job in job_list
        ):
            raise ContractViolation("Replay score batch exceeds configured fanout")
        if sum(job.token_budget for job in job_list) > self.max_tokens_per_parent:
            raise ContractViolation("Replay score batch exceeds the token reserve")
        if len({job.job_type for job in job_list}) != 1:
            raise ContractViolation("Replay score jobs must have one job type")
        if len({job.shared_prefix_key for job in job_list}) != 1:
            raise ContractViolation("Replay score jobs must share one prefix key")
        self._validate_cache_namespace(job_list[0].shared_prefix_key)
        for extra_key in per_job_extra_keys:
            self._validate_cache_namespace(extra_key)
        if any(job.is_cancelled_or_expired() for job in job_list):
            raise asyncio.CancelledError(
                "Replay score job was cancelled or expired before admission"
            )

        await self._reserve(parent_request_id, job_list)
        started = time.perf_counter()
        base_sampling = dict(sampling_params or {})
        try:
            request = self._make_request(
                rid=[job.job_id for job in job_list],
                input_ids=[list(tokens) for tokens in input_list],
                sampling_params=[
                    {
                        **base_sampling,
                        "max_new_tokens": (
                            0 if job.job_type is InternalJobType.BANK_INDEX else 1
                        ),
                        "custom_params": {
                            **(base_sampling.get("custom_params") or {}),
                            **per_job_custom[index],
                            "qwen_exo_kind": "internal",
                            "qwen_exo_job_type": job.job_type.value,
                            "qwen_exo_parent_request_id": job.parent_request_id,
                            "qwen_exo_state_budget_bytes": job.state_budget_bytes,
                        },
                    }
                    for index, job in enumerate(job_list)
                ],
                return_logprob=True,
                logprob_start_len=list(start_list),
                top_logprobs_num=0,
                stream=False,
                priority=min(job.priority for job in job_list),
                extra_key=list(per_job_extra_keys),
                no_logs=True,
                custom_labels={
                    "qwen_exo_visibility": "internal",
                    "qwen_exo_job_type": job_list[0].job_type.value,
                },
            )
            timeout = min(job.deadline_monotonic for job in job_list) - time.monotonic()
            if timeout <= 0:
                raise asyncio.TimeoutError(
                    "Replay score deadline elapsed before dispatch"
                )
            outputs = await asyncio.wait_for(self._collect(request), timeout=timeout)
            if len(outputs) != len(job_list):
                raise RuntimeError(
                    f"Replay score batch returned {len(outputs)} results for {len(job_list)} jobs"
                )
            elapsed = time.perf_counter() - started
            return tuple(
                self._score_result(job, output, elapsed)
                for job, output in zip(job_list, outputs)
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self._abort_jobs(job_list)
            raise
        except Exception:
            self._abort_jobs(job_list)
            raise
        finally:
            await self._release(parent_request_id, job_list)

    async def cancel_parent(self, parent_request_id: str) -> None:
        async with self._capacity_changed:
            self._cancelled_parents.add(parent_request_id)
            job_ids = tuple(self._active.get(parent_request_id, ()))
            self._capacity_changed.notify_all()
        for job_id in job_ids:
            self.tokenizer_manager.abort_request(job_id)

    async def finish_parent(self, parent_request_id: str) -> None:
        async with self._capacity_changed:
            self._active.pop(parent_request_id, None)
            self._reserved_tokens.pop(parent_request_id, None)
            self._cancelled_parents.discard(parent_request_id)
            self._capacity_changed.notify_all()

    async def _reserve(
        self, parent_request_id: str, jobs: tuple[InternalJob, ...]
    ) -> None:
        async with self._capacity_changed:
            if parent_request_id in self._cancelled_parents:
                raise asyncio.CancelledError("Parent request is cancelled")
            requested_tokens = sum(job.token_budget for job in jobs)
            reserved_tokens = self._reserved_tokens.get(parent_request_id, 0)
            if reserved_tokens + requested_tokens > self.max_tokens_per_parent:
                raise ContractViolation(
                    "Internal jobs exceed the parent cumulative token reserve"
                )
            deadline = min(job.deadline_monotonic for job in jobs)
            while (
                sum(len(active_jobs) for active_jobs in self._active.values())
                + len(jobs)
                > self.max_fanout
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        "Internal job deadline elapsed during global admission"
                    )
                try:
                    await asyncio.wait_for(
                        self._capacity_changed.wait(), timeout=remaining
                    )
                except asyncio.TimeoutError as exc:
                    raise asyncio.TimeoutError(
                        "Internal job deadline elapsed during global admission"
                    ) from exc
                if parent_request_id in self._cancelled_parents or any(
                    job.is_cancelled_or_expired() for job in jobs
                ):
                    raise asyncio.CancelledError(
                        "Internal job was cancelled while awaiting global admission"
                    )
            active = self._active.setdefault(parent_request_id, set())
            allowed_fanout = min(self.max_fanout, *(job.max_fanout for job in jobs))
            if len(active) + len(jobs) > allowed_fanout:
                raise ContractViolation("Parent already owns the maximum child fanout")
            duplicate = active.intersection(job.job_id for job in jobs)
            if duplicate:
                raise ContractViolation(
                    f"Internal job IDs are already active: {duplicate}"
                )
            active.update(job.job_id for job in jobs)
            self._reserved_tokens[parent_request_id] = (
                reserved_tokens + requested_tokens
            )

    async def _release(
        self, parent_request_id: str, jobs: tuple[InternalJob, ...]
    ) -> None:
        async with self._capacity_changed:
            active = self._active.get(parent_request_id)
            if active is None:
                return
            active.difference_update(job.job_id for job in jobs)
            if not active:
                self._active.pop(parent_request_id, None)
            self._capacity_changed.notify_all()

    async def _collect(self, request: Any) -> list[dict[str, Any]]:
        payloads = []
        async for payload in self.tokenizer_manager.generate_request(request, None):
            if isinstance(payload, list):
                payloads.extend(payload)
            else:
                payloads.append(payload)
        return payloads

    def _abort_jobs(self, jobs: tuple[InternalJob, ...]) -> None:
        for job in jobs:
            self.tokenizer_manager.abort_request(job.job_id)

    @staticmethod
    def _validate_cache_namespace(shared_prefix_key: str) -> None:
        key = str(shared_prefix_key)
        if not key.startswith("qwen-exo:v1:"):
            raise ContractViolation(
                "Internal jobs require a private qwen-exo:v1 cache namespace"
            )
        if key.startswith(
            (
                "qwen-exo:v1:external_memory:",
                "qwen-exo:v1:request_prefix:",
            )
        ):
            raise ContractViolation(
                "Internal jobs cannot enter a user-visible cache namespace"
            )

    def _make_request(self, **kwargs):
        if self._request_factory is not None:
            return self._request_factory(**kwargs)
        from sglang.srt.managers.io_struct import GenerateReqInput

        return GenerateReqInput(**kwargs)

    @staticmethod
    def _result(
        job: InternalJob, output: dict[str, Any], latency_seconds: float
    ) -> InternalJobResult:
        meta = output.get("meta_info") or {}
        completion_tokens = meta.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = len(output.get("output_ids") or ())
        return InternalJobResult(
            job=job,
            text=str(output.get("text") or ""),
            prompt_tokens=int(meta.get("prompt_tokens") or 0),
            completion_tokens=int(completion_tokens),
            finish_reason=meta.get("finish_reason"),
            latency_seconds=latency_seconds,
            metadata=dict(meta),
        )

    @classmethod
    def _score_result(
        cls, job: InternalJob, output: dict[str, Any], latency_seconds: float
    ) -> InternalScoreResult:
        meta = output.get("meta_info") or {}
        token_logprobs = tuple(
            value
            for item in (meta.get("input_token_logprobs") or ())
            if (value := cls._logprob_value(item)) is not None and math.isfinite(value)
        )
        if not token_logprobs:
            raise RuntimeError("Replay score result contains no finite label logprobs")
        return InternalScoreResult(
            job=job,
            token_logprobs=token_logprobs,
            mean_nll=-sum(token_logprobs) / len(token_logprobs),
            prompt_tokens=int(meta.get("prompt_tokens") or 0),
            finish_reason=meta.get("finish_reason"),
            latency_seconds=latency_seconds,
            metadata=dict(meta),
        )

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
