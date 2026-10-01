from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from qwen_exo_booster.contracts import ContractViolation, InternalJob, InternalJobType
from qwen_exo_booster.task_scheduling import current_internal_work_class

_DFLASH_ELIGIBLE_JOB_TYPES = frozenset(
    {
        InternalJobType.RESPONSE_COMPACTION,
        InternalJobType.SELF_ANSWER,
        InternalJobType.REFLECTION_MEMORY,
    }
)
_DFLASH_STRUCTURED_KEYS = ("json_schema", "regex", "ebnf", "structural_tag")


def _internal_custom_params(
    job: InternalJob,
    base_custom: dict[str, Any],
    override: dict[str, Any],
    sampling_params: dict[str, Any],
    *,
    allow_dflash: bool,
) -> dict[str, Any]:
    custom = {
        **base_custom,
        **override,
        "qwen_exo_kind": "internal",
        "qwen_exo_job_type": job.job_type.value,
        "qwen_exo_parent_request_id": job.parent_request_id,
        "qwen_exo_state_budget_bytes": job.state_budget_bytes,
        "qwen_exo_work_class": current_internal_work_class(job.job_type),
    }
    force_target_only = custom.get("qwen_exo_dflash") == "target_only"
    requested = custom.get("qwen_exo_dflash") == "eligible"
    eligible = (
        not force_target_only
        and allow_dflash
        and (job.job_type in _DFLASH_ELIGIBLE_JOB_TYPES or requested)
        and job.token_budget >= 32
        and not any(
            sampling_params.get(key) is not None for key in _DFLASH_STRUCTURED_KEYS
        )
    )
    if eligible:
        custom["qwen_exo_dflash"] = "eligible"
    elif custom.get("qwen_exo_dflash") != "target_only":
        custom.pop("qwen_exo_dflash", None)
    return custom


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


@dataclass(frozen=True, slots=True)
class InternalOptionScoreResult:
    job: InternalJob
    option_logprobs: tuple[float, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float


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
        self._lane_limits = {
            "foreground": self.max_fanout,
            "reflection": 1,
            "maintenance": 3,
        }
        self._lane_active = dict.fromkeys(self._lane_limits, 0)
        self._active_lanes: dict[tuple[str, str], str] = {}

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
        self._check_liveness(parent_request_id, job_list)
        lane_limit = self._lane_limits[current_internal_work_class(job_list[0].job_type)]
        if len(job_list) > lane_limit:
            results = []
            normalized_prompts = next(iter(prompt_kwargs.values()))
            for start in range(0, len(job_list), lane_limit):
                end = start + lane_limit
                results.extend(
                    await self.run_batch(
                        job_list[start:end],
                        normalized_prompts[start:end],
                        sampling_params,
                        custom_params_per_job=per_job_custom[start:end],
                        extra_keys=per_job_extra_keys[start:end],
                    )
                )
            return tuple(results)

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
                        "custom_params": _internal_custom_params(
                            job,
                            sampling_params.get("custom_params") or {},
                            per_job_custom[index],
                            sampling_params,
                            allow_dflash=True,
                        ),
                    }
                    for index, job in enumerate(job_list)
                ],
                return_logprob=False,
                stream=False,
                priority=self._priority(job_list[0]),
                extra_key=list(per_job_extra_keys),
                no_logs=True,
                custom_labels={
                    "qwen_exo_visibility": "internal",
                    "qwen_exo_job_type": job_list[0].job_type.value,
                },
            )
            deadline = self._earliest_deadline(job_list)
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise asyncio.TimeoutError(
                    "Internal job deadline elapsed before dispatch"
                )
            outputs = await asyncio.wait_for(self._collect(request), timeout=timeout)
            self._check_liveness(parent_request_id, job_list)
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
            await asyncio.shield(self._release(parent_request_id, job_list))

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
        self._check_liveness(parent_request_id, job_list)
        lane_limit = self._lane_limits[current_internal_work_class(job_list[0].job_type)]
        if len(job_list) > lane_limit:
            results = []
            for start in range(0, len(job_list), lane_limit):
                end = start + lane_limit
                results.extend(
                    await self.run_score_batch(
                        job_list[start:end],
                        input_list[start:end],
                        start_list[start:end],
                        sampling_params,
                        custom_params_per_job=per_job_custom[start:end],
                        extra_keys=per_job_extra_keys[start:end],
                    )
                )
            return tuple(results)

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
                        "custom_params": _internal_custom_params(
                            job,
                            base_sampling.get("custom_params") or {},
                            per_job_custom[index],
                            base_sampling,
                            allow_dflash=False,
                        ),
                    }
                    for index, job in enumerate(job_list)
                ],
                return_logprob=True,
                logprob_start_len=list(start_list),
                token_ids_logprob=[[int(tokens[-1])] for tokens in input_list],
                top_logprobs_num=0,
                stream=False,
                priority=self._priority(job_list[0]),
                extra_key=list(per_job_extra_keys),
                no_logs=True,
                custom_labels={
                    "qwen_exo_visibility": "internal",
                    "qwen_exo_job_type": job_list[0].job_type.value,
                },
            )
            deadline = self._earliest_deadline(job_list)
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise asyncio.TimeoutError(
                    "Replay score deadline elapsed before dispatch"
                )
            outputs = await asyncio.wait_for(self._collect(request), timeout=timeout)
            self._check_liveness(parent_request_id, job_list)
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
            await asyncio.shield(self._release(parent_request_id, job_list))

    async def run_option_score_batch(
        self,
        jobs: Iterable[InternalJob],
        input_ids: Iterable[Iterable[int]],
        option_token_ids: Iterable[int],
    ) -> tuple[InternalOptionScoreResult, ...]:
        """Read two next-token scores from one target-only prefill per job."""
        job_list = tuple(jobs)
        prefixes = tuple(tuple(tokens) for tokens in input_ids)
        options = tuple(option_token_ids)
        if not job_list or len(job_list) != len(prefixes):
            raise ContractViolation("Option score batch requires one prefix per job")
        if (
            len(options) != 2
            or any(type(token) is not int or token < 0 for token in options)
            or len(set(options)) != 2
        ):
            raise ContractViolation("Option scores require two distinct token IDs")
        if any(
            not tokens or any(type(token) is not int or token < 0 for token in tokens)
            for tokens in prefixes
        ):
            raise ContractViolation("Option score prefixes require non-empty token IDs")
        if len({job.job_id for job in job_list}) != len(job_list):
            raise ContractViolation("Option score job IDs must be unique")
        parents = {job.parent_request_id for job in job_list}
        if len(parents) != 1:
            raise ContractViolation("Option score batch must have one parent")
        parent_request_id = next(iter(parents))
        if len(job_list) > self.max_fanout or len(job_list) > min(
            job.max_fanout for job in job_list
        ):
            raise ContractViolation("Option score batch exceeds configured fanout")
        if sum(job.token_budget for job in job_list) > self.max_tokens_per_parent:
            raise ContractViolation("Option score batch exceeds the token reserve")
        if len({job.job_type for job in job_list}) != 1:
            raise ContractViolation("Option score batch must have one job type")
        if len({job.shared_prefix_key for job in job_list}) != 1:
            raise ContractViolation("Option score batch must share one prefix key")
        self._validate_cache_namespace(job_list[0].shared_prefix_key)
        self._check_liveness(parent_request_id, job_list)
        lane_limit = self._lane_limits[current_internal_work_class(job_list[0].job_type)]
        if len(job_list) > lane_limit:
            results = []
            for start in range(0, len(job_list), lane_limit):
                end = start + lane_limit
                results.extend(
                    await self.run_option_score_batch(
                        job_list[start:end], prefixes[start:end], options
                    )
                )
            return tuple(results)

        await self._reserve(parent_request_id, job_list)
        started = time.perf_counter()
        try:
            # SGLang prepends a None input-logprob row and drops the final
            # distribution. Include the last prefix token and one fixed sentinel
            # in the suffix so row 1 is exactly P(option | full prefix).
            request = self._make_request(
                rid=[job.job_id for job in job_list],
                input_ids=[[*tokens, 0] for tokens in prefixes],
                sampling_params=[
                    {
                        "max_new_tokens": 0,
                        "temperature": 1,
                        "custom_params": _internal_custom_params(
                            job, {}, {"qwen_exo_dflash": "target_only"}, {},
                            allow_dflash=False,
                        ),
                    }
                    for job in job_list
                ],
                return_logprob=True,
                logprob_start_len=[len(tokens) - 1 for tokens in prefixes],
                token_ids_logprob=[list(options) for _ in job_list],
                top_logprobs_num=0,
                stream=False,
                priority=self._priority(job_list[0]),
                extra_key=[job.shared_prefix_key for job in job_list],
                no_logs=True,
                custom_labels={
                    "qwen_exo_visibility": "internal",
                    "qwen_exo_job_type": job_list[0].job_type.value,
                },
            )
            deadline = self._earliest_deadline(job_list)
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise asyncio.TimeoutError("Option score deadline elapsed before dispatch")
            outputs = await asyncio.wait_for(self._collect(request), timeout=timeout)
            self._check_liveness(parent_request_id, job_list)
            if len(outputs) != len(job_list):
                raise RuntimeError("Option score batch returned an incorrect result count")
            elapsed = time.perf_counter() - started
            return tuple(
                self._option_score_result(job, tokens, options, output, elapsed)
                for job, tokens, output in zip(job_list, prefixes, outputs)
            )
        except (asyncio.CancelledError, Exception):
            self._abort_jobs(job_list)
            raise
        finally:
            await asyncio.shield(self._release(parent_request_id, job_list))

    @staticmethod
    def _option_score_result(
        job: InternalJob,
        prefix: tuple[int, ...],
        options: tuple[int, ...],
        output: dict[str, Any],
        latency_seconds: float,
    ) -> InternalOptionScoreResult:
        def sequence(value: Any, size: int) -> bool:
            return isinstance(value, (list, tuple)) and len(value) == size

        if not isinstance(output, dict) or not isinstance(output.get("meta_info"), dict):
            raise RuntimeError("Option score result has no metadata")
        meta = output["meta_info"]
        if meta.get("id") != job.job_id:
            raise RuntimeError("Option score result job identity is misaligned")
        finish_reason = meta.get("finish_reason")
        if (
            not isinstance(finish_reason, dict)
            or finish_reason.get("type") != "length"
            or type(finish_reason.get("length")) is not int
            or finish_reason["length"] != 0
        ):
            raise RuntimeError("Option score result did not finish a zero-token prefill")
        if (
            type(meta.get("completion_tokens")) is not int
            or meta["completion_tokens"] != 0
            or output.get("output_ids")
            or output.get("text")
        ):
            raise RuntimeError("Option score result contains generated tokens")
        if (
            type(meta.get("prompt_tokens")) is not int
            or meta["prompt_tokens"] != len(prefix) + 1
        ):
            raise RuntimeError("Option score result prompt length is misaligned")
        token_rows = meta.get("input_token_logprobs")
        if (
            not sequence(token_rows, 2)
            or not all(sequence(row, 3) for row in token_rows)
            or token_rows[0][0] is not None
            or type(token_rows[0][1]) is not int
            or token_rows[0][1] != prefix[-1]
            or type(token_rows[1][1]) is not int
            or token_rows[1][1] != 0
        ):
            raise RuntimeError("Option score result input rows are misaligned")
        rows = meta.get("input_token_ids_logprobs")
        if not sequence(rows, 2) or rows[0] is not None or not sequence(rows[1], 2):
            raise RuntimeError("Option score result option rows are misaligned")
        scores = []
        for token, entry in zip(options, rows[1]):
            if (
                not sequence(entry, 3)
                or type(entry[1]) is not int
                or entry[1] != token
                or type(entry[0]) not in (int, float)
                or not math.isfinite(entry[0])
            ):
                raise RuntimeError("Option score result contains invalid option logprobs")
            scores.append(float(entry[0]))
        return InternalOptionScoreResult(
            job=job,
            option_logprobs=tuple(scores),
            prompt_tokens=meta["prompt_tokens"],
            completion_tokens=0,
            latency_seconds=latency_seconds,
        )

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

    @staticmethod
    def _priority(job: InternalJob) -> int:
        return 0 if current_internal_work_class(job.job_type) == "foreground" else -100

    def _check_liveness(
        self, parent_request_id: str, jobs: tuple[InternalJob, ...]
    ) -> None:
        if parent_request_id in self._cancelled_parents or any(
            job.cancellation_token.cancelled for job in jobs
        ):
            raise asyncio.CancelledError("Internal job was cancelled")
        deadline = self._earliest_deadline(jobs)
        if deadline is not None and deadline <= time.monotonic():
            raise asyncio.TimeoutError("Internal job deadline elapsed")

    async def _reserve(
        self, parent_request_id: str, jobs: tuple[InternalJob, ...]
    ) -> None:
        lane = current_internal_work_class(jobs[0].job_type)
        requested_tokens = sum(job.token_budget for job in jobs)
        job_ids = {job.job_id for job in jobs}
        if len(job_ids) != len(jobs):
            raise ContractViolation("Internal job IDs must be unique")
        async with self._capacity_changed:
            deadline = self._earliest_deadline(jobs)
            while True:
                self._check_liveness(parent_request_id, jobs)
                reserved_tokens = self._reserved_tokens.get(parent_request_id, 0)
                if reserved_tokens + requested_tokens > self.max_tokens_per_parent:
                    raise ContractViolation(
                        "Internal jobs exceed the parent cumulative token reserve"
                    )
                if self._lane_active[lane] + len(jobs) <= self._lane_limits[lane]:
                    break
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise asyncio.TimeoutError(
                        "Internal job deadline elapsed during lane admission"
                    )
                await asyncio.wait_for(self._capacity_changed.wait(), timeout=remaining)
            active = self._active.setdefault(parent_request_id, set())
            duplicate = active.intersection(job_ids)
            if duplicate:
                raise ContractViolation(
                    f"Internal job IDs are already active: {duplicate}"
                )
            active.update(job_ids)
            for job_id in job_ids:
                self._active_lanes[parent_request_id, job_id] = lane
            self._lane_active[lane] += len(jobs)
            self._reserved_tokens[parent_request_id] = reserved_tokens + requested_tokens

    async def _release(
        self, parent_request_id: str, jobs: tuple[InternalJob, ...]
    ) -> None:
        async with self._capacity_changed:
            active = self._active.get(parent_request_id)
            for job in jobs:
                lane = self._active_lanes.pop((parent_request_id, job.job_id), None)
                if lane is not None:
                    self._lane_active[lane] -= 1
                if active is not None:
                    active.discard(job.job_id)
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

    @staticmethod
    def _earliest_deadline(jobs: tuple[InternalJob, ...]) -> float | None:
        return min(
            (
                job.deadline_monotonic
                for job in jobs
                if job.deadline_monotonic is not None
            ),
            default=None,
        )

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
        metadata = dict(meta)
        # Generation IDs are needed by diagnostic callers that must score the
        # exact sampled token rather than re-tokenizing its display text.
        if output.get("output_ids") is not None:
            metadata["output_ids"] = tuple(output.get("output_ids") or ())
        return InternalJobResult(
            job=job,
            text=str(output.get("text") or ""),
            prompt_tokens=int(meta.get("prompt_tokens") or 0),
            completion_tokens=int(completion_tokens),
            finish_reason=meta.get("finish_reason"),
            latency_seconds=latency_seconds,
            metadata=metadata,
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
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
        else:
            value = item
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
