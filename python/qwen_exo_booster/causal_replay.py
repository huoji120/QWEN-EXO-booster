from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

from qwen_exo_booster.contracts import (
    CancellationToken,
    EligibilityDecision,
    EligibilityStatus,
    InternalJob,
    InternalJobType,
    stable_digest,
)
from qwen_exo_booster.internal_jobs import InternalJobRunner, InternalScoreResult
from qwen_exo_booster.knowledge import KnowledgeCandidate
from qwen_exo_booster.observer import MidThinkEvent
from qwen_exo_booster.telemetry import TelemetryStore

_MEMORY_QUALIFIER = (
    "Unverified candidate reference. It may be incomplete, wrong, or from a "
    "different version. Verify exact identifiers before use:\n"
)


@dataclass(frozen=True, slots=True)
class ReplayLoss:
    candidate_id: str | None
    document_id: str | None
    nll: float
    gain_vs_baseline: float
    observed_token_kl: float
    observation_tokens: int
    source_positions: tuple[int, ...]
    decision_id: str | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "document_id": self.document_id,
            "nll": self.nll,
            "gain_vs_baseline": self.gain_vs_baseline,
            "kl_from_baseline": self.observed_token_kl,
            "kl_method": "selected_token_bernoulli",
            "observation_tokens": self.observation_tokens,
            "source_positions": list(self.source_positions),
            "decision_id": self.decision_id,
        }


@dataclass(frozen=True, slots=True)
class CausalReplayResult:
    parent_request_id: str
    event_id: str
    decision: str
    maybe_decision: str
    winner_candidate_id: str | None
    winner_document_id: str | None
    winner_gain: float | None
    winner_kl: float | None
    scheduled_next_turn: bool
    losses: tuple[ReplayLoss, ...]
    latency_seconds: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "replay_decision": self.decision,
            "maybe_gate_decision": self.maybe_decision,
            "winner_candidate_id": self.winner_candidate_id,
            "winner_document_id": self.winner_document_id,
            "winner_gain": self.winner_gain,
            "winner_kl": self.winner_kl,
            "scheduled_next_turn": self.scheduled_next_turn,
            "losses": [loss.public_dict() for loss in self.losses],
            "latency_seconds": self.latency_seconds,
        }


class CausalReplayService:
    """Counterfactually scores real future reasoning in scheduler child jobs."""

    def __init__(
        self,
        runner: InternalJobRunner,
        tokenizer: Any,
        telemetry: TelemetryStore,
        *,
        observation_tokens: int = 8,
        prefix_tokens: int = 1024,
        max_candidates: int = 2,
        reference_tokens: int = 128,
        minimum_gain: float = 0.02,
        switch_margin: float = 0.05,
        maybe_kl_cap: float = 4.0,
        timeout_seconds: float = 60.0,
    ):
        if (
            observation_tokens < 2
            or prefix_tokens < 1
            or max_candidates < 1
            or reference_tokens < 1
            or minimum_gain < 0
            or switch_margin < 0
            or maybe_kl_cap < 0
            or timeout_seconds <= 0
        ):
            raise ValueError("Invalid causal replay limits")
        self.runner = runner
        self.tokenizer = tokenizer
        self.telemetry = telemetry
        self.observation_tokens = int(observation_tokens)
        self.prefix_tokens = int(prefix_tokens)
        self.max_candidates = int(max_candidates)
        self.reference_tokens = int(reference_tokens)
        self.minimum_gain = float(minimum_gain)
        self.switch_margin = float(switch_margin)
        self.maybe_kl_cap = float(maybe_kl_cap)
        self.timeout_seconds = float(timeout_seconds)

    async def evaluate(
        self,
        *,
        parent_request_id: str,
        event: MidThinkEvent,
        prompt_ids: Iterable[int],
        output_ids: Iterable[int],
        candidates: Iterable[KnowledgeCandidate],
        decisions: Iterable[EligibilityDecision],
        active_candidate_id: str | None = None,
    ) -> CausalReplayResult:
        started = time.perf_counter()
        prompt = tuple(int(token) for token in prompt_ids)
        output = tuple(int(token) for token in output_ids)
        candidate_list = tuple(candidates)[: self.max_candidates]
        decision_map = {decision.candidate_id: decision for decision in decisions}
        eligible: list[tuple[KnowledgeCandidate, EligibilityDecision]] = []
        for candidate in candidate_list:
            decision = decision_map.get(candidate.candidate_id)
            if (
                decision is None
                or decision.status is not EligibilityStatus.ELIGIBLE
                or decision.parent_request_id != str(parent_request_id)
                or decision.reference_digest
                != stable_digest(candidate.reference_content)
            ):
                continue
            eligible.append((candidate, decision))
        prefix_end = int(event.generation_token_index)
        observation = output[prefix_end : prefix_end + self.observation_tokens]
        if len(observation) < self.observation_tokens:
            return self._completed(
                self._rejected(
                    parent_request_id,
                    event.event_id,
                    "insufficient_future_observation",
                    started,
                )
            )
        if not eligible:
            return self._completed(
                self._rejected(
                    parent_request_id,
                    event.event_id,
                    "reject_no_semantic_candidate",
                    started,
                )
            )

        causal_prefix = (prompt + output[:prefix_end])[-self.prefix_tokens :]
        if not causal_prefix:
            return self._completed(
                self._rejected(
                    parent_request_id,
                    event.event_id,
                    "reject_empty_prefix",
                    started,
                )
            )
        branch_ids: list[tuple[int, ...]] = [causal_prefix + observation]
        label_starts = [len(causal_prefix)]
        branch_candidates: list[
            tuple[KnowledgeCandidate, EligibilityDecision] | None
        ] = [None]
        branch_native_prefixes = [None]
        for candidate, decision in eligible:
            native_prefix = candidate.native_prefix
            if native_prefix is not None:
                branch_ids.append(native_prefix.token_ids + causal_prefix + observation)
                label_starts.append(len(native_prefix.token_ids) + len(causal_prefix))
            else:
                memory_ids = self._candidate_memory_ids(candidate)
                if not memory_ids:
                    continue
                branch_ids.append(causal_prefix + memory_ids + observation)
                label_starts.append(len(causal_prefix) + len(memory_ids))
            branch_candidates.append((candidate, decision))
            branch_native_prefixes.append(native_prefix)
        if len(branch_ids) == 1:
            return self._completed(
                self._rejected(
                    parent_request_id,
                    event.event_id,
                    "reject_empty_candidate_state",
                    started,
                )
            )

        deadline = time.monotonic() + self.timeout_seconds
        shared_prefix_key = (
            "qwen-exo:v1:causal-replay:"
            + stable_digest(parent_request_id, event.event_id)[:24]
        )
        custom_params_per_job = tuple(
            (
                {"qwen_exo_native_bank_selection": native_prefix.scheduler_payload()}
                if native_prefix is not None
                else {}
            )
            for native_prefix in branch_native_prefixes
        )
        extra_keys = tuple(
            (
                native_prefix.radix_namespace
                if native_prefix is not None
                else shared_prefix_key
            )
            for native_prefix in branch_native_prefixes
        )
        jobs = tuple(
            InternalJob(
                parent_request_id=str(parent_request_id),
                turn_id=f"{parent_request_id}:replay:{event.event_id}",
                job_id=(
                    "qwen-exo-replay-"
                    + stable_digest(parent_request_id, event.event_id, index)[:32]
                ),
                job_type=InternalJobType.CAUSAL_REPLAY,
                priority=-12,
                shared_prefix_key=shared_prefix_key,
                token_budget=1,
                state_budget_bytes=0,
                deadline_monotonic=deadline,
                cancellation_token=CancellationToken(
                    f"cancel:{parent_request_id}:replay:{index}"
                ),
                telemetry_correlation_id=f"{parent_request_id}:replay",
                max_fanout=len(branch_ids),
            )
            for index in range(len(branch_ids))
        )
        runner_overrides = (
            {
                "custom_params_per_job": custom_params_per_job,
                "extra_keys": extra_keys,
            }
            if any(item is not None for item in branch_native_prefixes)
            else {}
        )
        try:
            scores = await self.runner.run_score_batch(
                jobs,
                branch_ids,
                label_starts,
                {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "skip_special_tokens": True,
                },
                **runner_overrides,
            )
        except Exception as exc:
            self.telemetry.emit(
                str(parent_request_id),
                "causal_replay.failed_closed",
                {
                    "event_id": event.event_id,
                    "error_type": type(exc).__name__,
                },
            )
            return self._completed(
                self._rejected(
                    parent_request_id,
                    event.event_id,
                    f"failed_closed:{type(exc).__name__}",
                    started,
                )
            )

        baseline = scores[0]
        losses: list[ReplayLoss] = [
            ReplayLoss(
                candidate_id=None,
                document_id=None,
                nll=baseline.mean_nll,
                gain_vs_baseline=0.0,
                observed_token_kl=0.0,
                observation_tokens=len(baseline.token_logprobs),
                source_positions=(),
                decision_id=None,
            )
        ]
        for score, candidate_entry in zip(scores[1:], branch_candidates[1:]):
            if candidate_entry is None:
                continue
            candidate, eligibility = candidate_entry
            losses.append(
                ReplayLoss(
                    candidate_id=candidate.candidate_id,
                    document_id=candidate.document_id,
                    nll=score.mean_nll,
                    gain_vs_baseline=baseline.mean_nll - score.mean_nll,
                    observed_token_kl=self._observed_token_kl(score, baseline),
                    observation_tokens=min(
                        len(score.token_logprobs), len(baseline.token_logprobs)
                    ),
                    source_positions=candidate.source_positions,
                    decision_id=eligibility.decision_id,
                )
            )
        contenders = losses[1:]
        winner = min(contenders, key=lambda item: item.nll) if contenders else None
        if winner is None:
            replay_decision = "reject_no_challenger"
        elif winner.gain_vs_baseline < self.minimum_gain:
            replay_decision = "reject_insufficient_gain"
        elif (
            active_candidate_id is not None
            and winner.candidate_id != active_candidate_id
            and winner.gain_vs_baseline < self.switch_margin
        ):
            replay_decision = "reject_switch_margin"
        else:
            replay_decision = "shadow_would_switch"

        if replay_decision != "shadow_would_switch" or winner is None:
            maybe_decision = "not_compiled"
            scheduled = False
        elif winner.observed_token_kl > self.maybe_kl_cap:
            maybe_decision = "reject_maybe_kl"
            scheduled = False
        else:
            maybe_decision = "admit_maybe"
            scheduled = True
        result = CausalReplayResult(
            parent_request_id=str(parent_request_id),
            event_id=event.event_id,
            decision=replay_decision,
            maybe_decision=maybe_decision,
            winner_candidate_id=(winner.candidate_id if winner is not None else None),
            winner_document_id=(winner.document_id if winner is not None else None),
            winner_gain=(winner.gain_vs_baseline if winner is not None else None),
            winner_kl=(winner.observed_token_kl if winner is not None else None),
            scheduled_next_turn=scheduled,
            losses=tuple(losses),
            latency_seconds=time.perf_counter() - started,
        )
        return self._completed(result)

    def _completed(self, result: CausalReplayResult) -> CausalReplayResult:
        self.telemetry.emit(
            result.parent_request_id,
            "causal_replay.completed",
            result.public_dict(),
        )
        return result

    def _candidate_memory_ids(self, candidate: KnowledgeCandidate) -> tuple[int, ...]:
        reference_ids = list(
            self.tokenizer.encode(
                candidate.normalized_reference_content,
                add_special_tokens=False,
            )
        )
        selected = [
            reference_ids[position]
            for position in candidate.source_positions
            if 0 <= int(position) < len(reference_ids)
        ]
        if not selected:
            selected = reference_ids[: self.reference_tokens]
        selected = selected[: self.reference_tokens]
        qualifier = list(
            self.tokenizer.encode(_MEMORY_QUALIFIER, add_special_tokens=False)
        )
        return tuple(qualifier + selected)

    @staticmethod
    def _observed_token_kl(
        candidate: InternalScoreResult, baseline: InternalScoreResult
    ) -> float:
        count = min(len(candidate.token_logprobs), len(baseline.token_logprobs))
        if count == 0:
            return float("inf")
        total = 0.0
        epsilon = 1e-7
        for candidate_logp, baseline_logp in zip(
            candidate.token_logprobs[:count], baseline.token_logprobs[:count]
        ):
            p = min(max(math.exp(min(candidate_logp, 0.0)), epsilon), 1 - epsilon)
            q = min(max(math.exp(min(baseline_logp, 0.0)), epsilon), 1 - epsilon)
            total += p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        return total / count

    @staticmethod
    def _rejected(
        parent_request_id: str,
        event_id: str,
        decision: str,
        started: float,
    ) -> CausalReplayResult:
        return CausalReplayResult(
            parent_request_id=str(parent_request_id),
            event_id=str(event_id),
            decision=str(decision),
            maybe_decision="not_compiled",
            winner_candidate_id=None,
            winner_document_id=None,
            winner_gain=None,
            winner_kl=None,
            scheduled_next_turn=False,
            losses=(),
            latency_seconds=time.perf_counter() - started,
        )
