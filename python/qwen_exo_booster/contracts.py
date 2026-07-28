from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable


class ContractViolation(ValueError):
    """Raised when state would violate a QWEN-EXO runtime contract."""


class HybridLifecycleState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    CACHED = "cached"
    SUSPENDED = "suspended"
    EVICTED = "evicted"
    RELEASED = "released"


class HybridStateNamespace(str, Enum):
    REQUEST_PREFIX = "request_prefix"
    EXTERNAL_MEMORY = "external_memory"


class InternalJobType(str, Enum):
    REFERENCE_JUDGE = "reference_judge"
    CAPSULE_UPDATE = "capsule_update"
    RETRIEVAL_REFRESH = "retrieval_refresh"
    ADMISSION_PROBE = "admission_probe"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "true"
    INELIGIBLE = "false"
    INVALID = "invalid"


_ALLOWED_TRANSITIONS = {
    HybridLifecycleState.NEW: {
        HybridLifecycleState.ACTIVE,
        HybridLifecycleState.RELEASED,
    },
    HybridLifecycleState.ACTIVE: {
        HybridLifecycleState.CACHED,
        HybridLifecycleState.SUSPENDED,
        HybridLifecycleState.RELEASED,
    },
    HybridLifecycleState.CACHED: {
        HybridLifecycleState.ACTIVE,
        HybridLifecycleState.EVICTED,
        HybridLifecycleState.RELEASED,
    },
    HybridLifecycleState.SUSPENDED: {
        HybridLifecycleState.ACTIVE,
        HybridLifecycleState.EVICTED,
        HybridLifecycleState.RELEASED,
    },
    HybridLifecycleState.EVICTED: {
        HybridLifecycleState.ACTIVE,
        HybridLifecycleState.CACHED,
        HybridLifecycleState.RELEASED,
    },
    HybridLifecycleState.RELEASED: set(),
}


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_slots(name: str, values: Iterable[int]) -> tuple[int, ...]:
    slots = tuple(int(value) for value in values)
    if any(value < 0 for value in slots):
        raise ContractViolation(f"{name} cannot contain negative slot IDs")
    if len(slots) != len(set(slots)):
        raise ContractViolation(f"{name} cannot contain duplicate slot IDs")
    return slots


@dataclass(frozen=True, slots=True)
class HybridStateHandle:
    handle_id: str
    request_id: str
    prefix_identity: str
    boundary_fingerprint: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    tp_world_size: int
    tp_rank: int
    sequence_length: int
    namespace: HybridStateNamespace = HybridStateNamespace.REQUEST_PREFIX
    lifecycle: HybridLifecycleState = HybridLifecycleState.NEW
    full_kv_blocks: tuple[int, ...] = field(default_factory=tuple)
    recurrent_state_slots: tuple[int, ...] = field(default_factory=tuple)
    conv_state_slots: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        required_text = {
            "handle_id": self.handle_id,
            "request_id": self.request_id,
            "prefix_identity": self.prefix_identity,
            "boundary_fingerprint": self.boundary_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }
        missing = [name for name, value in required_text.items() if not str(value).strip()]
        if missing:
            raise ContractViolation(f"Missing hybrid state identity fields: {missing}")
        if self.tp_world_size < 1:
            raise ContractViolation("tp_world_size must be positive")
        if not 0 <= self.tp_rank < self.tp_world_size:
            raise ContractViolation("tp_rank must identify a rank in tp_world_size")
        if self.sequence_length < 0:
            raise ContractViolation("sequence_length cannot be negative")

        object.__setattr__(
            self,
            "full_kv_blocks",
            _validated_slots("full_kv_blocks", self.full_kv_blocks),
        )
        object.__setattr__(
            self,
            "recurrent_state_slots",
            _validated_slots("recurrent_state_slots", self.recurrent_state_slots),
        )
        object.__setattr__(
            self,
            "conv_state_slots",
            _validated_slots("conv_state_slots", self.conv_state_slots),
        )

        resident = self.lifecycle in {
            HybridLifecycleState.ACTIVE,
            HybridLifecycleState.CACHED,
            HybridLifecycleState.SUSPENDED,
        }
        if resident and not self.has_complete_hybrid_state:
            raise ContractViolation(
                "Resident hybrid state requires Full-Attention KV, recurrent, and conv slots"
            )
        if self.lifecycle in {
            HybridLifecycleState.EVICTED,
            HybridLifecycleState.RELEASED,
        } and self.has_any_component:
            raise ContractViolation(
                f"{self.lifecycle.value} hybrid state cannot retain GPU component slots"
            )

    @property
    def has_any_component(self) -> bool:
        return bool(
            self.full_kv_blocks
            or self.recurrent_state_slots
            or self.conv_state_slots
        )

    @property
    def has_complete_hybrid_state(self) -> bool:
        return bool(
            self.full_kv_blocks
            and self.recurrent_state_slots
            and self.conv_state_slots
        )

    def bind_components(
        self,
        *,
        full_kv_blocks: Iterable[int],
        recurrent_state_slots: Iterable[int],
        conv_state_slots: Iterable[int],
        lifecycle: HybridLifecycleState = HybridLifecycleState.ACTIVE,
    ) -> HybridStateHandle:
        if self.lifecycle not in {
            HybridLifecycleState.NEW,
            HybridLifecycleState.EVICTED,
        }:
            raise ContractViolation(
                f"Cannot bind components while handle is {self.lifecycle.value}"
            )
        return replace(
            self,
            full_kv_blocks=tuple(full_kv_blocks),
            recurrent_state_slots=tuple(recurrent_state_slots),
            conv_state_slots=tuple(conv_state_slots),
            lifecycle=lifecycle,
        )

    def transition(self, target: HybridLifecycleState) -> HybridStateHandle:
        if target not in _ALLOWED_TRANSITIONS[self.lifecycle]:
            raise ContractViolation(
                f"Invalid hybrid state transition: {self.lifecycle.value} -> {target.value}"
            )
        if target in {
            HybridLifecycleState.EVICTED,
            HybridLifecycleState.RELEASED,
        }:
            return replace(
                self,
                lifecycle=target,
                full_kv_blocks=(),
                recurrent_state_slots=(),
                conv_state_slots=(),
            )
        return replace(self, lifecycle=target)

    def assert_reusable_with(self, other: HybridStateHandle) -> None:
        logical_fields = (
            "prefix_identity",
            "boundary_fingerprint",
            "model_fingerprint",
            "tokenizer_fingerprint",
            "tp_world_size",
            "sequence_length",
            "namespace",
        )
        mismatched = [
            name for name in logical_fields if getattr(self, name) != getattr(other, name)
        ]
        if mismatched:
            raise ContractViolation(
                f"Hybrid prefix reuse fingerprint mismatch: {mismatched}"
            )
        if not self.has_complete_hybrid_state or not other.has_complete_hybrid_state:
            raise ContractViolation("Hybrid prefix reuse requires complete state on both handles")


@dataclass(frozen=True, slots=True)
class CancellationToken:
    token_id: str
    cancelled: bool = False

    def cancel(self) -> CancellationToken:
        return replace(self, cancelled=True)


@dataclass(frozen=True, slots=True)
class InternalJob:
    parent_request_id: str
    turn_id: str
    job_id: str
    job_type: InternalJobType
    priority: int
    shared_prefix_key: str
    token_budget: int
    state_budget_bytes: int
    deadline_monotonic: float
    cancellation_token: CancellationToken
    telemetry_correlation_id: str
    max_fanout: int
    recursion_depth: int = 0
    visibility: str = field(default="internal", init=False)

    def __post_init__(self) -> None:
        if not self.parent_request_id or not self.turn_id or not self.job_id:
            raise ContractViolation("Internal jobs require parent, turn, and job IDs")
        if not self.shared_prefix_key or not self.telemetry_correlation_id:
            raise ContractViolation("Internal jobs require prefix and telemetry identities")
        if self.token_budget < 1 or self.state_budget_bytes < 0:
            raise ContractViolation("Internal job budgets must be non-negative and non-empty")
        if self.max_fanout < 1:
            raise ContractViolation("Internal job max_fanout must be positive")
        if self.recursion_depth != 0:
            raise ContractViolation("Internal jobs cannot recursively create internal jobs")

    def is_cancelled_or_expired(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        return self.cancellation_token.cancelled or current >= self.deadline_monotonic


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    decision_id: str
    candidate_id: str
    parent_request_id: str
    question_digest: str
    reference_digest: str
    status: EligibilityStatus
    judge_method: str
    judge_model_fingerprint: str
    decision_margin: float | None

    def __post_init__(self) -> None:
        required = (
            self.decision_id,
            self.candidate_id,
            self.parent_request_id,
            self.question_digest,
            self.reference_digest,
            self.judge_method,
            self.judge_model_fingerprint,
        )
        if not all(str(value).strip() for value in required):
            raise ContractViolation("Eligibility decisions require complete identities")
        if self.status is EligibilityStatus.ELIGIBLE and self.decision_margin is None:
            raise ContractViolation("Eligible decisions require a decision margin")

    @property
    def eligible(self) -> bool:
        return self.status is EligibilityStatus.ELIGIBLE

    def require_eligible(self) -> None:
        if not self.eligible:
            raise ContractViolation(
                f"Candidate {self.candidate_id} is not semantically eligible"
            )

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        parent_request_id: str,
        question: str,
        reference: str,
        status: EligibilityStatus,
        judge_method: str,
        judge_model_fingerprint: str,
        decision_margin: float | None,
    ) -> EligibilityDecision:
        question_digest = stable_digest(question)
        reference_digest = stable_digest(reference)
        decision_id = stable_digest(
            candidate_id,
            parent_request_id,
            question_digest,
            reference_digest,
            status.value,
            judge_method,
            judge_model_fingerprint,
        )
        return cls(
            decision_id=decision_id,
            candidate_id=candidate_id,
            parent_request_id=parent_request_id,
            question_digest=question_digest,
            reference_digest=reference_digest,
            status=status,
            judge_method=judge_method,
            judge_model_fingerprint=judge_model_fingerprint,
            decision_margin=decision_margin,
        )


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    full_attention_kv_bytes: int
    recurrent_state_bytes: int
    conv_state_bytes: int
    internal_branch_bytes: int
    scoring_workspace_bytes: int
    output_workspace_bytes: int
    safety_reserve_bytes: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.components):
            raise ContractViolation("Resource estimate components cannot be negative")

    @property
    def components(self) -> tuple[int, ...]:
        return (
            self.full_attention_kv_bytes,
            self.recurrent_state_bytes,
            self.conv_state_bytes,
            self.internal_branch_bytes,
            self.scoring_workspace_bytes,
            self.output_workspace_bytes,
            self.safety_reserve_bytes,
        )

    @property
    def total_bytes(self) -> int:
        return sum(self.components)

    def fits(self, available_bytes: int, parent_reserved_bytes: int = 0) -> bool:
        usable = max(0, int(available_bytes) - int(parent_reserved_bytes))
        return self.total_bytes <= usable
