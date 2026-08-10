from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True, slots=True)
class SchedulerResourceEstimate:
    kv_tokens: int
    request_slots: int
    mamba_slots: int
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            self.kv_tokens < 1
            or self.request_slots < 0
            or self.mamba_slots < 0
            or self.workspace_bytes < 0
        ):
            raise ValueError("Scheduler admission estimates must be non-negative")


@dataclass(frozen=True, slots=True)
class SchedulerAdmissionDecision:
    admitted: bool
    reason: str
    estimate: SchedulerResourceEstimate
    available_kv_tokens: int
    available_request_slots: int
    available_mamba_slots: int | None
    available_workspace_bytes: int | None = None


class SchedulerAdmission:
    """Reserves waiting-queue capacity before allocator-specific admission.

    Each TP scheduler executes the same operation. ``consensus`` combines its
    rank-local decision with peers; no rank commits unless every rank admits.
    Reservations are released immediately before PrefillAdder performs the real
    KV and recurrent-state allocations.
    """

    def __init__(
        self,
        *,
        page_size: int,
        consensus: Callable[[bool], bool] | None = None,
    ):
        if page_size < 1:
            raise ValueError("Admission page size must be positive")
        self.page_size = int(page_size)
        self.consensus = consensus or (lambda accepted: accepted)
        self._reservations: dict[str, SchedulerResourceEstimate] = {}

    def estimate(
        self,
        *,
        prompt_tokens: int,
        max_new_tokens: int,
        needs_mamba: bool,
        request_slots: int = 1,
        workspace_bytes: int = 0,
        additional_mamba_slots: int = 0,
    ) -> SchedulerResourceEstimate:
        # One guard page matches PrefillAdder's strict upper-bound check.
        kv_tokens = (
            self._ceil_page(max(1, int(prompt_tokens)))
            + max(1, int(max_new_tokens))
            + self.page_size
        )
        return SchedulerResourceEstimate(
            kv_tokens=kv_tokens,
            request_slots=int(request_slots),
            mamba_slots=(1 + int(additional_mamba_slots)) if needs_mamba else 0,
            workspace_bytes=int(workspace_bytes),
        )

    def reserve(
        self,
        request_id: str,
        estimate: SchedulerResourceEstimate,
        *,
        available_kv_tokens: int,
        available_request_slots: int,
        available_mamba_slots: int | None,
        available_workspace_bytes: int | None = None,
    ) -> SchedulerAdmissionDecision:
        request_id = str(request_id)
        if request_id in self._reservations:
            return SchedulerAdmissionDecision(
                admitted=True,
                reason="already_reserved",
                estimate=self._reservations[request_id],
                available_kv_tokens=int(available_kv_tokens),
                available_request_slots=int(available_request_slots),
                available_mamba_slots=(
                    None
                    if available_mamba_slots is None
                    else int(available_mamba_slots)
                ),
                available_workspace_bytes=(
                    None
                    if available_workspace_bytes is None
                    else int(available_workspace_bytes)
                ),
            )

        reserved_kv = sum(item.kv_tokens for item in self._reservations.values())
        reserved_requests = sum(
            item.request_slots for item in self._reservations.values()
        )
        reserved_mamba = sum(item.mamba_slots for item in self._reservations.values())
        reserved_workspace = max(
            (item.workspace_bytes for item in self._reservations.values()),
            default=0,
        )
        local_reason = "admitted"
        if estimate.kv_tokens > int(available_kv_tokens) - reserved_kv:
            local_reason = "kv_capacity"
        elif estimate.request_slots > int(available_request_slots) - reserved_requests:
            local_reason = "request_slots"
        elif (
            estimate.mamba_slots
            and available_mamba_slots is not None
            and estimate.mamba_slots > int(available_mamba_slots) - reserved_mamba
        ):
            local_reason = "mamba_slots"
        elif available_workspace_bytes is not None and max(
            reserved_workspace, estimate.workspace_bytes
        ) > int(available_workspace_bytes):
            local_reason = "workspace_capacity"

        locally_admitted = local_reason == "admitted"
        globally_admitted = bool(self.consensus(locally_admitted))
        if globally_admitted:
            self._reservations[request_id] = estimate
            reason = "admitted"
        else:
            reason = local_reason if not locally_admitted else "peer_rank_capacity"
        return SchedulerAdmissionDecision(
            admitted=globally_admitted,
            reason=reason,
            estimate=estimate,
            available_kv_tokens=int(available_kv_tokens),
            available_request_slots=int(available_request_slots),
            available_mamba_slots=(
                None if available_mamba_slots is None else int(available_mamba_slots)
            ),
            available_workspace_bytes=(
                None
                if available_workspace_bytes is None
                else int(available_workspace_bytes)
            ),
        )

    def release(self, request_id: str) -> SchedulerResourceEstimate | None:
        return self._reservations.pop(str(request_id), None)

    def is_reserved(self, request_id: str) -> bool:
        return str(request_id) in self._reservations

    @property
    def reservation_count(self) -> int:
        return len(self._reservations)

    def _ceil_page(self, value: int) -> int:
        return ((value + self.page_size - 1) // self.page_size) * self.page_size
