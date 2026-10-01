"""CPU-only scheduling policy; physical KV/Mamba admission remains in SGLang."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


FOREGROUND = "foreground"
REFLECTION = "reflection"
MAINTENANCE = "maintenance"


def request_work_class(req: Any) -> str:
    params = getattr(req.sampling_params, "custom_params", None) or {}
    work_class = params.get("qwen_exo_work_class")
    if work_class in (FOREGROUND, REFLECTION, MAINTENANCE):
        return work_class
    if (
        params.get("qwen_exo_kind") == "internal"
        and params.get("qwen_exo_job_type") == "reflection_memory"
    ):
        return REFLECTION
    return FOREGROUND


def is_background_request(req: Any) -> bool:
    return request_work_class(req) != FOREGROUND


def work_class_preemption(req: Any, victim: Any) -> bool | None:
    """Override numeric priority only across the foreground/background boundary."""
    incoming_background = is_background_request(req)
    victim_background = is_background_request(victim)
    if incoming_background == victim_background:
        return None
    return victim_background


class WorkClassAdmission:
    """Supplemental background limits; SGLang owns physical admission/preemption."""

    def __init__(self, max_running_requests: int, active: Iterable[Any]):
        self.max_running_requests = max_running_requests
        self._owners: set[str] = set()
        self._counts = {FOREGROUND: 0, REFLECTION: 0, MAINTENANCE: 0}
        for req in active:
            if not req.finished():
                self.add(req)

    def add(self, req: Any) -> None:
        if req.rid not in self._owners:
            self._owners.add(req.rid)
            self._counts[request_work_class(req)] += 1

    def can_admit(self, req: Any) -> bool:
        if req.rid in self._owners:
            return True
        work_class = request_work_class(req)
        if work_class == FOREGROUND:
            # A full pool may have preemptible background owners. Foreground
            # must reach SGLang's physical-slot/KV gates and victim selection.
            return True
        if len(self._owners) >= self.max_running_requests:
            return False
        # Four background owners at most (1 reflection + 3 maintenance), and
        # leave a physical foreground slot whenever the pool has more than one.
        # A single-slot pool must serialize rather than permanently disable work.
        background = self._counts[REFLECTION] + self._counts[MAINTENANCE]
        if background >= max(1, self.max_running_requests - 1):
            return False
        limit = 1 if work_class == REFLECTION else 3
        return self._counts[work_class] < limit


class BackgroundPrefillFairness:
    """Alternate background prefill with useful foreground work at chunk boundaries.

    No request, cache, or token ownership lives here. An unscheduled chunk stays
    in Scheduler.chunked_req and uses the existing resume/abort/pause paths.
    """

    def __init__(self) -> None:
        self.last_was_background_prefill = False

    def should_yield(self, *, foreground_waiting: bool, foreground_decoding: bool) -> bool:
        return self.last_was_background_prefill and (
            foreground_waiting or foreground_decoding
        )

    def record_prefill(self, reqs: Iterable[Any]) -> None:
        self.last_was_background_prefill = any(is_background_request(req) for req in reqs)

    def record_decode(self) -> None:
        self.last_was_background_prefill = False


def fits_complete_prefill(req: Any, *, remaining_chunk_tokens: int, page_size: int) -> bool:
    """A yielded slot must not create a second persistent chunk owner."""
    tokens = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
    paged_tokens = -(-tokens // page_size) * page_size
    return paged_tokens <= remaining_chunk_tokens
