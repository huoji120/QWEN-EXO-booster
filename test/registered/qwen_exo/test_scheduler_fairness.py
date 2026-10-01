from types import SimpleNamespace

import pytest

from qwen_exo_booster.scheduler_fairness import (
    BackgroundPrefillFairness,
    WorkClassAdmission,
    fits_complete_prefill,
    is_background_request,
    request_work_class,
    work_class_preemption,
)


def request(rid, work_class=None, *, finished=False, **params):
    if work_class is not None:
        params["qwen_exo_work_class"] = work_class
    return SimpleNamespace(
        rid=rid,
        sampling_params=SimpleNamespace(custom_params=params),
        finished=lambda: finished,
    )


def test_foreground_descendants_and_legacy_reflection_classification():
    assert request_work_class(request("user")) == "foreground"
    assert request_work_class(request("judge", qwen_exo_kind="internal")) == "foreground"
    assert request_work_class(
        request("legacy", qwen_exo_kind="internal", qwen_exo_job_type="reflection_memory")
    ) == "reflection"
    assert request_work_class(
        request("descendant", "maintenance", qwen_exo_kind="internal", qwen_exo_job_type="judge")
    ) == "maintenance"


@pytest.mark.parametrize("low_first", [False, True])
def test_class_order_survives_numeric_priority_direction(low_first):
    foreground = request("probe")
    background = request("reflection", "reflection")
    foreground.priority = 0
    background.priority = -100
    queue = sorted([background, foreground], key=lambda req: req.priority, reverse=not low_first)
    queue.sort(key=is_background_request)
    assert queue == [foreground, background]
    assert work_class_preemption(foreground, background) is True
    assert work_class_preemption(background, foreground) is False
    assert work_class_preemption(foreground, request("user")) is None


def test_caps_include_overlap_and_chunk_owners_once():
    reflection = request("reflection", "reflection")
    maintenance = [request(f"maintenance-{index}", "maintenance") for index in range(3)]
    # An owner can appear in both running and overlap/last batches.
    admission = WorkClassAdmission(6, [reflection, *maintenance, reflection])
    assert not admission.can_admit(request("reflection-2", "reflection"))
    assert not admission.can_admit(request("maintenance-4", "maintenance"))
    user = request("user")
    judge = request("judge", qwen_exo_kind="internal")
    assert admission.can_admit(user)
    admission.add(user)
    assert admission.can_admit(judge)
    admission.add(judge)
    # Foreground still reaches the real slot/resource gate and preemption path.
    assert admission.can_admit(request("seventh"))
    assert admission.can_admit(reflection)


def test_small_pool_reserves_foreground_capacity():
    admission = WorkClassAdmission(3, [request("r", "reflection"), request("m", "maintenance")])
    assert not admission.can_admit(request("another", "maintenance"))
    assert admission.can_admit(request("foreground"))


def test_finished_owners_release_lane_and_physical_capacity():
    admission = WorkClassAdmission(
        2, [request("old", "reflection", finished=True), request("user")]
    )
    assert admission.can_admit(request("next", "reflection"))
    admission.add(request("next", "reflection"))
    assert not admission.can_admit(request("other", "maintenance"))


def test_chunk_decode_alternation_resumes_background_under_continuous_foreground():
    policy = BackgroundPrefillFairness()
    background = request("background", "reflection")
    decisions = []
    for _ in range(6):
        if policy.should_yield(foreground_waiting=False, foreground_decoding=True):
            decisions.append("foreground_decode")
            policy.record_decode()
        else:
            decisions.append("background_chunk")
            policy.record_prefill([background])
    assert decisions == ["background_chunk", "foreground_decode"] * 3


def test_complete_foreground_prefill_returns_next_turn_to_background():
    policy = BackgroundPrefillFairness()
    policy.record_prefill([request("background", "maintenance")])
    assert policy.should_yield(foreground_waiting=True, foreground_decoding=False)
    policy.record_prefill([request("scoring")])
    assert not policy.should_yield(foreground_waiting=True, foreground_decoding=False)


def test_background_runs_without_useful_foreground_work():
    policy = BackgroundPrefillFairness()
    policy.record_prefill([request("background", "reflection")])
    assert not policy.should_yield(foreground_waiting=False, foreground_decoding=False)


@pytest.mark.parametrize(
    "total,prefix,remaining,expected",
    [(65, 0, 64, False), (64, 0, 64, True), (65, 1, 64, True), (1, 0, 63, False)],
)
def test_yielded_slot_page_boundary_cannot_create_second_chunk(total, prefix, remaining, expected):
    req = request("foreground")
    req.full_untruncated_fill_ids = range(total)
    req.prefix_indices = range(prefix)
    assert fits_complete_prefill(req, remaining_chunk_tokens=remaining, page_size=64) is expected
