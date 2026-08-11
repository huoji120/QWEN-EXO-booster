import asyncio
import time
from types import SimpleNamespace

import pytest
from qwen_exo_booster.contracts import (
    CancellationToken,
    ContractViolation,
    InternalJob,
    InternalJobType,
)
from qwen_exo_booster.internal_jobs import InternalJobRunner


class FakeTokenizerManager:
    def __init__(self, outputs=None, delay=0, error=None):
        self.outputs = outputs or []
        self.delay = delay
        self.error = error
        self.requests = []
        self.aborted = []

    async def generate_request(self, request, raw_request):
        assert raw_request is None
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.delay:
            await asyncio.sleep(self.delay)
        yield list(self.outputs)

    def abort_request(self, rid):
        self.aborted.append(rid)


def job(index=0, **overrides):
    values = {
        "parent_request_id": "parent-1",
        "turn_id": "turn-1",
        "job_id": f"job-{index}",
        "job_type": InternalJobType.REFERENCE_JUDGE,
        "priority": -10,
        "shared_prefix_key": "qwen-exo:v1:test:prefix-key",
        "token_budget": 16,
        "state_budget_bytes": 1024,
        "deadline_monotonic": time.monotonic() + 5,
        "cancellation_token": CancellationToken(f"cancel-{index}"),
        "telemetry_correlation_id": "trace-1",
        "max_fanout": 4,
    }
    values.update(overrides)
    return InternalJob(**values)


def request_factory(**kwargs):
    return SimpleNamespace(**kwargs)


def test_internal_batch_uses_one_scheduler_request():
    manager = FakeTokenizerManager(
        outputs=[
            {"text": '{"supported":true}', "meta_info": {"completion_tokens": 4}},
            {"text": '{"supported":false}', "meta_info": {"completion_tokens": 4}},
        ]
    )
    runner = InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )

    results = asyncio.run(
        runner.run_batch(
            [job(0), job(1)],
            ["shared prefix A", "shared prefix B"],
            {"temperature": 0},
        )
    )

    assert [result.text for result in results] == [
        '{"supported":true}',
        '{"supported":false}',
    ]
    assert len(manager.requests) == 1
    request = manager.requests[0]
    assert request.rid == ["job-0", "job-1"]
    assert request.priority == -10
    assert request.no_logs is True
    assert request.custom_labels["qwen_exo_visibility"] == "internal"
    assert request.sampling_params[0]["custom_params"] == {
        "qwen_exo_kind": "internal",
        "qwen_exo_job_type": "reference_judge",
        "qwen_exo_parent_request_id": "parent-1",
        "qwen_exo_state_budget_bytes": 1024,
    }


def test_reflection_job_without_deadline_reaches_scheduler():
    manager = FakeTokenizerManager(
        outputs=[{"text": "ok", "meta_info": {"completion_tokens": 1}}]
    )
    runner = InternalJobRunner(
        manager, max_fanout=1, max_tokens_per_parent=16, request_factory=request_factory
    )
    reflection_job = job(
        job_type=InternalJobType.REFLECTION_MEMORY,
        deadline_monotonic=None,
        max_fanout=1,
    )

    result = asyncio.run(
        runner.run_batch([reflection_job], ["prompt"], {"temperature": 0})
    )

    assert result[0].text == "ok"
    assert len(manager.requests) == 1


def test_internal_batch_enforces_parent_token_reserve():
    runner = InternalJobRunner(
        FakeTokenizerManager(),
        max_fanout=4,
        max_tokens_per_parent=16,
        request_factory=request_factory,
    )

    with pytest.raises(ContractViolation, match="token reserve"):
        asyncio.run(runner.run_batch([job(0), job(1)], ["a", "b"], {"temperature": 0}))


def test_parent_token_reserve_is_cumulative_until_request_finishes():
    manager = FakeTokenizerManager(
        outputs=[{"text": "ok", "meta_info": {"completion_tokens": 1}}]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=4,
        max_tokens_per_parent=24,
        request_factory=request_factory,
    )

    async def exercise():
        await runner.run_batch([job(0)], ["first"], {"temperature": 0})
        with pytest.raises(ContractViolation, match="cumulative token reserve"):
            await runner.run_batch([job(1)], ["second"], {"temperature": 0})
        await runner.finish_parent("parent-1")
        await runner.run_batch([job(2)], ["third"], {"temperature": 0})

    asyncio.run(exercise())
    assert len(manager.requests) == 2


def test_parent_cancellation_aborts_active_children():
    manager = FakeTokenizerManager(
        outputs=[{"text": "never", "meta_info": {}}], delay=0.1
    )
    runner = InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )

    async def exercise():
        task = asyncio.create_task(
            runner.run_batch([job(0)], ["prompt"], {"temperature": 0})
        )
        await asyncio.sleep(0)
        await runner.cancel_parent("parent-1")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert "job-0" in manager.aborted


def test_non_timeout_scheduler_error_aborts_every_sibling():
    manager = FakeTokenizerManager(error=ValueError("scheduler failed"))
    runner = InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )

    with pytest.raises(ValueError, match="scheduler failed"):
        asyncio.run(
            runner.run_batch([job(0), job(1)], ["first", "second"], {"temperature": 0})
        )

    assert set(manager.aborted) == {"job-0", "job-1"}


def test_expired_job_never_reaches_scheduler():
    manager = FakeTokenizerManager()
    runner = InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runner.run_batch(
                [job(deadline_monotonic=time.monotonic() - 1)],
                ["prompt"],
                {"temperature": 0},
            )
        )
    assert manager.requests == []


def test_replay_score_batch_returns_label_nll_and_stays_internal():
    manager = FakeTokenizerManager(
        outputs=[
            {
                "text": "x",
                "meta_info": {
                    "input_token_logprobs": [
                        (None, 1),
                        (-1.0, 2),
                        (-3.0, 3),
                    ],
                    "prompt_tokens": 3,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=4,
        max_tokens_per_parent=64,
        request_factory=request_factory,
    )
    replay_job = job(
        0,
        job_type=InternalJobType.CAUSAL_REPLAY,
        token_budget=1,
        state_budget_bytes=0,
    )

    result = asyncio.run(
        runner.run_score_batch(
            (replay_job,),
            ((1, 2, 3),),
            (1,),
            {"temperature": 0},
        )
    )[0]

    assert result.token_logprobs == (-1.0, -3.0)
    assert result.mean_nll == 2.0
    request = manager.requests[0]
    assert request.return_logprob is True
    assert request.logprob_start_len == [1]
    assert request.no_logs is True
    assert request.sampling_params[0]["custom_params"]["qwen_exo_job_type"] == (
        "causal_replay"
    )
    assert request.sampling_params[0]["max_new_tokens"] == 1


def test_bank_score_export_generates_no_token_after_document_boundary():
    manager = FakeTokenizerManager(
        outputs=[
            {
                "text": "",
                "meta_info": {
                    "input_token_logprobs": [(None, 1), (-2.0, 2)],
                    "prompt_tokens": 2,
                    "finish_reason": {"type": "length", "length": 0},
                },
            }
        ]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=4,
        max_tokens_per_parent=64,
        request_factory=request_factory,
    )
    bank_job = job(
        0,
        job_type=InternalJobType.BANK_INDEX,
        token_budget=1,
        state_budget_bytes=0,
    )

    result = asyncio.run(
        runner.run_score_batch((bank_job,), ((1, 2),), (1,), {"temperature": 0})
    )[0]

    assert result.token_logprobs == (-2.0,)
    assert manager.requests[0].sampling_params[0]["max_new_tokens"] == 0


def test_internal_job_cannot_use_user_cache_namespace():
    manager = FakeTokenizerManager()
    runner = InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )

    with pytest.raises(ContractViolation, match="user-visible cache namespace"):
        asyncio.run(
            runner.run_batch(
                [job(shared_prefix_key=("qwen-exo:v1:external_memory:user-visible"))],
                ["prompt"],
                {"temperature": 0},
            )
        )

    assert manager.requests == []


class MeasuringTokenizerManager(FakeTokenizerManager):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def generate_request(self, request, raw_request):
        assert raw_request is None
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            yield [{"text": "ok", "meta_info": {"completion_tokens": 1}}]
        finally:
            self.active -= 1


def test_global_admission_serializes_internal_batches_across_parents():
    manager = MeasuringTokenizerManager(delay=0.02)
    runner = InternalJobRunner(
        manager,
        max_fanout=1,
        max_tokens_per_parent=64,
        request_factory=request_factory,
    )

    async def exercise():
        first = asyncio.create_task(
            runner.run_batch(
                [job(0, parent_request_id="parent-a", job_id="job-a")],
                ["first"],
                {"temperature": 0},
            )
        )
        await asyncio.sleep(0)
        second = asyncio.create_task(
            runner.run_batch(
                [job(1, parent_request_id="parent-b", job_id="job-b")],
                ["second"],
                {"temperature": 0},
            )
        )
        await asyncio.gather(first, second)

    asyncio.run(exercise())

    assert len(manager.requests) == 2
    assert manager.max_active == 1


def test_global_admission_honors_waiting_job_deadline():
    manager = MeasuringTokenizerManager(delay=0.05)
    runner = InternalJobRunner(
        manager,
        max_fanout=1,
        max_tokens_per_parent=64,
        request_factory=request_factory,
    )

    async def exercise():
        first = asyncio.create_task(
            runner.run_batch(
                [job(0, parent_request_id="parent-a", job_id="job-a")],
                ["first"],
                {"temperature": 0},
            )
        )
        await asyncio.sleep(0)
        with pytest.raises(asyncio.TimeoutError, match="global admission"):
            await runner.run_batch(
                [
                    job(
                        1,
                        parent_request_id="parent-b",
                        job_id="job-b",
                        deadline_monotonic=time.monotonic() + 0.01,
                    )
                ],
                ["second"],
                {"temperature": 0},
            )
        await first

    asyncio.run(exercise())

    assert len(manager.requests) == 1
