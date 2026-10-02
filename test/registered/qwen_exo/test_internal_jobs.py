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
from qwen_exo_booster.task_scheduling import (
    current_internal_work_class,
    internal_task_scope,
)


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
    assert request.priority == 0
    assert request.no_logs is True
    assert request.custom_labels["qwen_exo_visibility"] == "internal"
    assert request.sampling_params[0]["custom_params"] == {
        "qwen_exo_kind": "internal",
        "qwen_exo_job_type": "reference_judge",
        "qwen_exo_parent_request_id": "parent-1",
        "qwen_exo_state_budget_bytes": 1024,
        "qwen_exo_work_class": "foreground",
    }


def test_plain_long_internal_generation_marks_dflash_eligible():
    manager = FakeTokenizerManager(
        outputs=[{"text": "summary", "meta_info": {"completion_tokens": 4}}]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=1,
        max_tokens_per_parent=128,
        request_factory=request_factory,
    )

    asyncio.run(
        runner.run_batch(
            [
                job(
                    job_type=InternalJobType.RESPONSE_COMPACTION,
                    token_budget=96,
                    max_fanout=1,
                )
            ],
            ["plain summary prompt"],
            {"temperature": 0},
        )
    )

    custom = manager.requests[0].sampling_params[0]["custom_params"]
    assert custom["qwen_exo_dflash"] == "eligible"


def test_structured_internal_generation_can_use_dflash():
    """DFLASH verifies grammar-constrained blocks with per-position masks, so
    a structured output no longer forces an eligible job onto the slow
    target-only path that also blocks DFLASH user batches."""
    manager = FakeTokenizerManager(
        outputs=[{"text": "{}", "meta_info": {"completion_tokens": 2}}]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=1,
        max_tokens_per_parent=128,
        request_factory=request_factory,
    )

    asyncio.run(
        runner.run_batch(
            [
                job(
                    job_type=InternalJobType.SELF_ANSWER,
                    token_budget=96,
                    max_fanout=1,
                )
            ],
            ["structured answer prompt"],
            {"temperature": 0, "json_schema": {"type": "object"}},
        )
    )

    custom = manager.requests[0].sampling_params[0]["custom_params"]
    assert custom["qwen_exo_dflash"] == "eligible"


def test_explicit_target_only_overrides_dflash_eligible_job_type():
    manager = FakeTokenizerManager(
        outputs=[{"text": "reflection", "meta_info": {"completion_tokens": 4}}]
    )
    runner = InternalJobRunner(
        manager,
        max_fanout=1,
        max_tokens_per_parent=128,
        request_factory=request_factory,
    )

    asyncio.run(
        runner.run_batch(
            [
                job(
                    job_type=InternalJobType.REFLECTION_MEMORY,
                    token_budget=96,
                    deadline_monotonic=None,
                    max_fanout=1,
                )
            ],
            ["capture a flushed target recurrent state"],
            {
                "temperature": 0,
                "custom_params": {"qwen_exo_dflash": "target_only"},
            },
        )
    )

    custom = manager.requests[0].sampling_params[0]["custom_params"]
    assert custom["qwen_exo_dflash"] == "target_only"


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

    with pytest.raises(asyncio.TimeoutError):
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


def test_global_admission_honors_waiting_job_deadline(monkeypatch):
    from qwen_exo_booster import internal_jobs

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingManager(FakeTokenizerManager):
        async def generate_request(self, request, raw_request):
            self.requests.append(request)
            started.set()
            await release.wait()
            yield [{"text": "ok", "meta_info": {"completion_tokens": 1}}]

    manager = BlockingManager()
    runner = InternalJobRunner(
        manager, max_fanout=1, max_tokens_per_parent=64, request_factory=request_factory
    )

    async def exercise():
        first = asyncio.create_task(
            runner.run_batch(
                [job(0, parent_request_id="parent-a", job_id="job-a")],
                ["first"],
                {"temperature": 0},
            )
        )
        await started.wait()
        deadline = time.monotonic() + 60
        try:
            # Advance only the admission clock; do not race a 10 ms OS timer
            # against the pre-dispatch cancellation check or the event loop.
            with monkeypatch.context() as patch:
                patch.setattr(
                    internal_jobs,
                    "time",
                    SimpleNamespace(
                        monotonic=lambda: deadline + 1,
                        perf_counter=time.perf_counter,
                    ),
                )
                with pytest.raises(asyncio.TimeoutError):
                    await runner.run_batch(
                        [
                            job(
                                1,
                                parent_request_id="parent-b",
                                job_id="job-b",
                                deadline_monotonic=deadline,
                            )
                        ],
                        ["second"],
                        {"temperature": 0},
                    )
        finally:
            release.set()
            await first

    asyncio.run(exercise())
    assert len(manager.requests) == 1


def test_single_job_probe_is_admitted_while_a_sibling_job_runs():
    """A job's max_fanout bounds its batch, not the parent's concurrency.

    The mid-think query probe (one job, max_fanout=1) was rejected with
    "Parent already owns the maximum child fanout" whenever the concurrent
    self-ask job of the same request was still running, so mid-think recall
    silently returned no candidates. The parent's concurrent children are
    bounded by the runner fanout; each batch is bounded by its own contract.
    """
    release = asyncio.Event()

    class BlockingTokenizerManager(FakeTokenizerManager):
        async def generate_request(self, request, raw_request):
            self.requests.append(request)
            if request.rid == ["self-ask"]:
                await release.wait()
                yield [{"text": "answer", "meta_info": {"completion_tokens": 4}}]
            else:
                yield [{"text": "probe", "meta_info": {"completion_tokens": 1}}]

    manager = BlockingTokenizerManager()
    runner = InternalJobRunner(
        manager,
        max_fanout=4,
        max_tokens_per_parent=256,
        request_factory=request_factory,
    )

    async def exercise():
        self_ask = asyncio.create_task(
            runner.run_batch(
                [job(job_id="self-ask", token_budget=64, max_fanout=1)],
                ["self ask"],
                {"temperature": 0},
            )
        )
        await asyncio.sleep(0)
        probe = await runner.run_batch(
            [
                job(
                    job_id="probe",
                    job_type=InternalJobType.QUERY_PROBE,
                    token_budget=1,
                    max_fanout=1,
                )
            ],
            ["probe"],
            {"temperature": 0},
        )
        release.set()
        await self_ask
        with pytest.raises(ContractViolation, match="fanout"):
            await runner.run_batch(
                [job(index=i, max_fanout=2) for i in range(3)],
                ["a", "b", "c"],
                {"temperature": 0},
            )
        return probe

    (result,) = asyncio.run(exercise())
    assert result.text == "probe"


def option_output(index=0, prefix=(10, 11), scores=(-0.25, -2.5)):
    return {
        "text": "",
        "output_ids": [],
        "meta_info": {
            "id": f"job-{index}",
            "prompt_tokens": len(prefix) + 1,
            "completion_tokens": 0,
            "finish_reason": {"type": "length", "length": 0},
            "input_token_logprobs": [(None, prefix[-1], None), (-8.0, 0, None)],
            "input_token_ids_logprobs": [
                None,
                [(scores[0], 32, None), (scores[1], 33, None)],
            ],
        },
    }


def option_runner(manager):
    return InternalJobRunner(
        manager, max_fanout=4, max_tokens_per_parent=64, request_factory=request_factory
    )


def test_option_scores_read_same_prefix_distribution_without_generation():
    manager = FakeTokenizerManager(
        outputs=[
            option_output(),
            option_output(1, prefix=(12,), scores=(-3.0, -0.1)),
        ]
    )
    runner = option_runner(manager)
    jobs = (job(), job(1))
    results = asyncio.run(
        runner.run_option_score_batch(jobs, ((10, 11), (12,)), (32, 33))
    )
    assert tuple(result.job for result in results) == jobs
    assert tuple(result.option_logprobs for result in results) == (
        (-0.25, -2.5),
        (-3.0, -0.1),
    )
    assert tuple(result.prompt_tokens for result in results) == (3, 2)
    assert tuple(result.completion_tokens for result in results) == (0, 0)
    assert len(manager.requests) == 1
    request = manager.requests[0]
    # Exposes the distribution BEFORE sentinel, not after it or before prefix[-1].
    assert request.input_ids == [[10, 11, 0], [12, 0]]
    assert request.logprob_start_len == [1, 0]
    assert request.token_ids_logprob == [[32, 33], [32, 33]]
    assert all(params["max_new_tokens"] == 0 for params in request.sampling_params)
    assert all(
        params["custom_params"]["qwen_exo_dflash"] == "target_only"
        for params in request.sampling_params
    )
    assert not runner._active
    # Existing runner accounts cumulative parent usage until finish_parent.
    assert runner._reserved_tokens == {"parent-1": 32}
    asyncio.run(runner.finish_parent("parent-1"))
    assert not runner._reserved_tokens


@pytest.mark.parametrize(
    "jobs,prefixes,options",
    [
        ((), (), (32, 33)),
        ((job(),), (), (32, 33)),
        ((job(),), ((),), (32, 33)),
        ((job(),), ((10,),), (32, 32)),
        ((job(),), ((10,),), (32,)),
        ((job(),), ((10,),), (32, 33, 34)),
        ((job(),), ((True,),), (32, 33)),
        ((job(),), ((10,),), (32, -1)),
        ((job(), job()), ((10,), (11,)), (32, 33)),
    ],
)
def test_option_scores_reject_invalid_inputs_before_dispatch(jobs, prefixes, options):
    manager = FakeTokenizerManager()
    runner = option_runner(manager)
    with pytest.raises(ContractViolation):
        asyncio.run(runner.run_option_score_batch(jobs, prefixes, options))
    assert not manager.requests
    assert not runner._active
    assert not runner._reserved_tokens


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "different-job"),
        ("completion_tokens", 1),
        ("completion_tokens", None),
        ("finish_reason", {"type": "abort"}),
        ("finish_reason", {"type": "length", "length": 1}),
        ("prompt_tokens", 2),
        ("input_token_logprobs", [(None, 10, None), (-8.0, 0, None)]),
        ("input_token_logprobs", [(None, 11, None), (-8.0, 5, None)]),
        ("input_token_ids_logprobs", []),
        ("input_token_ids_logprobs", [[(-0.25, 32, None), (-2.5, 33, None)]]),
        ("input_token_ids_logprobs", [None, [(-2.5, 33, None), (-0.25, 32, None)]]),
        (
            "input_token_ids_logprobs",
            [None, [(float("nan"), 32, None), (-2.5, 33, None)]],
        ),
        (
            "input_token_ids_logprobs",
            [None, [(-0.25, 32, None), (float("-inf"), 33, None)]],
        ),
        ("input_token_ids_logprobs", [None, [(-0.25, 32, None), (None, 33, None)]]),
        ("input_token_ids_logprobs", [None, [(-0.25, 32, None)]]),
    ],
)
def test_option_scores_reject_malformed_results_and_release_capacity(field, value):
    output = option_output()
    output["meta_info"][field] = value
    manager = FakeTokenizerManager(outputs=[output])
    runner = option_runner(manager)
    with pytest.raises(RuntimeError):
        asyncio.run(runner.run_option_score_batch((job(),), ((10, 11),), (32, 33)))
    assert manager.aborted == ["job-0"]
    assert not runner._active


def test_option_scores_reject_reordered_batch_and_abort_every_job():
    manager = FakeTokenizerManager(outputs=[option_output(1), option_output(0)])
    runner = option_runner(manager)
    with pytest.raises(RuntimeError, match="identity"):
        asyncio.run(
            runner.run_option_score_batch(
                (job(), job(1)), ((10, 11), (10, 11)), (32, 33)
            )
        )
    assert manager.aborted == ["job-0", "job-1"]
    assert not runner._active


def test_option_scores_reject_missing_results():
    manager = FakeTokenizerManager()
    runner = option_runner(manager)
    with pytest.raises(RuntimeError, match="result count"):
        asyncio.run(runner.run_option_score_batch((job(),), ((10, 11),), (32, 33)))
    assert manager.aborted == ["job-0"]
    assert not runner._active


def test_option_scores_cancel_inflight_and_release_capacity():
    async def exercise():
        entered = asyncio.Event()
        blocked = asyncio.Event()

        class BlockingManager(FakeTokenizerManager):
            async def generate_request(self, request, raw_request):
                entered.set()
                await blocked.wait()
                yield [option_output()]

        manager = BlockingManager()
        runner = option_runner(manager)
        task = asyncio.create_task(
            runner.run_option_score_batch((job(),), ((10, 11),), (32, 33))
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert manager.aborted == ["job-0"]
        assert not runner._active

    asyncio.run(exercise())


def test_option_scores_parent_cancellation_rejects_even_a_returned_result():
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BlockingManager(FakeTokenizerManager):
            async def generate_request(self, request, raw_request):
                entered.set()
                await release.wait()
                yield [option_output()]

        manager = BlockingManager()
        runner = option_runner(manager)
        task = asyncio.create_task(
            runner.run_option_score_batch((job(),), ((10, 11),), (32, 33))
        )
        await entered.wait()
        await runner.cancel_parent("parent-1")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not runner._active
        with pytest.raises(asyncio.CancelledError):
            await runner.run_option_score_batch((job(1),), ((10, 11),), (32, 33))
        await runner.finish_parent("parent-1")
        assert not runner._reserved_tokens

    asyncio.run(exercise())


class LaneTokenizerManager(FakeTokenizerManager):
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Queue()
        self.releases = {}
        self.active = dict.fromkeys(("foreground", "reflection", "maintenance"), 0)
        self.maximum = self.active.copy()

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        lane = request.sampling_params[0]["custom_params"]["qwen_exo_work_class"]
        release = self.releases.setdefault(request.rid[0], asyncio.Event())
        self.active[lane] += len(request.rid)
        self.maximum[lane] = max(self.maximum[lane], self.active[lane])
        self.entered.put_nowait(tuple(request.rid))
        try:
            await release.wait()
            yield [{"text": rid, "meta_info": {}} for rid in request.rid]
        finally:
            self.active[lane] -= len(request.rid)


def test_background_lanes_do_not_consume_foreground_admission():
    async def exercise():
        manager = LaneTokenizerManager()
        runner = InternalJobRunner(
            manager,
            max_fanout=2,
            max_tokens_per_parent=64,
            request_factory=request_factory,
        )

        def submit(lane, index):
            with internal_task_scope(lane):
                return asyncio.create_task(
                    runner.run_batch(
                        [job(index, parent_request_id=f"parent-{index}")],
                        ["prompt"],
                        {},
                        custom_params_per_job=[{"qwen_exo_work_class": "foreground"}],
                    )
                )

        tasks = [submit("reflection", 0)]
        tasks.extend(submit("maintenance", index) for index in range(1, 4))
        for _ in range(4):
            await asyncio.wait_for(manager.entered.get(), 1)
        waiting = [submit("reflection", 4), submit("maintenance", 5)]
        foreground = [submit("foreground", index) for index in (6, 7)]
        admitted = {await asyncio.wait_for(manager.entered.get(), 1) for _ in range(2)}
        assert admitted == {("job-6",), ("job-7",)}
        assert manager.active == {"foreground": 2, "reflection": 1, "maintenance": 3}
        assert not any(task.done() for task in waiting)
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)
        for release in manager.releases.values():
            release.set()
        await asyncio.gather(*tasks, *foreground)
        assert runner._lane_active == dict.fromkeys(manager.active, 0)
        assert manager.maximum == {"foreground": 2, "reflection": 1, "maintenance": 3}
        assert all(
            request.priority == (0 if request.rid[0] in {"job-6", "job-7"} else -100)
            for request in manager.requests
        )

    asyncio.run(exercise())


def test_scopes_follow_spawned_children_without_leaking_or_promotion():
    async def exercise():
        release = asyncio.Event()

        @internal_task_scope("foreground")
        async def descendant():
            await release.wait()
            return current_internal_work_class(InternalJobType.QUERY_PROBE)

        with internal_task_scope("reflection"):
            task = asyncio.create_task(descendant())
        assert current_internal_work_class(InternalJobType.QUERY_PROBE) == "foreground"
        release.set()
        assert await task == "reflection"
        assert await descendant() == "foreground"
        assert (
            current_internal_work_class(InternalJobType.REFLECTION_MEMORY)
            == "reflection"
        )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "work_class,expected_sizes",
    [
        ("reflection", [1, 1, 1, 1]),
        ("maintenance", [3, 1]),
    ],
)
@pytest.mark.parametrize("method", ["generation", "score", "options"])
def test_background_batches_split_without_losing_results_or_budgets(
    work_class, expected_sizes, method
):
    class BatchManager(FakeTokenizerManager):
        async def generate_request(self, request, raw_request):
            self.requests.append(request)
            outputs = []
            for index, rid in enumerate(request.rid):
                ordinal = int(rid.split("-")[-1])
                if method == "options":
                    outputs.append(
                        option_output(
                            ordinal,
                            prefix=tuple(request.input_ids[index][:-1]),
                            scores=(-ordinal - 0.25, -2.5),
                        )
                    )
                elif method == "score":
                    outputs.append(
                        {
                            "meta_info": {
                                "input_token_logprobs": [
                                    (None, 1),
                                    (-ordinal - 1.0, 2),
                                ],
                                "prompt_tokens": 2,
                            }
                        }
                    )
                else:
                    outputs.append({"text": str(ordinal), "meta_info": {}})
            yield outputs

    async def exercise():
        manager = BatchManager()
        runner = option_runner(manager)
        jobs = tuple(job(index) for index in range(4))
        with internal_task_scope(work_class):
            if method == "options":
                result = await runner.run_option_score_batch(
                    jobs, [(10, 11)] * 4, (32, 33)
                )
                assert [item.option_logprobs[0] for item in result] == [
                    -0.25,
                    -1.25,
                    -2.25,
                    -3.25,
                ]
            elif method == "score":
                result = await runner.run_score_batch(jobs, [(1, 2)] * 4, [1] * 4)
                assert [item.mean_nll for item in result] == [1, 2, 3, 4]
            else:
                result = await runner.run_batch(jobs, ["prompt"] * 4, {})
                assert [item.text for item in result] == ["0", "1", "2", "3"]
        assert tuple(item.job for item in result) == jobs
        assert [len(request.rid) for request in manager.requests] == expected_sizes
        assert runner._reserved_tokens == {"parent-1": 64}
        assert not any(runner._lane_active.values())

    asyncio.run(exercise())


def test_cancelled_background_waiter_and_active_job_release_lane():
    async def exercise():
        manager = LaneTokenizerManager()
        runner = option_runner(manager)

        def submit(index):
            with internal_task_scope("reflection"):
                return asyncio.create_task(
                    runner.run_batch(
                        [job(index, parent_request_id=f"parent-{index}")],
                        ["prompt"],
                        {},
                    )
                )

        active = submit(0)
        assert await asyncio.wait_for(manager.entered.get(), 1) == ("job-0",)
        waiting = submit(1)
        await asyncio.sleep(0)
        await runner.cancel_parent("parent-1")
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert "parent-1" not in runner._reserved_tokens
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        next_job = submit(2)
        assert await asyncio.wait_for(manager.entered.get(), 1) == ("job-2",)
        manager.releases["job-2"].set()
        assert (await next_job)[0].text == "job-2"
        assert manager.maximum["reflection"] == 1
        assert not any(runner._lane_active.values())
        assert manager.aborted == ["job-0"]

    asyncio.run(exercise())


def test_waiting_background_batches_recheck_cumulative_budget_after_admission():
    async def exercise():
        manager = LaneTokenizerManager()
        runner = InternalJobRunner(
            manager,
            max_fanout=4,
            max_tokens_per_parent=16,
            request_factory=request_factory,
        )
        with internal_task_scope("reflection"):
            blocker = asyncio.create_task(
                runner.run_batch([job(0, parent_request_id="blocker")], ["hold"], {})
            )
            assert await asyncio.wait_for(manager.entered.get(), 1) == ("job-0",)
            waiting = [
                asyncio.create_task(runner.run_batch([job(index)], ["prompt"], {}))
                for index in (1, 2)
            ]
        await asyncio.sleep(0)
        manager.releases["job-0"].set()
        await blocker
        (admitted,) = await asyncio.wait_for(manager.entered.get(), 1)
        manager.releases[admitted].set()
        results = await asyncio.gather(*waiting, return_exceptions=True)
        assert sum(isinstance(result, ContractViolation) for result in results) == 1
        assert sum(isinstance(result, tuple) for result in results) == 1
        assert runner._reserved_tokens["parent-1"] == 16
        assert not any(runner._lane_active.values())
        assert len(manager.requests) == 2

    asyncio.run(exercise())


def test_queued_background_deadline_is_timeout_not_parent_cancellation(monkeypatch):
    from qwen_exo_booster import internal_jobs

    clock = [time.monotonic()]
    monkeypatch.setattr(
        internal_jobs,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            perf_counter=time.perf_counter,
        ),
    )

    async def exercise():
        manager = LaneTokenizerManager()
        runner = option_runner(manager)
        waiting = asyncio.Event()

        class AdmissionCondition(asyncio.Condition):
            async def wait(self):
                waiting.set()
                return await super().wait()

        runner._capacity_changed = AdmissionCondition(runner._lock)
        with internal_task_scope("reflection"):
            active = asyncio.create_task(
                runner.run_batch(
                    [
                        job(
                            0,
                            parent_request_id="active",
                            job_type=InternalJobType.REFLECTION_MEMORY,
                            deadline_monotonic=None,
                        )
                    ],
                    ["hold"],
                    {},
                )
            )
            assert await asyncio.wait_for(manager.entered.get(), 1) == ("job-0",)
            queued = asyncio.create_task(
                runner.run_batch(
                    [
                        job(
                            1,
                            parent_request_id="queued",
                            deadline_monotonic=clock[0] + 60,
                        )
                    ],
                    ["queued"],
                    {},
                )
            )
        await asyncio.wait_for(waiting.wait(), 1)
        clock[0] += 61
        async with runner._capacity_changed:
            runner._capacity_changed.notify_all()
        with pytest.raises(asyncio.TimeoutError):
            await queued
        assert "queued" not in runner._reserved_tokens
        assert manager.aborted == []
        manager.releases["job-0"].set()
        await active
        assert not any(runner._lane_active.values())
        assert len(manager.requests) == 1

    asyncio.run(exercise())


def test_option_score_deadline_after_output_remains_execution_timeout(monkeypatch):
    from qwen_exo_booster import internal_jobs

    clock = [time.monotonic()]
    deadline = clock[0] + 60
    monkeypatch.setattr(
        internal_jobs,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            perf_counter=time.perf_counter,
        ),
    )

    class ExpiringManager(FakeTokenizerManager):
        async def generate_request(self, request, raw_request):
            self.requests.append(request)
            clock[0] = deadline + 1
            yield [option_output()]

    manager = ExpiringManager()
    runner = option_runner(manager)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            runner.run_option_score_batch(
                [job(deadline_monotonic=deadline)], [(10, 11)], (32, 33)
            )
        )
    assert manager.aborted == ["job-0"]
    assert not any(runner._lane_active.values())
    assert not runner._cancelled_parents
