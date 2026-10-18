import asyncio
import math
import re
from types import SimpleNamespace

import pytest
from qwen_exo_booster.internal_jobs import InternalJobRunner
from qwen_exo_booster.system_one import (
    SystemOneError,
    SystemOneEvaluator,
    evaluate_system_one_request,
    parse_system_one_request,
)

class FakeTokenizer:
    """One token per character, so every option letter is a single token."""

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"


class FakeTokenizerManager:
    """Prefers the option letter named by a ``PREFER:X`` marker in the prompt."""

    def __init__(self):
        self.requests = []
        self.aborted = []

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        outputs = []
        for rid, input_ids, options in zip(
            request.rid, request.input_ids, request.token_ids_logprob
        ):
            prefix = input_ids[:-1]
            match = re.search(r"PREFER:([A-Z])", "".join(map(chr, prefix)))
            preferred = ord(match.group(1)) if match else None
            outputs.append(
                {
                    "text": "",
                    "output_ids": [],
                    "meta_info": {
                        "id": rid,
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": 0,
                        "finish_reason": {"type": "length", "length": 0},
                        "input_token_logprobs": [
                            (None, prefix[-1], None),
                            (-8.0, 0, None),
                        ],
                        "input_token_ids_logprobs": [
                            None,
                            [
                                (-0.1 if token == preferred else -5.0, token, None)
                                for token in options
                            ],
                        ],
                    },
                }
            )
        yield outputs

    def abort_request(self, rid):
        self.aborted.append(rid)


class RequestConnection:
    def __init__(self):
        self.disconnected = asyncio.Event()
        self.watching = asyncio.Event()
        self.closed = asyncio.Event()

    async def receive(self):
        self.watching.set()
        try:
            await self.disconnected.wait()
            return {"type": "http.disconnect"}
        finally:
            self.closed.set()


class ControlledTokenizerManager(FakeTokenizerManager):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Queue()
        self.stopped = asyncio.Queue()
        self.dispatched = []
        self.gates = {}
        self.running = set()

    async def generate_request(self, request, raw_request):
        self.dispatched.append(request)
        self.gates[request.rid[0]] = gate = asyncio.Event()
        self.running.update(request.rid)
        self.started.put_nowait(request)
        try:
            await gate.wait()
            async for output in super().generate_request(request, raw_request):
                yield output
        finally:
            self.running.difference_update(request.rid)
            self.stopped.put_nowait(request)


def evaluator(max_fanout=4, **kwargs):
    manager = FakeTokenizerManager()
    runner = InternalJobRunner(
        manager,
        max_fanout=max_fanout,
        max_tokens_per_parent=64,
        request_factory=lambda **kw: SimpleNamespace(**kw),
    )
    return SystemOneEvaluator(runner, FakeTokenizer(), **kwargs), manager, runner


def run(coro):
    return asyncio.run(coro)


def test_noul_returns_probability_and_boolean_verdict():
    system_one, manager, runner = evaluator()
    response = run(
        system_one.evaluate(
            {
                "model": "jev-latest",
                "state": {"order": 42, "status": "refunded"},
                "questions": {
                    "is_refunded": {"type": "noul", "instructions": "PREFER:A refunded?"},
                    "is_open": {
                        "type": "noul",
                        "instructions": "PREFER:B open?",
                        "criteria": {"true": "still open", "false": "closed"},
                    },
                },
            }
        )
    )
    answers = response["answers"]
    assert list(answers) == ["is_refunded", "is_open"]
    assert answers["is_refunded"]["type"] == "noul"
    assert answers["is_refunded"]["verdict"] is True
    assert answers["is_refunded"]["noul"] > 0.95
    assert answers["is_open"]["verdict"] is False
    assert answers["is_open"]["noul"] < 0.05
    assert response["usage"]["input_tokens"] > 0
    assert response["usage"]["output_tokens"] == 0
    # Both questions share one prefill batch, one cache namespace, target-only.
    assert len(manager.requests) == 1
    request = manager.requests[0]
    assert len(set(request.extra_key)) == 1
    assert request.extra_key[0].startswith("qwen-exo:v1:system-one:")
    assert all(p["max_new_tokens"] == 0 for p in request.sampling_params)
    assert all(
        p["custom_params"]["qwen_exo_kind"] == "internal"
        and p["custom_params"]["qwen_exo_job_type"] == "system_one"
        for p in request.sampling_params
    )
    assert not runner._active and not runner._reserved_tokens


def test_choice_and_score_follow_typesafe_shapes():
    system_one, _, _ = evaluator()
    response = run(
        system_one.evaluate(
            {
                "model": "jev-latest",
                "state": "The export button crashes Safari. PREFER:C",
                "questions": {
                    "label": {
                        "type": "choice",
                        "criteria": {"ui": "UI", "perf": None, "crash": "Crash"},
                    },
                    "severity": {
                        "type": "score",
                        "instructions": "How severe?",
                        "criteria": ["cosmetic", "degraded", "blocking"],
                    },
                    "single": {"type": "score", "criteria": ["only"]},
                },
            }
        )
    )
    label = response["answers"]["label"]
    assert label["choice"] == "crash"
    assert set(label["probabilities"]) == {"ui", "perf", "crash"}
    assert math.isclose(sum(label["probabilities"].values()), 1.0)
    assert 0 < label["confidence"] <= 1
    severity = response["answers"]["severity"]
    assert severity["legend"] == {"0": "cosmetic", "1": "degraded", "2": "blocking"}
    assert set(severity["probabilities"]) == {"0", "1", "2"}
    assert 1.9 < severity["score"] <= 2.0
    assert response["answers"]["single"] == {
        "type": "score",
        "score": 0.0,
        "confidence": 1.0,
        "legend": {"0": "only"},
        "probabilities": {"0": 1.0},
    }


def test_questions_beyond_fanout_run_in_waves():
    system_one, manager, runner = evaluator(max_fanout=2)
    questions = {
        f"q{i}": {"type": "noul", "instructions": "PREFER:A"} for i in range(5)
    }
    response = run(
        system_one.evaluate({"model": "m", "state": "s", "questions": questions})
    )
    assert list(response["answers"]) == list(questions)
    assert [len(r.rid) for r in manager.requests] == [2, 2, 1]
    assert not runner._reserved_tokens


@pytest.mark.parametrize(
    "payload,status",
    [
        ([], 400),
        ({"state": "s", "questions": {"q": {"type": "noul"}}}, 400),
        ({"model": "m", "questions": {"q": {"type": "noul"}}}, 400),
        ({"model": "m", "state": None, "questions": {"q": {"type": "noul"}}}, 400),
        ({"model": "m", "state": "s", "questions": {}}, 400),
        ({"model": "m", "state": "s", "questions": {"q": {"type": "bool"}}}, 400),
        ({"model": "m", "state": "s", "questions": {"q": {"type": "choice"}}}, 400),
        ({"model": "m", "state": "s", "questions": {"q": {"type": "score", "criteria": []}}}, 400),
        ({"model": "m", "state": "s", "questions": {"q": {"type": "noul", "criteria": [1]}}}, 400),
        ({"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}, "stream": True}, 400),
        ({"model": "m", "state": "s", "questions": {f"q{i}": {"type": "noul"} for i in range(3)}}, 413),
    ],
)
def test_invalid_requests_are_rejected_before_dispatch(payload, status):
    system_one, manager, _ = evaluator(max_questions=2)
    with pytest.raises(SystemOneError) as caught:
        run(system_one.evaluate(payload))
    assert caught.value.status_code == status
    assert caught.value.public_dict()["error"]["message"]
    assert not manager.requests


def test_overlong_prompt_is_rejected_with_413():
    system_one, manager, _ = evaluator(max_prompt_tokens=200)
    with pytest.raises(SystemOneError) as caught:
        run(
            system_one.evaluate(
                {"model": "m", "state": "x" * 500, "questions": {"q": {"type": "noul"}}}
            )
        )
    assert caught.value.status_code == 413
    assert not manager.requests


def test_parse_accepts_typesafe_examples():
    model, state, questions = parse_system_one_request(
        {
            "model": "jev-latest",
            "state": ["sample"],
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": None,
                    "criteria": {"a": {"description": "A"}, "b": ["B"], "c": None},
                }
            },
            "stream": False,
        },
        max_questions=8,
    )
    assert model == "jev-latest"
    assert state == ["sample"]
    assert [key for key, _ in questions[0].options] == ["a", "b", "c"]


def test_long_state_scores_one_question_first_to_seed_prefix_cache():
    system_one, manager, _ = evaluator(max_fanout=4)
    questions = {f"q{i}": {"type": "noul", "instructions": "PREFER:A"} for i in range(5)}
    run(system_one.evaluate({"model": "m", "state": "x" * 600, "questions": questions}))
    assert [len(r.rid) for r in manager.requests] == [1, 4]


@pytest.mark.parametrize("cancel_kind", ["disconnect", "handler"])
@pytest.mark.parametrize("waiting_for_capacity", [False, True])
def test_request_cancellation_stops_own_jobs_without_affecting_other_parent(
    cancel_kind, waiting_for_capacity
):
    async def scenario():
        baseline = asyncio.all_tasks()
        system_one, _, runner = evaluator(
            max_fanout=1 if waiting_for_capacity else 2
        )
        runner.tokenizer_manager = manager = ControlledTokenizerManager()
        admissions = asyncio.Queue()
        reserve = runner._reserve

        async def observe_reserve(parent_request_id, jobs):
            admissions.put_nowait(parent_request_id)
            await reserve(parent_request_id, jobs)

        runner._reserve = observe_reserve
        survivor_connection = RequestConnection()
        survivor = asyncio.create_task(
            evaluate_system_one_request(
                system_one,
                {
                    "model": "m",
                    "state": "PREFER:A",
                    "questions": {"keep": {"type": "noul"}},
                },
                survivor_connection,
            )
        )
        survivor_request = await asyncio.wait_for(manager.started.get(), timeout=5)
        survivor_parent = admissions.get_nowait()
        victim_connection = RequestConnection()
        victim = asyncio.create_task(
            evaluate_system_one_request(
                system_one,
                {
                    "model": "m",
                    "state": "x" * 600,
                    "questions": {f"q{i}": {"type": "noul"} for i in range(3)},
                },
                victim_connection,
            )
        )
        victim_parent = await asyncio.wait_for(admissions.get(), timeout=5)
        victim_request = None
        if not waiting_for_capacity:
            victim_request = await asyncio.wait_for(manager.started.get(), timeout=5)
        await asyncio.wait_for(victim_connection.watching.wait(), timeout=5)
        if cancel_kind == "disconnect":
            victim_connection.disconnected.set()
            with pytest.raises(SystemOneError) as caught:
                await asyncio.wait_for(victim, timeout=5)
            assert caught.value.status_code == 499
            assert caught.value.code == "client_disconnected"
        else:
            victim.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(victim, timeout=5)

        assert victim_connection.closed.is_set()
        assert manager.running == set(survivor_request.rid)
        assert runner._active == {survivor_parent: set(survivor_request.rid)}
        assert runner._reserved_tokens == {survivor_parent: 1}
        assert runner._lane_active["foreground"] == 1
        assert set(runner._active_lanes) == {
            (survivor_parent, rid) for rid in survivor_request.rid
        }
        assert victim_parent not in runner._cancelled_parents
        assert set(manager.aborted) == (
            set(victim_request.rid) if victim_request is not None else set()
        )
        assert not survivor.done()

        manager.gates[survivor_request.rid[0]].set()
        response = await asyncio.wait_for(survivor, timeout=5)
        assert response["model"] == "qwen-exo-jev"
        assert response["answers"]["keep"]["verdict"] is True
        assert response["usage"]["output_tokens"] == 0
        assert survivor_connection.closed.is_set()
        # Opening capacity must not dispatch a cancelled waiter or later waves.
        assert manager.dispatched == [
            survivor_request,
            *([victim_request] if victim_request is not None else []),
        ]
        assert not manager.running
        assert not runner._active
        assert not runner._reserved_tokens
        assert not runner._active_lanes
        assert not any(runner._lane_active.values())
        assert asyncio.all_tasks() == baseline

    run(scenario())


def test_repeated_handler_cancellation_waits_for_runner_cleanup():
    async def scenario():
        baseline = asyncio.all_tasks()
        system_one, _, runner = evaluator(max_fanout=1)
        runner.tokenizer_manager = manager = ControlledTokenizerManager()
        connection = RequestConnection()
        handler = asyncio.create_task(
            evaluate_system_one_request(
                system_one,
                {"model": "m", "state": "s", "questions": {"q": {"type": "noul"}}},
                connection,
            )
        )
        request = await asyncio.wait_for(manager.started.get(), timeout=5)
        # Hold admission's lock so release and finish_parent cannot complete yet.
        await runner._lock.acquire()
        try:
            handler.cancel()
            await asyncio.wait_for(connection.closed.wait(), timeout=5)
            await asyncio.wait_for(manager.stopped.get(), timeout=5)
            handler.cancel()
            await asyncio.sleep(0)
            assert not handler.done()
        finally:
            runner._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(handler, timeout=5)
        assert set(manager.aborted) == set(request.rid)
        assert not manager.running
        assert not runner._active
        assert not runner._reserved_tokens
        assert not runner._active_lanes
        assert not any(runner._lane_active.values())
        assert asyncio.all_tasks() == baseline

    run(scenario())


@pytest.mark.parametrize("cancel_kind", ["disconnect", "handler"])
def test_cancellation_after_parent_finalization_started_releases_reservations(cancel_kind):
    async def scenario():
        baseline = asyncio.all_tasks()
        system_one, _, runner = evaluator()
        entered = asyncio.Event()
        finish_parent = runner.finish_parent

        async def blocked_finalization(parent_id):
            await runner._lock.acquire()
            entered.set()
            await finish_parent(parent_id)

        runner.finish_parent = blocked_finalization
        connection = RequestConnection()
        handler = asyncio.create_task(evaluate_system_one_request(
            system_one,
            {"model": "m", "state": "s", "questions": {
                "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}
            }},
            connection,
        ))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            if cancel_kind == "disconnect":
                connection.disconnected.set()
            else:
                handler.cancel()
            await asyncio.wait_for(connection.closed.wait(), timeout=5)
            assert not handler.done()
        finally:
            runner._lock.release()
        if cancel_kind == "disconnect":
            with pytest.raises(SystemOneError) as error:
                await asyncio.wait_for(handler, timeout=5)
            assert error.value.status_code == 499
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(handler, timeout=5)
        assert not runner._reserved_tokens
        assert asyncio.all_tasks() == baseline

    run(scenario())


@pytest.mark.parametrize("disconnect_ready", [False, True])
def test_completed_request_wins_disconnect_race_without_leaking_watcher(disconnect_ready):
    async def scenario():
        baseline = asyncio.all_tasks()
        system_one, manager, runner = evaluator()
        connection = RequestConnection()
        if disconnect_ready:
            evaluate = system_one.evaluate

            async def complete_then_disconnect(payload):
                result = await evaluate(payload)
                connection.disconnected.set()
                return result

            system_one.evaluate = complete_then_disconnect
        response = await evaluate_system_one_request(
            system_one,
            {
                "model": "m",
                "state": "s",
                "questions": {"q": {"type": "choice", "criteria": {"only": "Only"}}},
            },
            connection,
        )
        assert response == {
            "model": "qwen-exo-jev",
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "only",
                    "confidence": 1.0,
                    "probabilities": {"only": 1.0},
                }
            },
            "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0},
        }
        assert connection.closed.is_set()
        assert not manager.requests
        assert not manager.aborted
        assert not runner._active
        assert not runner._reserved_tokens
        assert asyncio.all_tasks() == baseline

    run(scenario())
