import asyncio
import math
import re
from types import SimpleNamespace

import pytest
from qwen_exo_booster.decision_context import CanonicalDecisionContext
from qwen_exo_booster.internal_jobs import InternalJobRunner
from qwen_exo_booster.system_one import (
    SystemOneError,
    SystemOneEvaluator,
    parse_system_one_request,
    render_shared_decision_prompt,
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


def test_cross_role_prompts_share_exact_canonical_prefix_tokens():
    tokenizer = FakeTokenizer()
    text = '{"input":[{"role":"user","content":"' + ("x" * 2048) + '"}]}'
    context = CanonicalDecisionContext(
        text=text,
        token_ids=tuple(tokenizer.encode(text, add_special_tokens=False)),
        digest="context-digest",
        lineage_namespace="request:test",
    )
    system = render_shared_decision_prompt(tokenizer, context, "SYSTEM ONE OPTIONS: A/B")
    judge = render_shared_decision_prompt(tokenizer, context, "REFERENCE JUDGE: candidate")
    assert system.prefix_token_count == judge.prefix_token_count
    assert system.token_ids[: system.prefix_token_count] == judge.token_ids[
        : judge.prefix_token_count
    ]
    assert system.token_ids[system.prefix_token_count :] != judge.token_ids[
        judge.prefix_token_count :
    ]
    assert system.prefix_digest == judge.prefix_digest
