import asyncio
import json
from types import SimpleNamespace

import pytest

from qwen_exo_booster.internal_jobs import InternalJobResult
from qwen_exo_booster.mid_think_questions import MidThinkQuestionService


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "<sys>" + messages[0]["content"] + "<user>" + messages[1]["content"]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) % 512 for c in str(text)]


class _Runner:
    """Returns a fixed JSON body and records how many jobs each call scored."""

    def __init__(self, body):
        self._body = body
        self.calls = []

    async def run_batch(self, jobs, prompts, sampling_params):
        self.calls.append((tuple(jobs), tuple(prompts), dict(sampling_params)))
        return tuple(
            InternalJobResult(
                job=job,
                text=self._body,
                finish_reason={"type": "stop"},
                prompt_tokens=len(prompt),
                completion_tokens=32,
                latency_seconds=0.01,
            )
            for job, prompt in zip(jobs, prompts)
        )


def _service(body, **kwargs):
    return MidThinkQuestionService(
        _Runner(body), _Tokenizer(), SimpleNamespace(emit=lambda *a, **k: None), **kwargs
    )


def test_generates_exactly_n_divergent_questions_in_one_batch():
    body = '{"questions":["其一?","其二?","其三?","多余?"]}'
    service = _service(body, question_count=3)
    result = asyncio.run(
        service.generate(
            parent_request_id="r1",
            turn_id="r1",
            event_id="r1:mid-think:1",
            user_question="原始任务",
            partial_output="卡住的推理",
        )
    )
    assert result is not None
    # Capped to the requested count even when the model over-produces.
    assert result.questions == ("其一?", "其二?", "其三?")
    # A single job (no per-question fan-out, no retrieval), and the schema was
    # attached so the model must emit parseable JSON.
    assert len(service.runner.calls[0][0]) == 1
    assert "json_schema" in service.runner.calls[0][2]
    # The injected block lists the questions; it is what lands at the boundary.
    text = result.injection_text
    assert "其一?" in text and "其三?" in text and "多余?" not in text


def test_long_question_is_kept_up_to_200_chars_not_truncated_at_120():
    # Regression: an earlier _MAX_QUESTION_CHARS=120 chopped whole sentences
    # mid-word. A concrete cutoff question up to 200 chars must survive intact.
    question = "Did I verify the one load-bearing assumption " + "x" * 100 + "?"
    assert 120 < len(question) <= 200
    body = json.dumps({"questions": [question]}, ensure_ascii=False)
    result = asyncio.run(
        _service(body, question_count=1).generate(
            parent_request_id="r1",
            turn_id="r1",
            event_id=None,
            user_question="task",
            partial_output="stuck reasoning",
        )
    )
    assert result is not None
    assert result.questions == (question,)
    assert question in result.injection_text


def test_empty_or_unparseable_output_yields_no_injection():
    # Fails open: an unusable generation must not inject a malformed block.
    assert (
        asyncio.run(
            _service("not json").generate(
                parent_request_id="r",
                turn_id="r",
                event_id=None,
                user_question="q",
                partial_output="p",
            )
        )
        is None
    )
    assert (
        asyncio.run(
            _service('{"questions":[]}').generate(
                parent_request_id="r",
                turn_id="r",
                event_id=None,
                user_question="q",
                partial_output="p",
            )
        )
        is None
    )


def test_rejects_invalid_construction():
    with pytest.raises(ValueError):
        _service("{}", question_count=0)
    with pytest.raises(ValueError):
        _service("{}", max_output_tokens=8)
