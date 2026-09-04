import asyncio
from types import SimpleNamespace

import pytest

from qwen_exo_booster.internal_jobs import InternalScoreResult
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.pmi_gate import PmiJudgeGate


class _Tokenizer:
    """Character tokenizer with a recognizable chat-template wrapper."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) for c in str(text)]

    def apply_chat_template(self, messages, **_kwargs):
        return "<sys>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<a>"


class _ScoreRunner:
    """Scores a label span by how many of its characters appear in the prefix.

    A question that shares characters with a document head makes that head
    more "predictable"; the neutral prefix shares almost nothing.
    """

    max_fanout = 32

    def __init__(self):
        self.calls = []

    async def run_score_batch(self, jobs, input_ids, label_starts, sampling_params, **_):
        jobs = tuple(jobs)
        inputs = tuple(tuple(x) for x in input_ids)
        starts = tuple(label_starts)
        self.calls.append((len(jobs), tuple(len(x) - s for x, s in zip(inputs, starts))))
        results = []
        for job, ids, start in zip(jobs, inputs, starts):
            prefix = set(ids[:start])
            label = ids[start:]
            logprobs = tuple(-0.5 if t in prefix else -3.0 for t in label)
            results.append(
                InternalScoreResult(
                    job=job,
                    token_logprobs=logprobs,
                    mean_nll=-sum(logprobs) / len(logprobs),
                    prompt_tokens=len(ids),
                    finish_reason={"type": "stop"},
                    latency_seconds=0.01,
                )
            )
        return tuple(results)


def _candidates(tmp_path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    repo.upsert("notes.md", "---\ntitle: notes\n---\nmemory_schema: 3\nscope: x\n笔记整理回读核对")
    repo.upsert("lint.md", "---\ntitle: lint\n---\ntsc错误计数门禁")
    docs = {d.relative_path: d for d in repo.snapshot.documents}
    return repo, tuple(
        repo.candidate_for_document(docs[p].document_id, "q") for p in ("notes.md", "lint.md")
    )


def test_pmi_gate_scores_one_batch_and_caches_the_neutral_term(tmp_path):
    _repo, cands = _candidates(tmp_path)
    runner = _ScoreRunner()
    gate = PmiJudgeGate(runner, _Tokenizer(), threshold=0.0, head_tokens=64)

    first = asyncio.run(
        gate.evaluate(parent_request_id="r1", question="笔记整理时遇到什么问题", candidates=cands)
    )
    second = asyncio.run(
        gate.evaluate(parent_request_id="r2", question="今天天气怎么样", candidates=cands)
    )

    # First call: 2 conditional + 2 neutral rows in ONE batch; second: neutral cached.
    assert runner.calls[0][0] == 4
    assert runner.calls[1][0] == 2
    assert first.status == "passed"
    assert first.max_pmi is not None and first.max_pmi > 0
    notes_id, lint_id = cands[0].candidate_id, cands[1].candidate_id
    assert first.scores[notes_id] > first.scores[lint_id]
    assert second.status == "skipped"
    assert second.skip_judge is True
    assert second.cached_neutral_count == 2
    # The rule-card metadata lines are stripped from the scored head.
    assert "memory_schema" not in gate.rule_card_head(cands[0].normalized_reference_content)


def test_pmi_gate_declines_a_question_longer_than_its_budget(tmp_path):
    """A long question must cost nothing rather than score and always pass.

    Agentic tool turns ask with the original task plus the execution
    trajectory. That prefix overlaps lexically with every rule card, so on live
    traffic all 17 evaluations passed the gate while each spent 5-7s on a
    teacher-forced batch. Declining keeps the judge running (fail open) and
    stops the gate from being pure latency on those turns.
    """
    _repo, cands = _candidates(tmp_path)
    runner = _ScoreRunner()
    gate = PmiJudgeGate(runner, _Tokenizer(), head_tokens=64, max_question_tokens=32)

    result = asyncio.run(
        gate.evaluate(parent_request_id="r", question="笔记" * 200, candidates=cands)
    )

    assert result.status == "not_run_question_too_long"
    assert result.skip_judge is False
    assert runner.calls == []  # nothing was scored


def test_pmi_gate_threshold_and_empty_heads(tmp_path):
    repo, cands = _candidates(tmp_path)
    gate = PmiJudgeGate(_ScoreRunner(), _Tokenizer(), threshold=10.0)
    result = asyncio.run(
        gate.evaluate(parent_request_id="r", question="笔记整理", candidates=cands)
    )
    assert result.status == "skipped"  # nothing clears an absurd threshold

    empty = SimpleNamespace(
        candidate_id="e", reference_digest="d", normalized_reference_content=""
    )
    result = asyncio.run(
        gate.evaluate(parent_request_id="r", question="q", candidates=(empty,))
    )
    assert result.status == "not_run" and result.skip_judge is False
    with pytest.raises(ValueError):
        PmiJudgeGate(_ScoreRunner(), _Tokenizer(), head_tokens=2)
