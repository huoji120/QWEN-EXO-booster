import asyncio
import json

from qwen_exo_booster import judge as judge_module
from qwen_exo_booster.contracts import stable_digest
from qwen_exo_booster.internal_jobs import InternalOptionScoreResult
from qwen_exo_booster.judge import ReferenceJudge
from qwen_exo_booster.knowledge import KnowledgeRepository


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, **kwargs):
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<answer>"


class FakeRunner:
    max_fanout = 32

    def __init__(self):
        self.calls = 0

    async def run_option_score_batch(self, jobs, input_ids, option_token_ids):
        self.calls += 1
        results = []
        for job, prefix in zip(jobs, input_ids):
            prompt = "".join(chr(token) for token in prefix)
            reference = json.loads(prompt.split("<user>", 1)[1].removesuffix("<answer>"))["reference"]
            scores = (-0.1, -2.5) if "AppID" in reference else (-3.0, -0.2)
            results.append(InternalOptionScoreResult(
                job=job, option_logprobs=scores, prompt_tokens=len(prefix) + 1,
                completion_tokens=0, latency_seconds=0.01,
            ))
        return tuple(results)


def make_judge(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    good = repository.upsert("wfp.md", "Use an AppID condition for outbound authorization.")
    other = repository.upsert("other.md", "Use compost and water the garden.")
    candidates = tuple(
        repository.candidate_for_document(document.document_id, "question")
        for document in (good, other)
    )
    runner = FakeRunner()
    return ReferenceJudge(runner, repository, FakeTokenizer(), model_fingerprint="model"), runner, candidates


def evaluate(judge, candidates, *, question, parent, reuse_scope=None):
    return asyncio.run(judge.judge(
        parent_request_id=parent, turn_id=f"{parent}:turn", question=question,
        candidates=candidates, telemetry_correlation_id=f"{parent}:trace",
        reuse_scope=reuse_scope,
    ))


def test_session_reuses_admission_across_agent_turns_and_rebinds_identity(tmp_path):
    """Agent tool turns re-ask a slightly different retrieval question each
    time; within one conversation and task the document admission is reused
    instead of re-prefilling every candidate document."""
    judge, runner, candidates = make_judge(tmp_path)
    first = evaluate(judge, candidates, question="turn 1: configure outbound authorization", parent="turn-1", reuse_scope="conversation-a")
    second = evaluate(judge, candidates, question="turn 2: after tool output", parent="turn-2", reuse_scope="conversation-a")

    assert runner.calls == 1
    assert second.cache_hit_count == 2 and second.executed_count == 0
    for before, after in zip(first.decisions, second.decisions):
        assert after.status is before.status
        assert after.decision_margin == before.decision_margin
        assert after.parent_request_id == "turn-2"
        assert after.question_digest == stable_digest("turn 2: after tool output")
        assert after.judge_method == "direct_binary_logits_session_cache"


def test_session_reuse_is_scoped_and_expires(tmp_path, monkeypatch):
    judge, runner, candidates = make_judge(tmp_path)
    evaluate(judge, candidates, question="q1", parent="a", reuse_scope="conversation-a")
    evaluate(judge, candidates, question="q2", parent="b", reuse_scope="conversation-b")
    assert runner.calls == 2, "another conversation or task never reuses decisions"
    evaluate(judge, candidates, question="q3", parent="c")
    assert runner.calls == 3, "requests without a scope keep per-question judging"

    clock = [judge_module.time.monotonic()]
    monkeypatch.setattr(judge_module.time, "monotonic", lambda: clock[0])
    evaluate(judge, candidates, question="q4", parent="d", reuse_scope="conversation-a")
    assert runner.calls == 3, "an unexpired session decision is reused"
    clock[0] += judge_module._SESSION_DECISION_TTL_SECONDS + 1
    evaluate(judge, candidates, question="q6", parent="f", reuse_scope="conversation-a")
    assert runner.calls == 4, "expired session decisions are judged again"


def test_session_decision_is_refreshed_after_bounded_reuse(tmp_path):
    """A conversation can drift to a new sub-problem without changing its
    original task; a document's admission is re-judged after a bounded number
    of reused turns instead of being frozen for the whole session."""
    judge, runner, candidates = make_judge(tmp_path)
    for turn in range(1 + judge_module._SESSION_DECISION_MAX_REUSES):
        evaluate(judge, candidates, question=f"turn {turn}", parent=f"t{turn}", reuse_scope="conversation-a")
    assert runner.calls == 1
    evaluate(judge, candidates, question="next turn", parent="tn", reuse_scope="conversation-a")
    assert runner.calls == 2
