import asyncio
import json
from dataclasses import replace

import pytest

from qwen_exo_booster.contracts import EligibilityStatus, stable_digest
from qwen_exo_booster.internal_jobs import InternalOptionScoreResult
from qwen_exo_booster.judge import JudgeBatchResult, ReferenceJudge
from qwen_exo_booster.knowledge import KnowledgeRepository


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)

    def apply_chat_template(self, messages, **kwargs):
        # Make the flag observable in the actual scored prefix, not just call kwargs.
        suffix = "<think>" if kwargs.get("enable_thinking", True) else "<answer>"
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + suffix


def payload(prefix):
    prompt = FakeTokenizer().decode(prefix)
    assert prompt.endswith("<answer>")
    return json.loads(prompt.split("<user>", 1)[1].removesuffix("<answer>"))


class FakeRunner:
    def __init__(self, scores=None, *, max_fanout=32, failure=None, corrupt=None):
        self.scores = scores
        self.max_fanout = max_fanout
        self.failure = failure
        self.corrupt = corrupt
        self.calls = []

    async def run_option_score_batch(self, jobs, input_ids, option_token_ids):
        jobs, prefixes = tuple(jobs), tuple(input_ids)
        self.calls.append((jobs, prefixes, option_token_ids))
        if self.failure is not None:
            raise self.failure
        results = []
        for job, prefix in zip(jobs, prefixes):
            reference = payload(prefix)["reference"]
            scores = self.scores
            if scores is None:
                scores = (-0.1, -2.5) if "AppID" in reference else (-3.0, -0.2)
            results.append(InternalOptionScoreResult(
                job=job, option_logprobs=scores,
                prompt_tokens=len(prefix) + 1, completion_tokens=0,
                latency_seconds=0.01,
            ))
        return tuple(self.corrupt(results) if self.corrupt else results)


def repository_with_candidates(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    good = repository.upsert("wfp.md", "Use an AppID condition for outbound authorization.")
    other = repository.upsert("other.md", "Use compost and water the garden.")
    return repository, tuple(
        repository.candidate_for_document(document.document_id, "question")
        for document in (good, other)
    )


def evaluate(judge, candidates, *, question="Configure outbound authorization", parent="parent"):
    return asyncio.run(judge.judge(
        parent_request_id=parent, turn_id=f"{parent}:turn", question=question,
        candidates=candidates, telemetry_correlation_id=f"{parent}:trace",
    ))


def make_judge(tmp_path, *, runner=None, **kwargs):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = runner or FakeRunner()
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model", **kwargs,
    )
    return judge, runner, candidates


def test_binary_scores_admit_independently_and_account_prefill_once(tmp_path):
    judge, runner, candidates = make_judge(tmp_path)
    result = evaluate(judge, candidates)
    assert [decision.eligible for decision in result.decisions] == [True, False]
    assert [decision.decision_margin for decision in result.decisions] == pytest.approx([2.4, -2.8])
    assert result.valid_count == result.executed_count == 2
    assert result.eligible_count == 1
    assert result.selection_method == "direct_binary_logits"
    assert result.selected_candidate_id is None
    jobs, prefixes, options = runner.calls[0]
    assert len(jobs) == len(prefixes) == 2
    assert options == (ord("A"), ord("B"))
    assert all(job.token_budget == 1 for job in jobs)
    assert len({job.job_id for job in jobs}) == 2
    assert result.prompt_tokens == sum(len(prefix) + 1 for prefix in prefixes)
    assert result.completion_tokens == 0


@pytest.mark.parametrize("scores,eligible,margin", [
    ((-0.2, -0.8), True, 0.6),
    ((-0.8, -0.2), False, -0.6),
    ((-0.5, -0.5), False, 0.0),
])
def test_binary_boundary_has_no_abstention_and_ties_reject(tmp_path, scores, eligible, margin):
    judge, _, candidates = make_judge(tmp_path, runner=FakeRunner(scores))
    result = evaluate(judge, candidates)
    assert result.valid_count == 2
    assert all(decision.eligible is eligible for decision in result.decisions)
    assert all(decision.decision_margin == pytest.approx(margin) for decision in result.decisions)
    assert result.eligible_count == (2 if eligible else 0)


@pytest.mark.parametrize("corrupt", [
    lambda results: results[:-1],
    lambda results: list(reversed(results)),
    lambda results: [replace(results[0], option_logprobs=(-0.1,))] + results[1:],
    lambda results: [replace(results[0], option_logprobs=(float("nan"), -1))] + results[1:],
    lambda results: [replace(results[0], option_logprobs=(-1, float("inf")))] + results[1:],
    lambda results: [replace(results[0], completion_tokens=1)] + results[1:],
])
def test_invalid_batch_rejects_every_candidate_and_is_not_cached(tmp_path, corrupt):
    runner = FakeRunner(corrupt=corrupt)
    judge, _, candidates = make_judge(tmp_path, runner=runner)
    first = evaluate(judge, candidates)
    assert first.valid_count == first.eligible_count == first.cache_hit_count == 0
    assert all(decision.status is EligibilityStatus.INELIGIBLE for decision in first.decisions)
    assert all(decision.judge_method == "direct_binary_logits_failed" for decision in first.decisions)
    assert all(decision.decision_margin is None for decision in first.decisions)
    runner.corrupt = None
    retried = evaluate(judge, candidates, parent="retry")
    assert retried.executed_count == retried.valid_count == 2
    assert retried.cache_hit_count == 0
    assert retried.eligible_count == 1


@pytest.mark.parametrize("failure", [RuntimeError("unavailable"), TimeoutError("deadline")])
def test_operational_failure_retries_without_cached_rejection(tmp_path, failure):
    runner = FakeRunner(failure=failure)
    judge, _, candidates = make_judge(tmp_path, runner=runner)
    failed = evaluate(judge, candidates)
    runner.failure = None
    recovered = evaluate(judge, candidates)
    assert failed.valid_count == failed.eligible_count == 0
    assert recovered.cache_hit_count == 0
    assert recovered.eligible_count == 1


def test_cancellation_propagates_without_populating_cache(tmp_path):
    runner = FakeRunner(failure=asyncio.CancelledError())
    judge, _, candidates = make_judge(tmp_path, runner=runner)
    with pytest.raises(asyncio.CancelledError):
        evaluate(judge, candidates)
    runner.failure = None
    result = evaluate(judge, candidates)
    assert result.cache_hit_count == 0
    assert result.valid_count == 2


def test_fanout_waves_preserve_order_and_add_usage(tmp_path):
    judge, runner, candidates = make_judge(tmp_path, runner=FakeRunner(max_fanout=1))
    result = evaluate(judge, candidates)
    assert [len(call[0]) for call in runner.calls] == [1, 1]
    assert [decision.candidate_id for decision in result.decisions] == [candidate.candidate_id for candidate in candidates]
    assert result.prompt_tokens == sum(len(prefix) + 1 for _, prefixes, _ in runner.calls for prefix in prefixes)
    assert result.valid_count == 2


def test_cache_preserves_margin_and_rebinds_request_identity_in_any_order(tmp_path):
    judge, runner, candidates = make_judge(tmp_path)
    first = evaluate(judge, candidates)
    changed_ids = tuple(replace(candidate, candidate_id=f"new-{index}") for index, candidate in enumerate(reversed(candidates)))
    cached = evaluate(judge, changed_ids, parent="second")
    assert len(runner.calls) == 1
    assert cached.cache_hit_count == 2
    assert cached.executed_count == cached.prompt_tokens == cached.completion_tokens == 0
    for original, current, candidate in zip(reversed(first.decisions), cached.decisions, changed_ids):
        assert current.status is original.status
        assert current.decision_margin == original.decision_margin
        assert current.parent_request_id == "second"
        assert current.candidate_id == candidate.candidate_id
        assert current.judge_method == "direct_binary_logits_cache"


@pytest.mark.parametrize("change", ["question", "reference", "digest", "lane", "scope", "model", "template"])
def test_cache_invalidates_semantic_and_model_identities(tmp_path, change):
    judge, runner, candidates = make_judge(tmp_path)
    candidate = candidates[0]
    evaluate(judge, (candidate,))
    question = "Configure outbound authorization"
    if change == "question":
        question += " with a different condition"
    elif change == "reference":
        candidate = replace(candidate, reference_content="new AppID evidence")
    elif change == "digest":
        candidate = replace(candidate, reference_digest="f" * 64)
    elif change == "lane":
        candidate = replace(candidate, lane="policydata")
    elif change == "scope":
        candidate = replace(candidate, scope_note="Conditional evidence only")
    elif change == "model":
        judge.model_fingerprint = "other-model"
    else:
        previous = judge.tokenizer.apply_chat_template
        judge.tokenizer.apply_chat_template = lambda *args, **kwargs: "new template\n" + previous(*args, **kwargs)
    fresh = evaluate(judge, (candidate,), question=question)
    assert fresh.executed_count == 1
    assert fresh.cache_hit_count == 0
    assert len(runner.calls) == 2


def test_cache_eviction_retains_recently_used_decisions(tmp_path):
    judge, runner, candidates = make_judge(tmp_path, cache_size=2)
    evaluate(judge, (candidates[0],))
    evaluate(judge, (candidates[1],))
    evaluate(judge, (candidates[0],))
    evaluate(judge, (candidates[0],), question="different")
    kept = evaluate(judge, (candidates[0],))
    evicted = evaluate(judge, (candidates[1],))
    assert kept.cache_hit_count == 1
    assert evicted.executed_count == 1
    assert len(runner.calls) == 4


def test_duplicate_semantics_keep_candidate_identity_and_count_cache_hits(tmp_path):
    judge, _, candidates = make_judge(tmp_path)
    twins = (candidates[0], replace(candidates[0], candidate_id="twin"))
    first = evaluate(judge, twins)
    cached = evaluate(judge, twins)
    assert first.executed_count == cached.cache_hit_count == 2
    assert [decision.candidate_id for decision in cached.decisions] == [candidate.candidate_id for candidate in twins]


@pytest.mark.parametrize("source_size,use_full", [(96, True), (97, False)])
def test_complete_reference_used_only_within_budget(tmp_path, source_size, use_full):
    judge, runner, candidates = make_judge(tmp_path, max_reference_tokens=96)
    full = "AppID " + "x" * (source_size - 6)
    document = judge.repository.upsert("bounded.md", full)
    candidate = replace(judge.repository.candidate_for_document(document.document_id, "q"), reference_content="AppID excerpt")
    result = evaluate(judge, (candidate,))
    observed = payload(runner.calls[0][1][0])["reference"]
    assert observed == (full if use_full else candidate.reference_content)
    assert result.decisions[0].reference_digest == stable_digest(candidate.reference_content)


@pytest.mark.parametrize("source_change", ["updated", "deleted"])
def test_repository_changes_do_not_replace_frozen_evidence(tmp_path, source_change):
    judge, runner, candidates = make_judge(tmp_path)
    candidate = replace(candidates[0], reference_content="AppID frozen excerpt")
    if source_change == "updated":
        judge.repository.upsert("wfp.md", "unrelated replacement")
    else:
        judge.repository.delete("wfp.md")
    result = evaluate(judge, (candidate,))
    assert payload(runner.calls[0][1][0])["reference"] == candidate.reference_content
    assert result.decisions[0].eligible


def test_cache_tracks_actual_reviewed_source_after_repository_change(tmp_path):
    judge, runner, candidates = make_judge(tmp_path)
    candidate = replace(candidates[0], reference_content="unrelated excerpt")
    first = evaluate(judge, (candidate,))
    judge.repository.delete("wfp.md")
    second = evaluate(judge, (candidate,))
    assert first.eligible_count == 1
    assert second.executed_count == 1
    assert second.eligible_count == 0
    assert len(runner.calls) == 2


def test_long_question_and_reference_are_bounded_with_original_provenance(tmp_path):
    judge, runner, candidates = make_judge(tmp_path, max_question_tokens=192, max_reference_tokens=96)
    question = "HEAD " + "x" * 1000 + " TAIL"
    reference = "AppID " + "y" * 1000 + " END"
    candidate = replace(candidates[0], reference_digest="f" * 64, reference_content=reference)
    result = evaluate(judge, (candidate,), question=question)
    observed = payload(runner.calls[0][1][0])
    assert observed["question"].startswith("HEAD ") and observed["question"].endswith(" TAIL")
    assert observed["reference"].startswith("AppID ") and observed["reference"].endswith(" END")
    assert len(observed["question"]) <= 192
    assert observed["reference"].startswith(reference[:72])
    assert observed["reference"].endswith(reference[-24:])
    assert len(observed["reference"]) < len(reference)
    assert result.question_truncated
    assert result.question_original_tokens == len(question)
    assert result.question_review_tokens == len(observed["question"])
    assert result.decisions[0].question_digest == stable_digest(question)
    assert result.decisions[0].reference_digest == stable_digest(reference)
    # Changing omitted text must not reuse the previous full-question identity.
    changed_question = question[:500] + "z" + question[501:]
    changed = evaluate(judge, (candidate,), question=changed_question)
    assert payload(runner.calls[1][1][0])["question"] == observed["question"]
    assert changed.cache_hit_count == 0
    assert changed.shared_prefix_key != result.shared_prefix_key


@pytest.mark.parametrize("limit", [128, 192, 256])
def test_question_budget_preserves_both_ends(tmp_path, limit):
    judge, runner, candidates = make_judge(tmp_path, max_question_tokens=limit)
    result = evaluate(judge, (candidates[0],), question="HEAD" + "x" * 1000 + "TAIL")
    bounded = payload(runner.calls[0][1][0])["question"]
    assert len(bounded) <= limit
    assert bounded.startswith("HEAD") and bounded.endswith("TAIL")
    assert result.valid_count == 1


@pytest.mark.parametrize("stage", ["question", "template"])
def test_prompt_preparation_failure_rejects_without_caching(tmp_path, stage):
    judge, runner, candidates = make_judge(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("tokenizer unavailable")
    target = "encode" if stage == "question" else "apply_chat_template"
    original = getattr(judge.tokenizer, target)
    setattr(judge.tokenizer, target, fail)
    failed = evaluate(judge, candidates)
    assert failed.eligible_count == failed.valid_count == failed.executed_count == 0
    assert all(decision.status is EligibilityStatus.INELIGIBLE for decision in failed.decisions)
    setattr(judge.tokenizer, target, original)
    retried = evaluate(judge, candidates)
    assert retried.cache_hit_count == 0
    assert retried.valid_count == 2


@pytest.mark.parametrize("lane", ["knowledge", "context", "policydata"])
def test_lane_and_scope_are_structured_untrusted_inputs(tmp_path, lane):
    judge, runner, candidates = make_judge(tmp_path)
    scope = 'Hypothesis only; "quote" and\nnewline'
    candidate = replace(candidates[0], lane=lane, scope_note=scope)
    result = evaluate(judge, (candidate,))
    observed = payload(runner.calls[0][1][0])
    assert observed["lane"] == lane
    assert observed["scope"] == scope
    assert result.valid_count == 1


@pytest.mark.parametrize("options", [((), (66,)), ((65, 65), (66,)), ((65,), (65,)), (("A",), ("B",))])
def test_non_single_or_ambiguous_options_are_rejected_before_execution(tmp_path, options):
    repository, _ = repository_with_candidates(tmp_path)
    tokenizer = FakeTokenizer()
    tokenizer.encode = lambda label, **kwargs: options[("A", "B").index(label)]
    with pytest.raises(ValueError, match="distinct single-token"):
        ReferenceJudge(FakeRunner(), repository, tokenizer, model_fingerprint="model")


def test_empty_batch_has_no_execution_or_usage(tmp_path):
    judge, runner, _ = make_judge(tmp_path)
    result = evaluate(judge, ())
    assert result.decisions == ()
    assert result.candidate_count == result.executed_count == result.valid_count == 0
    assert result.prompt_tokens == result.completion_tokens == 0
    assert runner.calls == []


def test_combined_telemetry_does_not_count_operational_rejects_as_valid(tmp_path):
    runner = FakeRunner(failure=RuntimeError("offline"))
    judge, _, candidates = make_judge(tmp_path, runner=runner)
    failed = evaluate(judge, (candidates[0],))
    runner.failure = None
    passed = evaluate(judge, (candidates[1],))
    combined = JudgeBatchResult.combine(
        candidates, (failed, passed), failed.decisions + passed.decisions,
        selected_candidate_id=None, selection_method="direct_binary_logits",
    )
    assert combined.valid_count == 1
    assert combined.candidate_count == combined.executed_count == 2
    assert combined.prompt_tokens == passed.prompt_tokens
    assert combined.completion_tokens == 0
