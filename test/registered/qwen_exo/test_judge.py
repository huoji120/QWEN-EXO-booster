from dataclasses import replace
import json
import asyncio
import os

from qwen_exo_booster.contracts import EligibilityStatus, stable_digest
from qwen_exo_booster.internal_jobs import InternalJobResult
from qwen_exo_booster.judge import (
    ReferenceJudge,
    parse_reference_selection,
    parse_reference_support,
)
from qwen_exo_booster.knowledge import KnowledgeRepository


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, token_ids, **kwargs):
        return " ".join(token_ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"]


class FakeRunner:
    def __init__(self, malformed=False, finish_type="stop", fixed_text=None):
        self.malformed = malformed
        self.finish_type = finish_type
        self.fixed_text = fixed_text
        self.jobs = ()
        self.prompts = ()
        self.sampling_params = None
        self.calls = 0

    async def run_batch(self, jobs, prompts, sampling_params):
        self.calls += 1
        self.jobs = tuple(jobs)
        self.prompts = tuple(prompts)
        self.sampling_params = sampling_params
        results = []
        for job, prompt in zip(self.jobs, self.prompts):
            if self.fixed_text is not None:
                text = self.fixed_text
            elif self.malformed:
                text = '{"supported":true,"extra":1}'
            else:
                text = (
                    '{"supported":true}'
                    if "FWPM_LAYER_ALE_AUTH_CONNECT_V4" in prompt
                    else '{"supported":false}'
                )
            results.append(
                InternalJobResult(
                    job=job,
                    text=text,
                    prompt_tokens=100,
                    completion_tokens=4,
                    finish_reason={"type": self.finish_type},
                    latency_seconds=0.01,
                )
            )
        return tuple(results)


def repository_with_candidates(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert(
        "wfp.md", "Use FWPM_LAYER_ALE_AUTH_CONNECT_V4 with an AppID condition."
    )
    repository.upsert("ctf.md", "Heap exploitation and return oriented programming.")
    wfp = repository.rank("FWPM_LAYER_ALE_AUTH_CONNECT_V4")[0]
    ctf = repository.rank("heap exploitation")[0]
    return repository, (wfp, ctf)


def test_parse_reference_support_is_strict_and_fail_closed():
    assert parse_reference_support('{"supported":true}') is True
    assert parse_reference_support('{"supported":false}') is False
    assert parse_reference_support('{"supported":true,"extra":1}') is None
    assert parse_reference_support('{"supported":true,"supported":false}') is None
    assert parse_reference_support("true") is None


def test_parse_reference_selection_is_strict_and_allows_abstention():
    aliases = ("A", "B")
    assert parse_reference_selection('{"winner":"A"}', aliases) == (True, "A")
    assert parse_reference_selection('{"winner":null}', aliases) == (True, None)
    assert parse_reference_selection('{"winner":"Z"}', aliases) == (False, None)
    assert parse_reference_selection('{"winner":"A","extra":1}', aliases) == (
        False,
        None,
    )
    assert parse_reference_selection('{"winner":"A","winner":"B"}', aliases) == (
        False,
        None,
    )


def test_select_best_compares_candidates_in_one_bounded_job(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner(fixed_text='{"winner":"A"}')
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    result = asyncio.run(
        judge.select_best(
            parent_request_id="parent-listwise",
            turn_id="turn-listwise",
            question="Which reference best answers the WFP question?",
            candidates=candidates,
            telemetry_correlation_id="trace-listwise",
        )
    )

    assert len(runner.jobs) == len(runner.prompts) == 1
    prompt_payload = json.loads(runner.prompts[0].split("<user>", 1)[1])
    assert {item["source"] for item in prompt_payload["candidates"]} == {
        "wfp.md",
        "ctf.md",
    }
    winner_source = next(
        item["source"] for item in prompt_payload["candidates"] if item["id"] == "A"
    )
    winner_candidate = next(
        candidate
        for candidate in candidates
        if candidate.relative_path == winner_source
    )
    assert result.selection_method == "comparative_listwise"
    assert result.presented_candidate_count == 2
    assert result.executed_count == 1
    assert result.valid_count == 2
    assert result.eligible_count == 1
    assert result.selected_candidate_id == winner_candidate.candidate_id
    assert sum(decision.eligible for decision in result.decisions) == 1
    schema = json.loads(runner.sampling_params["json_schema"])
    assert schema["properties"]["winner"]["enum"] == [None, "A", "B"]
    assert '"score"' not in runner.prompts[0]


def test_select_best_presents_eight_candidates(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    documents = [
        repository.upsert(f"reference-{index}.md", f"reference content {index}")
        for index in range(8)
    ]
    candidates = tuple(repository.rank(document.content)[0] for document in documents)
    runner = FakeRunner(fixed_text='{"winner":"H"}')
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    result = asyncio.run(
        judge.select_best(
            parent_request_id="parent-eight",
            turn_id="turn-eight",
            question="Which reference is relevant?",
            candidates=candidates,
            telemetry_correlation_id="trace-eight",
        )
    )

    assert result.presented_candidate_count == 8
    assert result.valid_count == 8
    assert result.eligible_count == 1
    assert result.selected_candidate_id == candidates[7].candidate_id
    assert json.loads(runner.sampling_params["json_schema"])["properties"]["winner"][
        "enum"
    ] == [None, "A", "B", "C", "D", "E", "F", "G", "H"]


def test_select_best_can_reject_every_candidate(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    judge = ReferenceJudge(
        FakeRunner(fixed_text='{"winner":null}'),
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.select_best(
            parent_request_id="parent-none",
            turn_id="turn-none",
            question="Unrelated clinical dosing question",
            candidates=candidates,
            telemetry_correlation_id="trace-none",
        )
    )

    assert result.valid_count == 2
    assert result.eligible_count == 0
    assert result.selected_candidate_id is None
    assert all(
        decision.status is EligibilityStatus.INELIGIBLE for decision in result.decisions
    )


def test_select_best_invalid_winner_fails_closed(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    judge = ReferenceJudge(
        FakeRunner(fixed_text='{"winner":"Z"}'),
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.select_best(
            parent_request_id="parent-invalid",
            turn_id="turn-invalid",
            question="WFP question",
            candidates=candidates,
            telemetry_correlation_id="trace-invalid",
        )
    )

    assert result.valid_count == result.eligible_count == 0
    assert all(
        decision.status is EligibilityStatus.INVALID for decision in result.decisions
    )


def test_select_best_cache_is_candidate_order_independent(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner(fixed_text='{"winner":"A"}')
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    async def select(parent_request_id, items):
        return await judge.select_best(
            parent_request_id=parent_request_id,
            turn_id=f"{parent_request_id}:turn",
            question="WFP question",
            candidates=items,
            telemetry_correlation_id=f"{parent_request_id}:trace",
        )

    first = asyncio.run(select("first", candidates))
    repeated = asyncio.run(select("second", tuple(reversed(candidates))))

    assert first.selected_candidate_id == repeated.selected_candidate_id
    assert first.executed_count == 1
    assert repeated.executed_count == 0
    assert repeated.cache_hit_count == 1
    assert all(
        decision.judge_method == "sglang_constrained_listwise_cache"
        for decision in repeated.decisions
    )
    assert runner.calls == 1


def test_judge_batches_candidates_with_shared_prefix(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-1",
            turn_id="turn-1",
            question="How should WFP AppID filtering be configured?",
            candidates=candidates,
            telemetry_correlation_id="trace-1",
        )
    )

    assert [decision.status for decision in result.decisions] == [
        EligibilityStatus.ELIGIBLE,
        EligibilityStatus.INELIGIBLE,
    ]
    assert result.valid_count == 2
    assert result.eligible_count == 1
    assert len({job.shared_prefix_key for job in runner.jobs}) == 1
    assert len(os.path.commonprefix(runner.prompts)) > 100
    assert runner.sampling_params["json_schema"]


def test_judge_uses_frozen_candidate_content_after_repository_update(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert(
        "wfp.md", "Use FWPM_LAYER_ALE_AUTH_CONNECT_V4 with an AppID condition."
    )
    candidate = repository.rank("FWPM_LAYER_ALE_AUTH_CONNECT_V4")[0]
    repository.upsert("wfp.md", "replacement content that was never proposed")
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-1",
            turn_id="turn-1",
            question="How should WFP filtering be configured?",
            candidates=(candidate,),
            telemetry_correlation_id="trace-1",
        )
    )

    assert result.decisions[0].status is EligibilityStatus.ELIGIBLE
    assert "FWPM_LAYER_ALE_AUTH_CONNECT_V4" in runner.prompts[0]
    assert "replacement content" not in runner.prompts[0]


def test_judge_truncates_long_question_and_still_reviews(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
        max_question_tokens=64,
    )
    question = (
        "head-marker "
        + " ".join(f"word-{index}" for index in range(3000))
        + " tail-marker"
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-long",
            turn_id="turn-long",
            question=question,
            candidates=(candidates[0],),
            telemetry_correlation_id="trace-long",
        )
    )

    assert result.decisions[0].status is EligibilityStatus.ELIGIBLE
    assert result.valid_count == result.eligible_count == 1
    assert result.executed_count == 1
    assert result.question_truncated is True
    assert result.question_original_tokens == 3002
    assert result.question_review_tokens <= 64
    assert len(runner.prompts) == 1
    assert "head-marker" in runner.prompts[0]
    assert "tail-marker" in runner.prompts[0]
    assert "middle question tokens omitted" in runner.prompts[0]
    assert result.decisions[0].question_digest == stable_digest(question)


def test_select_best_truncates_long_question_and_still_reviews(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner(fixed_text='{"winner":"A"}')
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
        max_question_tokens=64,
    )
    question = (
        "original-task "
        + " ".join(f"trajectory-{index}" for index in range(3000))
        + " latest-evidence"
    )

    result = asyncio.run(
        judge.select_best(
            parent_request_id="parent-long-listwise",
            turn_id="turn-long-listwise",
            question=question,
            candidates=candidates,
            telemetry_correlation_id="trace-long-listwise",
        )
    )

    prompt_payload = json.loads(runner.prompts[0].split("<user>", 1)[1])
    review_question = prompt_payload["question"]
    assert result.selection_method == "comparative_listwise"
    assert result.valid_count == 2
    assert result.eligible_count == 1
    assert result.executed_count == 1
    assert result.presented_candidate_count == 2
    assert result.question_truncated is True
    assert result.question_original_tokens == 3002
    assert result.question_review_tokens <= 64
    assert "original-task" in review_question
    assert "latest-evidence" in review_question
    assert "middle question tokens omitted" in review_question
    assert all(
        decision.question_digest == stable_digest(question)
        for decision in result.decisions
    )


def test_malformed_output_marks_every_candidate_invalid(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    judge = ReferenceJudge(
        FakeRunner(malformed=True),
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-1",
            turn_id="turn-1",
            question="question",
            candidates=candidates,
            telemetry_correlation_id="trace-1",
        )
    )

    assert all(
        decision.status is EligibilityStatus.INVALID for decision in result.decisions
    )
    assert result.valid_count == 0
    assert result.eligible_count == 0


def test_valid_json_without_normal_stop_is_invalid(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    judge = ReferenceJudge(
        FakeRunner(finish_type="length"),
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-truncated",
            turn_id="turn-truncated",
            question="WFP question",
            candidates=(candidates[0],),
            telemetry_correlation_id="trace-truncated",
        )
    )

    assert result.decisions[0].status is EligibilityStatus.INVALID
    assert result.eligible_count == 0


def test_candidate_permutation_does_not_change_decisions(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    judge = ReferenceJudge(
        FakeRunner(), repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    async def evaluate(items):
        return await judge.judge(
            parent_request_id="parent-1",
            turn_id="turn-1",
            question="WFP question",
            candidates=items,
            telemetry_correlation_id="trace-1",
        )

    forward = asyncio.run(evaluate(candidates))
    reverse = asyncio.run(evaluate(tuple(reversed(candidates))))
    forward_by_candidate = {
        decision.candidate_id: decision.status for decision in forward.decisions
    }
    reverse_by_candidate = {
        decision.candidate_id: decision.status for decision in reverse.decisions
    }

    assert forward_by_candidate == reverse_by_candidate


def test_judge_identifies_policy_lane_and_uses_operational_applicability(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    policy_candidate = replace(candidates[0], lane="policydata")
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-policy",
            turn_id="turn-policy",
            question="Implement and verify the requested repository change",
            candidates=(policy_candidate,),
            telemetry_correlation_id="trace-policy",
        )
    )

    assert result.decisions[0].status is EligibilityStatus.ELIGIBLE
    assert '"lane":"policydata"' in runner.prompts[0]
    assert "policy need not contain the task's answer" in runner.prompts[0]


def test_judge_reuses_only_exact_valid_semantic_decisions(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    async def evaluate(parent_request_id, question):
        return await judge.judge(
            parent_request_id=parent_request_id,
            turn_id=f"turn-{parent_request_id}",
            question=question,
            candidates=candidates,
            telemetry_correlation_id=f"trace-{parent_request_id}",
        )

    first = asyncio.run(evaluate("first", "WFP question"))
    repeated = asyncio.run(evaluate("second", "WFP question"))
    changed = asyncio.run(evaluate("third", "Different WFP question"))

    assert first.cache_hit_count == 0
    assert first.executed_count == 2
    assert repeated.cache_hit_count == 2
    assert repeated.executed_count == 0
    assert runner.calls == 2
    assert all(
        decision.parent_request_id == "second" for decision in repeated.decisions
    )
    assert all(
        decision.judge_method == "sglang_constrained_binary_cache"
        for decision in repeated.decisions
    )
    assert changed.cache_hit_count == 0
    assert changed.executed_count == 2


def test_judge_cache_uses_semantic_reference_not_request_candidate_id(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner, repository, FakeTokenizer(), model_fingerprint="model-fingerprint"
    )
    request_specific = replace(candidates[0], candidate_id="request-specific-id")

    async def evaluate(candidate, parent):
        return await judge.judge(
            parent_request_id=parent,
            turn_id=f"turn-{parent}",
            question="WFP question",
            candidates=(candidate,),
            telemetry_correlation_id=f"trace-{parent}",
        )

    async def run_twice():
        first = await evaluate(candidates[0], "first")
        repeated = await evaluate(request_specific, "second")
        return first, repeated

    first, repeated = asyncio.run(run_twice())

    assert first.executed_count == 1
    assert repeated.cache_hit_count == 1
    assert repeated.decisions[0].candidate_id == "request-specific-id"
    assert runner.calls == 1


def test_judge_never_caches_invalid_results(tmp_path):
    repository, candidates = repository_with_candidates(tmp_path)
    runner = FakeRunner(malformed=True)
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    async def evaluate(parent_request_id):
        return await judge.judge(
            parent_request_id=parent_request_id,
            turn_id=f"turn-{parent_request_id}",
            question="WFP question",
            candidates=candidates,
            telemetry_correlation_id=f"trace-{parent_request_id}",
        )

    first = asyncio.run(evaluate("first-invalid"))
    repeated = asyncio.run(evaluate("second-invalid"))

    assert first.cache_hit_count == repeated.cache_hit_count == 0
    assert first.executed_count == repeated.executed_count == 2
    assert runner.calls == 2


def test_judge_bounds_oversize_reference_and_keeps_batch_alive(tmp_path):
    repository = KnowledgeRepository(tmp_path)
    repository.upsert(
        "long.md", "head FWPM_LAYER_ALE_AUTH_CONNECT_V4 " + "filler " * 8000 + "tail"
    )
    repository.upsert("short.md", "unrelated gardening notes")
    long_candidate = repository.rank("FWPM_LAYER_ALE_AUTH_CONNECT_V4")[0]
    short_candidate = repository.rank("gardening")[0]
    runner = FakeRunner()
    judge = ReferenceJudge(
        runner,
        repository,
        FakeTokenizer(),
        model_fingerprint="model-fingerprint",
    )

    result = asyncio.run(
        judge.judge(
            parent_request_id="parent-long",
            turn_id="turn-1",
            question="How should WFP AppID filtering be configured?",
            candidates=(long_candidate, short_candidate),
            telemetry_correlation_id="trace-long",
        )
    )

    assert result.valid_count == 2
    assert result.decisions[0].status is EligibilityStatus.ELIGIBLE
    assert "filler" * 4000 not in runner.prompts[0]
    assert "中间省略" in runner.prompts[0]
