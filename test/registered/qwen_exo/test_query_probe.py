import asyncio
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags
from qwen_exo_booster.contracts import (
    CancellationToken,
    EligibilityDecision,
    EligibilityStatus,
    InternalJob,
    InternalJobType,
    stable_digest,
)
from qwen_exo_booster.internal_jobs import InternalJobResult, InternalJobRunner
from qwen_exo_booster.knowledge import KnowledgeRepository, NativePrefixSelection
from qwen_exo_booster.pipeline import MemoryPipeline
from qwen_exo_booster.query_probe import (
    QueryProbePlan,
    QueryProbeService,
    QueryRoleText,
    QueryStateSpan,
)


class FakeTelemetry:
    def __init__(self):
        self.events = []

    def emit(self, request_id, event_type, payload):
        self.events.append((request_id, event_type, payload))


class FakeProbeRunner:
    max_fanout = 8

    def __init__(self):
        self.calls = []

    async def run_batch(
        self,
        jobs,
        prompts,
        sampling_params,
        *,
        custom_params_per_job=None,
        extra_keys=None,
    ):
        del extra_keys
        self.calls.append((jobs, prompts, sampling_params, custom_params_per_job))
        spans = custom_params_per_job[0]["qwen_exo_query_spans"]
        flattened = []
        for index, _span in enumerate(spans):
            flattened.extend(
                [1.0, 0.0, 0.0, 1.0] if index % 2 == 0 else [0.0, 1.0, 1.0, 0.0]
            )
        return (
            InternalJobResult(
                job=jobs[0],
                text="",
                prompt_tokens=len(prompts[0]),
                completion_tokens=0,
                finish_reason={"type": "length"},
                latency_seconds=0.01,
                metadata={"qwen_exo_user_query_full_heads": [flattened]},
            ),
        )


class FakeTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))


@dataclass(frozen=True)
class FakeRequest:
    request_id: str
    input: object
    instructions: str | None = None
    previous_response_id: str | None = None
    extra_key: str | None = None

    def model_copy(self, update):
        return replace(self, **update)


def _states(count, role="current_user"):
    return tuple(
        QueryStateSpan(role, index, index + 1, index, index + 1)
        for index in range(count)
    )


class FakeQKBank:
    def __init__(self, candidate):
        self.candidate = candidate
        self.rank_calls = []
        self.snapshot = SimpleNamespace(source_digest="bank-a")

    async def ensure_ready(self):
        return SimpleNamespace(ready=True)

    def rank(
        self,
        query_heads,
        *,
        query_states,
        query_identity,
        limit,
        min_tensor_score=0.0,
        min_document_margin=0.005,
        audit=None,
    ):
        self.rank_calls.append(
            (query_heads, query_states, query_identity, limit, min_document_margin)
        )
        if audit is not None:
            audit.update(
                status="ready",
                reason="candidates_ready",
                min_tensor_score=min_tensor_score,
                min_document_margin=min_document_margin,
                candidate_count=1,
            )
        return (self.candidate,)

    @staticmethod
    def bind_native_prefix(candidate, *, query, preferred_page_ids=()):
        del query, preferred_page_ids
        return candidate


class FakeRequestJudge:
    async def judge(
        self,
        *,
        parent_request_id,
        turn_id,
        question,
        candidates,
        telemetry_correlation_id,
    ):
        del turn_id, telemetry_correlation_id
        decisions = tuple(
            EligibilityDecision.create(
                candidate_id=candidate.candidate_id,
                parent_request_id=parent_request_id,
                question=question,
                reference=candidate.reference_content,
                status=EligibilityStatus.ELIGIBLE,
                judge_method="fake_batch_judge",
                judge_model_fingerprint="fake-model",
                decision_margin=0.0,
            )
            for candidate in candidates
        )
        return SimpleNamespace(
            decisions=decisions,
            valid_count=len(decisions),
            eligible_count=len(decisions),
            latency_seconds=0.001,
            cache_hit_count=0,
            executed_count=len(decisions),
            selection_method="independent_binary",
            selected_candidate_id=(decisions[0].candidate_id if decisions else None),
            presented_candidate_count=len(decisions),
        )


def _job(job_type):
    return InternalJob(
        parent_request_id="parent",
        turn_id="turn",
        job_id="job",
        job_type=job_type,
        priority=-1,
        shared_prefix_key="qwen-exo:v1:test:query-probe",
        token_budget=1,
        state_budget_bytes=0,
        deadline_monotonic=time.monotonic() + 5,
        cancellation_token=CancellationToken("cancel"),
        telemetry_correlation_id="trace",
        max_fanout=1,
    )


def test_query_probe_returns_finite_raw_q_heads_from_hidden_job():
    runner = FakeProbeRunner()
    telemetry = FakeTelemetry()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        telemetry,
        max_prompt_tokens=4,
        query_head_count=2,
        head_dim=2,
    )

    result = asyncio.run(
        probe.probe(
            "resp-1", QueryProbePlan.current_user("one two three four five six")
        )
    )

    assert result.status == "ready"
    assert len(result.query_heads) == 4
    assert result.query_heads[0][0] == (1.0, 0.0)
    assert result.query_heads[1][0] == (0.0, 1.0)
    jobs, prompts, _sampling, custom = runner.calls[0]
    assert jobs[0].job_type is InternalJobType.QUERY_PROBE
    assert prompts == ((3, 4, 5, 6),)
    assert custom[0]["qwen_exo_query_spans"] == [
        {"start": 0, "end": 1, "role": "current_user"},
        {"start": 1, "end": 2, "role": "current_user"},
        {"start": 2, "end": 3, "role": "current_user"},
        {"start": 3, "end": 4, "role": "current_user"},
    ]
    assert {state.role for state in result.query_states} == {"current_user"}
    assert [event[1] for event in telemetry.events] == [
        "query_probe.started",
        "query_probe.completed",
    ]


def test_query_probe_reuses_exact_raw_q_heads_without_disabling_probe():
    runner = FakeProbeRunner()
    telemetry = FakeTelemetry()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        telemetry,
        max_prompt_tokens=4,
        query_head_count=2,
        head_dim=2,
    )

    async def run_twice():
        return (
            await probe.probe(
                "resp-first", QueryProbePlan.current_user("one two three four")
            ),
            await probe.probe(
                "resp-second", QueryProbePlan.current_user("one two three four")
            ),
        )

    first, repeated = asyncio.run(run_twice())

    assert first.cache_hit is False
    assert repeated.cache_hit is True
    assert repeated.query_heads == first.query_heads
    assert len(runner.calls) == 1
    completed = [
        event for event in telemetry.events if event[1] == "query_probe.completed"
    ]
    assert completed[-1][2]["cache_hit"] is True


def test_query_probe_conditions_q_spans_on_cognition_prefix():
    runner = FakeProbeRunner()
    telemetry = FakeTelemetry()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        telemetry,
        max_prompt_tokens=6,
        cognition_token_ids=(90, 91),
        query_head_count=2,
        head_dim=2,
    )

    result = asyncio.run(
        probe.probe(
            "resp-cognition", QueryProbePlan.current_user("one two three four five six")
        )
    )

    _jobs, prompts, _sampling, custom = runner.calls[0]
    assert result.prompt_tokens == 6
    assert prompts == ((90, 91, 3, 4, 5, 6),)
    assert custom[0]["qwen_exo_query_spans"] == [
        {"start": 2, "end": 3, "role": "current_user"},
        {"start": 3, "end": 4, "role": "current_user"},
        {"start": 4, "end": 5, "role": "current_user"},
        {"start": 5, "end": 6, "role": "current_user"},
    ]
    assert telemetry.events[0][2]["cognition_tokens"] == 2


def test_query_probe_preserves_anchor_roles_and_bounds_trajectory():
    runner = FakeProbeRunner()
    telemetry = FakeTelemetry()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        telemetry,
        max_prompt_tokens=8,
        query_head_count=2,
        head_dim=2,
    )
    plan = QueryProbePlan(
        (
            QueryRoleText("original_task", "original anchor words"),
            QueryRoleText("current_user", "current request words"),
            QueryRoleText("trajectory_compaction", " ".join(["history"] * 100)),
        )
    )

    result = asyncio.run(probe.probe("resp-roles", plan))

    roles = [state.role for state in result.query_states]
    assert "original_task" in roles
    assert "current_user" in roles
    assert roles.count("trajectory_compaction") <= 2
    assert result.prompt_tokens == 8
    assert all(
        left.prompt_end <= right.prompt_start
        for left, right in zip(result.query_states, result.query_states[1:])
    )
    assert telemetry.events[-1][2]["role_plan_digest"] == plan.identity


def test_query_probe_cache_identity_includes_role_plan():
    runner = FakeProbeRunner()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        FakeTelemetry(),
        max_prompt_tokens=4,
        query_head_count=2,
        head_dim=2,
    )

    async def probe_roles():
        return (
            await probe.probe("anchor", QueryProbePlan.current_user("same words")),
            await probe.probe(
                "trajectory",
                QueryProbePlan((QueryRoleText("trajectory_compaction", "same words"),)),
            ),
        )

    anchor, trajectory = asyncio.run(probe_roles())
    assert anchor.cache_hit is False
    assert trajectory.cache_hit is False
    assert anchor.role_plan_digest != trajectory.role_plan_digest
    assert len(runner.calls) == 2


def test_request_query_plan_separates_task_current_and_history():
    plan = MemoryPipeline._request_query_plan(
        [
            {"role": "user", "content": "root task"},
            {"role": "assistant", "content": "working"},
            {"type": "function_call_output", "output": "tool observation"},
            {"role": "user", "content": "current correction"},
        ],
        original_task="root task",
        compaction_context="compacted history",
    )

    assert [segment.role for segment in plan.segments] == [
        "original_task",
        "current_user",
        "trajectory_compaction",
    ]
    assert plan.segments[0].text == "root task"
    assert plan.segments[1].text == "current correction"
    assert "compacted history" in plan.segments[2].text
    assert "tool observation" in plan.segments[2].text


def test_request_query_plan_deduplicates_equal_original_and_current():
    plan = MemoryPipeline._request_query_plan(
        "same first request", original_task="same first request"
    )

    assert [segment.role for segment in plan.segments] == ["original_task"]
    assert plan.segments[0].text == "same first request"


def test_latest_user_ignores_roleless_reasoning_after_explicit_user():
    value = [
        {"role": "user", "content": "explicit user task"},
        {"type": "reasoning", "content": "roleless hidden reasoning"},
    ]

    assert MemoryPipeline._latest_user_text(value) == "explicit user task"
    plan = MemoryPipeline._request_query_plan(value)
    assert [segment.role for segment in plan.segments] == [
        "original_task",
        "trajectory_compaction",
    ]
    assert plan.segments[0].text == "explicit user task"
    assert all(segment.role != "current_user" for segment in plan.segments)
    assert "roleless hidden reasoning" in plan.segments[1].text


def test_roleless_reasoning_only_plan_has_no_current_user_anchor():
    value = [
        {"type": "reasoning", "content": "roleless hidden reasoning"},
        {"type": "function_call_output", "output": "tool-only trajectory"},
    ]

    assert MemoryPipeline._latest_user_text(value) == ""
    plan = MemoryPipeline._request_query_plan(value)
    assert [segment.role for segment in plan.segments] == ["trajectory_compaction"]
    assert all(segment.role != "current_user" for segment in plan.segments)


def test_query_probe_fails_closed_on_interior_nonfinite_state():
    class NonfiniteRunner(FakeProbeRunner):
        async def run_batch(
            self,
            jobs,
            prompts,
            sampling_params,
            *,
            custom_params_per_job=None,
            extra_keys=None,
        ):
            del extra_keys
            self.calls.append((jobs, prompts, sampling_params, custom_params_per_job))
            return (
                InternalJobResult(
                    job=jobs[0],
                    text="",
                    prompt_tokens=len(prompts[0]),
                    completion_tokens=0,
                    finish_reason={"type": "length"},
                    latency_seconds=0.01,
                    metadata={
                        "qwen_exo_user_query_full_heads": [
                            [
                                1.0,
                                0.0,
                                0.0,
                                1.0,
                                float("nan"),
                                0.0,
                                0.0,
                                1.0,
                                0.0,
                                1.0,
                                1.0,
                                0.0,
                            ]
                        ]
                    },
                ),
            )

    runner = NonfiniteRunner()
    telemetry = FakeTelemetry()
    probe = QueryProbeService(
        runner,
        FakeTokenizer(),
        telemetry,
        max_prompt_tokens=4,
        query_head_count=2,
        head_dim=2,
    )
    plan = QueryProbePlan(
        (
            QueryRoleText("original_task", "original"),
            QueryRoleText("current_user", "current"),
            QueryRoleText("trajectory_compaction", "trajectory"),
        )
    )

    async def run_twice():
        return await probe.probe("first", plan), await probe.probe("second", plan)

    first, second = asyncio.run(run_twice())

    assert first.status == second.status == "no_q_signal"
    assert first.query_heads == second.query_heads == ()
    assert first.query_states == second.query_states == ()
    assert first.cache_hit is second.cache_hit is False
    assert len(runner.calls) == 2
    assert telemetry.events[-1][2]["role_counts"] == {}


def test_internal_query_probe_uses_one_hidden_token_for_prefill_metadata():
    requests = []

    class Manager:
        async def generate_request(self, request, raw_request):
            del raw_request
            requests.append(request)
            yield [{"text": "", "meta_info": {}}]

        def abort_request(self, _rid):
            return None

    runner = InternalJobRunner(
        Manager(),
        max_fanout=1,
        max_tokens_per_parent=8,
        request_factory=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    asyncio.run(runner.run_batch((_job(InternalJobType.QUERY_PROBE),), ((1, 2),), {}))

    assert requests[0].sampling_params[0]["max_new_tokens"] == 1
    assert requests[0].sampling_params[0]["custom_params"]["qwen_exo_job_type"] == (
        "query_probe"
    )


def test_request_memory_uses_only_query_q_and_skips_text_rankers(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    repository = KnowledgeRepository(knowledge_dir)
    document = repository.upsert("wfp.md", "WFP native query key material")
    repository.rank = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("text ranker must not run")
    )
    native = NativePrefixSelection(
        source_digest="a" * 64,
        page_id=3,
        document_id=document.document_id,
        local_positions=tuple(range(64)),
        source_positions=tuple(range(4)),
        token_ids=tuple(range(100, 164)),
        prefix_identity="query-qk-native",
        radix_namespace="qwen-exo:v1:tensor-bank-native:query-qk",
    )
    candidate = replace(
        repository.candidate_for_document(document.document_id, "query-probe"),
        score=0.91,
        lexical_score=0.0,
        tensor_score=0.91,
        page_ids=(3,),
        source_positions=native.source_positions,
        virtual_positions=tuple(range(4)),
        token_attributions=((0, 3, 0.91),),
        native_prefix=native,
        candidate_origin="attention_q_native_tensor_bank",
    )
    bank = FakeQKBank(candidate)
    config = QwenExoConfig(
        state_directory=tmp_path / "state",
        knowledge_directory=knowledge_dir,
        policy_data_directory=tmp_path / "policydata",
        max_internal_fanout=8,
        max_internal_tokens=1024,
        max_candidates=8,
        max_memory_tokens=256,
        observer_mode="shadow",
        feature_flags=QwenExoFeatureFlags(
            hybrid_prefix=True,
            external_memory=True,
            reference_judge=True,
            policy_data=False,
            capsule=False,
            observer=True,
            adaptive_refresh=False,
        ),
        model_path="model",
        tp_size=2,
    )
    pipeline = MemoryPipeline(
        config,
        repository,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=FakeRequestJudge(),
    )
    request = FakeRequest(
        request_id="resp-qk",
        input="WFP question",
        instructions="Keep this unchanged.",
    )

    prepared, state = asyncio.run(
        pipeline.prepare_responses_request(
            request,
            query_heads=(((1.0, 0.0),),),
            query_states=_states(1),
            query_role_plan_digest="role-plan-current",
            query_probe_status="ready",
            query_probe_prompt_tokens=2,
        )
    )
    repeated_request = replace(request, request_id="resp-qk-repeated")
    _repeated_prepared, repeated_state = asyncio.run(
        pipeline.prepare_responses_request(
            repeated_request,
            query_heads=(((1.0, 0.0),),),
            query_states=_states(1),
            query_role_plan_digest="role-plan-current",
            query_probe_status="ready",
            query_probe_prompt_tokens=2,
        )
    )

    assert prepared.instructions == request.instructions
    assert state.knowledge_admission_mode == "semantic_eligibility"
    assert len(state.decisions) == 1
    assert state.radix_prefix_page_id == 3
    assert state.radix_prefix_selection_reason == "query_qk"
    assert state.selected_document_ids == (document.document_id,)
    assert state.candidates[0].candidate_origin == "attention_q_native_tensor_bank"
    assert state.public_dict()["query_probe"] == {
        "status": "ready",
        "prompt_tokens": 2,
        "query_count": 1,
        "query_head_count": 1,
        "head_dim": 2,
        "role_plan_digest": "role-plan-current",
        "role_counts": {"current_user": 1},
        "query_states": [_states(1)[0].public_dict()],
    }
    assert state.qk_rank_cache_hit is False
    assert repeated_state.qk_rank_cache_hit is True
    assert repeated_state.public_dict()["qk_retrieval"]["cache_hit"] is True
    stable_identity = f"request-query-probe:{stable_digest('WFP question')}"
    assert bank.rank_calls == [((((1.0, 0.0),),), _states(1), stable_identity, 8, 0.02)]
    bank.snapshot = SimpleNamespace(source_digest="bank-b")
    changed_request = replace(request, request_id="resp-qk-changed-bank")
    _changed_prepared, changed_state = asyncio.run(
        pipeline.prepare_responses_request(
            changed_request,
            query_heads=(((1.0, 0.0),),),
            query_states=_states(1),
            query_role_plan_digest="role-plan-current",
            query_probe_status="ready",
            query_probe_prompt_tokens=2,
        )
    )
    assert changed_state.qk_rank_cache_hit is False
    assert len(bank.rank_calls) == 2
