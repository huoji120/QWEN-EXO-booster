import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.causal_replay import CausalReplayService
from qwen_exo_booster.contracts import EligibilityDecision, EligibilityStatus
from qwen_exo_booster.internal_jobs import (
    InternalJobRunner,
    InternalScoreResult,
)
from qwen_exo_booster.judge import ReferenceJudge
from qwen_exo_booster.knowledge import KnowledgeRepository, NativePrefixSelection
from qwen_exo_booster.observer import MidThinkEvent
from qwen_exo_booster.pipeline import MemoryPipeline
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.refresh import SelfAskRefreshService
from qwen_exo_booster.telemetry import TelemetryStore
from qwen_exo_booster.query_probe import QueryStateSpan


class FakeManager:
    def __init__(
        self,
        invalid_answer=False,
        *,
        all_supported=False,
        answer=None,
        question=None,
        question_kind="factual",
        question_payload=None,
        integrity_payload=None,
    ):
        self.invalid_answer = invalid_answer
        self.all_supported = all_supported
        self.answer = answer or "Use WFP_LAYER_ALE_AUTH_CONNECT_V4."
        self.question = question or "What is the required WFP AppID?"
        self.question_kind = question_kind
        self.question_payload = question_payload
        self.integrity_payload = integrity_payload
        self.requests = []

    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, token_ids, **_kwargs):
        return " ".join(token_ids)

    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(message["content"] for message in messages)

    async def generate_request(self, request, _raw_request):
        self.requests.append(request)
        outputs = []
        for prompt in request.text:
            if "internal document-retrieval question classifier" in prompt:
                text = self.question_payload or json.dumps(
                    {
                        "tool": "submit_self_question",
                        "kind": self.question_kind,
                        "arguments": {
                            "question": self.question,
                        },
                    },
                    ensure_ascii=False,
                )
            elif "Judge whether the supplied candidate" in prompt:
                text = (
                    '{"supported":true}'
                    if self.all_supported or "WFP_LAYER" in prompt
                    else '{"supported":false}'
                )
            elif "Answer the classified self-question" in prompt:
                text = (
                    "<think>truncated answer</think>"
                    if self.invalid_answer
                    else self.answer
                )
            elif "dedicated Context Integrity Check reviewer" in prompt:
                text = self.integrity_payload or json.dumps(
                    {
                        "status": "consistent",
                        "confirmed_facts": [
                            "No material contradiction was established."
                        ],
                        "invalid_claims": [],
                        "contradictions": [],
                        "stale_assumptions": [],
                        "correction": None,
                        "evidence_needed": [],
                        "confidence": 1.0,
                    }
                )
            else:
                raise AssertionError(f"Unexpected prompt: {prompt}")
            outputs.append(
                {
                    "text": text,
                    "meta_info": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "finish_reason": {"type": "stop"},
                    },
                }
            )
        yield outputs

    def abort_request(self, _request_id):
        pass


class BlockingManager(FakeManager):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.aborted = []
        self._release = asyncio.Event()

    async def generate_request(self, request, _raw_request):
        self.requests.append(request)
        self.started.set()
        try:
            await self._release.wait()
        finally:
            self.stopped.set()
        yield []

    def abort_request(self, request_id):
        self.aborted.append(request_id)


class BlockingTensorBank:
    def __init__(self, child_tasks, started, stopped):
        self.child_tasks = child_tasks
        self.started = started
        self.stopped = stopped
        self._release = asyncio.Event()

    async def ensure_ready(self):
        task = asyncio.current_task()
        assert task is not None
        self.child_tasks.append(task)
        self.started.set()
        try:
            await self._release.wait()
        finally:
            self.stopped.set()

    def rank(self, *_args, **_kwargs):
        raise AssertionError("Cancelled Tensor Bank task continued into rank")

    async def ensure_resident(self, *_args, **_kwargs):
        raise AssertionError("Cancelled Tensor Bank task continued into residency")


class RecordingTensorBank:
    def __init__(self):
        self.rank_kwargs = None
        self.resident_page_ids = None

    async def ensure_ready(self):
        return SimpleNamespace(ready=True)

    def rank(self, *_args, **kwargs):
        self.rank_kwargs = kwargs
        return ()

    async def ensure_resident(self, page_ids):
        self.resident_page_ids = tuple(page_ids)


class JudgeGatedBindingTensorBank:
    def __init__(self, manager):
        self.manager = manager
        self.bind_calls = []
        self.resident_page_ids = None

    def bind_native_prefix(self, candidate, *, query, preferred_page_ids=()):
        assert any(
            "Judge whether the supplied candidate" in prompt
            for request in self.manager.requests
            for prompt in request.text
        )
        page_id = preferred_page_ids[0] if preferred_page_ids else candidate.page_ids[0]
        native = NativePrefixSelection(
            source_digest="c" * 64,
            page_id=page_id,
            document_id=candidate.document_id,
            local_positions=tuple(range(64)),
            source_positions=(1, 2, 3, 4),
            token_ids=tuple(range(64)),
            prefix_identity=f"judge-gated-{page_id}",
            radix_namespace=f"qwen-exo:v1:tensor-bank-native:{page_id}",
        )
        self.bind_calls.append((candidate.candidate_id, query))
        return replace(
            candidate,
            source_positions=native.source_positions,
            virtual_positions=tuple(range(len(native.source_positions))),
            native_prefix=native,
            candidate_origin="admitted_native_tensor_bank",
        )

    async def ensure_resident(self, page_ids):
        self.resident_page_ids = tuple(page_ids)


def request_factory(**kwargs):
    return SimpleNamespace(**kwargs)


def repository(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "wfp.md").write_text(
        "# WFP\nUse WFP_LAYER_ALE_AUTH_CONNECT_V4 for outbound connect authorization.",
        encoding="utf-8",
    )
    (root / "ctf.md").write_text(
        "# CTF\nROP chains often use stack pivots and leaked libc addresses.",
        encoding="utf-8",
    )
    repo = KnowledgeRepository(root)
    repo.refresh()
    return repo


def build_service(tmp_path, manager, **kwargs):
    repo = repository(tmp_path)
    runner = InternalJobRunner(
        manager,
        max_fanout=8,
        max_tokens_per_parent=1024,
        request_factory=request_factory,
    )
    judge = ReferenceJudge(
        runner,
        repo,
        manager,
        model_fingerprint="model-fingerprint",
    )
    return (
        SelfAskRefreshService(
            runner,
            repo,
            judge,
            TelemetryStore(tmp_path / "trace.jsonl"),
            max_candidates=8,
            **kwargs,
        ),
        repo,
    )


@pytest.mark.asyncio
async def test_tensor_refresh_forwards_resolved_admission_margin(tmp_path):
    bank = RecordingTensorBank()
    service, _repo = build_service(
        tmp_path,
        FakeManager(),
        tensor_bank=bank,
        query_probe=SimpleNamespace(
            probe=lambda *_args, **_kwargs: asyncio.sleep(
                0,
                result=SimpleNamespace(
                    status="ready",
                    query_heads=(((1.0, 0.0),),),
                    query_states=(QueryStateSpan("current_user", 0, 1, 0, 1),),
                ),
            )
        ),
        qk_admission_margin=0.02,
        qk_min_tensor_score=8.0,
    )

    candidates = await service._tensor_candidates(
        SimpleNamespace(event_id="event-margin", request_id="request-margin"),
        "Which exact constant is needed?",
    )

    assert candidates == ()
    assert bank.rank_kwargs["min_document_margin"] == 0.02
    assert bank.rank_kwargs["min_tensor_score"] == 8.0
    assert bank.resident_page_ids is None


@pytest.mark.parametrize("candidate_source", ("fixed", "tensor"))
@pytest.mark.asyncio
async def test_refresh_cancellation_awaits_all_child_tasks(tmp_path, candidate_source):
    manager = BlockingManager()
    service, repo = build_service(tmp_path, manager)
    child_tasks = []
    proposal_started = asyncio.Event()
    proposal_stopped = asyncio.Event()
    original_self_ask = service._self_ask

    async def tracked_self_ask(*args, **kwargs):
        task = asyncio.current_task()
        assert task is not None
        child_tasks.append(task)
        return await original_self_ask(*args, **kwargs)

    service._self_ask = tracked_self_ask
    refresh_kwargs = {}
    if candidate_source == "fixed":
        document = repo.snapshot.documents[0]

        async def fixed_candidates(candidates):
            task = asyncio.current_task()
            assert task is not None
            child_tasks.append(task)
            proposal_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                proposal_stopped.set()
            return candidates

        service._fixed_candidates = fixed_candidates
        refresh_kwargs["candidates"] = (
            repo.candidate_for_document(document.document_id, "event-cancel"),
        )
    else:
        service.tensor_bank = BlockingTensorBank(
            child_tasks, proposal_started, proposal_stopped
        )
        service.query_probe = SimpleNamespace(
            probe=lambda *_args, **_kwargs: asyncio.sleep(
                0,
                result=SimpleNamespace(
                    status="ready",
                    query_heads=(((1.0, 0.0),),),
                    query_states=(QueryStateSpan("current_user", 0, 1, 0, 1),),
                ),
            )
        )
        refresh_kwargs["event"] = MidThinkEvent(
            event_id="event-cancel",
            request_id="request-cancel",
            token_index=16,
            trigger_reasons=("selected_token_surprisal_window",),
            current_surprisal=7.0,
            window_mean=6.5,
            history_mean=2.0,
            ema_surprisal=4.0,
            recovery_window_mean=6.2,
            uncertainty_state="persistent_uncertainty",
            pre_q_sketches=tuple((1.0, 0.0) for _ in range(8)),
            post_q_sketches=tuple((1.0, 0.0) for _ in range(4)),
        )

    refresh_task = asyncio.create_task(
        service.refresh(
            parent_request_id="request-cancel",
            turn_id="turn-cancel",
            user_question="Which exact constant is needed?",
            partial_output="The exact identifier is uncertain.",
            **refresh_kwargs,
        )
    )
    await asyncio.wait_for(
        asyncio.gather(manager.started.wait(), proposal_started.wait()),
        timeout=1,
    )

    refresh_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh_task

    assert len(child_tasks) == 2
    assert all(task.done() and task.cancelled() for task in child_tasks)
    assert manager.stopped.is_set()
    assert proposal_stopped.is_set()
    assert len(manager.aborted) == 1
    assert service.runner._active == {}
    assert service._inflight == set()
    assert service.record("request-cancel") is None


@pytest.mark.asyncio
async def test_self_ask_failure_cancels_and_awaits_proposed_task(tmp_path):
    service, repo = build_service(tmp_path, FakeManager())
    proposal_started = asyncio.Event()
    proposal_stopped = asyncio.Event()
    proposed_task = None

    async def failing_self_ask(*_args, **_kwargs):
        await proposal_started.wait()
        raise RuntimeError("self ask failed")

    async def fixed_candidates(candidates):
        nonlocal proposed_task
        proposed_task = asyncio.current_task()
        proposal_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            proposal_stopped.set()
        return candidates

    service._self_ask = failing_self_ask
    service._fixed_candidates = fixed_candidates
    document = repo.snapshot.documents[0]
    record = await service.refresh(
        parent_request_id="request-failure",
        turn_id="turn-failure",
        user_question="Which exact constant is needed?",
        partial_output="The exact identifier is uncertain.",
        candidates=(
            repo.candidate_for_document(document.document_id, "event-failure"),
        ),
    )

    assert record.status == "failed_closed:RuntimeError"
    assert proposed_task is not None
    assert proposed_task.done() and proposed_task.cancelled()
    assert proposal_stopped.is_set()
    assert service.runner._active == {}
    assert service._inflight == set()


@pytest.mark.asyncio
async def test_self_ask_uses_json_tool_contract_for_chinese_question(tmp_path):
    expected = "生成的异步包装器当前签名是什么，位于哪个源文件和行号？"
    manager = FakeManager(question=expected, question_kind="factual")
    service, _repo = build_service(tmp_path, manager)

    question = await service._self_ask(
        parent_request_id="request-chinese",
        turn_id="turn-chinese",
        user_question="修复 Python SDK 的异步包装器。",
        partial_output="最新测试显示 keyword-only timeout 没有被转发。",
        purpose="post_tool",
    )

    assert question is not None
    assert question.text == expected
    assert question.kind == "factual"
    request = manager.requests[0]
    schema = json.loads(request.sampling_params[0]["json_schema"])
    assert schema["properties"]["tool"]["enum"] == [
        "submit_self_question",
        "skip_self_question",
    ]
    assert "required" not in schema["properties"]["arguments"]
    assert schema["properties"]["arguments"]["properties"]["question"] == {
        "type": "string",
        "maxLength": 96,
    }
    assert schema["properties"]["kind"] == {
        "type": "string",
        "enum": ["skip", "factual"],
    }
    assert request.sampling_params[0]["max_new_tokens"] == 160
    assert "call the submit_self_question tool exactly once" in request.text[0]
    assert "skip_self_question tool with top-level kind=skip" in request.text[0]
    assert "QUESTION|" not in request.text[0]
    assert "Never create verification questions" in request.text[0]


@pytest.mark.asyncio
async def test_self_ask_can_skip_when_no_material_question_exists(tmp_path):
    manager = FakeManager(
        question_payload=json.dumps(
            {"tool": "skip_self_question", "kind": "skip", "arguments": {}}
        )
    )
    service, _repo = build_service(tmp_path, manager)

    question = await service._self_ask(
        parent_request_id="request-skip",
        turn_id="turn-skip",
        user_question="Implement the requested change.",
        partial_output="The change is complete and the focused test passes.",
        purpose="post_tool",
    )

    assert question is None


@pytest.mark.asyncio
async def test_self_ask_reuses_only_identical_classification_prompts(tmp_path):
    manager = FakeManager(question="Which exact SDK symbol is required?")
    service, _repo = build_service(tmp_path, manager)

    first = await service._self_ask(
        parent_request_id="request-cache-1",
        turn_id="turn-cache-1",
        user_question="Repair the SDK wrapper.",
        partial_output="first tool observation",
        purpose="post_tool",
    )
    repeated = await service._self_ask(
        parent_request_id="request-cache-2",
        turn_id="turn-cache-2",
        user_question="Repair the SDK wrapper.",
        partial_output="different tool observation",
        purpose="post_tool",
    )
    changed = await service._self_ask(
        parent_request_id="request-cache-3",
        turn_id="turn-cache-3",
        user_question="Repair the SDK wrapper.",
        partial_output="different reasoning excerpt",
        purpose="mid_think",
    )

    assert first == repeated == changed
    assert len(manager.requests) == 2
    cache_hits = [
        event
        for event in service.telemetry.events()
        if event.event_type == "self_ask.cache_hit"
    ]
    assert len(cache_hits) == 1
    assert cache_hits[0].request_id == "request-cache-2"
    assert cache_hits[0].payload["skipped"] is False


@pytest.mark.asyncio
async def test_self_ask_caches_a_schema_valid_skip(tmp_path):
    manager = FakeManager(
        question_payload=json.dumps(
            {"tool": "skip_self_question", "kind": "skip", "arguments": {}}
        )
    )
    service, _repo = build_service(tmp_path, manager)

    first = await service._self_ask(
        parent_request_id="request-skip-cache-1",
        turn_id="turn-skip-cache-1",
        user_question="Implement the requested change.",
        partial_output="first result",
        purpose="post_tool",
    )
    repeated = await service._self_ask(
        parent_request_id="request-skip-cache-2",
        turn_id="turn-skip-cache-2",
        user_question="Implement the requested change.",
        partial_output="second result",
        purpose="post_tool",
    )

    assert first is None and repeated is None
    assert len(manager.requests) == 1
    cache_hit = next(
        event
        for event in service.telemetry.events()
        if event.event_type == "self_ask.cache_hit"
    )
    assert cache_hit.payload["skipped"] is True


@pytest.mark.asyncio
async def test_self_ask_rejects_question_beyond_schema_limit(tmp_path):
    manager = FakeManager(question="x" * 97)
    service, _repo = build_service(tmp_path, manager)

    with pytest.raises(ValueError, match="invalid length"):
        await service._self_ask(
            parent_request_id="request-long-question",
            turn_id="turn-long-question",
            user_question="Find the unresolved boundary.",
            partial_output="The focused test failed.",
            purpose="post_tool",
        )


@pytest.mark.asyncio
async def test_self_ask_rejects_wrong_json_tool(tmp_path):
    manager = FakeManager(
        question_payload=json.dumps(
            {
                "tool": "answer_question",
                "kind": "factual",
                "arguments": {"question": "Which wrapper is stale?"},
            }
        )
    )
    service, _repo = build_service(tmp_path, manager)

    with pytest.raises(ValueError, match="wrong tool"):
        await service._self_ask(
            parent_request_id="request-wrong-tool",
            turn_id="turn-wrong-tool",
            user_question="Find the stale wrapper.",
            partial_output="The focused test failed.",
            purpose="post_tool",
        )


@pytest.mark.asyncio
async def test_self_ask_rejects_decision_question_kind(tmp_path):
    manager = FakeManager(
        question_payload=json.dumps(
            {
                "tool": "submit_self_question",
                "kind": "decision",
                "arguments": {
                    "question": "Should the server use WebSocket or TCP?",
                },
            }
        )
    )
    service, _repo = build_service(tmp_path, manager)

    with pytest.raises(ValueError, match="kind is invalid"):
        await service._self_ask(
            parent_request_id="request-decision-kind",
            turn_id="turn-decision-kind",
            user_question="Implement client communication.",
            partial_output="A transport choice is unresolved.",
            purpose="post_tool",
        )


@pytest.mark.asyncio
async def test_refresh_replays_only_eligible_references(tmp_path):
    service, repo = build_service(tmp_path, FakeManager())

    record = await service.refresh(
        parent_request_id="request-1",
        turn_id="turn-1",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="I am uncertain about the exact constant.",
    )

    selected_paths = {
        repo.get(document_id).relative_path
        for document_id in record.selected_document_ids
    }
    assert record.status == "semantic_ready"
    assert selected_paths == {"wfp.md"}
    assert "WFP_LAYER_ALE_AUTH_CONNECT_V4" in record.answer
    assert service.record("request-1") is record


@pytest.mark.asyncio
async def test_refresh_uses_policy_data_without_reference_judge(tmp_path):
    manager = FakeManager(answer="Run focused regression tests before delivery.")
    service, _repo = build_service(tmp_path, manager)
    policy_data = PolicyDataRepository(tmp_path / "policydata")
    policy_data.upsert(
        "delivery.md",
        "For WFP code changes, run focused regression tests before delivery.",
    )
    service.policy_data = policy_data

    record = await service.refresh(
        parent_request_id="request-policy-direct",
        turn_id="turn-policy-direct",
        user_question="Implement the WFP change safely.",
        partial_output="I need the required delivery policy.",
    )

    assert record.status == "semantic_ready"
    assert record.selected_lanes[0] == "policydata"
    assert record.answer == "Run focused regression tests before delivery."
    judge_prompts = [
        request.text[0]
        for request in manager.requests
        if "Judge whether the supplied candidate" in request.text[0]
    ]
    assert all("focused regression tests" not in prompt for prompt in judge_prompts)


@pytest.mark.asyncio
async def test_qk_policydata_requires_semantic_applicability_judge(tmp_path):
    manager = FakeManager()
    service, repo = build_service(tmp_path, manager)
    ctf = next(
        document
        for document in repo.snapshot.documents
        if document.relative_path == "ctf.md"
    )
    qk_policy = replace(
        repo.candidate_for_document(ctf.document_id, "qk-policy"),
        lane="policydata",
        candidate_origin="attention_q_native_tensor_bank",
    )

    record = await service.refresh(
        parent_request_id="request-qk-policy",
        turn_id="turn-qk-policy",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
        candidates=(qk_policy,),
    )

    assert record.status == "no_eligible_reference"
    assert record.selected_document_ids == ()
    assert record.answer is None
    assert service.eligible_candidates("request-qk-policy") == ()
    judge_prompts = [
        request.text[0]
        for request in manager.requests
        if "Judge whether the supplied candidate" in request.text[0]
    ]
    assert len(judge_prompts) == 1
    assert '"lane":"policydata"' in judge_prompts[0]
    completed = [
        event.payload
        for event in service.telemetry.events("request-qk-policy")
        if event.event_type == "semantic_judge.completed"
    ]
    assert completed[-1]["candidate_count"] == 1
    assert completed[-1]["bypassed_count"] == 0


@pytest.mark.asyncio
async def test_execution_policy_is_excluded_from_self_ask_evidence(tmp_path):
    manager = FakeManager()
    service, _repo = build_service(tmp_path, manager)
    policy_data = PolicyDataRepository(tmp_path / "policydata")
    document = policy_data.upsert(
        "coding-agent-execution-policy.md",
        """---
canonical: true
quality: 1.0
source_kind: coding_agent_execution_policy
---
# Engineering Change Execution Policy

Act as an evidence-first coding agent.
""",
    )
    service.policy_data = policy_data
    candidate = policy_data.candidate_for_document(document.document_id, "base")

    record = await service.refresh(
        parent_request_id="request-base-policy",
        turn_id="turn-base-policy",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
        candidates=(candidate,),
    )

    assert record.status == "no_eligible_reference"
    assert record.candidate_ids == ()
    assert record.selected_document_ids == ()
    assert service.eligible_candidates("request-base-policy") == ()
    assert all(
        "coding-agent-execution-policy.md" not in str(request.text)
        for request in manager.requests
    )


@pytest.mark.asyncio
async def test_refresh_fails_closed_when_self_answer_is_invalid(tmp_path):
    service, _repo = build_service(tmp_path, FakeManager(invalid_answer=True))

    record = await service.refresh(
        parent_request_id="request-2",
        turn_id="turn-2",
        user_question="Which WFP AppID?",
        partial_output="Unknown.",
    )

    assert record.status.startswith("failed_closed:")
    assert record.selected_document_ids == ()
    assert record.answer is None


@pytest.mark.asyncio
async def test_not_covered_answer_is_not_committed_as_think_context(tmp_path):
    service, _repo = build_service(tmp_path, FakeManager(answer="Not covered."))

    record = await service.refresh(
        parent_request_id="request-not-covered",
        turn_id="turn-not-covered",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
    )

    assert record.status == "no_answering_evidence"
    assert record.question_kind == "factual"
    assert record.reference_status == "eligible_but_not_answering"
    assert record.answer is None
    assert record.semantic_injection is None
    assert record.selected_document_ids == ()
    assert service.eligible_candidates("request-not-covered") == ()
    completed = [
        event
        for event in service.telemetry.events("request-not-covered")
        if event.event_type == "evidence_answer.completed"
    ]
    assert completed[-1].payload["decision"] == "not_covered"
    assert completed[-1].payload["semantic_injection_tokens"] == 0


@pytest.mark.asyncio
async def test_refresh_binds_and_residents_qk_native_state_only_after_judge(tmp_path):
    manager = FakeManager(all_supported=True)
    bank = JudgeGatedBindingTensorBank(manager)
    service, repo = build_service(tmp_path, manager, tensor_bank=bank)
    document = next(
        item for item in repo.snapshot.documents if item.relative_path == "wfp.md"
    )
    candidate = replace(
        repo.candidate_for_document(document.document_id, "event-judge-gated"),
        score=0.95,
        lexical_score=0.0,
        tensor_score=0.95,
        page_ids=(7,),
        source_positions=(),
        virtual_positions=(),
        native_prefix=None,
        candidate_origin="attention_q_native_tensor_bank",
    )

    record = await service.refresh(
        parent_request_id="request-judge-gated",
        turn_id="turn-judge-gated",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
        candidates=(candidate,),
    )

    assert record.status == "semantic_ready"
    assert bank.bind_calls == [
        (candidate.candidate_id, "What is the required WFP AppID?")
    ]
    assert bank.resident_page_ids == (7,)
    (eligible,) = service.eligible_candidates("request-judge-gated")
    assert eligible.native_prefix is not None
    assert eligible.candidate_origin == "admitted_native_tensor_bank"


@pytest.mark.asyncio
async def test_mid_think_self_ask_is_independent_from_tensor_candidates(tmp_path):
    manager = FakeManager()
    service, repo = build_service(tmp_path, manager)
    documents = {
        document.relative_path: document for document in repo.snapshot.documents
    }
    tensor_candidates = (
        repo.candidate_for_document(documents["wfp.md"].document_id, "event-1"),
        repo.candidate_for_document(documents["ctf.md"].document_id, "event-1"),
    )
    event = MidThinkEvent(
        event_id="event-1",
        request_id="request-3",
        token_index=16,
        trigger_reasons=("selected_token_surprisal_window",),
        current_surprisal=7.0,
        window_mean=6.5,
        history_mean=2.0,
        ema_surprisal=4.0,
        recovery_window_mean=6.2,
        uncertainty_state="persistent_uncertainty",
        pre_q_sketches=tuple((1.0, 0.0) for _ in range(8)),
        post_q_sketches=tuple((1.0, 0.0) for _ in range(4)),
    )

    record = await service.refresh(
        parent_request_id="request-3",
        turn_id="turn-3",
        user_question="Which exact constant is needed?",
        partial_output="The exact identifier is uncertain.",
        event=event,
        candidates=tensor_candidates,
    )

    self_ask_prompt = manager.requests[0].text[0]
    assert "wfp.md" not in self_ask_prompt.lower()
    assert "ctf.md" not in self_ask_prompt.lower()
    assert "WFP_LAYER_ALE_AUTH_CONNECT_V4" not in self_ask_prompt
    assert record.event_id == "event-1"
    assert record.candidate_ids == tuple(
        candidate.candidate_id for candidate in tensor_candidates
    )
    assert record.status == "semantic_ready"
    assert record.selected_document_ids == (documents["wfp.md"].document_id,)
    winner = service.eligible_candidates("request-3")[0]
    admitted = await service.complete_replay(
        "request-3",
        replay_decision="shadow_would_switch",
        winner_candidate_id=winner.candidate_id,
        gain=0.5,
        kl=0.1,
        maybe_decision="admit_maybe",
        scheduled_next_turn=True,
    )
    assert admitted is not None
    assert admitted.status == "ready_for_safe_replay"
    assert admitted.selected_document_ids == (winner.document_id,)
    assert admitted.selected_reference_digests == (winner.reference_digest,)
    assert admitted.selected_lanes == ("knowledge",)
    assert admitted.candidate_page_ids == winner.page_ids
    assert admitted.source_positions == winner.source_positions
    assert admitted.replay_winner_decision_id is not None
    assert "WFP_LAYER_ALE_AUTH_CONNECT_V4" in admitted.answer
    assert admitted.semantic_injection is not None


@pytest.mark.asyncio
async def test_replay_winner_narrowing_drops_multi_source_answer(tmp_path):
    marker = "B_ONLY_MARKER_MUST_NOT_RESTORE"
    manager = FakeManager(
        all_supported=True,
        answer=f"Source A is relevant; source B says {marker}.",
    )
    service, repo = build_service(tmp_path, manager)
    documents = {
        document.relative_path: document for document in repo.snapshot.documents
    }
    candidates = (
        repo.candidate_for_document(documents["wfp.md"].document_id, "event-ab"),
        repo.candidate_for_document(documents["ctf.md"].document_id, "event-ab"),
    )
    record = await service.refresh(
        parent_request_id="request-ab",
        turn_id="turn-ab",
        user_question="Which facts are required?",
        partial_output="Both sources may matter.",
        candidates=candidates,
    )
    assert record.answer is not None and marker in record.answer
    assert len(service.eligible_candidates("request-ab")) == 2

    winner = candidates[0]
    admitted = await service.complete_replay(
        "request-ab",
        replay_decision="shadow_would_switch",
        winner_candidate_id=winner.candidate_id,
        gain=0.5,
        kl=0.1,
        maybe_decision="admit_maybe",
        scheduled_next_turn=True,
    )

    assert admitted is not None
    assert admitted.status == "ready_for_safe_replay"
    assert admitted.selected_document_ids == (winner.document_id,)
    assert admitted.answer is None
    assert admitted.semantic_injection is None
    attachment, _attached_tokens = MemoryPipeline._compile_attachment(
        SimpleNamespace(
            tokenizer=manager,
            config=SimpleNamespace(max_memory_tokens=1024),
        ),
        (winner,),
        restored_answer=admitted.answer,
    )
    assert attachment is not None
    assert marker not in attachment


class _ReplayTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in str(text)]


class _ReplayRunner:
    def __init__(self):
        self.calls = 0

    async def run_score_batch(self, jobs, input_ids, label_starts, sampling_params):
        self.calls += 1
        results = []
        for index, job in enumerate(jobs):
            logprob = -2.0 if index == 0 else -1.0
            results.append(
                InternalScoreResult(
                    job=job,
                    token_logprobs=tuple(logprob for _ in range(8)),
                    mean_nll=-logprob,
                    prompt_tokens=len(input_ids[index]),
                    finish_reason={"type": "stop"},
                    latency_seconds=0.01,
                )
            )
        return tuple(results)


@pytest.mark.asyncio
async def test_causal_replay_admits_only_eligible_gain_and_schedules_maybe(tmp_path):
    repo = repository(tmp_path)
    document = next(
        item for item in repo.snapshot.documents if item.relative_path == "wfp.md"
    )
    candidate = replace(
        repo.candidate_for_document(document.document_id, "event-2"),
        source_positions=tuple(range(16)),
        candidate_origin="attention_q_tensor_bank",
    )
    decision = EligibilityDecision.create(
        candidate_id=candidate.candidate_id,
        parent_request_id="request-replay",
        question="What is the required WFP AppID?",
        reference=candidate.reference_content,
        status=EligibilityStatus.ELIGIBLE,
        judge_method="strict-test",
        judge_model_fingerprint="model",
        decision_margin=0.0,
    )
    event = MidThinkEvent(
        event_id="event-2",
        request_id="request-replay",
        token_index=4,
        trigger_reasons=("selected_token_surprisal_window",),
        current_surprisal=7.0,
        window_mean=6.0,
        history_mean=1.0,
        ema_surprisal=4.0,
        recovery_window_mean=6.0,
        uncertainty_state="persistent_uncertainty",
        pre_q_sketches=tuple((1.0, 0.0) for _ in range(8)),
        post_q_sketches=tuple((1.0, 0.0) for _ in range(4)),
        generation_index=0,
        generation_token_index=4,
    )
    runner = _ReplayRunner()
    service = CausalReplayService(
        runner,
        _ReplayTokenizer(),
        TelemetryStore(tmp_path / "replay-trace.jsonl"),
    )

    result = await service.evaluate(
        parent_request_id="request-replay",
        event=event,
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(12)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert runner.calls == 1
    assert result.decision == "shadow_would_switch"
    assert result.maybe_decision == "admit_maybe"
    assert result.scheduled_next_turn is True
    assert result.winner_candidate_id == candidate.candidate_id
    assert result.losses[1].decision_id == decision.decision_id


@pytest.mark.asyncio
async def test_causal_replay_never_scores_ineligible_candidate(tmp_path):
    repo = repository(tmp_path)
    document = next(
        item for item in repo.snapshot.documents if item.relative_path == "wfp.md"
    )
    candidate = repo.candidate_for_document(document.document_id, "event-3")
    decision = EligibilityDecision.create(
        candidate_id=candidate.candidate_id,
        parent_request_id="request-reject",
        question="question",
        reference=candidate.reference_content,
        status=EligibilityStatus.INELIGIBLE,
        judge_method="strict-test",
        judge_model_fingerprint="model",
        decision_margin=None,
    )
    event = MidThinkEvent(
        event_id="event-3",
        request_id="request-reject",
        token_index=4,
        trigger_reasons=("selected_token_surprisal_window",),
        current_surprisal=7.0,
        window_mean=6.0,
        history_mean=1.0,
        ema_surprisal=4.0,
        recovery_window_mean=6.0,
        uncertainty_state="persistent_uncertainty",
        pre_q_sketches=(),
        post_q_sketches=(),
        generation_token_index=4,
    )
    runner = _ReplayRunner()
    service = CausalReplayService(
        runner,
        _ReplayTokenizer(),
        TelemetryStore(tmp_path / "reject-trace.jsonl"),
    )

    result = await service.evaluate(
        parent_request_id="request-reject",
        event=event,
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(12)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert runner.calls == 0
    assert result.decision == "reject_no_semantic_candidate"
    assert result.scheduled_next_turn is False


@pytest.mark.asyncio
async def test_maybe_gate_rejects_distribution_shift_above_kl_cap(tmp_path):
    repo = repository(tmp_path)
    document = next(
        item for item in repo.snapshot.documents if item.relative_path == "wfp.md"
    )
    candidate = repo.candidate_for_document(document.document_id, "event-kl")
    decision = EligibilityDecision.create(
        candidate_id=candidate.candidate_id,
        parent_request_id="request-kl",
        question="question",
        reference=candidate.reference_content,
        status=EligibilityStatus.ELIGIBLE,
        judge_method="strict-test",
        judge_model_fingerprint="model",
        decision_margin=0.0,
    )
    event = MidThinkEvent(
        event_id="event-kl",
        request_id="request-kl",
        token_index=4,
        trigger_reasons=("selected_token_surprisal_window",),
        current_surprisal=7.0,
        window_mean=6.0,
        history_mean=1.0,
        ema_surprisal=4.0,
        recovery_window_mean=6.0,
        uncertainty_state="persistent_uncertainty",
        pre_q_sketches=(),
        post_q_sketches=(),
        generation_token_index=4,
    )
    service = CausalReplayService(
        _ReplayRunner(),
        _ReplayTokenizer(),
        TelemetryStore(tmp_path / "kl-trace.jsonl"),
        maybe_kl_cap=0.001,
    )

    result = await service.evaluate(
        parent_request_id="request-kl",
        event=event,
        prompt_ids=(1, 2, 3),
        output_ids=tuple(range(12)),
        candidates=(candidate,),
        decisions=(decision,),
    )

    assert result.decision == "shadow_would_switch"
    assert result.maybe_decision == "reject_maybe_kl"
    assert result.scheduled_next_turn is False


@pytest.mark.asyncio
async def test_post_tool_recall_runs_per_tool_turn_and_latest_record_wins(tmp_path):
    manager = FakeManager()
    service, _repo = build_service(tmp_path, manager)

    first = await service.refresh(
        parent_request_id="request-tools",
        turn_id="request-tools:post_tool:0",
        user_question="Which WFP AppID constant is required?",
        partial_output="TOOL OBSERVATION: first result",
        purpose="post_tool",
    )
    second = await service.refresh(
        parent_request_id="request-tools",
        turn_id="request-tools:post_tool:1",
        user_question="Which WFP AppID constant is required?",
        partial_output="TOOL OBSERVATION: corrected result",
        purpose="post_tool",
    )

    assert first.turn_id.endswith(":0")
    assert second.turn_id.endswith(":1")
    assert second.status == "ready_for_safe_replay"
    assert second.maybe_decision == "admit_post_tool"
    assert service.record("request-tools") is second
    self_ask_prompts = [
        request.text[0]
        for request in manager.requests
        if "internal document-retrieval question classifier" in request.text[0]
    ]
    assert len(self_ask_prompts) == 1
    assert "corrected result" not in self_ask_prompts[0]
    cache_hit = next(
        event
        for event in service.telemetry.events()
        if event.event_type == "self_ask.cache_hit"
    )
    assert cache_hit.payload["turn_id"] == "request-tools:post_tool:1"


@pytest.mark.asyncio
async def test_policy_candidate_never_becomes_a_gap_reflection(tmp_path):
    manager = FakeManager()
    service, repo = build_service(tmp_path, manager)
    service.context_evidence_mode = "active"
    ctf = next(
        document
        for document in repo.snapshot.documents
        if document.relative_path == "ctf.md"
    )
    policy_candidate = replace(
        repo.candidate_for_document(ctf.document_id, "policy-only"),
        lane="policydata",
    )

    record = await service.refresh(
        parent_request_id="request-policy-context",
        turn_id="request-policy-context:post_tool:0",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The answer still needs direct evidence.",
        purpose="post_tool",
        candidates=(policy_candidate,),
        latest_tool_observation=(
            "Direct test output used WFP_LAYER_ALE_AUTH_CONNECT_V4 successfully."
        ),
    )

    assert record.status == "ready_for_safe_replay"
    assert record.selected_lanes == ("policydata", "context")
    assert record.context_status == "eligible"
    assert record.reflection_kind == "combined_evidence"
    assert record.semantic_injection == (
        "\n\nSelf-question: What is the required WFP AppID?\n"
        "Self-answer: Use WFP_LAYER_ALE_AUTH_CONNECT_V4.\n"
    )
    assert "GAP=" not in record.semantic_injection
    assert "qwen_exo" not in record.semantic_injection
    answer_prompt = next(
        request.text[0]
        for request in manager.requests
        if "Answer the classified self-question" in request.text[0]
    )
    answer_payload = json.loads(answer_prompt.rsplit("\n", 1)[-1])
    assert answer_payload["question"]["kind"] == "factual"
    assert answer_payload["knowledge"][0]["lane"] == "policydata"
    assert (
        "WFP_LAYER_ALE_AUTH_CONNECT_V4"
        in answer_payload["recent_context"][0]["content"]
    )
    assert "The answer still needs direct evidence" not in answer_prompt


@pytest.mark.asyncio
async def test_post_tool_context_evidence_admits_direct_observation(tmp_path):
    service, repo = build_service(tmp_path, FakeManager())
    service.context_evidence_mode = "active"
    ctf = next(
        document
        for document in repo.snapshot.documents
        if document.relative_path == "ctf.md"
    )

    record = await service.refresh(
        parent_request_id="request-context",
        turn_id="request-context:post_tool:0",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
        purpose="post_tool",
        candidates=(repo.candidate_for_document(ctf.document_id, "context"),),
        latest_tool_observation=(
            "Direct test output used WFP_LAYER_ALE_AUTH_CONNECT_V4 successfully."
        ),
    )

    assert record.status == "context_evidence_ready"
    assert record.reference_status == "no_eligible_reference"
    assert record.context_status == "eligible"
    assert record.selected_document_ids == ()
    assert record.selected_reference_digests == ()
    assert record.selected_lanes == ("context",)
    assert record.context_source_digests
    assert record.context_decision_ids
    assert service.eligible_candidates("request-context") == ()
    completed = [
        event
        for event in service.telemetry.events("request-context")
        if event.event_type == "context_evidence.completed"
    ]
    assert completed[-1].payload["status"] == "eligible"
    assert completed[-1].payload["answer"]["redacted"] is True
    routed = [
        event
        for event in service.telemetry.events("request-context")
        if event.event_type == "self_ask.routed"
    ]
    assert routed[-1].payload["evidence_route"] == "request_local_tool_observation"
    proposed = [
        event
        for event in service.telemetry.events("request-context")
        if event.event_type == "tensor.candidates_proposed"
    ]
    assert len(proposed[-1].payload["candidates"]) == 1


@pytest.mark.asyncio
async def test_context_evidence_does_not_treat_reasoning_as_evidence(tmp_path):
    service, repo = build_service(tmp_path, FakeManager())
    service.context_evidence_mode = "active"
    ctf = next(
        document
        for document in repo.snapshot.documents
        if document.relative_path == "ctf.md"
    )

    record = await service.refresh(
        parent_request_id="request-reasoning-only",
        turn_id="request-reasoning-only:post_tool:0",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="I think it is WFP_LAYER_ALE_AUTH_CONNECT_V4.",
        purpose="post_tool",
        candidates=(repo.candidate_for_document(ctf.document_id, "reasoning"),),
        latest_tool_observation="The file write completed successfully.",
    )

    assert record.status == "no_eligible_reference"
    assert record.reference_status == "no_eligible_reference"
    assert record.context_status == "ineligible"
    assert record.answer is None
    assert record.semantic_injection is None
    assert record.context_source_digests == ()


def test_context_modes_reject_removed_shadow(tmp_path):
    with pytest.raises(ValueError, match="Context evidence mode"):
        build_service(tmp_path, FakeManager(), context_evidence_mode="shadow")
    integrity_path = tmp_path / "integrity"
    integrity_path.mkdir()
    with pytest.raises(ValueError, match="Context integrity mode"):
        build_service(integrity_path, FakeManager(), context_integrity_mode="shadow")


@pytest.mark.asyncio
async def test_active_context_integrity_commits_grounded_correction(tmp_path):
    observation = (
        "The direct probe selected WFP_LAYER_ALE_AUTH_CONNECT_V4 for outbound connect."
    )
    manager = FakeManager(
        integrity_payload=json.dumps(
            {
                "status": "corrected",
                "confirmed_facts": [
                    "The latest probe names the outbound connect layer."
                ],
                "invalid_claims": [
                    "The prior capsule named an inbound accept layer for this task."
                ],
                "contradictions": [
                    "The current tool result contradicts the prior inbound classification."
                ],
                "stale_assumptions": ["The prior direction assumption is stale."],
                "correction": (
                    "Use WFP_LAYER_ALE_AUTH_CONNECT_V4 for outbound connect. "
                    "<evidence>WFP_LAYER_ALE_AUTH_CONNECT_V4</evidence>"
                ),
                "evidence_needed": [],
                "confidence": 0.9,
            }
        )
    )
    service, _ = build_service(
        tmp_path,
        manager,
        context_integrity_mode="active",
    )

    result = await service.context_integrity_check(
        parent_request_id="request-integrity",
        turn_id="request-integrity:post_tool:0",
        original_task="Configure outbound WFP filtering",
        session_context="Prior capsule assumed an inbound accept layer.",
        current_tool_observation=observation,
    )
    integrity_prompt = manager.requests[-1].text[0]
    assert "latest_tool_content" in integrity_prompt
    assert "recent_session_context" in integrity_prompt
    assert "current_tool_call" not in integrity_prompt
    assert "prior_tool_result_ledger" not in integrity_prompt
    record = await service.commit_context_integrity(
        parent_request_id="request-integrity",
        turn_id="request-integrity:post_tool:0",
        result=result,
    )

    assert result.status == "corrected"
    assert result.injectable is True
    assert result.evidence_quote == "WFP_LAYER_ALE_AUTH_CONNECT_V4"
    assert record.status == "context_integrity_ready"
    assert record.semantic_injection == (
        "\n\nContext Integrity Correction: "
        "Use WFP_LAYER_ALE_AUTH_CONNECT_V4 for outbound connect.\n"
    )
    event_types = [
        event.event_type for event in service.telemetry.events("request-integrity")
    ]
    assert "context_integrity.completed" in event_types
    assert "context_integrity.applied" in event_types


@pytest.mark.asyncio
async def test_context_integrity_source_budget_follows_configured_total(tmp_path):
    manager = FakeManager()
    service, _ = build_service(
        tmp_path,
        manager,
        context_integrity_mode="active",
        context_integrity_max_tokens=512,
    )
    words = [f"token-{index}" for index in range(400)]
    sources, budgets = service._fit_integrity_sources(
        original_task=" ".join(words),
        session_context=" ".join(words),
        current_tool_observation=" ".join(words),
    )

    assert sum(budgets.values()) <= 512
    assert sum(len(manager.encode(text)) for text in sources.values()) <= 512
    assert "token-399" in sources["recent_session_context"]
    assert "token-0" not in sources["recent_session_context"]


@pytest.mark.asyncio
async def test_context_evidence_failure_does_not_fail_refresh(tmp_path, monkeypatch):
    service, repo = build_service(tmp_path, FakeManager())
    service.context_evidence_mode = "active"
    ctf = next(
        document
        for document in repo.snapshot.documents
        if document.relative_path == "ctf.md"
    )

    def fail_context_candidates(**_kwargs):
        raise ValueError("broken context tokenizer")

    monkeypatch.setattr(service, "_context_candidates", fail_context_candidates)
    record = await service.refresh(
        parent_request_id="request-context-failure",
        turn_id="request-context-failure:post_tool:0",
        user_question="Which WFP AppID controls outbound connect?",
        partial_output="The exact constant remains uncertain.",
        purpose="post_tool",
        candidates=(repo.candidate_for_document(ctf.document_id, "failure"),),
        latest_tool_observation="Direct test output is available.",
    )

    assert record.status == "no_eligible_reference"
    assert record.context_status == "failed_closed:ValueError"
    assert record.answer is None
    assert record.semantic_injection is None
