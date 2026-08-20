import asyncio
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from qwen_exo_booster.config import QwenExoConfig, QwenExoFeatureFlags
from qwen_exo_booster.contracts import (
    EligibilityDecision,
    EligibilityStatus,
    stable_digest,
)
from qwen_exo_booster.knowledge import KnowledgeRepository, NativePrefixSelection
from qwen_exo_booster.pipeline import MemoryPipeline, response_memory_metadata
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.query_probe import QueryStateSpan


@dataclass(frozen=True)
class FakeRequest:
    request_id: str
    input: object
    instructions: str | None = None
    previous_response_id: str | None = None
    extra_key: str | None = None

    def model_copy(self, update):
        return replace(self, **update)


class FakeTokenizer:
    def __init__(self):
        self._tokens = {}
        self._next = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        for word in text.split():
            if word not in self._tokens:
                self._tokens[word] = self._next
                self._next += 1
            result.append(self._tokens[word])
        return result

    def decode(self, token_ids, **_kwargs):
        reverse = {value: key for key, value in self._tokens.items()}
        return " ".join(reverse[token_id] for token_id in token_ids)


class FakeQKTensorBank:
    def __init__(self, candidates=()):
        self.candidates = tuple(candidates)
        self.rank_calls = []
        self.rank_gates = []
        self.ensure_ready_calls = 0

    async def ensure_ready(self):
        self.ensure_ready_calls += 1
        return SimpleNamespace(ready=True)

    def rank(
        self,
        query_heads,
        query_states,
        *,
        query_identity,
        limit,
        min_tensor_score=0.0,
        min_document_margin=0.005,
        audit=None,
    ):
        self.rank_calls.append(
            (query_heads, query_states, query_identity, limit, min_document_margin)
        )
        self.rank_gates.append((min_tensor_score, min_document_margin))
        candidates = self.candidates[:limit]
        if audit is not None:
            audit.update(
                status="ready",
                reason=(
                    "candidates_ready" if candidates else "all_scores_below_threshold"
                ),
                min_tensor_score=min_tensor_score,
                min_document_margin=min_document_margin,
                candidate_count=len(candidates),
            )
        return candidates

    def bind_native_prefix(self, candidate, *, query, preferred_page_ids=()):
        del query
        match = next(
            (
                item
                for item in self.candidates
                if item.lane == candidate.lane
                and item.document_id == candidate.document_id
                and item.reference_digest == candidate.reference_digest
            ),
            None,
        )
        if match is None:
            return candidate
        if match.native_prefix is None:
            page_id = preferred_page_ids[0] if preferred_page_ids else match.page_ids[0]
            native = NativePrefixSelection(
                source_digest="b" * 64,
                page_id=page_id,
                document_id=match.document_id,
                local_positions=tuple(range(64)),
                source_positions=tuple(range(4)),
                token_ids=tuple(range(page_id * 100, page_id * 100 + 64)),
                prefix_identity=f"admitted-{page_id}",
                radix_namespace=f"qwen-exo:v1:tensor-bank-native:{page_id}",
            )
            return replace(
                match,
                source_positions=native.source_positions,
                virtual_positions=tuple(range(len(native.source_positions))),
                native_prefix=native,
                candidate_origin="admitted_native_tensor_bank",
            )
        if preferred_page_ids and match.native_prefix.page_id != preferred_page_ids[0]:
            native = replace(match.native_prefix, page_id=preferred_page_ids[0])
            return replace(match, page_ids=preferred_page_ids, native_prefix=native)
        return match

    def selection_for_page(self, page_id):
        return next(
            item.native_prefix
            for item in self.candidates
            if item.native_prefix is not None and item.native_prefix.page_id == page_id
        )

    def page_lane(self, page_id):
        return next(
            item.lane
            for item in self.candidates
            if item.native_prefix is not None and item.native_prefix.page_id == page_id
        )


class MarginGatedFakeQKTensorBank(FakeQKTensorBank):
    def __init__(self, candidates, *, observed_margin):
        super().__init__(candidates)
        self.observed_margin = float(observed_margin)

    def rank(
        self,
        query_heads,
        query_states,
        *,
        query_identity,
        limit,
        min_tensor_score=0.0,
        min_document_margin=0.005,
        audit=None,
    ):
        self.rank_calls.append(
            (query_heads, query_states, query_identity, limit, min_document_margin)
        )
        self.rank_gates.append((min_tensor_score, min_document_margin))
        rejected = self.observed_margin < float(min_document_margin)
        candidates = () if rejected else self.candidates[:limit]
        if audit is not None:
            audit.update(
                status="rejected" if rejected else "ready",
                reason=(
                    "document_margin_too_small" if rejected else "candidates_ready"
                ),
                min_tensor_score=min_tensor_score,
                min_document_margin=min_document_margin,
                observed_margin=self.observed_margin,
                candidate_count=len(candidates),
            )
        return candidates


class FakeCognitionTensorBank(FakeQKTensorBank):
    def __init__(self, selection, lane="cognition"):
        super().__init__()
        self.selection = selection
        self.lane = lane
        self.snapshot = SimpleNamespace(pages=(SimpleNamespace(lane=lane),))

    def cognition_selection(self):
        return self.selection

    def page_lane(self, _page_id):
        return self.lane

    @staticmethod
    def cognition_token_ids():
        return (71, 72)


class FakeReferenceJudge:
    def __init__(
        self,
        supported=True,
        winner_path=None,
        *,
        question_original_tokens=0,
        question_review_tokens=0,
    ):
        self.supported = supported
        self.winner_path = winner_path
        self.question_original_tokens = int(question_original_tokens)
        self.question_review_tokens = int(question_review_tokens)
        self.calls = []
        self.selection_calls = []

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
        candidates = tuple(candidates)
        self.calls.append((parent_request_id, question, candidates))
        decisions = []
        for candidate in candidates:
            supported = (
                self.supported.get(candidate.relative_path, False)
                if isinstance(self.supported, dict)
                else bool(self.supported)
            )
            decisions.append(
                EligibilityDecision.create(
                    candidate_id=candidate.candidate_id,
                    parent_request_id=parent_request_id,
                    question=question,
                    reference=candidate.reference_content,
                    status=(
                        EligibilityStatus.ELIGIBLE
                        if supported
                        else EligibilityStatus.INELIGIBLE
                    ),
                    judge_method="fake_batch_judge",
                    judge_model_fingerprint="fake-model",
                    decision_margin=0.0,
                )
            )
        decisions = tuple(decisions)
        return SimpleNamespace(
            decisions=decisions,
            candidate_count=len(candidates),
            valid_count=len(candidates),
            eligible_count=sum(decision.eligible for decision in decisions),
            latency_seconds=0.001,
            cache_hit_count=0,
            executed_count=len(candidates),
            selection_method="independent_binary",
            selected_candidate_id=(
                next(
                    (
                        decision.candidate_id
                        for decision in decisions
                        if decision.eligible
                    ),
                    None,
                )
            ),
            presented_candidate_count=len(candidates),
            question_truncated=(
                self.question_original_tokens > self.question_review_tokens
            ),
            question_original_tokens=self.question_original_tokens,
            question_review_tokens=self.question_review_tokens,
        )

    async def select_best(
        self,
        *,
        parent_request_id,
        turn_id,
        question,
        candidates,
        telemetry_correlation_id,
    ):
        del turn_id, telemetry_correlation_id
        candidates = tuple(candidates)
        self.calls.append((parent_request_id, question, candidates))
        self.selection_calls.append((parent_request_id, question, candidates))
        supported_candidates = tuple(
            candidate
            for candidate in candidates
            if (
                self.supported.get(candidate.relative_path, False)
                if isinstance(self.supported, dict)
                else bool(self.supported)
            )
        )
        if self.winner_path is None:
            winner = next(iter(supported_candidates), None)
        else:
            winner = next(
                (
                    candidate
                    for candidate in supported_candidates
                    if candidate.relative_path == self.winner_path
                ),
                None,
            )
        decisions = tuple(
            EligibilityDecision.create(
                candidate_id=candidate.candidate_id,
                parent_request_id=parent_request_id,
                question=question,
                reference=candidate.reference_content,
                status=(
                    EligibilityStatus.ELIGIBLE
                    if candidate is winner
                    else EligibilityStatus.INELIGIBLE
                ),
                judge_method="fake_listwise_judge",
                judge_model_fingerprint="fake-model",
                decision_margin=0.0,
            )
            for candidate in candidates
        )
        return SimpleNamespace(
            decisions=decisions,
            candidate_count=len(candidates),
            valid_count=len(candidates),
            eligible_count=1 if winner is not None else 0,
            latency_seconds=0.001,
            cache_hit_count=0,
            executed_count=1,
            selection_method="comparative_listwise",
            selected_candidate_id=(winner.candidate_id if winner is not None else None),
            presented_candidate_count=len(candidates),
            question_truncated=(
                self.question_original_tokens > self.question_review_tokens
            ),
            question_original_tokens=self.question_original_tokens,
            question_review_tokens=self.question_review_tokens,
        )


class FakeTelemetry:
    def __init__(self):
        self.events = []

    def emit(self, request_id, event_type, payload):
        self.events.append((request_id, event_type, payload))

    def by_type(self, event_type):
        return [
            payload
            for _request_id, current_type, payload in self.events
            if current_type == event_type
        ]


def config(tmp_path, *, policy_data=True):
    return QwenExoConfig(
        state_directory=tmp_path / "state",
        knowledge_directory=tmp_path / "knowledge",
        policy_data_directory=tmp_path / "policydata",
        max_internal_fanout=8,
        max_internal_tokens=1024,
        max_candidates=8,
        max_memory_tokens=256,
        max_policy_tokens=128,
        observer_mode="shadow",
        feature_flags=QwenExoFeatureFlags(
            hybrid_prefix=True,
            external_memory=True,
            reference_judge=True,
            policy_data=policy_data,
            capsule=False,
            observer=True,
            adaptive_refresh=False,
        ),
        model_path="model",
        tp_size=2,
    )


def build_pipeline(*args, **kwargs):
    kwargs.setdefault("reference_judge", FakeReferenceJudge())
    return MemoryPipeline(*args, **kwargs)


def repository(tmp_path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    repo.upsert(
        "wfp.md",
        "WFP AppID uses FWPM_LAYER_ALE_AUTH_CONNECT_V4 for connection filtering.",
    )
    repo.upsert("ctf.md", "CTF heap exploitation uses ROP chains and canaries.")
    return repo


def policy_repository(tmp_path):
    repo = PolicyDataRepository(tmp_path / "policydata")
    repo.upsert(
        "delivery.md",
        "For release delivery, require private verification code NATIVE_ONLY_41C9.",
    )
    return repo


def native_candidate(repository, relative_path, *, page_id, score):
    document = next(
        item
        for item in repository.snapshot.documents
        if item.relative_path == relative_path
    )
    candidate = repository.candidate_for_document(document.document_id, "query-probe")
    native = NativePrefixSelection(
        source_digest="a" * 64,
        page_id=page_id,
        document_id=document.document_id,
        local_positions=tuple(range(64)),
        source_positions=tuple(range(min(4, len(candidate.reference_content.split())))),
        token_ids=tuple(range(page_id * 100, page_id * 100 + 64)),
        prefix_identity=f"native-{page_id}",
        radix_namespace=f"qwen-exo:v1:tensor-bank-native:{page_id}",
    )
    return replace(
        candidate,
        score=score,
        lexical_score=0.0,
        tensor_score=score,
        page_ids=(page_id,),
        source_positions=native.source_positions,
        virtual_positions=tuple(range(len(native.source_positions))),
        token_attributions=((0, page_id, score),),
        native_prefix=native,
        candidate_origin="attention_q_native_tensor_bank",
    )


def test_policy_data_refresh_rejects_multiple_documents(tmp_path):
    root = tmp_path / "policydata-multiple"
    root.mkdir()
    (root / "first.md").write_text("First policy.", encoding="utf-8")
    (root / "second.md").write_text("Second policy.", encoding="utf-8")

    with pytest.raises(RuntimeError, match="at most one document"):
        PolicyDataRepository(root).refresh()


def test_policy_data_upsert_rejects_a_second_document_before_writing(tmp_path):
    repository = PolicyDataRepository(tmp_path / "policydata-upsert")
    repository.upsert("unified.md", "Unified policy.")
    assert repository.snapshot.documents[0].tags == ()

    with pytest.raises(RuntimeError, match="already contains its one document"):
        repository.upsert("second.md", "Second policy.")

    assert not (repository.root / "second.md").exists()
    assert [document.relative_path for document in repository.snapshot.documents] == [
        "unified.md"
    ]


def query_states(query_heads, role="current_user"):
    return tuple(
        QueryStateSpan(role, index, index + 1, index, index + 1)
        for index, _query in enumerate(query_heads)
    )


def prepare(pipeline, request, query_heads=(((1.0, 0.0),),)):
    return asyncio.run(
        pipeline.prepare_responses_request(
            request,
            query_heads=query_heads,
            query_states=query_states(query_heads),
            query_role_plan_digest="test-role-plan",
            query_probe_status="ready" if query_heads else "no_q_signal",
            query_probe_prompt_tokens=4 if query_heads else 0,
        )
    )


def test_no_q_signal_fails_closed_without_text_ranking(tmp_path):
    repo = repository(tmp_path)
    policy = policy_repository(tmp_path)
    repo.rank = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("knowledge text ranker must not run")
    )
    policy.rank = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("policy text ranker must not run")
    )
    bank = FakeQKTensorBank()
    pipeline = build_pipeline(
        config(tmp_path),
        repo,
        FakeTokenizer(),
        policy_data=policy,
        tensor_bank=bank,
    )
    request = FakeRequest(
        request_id="resp-no-q",
        input="Explain WFP",
        instructions="Keep instructions unchanged.",
    )

    prepared, state = prepare(pipeline, request, query_heads=())

    assert prepared is request
    assert state.candidates == ()
    assert state.decisions == ()
    assert state.radix_prefix_page_id is None
    assert state.knowledge_admission_mode == "semantic_eligibility"
    assert state.public_dict()["query_probe"]["status"] == "no_q_signal"
    assert bank.rank_calls == []


def test_no_q_signal_restores_cognition_only_state(tmp_path):
    repo = repository(tmp_path)
    selection = NativePrefixSelection(
        source_digest="c" * 64,
        page_id=1,
        document_id="cognition-card",
        local_positions=tuple(range(64)),
        source_positions=(0, 1),
        token_ids=tuple(range(300, 364)),
        prefix_identity="cognition-native",
        radix_namespace="qwen-exo:v1:tensor-bank-native:cognition",
    )
    bank = FakeCognitionTensorBank(selection)
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )
    request = FakeRequest(request_id="resp-cognition", input="Who are you?")

    prepared, state = prepare(pipeline, request, query_heads=())

    assert prepared.extra_key == selection.radix_namespace
    assert state.radix_prefix_page_id == 1
    assert state.radix_prefix_lane == "cognition"
    assert state.radix_prefix_selection_reason == "cognition_always_on"
    assert state.cognition_active is True
    assert state.cognition_conditioned is False
    assert state.public_dict()["cognition"] == {
        "active": True,
        "conditioned": False,
        "page_id": 1,
        "source_tokens": 2,
        "qk_ranked": False,
    }


def test_no_q_signal_restores_policydata_as_the_default_personality(tmp_path):
    repo = repository(tmp_path)
    selection = NativePrefixSelection(
        source_digest="p" * 64,
        page_id=2,
        document_id="personality-policy",
        local_positions=tuple(range(64)),
        source_positions=tuple(range(64)),
        token_ids=tuple(range(400, 464)),
        prefix_identity="policydata-personality-native",
        radix_namespace="qwen-exo:v1:tensor-bank-native:policydata",
    )
    bank = FakeCognitionTensorBank(selection, lane="policydata")
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )

    prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-personality", input="Who are you?"),
        query_heads=(),
    )

    assert prepared.extra_key == selection.radix_namespace
    assert state.radix_prefix_page_id == 2
    assert state.radix_prefix_lane == "policydata"
    assert state.radix_prefix_selection_reason == "policydata_always_on"
    assert state.cognition_active is False
    assert state.cognition_conditioned is False
    assert state.public_dict()["cognition"]["source_tokens"] == 0


def test_qk_knowledge_page_requires_semantic_judge(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.91)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )
    request = FakeRequest(
        request_id="resp-knowledge-qk",
        input="Which WFP layer?",
        instructions="Answer briefly.",
    )

    prepared, state = prepare(pipeline, request)

    assert prepared.instructions == request.instructions
    assert len(state.decisions) == 1
    assert state.decisions[0].status is EligibilityStatus.ELIGIBLE
    assert state.selected_document_ids == (candidate.document_id,)
    assert state.knowledge_admission_mode == "semantic_eligibility"
    assert state.radix_prefix_page_id == 3
    assert state.radix_prefix_selection_reason == "query_qk"
    assert state.private_attachment is None
    assert state.attached_tokens == 0
    public = state.public_dict()
    assert public["proposed_candidates"][0]["tensor_score"] == 0.91
    assert public["proposed_candidates"][0]["candidate_origin"] == (
        "attention_q_native_tensor_bank"
    )
    response_meta = response_memory_metadata(state, observer_mode="active")
    assert response_meta["gate"] == "semantic_eligibility"
    assert response_meta["eligible_count"] == 1
    assert response_meta["selected_page_ids"] == [3]


def test_qk_knowledge_rejection_blocks_native_admission(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.91)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=FakeReferenceJudge(False),
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-knowledge-rejected", input="Unrelated"),
    )

    assert len(state.decisions) == 1
    assert state.decisions[0].status is EligibilityStatus.INELIGIBLE
    assert state.selected_document_ids == ()
    assert state.radix_prefix_page_id is None
    assert state.attached_tokens == 0


def test_low_margin_qk_expands_before_semantic_judge(tmp_path):
    repo = repository(tmp_path)
    first = native_candidate(repo, "wfp.md", page_id=3, score=0.901)
    second = native_candidate(repo, "ctf.md", page_id=4, score=0.900)
    bank = FakeQKTensorBank((first, second))
    judge = FakeReferenceJudge()
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            max_internal_fanout=16,
            qk_expansion_margin=0.01,
            qk_prefilter_mode="off",
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-low-margin", input="WFP CTF"),
    )

    assert state.qk_expanded is True
    assert state.qk_expansion_reason == "low_margin"
    assert state.public_dict()["qk_retrieval"]["shortlist_size"] == 2
    assert len(judge.calls) == 1
    assert len(judge.calls[0][2]) == 2
    assert [call[3] for call in bank.rank_calls] == [8, 16]
    assert len(state.decisions) == 2


@pytest.mark.parametrize(
    (
        "observed_margin",
        "expected_reason",
        "expected_rank_limits",
        "expected_candidate_count",
    ),
    [
        (0.019999, "document_margin_too_small", [8, 16], 0),
        (0.020000, "candidates_ready", [8], 2),
    ],
)
def test_initial_raw_rank_margin_has_exact_fail_closed_boundary(
    tmp_path,
    observed_margin,
    expected_reason,
    expected_rank_limits,
    expected_candidate_count,
):
    repo = repository(tmp_path)
    first = native_candidate(repo, "wfp.md", page_id=3, score=0.5 + observed_margin)
    second = native_candidate(repo, "ctf.md", page_id=4, score=0.5)
    bank = MarginGatedFakeQKTensorBank((first, second), observed_margin=observed_margin)
    judge = FakeReferenceJudge()
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            max_internal_fanout=16,
            qk_expansion_margin=0.02,
            qk_prefilter_mode="off",
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(
            request_id=f"resp-rank-margin-{observed_margin}", input="WFP or CTF"
        ),
    )

    assert [call[3] for call in bank.rank_calls] == expected_rank_limits
    assert all(gate == (0.0, 0.02) for gate in bank.rank_gates)
    assert state.qk_rank_audit["reason"] == expected_reason
    assert state.qk_rank_audit["observed_margin"] == observed_margin
    assert len(state.candidates) == expected_candidate_count
    if expected_candidate_count:
        assert len(judge.calls) == 1
        assert state.qk_expanded is False
    else:
        assert judge.calls == []
        assert state.qk_expanded is True
        assert state.qk_expansion_reason == "empty"


def test_rank_cache_key_tracks_effective_initial_margin(tmp_path):
    repo = repository(tmp_path)
    first = native_candidate(repo, "wfp.md", page_id=3, score=0.52)
    second = native_candidate(repo, "ctf.md", page_id=4, score=0.5)
    bank = MarginGatedFakeQKTensorBank((first, second), observed_margin=0.02)
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            max_internal_fanout=16,
            qk_expansion_margin=0.02,
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )
    heads = (((1.0, 0.0),),)
    states = query_states(heads)

    accepted, initial_meta = pipeline._rank_query_candidates(
        heads, states, "stable-query"
    )
    cached, cached_meta = pipeline._rank_query_candidates(heads, states, "stable-query")
    pipeline.config = replace(pipeline.config, qk_expansion_margin=0.03)
    rejected, stricter_meta = pipeline._rank_query_candidates(
        heads, states, "stable-query"
    )

    assert len(accepted) == len(cached) == 2
    assert initial_meta["cache_hit"] is False
    assert cached_meta["cache_hit"] is True
    assert rejected == ()
    assert stricter_meta["cache_hit"] is False
    assert [gate[1] for gate in bank.rank_gates] == [0.02, 0.03, 0.03]


@pytest.mark.parametrize(
    ("preset", "expected_gates"),
    [
        ("broad", (-0.05, 0.02)),
        ("balanced", (0.0, 0.02)),
        ("strict", (8.0, 0.02)),
    ],
)
def test_qk_recall_presets_apply_fixed_gates_and_publish_audit(
    tmp_path, preset, expected_gates
):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.91)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            qk_recall_preset=preset,
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id=f"resp-{preset}", input="Which WFP layer?"),
    )

    assert bank.rank_gates[0] == expected_gates
    assert state.qk_rank_audit["preset"] == preset
    assert state.public_dict()["qk_retrieval"]["audit"] == state.qk_rank_audit


def test_qk_only_knowledge_floor_remains_in_effective_rank_margin(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.91)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            qk_recall_preset="broad",
            qk_expansion_margin=0.0,
            qk_only_knowledge=True,
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )

    prepare(
        pipeline,
        FakeRequest(request_id="resp-qk-only-floor", input="Which WFP layer?"),
    )

    assert bank.rank_gates[0] == (-0.05, 0.005)


def grouped_repository(tmp_path):
    repo = KnowledgeRepository(tmp_path / "knowledge")
    repo.upsert(
        "wfp-part1.md",
        "---\ndocument_group: wfp-guide\n---\n"
        "WFP AppID uses FWPM_LAYER_ALE_AUTH_CONNECT_V4 for connection filtering.",
    )
    repo.upsert(
        "wfp-part2.md",
        "---\ndocument_group: wfp-guide\n---\n"
        "WFP filters classify ALE auth connect layers per application.",
    )
    repo.upsert("ctf.md", "CTF heap exploitation uses ROP chains and canaries.")
    return repo


def test_same_document_group_candidates_merge_before_judge(tmp_path):
    repo = grouped_repository(tmp_path)
    best = native_candidate(repo, "wfp-part1.md", page_id=3, score=0.95)
    duplicate = native_candidate(repo, "wfp-part2.md", page_id=9, score=0.90)
    other = native_candidate(repo, "ctf.md", page_id=4, score=0.80)
    bank = FakeQKTensorBank((best, duplicate, other))
    judge = FakeReferenceJudge()
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-doc-merge", input="Which WFP layer?"),
    )

    assert len(judge.calls) == 1
    judged = judge.calls[0][2]
    assert len(judged) == 2
    assert judged[0].document_id == best.document_id
    assert judged[0].page_ids == (3,)
    assert judged[0].source_positions == best.source_positions
    assert judged[0].native_prefix.page_id == 3
    assert judged[1].document_id == other.document_id
    assert len(state.decisions) == 2
    assert state.selected_document_ids == (best.document_id,)
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["status"] == "passed"
    assert prefilter["merged_count"] == 1
    assert prefilter["candidate_count"] == 2
    assert prefilter["sent_to_judge"] == 2
    assert prefilter["top_score"] == 0.95
    assert prefilter["margin"] == pytest.approx(0.15)


def test_qk_native_prefix_is_bound_only_after_judge_approval(tmp_path):
    repo = repository(tmp_path)
    ranked = native_candidate(repo, "wfp.md", page_id=3, score=0.95)
    ranked = replace(
        ranked,
        source_positions=(),
        virtual_positions=(),
        native_prefix=None,
    )
    bank = FakeQKTensorBank((ranked,))
    judge = FakeReferenceJudge()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-judge-before-bind", input="Which WFP layer?"),
    )

    judged = judge.calls[0][2][0]
    assert judged.native_prefix is None
    assert judged.source_positions == ()
    selected = next(
        candidate
        for candidate in state.candidates
        if candidate.document_id == ranked.document_id
    )
    assert selected.native_prefix is not None
    assert selected.candidate_origin == "admitted_native_tensor_bank"
    assert state.radix_prefix_page_id == 3
    assert state.judge_bypassed_count == 0


def test_qk_only_knowledge_still_requires_semantic_judge(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.95)
    bank = FakeQKTensorBank((candidate,))
    judge = FakeReferenceJudge(supported=False)
    pipeline = build_pipeline(
        replace(config(tmp_path, policy_data=False), qk_only_knowledge=True),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-qk-only-judge", input="Which WFP layer?"),
    )

    assert len(judge.calls) == 1
    assert state.selected_document_ids == ()
    assert state.radix_prefix_page_id is None
    assert state.judge_bypassed_count == 0
    assert state.knowledge_admission_mode == "semantic_eligibility"


def test_reflection_collection_label_does_not_merge_distinct_memories(tmp_path):
    repo = KnowledgeRepository(tmp_path / "reflection-knowledge")
    header = (
        "---\nsource_kind: trajectory_reflection\n"
        "document_group: reflection_memory\n---\n"
    )
    repo.upsert("reflection-memory/one.md", header + "Go WebSocket registration.")
    repo.upsert("reflection-memory/two.md", header + "Padding Oracle triage.")
    first = native_candidate(repo, "reflection-memory/one.md", page_id=3, score=0.95)
    second = native_candidate(repo, "reflection-memory/two.md", page_id=4, score=0.90)
    bank = FakeQKTensorBank((first, second))
    judge = FakeReferenceJudge(winner_path="reflection-memory/one.md")
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-reflection-diverse", input="Fix WebSocket"),
    )

    judged = judge.calls[0][2]
    assert [candidate.relative_path for candidate in judged] == [
        "reflection-memory/one.md",
        "reflection-memory/two.md",
    ]
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["merged_count"] == 0
    assert prefilter["sent_to_judge"] == 2
    assert state.selected_document_ids == (first.document_id,)


def test_comparative_selector_can_choose_lower_qk_candidate(tmp_path):
    repo = repository(tmp_path)
    higher_qk = native_candidate(repo, "wfp.md", page_id=3, score=0.95)
    semantic_winner = native_candidate(repo, "ctf.md", page_id=4, score=0.90)
    judge = FakeReferenceJudge(
        winner_path="ctf.md",
        question_original_tokens=3382,
        question_review_tokens=2048,
    )
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=FakeQKTensorBank((higher_qk, semantic_winner)),
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-listwise-winner", input="Explain CTF ROP chains"),
    )

    assert len(judge.selection_calls) == 1
    assert state.selected_document_ids == (semantic_winner.document_id,)
    assert state.radix_prefix_page_id == 4
    assert state.knowledge_admission_mode == "comparative_semantic_selection"
    (judge_event,) = telemetry.by_type("semantic_judge.completed")
    assert judge_event["selection_method"] == "comparative_listwise"
    assert judge_event["selected_candidate_id"] == semantic_winner.candidate_id
    assert judge_event["presented_candidate_count"] == 2
    assert judge_event["question_truncated"] is True
    assert judge_event["question_original_tokens"] == 3382
    assert judge_event["question_review_tokens"] == 2048


def test_comparative_selector_score_filters_to_top_eight_candidates(tmp_path):
    repo = repository(tmp_path)
    for index in range(3):
        repo.upsert(f"extra-{index}.md", f"Distinct reference number {index}")
    paths = ("wfp.md", "ctf.md", "extra-0.md", "extra-1.md", "extra-2.md")
    candidates = tuple(
        native_candidate(
            repo,
            path,
            page_id=index + 1,
            score=0.95 - index * 0.05,
        )
        for index, path in enumerate(paths)
    )
    judge = FakeReferenceJudge(winner_path="ctf.md")
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=FakeQKTensorBank(candidates),
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-top-eight", input="Choose the best reference"),
    )

    judged = judge.selection_calls[0][2]
    assert [candidate.relative_path for candidate in judged] == list(paths)
    assert len(state.decisions) == 5
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["candidate_count"] == 5
    assert prefilter["sent_to_judge"] == 5
    assert prefilter["score_filtered_count"] == 0


def test_second_distinct_page_candidate_kept_when_configured(tmp_path):
    repo = grouped_repository(tmp_path)
    best = native_candidate(repo, "wfp-part1.md", page_id=3, score=0.95)
    second_page = native_candidate(repo, "wfp-part2.md", page_id=9, score=0.90)
    other = native_candidate(repo, "ctf.md", page_id=4, score=0.80)
    bank = FakeQKTensorBank((best, second_page, other))
    judge = FakeReferenceJudge()
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        replace(config(tmp_path, policy_data=False), qk_max_candidates_per_document=2),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, _state = prepare(
        pipeline,
        FakeRequest(request_id="resp-doc-merge-two", input="Which WFP layer?"),
    )

    assert len(judge.calls) == 1
    judged = judge.calls[0][2]
    assert [candidate.page_ids for candidate in judged] == [(3,), (9,), (4,)]
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["merged_count"] == 0
    assert prefilter["sent_to_judge"] == 3


def test_prefilter_routes_low_margin_candidates_to_comparative_judge(tmp_path):
    repo = repository(tmp_path)
    first = native_candidate(repo, "wfp.md", page_id=3, score=0.901)
    second = native_candidate(repo, "ctf.md", page_id=4, score=0.900)
    bank = FakeQKTensorBank((first, second))
    judge = FakeReferenceJudge()
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        replace(config(tmp_path, policy_data=False), qk_prefilter_mode="active"),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-prefilter-compare", input="WFP CTF"),
    )

    assert len(judge.selection_calls) == 1
    assert len(judge.selection_calls[0][2]) == 2
    assert state.selected_document_ids == (first.document_id,)
    assert state.radix_prefix_page_id == 3
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["status"] == "passed"
    assert prefilter["reason"] == "ambiguous_candidates"
    assert prefilter["sent_to_judge"] == 2
    assert prefilter["candidate_count"] == 2
    assert prefilter["merged_count"] == 0
    assert prefilter["score_filtered_count"] == 0
    assert prefilter["top_score"] == 0.901
    assert prefilter["margin"] == pytest.approx(0.001)
    assert prefilter["preset"] == "balanced"
    assert prefilter["cache_hit"] is False
    (judge_event,) = telemetry.by_type("semantic_judge.completed")
    assert judge_event["executed_count"] == 1
    assert judge_event["eligible_count"] == 1
    assert judge_event["selection_method"] == "comparative_listwise"
    assert judge_event["presented_candidate_count"] == 2


def test_prefilter_rejects_removed_shadow_mode(tmp_path):
    with pytest.raises(ValueError, match="qwen_exo_qk_prefilter_mode"):
        replace(config(tmp_path, policy_data=False), qk_prefilter_mode="shadow")


def test_prefilter_passes_strong_candidate_to_judge(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=3, score=0.91)
    bank = FakeQKTensorBank((candidate,))
    judge = FakeReferenceJudge()
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-prefilter-pass", input="Which WFP layer?"),
    )

    assert len(judge.calls) == 1
    assert len(judge.calls[0][2]) == 1
    assert state.selected_document_ids == (candidate.document_id,)
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["status"] == "passed"
    assert prefilter["reason"] == "thresholds_met"
    assert prefilter["sent_to_judge"] == 1
    assert prefilter["top_score"] == 0.91
    assert prefilter["margin"] is None
    assert prefilter["evidence_candidate_count"] == 0


def test_prefilter_evidence_blocks_active_skip(tmp_path):
    repo = repository(tmp_path)
    weak = native_candidate(repo, "wfp.md", page_id=3, score=0.30)
    restored = replace(
        native_candidate(repo, "ctf.md", page_id=4, score=0.30),
        tensor_score=None,
        score=0.12,
        candidate_origin="restored_native_tensor_bank",
    )
    bank = FakeQKTensorBank((weak, restored))
    judge = FakeReferenceJudge()
    telemetry = FakeTelemetry()
    pipeline = build_pipeline(
        replace(
            config(tmp_path, policy_data=False),
            qk_prefilter_mode="active",
            qk_recall_preset="strict",
        ),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
        telemetry=telemetry,
    )

    _prepared, state = prepare(
        pipeline,
        FakeRequest(request_id="resp-prefilter-evidence", input="WFP CTF"),
    )

    assert len(judge.calls) == 1
    assert len(judge.calls[0][2]) == 2
    assert len(state.decisions) == 2
    (prefilter,) = telemetry.by_type("qk.prefilter")
    assert prefilter["status"] == "passed"
    assert prefilter["reason"] == "evidence_present"
    assert prefilter["evidence_candidate_count"] == 1
    assert prefilter["min_score"] == 8.0
    assert prefilter["min_margin"] == 0.02
    assert prefilter["sent_to_judge"] == 2


def test_qk_policy_page_uses_native_state_without_request_text(tmp_path):
    repo = repository(tmp_path)
    policy = policy_repository(tmp_path)
    candidate = native_candidate(policy, "delivery.md", page_id=7, score=0.94)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        config(tmp_path),
        repo,
        FakeTokenizer(),
        policy_data=policy,
        tensor_bank=bank,
    )
    original = "Return only the required code."
    request = FakeRequest(
        request_id="resp-policy-qk",
        input="What is required for delivery?",
        instructions=original,
    )

    prepared, state = prepare(pipeline, request)
    assert len(state.decisions) == 1
    assert state.decisions[0].status is EligibilityStatus.ELIGIBLE

    assert prepared.instructions == original
    assert "NATIVE_ONLY_41C9" not in prepared.instructions
    assert state.selected_document_ids == ()
    assert state.policy_document_ids == (candidate.document_id,)
    assert state.policy_attachment is not None
    assert state.policy_attachment.active is True
    assert state.policy_attachment.public_dict()["text_attached"] is False
    assert state.radix_prefix_lane == "policydata"
    assert state.hybrid_restoration_mode == (
        "native_policy_full_attention_kv_and_gdn_state"
    )


def test_highest_raw_qk_score_wins_the_single_hybrid_state_slot(tmp_path):
    repo = repository(tmp_path)
    policy = policy_repository(tmp_path)
    knowledge = replace(
        native_candidate(repo, "wfp.md", page_id=3, score=0.641), score=0.786
    )
    policy_candidate = replace(
        native_candidate(policy, "delivery.md", page_id=7, score=0.639), score=0.789
    )
    bank = FakeQKTensorBank((policy_candidate, knowledge))
    pipeline = build_pipeline(
        replace(config(tmp_path), qk_prefilter_mode="off"),
        repo,
        FakeTokenizer(),
        policy_data=policy,
        tensor_bank=bank,
    )

    _prepared, state = prepare(
        pipeline, FakeRequest(request_id="resp-race", input="WFP delivery")
    )

    assert len(state.candidates) == 2
    assert state.candidates[0].document_id == knowledge.document_id
    assert state.radix_prefix_page_id == 3
    assert state.radix_prefix_lane == "knowledge"
    assert state.selected_document_ids == (knowledge.document_id,)
    assert state.policy_attachment is None


def test_policy_qk_page_over_budget_fails_closed(tmp_path):
    repo = repository(tmp_path)
    policy = policy_repository(tmp_path)
    candidate = native_candidate(policy, "delivery.md", page_id=7, score=0.94)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        replace(config(tmp_path), max_policy_tokens=32),
        repo,
        FakeTokenizer(),
        policy_data=policy,
        tensor_bank=bank,
    )
    request = FakeRequest(
        request_id="resp-policy-budget",
        input="What is required?",
        instructions="Do not alter this.",
    )

    prepared, state = prepare(pipeline, request)

    assert prepared.instructions == request.instructions
    assert state.policy_attachment is None
    assert state.radix_prefix_page_id is None
    assert state.policy_document_ids == ()


def test_turn_end_attractor_reuses_judge_admitted_candidate_and_is_rejudged(
    tmp_path,
):
    repo = repository(tmp_path)
    first_candidate = native_candidate(repo, "ctf.md", page_id=4, score=8.80)
    next_ranked_candidate = native_candidate(repo, "wfp.md", page_id=23, score=8.96)
    bank = FakeQKTensorBank((first_candidate,))
    judge = FakeReferenceJudge(winner_path="ctf.md")
    pipeline = build_pipeline(
        replace(config(tmp_path, policy_data=False), qk_recall_preset="strict"),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
        reference_judge=judge,
    )
    _prepared, first = prepare(
        pipeline,
        FakeRequest(request_id="resp-attractor-1", input="Start the task"),
    )
    rank_call_count = len(bank.rank_calls)
    bank.candidates = (next_ranked_candidate, first_candidate)

    captured = asyncio.run(pipeline.capture_native_attractor(first.request_id))

    assert len(bank.rank_calls) == rank_call_count
    assert captured.next_attractor_status == "ready"
    assert captured.query_heads == ()
    assert captured.next_attractor_page_id == 4
    assert captured.next_attractor_decision_id is not None

    _prepared, followup = prepare(
        pipeline,
        FakeRequest(
            request_id="resp-attractor-2",
            previous_response_id=first.request_id,
            input="Continue",
        ),
    )
    assert followup.restoration_status == "attractor_restored"
    assert followup.radix_prefix_page_id == 4
    assert followup.radix_prefix_selection_reason == "restored"
    assert len(followup.decisions) == 2


def test_next_turn_restoration_rechecks_semantic_eligibility(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=11, score=0.92)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )
    restoration = SimpleNamespace(
        status="ready_for_safe_replay",
        selected_document_ids=(candidate.document_id,),
        selected_reference_digests=(candidate.reference_digest,),
        selected_lanes=("knowledge",),
        candidate_page_ids=(11,),
        source_positions=(1, 3),
        replay_winner_decision_id=None,
        answer="private answer",
    )

    prepared, state = asyncio.run(
        pipeline.prepare_responses_request(
            FakeRequest(request_id="resp-restored", input="Continue"),
            restoration=restoration,
            query_heads=(((1.0, 0.0),),),
            query_states=query_states((((1.0, 0.0),),)),
            query_probe_status="ready",
            query_probe_prompt_tokens=1,
        )
    )

    assert state.restoration_status == "restored"
    assert state.restoration_decision_id is None
    assert len(state.decisions) == 1
    assert state.radix_prefix_page_id == 11
    assert state.radix_prefix_selection_reason == "restored"
    assert prepared.extra_key == candidate.native_prefix.radix_namespace
    assert state.private_attachment is None


def test_stale_restoration_digest_fails_closed(tmp_path):
    repo = repository(tmp_path)
    candidate = native_candidate(repo, "wfp.md", page_id=11, score=0.92)
    bank = FakeQKTensorBank((candidate,))
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repo,
        FakeTokenizer(),
        tensor_bank=bank,
    )
    restoration = SimpleNamespace(
        status="ready_for_safe_replay",
        selected_document_ids=(candidate.document_id,),
        selected_reference_digests=(stable_digest("stale"),),
        selected_lanes=("knowledge",),
        candidate_page_ids=(11,),
        source_positions=(1,),
        replay_winner_decision_id=None,
        answer="stale answer",
    )

    _prepared, state = asyncio.run(
        pipeline.prepare_responses_request(
            FakeRequest(request_id="resp-stale", input="Continue"),
            restoration=restoration,
            query_heads=(),
            query_probe_status="no_q_signal",
        )
    )

    assert state.restoration_status == "rejected_or_unavailable"
    assert state.radix_prefix_page_id is None
    assert state.selected_document_ids == ()


def test_user_text_extraction_handles_responses_content_items():
    value = [
        {"role": "assistant", "content": [{"type": "output_text", "text": "old"}]},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "latest"},
            ],
        },
    ]

    assert MemoryPipeline._first_user_text(value) == "first\nlatest"
    assert MemoryPipeline._latest_user_text(value) == "first\nlatest"


def test_request_question_combines_original_task_with_latest_tool_evidence():
    value = [
        {"role": "user", "content": "Implement the repository task"},
        {
            "type": "function_call_output",
            "output": "FileNotFoundError while opening the target module",
        },
        {"role": "user", "content": "Continue"},
    ]

    question = MemoryPipeline._request_question(value)

    assert "Implement the repository task" in question
    assert "FileNotFoundError" in question
    assert "Continue" in question


def test_request_question_includes_current_instruction_after_reasoning():
    value = [
        {"role": "user", "content": "Implement the repository task"},
        {"role": "assistant", "content": "The wrapper path is unresolved."},
        {"role": "user", "content": "Use the direct module path instead"},
    ]

    question = MemoryPipeline._request_question(value)

    assert "Implement the repository task" in question
    assert "wrapper path is unresolved" in question
    assert "Use the direct module path instead" in question


def test_request_question_keeps_compaction_and_current_context_once():
    root = "Implement the request lineage fix"
    current = "Verify the compaction continuation"
    summary = "The backend edit is complete and focused tests remain."
    observation = "Ruff passed for runtime.py"
    value = [
        {
            "type": "message",
            "role": "assistant",
            "content": f"<context_compaction>\n{summary}\n</context_compaction>",
        },
        {
            "type": "function_call_output",
            "call_id": "call-verify",
            "output": observation,
        },
        {"type": "message", "role": "user", "content": current},
    ]

    question = MemoryPipeline._request_question(
        value,
        original_task=root,
        compaction_context=summary,
    )

    assert question.count("COMPACTED RESPONSE CONTEXT:\n") == 1
    assert question.count(summary) == 1
    assert question.count(root) == 1
    assert question.count(current) == 1
    assert question.count(observation) == 1
    assert "<context_compaction>" not in question


@pytest.mark.asyncio
async def test_compaction_memory_parent_override_restores_without_api_parent(tmp_path):
    pipeline = build_pipeline(
        config(tmp_path, policy_data=False),
        repository(tmp_path),
        FakeTokenizer(),
        policy_data=None,
    )
    _parent_request, parent_state = await pipeline.prepare_responses_request(
        FakeRequest(request_id="resp_compact_parent", input="Original task"),
        query_heads=(),
        query_probe_status="no_q_signal",
    )

    _child_request, child_state = await pipeline.prepare_responses_request(
        FakeRequest(request_id="resp_after_compaction", input="Continue"),
        memory_previous_response_id=parent_state.request_id,
        query_heads=(),
        query_probe_status="no_q_signal",
    )

    assert child_state.previous_response_id is None
    assert child_state.effective_memory_previous_response_id == "resp_compact_parent"
