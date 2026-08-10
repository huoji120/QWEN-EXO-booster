from dataclasses import replace

from qwen_exo_booster.contracts import EligibilityDecision, EligibilityStatus
from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.recall_trace import recall_trace_payload
from qwen_exo_booster.recall_visualization import render_recall_trace_html


def _decision(candidate, status):
    decision = EligibilityDecision.create(
        candidate_id=candidate.candidate_id,
        parent_request_id="request-1",
        question="How should WFP code be changed and tested?",
        reference=candidate.reference_content,
        status=status,
        judge_method="batch_reference_judge",
        judge_model_fingerprint="model",
        decision_margin=0.0,
    )
    return {
        "decision_id": decision.decision_id,
        "candidate_id": decision.candidate_id,
        "status": decision.status.value,
        "judge_method": decision.judge_method,
        "decision_margin": decision.decision_margin,
    }


def test_recall_visualization_is_chinese_and_clears_directly():
    html = render_recall_trace_html(
        {
            "schema": "inflight-memory-visualization-v2",
            "bank": {},
            "delta": {},
            "turns": [],
        }
    )

    assert '<html lang="zh-CN">' in html
    assert "<title>QWEN EXO / 飞行中召回轨迹</title>" in html
    assert "暂无轨迹。新的请求完成后会自动记录。" in html
    assert "qwen-exo-admin-token" not in html
    assert 'fetch("/qwen-exo/recall-trace",{method:"DELETE"})' in html


def test_recall_trace_preserves_current_policy_and_knowledge_contract(tmp_path):
    knowledge = KnowledgeRepository(tmp_path / "knowledge")
    knowledge.upsert(
        "wfp.md",
        "WFP AppID filtering uses FWPM_LAYER_ALE_AUTH_CONNECT_V4.",
    )
    knowledge.upsert("unrelated.md", "Unrelated package release notes.")
    policy = PolicyDataRepository(tmp_path / "policydata")
    policy.upsert(
        "verification.md",
        "Run observable regression tests before delivering a code change.",
    )

    knowledge_candidate = replace(
        knowledge.rank("WFP AppID filtering")[0],
        page_ids=(7,),
        source_positions=(10, 12),
        virtual_positions=(0, 1),
        token_attributions=((0, 7, 0.9),),
        candidate_origin="attention_q_tensor_bank",
    )
    rejected_candidate = knowledge.rank("package release notes")[0]
    policy_candidate = policy.rank("code regression tests", limit=8)[0]
    memory_payload = {
        "previous_response_id": None,
        "retrieval_latency_seconds": 0.02,
        "judge_latency_seconds": 0.03,
        "knowledge_admission_mode": "native_qk",
        "policy_data": {
            "source_digest": policy.snapshot.source_digest,
            "document_ids": [policy_candidate.document_id],
            "attached_tokens": 9,
            "active": True,
        },
        "proposed_candidates": [
            policy_candidate.public_dict(),
            knowledge_candidate.public_dict(),
            rejected_candidate.public_dict(),
        ],
        "semantic_decisions": [],
        "selected_document_ids": [knowledge_candidate.document_id],
        "next_turn_restoration": {
            "status": "restored",
            "document_ids": [knowledge_candidate.document_id],
            "page_ids": [7],
            "source_positions": [10, 12],
        },
    }
    events = [
        {
            "event_id": 1,
            "timestamp": 10.0,
            "request_id": "request-1",
            "event_type": "request.started",
            "payload": {"input": "How should WFP code be changed and tested?"},
        },
        {
            "event_id": 2,
            "timestamp": 10.1,
            "request_id": "request-1",
            "event_type": "memory.prepared",
            "payload": memory_payload,
        },
        {
            "event_id": 3,
            "timestamp": 10.2,
            "request_id": "request-1",
            "event_type": "observer.decode_summary",
            "payload": {
                "new_tokens": 12,
                "token_count": 12,
                "max_surprisal": 4.2,
                "ema_surprisal": 2.1,
                "triggered": False,
            },
        },
        {
            "event_id": 4,
            "timestamp": 10.3,
            "request_id": "request-1",
            "event_type": "causal_replay.completed",
            "payload": {
                "replay_decision": "shadow_would_switch",
                "maybe_gate_decision": "admit_maybe",
                "winner_candidate_id": knowledge_candidate.candidate_id,
                "winner_gain": 0.4,
                "winner_kl": 0.2,
                "scheduled_next_turn": True,
                "losses": [{"observation_tokens": 8}],
                "latency_seconds": 0.04,
            },
        },
        {
            "event_id": 5,
            "timestamp": 10.31,
            "request_id": "request-1",
            "event_type": "maybe.completed",
            "payload": {
                "status": "ready_for_safe_replay",
                "selected_document_ids": [knowledge_candidate.document_id],
                "candidate_page_ids": [7],
                "maybe_scheduled_next_turn": True,
                "maybe_decision": "admit_maybe",
            },
        },
        {
            "event_id": 6,
            "timestamp": 10.32,
            "request_id": "request-1",
            "event_type": "adaptive.transition",
            "payload": {"from": "replay_scoring", "to": "next_turn_ready"},
        },
        {
            "event_id": 7,
            "timestamp": 10.33,
            "request_id": "request-1",
            "event_type": "post_tool_recall.completed",
            "payload": {"status": "ready_for_safe_replay", "admitted": True},
        },
        {
            "event_id": 8,
            "timestamp": 10.34,
            "request_id": "request-1",
            "event_type": "request.stage_summary",
            "payload": {"schema": "qwen-exo-stage-summary-v1"},
        },
        {
            "event_id": 4,
            "timestamp": 10.4,
            "request_id": "request-1",
            "event_type": "request.completed",
            "payload": {"output": "Use the ALE AUTH CONNECT layer."},
        },
    ]

    payload = recall_trace_payload(
        policy_snapshot=policy.snapshot,
        knowledge_snapshot=knowledge.snapshot,
        events=events,
    )

    assert payload["schema"] == "inflight-memory-visualization-v2"
    assert payload["bank"]["semantics"] == (
        "request_qk_native+adaptive_semantic_admission"
    )
    assert payload["delta"]["policy_data"] == {
        "always_on": True,
        "requires_qk_relevance": False,
        "independent_recall_lane": False,
        "personality_prefix": True,
    }
    assert payload["delta"]["knowledge"] == {
        "request_admission": "native_qk",
        "adaptive_requires_semantic_eligibility": True,
    }
    assert len(payload["bank"]["policy_documents"]) == 1
    assert len(payload["bank"]["documents"]) == 2

    turn = payload["turns"][0]
    assert turn["policy_data_active"] is True
    assert turn["policy_data_document_ids"] == [policy_candidate.document_id]
    assert turn["selected_document_ids"] == [knowledge_candidate.document_id]
    selected_by_lane = {
        item["lane"]: item["document_id"] for item in turn["selected_support_chunks"]
    }
    assert selected_by_lane == {
        "policydata": policy_candidate.document_id,
        "knowledge": knowledge_candidate.document_id,
    }
    candidate_by_id = {
        item["candidate_id"]: item for item in turn["knowledge_candidates"]
    }
    assert candidate_by_id[policy_candidate.candidate_id]["policy"] is True
    assert candidate_by_id[knowledge_candidate.candidate_id]["policy"] is False
    assert (
        candidate_by_id[rejected_candidate.candidate_id]["semantic_support"][
            "supported"
        ]
        is False
    )
    assert turn["mid_think_monitoring"] is True
    assert turn["think_recall_events"][0]["latest_surprisal"] == 4.2

    assert turn["selected_page_ids"] == [7]
    assert turn["pending_maybe_restored"] is True
    assert turn["think_recall_events"][0]["winner_gain"] == 0.4
    assert turn["think_recall_events"][0]["maybe_kl"] == 0.2
    assert turn["post_tool_recall_active"] is True
    assert turn["next_turn_restoration"]["status"] == "restored"
    assert turn["adaptive_retrieval_transitions"][0]["to"] == "next_turn_ready"
    attribution = turn["token_memory_attribution"]
    assert attribution["method"] == "tp_synchronized_q_to_fp8_page_k"
    assert attribution["tokens"][0]["page_id"] == 7
    assert turn["stage_summary"]["schema"] == "qwen-exo-stage-summary-v1"


def test_recall_trace_distinguishes_policy_reflection_from_factual_answer(tmp_path):
    knowledge = KnowledgeRepository(tmp_path / "knowledge")
    policy = PolicyDataRepository(tmp_path / "policydata")
    policy.upsert(
        "reflection.md",
        "Compare unresolved assumptions with direct behavioral evidence.",
    )
    candidate = policy.rank("behavioral evidence", limit=1)[0]
    events = [
        {
            "event_id": 1,
            "timestamp": 1.0,
            "request_id": "request-reflection",
            "event_type": "request.started",
            "payload": {"input": "Change and verify repository behavior"},
        },
        {
            "event_id": 2,
            "timestamp": 1.1,
            "request_id": "request-reflection",
            "event_type": "memory.prepared",
            "payload": {
                "policy_data": {
                    "source_digest": policy.snapshot.source_digest,
                    "document_ids": [candidate.document_id],
                    "attached_tokens": 8,
                    "active": True,
                },
                "proposed_candidates": [candidate.public_dict()],
                "semantic_decisions": [
                    _decision(candidate, EligibilityStatus.ELIGIBLE)
                ],
            },
        },
        {
            "event_id": 3,
            "timestamp": 1.2,
            "request_id": "request-reflection",
            "event_type": "observer.decode_summary",
            "payload": {
                "token_count": 16,
                "new_tokens": 16,
                "triggered": True,
                "trigger_reasons": ["attention_q_drift"],
            },
        },
        {
            "event_id": 4,
            "timestamp": 1.3,
            "request_id": "request-reflection",
            "event_type": "refresh.started",
            "payload": {},
        },
        {
            "event_id": 5,
            "timestamp": 1.4,
            "request_id": "request-reflection",
            "event_type": "refresh.completed",
            "payload": {
                "status": "policy_reflection_ready",
                "reflection_kind": "policy_critique",
                "selected_document_ids": [candidate.document_id],
                "maybe_scheduled_next_turn": True,
                "maybe_decision": "admit_policy_reflection",
            },
        },
        {
            "event_id": 6,
            "timestamp": 1.5,
            "request_id": "request-reflection",
            "event_type": "request.completed",
            "payload": {"output": "continued"},
        },
    ]

    payload = recall_trace_payload(
        policy_snapshot=policy.snapshot,
        knowledge_snapshot=knowledge.snapshot,
        events=events,
    )
    turn = payload["turns"][0]
    recall = turn["think_recall_events"][0]

    assert turn["policy_reflection_active"] is True
    assert recall["evidence_answer"]["decision"] == "not_generated"
    assert recall["reasoning_reflection"] == {
        "decision": "generated",
        "kind": "policy_critique",
    }
