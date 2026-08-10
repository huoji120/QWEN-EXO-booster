import json

from qwen_exo_booster.knowledge import KnowledgeRepository
from qwen_exo_booster.policy_data import PolicyDataRepository
from qwen_exo_booster.recall_trace import recall_trace_payload


def test_observer_snapshots_do_not_repeat_large_candidate_arrays(tmp_path):
    knowledge = KnowledgeRepository(tmp_path / "knowledge")
    knowledge.upsert(
        "identity.md",
        "A benign operational identity label for a local assistant.",
    )
    policy = PolicyDataRepository(tmp_path / "policydata")
    candidate = knowledge.rank("operational identity label")[0].public_dict()
    candidate.update(
        {
            "page_ids": [7],
            "source_positions": list(range(2048)),
            "virtual_positions": list(range(2048)),
            "token_attributions": [
                {"query_token_offset": index, "page_id": 7, "score": 0.5}
                for index in range(64)
            ],
            "candidate_origin": "attention_q_native_tensor_bank",
        }
    )
    events = [
        {
            "event_id": 1,
            "timestamp": 1.0,
            "request_id": "identity-request",
            "event_type": "request.started",
            "payload": {"input": "What is the operational identity label?"},
        },
        {
            "event_id": 2,
            "timestamp": 1.1,
            "request_id": "identity-request",
            "event_type": "memory.prepared",
            "payload": {
                "knowledge_admission_mode": "native_qk",
                "selected_document_ids": [candidate["document_id"]],
                "proposed_candidates": [candidate],
                "semantic_decisions": [],
            },
        },
    ]
    events.extend(
        {
            "event_id": index + 3,
            "timestamp": 1.1 + index / 1000,
            "request_id": "identity-request",
            "event_type": "observer.decode_summary",
            "payload": {
                "token_count": index + 1,
                "new_tokens": 1,
                "max_surprisal": 0.1,
                "ema_surprisal": 0.1,
                "triggered": False,
            },
        }
        for index in range(128)
    )

    payload = recall_trace_payload(
        policy_snapshot=policy.snapshot,
        knowledge_snapshot=knowledge.snapshot,
        events=events,
    )
    turn = payload["turns"][0]
    observations = turn["think_recall_events"]

    assert len(observations) == 128
    assert turn["knowledge_candidates"][0]["source_positions"] == list(range(2048))
    assert turn["knowledge_candidates"][0]["token_attributions"]
    assert len(json.dumps(observations, ensure_ascii=False)) < 1_000_000
    for observation in observations:
        candidate_row = observation["candidates"][0]
        support_row = observation["semantic_support"][0]
        assert candidate_row["source_positions"] == []
        assert candidate_row["source_positions_count"] == 2048
        assert candidate_row["virtual_positions"] == []
        assert candidate_row["token_attributions"] == []
        assert candidate_row["token_attribution_count"] == 64
        assert support_row["source_positions"] == []
        assert support_row["source_positions_count"] == 2048
