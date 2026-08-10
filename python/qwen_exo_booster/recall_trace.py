from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Iterable

from qwen_exo_booster.knowledge import (
    KnowledgeDocument,
    KnowledgeSnapshot,
    lexical_terms,
)
from qwen_exo_booster.telemetry import TraceEvent

RECALL_TRACE_SCHEMA = "inflight-memory-visualization-v2"


def anonymous_document_id(document: KnowledgeDocument) -> str:
    value = document.sha256 or document.relative_path or document.document_id
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def _document_summary(document: KnowledgeDocument, *, policy: bool) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "anonymous_id": anonymous_document_id(document),
        "sha256": document.sha256,
        "tokens": len(lexical_terms(document.normalized_content)),
        "pages": 1,
        "relative_path": document.relative_path,
        "canonical": document.canonical,
        "quality": document.quality,
        "source_kind": document.source_kind,
        "policy": policy,
    }


def _event_dict(event: TraceEvent | dict[str, Any]) -> dict[str, Any]:
    return event.to_dict() if isinstance(event, TraceEvent) else dict(event)


def _snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    text = str(value)
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def _candidate_rows(
    memory: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = {
        str(item.get("candidate_id")): item
        for item in memory.get("semantic_decisions") or ()
    }
    admission_mode = str(
        memory.get("knowledge_admission_mode") or "semantic_eligibility"
    )
    selected_document_ids = set(memory.get("selected_document_ids") or ())
    selected_policy_document_ids = set(
        (memory.get("policy_data") or {}).get("document_ids") or ()
    )
    supports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for candidate in memory.get("proposed_candidates") or ():
        candidate_id = str(candidate.get("candidate_id") or "")
        lane = str(
            candidate.get("lane")
            or ("policydata" if candidate.get("policy") else "knowledge")
        )
        decision = decisions.get(candidate_id)
        status = str((decision or {}).get("status") or "")
        if admission_mode == "native_qk":
            selected_ids = (
                selected_policy_document_ids
                if lane == "policydata"
                else selected_document_ids
            )
            supported = candidate.get("document_id") in selected_ids
        else:
            supported = (
                True if status == "true" else False if status == "false" else None
            )
        support = {
            "section_id": candidate_id,
            "candidate_id": candidate_id,
            "document": candidate.get("relative_path"),
            "document_id": candidate.get("document_id"),
            "lane": lane,
            "supported": supported,
            "judge_completed": bool(decision),
            "judge_tokens": None,
            "decision_id": (decision or {}).get("decision_id"),
            "judge_method": (decision or {}).get("judge_method"),
            "decision_margin": (decision or {}).get("decision_margin"),
            "admission_method": admission_mode if lane == "knowledge" else None,
            "page_ids": list(candidate.get("page_ids") or ()),
            "source_positions": list(candidate.get("source_positions") or ()),
            "virtual_positions": list(candidate.get("virtual_positions") or ()),
        }
        supports.append(support)
        candidates.append(
            {
                "section_id": candidate_id,
                "candidate_id": candidate_id,
                "document": candidate.get("relative_path"),
                "document_id": candidate.get("document_id"),
                "score": candidate.get("score"),
                "canonical": bool(candidate.get("canonical", False)),
                "quality": candidate.get("quality_prior"),
                "policy": lane == "policydata",
                "lane": lane,
                "page_ids": list(candidate.get("page_ids") or ()),
                "source_positions": list(candidate.get("source_positions") or ()),
                "virtual_positions": list(candidate.get("virtual_positions") or ()),
                "token_attributions": list(candidate.get("token_attributions") or ()),
                "origin": candidate.get("candidate_origin") or f"{lane}_lexical_rank",
                "matched_question": None,
                "semantic_support": support,
            }
        )
    return candidates, supports


def _compact_event_candidates(
    candidates: list[dict[str, Any]],
    semantic_support: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep per-observation snapshots small; full rows live on the turn."""
    compact_support: list[dict[str, Any]] = []
    for support in semantic_support:
        row = dict(support)
        for key in ("source_positions", "virtual_positions", "page_ids"):
            value = row.get(key)
            row[f"{key}_count"] = len(value) if isinstance(value, list) else 0
            row[key] = []
        compact_support.append(row)
    support_by_id = {str(row.get("candidate_id")): row for row in compact_support}
    compact_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        for key in ("source_positions", "virtual_positions", "page_ids"):
            value = row.get(key)
            row[f"{key}_count"] = len(value) if isinstance(value, list) else 0
            row[key] = []
        attributions = row.get("token_attributions")
        row["token_attribution_count"] = (
            len(attributions) if isinstance(attributions, list) else 0
        )
        row["token_attributions"] = []
        row["semantic_support"] = support_by_id.get(str(row.get("candidate_id")), {})
        compact_candidates.append(row)
    return compact_candidates, compact_support


def _refresh_timing(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, float]:
    started = next(
        (event for event in events if event.get("event_type") == "refresh.started"),
        None,
    )
    completed = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "refresh.completed"
        ),
        None,
    )
    latency = 0.0
    if started is not None and completed is not None:
        latency = max(
            0.0,
            float(completed.get("timestamp") or 0)
            - float(started.get("timestamp") or 0),
        )
    return completed, latency


def _turn_trace(request_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    memory_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "memory.prepared"
        ),
        None,
    )
    memory = dict((memory_event or {}).get("payload") or {})
    tensor_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "tensor.candidates_proposed"
        ),
        None,
    )
    semantic_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "semantic_judge.completed"
        ),
        None,
    )
    if tensor_event is not None:
        combined = {
            str(candidate.get("candidate_id")): candidate
            for candidate in memory.get("proposed_candidates") or ()
        }
        for candidate in (tensor_event.get("payload") or {}).get("candidates") or ():
            combined[str(candidate.get("candidate_id"))] = candidate
        memory["proposed_candidates"] = list(combined.values())
    if semantic_event is not None:
        decisions = {
            str(decision.get("candidate_id")): decision
            for decision in memory.get("semantic_decisions") or ()
        }
        for decision in (semantic_event.get("payload") or {}).get("decisions") or ():
            decisions[str(decision.get("candidate_id"))] = decision
        memory["semantic_decisions"] = list(decisions.values())
    candidates, semantic_support = _candidate_rows(memory)
    event_candidates, event_semantic_support = _compact_event_candidates(
        candidates, semantic_support
    )
    refresh_event, refresh_latency = _refresh_timing(events)
    maybe_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "maybe.completed"
        ),
        None,
    )
    replay_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "causal_replay.completed"
        ),
        None,
    )
    final_refresh_event = maybe_event or refresh_event
    refresh = dict((final_refresh_event or {}).get("payload") or {})
    replay = dict((replay_event or {}).get("payload") or {})
    refresh_status = str(refresh.get("status") or "")
    reflection_kind = str(refresh.get("reflection_kind") or "none")
    admitted = bool(
        refresh.get("maybe_scheduled_next_turn")
        or replay.get("scheduled_next_turn")
        or refresh_status == "ready_for_safe_replay"
    )
    selected_document_ids = list(
        refresh.get("selected_document_ids")
        or memory.get("selected_document_ids")
        or ()
    )
    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.get("document_id") in selected_document_ids
        ),
        None,
    )
    observer_events = [
        event
        for event in events
        if event.get("event_type") == "observer.decode_summary"
    ]
    selected_source_positions = list((selected or {}).get("source_positions") or ())
    recall_events = []
    for event in observer_events:
        observation = dict(event.get("payload") or {})
        triggered = bool(observation.get("triggered"))
        question = refresh.get("question")
        recall_events.append(
            {
                "token_index": observation.get("token_count"),
                "layer": None,
                "latest_surprisal": observation.get("max_surprisal"),
                "window_mean": observation.get("ema_surprisal"),
                "history_mean": observation.get("ema_surprisal"),
                "uncertainty_state": (
                    "uncertainty_detected" if triggered else "stable"
                ),
                "recovery_observation_tokens": 0,
                "recovery_window_mean": observation.get("ema_surprisal"),
                "bank_challenger": bool(candidates),
                "bank_challenger_margin": None,
                "self_ask": question if isinstance(question, str) else None,
                "self_ask_questions": (
                    [question] if isinstance(question, str) and question else []
                ),
                "self_ask_raw_text": question,
                "self_ask_complete": bool(refresh_event),
                "self_ask_decision": refresh_status or "not_scheduled",
                "self_question_committed": False,
                "self_question_commit_text": None,
                "self_question_commit_tokens": None,
                "self_question_commit_decision": "scheduler_native_sibling_job",
                "self_question_continuation_required": False,
                "self_question_continuation_tokens": 0,
                "knowledge_reference": None,
                "self_ask_question_evidence": [],
                "self_ask_tokens": None,
                "self_ask_layer": None,
                "self_ask_visible": False,
                "document_retrieval": {
                    "decision": "selected" if selected_document_ids else "not_selected",
                    "selection_source": (selected or {}).get("origin")
                    or "semantic_rejudge",
                },
                "semantic_support": event_semantic_support,
                "evidence_answer": {
                    "decision": (
                        "generated"
                        if admitted and reflection_kind == "none"
                        else "not_generated"
                    ),
                    "tokens": None,
                },
                "reasoning_reflection": {
                    "decision": (
                        "generated"
                        if admitted and reflection_kind != "none"
                        else "not_generated"
                    ),
                    "kind": reflection_kind,
                },
                "tensor_replay_decision": replay.get("replay_decision")
                or refresh_status
                or "not_scheduled",
                "tensor_recall": {
                    "state": (
                        "replay_admitted"
                        if admitted
                        else "replay_rejected" if refresh_event else "observer_shadow"
                    ),
                    "trigger_reasons": observation.get("trigger_reasons") or [],
                },
                "candidates": event_candidates,
                "replay_prefix_start": None,
                "replay_prefix_end": None,
                "observation_tokens": (
                    (replay.get("losses") or [{}])[0].get("observation_tokens")
                    if replay.get("losses")
                    else observation.get("new_tokens")
                ),
                "baseline_section_id": None,
                "winner_section_id": replay.get("winner_candidate_id")
                or (selected or {}).get("candidate_id"),
                "winner_gain": replay.get("winner_gain"),
                "replay_decision": replay.get("replay_decision")
                or refresh_status
                or "not_scheduled",
                "replay_candidate_source": "scheduler_native_reference_judge",
                "replay_losses": list(replay.get("losses") or ()),
                "maybe_nll": None,
                "maybe_gain": replay.get("winner_gain"),
                "maybe_kl": replay.get("winner_kl"),
                "maybe_kl_cap": None,
                "maybe_decision": replay.get("maybe_gate_decision")
                or refresh.get("maybe_decision")
                or "not_compiled",
                "maybe_scheduled": admitted,
                "selected_document": (selected or {}).get("document"),
                "selected_section_id": (selected or {}).get("candidate_id"),
                "recalled_source_text": None,
                "recalled_source_ranges": [],
                "recalled_source_range_count": len(selected_source_positions),
                "recalled_text": None,
                "external_learning": None,
                "observer": observation,
            }
        )

    started_event = next(
        (event for event in events if event.get("event_type") == "request.started"),
        None,
    )
    completed_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "request.completed"
        ),
        None,
    )
    capsule_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") in {"capsule.updated", "capsule.failed_closed"}
        ),
        None,
    )
    first_timestamp = float(events[0].get("timestamp") or 0) if events else 0.0
    last_timestamp = (
        float(events[-1].get("timestamp") or 0) if events else first_timestamp
    )
    post_tool_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "post_tool_recall.completed"
        ),
        None,
    )
    stage_summary_event = next(
        (
            event
            for event in reversed(events)
            if event.get("event_type") == "request.stage_summary"
        ),
        None,
    )
    adaptive_transitions = [
        dict(event.get("payload") or {})
        for event in events
        if event.get("event_type") == "adaptive.transition"
    ]
    restoration = dict(memory.get("next_turn_restoration") or {})
    selected_page_ids = list(
        dict.fromkeys(
            [*(refresh.get("candidate_page_ids") or ())]
            + [
                page_id
                for candidate in candidates
                if candidate.get("document_id") in selected_document_ids
                for page_id in candidate.get("page_ids") or ()
            ]
        )
    )
    token_attributions = [
        {
            **attribution,
            "candidate_id": candidate.get("candidate_id"),
            "document_id": candidate.get("document_id"),
            "lane": candidate.get("lane"),
        }
        for candidate in candidates
        for attribution in candidate.get("token_attributions") or ()
    ]
    policy_data = dict(memory.get("policy_data") or {})
    policy_document_ids = list(policy_data.get("document_ids") or ())
    selected_source_ids = set(selected_document_ids) | set(policy_document_ids)
    selected_chunks = [
        {
            "file": candidate.get("document"),
            "document_id": candidate.get("document_id"),
            "candidate_id": candidate.get("candidate_id"),
            "score": candidate.get("score"),
            "policy": bool(candidate.get("policy")),
            "lane": str(candidate.get("lane") or "knowledge"),
            "page_ids": list(candidate.get("page_ids") or ()),
            "source_positions": list(candidate.get("source_positions") or ()),
            "virtual_positions": list(candidate.get("virtual_positions") or ()),
        }
        for candidate in candidates
        if candidate.get("document_id") in selected_source_ids
    ]
    return {
        "time_unix_ns": int(first_timestamp * 1_000_000_000),
        "trajectory_id": (
            ((capsule_event or {}).get("payload") or {}).get("trajectory_id")
            or request_id
        ),
        "turn_id": request_id,
        "response_id": request_id,
        "parent_response_id": memory.get("previous_response_id"),
        "memory_condition": "on",
        "strategy": "request_qk_native+adaptive_reference_judge",
        "gate": str(memory.get("knowledge_admission_mode") or "native_qk"),
        "top_k": len(candidates),
        "knowledge_admitted": bool(selected_document_ids),
        "selected_document_id": (
            selected_document_ids[0] if selected_document_ids else None
        ),
        "selected_document_ids": selected_document_ids,
        "selected_document_anonymous_ids": selected_document_ids,
        "selected_document_anonymous_id": (
            selected_document_ids[0] if selected_document_ids else None
        ),
        "selected_page_ids": selected_page_ids,
        "document_probability": None,
        "no_memory_probability": None,
        "document_agreeing_layers": None,
        "document_contributing_layers": None,
        "policy_delta_active": bool(policy_data.get("active")),
        "policy_data_active": bool(policy_data.get("active")),
        "policy_data_source_digest": policy_data.get("source_digest"),
        "policy_data_document_ids": policy_document_ids,
        "policy_data_tokens": policy_data.get("attached_tokens", 0),
        "policy_reflection_active": reflection_kind != "none",
        "collection_exact_probe": None,
        "generation_delta_source": "native_sglang_hybrid_state",
        "timing": {
            "total_seconds": max(0.0, last_timestamp - first_timestamp),
            "retrieval_seconds": memory.get("retrieval_latency_seconds", 0),
            "reference_judge_seconds": memory.get("judge_latency_seconds", 0),
            "reference_judge_cache_hits": memory.get("judge_cache_hit_count", 0),
            "reference_judge_executed": memory.get("judge_executed_count", 0),
        },
        "think_recall_stage_timing": {
            "self_ask_seconds": refresh_latency,
            "exact_replay_seconds": float(replay.get("latency_seconds") or 0.0),
        },
        "think_recall_events": recall_events,
        "prefill_chunks": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "lane": candidate.get("lane"),
                "page_ids": candidate.get("page_ids"),
                "semantically_admitted": bool(
                    (candidate.get("semantic_support") or {}).get("supported")
                ),
            }
            for candidate in candidates
        ],
        "selected_support_chunks": selected_chunks,
        "knowledge_candidates": candidates,
        "semantic_support": semantic_support,
        "pending_maybe_restored": admitted,
        "pending_maybe_compilation": refresh or None,
        "post_tool_recall_active": post_tool_event is not None,
        "post_tool_recall": dict((post_tool_event or {}).get("payload") or {}) or None,
        "post_tool_recall_compilation": dict(
            (post_tool_event or {}).get("payload") or {}
        )
        or None,
        "post_tool_duplicate_suppressed": False,
        "post_tool_observation_resolved": bool(post_tool_event),
        "mid_think_monitoring": bool(observer_events),
        "mid_think_monitor_only": not any(
            bool((event.get("payload") or {}).get("triggered"))
            for event in observer_events
        ),
        "external_learning_restored": False,
        "external_learning_questions": [],
        "external_learning_answer_preview": None,
        "prompt": _snapshot((started_event or {}).get("payload", {}).get("input")),
        "answer": _snapshot((completed_event or {}).get("payload", {}).get("output")),
        "reasoning": _snapshot(
            (completed_event or {}).get("payload", {}).get("reasoning")
        ),
        "output_types": [],
        "tool_calls": [],
        "layers": [],
        "decode_events": [
            dict(event.get("payload") or {}) for event in observer_events
        ],
        "adaptive_retrieval_transitions": adaptive_transitions,
        "stage_summary": dict((stage_summary_event or {}).get("payload") or {}) or None,
        "next_turn_restoration": restoration or None,
        "runtime_hypothesis_active": False,
        "memory_collision": None,
        "runtime_hypothesis_slots": [],
        "runtime_hypothesis_delta": {
            "composed": False,
            "base_delta_preserved": True,
            "layers": [],
        },
        "token_memory_attribution": {
            "method": "tp_synchronized_q_to_fp8_page_k",
            "causal_claim": False,
            "tracked_layers": ["last_full_attention"],
            "tokens": token_attributions,
        },
        "execution_capsule": dict((capsule_event or {}).get("payload") or {}),
    }


def recall_trace_payload(
    *,
    policy_snapshot: KnowledgeSnapshot,
    knowledge_snapshot: KnowledgeSnapshot,
    events: Iterable[TraceEvent | dict[str, Any]],
    max_turns: int = 100,
) -> dict[str, Any]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for raw_event in events:
        event = _event_dict(raw_event)
        request_id = str(event.get("request_id") or "")
        if not request_id or request_id in {"runtime", "admin"} or ":" in request_id:
            continue
        grouped.setdefault(request_id, []).append(event)
    selected_groups = list(grouped.items())[-max(0, int(max_turns)) :]
    turns = [
        _turn_trace(request_id, request_events)
        for request_id, request_events in selected_groups
    ]
    policy_documents = [
        _document_summary(document, policy=True)
        for document in policy_snapshot.documents
    ]
    knowledge_documents = [
        _document_summary(document, policy=False)
        for document in knowledge_snapshot.documents
    ]
    policy_tokens = max(
        (int(turn.get("policy_data_tokens") or 0) for turn in turns),
        default=0,
    )
    return {
        "schema": RECALL_TRACE_SCHEMA,
        "bank": {
            "schema": "qwen-exo-policy-knowledge-v1",
            "semantics": "request_qk_native+adaptive_semantic_admission",
            "source_tokens": None,
            "policy_tokens": policy_tokens,
            "policy_source_digest": policy_snapshot.source_digest,
            "knowledge_source_digest": knowledge_snapshot.source_digest,
            "policy_documents": policy_documents,
            "documents": knowledge_documents,
        },
        "delta": {
            "policy_data": {
                "always_on": True,
                "requires_qk_relevance": False,
                "independent_recall_lane": False,
                "personality_prefix": True,
            },
            "knowledge": {
                "request_admission": "native_qk",
                "adaptive_requires_semantic_eligibility": True,
            },
            "native_hybrid_state": True,
        },
        "turns": turns,
    }
