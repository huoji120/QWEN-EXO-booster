#!/usr/bin/env python3
"""Compare compressed and model-native Q/K recall on persisted trajectories."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from qwen_exo_booster.knowledge import normalize_markdown
from qwen_exo_booster.native_state_bank import _dequantize_fp8, _load_page_payload
from qwen_exo_booster.tensor_bank import _SINK_TOKEN_TEXT

_CLEAR_SEPARATION_MARGIN = 0.02


@dataclass(frozen=True, slots=True)
class RecallCase:
    case_id: str
    request_id: str
    query: str
    gold_path: str | None
    query_source: str


@dataclass(frozen=True, slots=True)
class PageKeys:
    page_id: int
    relative_path: str
    cognition_tokens: int
    source_tokens: int
    searchable_positions: torch.Tensor
    keys: torch.Tensor


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reflection_records(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, dict):
        rows = (
            value.get("records") or value.get("entries") or value.get("memories") or ()
        )
    else:
        rows = ()
    return tuple(row for row in rows if isinstance(row, dict))


def request_inputs(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                not isinstance(event, dict)
                or event.get("event_type") != "request.started"
            ):
                continue
            request_id = str(event.get("request_id") or "")
            payload = event.get("payload") or {}
            query = str(payload.get("input") or "").strip()
            if request_id and query:
                result[request_id] = query
    return result


def build_cases(
    reflection_path: Path,
    trace_path: Path,
    bank_paths: frozenset[str],
    explicit: tuple[str, ...],
) -> tuple[RecallCase, ...]:
    inputs = request_inputs(trace_path)
    cases: list[RecallCase] = []
    seen: set[tuple[str, str]] = set()
    for record in reflection_records(load_json(reflection_path)):
        document_path = str(
            record.get("document_path") or record.get("target_document_path") or ""
        )
        request_id = str(record.get("trajectory_id") or "")
        query = inputs.get(request_id, "")
        query_source = "request.started"
        if not query:
            query = "\n".join(
                str(record.get(field) or "").strip()
                for field in ("title", "evidence")
                if str(record.get(field) or "").strip()
            )
            if not query:
                query = str(record.get("reflection") or "").strip()
            query_source = "reflection.title_evidence"
        identity = (request_id, document_path)
        if (
            document_path not in bank_paths
            or not request_id
            or not query
            or identity in seen
        ):
            continue
        seen.add(identity)
        cases.append(
            RecallCase(
                case_id=str(record.get("title") or request_id),
                request_id=request_id,
                query=query,
                gold_path=document_path,
                query_source=query_source,
            )
        )
    for raw in explicit:
        request_id, separator, document_path = raw.partition("=")
        query = inputs.get(request_id, "")
        identity = (request_id, document_path)
        if (
            not separator
            or document_path not in bank_paths
            or not query
            or identity in seen
        ):
            raise ValueError(
                "Explicit cases require REQUEST_ID=GOLD_PATH present in trace and Bank"
            )
        seen.add(identity)
        cases.append(
            RecallCase(
                case_id=f"explicit:{request_id}",
                request_id=request_id,
                query=query,
                gold_path=document_path,
                query_source="request.started",
            )
        )
    return tuple(cases)


def fixture_cases(path: Path, bank_paths: frozenset[str]) -> tuple[RecallCase, ...]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("QK fixture must be a JSON array")
    cases: list[RecallCase] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"QK fixture row {index} must be an object")
        case_id = str(raw.get("case_id") or f"fixture-{index}").strip()
        query = str(raw.get("query") or "").strip()
        expected_absent = bool(raw.get("expected_absent"))
        gold_path = None if expected_absent else str(raw.get("gold_path") or "").strip()
        if (
            not case_id
            or not query
            or (gold_path is not None and gold_path not in bank_paths)
        ):
            raise ValueError(f"QK fixture row {index} is incomplete or stale")
        cases.append(
            RecallCase(
                case_id=case_id,
                request_id=f"fixture:{case_id}",
                query=query,
                gold_path=gold_path,
                query_source=(
                    "reviewed.negative_fixture"
                    if expected_absent
                    else "reviewed.fixture"
                ),
            )
        )
    return tuple(cases)


def policy_prefix_ids(
    tokenizer: Any, policy_data_directory: Path, cognition_directory: Path
) -> tuple[int, ...]:
    policy_files = sorted(policy_data_directory.rglob("*.md"))
    if policy_files:
        if len(policy_files) != 1:
            raise RuntimeError(
                "The experiment requires exactly one PolicyData document"
            )
        content = normalize_markdown(policy_files[0].read_text(encoding="utf-8"))
        return tuple(
            int(token)
            for token in tokenizer.encode(content, add_special_tokens=False)[:256]
        )
    cognition_files = sorted(cognition_directory.rglob("*.md"))
    if len(cognition_files) != 1:
        raise RuntimeError("The experiment requires exactly one Cognition document")
    content = normalize_markdown(cognition_files[0].read_text(encoding="utf-8"))
    return tuple(
        int(token) for token in tokenizer.encode(content, add_special_tokens=False)
    )


def query_spans(token_count: int) -> tuple[tuple[int, int], ...]:
    if token_count < 1:
        return ()
    span_count = min(8, token_count)
    width = math.ceil(token_count / span_count)
    return tuple(
        (start, min(start + width, token_count))
        for start in range(0, token_count, width)
    )[-8:]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Generate failed with HTTP {exc.code}: {detail}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Generate returned a non-object")
    return value


def _last_tensor(value: Any, expected_dims: int) -> torch.Tensor:
    try:
        tensor = torch.tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        tensor = None
    if tensor is not None:
        if tensor.ndim == expected_dims - 1:
            return tensor.unsqueeze(0)
        if tensor.ndim == expected_dims:
            return tensor
        if tensor.ndim == expected_dims + 1:
            for index in range(tensor.shape[0] - 1, -1, -1):
                candidate = tensor[index]
                if bool(torch.isfinite(candidate).any()):
                    return candidate
    if isinstance(value, (list, tuple)):
        for candidate in reversed(value):
            if candidate is None:
                continue
            try:
                return _last_tensor(candidate, expected_dims)
            except RuntimeError:
                continue
    rank = tensor.ndim if tensor is not None else "ragged"
    preview = repr(value)
    raise RuntimeError(
        f"Captured tensor has rank {rank}; expected {expected_dims - 1}, "
        f"{expected_dims}, or {expected_dims + 1}; value={preview[:600]}"
    )


def capture_query(
    base_url: str,
    tokenizer: Any,
    cognition_ids: tuple[int, ...],
    query: str,
    *,
    max_prompt_tokens: int,
    num_query_heads: int,
    head_dim: int,
    timeout: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    query_ids = tuple(
        int(token)
        for token in tokenizer.encode(query.strip(), add_special_tokens=False)
    )
    nonce_ids = tuple(
        int(token)
        for token in tokenizer.encode(
            f"\n[QK-EVAL {time.time_ns()}]\n", add_special_tokens=False
        )
    )
    prefix_ids = cognition_ids + nonce_ids
    capacity = max_prompt_tokens - len(prefix_ids)
    if capacity < 1:
        raise RuntimeError("Cognition and nonce prefix leave no query capacity")
    query_ids = query_ids[-capacity:]
    local_spans = query_spans(len(query_ids))
    spans = tuple(
        (start + len(prefix_ids), end + len(prefix_ids)) for start, end in local_spans
    )
    request_id = f"qk-experiment-{time.time_ns()}"
    response = post_json(
        f"{base_url.rstrip('/')}/generate",
        {
            "input_ids": list(prefix_ids + query_ids),
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 1,
                "custom_params": {
                    "qwen_exo_kind": "internal",
                    "qwen_exo_parent_request_id": request_id,
                    "qwen_exo_job_type": "query_probe",
                    "qwen_exo_full_qk_experiment": True,
                    "qwen_exo_query_spans": [
                        {"start": start, "end": end} for start, end in spans
                    ],
                },
            },
            "rid": request_id,
            "stream": False,
        },
        timeout,
    )
    meta = response.get("meta_info") or {}
    if "qwen_exo_user_query_full_heads" not in meta:
        raise RuntimeError("Server did not return full Q-head experiment metadata")
    if "qwen_exo_user_query_sketch" not in meta:
        raise RuntimeError("Server did not return the baseline 32D Q sketch")
    flattened_heads = _last_tensor(meta["qwen_exo_user_query_full_heads"], 3).reshape(
        -1
    )
    head_width = num_query_heads * head_dim
    if flattened_heads.numel() % head_width:
        raise RuntimeError(
            "Captured full Q metadata cannot be reshaped to model heads: "
            f"elements={flattened_heads.numel()} heads={num_query_heads} dim={head_dim}"
        )
    full_heads = flattened_heads.reshape(-1, num_query_heads, head_dim)
    flattened_sketch = _last_tensor(meta["qwen_exo_user_query_sketch"], 2).reshape(-1)
    if flattened_sketch.numel() % 32:
        raise RuntimeError("Captured 32D Q metadata has a non-divisible width")
    sketch = flattened_sketch.reshape(-1, 32)
    if full_heads.shape[0] != sketch.shape[0]:
        raise RuntimeError(
            "Full and compressed Q metadata disagree on span count: "
            f"full={full_heads.shape[0]} compressed={sketch.shape[0]}"
        )
    finite_full = torch.isfinite(full_heads).all(dim=(-1, -2))
    finite_sketch = torch.isfinite(sketch).all(dim=-1)
    keep = finite_full & finite_sketch
    if not bool(keep.any()):
        raise RuntimeError("Server returned no finite Q spans")
    return (
        full_heads[keep],
        sketch[keep],
        {
            "request_id": request_id,
            "prompt_tokens": len(prefix_ids) + len(query_ids),
            "query_tokens": len(query_ids),
            "span_count": int(keep.sum().item()),
        },
    )


def load_pages(
    bank_path: Path,
    native_root: Path,
    tokenizer: Any,
    *,
    tp_size: int,
) -> tuple[str, tuple[PageKeys, ...]]:
    bank = torch.load(bank_path, map_location="cpu", weights_only=True, mmap=True)
    source_digest = str(bank["source_digest"])
    special_ids = frozenset(int(token) for token in tokenizer.all_special_ids)
    pages: list[PageKeys] = []
    for page in bank.get("pages") or ():
        if str(page.get("lane")) != "knowledge":
            continue
        page_id = int(page["page_id"])
        rank_keys: list[torch.Tensor] = []
        token_ids: tuple[int, ...] | None = None
        for rank in range(tp_size):
            payload = _load_page_payload(
                native_root,
                source_digest=source_digest,
                page_id=page_id,
                rank=rank,
            )
            layer_ids = tuple(int(item) for item in payload.get("full_layer_ids") or ())
            final = (payload.get("full_attention") or {}).get(str(layer_ids[-1])) or {}
            keys = _dequantize_fp8(final["key"], dtype=torch.float32)
            if keys.ndim != 3:
                raise RuntimeError("Native K must have [tokens, heads, head_dim] shape")
            observed_ids = tuple(int(token) for token in payload.get("token_ids") or ())
            if token_ids is None:
                token_ids = observed_ids
            elif token_ids != observed_ids:
                raise RuntimeError("TP ranks disagree on page token IDs")
            rank_keys.append(keys)
        keys = torch.cat(rank_keys, dim=1)
        cognition_tokens = int(page.get("cognition_token_count") or 0)
        source_tokens = int(page["token_end"])
        searchable: list[int] = []
        for position in range(
            cognition_tokens, min(source_tokens, len(token_ids or ()))
        ):
            token_id = int((token_ids or ())[position])
            if token_id in special_ids:
                continue
            text = str(tokenizer.decode((token_id,), skip_special_tokens=False)).strip()
            if text and text not in _SINK_TOKEN_TEXT:
                searchable.append(position)
        if not searchable:
            continue
        pages.append(
            PageKeys(
                page_id=page_id,
                relative_path=str(page["relative_path"]),
                cognition_tokens=cognition_tokens,
                source_tokens=source_tokens,
                searchable_positions=torch.tensor(searchable, dtype=torch.long),
                keys=keys,
            )
        )
    if not pages:
        raise RuntimeError("Tensor Bank contains no searchable Knowledge pages")
    return source_digest, tuple(pages)


def compress_32(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] % 32 == 0:
        return value.reshape(*value.shape[:-1], 32, -1).mean(dim=-1)
    return torch.nn.functional.adaptive_avg_pool1d(
        value.reshape(-1, 1, value.shape[-1]), 32
    ).reshape(*value.shape[:-1], 32)


def aggregate_page(
    similarities: torch.Tensor, *, token_top: int = 4, query_top: int = 3
) -> float:
    token_top = min(token_top, similarities.shape[1])
    per_query = torch.topk(similarities, k=token_top, dim=1).values.mean(dim=1)
    query_top = min(query_top, per_query.shape[0])
    return float(torch.topk(per_query, k=query_top).values.mean().item())


def score_page(
    full_q: torch.Tensor,
    sketch_q: torch.Tensor,
    page: PageKeys,
    device: torch.device,
) -> dict[str, float]:
    positions = page.searchable_positions
    keys = page.keys.index_select(0, positions).to(device=device, dtype=torch.float32)
    full_q = full_q.to(device=device, dtype=torch.float32)
    sketch_q = sketch_q.to(device=device, dtype=torch.float32)
    if full_q.shape[1] % keys.shape[1]:
        raise RuntimeError(
            "Q heads cannot be mapped evenly onto KV heads: "
            f"Q={tuple(full_q.shape)} K={tuple(keys.shape)}"
        )
    group_width = full_q.shape[1] // keys.shape[1]
    grouped_q = full_q.reshape(
        full_q.shape[0], keys.shape[1], group_width, full_q.shape[2]
    )

    mean_q = full_q.mean(dim=1)
    mean_k = keys.mean(dim=1)
    q_256 = torch.nn.functional.normalize(mean_q, dim=-1)
    k_256 = torch.nn.functional.normalize(mean_k, dim=-1)
    sim_256 = q_256 @ k_256.transpose(0, 1)

    q_32 = torch.nn.functional.normalize(compress_32(mean_q), dim=-1)
    k_32 = torch.nn.functional.normalize(compress_32(mean_k), dim=-1)
    sim_32 = q_32 @ k_32.transpose(0, 1)
    baseline_sim = torch.nn.functional.normalize(sketch_q, dim=-1) @ k_32.transpose(
        0, 1
    )

    q_group = torch.nn.functional.normalize(grouped_q.mean(dim=2), dim=-1)
    k_group = torch.nn.functional.normalize(keys, dim=-1)
    sim_1024 = torch.einsum("sgd,tgd->sgt", q_group, k_group).mean(dim=1)

    q_heads = torch.nn.functional.normalize(grouped_q, dim=-1)
    head_cosines = torch.einsum("sgmd,tgd->sgmt", q_heads, k_group)
    sim_exact_cos = head_cosines.mean(dim=(1, 2))
    head_logits = torch.einsum("sgmd,tgd->sgmt", grouped_q, keys) / math.sqrt(
        full_q.shape[-1]
    )
    sim_exact_dot = head_logits.mean(dim=(1, 2))
    sim_qnorm_rawk = torch.einsum("sgmd,tgd->sgmt", q_heads, keys).mean(dim=(1, 2))
    sim_rawq_normk = torch.einsum("sgmd,tgd->sgmt", grouped_q, k_group).mean(dim=(1, 2))
    head_logits_by_token = head_logits.permute(0, 3, 1, 2).reshape(
        head_logits.shape[0], head_logits.shape[3], -1
    )
    head_counts = (2, 3, 4, 6, 8, 10, 12, 16, 20, 24)
    top_head_similarities = {
        count: torch.topk(
            head_logits_by_token,
            k=min(count, head_logits_by_token.shape[-1]),
            dim=-1,
        ).values.mean(dim=-1)
        for count in head_counts
    }

    return {
        "baseline_32d": aggregate_page(baseline_sim),
        "reconstructed_32d": aggregate_page(sim_32),
        "mean_head_256d": aggregate_page(sim_256),
        "grouped_1024d": aggregate_page(sim_1024),
        "exact_24head_cosine": aggregate_page(sim_exact_cos),
        "exact_24head_dot": aggregate_page(sim_exact_dot),
        "exact_24head_dot_max": aggregate_page(sim_exact_dot, token_top=1, query_top=1),
        "qnorm_rawk_dot": aggregate_page(sim_qnorm_rawk),
        "rawq_normk_dot": aggregate_page(sim_rawq_normk),
        "top2_head_dot": aggregate_page(top_head_similarities[2]),
        "top3_head_dot": aggregate_page(top_head_similarities[3]),
        "top4_head_dot": aggregate_page(top_head_similarities[4]),
        "top6_head_dot": aggregate_page(top_head_similarities[6]),
        "top8_head_dot": aggregate_page(top_head_similarities[8]),
        "top10_head_dot": aggregate_page(top_head_similarities[10]),
        "top12_head_dot": aggregate_page(top_head_similarities[12]),
        "top16_head_dot": aggregate_page(top_head_similarities[16]),
        "top20_head_dot": aggregate_page(top_head_similarities[20]),
        "top24_head_dot": aggregate_page(top_head_similarities[24]),
        "top4_head_dot_token1": aggregate_page(top_head_similarities[4], token_top=1),
        "top4_head_dot_token8": aggregate_page(top_head_similarities[4], token_top=8),
        "top4_head_dot_token16": aggregate_page(top_head_similarities[4], token_top=16),
        "top4_head_dot_query1": aggregate_page(top_head_similarities[4], query_top=1),
        "top4_head_dot_query2": aggregate_page(top_head_similarities[4], query_top=2),
        "top4_head_dot_query4": aggregate_page(top_head_similarities[4], query_top=4),
        "top4_head_dot_all_queries": aggregate_page(
            top_head_similarities[4], query_top=8
        ),
    }


def add_zscore_fusion(
    by_method: dict[str, list[dict[str, Any]]],
    *,
    name: str,
    left: str,
    right: str,
    left_weight: float,
) -> None:
    left_rows = by_method[left]
    right_by_path = {
        str(row["relative_path"]): float(row["score"]) for row in by_method[right]
    }
    left_scores = torch.tensor(
        [float(row["score"]) for row in left_rows], dtype=torch.float32
    )
    right_scores = torch.tensor(
        [right_by_path[str(row["relative_path"])] for row in left_rows],
        dtype=torch.float32,
    )
    left_z = (left_scores - left_scores.mean()) / left_scores.std().clamp_min(1e-6)
    right_z = (right_scores - right_scores.mean()) / right_scores.std().clamp_min(1e-6)
    fused = left_weight * left_z + (1 - left_weight) * right_z
    by_method[name] = [
        {**row, "score": float(score)}
        for row, score in zip(left_rows, fused.tolist(), strict=True)
    ]


def rank_case(
    case: RecallCase,
    full_q: torch.Tensor,
    sketch_q: torch.Tensor,
    pages: tuple[PageKeys, ...],
    device: torch.device,
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        scores = score_page(full_q, sketch_q, page, device)
        for method, score in scores.items():
            by_method.setdefault(method, []).append(
                {
                    "relative_path": page.relative_path,
                    "page_id": page.page_id,
                    "score": score,
                }
            )
    for left_weight in (0.25, 0.5, 0.75):
        add_zscore_fusion(
            by_method,
            name=f"top4_dot_cosine_z{int(left_weight * 100)}",
            left="top4_head_dot",
            right="exact_24head_cosine",
            left_weight=left_weight,
        )
    methods: dict[str, Any] = {}
    for method, rows in by_method.items():
        rows.sort(key=lambda item: (-float(item["score"]), item["relative_path"]))
        gold_index = next(
            (
                index
                for index, item in enumerate(rows)
                if item["relative_path"] == case.gold_path
            ),
            None,
        )
        gold_score = (
            float(rows[gold_index]["score"]) if gold_index is not None else None
        )
        best_negative = next(
            (
                float(item["score"])
                for item in rows
                if item["relative_path"] != case.gold_path
            ),
            None,
        )
        top1_margin = (
            float(rows[0]["score"]) - float(rows[1]["score"]) if len(rows) > 1 else None
        )
        methods[method] = {
            "rank": gold_index + 1 if gold_index is not None else None,
            "top1": gold_index == 0,
            "top3": gold_index is not None and gold_index < 3,
            "mrr": 1.0 / (gold_index + 1) if gold_index is not None else 0.0,
            "gold_score": gold_score,
            "best_negative_score": best_negative,
            "gold_margin": (
                gold_score - best_negative
                if gold_score is not None and best_negative is not None
                else None
            ),
            "score_min": min(float(item["score"]) for item in rows),
            "score_max": max(float(item["score"]) for item in rows),
            "top": rows[:8],
            "top1_margin": top1_margin,
            "clearly_separated": (
                top1_margin is not None and top1_margin >= _CLEAR_SEPARATION_MARGIN
            ),
        }
    return {
        "case_id": case.case_id,
        "source_request_id": case.request_id,
        "gold_path": case.gold_path,
        "query_source": case.query_source,
        "methods": methods,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("gold_path")]
    methods = sorted(
        {method for row in labeled_rows for method in (row.get("methods") or {}).keys()}
    )
    output: dict[str, Any] = {}
    for method in methods:
        values = [row["methods"][method] for row in labeled_rows]
        margins = [
            float(value["gold_margin"])
            for value in values
            if value.get("gold_margin") is not None
        ]
        top1_margins = [
            float(value["top1_margin"])
            for value in values
            if value.get("top1_margin") is not None
        ]
        output[method] = {
            "count": len(values),
            "top1": sum(bool(value["top1"]) for value in values) / len(values),
            "top3": sum(bool(value["top3"]) for value in values) / len(values),
            "mrr": sum(float(value["mrr"]) for value in values) / len(values),
            "positive_margin_rate": (
                sum(margin > 0 for margin in margins) / len(margins) if margins else 0.0
            ),
            "mean_gold_margin": sum(margins) / len(margins) if margins else None,
            "clear_separation_threshold": _CLEAR_SEPARATION_MARGIN,
            "clear_separation_rate": (
                sum(margin >= _CLEAR_SEPARATION_MARGIN for margin in top1_margins)
                / len(top1_margins)
                if top1_margins
                else 0.0
            ),
            "mean_top1_margin": (
                sum(top1_margins) / len(top1_margins) if top1_margins else None
            ),
            "minimum_top1_margin": min(top1_margins) if top1_margins else None,
        }
    return output


def summarize_consensus_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    method_names = ("top3_head_dot", "top4_head_dot")
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("gold_path"):
            continue
        methods = [row["methods"][name] for name in method_names]
        top_paths = [str(method["top"][0]["relative_path"]) for method in methods]
        margins = [float(method["top1_margin"]) for method in methods]
        accepted = len(set(top_paths)) == 1 and min(margins) >= _CLEAR_SEPARATION_MARGIN
        decisions.append(
            {
                "case_id": row["case_id"],
                "accepted": accepted,
                "selected_path": top_paths[0] if accepted else None,
                "gold_path": row["gold_path"],
                "correct": accepted and top_paths[0] == row["gold_path"],
                "top_paths": dict(zip(method_names, top_paths, strict=True)),
                "top1_margins": dict(zip(method_names, margins, strict=True)),
            }
        )
    accepted_rows = [decision for decision in decisions if decision["accepted"]]
    correct = sum(bool(decision["correct"]) for decision in accepted_rows)
    return {
        "status": "experimental_not_production_gated",
        "methods": list(method_names),
        "minimum_margin": _CLEAR_SEPARATION_MARGIN,
        "case_count": len(decisions),
        "accepted_count": len(accepted_rows),
        "coverage": len(accepted_rows) / len(decisions) if decisions else 0.0,
        "labeled_precision": (correct / len(accepted_rows) if accepted_rows else None),
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model-path", type=Path, default=Path("/models/qwen-exo"))
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--policydata", type=Path, required=True)
    parser.add_argument("--cognition", type=Path, required=True)
    parser.add_argument("--reflection-state", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=12288)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model_config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    text_config = getattr(model_config, "text_config", model_config)
    num_query_heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    source_digest, pages = load_pages(
        args.bank, args.native_root, tokenizer, tp_size=args.tp_size
    )
    bank_paths = frozenset(page.relative_path for page in pages)
    cases = build_cases(args.reflection_state, args.trace, bank_paths, tuple(args.case))
    if args.fixture is not None:
        cases = (*cases, *fixture_cases(args.fixture, bank_paths))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise RuntimeError("No labeled trajectory cases matched the live Tensor Bank")
    cognition_ids = policy_prefix_ids(tokenizer, args.policydata, args.cognition)
    expected_cognition = {
        page.cognition_tokens for page in pages if page.cognition_tokens > 0
    }
    if expected_cognition and expected_cognition != {len(cognition_ids)}:
        raise RuntimeError(
            f"Query prefix has {len(cognition_ids)} tokens; Bank expects {expected_cognition}"
        )

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        captured_at = time.perf_counter()
        full_q, sketch_q, capture = capture_query(
            args.base_url,
            tokenizer,
            cognition_ids,
            case.query,
            max_prompt_tokens=args.max_prompt_tokens,
            num_query_heads=num_query_heads,
            head_dim=head_dim,
            timeout=args.timeout,
        )
        ranked = rank_case(case, full_q, sketch_q, pages, device)
        ranked["capture"] = capture
        ranked["elapsed_seconds"] = round(time.perf_counter() - captured_at, 3)
        rows.append(ranked)
        ranks = {method: value["rank"] for method, value in ranked["methods"].items()}
        print(
            f"[{index}/{len(cases)}] {case.case_id[:52]} ranks={ranks}",
            flush=True,
        )

    report = {
        "schema": "qwen-exo-model-native-qk-experiment-v1",
        "source_digest": source_digest,
        "model_path": str(args.model_path),
        "bank_path": str(args.bank),
        "case_count": len(rows),
        "labeled_case_count": sum(bool(row.get("gold_path")) for row in rows),
        "negative_case_count": sum(not row.get("gold_path") for row in rows),
        "page_count": len(pages),
        "cognition_tokens": len(cognition_ids),
        "dimensions": {
            "baseline": 32,
            "mean_head": 256,
            "grouped_kv_heads": int(pages[0].keys.shape[1] * pages[0].keys.shape[2]),
            "query_heads": int(full_q.shape[1]),
            "head_dim": int(full_q.shape[2]),
        },
        "summary": summarize(rows),
        "candidate_consensus_gate": summarize_consensus_gate(rows),
        "negative_observations": [
            {
                "case_id": row["case_id"],
                "top": row["methods"]["top4_head_dot"]["top"][:3],
                "top1_margin": row["methods"]["top4_head_dot"]["top1_margin"],
            }
            for row in rows
            if not row.get("gold_path")
        ],
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
