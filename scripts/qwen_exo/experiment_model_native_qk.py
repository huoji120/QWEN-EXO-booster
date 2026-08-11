#!/usr/bin/env python3
"""Compare compressed and model-native Q/K recall on persisted trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
    original_task: str = ""
    current_user: str = ""
    trajectory_compaction: str = ""
    trajectory_source: str = ""
    hard_negative_paths: tuple[str, ...] = ()
    trajectory_tokens: tuple[int, ...] = ()
    trajectory_provenance: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class QuerySpan:
    role: str
    prompt_start: int
    prompt_end: int
    source_start: int
    source_end: int

    @property
    def anchor(self) -> bool:
        return self.role in {"original_task", "current_user"}

    def public_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "prompt_start": self.prompt_start,
            "prompt_end": self.prompt_end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "anchor": self.anchor,
        }


@dataclass(frozen=True, slots=True)
class PageKeys:
    page_id: int
    relative_path: str
    cognition_tokens: int
    source_tokens: int
    source_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
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
                original_task=query,
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
                original_task=query,
            )
        )
    return tuple(cases)


def _knowledge_document_path(root: Path, relative_path: str) -> Path:
    raw = str(relative_path).strip()
    relative = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or relative.is_absolute()
        or (relative.parts and ":" in relative.parts[0])
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Unsafe relative Knowledge path: {relative_path!r}")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"Knowledge path escapes --knowledge-root: {relative_path!r}"
        ) from exc
    if not candidate.is_file():
        raise ValueError(f"Knowledge path is not a file: {relative_path!r}")
    return candidate


def _materialize_trajectory_documents(
    tokenizer: Any,
    knowledge_root: Path,
    relative_paths: tuple[str, ...],
    *,
    max_tokens: int,
) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    if max_tokens < 1:
        raise ValueError("trajectory_max_tokens must be positive")
    source_rows: list[tuple[str, str, tuple[int, ...]]] = []
    for relative_path in relative_paths:
        document_path = _knowledge_document_path(knowledge_root, relative_path)
        source_text = document_path.read_text(encoding="utf-8")
        source_tokens = tuple(
            int(token)
            for token in tokenizer.encode(source_text, add_special_tokens=False)
        )
        source_rows.append((relative_path, source_text, source_tokens))
    combined_text = "\n\n".join(row[1] for row in source_rows)

    def encode_exact(text: str) -> tuple[int, ...]:
        return tuple(
            int(token) for token in tokenizer.encode(text, add_special_tokens=False)
        )

    combined_tokens = encode_exact(combined_text)
    selected_text = combined_text
    if len(combined_tokens) > max_tokens:
        low, high = 0, len(combined_text)
        while low < high:
            middle = (low + high + 1) // 2
            if len(encode_exact(combined_text[:middle])) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        selected_text = combined_text[:low]
        combined_tokens = encode_exact(selected_text)
        while len(combined_tokens) > max_tokens:
            selected_text = selected_text[:-1]
            combined_tokens = encode_exact(selected_text)
    if not combined_tokens:
        raise ValueError("Trajectory Knowledge documents encoded no tokens")

    sources: list[dict[str, Any]] = []
    cursor = 0
    for source_index, (relative_path, source_text, source_tokens) in enumerate(
        source_rows
    ):
        if source_index:
            cursor += 2
        source_start = cursor
        source_end = source_start + len(source_text)
        selected_character_count = max(
            0, min(len(selected_text), source_end) - source_start
        )
        selected_source_text = source_text[:selected_character_count]
        sources.append(
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
                "source_character_count": len(source_text),
                "source_token_count": len(source_tokens),
                "selected_character_count": selected_character_count,
                "selected_token_count": len(encode_exact(selected_source_text)),
                "truncated": selected_character_count < len(source_text),
            }
        )
        cursor = source_end
    return (
        selected_text,
        combined_tokens,
        {
            "source_text_kind": "exact_knowledge_document_text",
            "materialization": "read_only_exact_prefix_tokenization",
            "declared_max_tokens": max_tokens,
            "selected_character_count": len(selected_text),
            "selected_token_count": len(combined_tokens),
            "truncated": len(selected_text) < len(combined_text),
            "sources": sources,
        },
    )


def fixture_cases(
    path: Path,
    bank_paths: frozenset[str],
    *,
    tokenizer: Any | None = None,
    knowledge_root: Path | None = None,
    require_source_backed_trajectory: bool = False,
) -> tuple[RecallCase, ...]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("QK fixture must be a JSON array")
    cases: list[RecallCase] = []
    seen_case_ids: set[str] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"QK fixture row {index} must be an object")
        case_id = str(raw.get("case_id") or f"fixture-{index}").strip()
        if case_id in seen_case_ids:
            raise ValueError(f"QK fixture has duplicate case_id {case_id!r}")
        seen_case_ids.add(case_id)
        for field in (
            "query",
            "original_task",
            "current_user",
            "trajectory_compaction",
            "trajectory_source",
        ):
            value = raw.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"QK fixture row {index} field {field} must be a string"
                )
        legacy_query = str(raw.get("query") or "").strip()
        original_task = str(raw.get("original_task") or "").strip()
        if legacy_query and original_task and legacy_query != original_task:
            raise ValueError(
                f"QK fixture row {index} has conflicting query and original_task"
            )
        original_task = original_task or legacy_query
        current_user = str(raw.get("current_user") or "").strip()
        trajectory_compaction = str(raw.get("trajectory_compaction") or "").strip()
        had_inline_trajectory = bool(trajectory_compaction)
        hard_negative_value = raw.get("hard_negative_paths") or []
        trajectory_path_value = raw.get("trajectory_document_paths") or []
        for field, value in (
            ("hard_negative_paths", hard_negative_value),
            ("trajectory_document_paths", trajectory_path_value),
        ):
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item.strip() for item in value
            ):
                raise ValueError(
                    f"QK fixture row {index} {field} must be a string array"
                )
        trajectory_source = str(raw.get("trajectory_source") or "").strip()
        hard_negative_paths = tuple(
            dict.fromkeys(str(item).strip() for item in hard_negative_value)
        )
        trajectory_document_paths = tuple(
            dict.fromkeys(str(item).strip() for item in trajectory_path_value)
        )
        expected_absent = bool(raw.get("expected_absent"))
        gold_path = None if expected_absent else str(raw.get("gold_path") or "").strip()
        stale_paths = sorted(
            candidate
            for candidate in (
                *hard_negative_paths,
                *trajectory_document_paths,
                *((gold_path,) if gold_path else ()),
            )
            if candidate not in bank_paths
        )
        if (
            not case_id
            or not original_task
            or stale_paths
            or (gold_path is not None and gold_path in hard_negative_paths)
        ):
            detail = f"; stale Bank paths={stale_paths}" if stale_paths else ""
            raise ValueError(f"QK fixture row {index} is incomplete or stale{detail}")
        trajectory_tokens: tuple[int, ...] = ()
        trajectory_provenance: dict[str, Any] | None = None
        if trajectory_document_paths:
            if tokenizer is None or knowledge_root is None:
                raise ValueError(
                    "trajectory_document_paths require tokenizer and --knowledge-root"
                )
            max_tokens = raw.get("trajectory_max_tokens")
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
                raise ValueError(
                    f"QK fixture row {index} trajectory_max_tokens must be an integer"
                )
            if require_source_backed_trajectory and max_tokens != 4500:
                raise ValueError(
                    f"QK fixture row {index} must declare trajectory_max_tokens=4500"
                )
            (
                trajectory_compaction,
                trajectory_tokens,
                trajectory_provenance,
            ) = _materialize_trajectory_documents(
                tokenizer,
                knowledge_root,
                trajectory_document_paths,
                max_tokens=max_tokens,
            )
            trajectory_source = "knowledge_documents_exact_text"
        elif trajectory_compaction:
            trajectory_tokens = (
                _encode(tokenizer, trajectory_compaction) if tokenizer else ()
            )
            trajectory_provenance = {
                "source_text_kind": "reviewed_reconstruction",
                "materialization": "fixture_inline_text",
                "selected_token_count": len(trajectory_tokens),
                "sources": [],
            }
        if require_source_backed_trajectory and (
            not trajectory_document_paths or had_inline_trajectory
        ):
            raise ValueError(
                f"QK fixture row {index} requires only source-backed trajectory documents"
            )
        cases.append(
            RecallCase(
                case_id=case_id,
                request_id=f"fixture:{case_id}",
                query=original_task,
                gold_path=gold_path,
                query_source=(
                    "reviewed.negative_fixture"
                    if expected_absent
                    else "reviewed.fixture"
                ),
                original_task=original_task,
                current_user=current_user,
                trajectory_compaction=trajectory_compaction,
                trajectory_source=trajectory_source,
                hard_negative_paths=hard_negative_paths,
                trajectory_tokens=trajectory_tokens,
                trajectory_provenance=trajectory_provenance,
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


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(
        int(token)
        for token in tokenizer.encode(str(text).strip(), add_special_tokens=False)
    )


def _query_spans(token_count: int, state_limit: int) -> tuple[tuple[int, int], ...]:
    if token_count < 1 or state_limit < 1:
        return ()
    state_count = min(int(state_limit), int(token_count))
    width = math.ceil(token_count / state_count)
    return tuple(
        (start, min(start + width, token_count))
        for start in range(0, token_count, width)
    )[:state_count]


def _bounded_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars < 1 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[bounded]...\n{text[-tail:]}"


def plan_legacy_flat_query(
    tokenizer: Any,
    case: RecallCase,
    *,
    capacity: int,
    prompt_offset: int,
) -> tuple[tuple[int, ...], tuple[QuerySpan, ...], dict[str, Any]]:
    original = str(case.original_task or case.query).strip()
    current = str(case.current_user).strip()
    exact_source = (case.trajectory_provenance or {}).get(
        "source_text_kind"
    ) == "exact_knowledge_document_text"
    trajectory = str(case.trajectory_compaction)
    if not exact_source:
        trajectory = trajectory.strip()
    parts: list[tuple[str, str]] = []
    if original:
        parts.append(("ORIGINAL TASK", _bounded_text(original, 8000)))
    deduplicated_current = bool(current and current == original)
    if current and not deduplicated_current:
        parts.append(("CURRENT USER REQUEST", _bounded_text(current, 4000)))
    if trajectory:
        parts.append(("RECENT EXECUTION TRAJECTORY", trajectory))
    flat_text = "\n\n".join(f"{label}:\n{text}" for label, text in parts)
    encoded = tuple(
        int(token) for token in tokenizer.encode(flat_text, add_special_tokens=False)
    )
    if not encoded:
        raise RuntimeError("Legacy flat query encoded no tokens")
    selected = encoded[-capacity:]
    source_start = len(encoded) - len(selected)
    span_bounds = _query_spans(len(selected), 8)
    spans = tuple(
        QuerySpan(
            role="legacy_flat",
            prompt_start=prompt_offset + start,
            prompt_end=prompt_offset + end,
            source_start=source_start + start,
            source_end=source_start + end,
        )
        for start, end in span_bounds
    )
    return (
        selected,
        spans,
        {
            "plan": "legacy_flat",
            "capacity": capacity,
            "labels": [label for label, _text in parts],
            "deduplicated_current_user": deduplicated_current,
            "source_token_count": len(encoded),
            "selected_source_start": source_start,
            "selected_token_count": len(selected),
            "tail_truncated": source_start > 0,
            "state_count": len(spans),
            "state_partition": "even_ceil_width_untyped",
        },
    )


def plan_role_query(
    tokenizer: Any,
    case: RecallCase,
    *,
    capacity: int,
    prompt_offset: int,
    strict_trajectory_fraction: bool = False,
) -> tuple[tuple[int, ...], tuple[QuerySpan, ...], dict[str, Any]]:
    original_task = str(case.original_task or case.query).strip()
    current_user = str(case.current_user).strip()
    deduplicated_roles: list[str] = []
    if current_user and current_user == original_task:
        current_user = ""
        deduplicated_roles.append("current_user")
    encoded = [
        ("original_task", _encode(tokenizer, original_task)),
        ("current_user", _encode(tokenizer, current_user)),
        (
            "trajectory_compaction",
            case.trajectory_tokens or _encode(tokenizer, case.trajectory_compaction),
        ),
    ]
    encoded = [(role, tokens) for role, tokens in encoded if tokens]
    if not encoded:
        raise RuntimeError("Query role plan encoded no tokens")
    anchors = [
        (role, tokens)
        for role, tokens in encoded
        if role in {"original_task", "current_user"}
    ]
    if not anchors:
        raise RuntimeError("Query role plan has no anchor role")
    if len(anchors) > capacity:
        raise RuntimeError("Query budget cannot represent every anchor role")
    trajectory = next(
        (tokens for role, tokens in encoded if role == "trajectory_compaction"), ()
    )
    if strict_trajectory_fraction:
        anchor_token_count = sum(len(tokens) for _role, tokens in anchors)
        trajectory_budget = min(
            len(trajectory),
            max(0, capacity // 4),
            max(0, anchor_token_count // 3),
            capacity - len(anchors),
        )
    else:
        trajectory_budget = min(
            len(trajectory), max(0, capacity // 4), capacity - len(anchors)
        )
    role_budgets: dict[str, int] = {}
    remaining = capacity - trajectory_budget
    pending = list(anchors)
    while pending and remaining:
        next_pending: list[tuple[str, tuple[int, ...]]] = []
        for role, tokens in pending:
            if remaining < 1:
                next_pending.append((role, tokens))
                continue
            allocated = role_budgets.get(role, 0)
            if allocated < len(tokens):
                role_budgets[role] = allocated + 1
                remaining -= 1
            if role_budgets.get(role, 0) < len(tokens):
                next_pending.append((role, tokens))
        pending = next_pending
    if trajectory_budget:
        role_budgets["trajectory_compaction"] = trajectory_budget

    selected: list[tuple[str, tuple[int, ...], int, int]] = []
    for role, tokens in encoded:
        budget = role_budgets.get(role, 0)
        if budget:
            source_start = len(tokens) - budget
            selected.append((role, tokens[source_start:], source_start, len(tokens)))
    anchor_roles = [
        role
        for role, _tokens, _source_start, _source_count in selected
        if role in {"original_task", "current_user"}
    ]
    state_counts = {role: 1 for role in anchor_roles}
    trajectory_tokens = next(
        (
            tokens
            for role, tokens, _source_start, _source_count in selected
            if role == "trajectory_compaction"
        ),
        (),
    )
    trajectory_states = min(2, len(trajectory_tokens), max(0, 8 - len(anchor_roles)))
    if trajectory_states:
        state_counts["trajectory_compaction"] = trajectory_states
    remaining_states = 8 - sum(state_counts.values())
    while anchor_roles and remaining_states:
        progressed = False
        for role in anchor_roles:
            token_count = next(
                len(tokens)
                for item_role, tokens, _source_start, _source_count in selected
                if item_role == role
            )
            if state_counts[role] < token_count and remaining_states:
                state_counts[role] += 1
                remaining_states -= 1
                progressed = True
        if not progressed:
            break

    query_tokens: list[int] = []
    spans: list[QuerySpan] = []
    selected_roles: list[dict[str, Any]] = []
    for role, tokens, source_start, source_count in selected:
        prompt_role_start = prompt_offset + len(query_tokens)
        query_tokens.extend(tokens)
        role_spans = _query_spans(len(tokens), state_counts.get(role, 0))
        for start, end in role_spans:
            spans.append(
                QuerySpan(
                    role=role,
                    prompt_start=prompt_role_start + start,
                    prompt_end=prompt_role_start + end,
                    source_start=source_start + start,
                    source_end=source_start + end,
                )
            )
        selected_roles.append(
            {
                "role": role,
                "source_token_count": source_count,
                "selected_source_start": source_start,
                "selected_token_count": len(tokens),
                "state_count": len(role_spans),
            }
        )
    if not spans or len(spans) > 8:
        raise RuntimeError("Query role plan did not produce one to eight states")
    selected_trajectory_tokens = sum(
        int(role["selected_token_count"])
        for role in selected_roles
        if role["role"] == "trajectory_compaction"
    )
    if strict_trajectory_fraction and selected_trajectory_tokens * 4 > len(
        query_tokens
    ):
        raise RuntimeError("Trajectory exceeds one quarter of role-plan query tokens")
    return (
        tuple(query_tokens),
        tuple(spans),
        {
            "capacity": capacity,
            "deduplicated_roles": deduplicated_roles,
            **(
                {
                    "selected_token_count": len(query_tokens),
                    "selected_trajectory_tokens": selected_trajectory_tokens,
                    "selected_trajectory_fraction": (
                        selected_trajectory_tokens / len(query_tokens)
                        if query_tokens
                        else 0.0
                    ),
                }
                if strict_trajectory_fraction
                else {}
            ),
            "roles": selected_roles,
        },
    )


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
    case: RecallCase,
    *,
    max_prompt_tokens: int,
    num_query_heads: int,
    head_dim: int,
    timeout: float,
    capture_plan: str = "role",
    nonce_ids: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[QuerySpan, ...], dict[str, Any]]:
    paired_capture = nonce_ids is not None
    if nonce_ids is None:
        nonce_ids = _encode(tokenizer, f"\n[QK-EVAL {time.time_ns()}]\n")
    prefix_ids = cognition_ids + nonce_ids
    capacity = max_prompt_tokens - len(prefix_ids)
    if capacity < 1:
        raise RuntimeError("Cognition and nonce prefix leave no query capacity")
    if capture_plan == "role":
        query_ids, spans, allocation = plan_role_query(
            tokenizer,
            case,
            capacity=capacity,
            prompt_offset=len(prefix_ids),
            strict_trajectory_fraction=paired_capture,
        )
    elif capture_plan == "legacy_flat":
        query_ids, spans, allocation = plan_legacy_flat_query(
            tokenizer,
            case,
            capacity=capacity,
            prompt_offset=len(prefix_ids),
        )
    else:
        raise ValueError(f"Unknown capture plan: {capture_plan!r}")
    request_id = (
        f"qk-experiment-{capture_plan}-{time.time_ns()}"
        if paired_capture
        else f"qk-experiment-{time.time_ns()}"
    )
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
                        {"start": span.prompt_start, "end": span.prompt_end}
                        for span in spans
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
    expected_states = len(spans)
    if full_heads.shape[0] != expected_states or sketch.shape[0] != expected_states:
        raise RuntimeError(
            "Captured Q state count disagrees with the exact role plan: "
            f"planned={expected_states} full={full_heads.shape[0]} "
            f"compressed={sketch.shape[0]}"
        )
    if not bool(torch.isfinite(full_heads).all()) or not bool(
        torch.isfinite(sketch).all()
    ):
        raise RuntimeError("Server returned a non-finite planned Q state")
    return (
        full_heads,
        sketch,
        spans,
        {
            "request_id": request_id,
            **(
                {
                    "capture_plan": capture_plan,
                    "nonce_token_ids": list(nonce_ids),
                }
                if paired_capture
                else {}
            ),
            "prompt_tokens": len(prefix_ids) + len(query_ids),
            "cognition_tokens": len(cognition_ids),
            "nonce_tokens": len(nonce_ids),
            "query_tokens": len(query_ids),
            "span_count": expected_states,
            "role_allocation": allocation,
            "role_spans": [span.public_dict() for span in spans],
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
            if not layer_ids:
                raise RuntimeError("Native page has no persisted Full-Attention layer")
            final = (payload.get("full_attention") or {}).get(str(layer_ids[-1])) or {}
            if "key" not in final:
                raise RuntimeError("Native page has no persisted final-layer K state")
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
        if (
            token_ids is None
            or len(token_ids) < source_tokens
            or keys.shape[0] < source_tokens
        ):
            raise RuntimeError(
                "Native page K/token IDs do not cover its source coordinates"
            )
        source_positions = tuple(
            int(position)
            for position in (page.get("source_positions") or range(source_tokens))
        )
        if len(source_positions) < source_tokens:
            raise RuntimeError("Tensor Bank page source-position map is incomplete")
        searchable: list[int] = []
        for position in range(cognition_tokens, source_tokens):
            token_id = int(token_ids[position])
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
                source_positions=source_positions,
                token_ids=token_ids,
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


def _decode_bounded(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    text = str(tokenizer.decode(token_ids, skip_special_tokens=False)).strip()
    return text if len(text) <= 240 else text[:237] + "..."


def _score_attribution(
    tokenizer: Any,
    page: PageKeys,
    span: QuerySpan,
    query_index: int,
    query_score: float,
    support_positions: tuple[int, ...],
    *,
    window_start: int | None,
    window_end: int | None,
) -> dict[str, Any]:
    bounded_support = tuple(sorted(support_positions))[:4]
    support_tokens = [
        {
            "raw_position": position,
            "source_position": int(page.source_positions[position]),
            "token_id": int(page.token_ids[position]),
            "text": _decode_bounded(tokenizer, (int(page.token_ids[position]),)),
        }
        for position in bounded_support
    ]
    if window_start is not None and window_end is not None:
        snippet_ids = tuple(
            int(token) for token in page.token_ids[window_start:window_end]
        )
        source_window_start = int(page.source_positions[window_start])
        source_window_end = int(page.source_positions[window_end - 1]) + 1
    else:
        snippet_ids = tuple(
            int(page.token_ids[position]) for position in bounded_support
        )
        source_window_start = None
        source_window_end = None
    return {
        "query_index": query_index,
        "query_role": span.role,
        "query_prompt_start": span.prompt_start,
        "query_prompt_end": span.prompt_end,
        "query_source_start": span.source_start,
        "query_source_end": span.source_end,
        "query_score": query_score,
        "window_raw_start": window_start,
        "window_raw_end": window_end,
        "window_source_start": source_window_start,
        "window_source_end": source_window_end,
        "support_raw_positions": list(bounded_support),
        "support_source_positions": [
            int(page.source_positions[position]) for position in bounded_support
        ],
        "support_tokens": support_tokens,
        "decoded_support_snippet": _decode_bounded(tokenizer, snippet_ids),
    }


def _aggregate_role_queries(
    tokenizer: Any,
    page: PageKeys,
    spans: tuple[QuerySpan, ...],
    query_rows: list[tuple[int, float, tuple[int, ...], int | None, int | None]],
) -> dict[str, Any] | None:
    if not query_rows:
        return None
    query_rows.sort(key=lambda item: (-item[1], item[0]))
    selected = query_rows[: min(4, len(query_rows))]
    if not all(math.isfinite(item[1]) for item in selected):
        return None
    winner = selected[0]
    return {
        "score": sum(item[1] for item in selected) / len(selected),
        "attribution": _score_attribution(
            tokenizer,
            page,
            spans[winner[0]],
            winner[0],
            winner[1],
            winner[2],
            window_start=winner[3],
            window_end=winner[4],
        ),
    }


def _global_role_score(
    tokenizer: Any,
    page: PageKeys,
    spans: tuple[QuerySpan, ...],
    token_scores: torch.Tensor,
    query_indices: tuple[int, ...],
) -> dict[str, Any] | None:
    searchable = page.searchable_positions.to(device=token_scores.device)
    required_support = min(4, int(searchable.numel()))
    if required_support < 1:
        return None
    query_rows: list[tuple[int, float, tuple[int, ...], int | None, int | None]] = []
    for query_index in query_indices:
        values = token_scores[query_index].index_select(0, searchable)
        finite = torch.isfinite(values)
        if int(finite.sum().item()) < required_support:
            continue
        masked = values.masked_fill(~finite, float("-inf"))
        top = torch.topk(masked, k=required_support, sorted=True)
        raw_positions = tuple(
            int(searchable[int(offset)].item()) for offset in top.indices.tolist()
        )
        query_rows.append(
            (query_index, float(top.values.mean().item()), raw_positions, None, None)
        )
    return _aggregate_role_queries(tokenizer, page, spans, query_rows)


def _local_role_score(
    tokenizer: Any,
    page: PageKeys,
    spans: tuple[QuerySpan, ...],
    token_scores: torch.Tensor,
    query_indices: tuple[int, ...],
    *,
    window_width: int = 16,
) -> dict[str, Any] | None:
    document_start = page.cognition_tokens
    document_end = page.source_tokens
    document_width = document_end - document_start
    width = min(max(1, window_width), document_width)
    if width < 1:
        return None
    search_mask = torch.zeros(
        page.source_tokens, dtype=torch.bool, device=token_scores.device
    )
    search_mask[page.searchable_positions.to(device=token_scores.device)] = True
    document_scores = token_scores[:, document_start:document_end]
    score_windows = document_scores.unfold(1, width, 1)
    search_windows = search_mask[document_start:document_end].unfold(0, width, 1)
    finite_windows = search_windows.unsqueeze(0) & torch.isfinite(score_windows)
    required_support = min(4, width)
    masked = score_windows.masked_fill(~finite_windows, float("-inf"))
    top = torch.topk(masked, k=required_support, dim=2, sorted=True)
    support_scores = top.values.mean(dim=2).masked_fill(
        finite_windows.sum(dim=2) < required_support, float("-inf")
    )
    query_rows: list[tuple[int, float, tuple[int, ...], int | None, int | None]] = []
    for query_index in query_indices:
        best_score, best_start = support_scores[query_index].max(dim=0)
        if not bool(torch.isfinite(best_score)):
            continue
        local_start = int(best_start.item())
        raw_window_start = document_start + local_start
        raw_window_end = raw_window_start + width
        raw_positions = tuple(
            raw_window_start + int(offset)
            for offset in top.indices[query_index, local_start].tolist()
            if bool(search_windows[local_start, int(offset)].item())
        )
        if len(raw_positions) != required_support:
            continue
        query_rows.append(
            (
                query_index,
                float(best_score.item()),
                raw_positions,
                raw_window_start,
                raw_window_end,
            )
        )
    return _aggregate_role_queries(tokenizer, page, spans, query_rows)


def _raw_top4_token_scores(
    full_q: torch.Tensor, page: PageKeys, device: torch.device
) -> torch.Tensor:
    queries = full_q.to(device=device, dtype=torch.float32)
    keys = page.keys[: page.source_tokens].to(device=device, dtype=torch.float32)
    if queries.shape[1] % keys.shape[1]:
        raise RuntimeError("Q heads cannot be mapped evenly onto persisted K heads")
    grouped_q = queries.reshape(
        queries.shape[0],
        keys.shape[1],
        queries.shape[1] // keys.shape[1],
        queries.shape[2],
    )
    head_logits = torch.einsum("qkrd,tkd->qtkr", grouped_q, keys) / math.sqrt(
        int(queries.shape[2])
    )
    flattened_heads = head_logits.flatten(start_dim=2)
    finite_heads = torch.isfinite(flattened_heads)
    if int(flattened_heads.shape[2]) < 4:
        raise RuntimeError("Persisted Q/K geometry cannot provide Top-4 head pairs")
    top_heads = torch.topk(
        flattened_heads.masked_fill(~finite_heads, float("-inf")),
        k=4,
        dim=2,
        sorted=True,
    ).values
    return top_heads.mean(dim=2).masked_fill(finite_heads.sum(dim=2) < 4, float("-inf"))


def score_role_variants(
    tokenizer: Any,
    full_q: torch.Tensor,
    spans: tuple[QuerySpan, ...],
    page: PageKeys,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    token_scores = _raw_top4_token_scores(full_q, page, device)
    all_rows = tuple(range(len(spans)))
    anchor_rows = tuple(index for index, span in enumerate(spans) if span.anchor)
    trajectory_rows = tuple(
        index
        for index, span in enumerate(spans)
        if span.role == "trajectory_compaction"
    )
    definitions = (
        ("legacy_global_all_roles_top4x4x4", _global_role_score, all_rows),
        ("anchor_global_top4x4x4", _global_role_score, anchor_rows),
        ("anchor_local_window16_top4x4x4", _local_role_score, anchor_rows),
        (
            "trajectory_diagnostic_global_top4x4x4",
            _global_role_score,
            trajectory_rows,
        ),
        (
            "trajectory_diagnostic_local_window16_top4x4x4",
            _local_role_score,
            trajectory_rows,
        ),
    )
    results: dict[str, dict[str, Any]] = {}
    for name, scorer, indices in definitions:
        if not indices:
            continue
        result = scorer(tokenizer, page, spans, token_scores, indices)
        if result is not None and math.isfinite(float(result["score"])):
            results[name] = result
    return results


def score_followup_variants(
    tokenizer: Any,
    legacy_q: torch.Tensor,
    legacy_spans: tuple[QuerySpan, ...],
    role_q: torch.Tensor,
    role_spans: tuple[QuerySpan, ...],
    page: PageKeys,
    device: torch.device,
    window_widths: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    legacy_scores = _raw_top4_token_scores(legacy_q, page, device)
    role_scores = _raw_top4_token_scores(role_q, page, device)
    legacy_rows = tuple(range(len(legacy_spans)))
    anchor_rows = tuple(index for index, span in enumerate(role_spans) if span.anchor)
    if not legacy_rows or not anchor_rows:
        raise RuntimeError("Paired scoring requires flat states and anchor role states")
    results: dict[str, dict[str, Any]] = {}
    legacy = _global_role_score(
        tokenizer, page, legacy_spans, legacy_scores, legacy_rows
    )
    anchor = _global_role_score(tokenizer, page, role_spans, role_scores, anchor_rows)
    if legacy is not None and math.isfinite(float(legacy["score"])):
        results["legacy_flat_global_top4x4x4"] = legacy
    if anchor is not None and math.isfinite(float(anchor["score"])):
        results["anchor_global_top4x4x4"] = anchor
    for width in window_widths:
        local = _local_role_score(
            tokenizer,
            page,
            role_spans,
            role_scores,
            anchor_rows,
            window_width=width,
        )
        if local is not None and math.isfinite(float(local["score"])):
            results[f"anchor_local_window{width}_top4x4x4"] = local
    return results


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
    tokenizer: Any,
    case: RecallCase,
    full_q: torch.Tensor,
    sketch_q: torch.Tensor,
    spans: tuple[QuerySpan, ...],
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
        for method, result in score_role_variants(
            tokenizer, full_q, spans, page, device
        ).items():
            by_method.setdefault(method, []).append(
                {
                    "relative_path": page.relative_path,
                    "page_id": page.page_id,
                    "score": float(result["score"]),
                    "attribution": result["attribution"],
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
    return _ranked_case_from_methods(case, by_method)


def _ranked_case_from_methods(
    case: RecallCase,
    by_method: dict[str, list[dict[str, Any]]],
    *,
    followup: bool = False,
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    hard_negative_set = frozenset(case.hard_negative_paths)
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
        gold_row = rows[gold_index] if gold_index is not None else None
        gold_score = float(gold_row["score"]) if gold_row is not None else None
        best_negative = next(
            (
                float(item["score"])
                for item in rows
                if item["relative_path"] != case.gold_path
            ),
            None,
        )
        named_hard_rows = [
            {**item, "rank": index + 1}
            for index, item in enumerate(rows)
            if item["relative_path"] in hard_negative_set
        ]
        best_named_hard = named_hard_rows[0] if named_hard_rows else None
        named_hard_score = (
            float(best_named_hard["score"]) if best_named_hard is not None else None
        )
        named_hard_margin = (
            gold_score - named_hard_score
            if gold_score is not None and named_hard_score is not None
            else None
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
            "gold_evidence": (
                {**gold_row, "rank": gold_index + 1} if gold_row else None
            ),
            "best_negative_score": best_negative,
            "gold_margin": (
                gold_score - best_negative
                if gold_score is not None and best_negative is not None
                else None
            ),
            "best_named_hard_negative": best_named_hard,
            "named_hard_negative_scores": [
                {
                    "relative_path": item["relative_path"],
                    "page_id": item["page_id"],
                    "score": item["score"],
                    "rank": item["rank"],
                }
                for item in named_hard_rows
            ],
            "relevant_vs_named_hard_negative_margin": named_hard_margin,
            "score_min": min(float(item["score"]) for item in rows),
            "score_max": max(float(item["score"]) for item in rows),
            "top": rows[:8],
            "top1_margin": top1_margin,
            "clearly_separated": (
                top1_margin is not None and top1_margin >= _CLEAR_SEPARATION_MARGIN
            ),
        }
        if followup:
            named_clear = (
                named_hard_margin is not None
                and named_hard_margin >= _CLEAR_SEPARATION_MARGIN
            )
            methods[method]["clearly_separated"] = named_clear
            methods[method]["named_hard_negative_clearly_separated"] = named_clear
            methods[method][
                "separation_basis"
            ] = "relevant_vs_named_hard_negative_margin"
    result = {
        "case_id": case.case_id,
        "source_request_id": case.request_id,
        "gold_path": case.gold_path,
        "hard_negative_paths": list(case.hard_negative_paths),
        "query_source": case.query_source,
        "trajectory_source": case.trajectory_source or None,
        "methods": methods,
    }
    if followup:
        result["trajectory_provenance"] = case.trajectory_provenance
    return result


def rank_paired_case(
    tokenizer: Any,
    case: RecallCase,
    legacy_q: torch.Tensor,
    legacy_spans: tuple[QuerySpan, ...],
    role_q: torch.Tensor,
    role_spans: tuple[QuerySpan, ...],
    pages: tuple[PageKeys, ...],
    device: torch.device,
    window_widths: tuple[int, ...],
) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        for method, result in score_followup_variants(
            tokenizer,
            legacy_q,
            legacy_spans,
            role_q,
            role_spans,
            page,
            device,
            window_widths,
        ).items():
            by_method.setdefault(method, []).append(
                {
                    "relative_path": page.relative_path,
                    "page_id": page.page_id,
                    "score": float(result["score"]),
                    "attribution": result["attribution"],
                }
            )
    expected_methods = {
        "legacy_flat_global_top4x4x4",
        "anchor_global_top4x4x4",
        *(f"anchor_local_window{width}_top4x4x4" for width in window_widths),
    }
    missing = sorted(expected_methods - by_method.keys())
    if missing:
        raise RuntimeError(f"Paired scoring did not produce methods: {missing}")
    return _ranked_case_from_methods(case, by_method, followup=True)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("gold_path")]
    methods = sorted(
        {method for row in labeled_rows for method in (row.get("methods") or {}).keys()}
    )
    output: dict[str, Any] = {}
    for method in methods:
        values = [
            row["methods"][method] for row in labeled_rows if method in row["methods"]
        ]
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


def _numeric_stats(values: list[float]) -> dict[str, Any] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "variance": statistics.pvariance(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _margin_sign_stability(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in values]
    counts = {str(sign): signs.count(sign) for sign in (-1, 0, 1)}
    return {
        "counts": counts,
        "all_same": len(set(signs)) == 1,
        "modal_fraction": max(counts.values()) / len(signs),
    }


def summarize_repetitions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for case_id in dict.fromkeys(str(row["case_id"]) for row in rows):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        methods = sorted(
            {
                method
                for row in case_rows
                for method in (row.get("methods") or {}).keys()
            }
        )
        method_output: dict[str, Any] = {}
        for method in methods:
            values = [
                row["methods"][method] for row in case_rows if method in row["methods"]
            ]
            ranks = [float(value["rank"]) for value in values if value.get("rank")]
            relevant_scores = [
                float(value["gold_score"])
                for value in values
                if value.get("gold_score") is not None
            ]
            margins = [
                float(value["relevant_vs_named_hard_negative_margin"])
                for value in values
                if value.get("relevant_vs_named_hard_negative_margin") is not None
            ]
            rank_counts = {
                str(int(rank)): ranks.count(rank) for rank in sorted(set(ranks))
            }
            method_output[method] = {
                "capture_count": len(values),
                "rank": _numeric_stats(ranks),
                "rank_stability": (
                    {
                        "counts": rank_counts,
                        "all_same": len(rank_counts) == 1,
                        "modal_fraction": max(rank_counts.values()) / len(ranks),
                    }
                    if ranks
                    else None
                ),
                "relevant_score": _numeric_stats(relevant_scores),
                "relevant_vs_named_hard_negative_margin": _numeric_stats(margins),
                "margin_sign_stability": _margin_sign_stability(margins),
            }
        output[case_id] = {
            "capture_count": len(case_rows),
            "methods": method_output,
        }
    return output


def _paired_method_comparison(
    rows: list[dict[str, Any]], *, baseline: str, candidate: str
) -> dict[str, Any]:
    margin_deltas: list[float] = []
    rank_deltas: list[float] = []
    margin_outcomes = {"wins": 0, "ties": 0, "regressions": 0}
    rank_outcomes = {"wins": 0, "ties": 0, "regressions": 0}
    for row in rows:
        baseline_value = (row.get("methods") or {}).get(baseline)
        candidate_value = (row.get("methods") or {}).get(candidate)
        if not baseline_value or not candidate_value:
            continue
        baseline_margin = baseline_value.get("relevant_vs_named_hard_negative_margin")
        candidate_margin = candidate_value.get("relevant_vs_named_hard_negative_margin")
        if baseline_margin is not None and candidate_margin is not None:
            delta = float(candidate_margin) - float(baseline_margin)
            margin_deltas.append(delta)
            outcome = "wins" if delta > 0 else "regressions" if delta < 0 else "ties"
            margin_outcomes[outcome] += 1
        baseline_rank = baseline_value.get("rank")
        candidate_rank = candidate_value.get("rank")
        if baseline_rank is not None and candidate_rank is not None:
            delta = float(baseline_rank) - float(candidate_rank)
            rank_deltas.append(delta)
            outcome = "wins" if delta > 0 else "regressions" if delta < 0 else "ties"
            rank_outcomes[outcome] += 1
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_capture_count": max(
            sum(margin_outcomes.values()), sum(rank_outcomes.values())
        ),
        "named_hard_negative_margin_delta": _numeric_stats(margin_deltas),
        "margin_delta_sign_stability": _margin_sign_stability(margin_deltas),
        "margin_outcomes": margin_outcomes,
        "gold_rank_improvement": _numeric_stats(rank_deltas),
        "rank_outcomes": rank_outcomes,
    }


def summarize_followup(
    rows: list[dict[str, Any]], window_widths: tuple[int, ...]
) -> dict[str, Any]:
    legacy = "legacy_flat_global_top4x4x4"
    anchor = "anchor_global_top4x4x4"
    method_names = (
        legacy,
        anchor,
        *(f"anchor_local_window{width}_top4x4x4" for width in window_widths),
    )
    methods: dict[str, Any] = {}
    for name in method_names:
        values = [
            row["methods"][name] for row in rows if name in (row.get("methods") or {})
        ]
        named_hard_margins = [
            float(value["relevant_vs_named_hard_negative_margin"])
            for value in values
            if value.get("relevant_vs_named_hard_negative_margin") is not None
        ]
        all_non_gold_margins = [
            float(value["gold_margin"])
            for value in values
            if value.get("gold_margin") is not None
        ]
        ranks = [
            float(value["rank"]) for value in values if value.get("rank") is not None
        ]
        accepted = [
            value
            for value in values
            if value.get("top1_margin") is not None
            and float(value["top1_margin"]) >= _CLEAR_SEPARATION_MARGIN
        ]
        abstained = [
            value
            for value in values
            if value.get("top1_margin") is None
            or float(value["top1_margin"]) < _CLEAR_SEPARATION_MARGIN
        ]
        correct_accepted = sum(bool(value["top1"]) for value in accepted)
        false_accepted = len(accepted) - correct_accepted
        correct_abstained = sum(bool(value["top1"]) for value in abstained)
        wrong_abstained = len(abstained) - correct_abstained
        named_clear_count = sum(
            margin >= _CLEAR_SEPARATION_MARGIN for margin in named_hard_margins
        )
        methods[name] = {
            "capture_count": len(values),
            "gold_rank": _numeric_stats(ranks),
            "clear_separation_threshold": _CLEAR_SEPARATION_MARGIN,
            "named_hard_negative": {
                "margin": _numeric_stats(named_hard_margins),
                "margin_sign_stability": _margin_sign_stability(named_hard_margins),
                "clear_separation_count": named_clear_count,
                "clear_separation_rate": (
                    named_clear_count / len(named_hard_margins)
                    if named_hard_margins
                    else 0.0
                ),
                "positive_but_not_clear_count": sum(
                    0 < margin < _CLEAR_SEPARATION_MARGIN
                    for margin in named_hard_margins
                ),
            },
            "all_non_gold": {
                "gold_margin": _numeric_stats(all_non_gold_margins),
                "mean_gold_margin": (
                    statistics.fmean(all_non_gold_margins)
                    if all_non_gold_margins
                    else None
                ),
                "minimum_gold_margin": (
                    min(all_non_gold_margins) if all_non_gold_margins else None
                ),
                "positive_gold_margin_count": sum(
                    margin > 0 for margin in all_non_gold_margins
                ),
                "clear_gold_margin_count": sum(
                    margin >= _CLEAR_SEPARATION_MARGIN
                    for margin in all_non_gold_margins
                ),
            },
            "raw_top1_gap_selector": {
                "acceptance_basis": "actual_top1_minus_top2_gap",
                "minimum_gap": _CLEAR_SEPARATION_MARGIN,
                "accepted_count": len(accepted),
                "correct_accepted": correct_accepted,
                "false_accepted": false_accepted,
                "precision": (correct_accepted / len(accepted) if accepted else None),
                "coverage": len(accepted) / len(values) if values else 0.0,
                "correct_abstained": correct_abstained,
                "wrong_abstained": wrong_abstained,
            },
        }
    comparisons = {
        "anchor_global_vs_legacy_flat": _paired_method_comparison(
            rows, baseline=legacy, candidate=anchor
        ),
        "windows_vs_anchor_global": {
            str(width): _paired_method_comparison(
                rows,
                baseline=anchor,
                candidate=f"anchor_local_window{width}_top4x4x4",
            )
            for width in window_widths
        },
    }
    per_case: dict[str, Any] = {}
    for case_id in dict.fromkeys(str(row["case_id"]) for row in rows):
        case_rows = [row for row in rows if row["case_id"] == case_id]
        per_case[case_id] = {
            "anchor_global_vs_legacy_flat": _paired_method_comparison(
                case_rows, baseline=legacy, candidate=anchor
            ),
            "windows_vs_anchor_global": {
                str(width): _paired_method_comparison(
                    case_rows,
                    baseline=anchor,
                    candidate=f"anchor_local_window{width}_top4x4x4",
                )
                for width in window_widths
            },
        }
    return {
        "methods": methods,
        "paired_comparisons": comparisons,
        "per_case_paired_comparisons": per_case,
        "repetition_stability": summarize_repetitions(rows),
    }


def separation_diagnosis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    legacy_name = "legacy_global_all_roles_top4x4x4"
    anchor_name = "anchor_global_top4x4x4"
    local_name = "anchor_local_window16_top4x4x4"
    trajectory_global_name = "trajectory_diagnostic_global_top4x4x4"
    trajectory_local_name = "trajectory_diagnostic_local_window16_top4x4x4"
    decisions: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("hard_negative_paths"):
            continue
        names = (
            legacy_name,
            anchor_name,
            local_name,
            trajectory_global_name,
            trajectory_local_name,
        )
        margins = {
            name: (row["methods"].get(name) or {}).get(
                "relevant_vs_named_hard_negative_margin"
            )
            for name in names
        }
        legacy = margins[legacy_name]
        anchor = margins[anchor_name]
        local = margins[local_name]
        clear_separation = {
            name: (
                margins[name] is not None
                and float(margins[name]) >= _CLEAR_SEPARATION_MARGIN
            )
            for name in names
        }
        below_clear_separation = {
            name: (
                margins[name] is not None
                and 0 < float(margins[name]) < _CLEAR_SEPARATION_MARGIN
            )
            for name in names
        }
        legacy_separates = clear_separation[legacy_name]
        role_separates = clear_separation[anchor_name]
        local_separates = clear_separation[local_name]
        role_delta = (
            float(anchor) - float(legacy)
            if anchor is not None and legacy is not None
            else None
        )
        local_delta = (
            float(local) - float(anchor)
            if local is not None and anchor is not None
            else None
        )
        if legacy_separates:
            verdict = "legacy_all_role_global_already_separates"
        elif role_separates:
            verdict = "role_gating_separates"
        elif local_separates and role_delta is not None and role_delta > 0:
            verdict = "combined_role_gating_and_local_windows_separate"
        elif local_separates:
            verdict = "local_windows_separate"
        elif any(below_clear_separation.values()):
            verdict = "positive_but_below_clear_separation_margin"
        else:
            verdict = "neither_separates"
        decisions.append(
            {
                "case_id": row["case_id"],
                "repetition": row.get("repetition"),
                "verdict": verdict,
                "relevant_vs_named_hard_negative_margins": margins,
                "clear_separation_threshold": _CLEAR_SEPARATION_MARGIN,
                "clear_separation_by_method": clear_separation,
                "positive_but_below_clear_separation_by_method": below_clear_separation,
                "legacy_separates": legacy_separates,
                "role_gating_separates": role_separates,
                "anchor_local_windows_separate": local_separates,
                "role_gating_margin_delta_vs_legacy": role_delta,
                "local_window_margin_delta_vs_anchor_global": local_delta,
            }
        )
    return decisions


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
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument(
        "--paired-role-window-followup",
        "--legacy-flat-capture",
        dest="paired_role_window_followup",
        action="store_true",
        help="Opt into paired historical-flat and role-separated v3 captures",
    )
    parser.add_argument(
        "--window-widths",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="Source-local anchor window widths for the opt-in v3 report",
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    window_widths = tuple(dict.fromkeys(int(width) for width in args.window_widths))
    if not window_widths or any(width < 1 for width in window_widths):
        raise ValueError("--window-widths must contain positive integers")
    if args.paired_role_window_followup:
        if args.fixture is None or args.knowledge_root is None:
            raise ValueError(
                "Paired role/window follow-up requires --fixture and --knowledge-root"
            )
        if args.case or args.max_cases:
            raise ValueError(
                "Paired role/window follow-up requires the complete fixture; "
                "--case and --max-cases are not allowed"
            )

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
    if args.paired_role_window_followup:
        cases = fixture_cases(
            args.fixture,
            bank_paths,
            tokenizer=tokenizer,
            knowledge_root=args.knowledge_root,
            require_source_backed_trajectory=True,
        )
        if len(cases) != 6:
            raise ValueError(
                f"Paired role/window follow-up requires exactly six cases, got {len(cases)}"
            )
        if any(not case.gold_path or not case.hard_negative_paths for case in cases):
            raise ValueError(
                "Every follow-up case requires a gold and named hard negatives"
            )
    else:
        cases = build_cases(
            args.reflection_state, args.trace, bank_paths, tuple(args.case)
        )
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
    total_pairs = len(cases) * args.repetitions
    capture_index = 0
    if args.paired_role_window_followup:
        for case in cases:
            for repetition in range(1, args.repetitions + 1):
                capture_index += 1
                captured_at = time.perf_counter()
                nonce_ids = _encode(tokenizer, f"\n[QK-EVAL {time.time_ns()}]\n")
                legacy_q, _legacy_sketch, legacy_spans, legacy_capture = capture_query(
                    args.base_url,
                    tokenizer,
                    cognition_ids,
                    case,
                    max_prompt_tokens=args.max_prompt_tokens,
                    num_query_heads=num_query_heads,
                    head_dim=head_dim,
                    timeout=args.timeout,
                    capture_plan="legacy_flat",
                    nonce_ids=nonce_ids,
                )
                role_q, _role_sketch, role_spans, role_capture = capture_query(
                    args.base_url,
                    tokenizer,
                    cognition_ids,
                    case,
                    max_prompt_tokens=args.max_prompt_tokens,
                    num_query_heads=num_query_heads,
                    head_dim=head_dim,
                    timeout=args.timeout,
                    capture_plan="role",
                    nonce_ids=nonce_ids,
                )
                ranked = rank_paired_case(
                    tokenizer,
                    case,
                    legacy_q,
                    legacy_spans,
                    role_q,
                    role_spans,
                    pages,
                    device,
                    window_widths,
                )
                ranked["repetition"] = repetition
                ranked["capture_pair"] = {
                    "shared_nonce_token_ids": list(nonce_ids),
                    "legacy_flat": legacy_capture,
                    "role_separated": role_capture,
                }
                ranked["elapsed_seconds"] = round(time.perf_counter() - captured_at, 3)
                rows.append(ranked)
                ranks = {
                    method: value["rank"] for method, value in ranked["methods"].items()
                }
                print(
                    f"[{capture_index}/{total_pairs}] {case.case_id[:52]} "
                    f"repetition={repetition} paired_ranks={ranks}",
                    flush=True,
                )
        report = {
            "schema": "qwen-exo-model-native-qk-experiment-v3",
            "mode": "paired_role_window_followup",
            "source_digest": source_digest,
            "model_path": str(args.model_path),
            "bank_path": str(args.bank),
            "knowledge_root": str(args.knowledge_root),
            "case_count": len(cases),
            "paired_capture_count": len(rows),
            "capture_count": len(rows) * 2,
            "repetitions": args.repetitions,
            "page_count": len(pages),
            "cognition_tokens": len(cognition_ids),
            "window_widths": list(window_widths),
            "clear_separation_threshold": _CLEAR_SEPARATION_MARGIN,
            "dimensions": {
                "grouped_kv_heads": int(
                    pages[0].keys.shape[1] * pages[0].keys.shape[2]
                ),
                "query_heads": num_query_heads,
                "head_dim": head_dim,
            },
            "capture_plans": {
                "legacy_flat": {
                    "labels": [
                        "ORIGINAL TASK",
                        "CURRENT USER REQUEST (only when distinct)",
                        "RECENT EXECUTION TRAJECTORY",
                    ],
                    "truncation": "whole encoded flat text tail-truncated to capacity",
                    "states": "evenly partitioned into up to 8 untyped spans",
                },
                "role_separated": {
                    "max_states": 8,
                    "anchor_roles": ["original_task", "current_user"],
                    "trajectory_max_fraction": 0.25,
                    "trajectory_max_states": 2,
                },
                "pairing": "identical cognition prefix and nonce token IDs per pair",
            },
            "role_window_methods": {
                "legacy_flat_global_top4x4x4": (
                    "Historical flat capture: Top-4 head pairs x Top-4 global "
                    "searchable tokens x Top-4 untyped query states"
                ),
                "anchor_global_top4x4x4": (
                    "Role capture: Top-4 head pairs x Top-4 global searchable "
                    "tokens x Top-4 anchor states"
                ),
                **{
                    f"anchor_local_window{width}_top4x4x4": (
                        "Role capture: Top-4 head pairs x Top-4 finite supports "
                        f"in a source-local width-{width} window x Top-4 anchor states"
                    )
                    for width in window_widths
                },
            },
            "summary": summarize_followup(rows, window_widths),
            "bounded_source_attribution": [
                {
                    "case_id": case.case_id,
                    "trajectory_source": case.trajectory_source,
                    "trajectory_provenance": case.trajectory_provenance,
                }
                for case in cases
            ],
            "rows": rows,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    else:
        for case in cases:
            for repetition in range(1, args.repetitions + 1):
                capture_index += 1
                captured_at = time.perf_counter()
                full_q, sketch_q, spans, capture = capture_query(
                    args.base_url,
                    tokenizer,
                    cognition_ids,
                    case,
                    max_prompt_tokens=args.max_prompt_tokens,
                    num_query_heads=num_query_heads,
                    head_dim=head_dim,
                    timeout=args.timeout,
                )
                ranked = rank_case(
                    tokenizer, case, full_q, sketch_q, spans, pages, device
                )
                ranked["repetition"] = repetition
                ranked["capture"] = capture
                ranked["elapsed_seconds"] = round(time.perf_counter() - captured_at, 3)
                rows.append(ranked)
                ranks = {
                    method: value["rank"] for method, value in ranked["methods"].items()
                }
                print(
                    f"[{capture_index}/{total_pairs}] {case.case_id[:52]} "
                    f"repetition={repetition} ranks={ranks}",
                    flush=True,
                )
        report = {
            "schema": "qwen-exo-model-native-qk-experiment-v2",
            "source_digest": source_digest,
            "model_path": str(args.model_path),
            "bank_path": str(args.bank),
            "case_count": len(cases),
            "capture_count": len(rows),
            "repetitions": args.repetitions,
            "labeled_case_count": sum(bool(case.gold_path) for case in cases),
            "negative_case_count": sum(not case.gold_path for case in cases),
            "page_count": len(pages),
            "cognition_tokens": len(cognition_ids),
            "dimensions": {
                "baseline": 32,
                "mean_head": 256,
                "grouped_kv_heads": int(
                    pages[0].keys.shape[1] * pages[0].keys.shape[2]
                ),
                "query_heads": int(full_q.shape[1]),
                "head_dim": int(full_q.shape[2]),
            },
            "role_window_methods": {
                "legacy_global_all_roles_top4x4x4": "Top-4 head pairs x Top-4 searchable tokens x Top-4 all-role query states",
                "anchor_global_top4x4x4": "Top-4 head pairs x Top-4 searchable tokens x Top-4 anchor query states",
                "anchor_local_window16_top4x4x4": "Top-4 head pairs x Top-4 finite searchable supports in a source-local width-16 window x Top-4 anchor states",
                "trajectory_diagnostic_global_top4x4x4": "Trajectory-only global diagnostic; never an origin gate",
                "trajectory_diagnostic_local_window16_top4x4x4": "Trajectory-only source-local diagnostic; never an origin gate",
            },
            "summary": summarize(rows),
            "repetition_summary": summarize_repetitions(rows),
            "role_window_separation": separation_diagnosis(rows),
            "candidate_consensus_gate": summarize_consensus_gate(rows),
            "negative_observations": [
                {
                    "case_id": row["case_id"],
                    "repetition": row["repetition"],
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
