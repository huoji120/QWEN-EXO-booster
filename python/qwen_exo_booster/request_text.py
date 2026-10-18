"""Plain-text views of Responses API input items.

Used for task tracking, capsules, reflection rows, score-bias user queries and
compaction. (Moved verbatim from the retired retrieval pipeline.)
"""

from __future__ import annotations

import json
from typing import Any


def latest_user_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return str(value or "").strip()
    for raw_item in reversed(value):
        item = raw_item.model_dump() if hasattr(raw_item, "model_dump") else raw_item
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") in {
            "function_call_output",
            "computer_call_output",
        }:
            continue
        role = str(item.get("role") or "")
        if role != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            pieces = []
            for part in content:
                part = part.model_dump() if hasattr(part, "model_dump") else part
                if isinstance(part, dict):
                    text = part.get("text") or part.get("input_text")
                    if text:
                        pieces.append(str(text))
            if pieces:
                return "\n".join(pieces).strip()
    return ""


def first_user_text(value: Any) -> str:
    if not isinstance(value, list):
        return latest_user_text(value)
    for raw_item in value:
        item = raw_item.model_dump() if hasattr(raw_item, "model_dump") else raw_item
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        text = latest_user_text([item])
        if text:
            return text
    return ""


def bounded_text(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars < 1 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n...[bounded]...\n{text[-tail:]}"


def response_item_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(
            text for item in value if (text := response_item_text(item).strip())
        )
    if isinstance(value, dict):
        for key in ("text", "output_text", "content", "summary"):
            if key in value:
                text = response_item_text(value[key]).strip()
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def trajectory_context(value: Any, *, max_chars: int = 12000) -> str:
    if not isinstance(value, list) or max_chars < 1:
        return ""
    rows_reversed = []
    remaining = int(max_chars)
    for raw_item in reversed(value):
        item = raw_item.model_dump() if hasattr(raw_item, "model_dump") else raw_item
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        role = str(item.get("role") or "")
        if role in {"user", "system", "developer"}:
            continue
        if item_type in {"function_call_output", "computer_call_output"}:
            label = "TOOL OBSERVATION"
            content = item.get("output", item.get("content"))
        elif item_type in {"function_call", "computer_call"}:
            label = "TOOL ACTION"
            content = {
                "name": item.get("name"),
                "arguments": item.get("arguments"),
            }
        elif role == "assistant" or item_type in {
            "reasoning",
            "output_text",
            "message",
        }:
            label = "ASSISTANT TRAJECTORY"
            content = item.get("content")
            if content is None:
                content = item.get("summary", item.get("text"))
        else:
            continue
        text = response_item_text(content).strip()
        if (
            item_type == "message"
            and role == "assistant"
            and text.startswith("<context_compaction>\n")
            and text.endswith("\n</context_compaction>")
        ):
            continue
        if not text:
            continue
        row = f"{label}:\n{text}"
        if len(row) > remaining:
            row = row[-remaining:]
        rows_reversed.append(row)
        remaining -= len(row)
        if remaining <= 0:
            break
    return "\n\n".join(reversed(rows_reversed))


def request_question(
    value: Any,
    *,
    original_task: str | None = None,
    compaction_context: str | None = None,
) -> str:
    extracted_original = first_user_text(value)
    current = latest_user_text(value)
    original = str(original_task or extracted_original).strip()
    compacted = str(compaction_context or "").strip()
    trajectory = trajectory_context(value)

    if not compacted and not trajectory and original and current == original:
        return original

    parts = []
    if compacted:
        parts.append(f"COMPACTED RESPONSE CONTEXT:\n{compacted}")
    if original:
        parts.append("ORIGINAL TASK:\n" f"{bounded_text(original, 8000)}")
    if current and current != original:
        parts.append("CURRENT USER REQUEST:\n" f"{bounded_text(current, 4000)}")
    if trajectory:
        parts.append(f"RECENT EXECUTION TRAJECTORY:\n{trajectory}")
    return "\n\n".join(parts) or current or original


__all__ = [
    "bounded_text",
    "first_user_text",
    "latest_user_text",
    "request_question",
    "response_item_text",
    "trajectory_context",
]
