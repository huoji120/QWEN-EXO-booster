from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from qwen_exo_booster.contracts import stable_digest


@dataclass(frozen=True, slots=True)
class CanonicalDecisionContext:
    """An immutable, lossless request snapshot shared by internal decisions.

    ``token_ids`` count the canonical data alone, without chat-template or
    decision-specific overhead. Consumers must fit the complete rendered prompt
    to the serving input cap, never a separate fixed question budget. A digest
    identifies the snapshot, not its cache namespace: appending history must not
    change an authenticated conversation's ``lineage_namespace``.
    """

    text: str
    token_ids: tuple[int, ...]
    digest: str
    lineage_namespace: str

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Unsupported decision context value: {type(value).__name__}")


def _request_field(request: Any, name: str, default: Any = None) -> Any:
    if isinstance(request, Mapping):
        return request.get(name, default)
    return getattr(request, name, default)


def build_decision_context(
    request: Any,
    tokenizer: Any,
    *,
    original_task: str | None = None,
    compaction_context: str | None = None,
    input_items: Any = None,
    lineage_namespace: str | None = None,
) -> CanonicalDecisionContext:
    """Serialize the complete effective request, without filtering or clipping.

    The input array retains role, item order, content-block and tool-call/result
    boundaries, including structured tool results. JSON framing keeps supplied
    data separate from the evaluator's instructions; consumers must explicitly
    treat all embedded tool output as untrusted. ``input_items`` is the caller's
    authoritative resolved history for a stateful continuation, not a summary.
    Compaction context can only preserve the summary actually available; this
    function never fabricates missing pre-compaction history.

    A missing lineage namespace isolates cache use to this request. Callers may
    supply an authenticated stable tenant/conversation namespace, never a hash
    of the evolving request history or an unverified client session label.
    """
    payload = {
        "schema": "qwen-exo-decision-context-v1",
        "instructions": _request_field(request, "instructions"),
        "tools": _request_field(request, "tools"),
        "tool_choice": _request_field(request, "tool_choice"),
        "parallel_tool_calls": _request_field(request, "parallel_tool_calls"),
        "input": (
            _request_field(request, "input") if input_items is None else input_items
        ),
        "original_task": original_task,
        "compaction_context": compaction_context,
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_value,
    )
    token_ids = tuple(tokenizer.encode(text, add_special_tokens=False))
    if not lineage_namespace:
        request_id = _request_field(request, "request_id")
        if not request_id:
            raise ValueError("Decision context requires a request ID or lineage namespace")
        lineage_namespace = "request:" + str(request_id)
    return CanonicalDecisionContext(
        text=text,
        token_ids=token_ids,
        digest=stable_digest(text),
        lineage_namespace=str(lineage_namespace),
    )
