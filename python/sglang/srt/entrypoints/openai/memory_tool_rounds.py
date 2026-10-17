"""Server-executed qwen-exo memory tool rounds for the non-harmony Responses path.

A *round* is: the model ends a turn with only ``qwen_exo_memory_*`` calls, the
server runs them, appends the tool results to the token prompt and lets the
model continue.  This module holds the pieces that do not need the serving
object: splitting parsed calls, building the tool-response token block from the
model's own chat template, and the ``mcp_call`` output items.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import orjson
from openai.types.responses.response_output_item import McpCall
from qwen_exo_booster.memory_tools import MEMORY_SERVER_LABEL, MEMORY_TOOL_NAMES

from sglang.srt.utils import random_uuid

MAX_MEMORY_TOOL_ROUNDS = 4
MEMORY_ROUND_META_KEY = "qwen_exo_server_tool_round"
ROUND_LIMIT_OUTPUT = (
    "Memory tool limit reached for this turn. Continue with what you already "
    "found and answer directly."
)
_SENTINEL = "QWEN_EXO_TOOL_RESULT_SENTINEL_{index}_7f3a"
# Chat-control markers a tool result must never smuggle into the prompt as
# special tokens (ChatML roles, tool/think tags).
_CONTROL_MARKER = re.compile(
    r"<(?=\||/?(?:tool_call|tool_response|think|function|parameter)\b)"
)


def neutralize_control_markers(text: str) -> str:
    """Break chat-control markers inside tool output with a zero-width joiner,
    so file contents cannot close the tool block or open a new role."""
    return _CONTROL_MARKER.sub("<\u200d", text)


def is_memory_tool(name: str | None) -> bool:
    return bool(name) and name in MEMORY_TOOL_NAMES


def split_memory_calls(calls: Iterable[Any]) -> tuple[list[Any], list[Any]]:
    """Partition parsed ``ToolCallItem``s into (memory, client) calls."""
    memory, client = [], []
    for call in calls:
        (memory if is_memory_tool(getattr(call, "name", None)) else client).append(call)
    return memory, client


class ToolResponseTemplate:
    """Token layout for appending tool results after a model turn.

    Derived once per call-count by rendering a sentinel conversation with the
    model's chat template, so the appended block matches what the template
    would render when the client later replays the same turn.
    """

    def __init__(self, tokenizer: Any, template_kwargs: dict[str, Any] | None):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(template_kwargs or {})
        self._layouts: dict[int, tuple[str, str, list[str]]] = {}

    def _layout(self, count: int) -> tuple[str, str, list[str]]:
        """Return (end_marker, bridge, pieces) where the block for outputs
        o_i is ``bridge + pieces[0] + o_0 + pieces[1] + ... + pieces[count]``."""
        cached = self._layouts.get(count)
        if cached is not None:
            return cached
        # Arguments as a dict, as _process_messages hands history tool calls to
        # the template (Qwen templates iterate ``arguments|items``).
        calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": "qwen_exo_memory_search", "arguments": {}},
            }
            for index in range(count)
        ]
        base = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ]
        results = [
            {
                "role": "tool",
                "tool_call_id": f"call_{index}",
                "content": _SENTINEL.format(index=index),
            }
            for index in range(count)
        ]
        before = self.tokenizer.apply_chat_template(
            base, tokenize=False, add_generation_prompt=False, **self.template_kwargs
        )
        after = self.tokenizer.apply_chat_template(
            base + results,
            tokenize=False,
            add_generation_prompt=False,
            **self.template_kwargs,
        )
        if not after.startswith(before):
            raise ValueError("chat template does not render tool results as a suffix")
        end_marker = "<|im_end|>"
        end_at = before.rfind(end_marker)
        if end_at < 0:
            raise ValueError("chat template has no <|im_end|> turn marker")
        bridge = before[end_at + len(end_marker) :]
        middle = after[len(before) :]
        pieces: list[str] = []
        cursor = 0
        for index in range(count):
            sentinel = _SENTINEL.format(index=index)
            at = middle.find(sentinel, cursor)
            if at < 0:
                raise ValueError("chat template dropped a tool result")
            pieces.append(middle[cursor:at])
            cursor = at + len(sentinel)
        pieces.append(middle[cursor:])
        layout = (end_marker, bridge, pieces)
        self._layouts[count] = layout
        return layout

    def continuation_ids(
        self,
        *,
        output_ids: Sequence[int],
        outputs: Sequence[str],
        generation_prompt_ids: Sequence[int],
    ) -> list[int]:
        """Token ids to append after ``output_ids`` so the model sees the tool
        results and starts a fresh assistant turn."""
        end_marker, bridge, pieces = self._layout(len(outputs))
        end_ids = self.tokenizer.encode(end_marker, add_special_tokens=False)
        ids: list[int] = []
        if list(output_ids[-len(end_ids) :]) != list(end_ids):
            ids.extend(end_ids)
        text = bridge + pieces[0]
        for output, piece in zip(outputs, pieces[1:]):
            text += neutralize_control_markers(output) + piece
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        ids.extend(generation_prompt_ids)
        return ids


def generation_prompt_ids(prompt_ids: Sequence[int], im_start_id: int) -> list[int]:
    """The assistant header the original prompt ended with (from the last
    ``<|im_start|>``), reused verbatim for every continuation turn."""
    for index in range(len(prompt_ids) - 1, -1, -1):
        if prompt_ids[index] == im_start_id:
            return list(prompt_ids[index:])
    raise ValueError("prompt has no <|im_start|> assistant header")


def mcp_call_item(call: dict[str, Any]) -> McpCall:
    return McpCall(
        id=call["id"],
        type="mcp_call",
        server_label=MEMORY_SERVER_LABEL,
        name=call["name"],
        arguments=call["arguments"],
        output=call["output"] if call["ok"] else None,
        error=None if call["ok"] else call["output"],
        status="completed" if call["ok"] else "failed",
    )


def new_memory_call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"mcp_{random_uuid()}",
        "name": name,
        "arguments": arguments or "{}",
        "ok": False,
        "output": "",
    }


def mcp_item_to_chat_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Render a replayed ``mcp_call`` input item as the assistant tool call and
    tool result the model originally saw."""
    raw = item.get("arguments")
    if isinstance(raw, dict):
        raw = orjson.dumps(raw).decode("utf-8")
    elif isinstance(raw, str):
        try:
            parsed = orjson.loads(raw) if raw else None
        except orjson.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            raw = "{}"
    else:
        raw = "{}"
    call_id = str(item.get("id") or f"mcp_{random_uuid()}")
    output = item.get("output")
    if not output:
        output = item.get("error") or "(tool returned no textual output)"
    if not isinstance(output, str):
        output = orjson.dumps(output).decode("utf-8")
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": raw},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": neutralize_control_markers(output),
        },
    ]


__all__ = [
    "MAX_MEMORY_TOOL_ROUNDS",
    "MEMORY_ROUND_META_KEY",
    "ROUND_LIMIT_OUTPUT",
    "ToolResponseTemplate",
    "generation_prompt_ids",
    "is_memory_tool",
    "mcp_call_item",
    "mcp_item_to_chat_messages",
    "neutralize_control_markers",
    "new_memory_call",
    "split_memory_calls",
]
