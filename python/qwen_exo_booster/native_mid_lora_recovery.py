"""Two causal prefixes for overlength self-Think examples, without corpus I/O.

Generation retains the original evidence and historical reasoning. Training first
hides tool-result bodies and then removes complete oldest unprotected units. All
reported indices refer to the original source, and audit fields contain hashes,
counts, and indices rather than source text.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any, Sequence

from .native_mid_lora_self_think import (
    PreparedExample,
    _hash,
    _ids,
    _render,
    clean_message,
    prepare_example,
)


TOOL_OUTPUT_HIDDEN = "[tool output hidden for training]"
_TEXT_BLOCKS = frozenset({"text", "input_text", "output_text"})
_REASONING_BLOCKS = frozenset({"thinking", "reasoning", "redacted_thinking"})
_PREFIX_OVERFLOWS = frozenset({
    "complete_prefix_insufficient_think_budget",
    "ordinary_complete_prefix_overflow",
})


def _reject_constant(value: str) -> None:
    raise ValueError("nonfinite_json_constant")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _generation_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize known text/reasoning shapes without the training Think cleaner."""
    result = deepcopy(message)
    reasoning: list[str] = []
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = result.pop(key, None)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError("source_reasoning_not_text")
            if value:
                reasoning.append(value)
    content = result.get("content")
    if isinstance(content, list):
        text: list[str] = []
        for block in content:
            kind = block.get("type")
            if kind in _TEXT_BLOCKS:
                text.append(block["text"])
            elif result["role"] == "assistant" and kind in _REASONING_BLOCKS:
                # Opaque/redacted blocks cannot be reconstructed as native text.
                # Exclude instead of silently deleting their historical evidence.
                values = [block[key] for key in ("text", "thinking", "reasoning")
                          if key in block]
                if not values or any(not isinstance(value, str) for value in values):
                    raise ValueError("source_reasoning_block_unsupported")
                reasoning.extend(value for value in values if value)
            else:
                raise ValueError("source_nontext_block_unsupported")
        result["content"] = "".join(text)
    if reasoning:
        # Native templates consume assistant reasoning_content. For other roles,
        # keep alias fields verbatim rather than pretend the template uses them.
        if result["role"] != "assistant":
            for key in ("reasoning_content", "thinking", "reasoning"):
                if key in message:
                    result[key] = deepcopy(message[key])
        else:
            result["reasoning_content"] = "\n".join(reasoning)
    return result


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _object_bodies(text: str, start: int, end: int) -> list[tuple[int, int]] | None:
    """Locate root body values, preserving every other byte of the envelope."""
    if text[start:start + 1] != "{":
        return None
    index = _skip_space(text, start + 1)
    seen: set[str] = set()
    bodies: list[tuple[int, int]] = []
    while index < end and text[index] != "}":
        key, index = _DECODER.raw_decode(text, index)
        if not isinstance(key, str) or key in seen:
            return None
        seen.add(key)
        index = _skip_space(text, index)
        if text[index:index + 1] != ":":
            return None
        value_start = _skip_space(text, index + 1)
        value, index = _DECODER.raw_decode(text, value_start)
        if key in {"output", "content"} and value != TOOL_OUTPUT_HIDDEN:
            bodies.append((value_start, index))
        index = _skip_space(text, index)
        if text[index:index + 1] == ",":
            index = _skip_space(text, index + 1)
        else:
            break
    return bodies if text[index:index + 1] == "}" and index + 1 == end else None


def _wrappers(text: str, tag: str):
    """Yield valid JSON envelopes; an unknown/malformed wrapper stays untouched."""
    opening, closing = f"<{tag}>", f"</{tag}>"
    cursor = 0
    while True:
        begin = text.find(opening, cursor)
        if begin < 0:
            return
        start = _skip_space(text, begin + len(opening))
        try:
            value, end = _DECODER.raw_decode(text, start)
            close_start = _skip_space(text, end)
            if text.startswith(closing, close_start):
                yield start, end, value
                cursor = close_start + len(closing)
                continue
        except (ValueError, RecursionError):
            pass
        # Do not reinterpret markers inside an invalid enclosing wrapper.
        close_start = text.find(closing, start)
        if close_start < 0:
            return
        cursor = close_start + len(closing)


def _hide_serialized_bodies(text: str) -> tuple[str, int]:
    spans: list[tuple[int, int]] = []
    for start, end, value in _wrappers(text, "tool_response"):
        if not isinstance(value, dict):
            continue
        try:
            bodies = _object_bodies(text, start, end)
        except (ValueError, RecursionError):
            continue
        if bodies:
            spans.extend(bodies)
    if not spans:
        return text, 0
    replacement = json.dumps(TOOL_OUTPUT_HIDDEN, ensure_ascii=False)
    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.extend((text[cursor:start], replacement))
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks), len(spans)


def _has_wrapper(message: dict[str, Any], tag: str) -> bool:
    content = message.get("content")
    return isinstance(content, str) and any(
        isinstance(value, dict) for _, _, value in _wrappers(content, tag))


def _units(history: Sequence[dict[str, Any]]) -> list[tuple[int, ...]]:
    units: list[tuple[int, ...]] = []
    index = 0
    while index < len(history):
        start = index
        message = history[index]
        index += 1
        if message["role"] == "assistant":
            if message.get("tool_calls"):
                while index < len(history) and history[index]["role"] == "tool":
                    index += 1
            if _has_wrapper(message, "tool_call"):
                while (index < len(history) and history[index]["role"] == "user"
                       and _has_wrapper(history[index], "tool_response")):
                    index += 1
        units.append(tuple(range(start, index)))
    return units


def prepare_recovered_example(
    tokenizer,
    messages: Sequence[dict[str, Any]],
    target_index: int,
    *,
    generation_context_tokens: int,
    total_tokens: int = 32768,
    max_think_tokens: int = 8192,
    min_think_tokens: int = 1,
    provenance: dict[str, Any] | None = None,
) -> PreparedExample:
    """Prepare an immutable-source recovery; never truncate generation or target."""
    if (type(total_tokens) is not int or not 2 <= total_tokens <= 32768
            or type(max_think_tokens) is not int or type(min_think_tokens) is not int
            or min_think_tokens < 1 or max_think_tokens < min_think_tokens):
        raise ValueError("token_budget_invalid")
    if type(generation_context_tokens) is not int or generation_context_tokens < 1:
        raise ValueError("generation_context_budget_invalid")
    if (type(target_index) is not int or not 0 < target_index < len(messages)
            or not isinstance(messages[target_index], dict)
            or messages[target_index].get("role") != "assistant"):
        raise ValueError("target_not_causal_assistant")
    # Snapshot only the causal source. Future messages are neither read nor hashed.
    original = deepcopy(messages[:target_index + 1])
    cleaned = [clean_message(message) for message in original]
    history, target = cleaned[:-1], cleaned[-1]
    users = [index for index, message in enumerate(history) if message["role"] == "user"]
    if not users:
        raise ValueError("target_without_user_root")
    if not ((target.get("content") or "").strip() or target.get("tool_calls")):
        raise ValueError("target_answer_empty")
    original_hash = _hash(cleaned)
    generation_history = [_generation_message(message) for message in original[:-1]]
    generation_prompt = _render(tokenizer, generation_history, thinking=True, generation=True)
    if not generation_prompt.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("template_think_prompt_mismatch")
    generation_ids = _ids(tokenizer, generation_prompt)
    minimum = max(min_think_tokens, len(_ids(tokenizer, "\n</think>")))
    generation_room = generation_context_tokens - len(generation_ids)
    if generation_room < minimum:
        raise ValueError("generation_complete_prefix_overflow")
    reserve = min(max_think_tokens, generation_room)
    if reserve < minimum:
        raise ValueError("token_budget_invalid")

    # Derive the original cleaned target's exact suffix without subjecting its
    # original full history to the training limit or admitting it to generation.
    original_prompt = _render(tokenizer, history, thinking=True, generation=True)
    original_full = _render(tokenizer, cleaned, thinking=True, generation=False)
    if (not original_full.startswith(original_prompt + "\n</think>\n\n")
            or not original_full.endswith("<|im_end|>\n")):
        raise ValueError("template_answer_boundary_mismatch")
    original_suffix = _ids(tokenizer, original_full[len(original_prompt + "\n</think>"):])
    if len(original_suffix) >= total_tokens:
        raise ValueError("answer_exceeds_total_budget")

    training = deepcopy(history)
    hidden_indices: list[int] = []
    hidden_count = 0
    for index, message in enumerate(training):
        content = message.get("content")
        count = 0
        if message["role"] == "tool" and content is not None:
            if content != TOOL_OUTPUT_HIDDEN:
                message["content"] = TOOL_OUTPUT_HIDDEN
                count = 1
        elif message["role"] == "user" and isinstance(content, str):
            message["content"], count = _hide_serialized_bodies(content)
        if count:
            hidden_indices.append(index)
            hidden_count += count

    units = _units(history)
    protected = {users[0], users[-1], len(history) - 1}
    protected.update(index for index, message in enumerate(history) if message["role"] == "system")
    removable = [unit for unit in units if not protected.intersection(unit)]

    def candidate(remove_count: int) -> tuple[PreparedExample | None, list[int]]:
        removed = {index for unit in removable[:remove_count] for index in unit}
        kept = [index for index in range(len(training)) if index not in removed]
        compact = [training[index] for index in kept] + [target]
        try:
            example = prepare_example(tokenizer, compact, len(kept), total_tokens=total_tokens,
                                      max_think_tokens=reserve, min_think_tokens=min_think_tokens)
        except ValueError as error:
            if str(error) not in _PREFIX_OVERFLOWS:
                raise
            return None, kept
        if example.answer_suffix_ids != original_suffix:
            raise ValueError("recovery_answer_suffix_changed")
        return example, kept

    example, kept = candidate(0)
    removed_count = 0
    if example is None or example.allowed_think_tokens < reserve:
        core, core_kept = candidate(len(removable))
        if core is None:
            raise ValueError("recovery_protected_core_overflow")
        if core.allowed_think_tokens < reserve:
            # The protected task/system/current-turn core cannot reserve maxThink.
            # Keep that core and its exact smaller budget; never cut its messages.
            example, kept, removed_count = core, core_kept, len(removable)
        else:
            low, high = 1, len(removable)
            while low < high:
                middle = (low + high) // 2
                trial, _ = candidate(middle)
                if trial is not None and trial.allowed_think_tokens >= reserve:
                    high = middle
                else:
                    low = middle + 1
            removed_count = low
            # Verify the selected exact rendering rather than trusting estimates
            # or token-count additivity across template/BPE boundaries.
            example, kept = candidate(removed_count)
            if example is None or example.allowed_think_tokens < reserve:
                raise ValueError("recovery_protected_core_overflow")

    if (len(example.question_input_ids) + example.allowed_think_tokens
            + len(example.answer_suffix_ids) > total_tokens
            or len(example.ordinary_input_ids) > total_tokens):
        raise ValueError("recovery_protected_core_overflow")
    if len(generation_ids) + example.allowed_think_tokens > generation_context_tokens:
        raise ValueError("generation_complete_prefix_overflow")
    removed_indices = sorted(index for unit in removable[:removed_count] for index in unit)
    metadata = deepcopy(provenance or {})
    metadata.update(example.provenance)
    metadata.update(
        sample_hash=original_hash,
        target_index=target_index,
        original_target_index=target_index,
        training_sample_hash=example.provenance["sample_hash"],
        generation_question_sha256=_hash(generation_ids),
        generation_question_tokens=len(generation_ids),
        generation_context_tokens=generation_context_tokens,
        generation_prefix_policy="full_original_history_no_truncation",
        generation_history_reasoning_policy="preserve_text_normalize_aliases",
        prefix_policy="tool_body_hidden_oldest_complete_units_removed",
        hidden_message_indices=hidden_indices,
        hidden_message_count=len(hidden_indices),
        hidden_body_count=hidden_count,
        hidden_messages_sha256=_hash([history[index] for index in hidden_indices]),
        removed_message_indices=removed_indices,
        removed_message_count=len(removed_indices),
        removed_unit_count=removed_count,
        removed_messages_sha256=_hash([history[index] for index in removed_indices]),
        training_kept_message_indices=kept,
        training_target_index=len(kept),
        protected_message_indices=sorted({index for unit in units
                                          if protected.intersection(unit) for index in unit}),
        requested_max_think_tokens=max_think_tokens,
        effective_max_think_tokens=reserve,
    )
    return replace(example, provenance=metadata, generation_input_ids=generation_ids,
                   generation_context_tokens=generation_context_tokens)
