"""Synthetic dual-prefix recovery contracts against a real local tokenizer.

Set NATIVE_MID_LORA_TEST_PROFILE to a local tokenizer profile. These tests
never load a model, contact serving, use CUDA, or inspect a training corpus.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest

from qwen_exo_booster.native_mid_lora_recovery import (
    TOOL_OUTPUT_HIDDEN,
    prepare_recovered_example,
)
from qwen_exo_booster.native_mid_lora_self_think import (
    _hash,
    _render,
    assemble_self_think,
    clean_message,
    example_from_record,
    example_to_record,
    generation_plan,
    load_profile_tokenizer,
    prepare_example,
)


@pytest.fixture(scope="module")
def tokenizer():
    profile = os.environ.get("NATIVE_MID_LORA_TEST_PROFILE")
    if not profile:
        pytest.skip("NATIVE_MID_LORA_TEST_PROFILE is not set")
    if not Path(profile).is_dir():
        pytest.skip("NATIVE_MID_LORA_TEST_PROFILE directory is absent")
    tokenizer, _ = load_profile_tokenizer(profile)
    return tokenizer


def _encode(tokenizer, text):
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _text(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


def _recover(tokenizer, messages, target_index, **kwargs):
    options = {"generation_context_tokens": 131072, "max_think_tokens": 64}
    options.update(kwargs)
    return prepare_recovered_example(tokenizer, messages, target_index, **options)


def _call(identifier, argument):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": identifier,
            "type": "function",
            "function": {"name": "synthetic_lookup", "arguments": {"query": argument}},
        }],
    }


def _wrapped(payload):
    return "<tool_response>" + json.dumps(payload, ensure_ascii=False) + "</tool_response>"


def _wrapper_payloads(text):
    """Parse synthetic wrappers without treating a quoted closing tag as XML."""
    decoder = json.JSONDecoder()
    payloads = []
    offset = 0
    while (start := text.find("<tool_response>", offset)) >= 0:
        start += len("<tool_response>")
        start += len(text[start:]) - len(text[start:].lstrip())
        payload, end = decoder.raw_decode(text, start)
        assert text[end:].lstrip().startswith("</tool_response>")
        payloads.append(payload)
        offset = text.index("</tool_response>", end) + len("</tool_response>")
    return payloads


@pytest.fixture
def native_messages():
    return [
        {"role": "system", "content": "SYNTHETIC_SYSTEM_ANCHOR"},
        {"role": "user", "content": "SYNTHETIC_TASK_ANCHOR"},
        {"role": "assistant", "content": "Historical answer.",
         "reasoning_content": "SYNTHETIC_HISTORICAL_THINK"},
        _call("synthetic-call-17", "SYNTHETIC_UNCHANGED_ARGUMENT"),
        {"role": "tool", "tool_call_id": "synthetic-call-17", "name": "synthetic_lookup",
         "is_error": True, "content": "SYNTHETIC_NATIVE_OUTPUT"},
        {"role": "user", "content": "SYNTHETIC_LATEST_USER"},
        {"role": "assistant", "content": "<think>SYNTHETIC_TARGET_THINK</think>SYNTHETIC_TARGET_ANSWER"},
        {"role": "user", "content": "SYNTHETIC_FUTURE_USER"},
        {"role": "assistant", "content": "SYNTHETIC_FUTURE_ANSWER"},
    ]


def test_hides_native_output_only_in_training_and_keeps_causal_generation(tokenizer, native_messages):
    original = deepcopy(native_messages)
    example = _recover(tokenizer, native_messages, 6)
    expected_training = [clean_message(message) for message in original[:7]]
    expected_training[4]["content"] = TOOL_OUTPUT_HIDDEN
    training_reference = prepare_example(tokenizer, expected_training, 6, max_think_tokens=64)
    assert example.question_input_ids == training_reference.question_input_ids
    train = _text(tokenizer, example.question_input_ids)
    generation = _text(tokenizer, example.generation_input_ids)
    assert "SYNTHETIC_NATIVE_OUTPUT" not in train
    assert TOOL_OUTPUT_HIDDEN in train
    assert "SYNTHETIC_HISTORICAL_THINK" not in train
    assert "SYNTHETIC_NATIVE_OUTPUT" in generation
    assert "SYNTHETIC_HISTORICAL_THINK" in generation
    assert TOOL_OUTPUT_HIDDEN not in generation
    for marker in ("SYNTHETIC_UNCHANGED_ARGUMENT", "SYNTHETIC_SYSTEM_ANCHOR",
                   "SYNTHETIC_TASK_ANCHOR", "SYNTHETIC_LATEST_USER"):
        assert marker in train and marker in generation
    for marker in ("SYNTHETIC_TARGET_THINK", "SYNTHETIC_TARGET_ANSWER",
                   "SYNTHETIC_FUTURE_USER", "SYNTHETIC_FUTURE_ANSWER"):
        assert marker not in train and marker not in generation
    expected_generation = _encode(tokenizer, _render(
        tokenizer, original[:6], thinking=True, generation=True))
    assert example.generation_input_ids == expected_generation
    assert generation_plan(example).question_input_ids == expected_generation
    assert native_messages == original
    baseline = prepare_example(tokenizer, original, 6, max_think_tokens=64)
    assert example.answer_suffix_ids == baseline.answer_suffix_ids
    assert example.answer_suffix_labels == baseline.answer_suffix_labels
    assert "SYNTHETIC_TARGET_ANSWER" in _text(tokenizer, example.answer_suffix_ids)
    assert "SYNTHETIC_TARGET_THINK" not in _text(tokenizer, example.answer_suffix_ids)
    metadata = example.provenance
    canonical_hash = _hash([clean_message(message) for message in original[:7]])
    assert metadata["sample_hash"] == canonical_hash
    assert metadata["original_target_index"] == metadata["target_index"] == 6
    assert metadata["generation_question_sha256"] == _hash(expected_generation)
    assert metadata["generation_question_tokens"] == len(expected_generation)
    assert metadata["generation_prefix_policy"] == "full_original_history_no_truncation"
    assert metadata["hidden_message_indices"] == [4]
    assert metadata["hidden_body_count"] == 1
    assert metadata["removed_message_indices"] == []
    assert metadata["removed_unit_count"] == 0
    assert isinstance(metadata["hidden_messages_sha256"], str)
    assert isinstance(metadata["removed_messages_sha256"], str)


@pytest.mark.parametrize("alias", ["reasoning_content", "thinking", "reasoning"])
def test_historical_reasoning_aliases_keep_text_in_full_generation(tokenizer, alias):
    messages = [
        {"role": "user", "content": "Synthetic original task."},
        {"role": "assistant", "content": "Synthetic earlier answer.", alias: "SYNTHETIC_ALIAS_THINK"},
        {"role": "user", "content": "Synthetic continuation."},
        {"role": "assistant", "content": "Synthetic target."},
    ]
    original = deepcopy(messages)
    expected = deepcopy(messages[:3])
    reasoning = expected[1].pop(alias)
    expected[1]["reasoning_content"] = reasoning
    example = _recover(tokenizer, messages, 3)
    assert example.generation_input_ids == _encode(tokenizer, _render(
        tokenizer, expected, thinking=True, generation=True))
    assert "SYNTHETIC_ALIAS_THINK" in _text(tokenizer, example.generation_input_ids)
    assert "SYNTHETIC_ALIAS_THINK" not in _text(tokenizer, example.question_input_ids)
    assert messages == original


@pytest.mark.parametrize("body_key", ["output", "content"])
def test_json_wrapper_hides_only_root_body_preserving_metadata_and_surrounding_text(tokenizer, body_key):
    payload = {
        "tool_call_id": "synthetic-wrapped-id", "name": "synthetic_lookup",
        "type": "tool_result", "is_error": True,
        "metadata": {"output": "SYNTHETIC_NESTED_METADATA"},
        body_key: {"nested": ["SYNTHETIC_WRAPPED_OUTPUT", "quoted </tool_response> and } text"]},
    }
    second = {"id": "synthetic-second-id", "is_error": False, body_key: "SYNTHETIC_SECOND_OUTPUT"}
    content = ("SYNTHETIC_BEFORE " + _wrapped(payload) + " SYNTHETIC_BETWEEN "
               + _wrapped(second) + " SYNTHETIC_AFTER")
    messages = [
        {"role": "user", "content": "Synthetic task."},
        _call("synthetic-wrapped-id", "SYNTHETIC_WRAPPED_ARGUMENT"),
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Synthetic target."},
    ]
    original = deepcopy(messages)
    example = _recover(tokenizer, messages, 3)
    train = _text(tokenizer, example.question_input_ids)
    generation = _text(tokenizer, example.generation_input_ids)
    expected = [deepcopy(payload), deepcopy(second)]
    for item in expected:
        item[body_key] = TOOL_OUTPUT_HIDDEN
    assert _wrapper_payloads(train) == expected
    for marker in ("SYNTHETIC_BEFORE", "SYNTHETIC_BETWEEN", "SYNTHETIC_AFTER",
                   "SYNTHETIC_NESTED_METADATA", "SYNTHETIC_WRAPPED_ARGUMENT"):
        assert marker in train and marker in generation
    for marker in ("SYNTHETIC_WRAPPED_OUTPUT", "SYNTHETIC_SECOND_OUTPUT"):
        assert marker not in train and marker in generation
    assert example.provenance["hidden_message_indices"] == [2]
    assert example.provenance["hidden_body_count"] == 2
    assert messages == original


@pytest.mark.parametrize("content", [
    'before <tool_response>{"output": "SYNTHETIC_UNCLOSED"}',
    'before <tool_response>{invalid json SYNTHETIC_INVALID}</tool_response> after',
    '<tool_response>["SYNTHETIC_NONOBJECT"]</tool_response>',
    '<tool_response>{"metadata": {"output": "SYNTHETIC_NESTED_ONLY"}}</tool_response>',
])
def test_invalid_or_bodyless_wrappers_remain_untouched(tokenizer, content):
    messages = [
        {"role": "user", "content": "SYNTHETIC_ROOT_TASK"},
        {"role": "assistant", "content": "Synthetic previous action."},
        {"role": "user", "content": content},
        {"role": "assistant", "content": "Synthetic answer."},
    ]
    example = _recover(tokenizer, messages, 3)
    ordinary = prepare_example(tokenizer, messages, 3, max_think_tokens=64)
    assert example.question_input_ids == ordinary.question_input_ids
    assert example.provenance["hidden_message_indices"] == []
    assert example.provenance["hidden_body_count"] == 0


def test_full_generation_can_exceed_32k_without_losing_original_output(tokenizer):
    output = "synthetic evidence " * 36000
    messages = [
        {"role": "user", "content": "SYNTHETIC_LONG_ROOT"},
        _call("synthetic-long-call", "read synthetic evidence"),
        {"role": "tool", "tool_call_id": "synthetic-long-call", "content": output},
        {"role": "assistant", "content": "SYNTHETIC_LONG_TARGET"},
    ]
    expected = _encode(tokenizer, _render(tokenizer, messages[:3], thinking=True, generation=True))
    assert len(expected) > 32768
    example = _recover(tokenizer, messages, 3, generation_context_tokens=len(expected) + 64)
    assert example.generation_input_ids == expected
    assert len(example.question_input_ids) + example.allowed_think_tokens + len(example.answer_suffix_ids) <= 32768
    assert example.allowed_think_tokens == 64
    assert example.provenance["removed_message_indices"] == []


def test_trims_complete_native_and_serialized_units_but_preserves_anchors(tokenizer):
    native = _call("synthetic-old-native", "SYNTHETIC_OLD_NATIVE " * 2000)
    serialized = {
        "role": "assistant",
        "content": '<tool_call>' + json.dumps({"name": "synthetic_lookup", "arguments": {
            "query": "SYNTHETIC_OLD_SERIALIZED " * 2000}}) + '</tool_call>',
    }
    final_call = _call("synthetic-final-call", "SYNTHETIC_FINAL_ARGUMENT")
    final_result = {"role": "tool", "tool_call_id": "synthetic-final-call", "content": "SYNTHETIC_FINAL_RESULT"}
    messages = [
        {"role": "system", "content": "SYNTHETIC_TRIM_SYSTEM"},
        {"role": "user", "content": "SYNTHETIC_TRIM_ROOT"},
        native,
        {"role": "tool", "tool_call_id": "synthetic-old-native", "content": "old result one"},
        {"role": "tool", "tool_call_id": "synthetic-old-native", "content": "old result two"},
        serialized,
        {"role": "user", "content": _wrapped({"id": "synthetic-old-serialized", "output": "old output"})},
        {"role": "user", "content": "SYNTHETIC_TRIM_LAST_USER"},
        final_call,
        final_result,
        {"role": "assistant", "content": "SYNTHETIC_TRIM_TARGET"},
    ]
    original = deepcopy(messages)
    kept_indices = [0, 1, 7, 8, 9]
    protected = [deepcopy(messages[index]) for index in kept_indices] + [deepcopy(messages[10])]
    reference = _recover(tokenizer, protected, len(protected) - 1)
    budget = len(reference.question_input_ids) + len(reference.answer_suffix_ids) + 64
    example = _recover(tokenizer, messages, 10, total_tokens=budget)
    assert example.provenance["removed_message_indices"] == [2, 3, 4, 5, 6]
    assert example.provenance["removed_unit_count"] == 2
    assert example.provenance["training_kept_message_indices"] == kept_indices
    assert example.question_input_ids == reference.question_input_ids
    assert example.answer_suffix_ids == reference.answer_suffix_ids
    assert example.allowed_think_tokens == 64
    assert example.generation_input_ids == _encode(tokenizer, _render(
        tokenizer, original[:10], thinking=True, generation=True))
    train = _text(tokenizer, example.question_input_ids)
    for marker in ("SYNTHETIC_TRIM_SYSTEM", "SYNTHETIC_TRIM_ROOT",
                   "SYNTHETIC_TRIM_LAST_USER", "SYNTHETIC_FINAL_ARGUMENT"):
        assert marker in train
    assert "SYNTHETIC_OLD_NATIVE" not in train
    assert "SYNTHETIC_OLD_SERIALIZED" not in train
    assert messages == original


def test_rejects_full_generation_that_cannot_fit_without_truncation(tokenizer, native_messages):
    example = _recover(tokenizer, native_messages, 6)
    original = deepcopy(native_messages)
    with pytest.raises(ValueError, match="^generation_complete_prefix_overflow$"):
        _recover(tokenizer, native_messages, 6,
                 generation_context_tokens=len(example.generation_input_ids))
    assert native_messages == original


def test_generation_capacity_caps_think_instead_of_truncating_prefix(tokenizer, native_messages):
    reference = _recover(tokenizer, native_messages, 6)
    room = max(8, len(reference.think_termination_ids))
    example = _recover(tokenizer, native_messages, 6,
                       generation_context_tokens=len(reference.generation_input_ids) + room)
    assert example.generation_input_ids == reference.generation_input_ids
    assert example.allowed_think_tokens == room


def test_rejects_impossible_protected_core_without_changing_target(tokenizer):
    messages = [
        {"role": "system", "content": "Synthetic protected policy."},
        {"role": "user", "content": "Synthetic protected task. " * 300},
        {"role": "assistant", "content": "SYNTHETIC_PROTECTED_TARGET"},
    ]
    reference = _recover(tokenizer, messages, 2)
    original = deepcopy(messages)
    budget = len(reference.question_input_ids) + len(reference.answer_suffix_ids)
    with pytest.raises(ValueError, match="^recovery_protected_core_overflow$"):
        _recover(tokenizer, messages, 2, total_tokens=budget)
    assert messages == original


def test_rejects_target_larger_than_training_budget_instead_of_shortening_it(tokenizer):
    messages = [
        {"role": "user", "content": "Synthetic task."},
        {"role": "assistant", "content": "SYNTHETIC_LARGE_TARGET " * 1000},
    ]
    original = deepcopy(messages)
    with pytest.raises(ValueError, match="^answer_exceeds_total_budget$"):
        _recover(tokenizer, messages, 1, total_tokens=128)
    assert messages == original


def test_protected_core_uses_remaining_think_budget_without_dropping_anchors(tokenizer):
    messages = [
        {"role": "system", "content": "SYNTHETIC_BUDGET_SYSTEM"},
        {"role": "user", "content": "SYNTHETIC_BUDGET_ROOT"},
        {"role": "assistant", "content": "SYNTHETIC_BUDGET_TARGET"},
    ]
    reference = _recover(tokenizer, messages, 2)
    room = max(8, len(reference.think_termination_ids))
    budget = len(reference.question_input_ids) + len(reference.answer_suffix_ids) + room
    example = _recover(tokenizer, messages, 2, total_tokens=budget)
    assert example.allowed_think_tokens == room
    assert example.question_input_ids == reference.question_input_ids
    assert example.answer_suffix_ids == reference.answer_suffix_ids
    assert example.provenance["removed_message_indices"] == []


def test_schema_two_roundtrip_and_training_only_assembly_masks(tokenizer, native_messages):
    example = _recover(tokenizer, native_messages, 6)
    record = example_to_record(example)
    assert record["schema"] == 2
    decoded = json.loads(json.dumps(record))
    assert example_from_record(decoded) == example
    assert decoded["generation_input_ids"] == list(example.generation_input_ids)
    assert decoded["generation_context_tokens"] == example.generation_context_tokens
    think = _encode(tokenizer, "A synthetic conclusion follows.") + example.think_termination_ids
    assembled = assemble_self_think(example, think, adapter_revision="synthetic-current-adapter",
                                   generation_seconds=0.125)
    prefix_length = len(example.question_input_ids) + len(think)
    assert assembled["input_ids"] == list(example.question_input_ids + think + example.answer_suffix_ids)
    assert len(assembled["input_ids"]) <= example.total_tokens
    assert assembled["labels"][:prefix_length] == [-100] * prefix_length
    assert assembled["labels"][prefix_length:] == list(example.answer_suffix_labels)
    separator = example.provenance["answer_separator_tokens"]
    assert assembled["labels"][prefix_length:prefix_length + separator] == [-100] * separator
    assert assembled["labels"][prefix_length + separator:] == list(example.answer_suffix_ids[separator:])
    ordinary_prefix = example.provenance["ordinary_prefix_tokens"]
    assert example.ordinary_labels[:ordinary_prefix] == (-100,) * ordinary_prefix
    assert example.ordinary_labels[ordinary_prefix:] == example.ordinary_input_ids[ordinary_prefix:]
    assert "SYNTHETIC_NATIVE_OUTPUT" not in _text(tokenizer, assembled["input_ids"])
    assert "SYNTHETIC_TARGET_ANSWER" in _text(tokenizer, assembled["input_ids"])
    assert any(label != -100 for label in assembled["labels"])


@pytest.mark.parametrize("field", ["generation_input_ids", "generation_context_tokens",
                                    "generation_question_sha256", "generation_question_tokens"])
def test_schema_two_rejects_generation_binding_tampering(tokenizer, native_messages, field):
    example = _recover(tokenizer, native_messages, 6)
    record = json.loads(json.dumps(example_to_record(example)))
    if field == "generation_input_ids":
        record[field] = record[field][1:]
    elif field == "generation_context_tokens":
        record[field] = len(example.generation_input_ids)
    elif field == "generation_question_tokens":
        record["provenance"][field] += 1
    else:
        record["provenance"][field] = _hash("synthetic-wrong-generation-binding")
    with pytest.raises(ValueError):
        example_from_record(record)


def test_schema_one_serialization_omits_dual_prefix_fields(tokenizer):
    messages = [{"role": "user", "content": "Synthetic ordinary task."},
                {"role": "assistant", "content": "Synthetic ordinary target."}]
    example = prepare_example(tokenizer, messages, 1, max_think_tokens=64)
    expected = asdict(example)
    expected.pop("generation_input_ids")
    expected.pop("generation_context_tokens")
    record = example_to_record(example)
    assert record == {"schema": 1, **expected}
    assert example_from_record(json.loads(json.dumps(record))) == example
    assert generation_plan(example).question_input_ids == example.question_input_ids


def test_assembly_rejects_think_and_total_budget_overflow(tokenizer, native_messages):
    example = _recover(tokenizer, native_messages, 6)
    think = _encode(tokenizer, "Synthetic reasoning.") + example.think_termination_ids
    with pytest.raises(ValueError, match="^think_generation_budget_overflow$"):
        assemble_self_think(replace(example, allowed_think_tokens=len(think) - 1), think,
                            adapter_revision="synthetic-adapter", generation_seconds=0.0)
    total = len(example.question_input_ids) + len(think) + len(example.answer_suffix_ids)
    with pytest.raises(ValueError, match="^assembled_sequence_overflow$"):
        assemble_self_think(replace(example, total_tokens=total - 1), think,
                            adapter_revision="synthetic-adapter", generation_seconds=0.0)
