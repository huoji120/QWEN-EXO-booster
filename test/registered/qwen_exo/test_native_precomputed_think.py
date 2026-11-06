"""Synthetic fixed-teacher Think cache contracts using the real local tokenizer.

Set NATIVE_MID_LORA_TEST_PROFILE to a local tokenizer profile; no model,
private corpus, serving endpoint, or CUDA execution is needed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from qwen_exo_booster.native_mid_lora_self_think import (
    _hash,
    load_profile_tokenizer,
    prepare_example,
)
from qwen_exo_booster.native_precomputed_think import (
    load_precomputed_thinks,
    make_record,
    validate_record,
)


_PREPARED_SHA = _hash("synthetic-original-prepared-data")
_TEACHER = {
    "teacher_fingerprint": _hash("synthetic-fixed-teacher"),
    "teacher_adapter_sha256": _hash("synthetic-fixed-teacher-adapter"),
    "teacher_profile_path": "synthetic/fixed-teacher-profile",
}
_SHAPING = {
    "suppress_after": 32,
    "suppress_strength": 1.0,
    "landing_start": 48,
    "landing_end": 64,
    "landing_bias": 1.0,
}
_SAMPLING = {"temperature": 0.0, "max_new_tokens": 64}


@pytest.fixture(scope="module")
def tokenizer():
    profile = os.environ.get("NATIVE_MID_LORA_TEST_PROFILE")
    if not profile:
        pytest.skip("NATIVE_MID_LORA_TEST_PROFILE is not set")
    if not Path(profile).is_dir():
        pytest.skip("NATIVE_MID_LORA_TEST_PROFILE directory is absent")
    tokenizer, _ = load_profile_tokenizer(profile)
    return tokenizer


@pytest.fixture
def examples(tokenizer):
    # Every message is synthetic. Include earlier context to exercise its mask,
    # not just the final user prompt and answer separator.
    return [
        prepare_example(
            tokenizer,
            [
                {"role": "user", "content": "Remember the synthetic label amber."},
                {"role": "assistant", "content": "The label is amber."},
                {"role": "user", "content": f"Return synthetic result {index}."},
                {"role": "assistant", "content": f"Synthetic result {index}."},
            ],
            3,
            max_think_tokens=64,
        )
        for index in range(2)
    ]


@pytest.fixture
def think_ids(tokenizer, examples):
    body = tuple(tokenizer.encode("The synthetic result follows directly.", add_special_tokens=False))
    return body + examples[0].think_termination_ids


def _record(example, tokens, **overrides):
    kwargs = {
        "seconds": 0.25,
        "prepared_data_sha256": _PREPARED_SHA,
        "teacher": dict(_TEACHER),
        "shaping": dict(_SHAPING),
        "sampling": dict(_SAMPLING),
    }
    kwargs.update(overrides)
    return make_record(example, tokens, **kwargs)


def _save(tmp_path, records):
    path = tmp_path / "synthetic-thinks.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _with_tokens(record, example, tokens):
    # Recompute both digests so malformed-sequence cases must reach protocol
    # validation rather than merely fail the first checksum comparison.
    mutated = deepcopy(record)
    mutated["token_ids"] = list(tokens)
    mutated["token_ids_sha256"] = _hash(tokens)
    mutated["assembled_sha256"] = _hash(
        example.question_input_ids + tuple(tokens) + example.answer_suffix_ids
    )
    mutated["generated_count"] = len(tokens)
    return mutated


def test_roundtrip_masks_all_context_and_think_but_supervises_answer(tmp_path, examples, think_ids):
    example = examples[0]
    record = _record(example, think_ids)
    assert record["schema"] == "native-precomputed-think-v1"
    assert record["sample_hash"] == example.provenance["sample_hash"]
    assert record["question_sha256"] == _hash(example.question_input_ids)
    assert record["prepared_data_sha256"] == _PREPARED_SHA
    assert record["token_ids"] == list(think_ids)
    assert record["token_ids_sha256"] == _hash(think_ids)
    assert record["generated_count"] == len(think_ids)
    assert record["seconds"] == 0.25
    assert record["shaping"] == _SHAPING
    assert record["sampling"] == _SAMPLING
    for key, value in _TEACHER.items():
        assert record[key] == value

    path = _save(tmp_path, [record])
    loaded = load_precomputed_thinks(path, [example], prepared_data_sha256=_PREPARED_SHA)
    assert loaded == {example.provenance["sample_hash"]: record}
    assembled = validate_record(loaded[record["sample_hash"]], example,
                                prepared_data_sha256=_PREPARED_SHA)
    prefix_length = len(example.question_input_ids) + len(think_ids)
    expected_ids = list(example.question_input_ids + think_ids + example.answer_suffix_ids)
    assert assembled["input_ids"] == expected_ids
    assert record["assembled_sha256"] == _hash(expected_ids)
    assert assembled["labels"][:prefix_length] == [-100] * prefix_length
    assert assembled["labels"][prefix_length:] == list(example.answer_suffix_labels)
    separator_length = example.provenance["answer_separator_tokens"]
    assert assembled["labels"][prefix_length:prefix_length + separator_length] == [-100] * separator_length
    assert assembled["labels"][prefix_length + separator_length:] == list(example.answer_suffix_ids[separator_length:])
    assert any(label != -100 for label in assembled["labels"])


@pytest.mark.parametrize("field", ["sample_hash", "question_sha256", "prepared_data_sha256"])
def test_rejects_wrong_record_binding(field, examples, think_ids):
    record = _record(examples[0], think_ids)
    record[field] = _hash("different-synthetic-binding")
    with pytest.raises(ValueError):
        validate_record(record, examples[0], prepared_data_sha256=_PREPARED_SHA)


def test_rejects_wrong_expected_example_and_source_digest(examples, think_ids):
    record = _record(examples[0], think_ids)
    with pytest.raises(ValueError):
        validate_record(record, examples[1], prepared_data_sha256=_PREPARED_SHA)
    with pytest.raises(ValueError):
        validate_record(record, examples[0], prepared_data_sha256=_hash("different-source"))


@pytest.mark.parametrize("coverage", ["duplicate", "missing", "extra"])
def test_requires_exact_record_coverage(tmp_path, examples, think_ids, coverage):
    first, second = (_record(example, think_ids) for example in examples)
    records, expected = {
        "duplicate": ([first, first, second], examples),
        "missing": ([first], examples),
        "extra": ([first, second], examples[:1]),
    }[coverage]
    with pytest.raises(ValueError):
        load_precomputed_thinks(_save(tmp_path, records), expected,
                                prepared_data_sha256=_PREPARED_SHA)


def test_filtered_subset_retains_original_prepared_hash(tmp_path, examples, think_ids):
    record = _record(examples[1], think_ids)
    loaded = load_precomputed_thinks(_save(tmp_path, [record]), examples[1:],
                                    prepared_data_sha256=_PREPARED_SHA)
    assert loaded == {examples[1].provenance["sample_hash"]: record}


def test_accepts_per_sample_shaping_budgets_and_arbitrary_record_order(tmp_path, examples, think_ids):
    first = _record(examples[0], think_ids)
    second = _record(examples[1], think_ids,
                     shaping={**_SHAPING, "landing_start": 24, "landing_end": 32},
                     sampling={**_SAMPLING, "max_new_tokens": 32})
    loaded = load_precomputed_thinks(_save(tmp_path, [second, first]), examples,
                                    prepared_data_sha256=_PREPARED_SHA)
    assert loaded == {record["sample_hash"]: record for record in (first, second)}


@pytest.mark.parametrize("field", list(_TEACHER))
def test_rejects_mixed_fixed_teachers(tmp_path, examples, think_ids, field):
    teacher = dict(_TEACHER)
    teacher[field] = "synthetic/other-profile" if field == "teacher_profile_path" else _hash("other-teacher")
    records = [_record(examples[0], think_ids), _record(examples[1], think_ids, teacher=teacher)]
    with pytest.raises(ValueError):
        load_precomputed_thinks(_save(tmp_path, records), examples,
                                prepared_data_sha256=_PREPARED_SHA)


@pytest.mark.parametrize("field", ["token_ids", "token_ids_sha256", "assembled_sha256", "generated_count"])
def test_rejects_token_and_assembled_tampering(examples, think_ids, field):
    record = _record(examples[0], think_ids)
    if field == "token_ids":
        record[field] = record[field][1:]
    elif field == "generated_count":
        record[field] += 1
    else:
        record[field] = _hash("tampered-digest")
    with pytest.raises(ValueError):
        validate_record(record, examples[0], prepared_data_sha256=_PREPARED_SHA)


@pytest.mark.parametrize("malformation", ["unterminated", "repeated_closer", "trailing_tokens", "repeated_opener", "forbidden", "overflow"])
def test_never_repairs_or_truncates_malformed_generated_think(examples, think_ids, malformation):
    example = examples[0]
    body = think_ids[:-len(example.think_termination_ids)]
    tokens = {
        "unterminated": body,
        "repeated_closer": think_ids + example.think_close_ids,
        "trailing_tokens": think_ids + body,
        "repeated_opener": example.think_open_ids + think_ids,
        "forbidden": (example.forbidden_think_token_ids[0],) + think_ids,
        "overflow": (body[0],) * example.allowed_think_tokens + example.think_close_ids,
    }[malformation]
    with pytest.raises(ValueError):
        _record(example, tokens)
    record = _with_tokens(_record(example, think_ids), example, tokens)
    with pytest.raises(ValueError):
        validate_record(record, example, prepared_data_sha256=_PREPARED_SHA)


def test_rejects_assembled_context_overflow_without_trimming(examples, think_ids):
    example = examples[0]
    record = _record(example, think_ids)
    too_small = replace(example, total_tokens=(len(example.question_input_ids)
                                              + len(think_ids)
                                              + len(example.answer_suffix_ids) - 1))
    with pytest.raises(ValueError):
        _record(too_small, think_ids)
    with pytest.raises(ValueError):
        validate_record(record, too_small, prepared_data_sha256=_PREPARED_SHA)


def test_recovered_cache_binds_full_generation_not_only_training_prefix(tokenizer):
    from qwen_exo_booster.native_mid_lora_recovery import prepare_recovered_example
    messages = [
        {"role": "user", "content": "Read the synthetic observation."},
        {"role": "assistant", "content": '<tool_call>{"name":"read","arguments":{}}</tool_call>'},
        {"role": "user", "content": '<tool_response>{"content":"UNIQUE_SYNTHETIC_EVIDENCE","is_error":false}</tool_response>'},
        {"role": "assistant", "content": "Synthetic target action."},
    ]
    example = prepare_recovered_example(tokenizer, messages, 3, generation_context_tokens=65536, max_think_tokens=64)
    tokens = tuple(tokenizer.encode("I read the evidence.\n</think>", add_special_tokens=False))
    record = make_record(example, tokens, seconds=1., prepared_data_sha256=_PREPARED_SHA,
                         teacher=_TEACHER, shaping=_SHAPING, sampling=_SAMPLING)
    assert record["generation_question_sha256"] != record["question_sha256"]
    assembled = validate_record(record, example, prepared_data_sha256=_PREPARED_SHA)
    assert "UNIQUE_SYNTHETIC_EVIDENCE" not in tokenizer.decode(assembled["input_ids"])
    changed = deepcopy(record)
    changed["generation_question_sha256"] = changed["question_sha256"]
    with pytest.raises(ValueError, match="precomputed_think_generation_prefix_mismatch"):
        validate_record(changed, example, prepared_data_sha256=_PREPARED_SHA)


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("supervision", ["answer_only", "think_and_answer"])
def test_supplied_reasoning_masks_and_source_binding(tokenizer, examples, tmp_path, present, supervision):
    from qwen_exo_booster.native_precomputed_think import make_supplied_record
    import hashlib
    example = replace(examples[0], provenance={**examples[0].provenance, "supplied_supervision": supervision})
    reason = "The observation determines the action." if present else ""
    tokens = tokenizer.encode(reason + "\n</think>", add_special_tokens=False)
    record = make_supplied_record(example, tokens, prepared_data_sha256=_PREPARED_SHA,
        source_sha256=_hash("provided-source"), source_reasoning_sha256=hashlib.sha256(reason.encode()).hexdigest(),
        reasoning_present=present, supervision=supervision)
    assembled = validate_record(record, example, prepared_data_sha256=_PREPARED_SHA)
    boundary = len(example.question_input_ids) + len(tokens)
    start = len(example.question_input_ids)
    assert all(label == -100 for label in assembled["labels"][:start])
    if present and supervision == "think_and_answer":
        assert list(assembled["labels"][start:boundary]) == tokens
    else:
        assert all(label == -100 for label in assembled["labels"][start:boundary])
    assert tuple(assembled["input_ids"][boundary:]) == example.answer_suffix_ids
    assert assembled["provenance"]["source_reasoning_present"] is present
    path = tmp_path / "provided.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert load_precomputed_thinks(path, [example], prepared_data_sha256=_PREPARED_SHA)[record["sample_hash"]]["token_ids"] == tokens
    corrupt = deepcopy(record)
    corrupt["token_ids"][0] += 1
    with pytest.raises(ValueError):
        validate_record(corrupt, example, prepared_data_sha256=_PREPARED_SHA)
    corrupt = deepcopy(record)
    corrupt["source_sha256"] = "not-a-digest"
    with pytest.raises(ValueError):
        validate_record(corrupt, example, prepared_data_sha256=_PREPARED_SHA)
    changed = deepcopy(record)
    changed["supervision"] = "answer_only" if supervision == "think_and_answer" else "think_and_answer"
    with pytest.raises(ValueError, match="supplied_think_supervision_mismatch"):
        validate_record(changed, example, prepared_data_sha256=_PREPARED_SHA)
    changed = deepcopy(record)
    changed["labels_sha256"] = _hash([])
    with pytest.raises(ValueError, match="supplied_think_labels_digest_mismatch"):
        validate_record(changed, example, prepared_data_sha256=_PREPARED_SHA)
