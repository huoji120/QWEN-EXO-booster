"""Content-blind, fixed-teacher Think cache bound to prepared examples.

Records contain exact generated token IDs, never repaired reasoning. Loading a
cache requires exact coverage of the supplied examples; the prepared-data digest
continues to identify the original prepared file when that set is filtered.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .native_mid_lora_self_think import PreparedExample, _hash, assemble_self_think, generation_plan


SCHEMA = "native-precomputed-think-v1"
ADAPTER_POLICY = "fixed_precomputed_teacher_no_sampling_gradient"
_TEACHER_FIELDS = ("teacher_fingerprint", "teacher_adapter_sha256", "teacher_profile_path")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")


def _require_digest(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("precomputed_think_digest_invalid")
    return value


def _example_identity(example: PreparedExample, prepared_data_sha256: str) -> tuple[str, str]:
    _require_digest(prepared_data_sha256)
    sample_hash = _require_digest(example.provenance.get("sample_hash"))
    question_hash = _require_digest(example.provenance.get("question_sha256"))
    if question_hash != _hash(example.question_input_ids):
        raise ValueError("precomputed_think_question_digest_mismatch")
    return sample_hash, question_hash


def _teacher(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("precomputed_think_teacher_invalid")
    fingerprint = _require_digest(value.get("teacher_fingerprint"))
    adapter = _require_digest(value.get("teacher_adapter_sha256"))
    profile = value.get("teacher_profile_path")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("precomputed_think_teacher_profile_invalid")
    return dict(zip(_TEACHER_FIELDS, (fingerprint, adapter, profile)))


def _json_value(value: Any) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_value(item) for key, item in value.items())
    return False


def _metadata(value: Any) -> dict[str, Any]:
    try:
        valid = isinstance(value, dict) and _json_value(value)
        if valid:
            return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("precomputed_think_metadata_invalid") from None


def _seconds(value: Any) -> float:
    try:
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            return float(value)
    except OverflowError:
        pass
    raise ValueError("precomputed_think_seconds_invalid")


def _assemble(example: PreparedExample, tokens: Sequence[int], teacher: dict[str, str],
              seconds: float) -> dict[str, Any]:
    assembled = assemble_self_think(example, tokens,
        adapter_revision=teacher["teacher_adapter_sha256"], generation_seconds=seconds)
    assembled["provenance"].update(teacher, adapter_policy=ADAPTER_POLICY)
    return assembled


def make_record(example: PreparedExample, token_ids: Sequence[int], *, seconds: float,
                prepared_data_sha256: str, teacher: dict[str, Any],
                shaping: dict[str, Any], sampling: dict[str, Any]) -> dict[str, Any]:
    """Bind exact stopped teacher tokens to one original prepared sample."""
    sample_hash, question_hash = _example_identity(example, prepared_data_sha256)
    binding = _teacher(teacher)
    shaping_copy, sampling_copy = _metadata(shaping), _metadata(sampling)
    elapsed = _seconds(seconds)
    if not isinstance(token_ids, (list, tuple)):
        raise ValueError("precomputed_think_tokens_invalid")
    tokens = list(token_ids)
    assembled = _assemble(example, tokens, binding, elapsed)
    return {
        "schema": SCHEMA,
        "sample_hash": sample_hash,
        "question_sha256": question_hash,
        "prepared_data_sha256": prepared_data_sha256,
        "generation_question_sha256": _hash(generation_plan(example).question_input_ids),
        **binding,
        "shaping": shaping_copy,
        "sampling": sampling_copy,
        "token_ids": tokens,
        "seconds": elapsed,
        "token_ids_sha256": _hash(tokens),
        "assembled_sha256": _hash(assembled["input_ids"]),
        "generated_count": len(tokens),
    }

def _assemble_supplied(example, tokens, source_sha256, reasoning_sha256, reasoning_present, supervision):
    if supervision not in ("answer_only", "think_and_answer"):
        raise ValueError("supplied_think_supervision_invalid")
    if example.provenance.get("supplied_supervision", "answer_only") != supervision:
        raise ValueError("supplied_think_supervision_mismatch")
    assembled = assemble_self_think(example, tokens, adapter_revision=source_sha256, generation_seconds=0.)
    if supervision == "think_and_answer" and reasoning_present:
        # Only the CURRENT target's supplied reasoning gets a label. Historical
        # turns and task/system prompts remain ignored by cross entropy.
        assembled["labels"] = [-100] * len(example.question_input_ids) + list(tokens) + list(example.answer_suffix_labels)
    assembled["provenance"].update(adapter_policy="provided_source_reasoning_" + supervision,
        source_reasoning_sha256=reasoning_sha256, source_reasoning_present=reasoning_present,
        supplied_supervision=supervision)
    return assembled


def make_supplied_record(example, token_ids, *, prepared_data_sha256, source_sha256,
                         source_reasoning_sha256, reasoning_present, supervision="answer_only"):
    """Frame existing source reasoning; no inference, no invented teacher weights."""
    sample_hash, question_hash = _example_identity(example, prepared_data_sha256)
    _require_digest(source_sha256)
    _require_digest(source_reasoning_sha256)
    tokens = list(token_ids)
    assembled = _assemble_supplied(example, tokens, source_sha256, source_reasoning_sha256, bool(reasoning_present), supervision)
    return {"schema": "native-supplied-think-v1", "sample_hash": sample_hash,
            "question_sha256": question_hash, "prepared_data_sha256": prepared_data_sha256,
            "generation_question_sha256": _hash(generation_plan(example).question_input_ids),
            "source_sha256": source_sha256, "source_reasoning_sha256": source_reasoning_sha256,
            "reasoning_present": bool(reasoning_present), "token_ids": tokens, "seconds": 0.,
            "token_ids_sha256": _hash(tokens), "assembled_sha256": _hash(assembled["input_ids"]),
            "generated_count": len(tokens), "origin": "provided_source_reasoning",
            "supervision": supervision, "labels_sha256": _hash(assembled["labels"])}



def validate_record(record: dict[str, Any], example: PreparedExample, *,
                    prepared_data_sha256: str) -> dict[str, Any]:
    """Return the answer-only assembled sequence, rejecting any binding drift."""
    sample_hash, question_hash = _example_identity(example, prepared_data_sha256)
    if not isinstance(record, dict) or record.get("schema") not in (SCHEMA, "native-supplied-think-v1"):
        raise ValueError("precomputed_think_schema_mismatch")
    if record.get("sample_hash") != sample_hash:
        raise ValueError("precomputed_think_sample_mismatch")
    if record.get("prepared_data_sha256") != prepared_data_sha256:
        raise ValueError("precomputed_think_prepared_data_mismatch")
    if record.get("question_sha256") != question_hash:
        raise ValueError("precomputed_think_question_mismatch")
    generation_hash = _hash(generation_plan(example).question_input_ids)
    if record.get("generation_question_sha256", question_hash) != generation_hash:
        raise ValueError("precomputed_think_generation_prefix_mismatch")
    supplied = record["schema"] == "native-supplied-think-v1"
    if supplied:
        _require_digest(record.get("source_sha256"))
        _require_digest(record.get("source_reasoning_sha256"))
        if (record.get("origin") != "provided_source_reasoning"
                or type(record.get("reasoning_present")) is not bool or record.get("seconds") != 0):
            raise ValueError("supplied_think_origin_invalid")
    else:
        binding = _teacher(record)
        _metadata(record.get("shaping"))
        _metadata(record.get("sampling"))
    elapsed = _seconds(record.get("seconds"))
    tokens = record.get("token_ids")
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("precomputed_think_tokens_invalid")
    count = record.get("generated_count")
    if type(count) is not int or count != len(tokens):
        raise ValueError("precomputed_think_generated_count_mismatch")
    if _require_digest(record.get("token_ids_sha256")) != _hash(tokens):
        raise ValueError("precomputed_think_token_digest_mismatch")
    if supplied:
        supervision = record.get("supervision", "answer_only")
        assembled = _assemble_supplied(example, tokens, record["source_sha256"], record["source_reasoning_sha256"], record["reasoning_present"], supervision)
        if supervision != "answer_only" or "labels_sha256" in record:
            if record.get("labels_sha256") != _hash(assembled["labels"]):
                raise ValueError("supplied_think_labels_digest_mismatch")
    else:
        assembled = _assemble(example, tokens, binding, elapsed)
    if _require_digest(record.get("assembled_sha256")) != _hash(assembled["input_ids"]):
        raise ValueError("precomputed_think_assembled_digest_mismatch")
    return assembled


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("precomputed_think_duplicate_json_key")
        result[key] = value
    return result


def load_precomputed_thinks(path: str | Path, examples: Iterable[PreparedExample], *,
                           prepared_data_sha256: str) -> dict[str, dict[str, Any]]:
    """Load JSONL with exact coverage and one immutable teacher binding.

    Shaping budgets may differ per sample. Neither a filtered examples list nor
    its ordering changes the caller-supplied original prepared-data digest.
    All rejection messages are content-safe constant codes.
    """
    _require_digest(prepared_data_sha256)
    expected = {}
    for example in examples:
        sample_hash, _ = _example_identity(example, prepared_data_sha256)
        if sample_hash in expected:
            raise ValueError("precomputed_think_duplicate_example")
        expected[sample_hash] = example
    records = {}
    teacher_binding = None
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("precomputed_think_blank_row")
                try:
                    record = json.loads(line, object_pairs_hook=_json_object)
                except (ValueError, RecursionError):
                    raise ValueError("precomputed_think_json_invalid") from None
                if not isinstance(record, dict):
                    raise ValueError("precomputed_think_record_invalid")
                sample_hash = _require_digest(record.get("sample_hash"))
                if sample_hash in records:
                    raise ValueError("precomputed_think_duplicate_record")
                if sample_hash not in expected:
                    raise ValueError("precomputed_think_extra_record")
                validate_record(record, expected[sample_hash], prepared_data_sha256=prepared_data_sha256)
                binding = ((record["schema"], record["source_sha256"]) if record["schema"] == "native-supplied-think-v1"
                           else (record["schema"], *(record[key] for key in _TEACHER_FIELDS)))
                if teacher_binding is not None and binding != teacher_binding:
                    raise ValueError("precomputed_think_teacher_inconsistent")
                teacher_binding = binding
                records[sample_hash] = record
    except (OSError, UnicodeError):
        raise ValueError("precomputed_think_file_unreadable") from None
    if records.keys() != expected.keys():
        raise ValueError("precomputed_think_missing_records")
    return records
