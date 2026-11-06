"""Causal, template-exact data preparation for answer-only self-Think LoRA.

No model, corpus, or CUDA is loaded at import time. GenerationPlan deliberately
contains no answer, provenance object, or reference to its PreparedExample.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = 1
ADAPTER_POLICY = "fresh_current_adapter_per_sample_no_sampling_gradient"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationPlan:
    question_input_ids: tuple[int, ...]
    allowed_think_tokens: int
    think_close_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedExample:
    question_input_ids: tuple[int, ...]
    answer_suffix_ids: tuple[int, ...]
    allowed_think_tokens: int
    provenance: dict[str, Any]
    ordinary_input_ids: tuple[int, ...]
    ordinary_labels: tuple[int, ...]
    think_close_ids: tuple[int, ...]
    total_tokens: int
    answer_suffix_labels: tuple[int, ...]
    think_open_ids: tuple[int, ...]
    think_termination_ids: tuple[int, ...]
    forbidden_think_token_ids: tuple[int, ...]
    generation_input_ids: tuple[int, ...] | None = None
    generation_context_tokens: int | None = None


def generation_plan(example: PreparedExample) -> GenerationPlan:
    prefix = example.generation_input_ids if example.generation_input_ids is not None else example.question_input_ids
    return GenerationPlan(tuple(prefix), example.allowed_think_tokens, tuple(example.think_close_ids))


def _ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))


def _positions(tokens: Sequence[int], marker: Sequence[int]) -> list[int]:
    width = len(marker)
    if not width:
        raise ValueError("empty_protocol_marker")
    return [index for index in range(len(tokens) - width + 1)
            if tuple(tokens[index:index + width]) == tuple(marker)]


def _strip_leading_think(text: str) -> str:
    # Only assistant framing at the start is reasoning. Do not regex serialized
    # JSON, tool arguments, user observations, or literal code inside the answer.
    remaining = text
    while remaining.lstrip().startswith("<think>"):
        start = len(remaining) - len(remaining.lstrip())
        end = remaining.find("</think>", start + len("<think>"))
        if end < 0:
            raise ValueError("source_think_unterminated")
        if "<think>" in remaining[start + len("<think>"):end]:
            raise ValueError("source_think_nested")
        remaining = remaining[end + len("</think>"):]
    if remaining.lstrip().startswith("</think>"):
        raise ValueError("source_think_unmatched_closer")
    return remaining


def clean_message(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
        raise ValueError("source_message_role_invalid")
    result = dict(message)
    content = result.get("content")
    if content is not None and not isinstance(content, (str, list)):
        raise ValueError("source_content_not_text")
    assistant = result["role"] == "assistant"
    for key in ("reasoning_content", "thinking", "reasoning"):
        result.pop(key, None)
    if isinstance(content, list):
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("source_content_block_invalid")
            if assistant and block.get("type") in {"thinking", "redacted_thinking", "reasoning"}:
                continue
            if block.get("type") not in {"text", "input_text", "output_text"} or not isinstance(block.get("text"), str):
                raise ValueError("source_nontext_block_unsupported")
            blocks.append(dict(block))
        # Match the template's concatenation before identifying assistant-only
        # leading framing; no tool JSON is parsed and rewritten here.
        content = "".join(block["text"] for block in blocks)
    if assistant and isinstance(content, str):
        content = _strip_leading_think(content)
    result["content"] = content
    return result


def _render(tokenizer, messages, *, thinking: bool, generation: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=generation, enable_thinking=thinking,
        reasoning_effort="xhigh", preserve_thinking=True)


def load_profile_tokenizer(profile: str | Path):
    """Use only the profile's local tokenizer and exact on-disk Jinja template."""
    from transformers import AutoTokenizer
    profile = Path(profile).resolve()
    template_path = profile / "chat_template.jinja"
    tokenizer = AutoTokenizer.from_pretrained(str(profile), local_files_only=True,
                                             trust_remote_code=False)
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")
    binding = {name: file_sha256(profile / name) for name in
               ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    return tokenizer, binding


def prepare_example(tokenizer, messages: Sequence[dict[str, Any]], target_index: int, *,
                    total_tokens: int = 32768, max_think_tokens: int = 8192,
                    min_think_tokens: int = 1, provenance: dict[str, Any] | None = None) -> PreparedExample:
    """Take one complete original prefix; never truncate or select later evidence."""
    if not 2 <= total_tokens <= 32768 or min_think_tokens < 1 or max_think_tokens < min_think_tokens:
        raise ValueError("token_budget_invalid")
    if not 0 < target_index < len(messages) or messages[target_index].get("role") != "assistant":
        raise ValueError("target_not_causal_assistant")
    cleaned = [clean_message(message) for message in messages[:target_index + 1]]
    history, target = cleaned[:-1], cleaned[-1]
    if not any(message["role"] == "user" for message in history):
        raise ValueError("target_without_user_root")
    if not ((target.get("content") or "").strip() or target.get("tool_calls")):
        raise ValueError("target_answer_empty")
    prompt = _render(tokenizer, history, thinking=True, generation=True)
    if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("template_think_prompt_mismatch")
    full = _render(tokenizer, cleaned, thinking=True, generation=False)
    before_answer = prompt + "\n</think>\n\n"
    if not full.startswith(before_answer) or not full.endswith("<|im_end|>\n"):
        raise ValueError("template_answer_boundary_mismatch")
    question = _ids(tokenizer, prompt)
    close, opening = _ids(tokenizer, "</think>"), _ids(tokenizer, "<think>")
    if len(close) != 1:
        raise ValueError("template_think_closer_not_atomic")
    # The empty-Think rendering has two adjacent newlines before its closer.
    # Their BPE may merge, unlike the actual generation prompt's final newline.
    # Only the atomic closer defines the gold suffix: never require the empty
    # rendering to share the question's token prefix or alter question IDs.
    through_close_text = prompt + "\n</think>"
    suffix_text = full[len(through_close_text):]
    suffix = _ids(tokenizer, suffix_text)
    through_close = _ids(tokenizer, through_close_text)
    separator = _ids(tokenizer, "\n\n")
    if (through_close[-len(close):] != close
            or through_close + suffix != _ids(tokenizer, full)
            or suffix[:len(separator)] != separator):
        raise ValueError("tokenizer_answer_boundary_not_separable")
    separator_size = len(separator)
    suffix_labels = (-100,) * separator_size + suffix[separator_size:]
    if len(suffix) >= total_tokens:
        raise ValueError("answer_exceeds_total_budget")
    termination = _ids(tokenizer, "\n</think>")
    allowed = min(max_think_tokens, total_tokens - len(question) - len(suffix))
    if allowed < max(min_think_tokens, len(termination)):
        raise ValueError("complete_prefix_insufficient_think_budget")
    # Keep identical xhigh system instructions and history in both arms. The
    # control teacher-forces an empty, fully masked Think instead of changing
    # enable_thinking, which would remove the template's reasoning policy.
    ordinary, ordinary_prefix = _ids(tokenizer, full), _ids(tokenizer, before_answer)
    if ordinary[:len(ordinary_prefix)] != ordinary_prefix:
        raise ValueError("tokenizer_ordinary_boundary_not_separable")
    if len(ordinary) > total_tokens:
        raise ValueError("ordinary_complete_prefix_overflow")
    forbidden = set()
    for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        encoded = _ids(tokenizer, marker)
        if len(encoded) != 1:
            raise ValueError("template_framing_marker_not_atomic")
        forbidden.update(encoded)
    eos = tokenizer.eos_token_id
    forbidden.update(eos if isinstance(eos, (list, tuple)) else (() if eos is None else (eos,)))
    metadata = dict(provenance or {})
    metadata.update(sample_hash=_hash(cleaned), target_index=target_index,
        question_sha256=_hash(question), answer_sha256=_hash(suffix),
        ordinary_sha256=_hash(ordinary), ordinary_prefix_tokens=len(ordinary_prefix),
        answer_separator_tokens=separator_size,
        adapter_policy=ADAPTER_POLICY, source_think_supervised=False,
        ordinary_policy="empty_think_matched_template", reasoning_effort="xhigh",
        prefix_policy="complete_original_prefix_no_truncation", question_tokens=len(question),
        answer_suffix_tokens=len(suffix), ordinary_tokens=len(ordinary))
    return PreparedExample(question, suffix, allowed, metadata, ordinary,
        (-100,) * len(ordinary_prefix) + ordinary[len(ordinary_prefix):], close, total_tokens,
        suffix_labels, opening, termination, tuple(sorted(forbidden)))


def assemble_self_think(example: PreparedExample, generated_think_ids: Sequence[int], *,
                        adapter_revision: str, generation_seconds: float) -> dict[str, Any]:
    """Attach a stopped, current-adapter sample; never repair malformed reasoning."""
    tokens = tuple(generated_think_ids)
    if not adapter_revision or not math.isfinite(generation_seconds) or generation_seconds < 0:
        raise ValueError("think_generation_metadata_invalid")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("think_token_invalid")
    if len(tokens) > example.allowed_think_tokens:
        raise ValueError("think_generation_budget_overflow")
    closes = _positions(tokens, example.think_close_ids)
    if closes != [len(tokens) - len(example.think_close_ids)]:
        raise ValueError("think_missing_or_repeated_closer")
    if _positions(tokens, example.think_open_ids):
        raise ValueError("think_repeated_opener")
    if set(tokens).intersection(example.forbidden_think_token_ids):
        raise ValueError("think_unexpected_eos_or_message_marker")
    ids = example.question_input_ids + tokens + example.answer_suffix_ids
    if len(ids) > example.total_tokens:
        raise ValueError("assembled_sequence_overflow")
    labels = (-100,) * (len(example.question_input_ids) + len(tokens)) + example.answer_suffix_labels
    metadata = dict(example.provenance)
    metadata.update(adapter_revision=adapter_revision, think_generation_seconds=generation_seconds,
                    generated_think_tokens=len(tokens), generated_think_sha256=_hash(tokens),
                    assembled_tokens=len(ids), source_think_supervised=False)
    return {"input_ids": list(ids), "labels": list(labels), "provenance": metadata}


def example_to_record(example: PreparedExample) -> dict[str, Any]:
    data = asdict(example)
    recovered = example.generation_input_ids is not None
    if not recovered:
        data.pop("generation_input_ids")
        data.pop("generation_context_tokens")
    return {"schema": 2 if recovered else SCHEMA, **data}


def example_from_record(record: dict[str, Any]) -> PreparedExample:
    data = dict(record)
    schema = data.pop("schema", None)
    if schema not in (SCHEMA, 2) or type(schema) is not int:
        raise ValueError("prepared_example_schema_mismatch")
    if schema == 2:
        full = data.get("generation_input_ids")
        cap = data.get("generation_context_tokens")
        if (not isinstance(full, (list, tuple)) or not full
                or any(type(value) is not int or value < 0 for value in full)
                or type(cap) is not int or cap < 2):
            raise ValueError("prepared_generation_context_invalid")
        data["generation_input_ids"] = tuple(full)
    elif "generation_input_ids" in data or "generation_context_tokens" in data:
        raise ValueError("prepared_generation_schema_mismatch")
    for key in ("question_input_ids", "answer_suffix_ids", "ordinary_input_ids", "ordinary_labels",
                "think_close_ids", "answer_suffix_labels", "think_open_ids", "think_termination_ids",
                "forbidden_think_token_ids"):
        data[key] = tuple(data[key])
    example = PreparedExample(**data)
    if schema == 2:
        if (len(example.generation_input_ids) + example.allowed_think_tokens > example.generation_context_tokens
                or _hash(example.generation_input_ids) != example.provenance.get("generation_question_sha256")
                or len(example.generation_input_ids) != example.provenance.get("generation_question_tokens")
                or example.provenance.get("generation_prefix_policy") != "full_original_history_no_truncation"):
            raise ValueError("prepared_generation_prefix_mismatch")
    if (len(example.ordinary_input_ids) != len(example.ordinary_labels)
            or len(example.answer_suffix_ids) != len(example.answer_suffix_labels)
            or not 2 <= example.total_tokens <= 32768
            or len(example.ordinary_input_ids) > example.total_tokens
            or len(example.question_input_ids) + example.allowed_think_tokens + len(example.answer_suffix_ids) > example.total_tokens):
        raise ValueError("prepared_example_budget_or_mask_invalid")
    if _hash(example.question_input_ids) != example.provenance.get("question_sha256") or _hash(example.answer_suffix_ids) != example.provenance.get("answer_sha256"):
        raise ValueError("prepared_example_token_digest_mismatch")
    if _hash(example.ordinary_input_ids) != example.provenance.get("ordinary_sha256"):
        raise ValueError("prepared_ordinary_digest_mismatch")
    ordinary_prefix = example.provenance.get("ordinary_prefix_tokens")
    separator = example.provenance.get("answer_separator_tokens")
    if (type(ordinary_prefix) is not int or not 0 < ordinary_prefix < len(example.ordinary_input_ids)
            or type(separator) is not int or not 0 <= separator < len(example.answer_suffix_ids)
            or example.ordinary_labels != (-100,) * ordinary_prefix + example.ordinary_input_ids[ordinary_prefix:]
            or example.answer_suffix_labels != (-100,) * separator + example.answer_suffix_ids[separator:]
            or example.allowed_think_tokens < len(example.think_close_ids)):
        raise ValueError("prepared_answer_mask_invalid")
    return example


def _validate_splits(examples: Iterable[PreparedExample]) -> None:
    groups, samples = {}, {}
    for example in examples:
        metadata = example.provenance
        split, group, sample = metadata.get("split"), metadata.get("task_group"), metadata.get("sample_hash")
        if split not in {"train", "heldout"} or not group or not sample:
            raise ValueError("prepared_split_provenance_missing")
        if groups.setdefault(group, split) != split or samples.setdefault(sample, split) != split:
            raise ValueError("prepared_train_heldout_prefix_leakage")


def validate_prepared_profile(examples: Sequence[PreparedExample], profile: str | Path) -> None:
    profile = Path(profile).resolve()
    expected = {name: file_sha256(profile / name) for name in
                ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    _validate_splits(examples)
    model_config = json.loads((profile / "config.json").read_text(encoding="utf-8"))
    model_capacity = model_config.get("text_config", model_config).get("max_position_embeddings")
    for example in examples:
        if example.generation_input_ids is not None:
            if type(model_capacity) is not int or example.generation_context_tokens > model_capacity:
                raise ValueError("prepared_generation_exceeds_model_capacity")
        if example.provenance.get("profile_binding") != expected:
            raise ValueError("prepared_profile_fingerprint_mismatch")
        if example.provenance.get("adapter_policy") != ADAPTER_POLICY:
            raise ValueError("prepared_adapter_policy_mismatch")
        if (example.provenance.get("ordinary_policy") != "empty_think_matched_template"
                or example.provenance.get("reasoning_effort") != "xhigh"):
            raise ValueError("prepared_control_prompt_policy_mismatch")


def load_prepared_examples(path: str | Path, *, split: str | None = None) -> list[PreparedExample]:
    if split is not None and split not in {"train", "heldout"}:
        raise ValueError("prepared_split_invalid")
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                examples.append(example_from_record(json.loads(line)))
    _validate_splits(examples)
    return [example for example in examples if split is None or example.provenance["split"] == split]


def iter_source_records(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank_source_row:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
                raise ValueError(f"source_requires_messages_array:{line_number}")
            yield line_number, record


def source_group_aliases(record: dict[str, Any]) -> tuple[str, ...]:
    explicit = []
    for container in (record, record.get("metadata", {}), record.get("provenance", {})):
        if isinstance(container, dict):
            for key in ("root_id", "rootId", "task_id", "taskId", "root_request_id"):
                if container.get(key) is not None:
                    explicit.append("root:" + _hash(str(container[key])))
    if explicit:
        # An explicit session/task identity is authoritative. Do not union it
        # with a repeated first-user prompt from unrelated trajectories.
        return tuple(dict.fromkeys(explicit))
    messages = record["messages"]
    first_user = next((message for message in messages if message.get("role") == "user"), None)
    if first_user is None:
        raise ValueError("source_without_user_root")
    return ("user:" + _hash(clean_message(first_user)["content"]),)


def group_source_records(path: str | Path, *, heldout_fraction: float, seed: str) -> dict[int, tuple[str, str]]:
    if not 0 < heldout_fraction < 1:
        raise ValueError("heldout_fraction_must_be_between_zero_and_one")
    parents: dict[str, str] = {}
    row_roots = {}

    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            previous = parents[key]
            parents[key] = root
            key = previous
        return root

    for line, record in iter_source_records(path):
        aliases = source_group_aliases(record)
        root = find(aliases[0])
        for alias in aliases[1:]:
            other = find(alias)
            if root != other:
                root, other = sorted((root, other))
                parents[other] = root
        row_roots[line] = aliases[0]
    result = {}
    for line, alias in row_roots.items():
        group = _hash(find(alias))
        bucket = int(_hash([seed, group])[:16], 16) / 2**64
        result[line] = (group, "heldout" if bucket < heldout_fraction else "train")
    return result


def verify_self_think_dataflow(tokenizer) -> dict[str, Any]:
    """Explicit CPU-only synthetic checks, callable by the integration owner."""
    messages = [{"role": "system", "content": "Use tools when needed."},
                {"role": "user", "content": "Inspect a local file."},
                {"role": "assistant", "content": '<think>source-only-secret</think>\n<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"text":"literal <think> stays"}</tool_result>'},
                {"role": "assistant", "content": "<think>future-source-secret</think>\nThe file is present."}]
    example = prepare_example(tokenizer, messages, 4)
    prompt_text = tokenizer.decode(example.question_input_ids, skip_special_tokens=False)
    assert "source-only-secret" not in prompt_text and "future-source-secret" not in prompt_text
    assert "The file is present." not in prompt_text and "literal <think> stays" in prompt_text
    assert set(asdict(generation_plan(example))) == {"question_input_ids", "allowed_think_tokens", "think_close_ids"}
    generated = _ids(tokenizer, "I will use the observed tool result.\n</think>")
    assembled = assemble_self_think(example, generated, adapter_revision="synthetic-revision-0", generation_seconds=0.0)
    assert all(label == -100 for label in assembled["labels"][:len(example.question_input_ids) + len(generated)])
    assert any(label != -100 for label in assembled["labels"])
    assert "source-only-secret" not in tokenizer.decode(example.ordinary_input_ids, skip_special_tokens=False)
    assert _ids(tokenizer, tokenizer.decode(assembled["input_ids"], skip_special_tokens=False)) == tuple(assembled["input_ids"])
    assert example_from_record(example_to_record(example)) == example
    rendered_messages = [clean_message(message) for message in messages]
    assert example.ordinary_input_ids == _ids(tokenizer, _render(tokenizer, rendered_messages, thinking=True, generation=False))
    assert example.provenance["ordinary_policy"] == "empty_think_matched_template"
    rendered_messages[-1]["reasoning_content"] = "I will use the observed tool result."
    assert tuple(assembled["input_ids"]) == _ids(tokenizer, _render(tokenizer, rendered_messages, thinking=True, generation=False))
    direct_close = _ids(tokenizer, "I used the tool result.</think>")
    assert assemble_self_think(example, direct_close, adapter_revision="r0", generation_seconds=0)["input_ids"]
    early = prepare_example(tokenizer, messages, 2)
    assert "literal <think> stays" not in tokenizer.decode(early.question_input_ids, skip_special_tokens=False)
    assert clean_message(messages[3])["content"] == messages[3]["content"]
    failures = {}
    for name, bad in {
        "missing_closer": _ids(tokenizer, "unfinished"),
        "repeated_closer": generated + generated,
        "repeated_opener": _ids(tokenizer, "<think>nested\n</think>"),
        "unexpected_eos": (example.forbidden_think_token_ids[0],) + generated,
        "budget_overflow": generated * (example.allowed_think_tokens // len(generated) + 1),
    }.items():
        try:
            assemble_self_think(example, bad, adapter_revision="r0", generation_seconds=0)
        except ValueError:
            failures[name] = True
        else:
            raise AssertionError(name)
    try:
        prepare_example(tokenizer, messages, 4, total_tokens=2)
    except ValueError:
        failures["whole_prefix_overflow"] = True
    else:
        raise AssertionError("whole_prefix_overflow")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mid-lora-data-gate-") as directory:
        fixture = Path(directory) / "synthetic.jsonl"
        rows = [
            {"root_id": "one", "messages": [{"role": "user", "content": "same"}]},
            {"root_id": "two", "messages": [{"role": "user", "content": "same"}]},
            {"root_id": "two", "messages": [{"role": "user", "content": "different"}]},
        ]
        fixture.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        grouped = group_source_records(fixture, heldout_fraction=0.2, seed="synthetic")
        assert grouped[2][0] != grouped[1][0]
        assert grouped[2][0] == grouped[3][0]
        assert grouped[2][1] == grouped[3][1]
        assert grouped == group_source_records(fixture, heldout_fraction=0.2, seed="synthetic")
    return {"cpu_synthetic_only": True, "template_roundtrip": True,
            "causal_gold_exclusion": True, "source_think_removed": True,
            "answer_only_masks": True, "root_user_union_split": True, "rejections": failures}
