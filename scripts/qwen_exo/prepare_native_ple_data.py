"""CPU-only native Flash-Next PLE training-data preparation (never training).

Inventory: --inventory --src SOURCE --output PRIVATE_DIR
Prepare: --prepare --src SOURCE --model NATIVE_PROFILE --output PRIVATE_DIR
         [--split-map PRIVATE.json | --prior-records OLD_RECORDS.jsonl]
Check: --check --output PRIVATE_DIR
Smoke: --synthetic-smoke --model NATIVE_PROFILE (temporary synthetic data only)

A split map has schema_version=1 and records [{source_line, message_sha256,
 session_id, task_id, split: train|heldout}]. IDs must come from an authoritative
export, not be invented from line numbers. Every session, task, and duplicate
conversation must remain in one split. Without a trusted map, the conservative
first_user_prompt_sha256 grouping joins exact first-user prompts plus preceding
system instructions and tool schemas; semantic task independence is NOT proven.
--prior-records freezes every old heldout record and its entire causal group.
No raw content, tokens, paths, or source exceptions are printed.

Window labels are unshifted: labels[t]=input_ids[t] on assistant text (including
Think/tool-call closing syntax and native turn EOS), else -100. Consumers shift
logits[:-1] against labels[1:]. A one-token overlap preserves every causal target
exactly once; each window also carries the real preceding n-gram history. Full
sessions are rendered before windowing; previous transformer/GDN context is NOT
magically retained across bounded windows. The manifest reports this limitation.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

# Must be set before importing tokenizer/Torch dependencies. No model is loaded.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "special_tokens_map.json", "added_tokens.json")


class PreparationError(ValueError):
    """Messages are stable codes and structural counts, never corpus content."""


def require(condition, code):
    if not condition:
        raise PreparationError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                require(isinstance(record, dict), "source_record_not_object")
                messages = record.get("messages")
                require(isinstance(messages, list) and messages, "source_messages_missing")
                yield line, record
            except (ValueError, TypeError, KeyError):
                raise PreparationError(f"source_structure_invalid_line_{line}") from None


def inventory_source(path):
    summary = []
    roles = collections.Counter()
    for line, record in records(path):
        messages = record["messages"]
        require(all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in messages),
                f"message_structure_invalid_line_{line}")
        roles.update(m["role"] for m in messages)
        first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
        require(first_user is not None, f"first_user_missing_line_{line}")
        prefix = {"system": [m for m in messages[:first_user] if m["role"] == "system"],
                  "first_user": messages[first_user], "tools": record.get("tools")}
        summary.append({"source_line": line, "message_sha256": digest(messages),
                        "legacy_message_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                        "first_user_prompt_sha256": digest(prefix),
                        "record_sha256": digest(record), "messages": len(messages)})
    require(summary, "empty_source")
    return {"schema_version": SCHEMA_VERSION, "status": "inventory_ready",
            "source_sha256": file_sha256(path), "source_records": len(summary),
            "causal_prompt_groups": len({r["first_user_prompt_sha256"] for r in summary}),
            "source_roles": dict(sorted(roles.items())), "records": summary}


def private_output(path):
    path = Path(path).resolve()
    require(not path.is_relative_to(REPO_ROOT), "output_must_be_outside_public_repository")
    require(not path.exists(), "output_already_exists")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_splits(inventory, split_map):
    require(split_map.get("schema_version") == SCHEMA_VERSION, "split_map_schema_mismatch")
    require(split_map.get("source_sha256") == inventory["source_sha256"], "split_map_source_mismatch")
    entries = split_map.get("records")
    require(isinstance(entries, list), "split_map_records_missing")
    mapped = {}
    groups = {"session_id": {}, "task_id": {}, "message_sha256": {}}
    for item in entries:
        require(isinstance(item, dict), "split_map_entry_invalid")
        line = item.get("source_line")
        require(isinstance(line, int) and line > 0 and line not in mapped, "split_map_line_invalid")
        require(item.get("split") in ("train", "heldout"), "split_map_split_invalid")
        for field in ("session_id", "task_id", "message_sha256"):
            value = item.get(field)
            require(isinstance(value, str) and bool(value), "split_map_group_missing")
            require(groups[field].get(value, item["split"]) == item["split"], "split_map_group_leakage")
            groups[field][value] = item["split"]
        # Hash private identifiers so they cannot leak through metadata logs.
        mapped[line] = {"split": item["split"], "session_sha256": digest(item["session_id"]),
                        "task_sha256": digest(item["task_id"]), "message_sha256": item["message_sha256"]}
    require(len(mapped) == inventory["source_records"], "split_map_record_count_mismatch")
    for item in inventory["records"]:
        require(item["source_line"] in mapped, "split_map_record_missing")
        require(mapped[item["source_line"]]["message_sha256"] == item["message_sha256"],
                "split_map_message_hash_mismatch")
    require({m["split"] for m in mapped.values()} == {"train", "heldout"}, "both_splits_required")
    return mapped


def derived_splits(inventory, heldout_groups=15, prior_records=None):
    groups = sorted({r["first_user_prompt_sha256"] for r in inventory["records"]})
    require(len(groups) > 1, "only_one_causal_prompt_group_cannot_split")
    require(heldout_groups > 0, "heldout_groups_out_of_range")
    heldout = set()
    prior_hash = None
    if prior_records is not None:
        prior_hash = file_sha256(prior_records)
        current = {r["source_line"]: r for r in inventory["records"]}
        seen = set()
        with Path(prior_records).open(encoding="utf-8") as handle:
            for raw in handle:
                old = json.loads(raw)
                line = old["source_line"]
                require(line in current and line not in seen, "prior_record_line_mismatch")
                require(old["message_sha256"] == current[line]["legacy_message_sha256"],
                        "prior_record_source_hash_mismatch")
                require(old["split"] in ("wt_full_write", "wt_full_heldout"), "prior_record_split_unknown")
                seen.add(line)
                if old["split"] == "wt_full_heldout":
                    heldout.add(current[line]["first_user_prompt_sha256"])
        require(len(seen) == inventory["source_records"] and heldout, "prior_record_coverage_mismatch")
    else:
        require(heldout_groups < len(groups), "heldout_groups_exhaust_corpus")
        heldout = set(groups[-heldout_groups:])
    require(len(heldout) < len(groups), "prior_heldout_groups_exhaust_corpus")
    return {"schema_version": SCHEMA_VERSION, "source_sha256": inventory["source_sha256"],
            "provenance": {"mode": "first_user_prompt_sha256", "verified_semantic_task_independence": False,
                           "group_definition": "exact_canonical_first_user_plus_prior_system_and_tools",
                           "limitations": "Different prompt wording or changed system instructions can split the same semantic task.",
                           "causal_prompt_groups": len(groups), "heldout_groups": len(heldout),
                           "prior_records_sha256": prior_hash, "old_heldout_groups_frozen": bool(prior_records)},
            "records": [{"source_line": r["source_line"], "message_sha256": r["message_sha256"],
                         "session_id": r["first_user_prompt_sha256"], "task_id": r["first_user_prompt_sha256"],
                         "split": "heldout" if r["first_user_prompt_sha256"] in heldout else "train"}
                        for r in inventory["records"]]}


def normalize(record):
    """Reuse study normalization; reject unsupported multimodal/developer records."""
    fixed = []
    for source in record["messages"]:
        message = dict(source)
        require(message.get("role") in ("system", "user", "assistant", "tool"), "unsupported_role")
        content = message.get("content")
        if isinstance(content, list):
            require(all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
                        for p in content), "multimodal_source_not_supported")
        else:
            require(content is None or isinstance(content, str), "content_shape_invalid")
        require(message.get("reasoning_content") is None or isinstance(message["reasoning_content"], str),
                "reasoning_shape_invalid")
        if message.get("tool_calls"):
            calls = []
            for original in message["tool_calls"]:
                call = dict(original)
                function = dict(call.get("function", call))
                args = function.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args) if args else {}
                require(isinstance(args, dict), "tool_arguments_not_object")
                require(isinstance(function.get("name"), str), "tool_name_missing")
                function["arguments"] = args
                call["function"] = function
                calls.append(call)
            message["tool_calls"] = calls
        fixed.append(message)
    return fixed


def tracked_template(template):
    import jinja2
    tree = jinja2.Environment().parse(template)
    branches = []
    for node in tree.find_all(jinja2.nodes.If):
        test = node.test
        if not isinstance(test, jinja2.nodes.Compare) or len(test.ops) != 1:
            continue
        expr, operand = test.expr, test.ops[0]
        if (isinstance(expr, jinja2.nodes.Getattr) and expr.attr == "role"
                and isinstance(expr.node, jinja2.nodes.Name) and expr.node.name == "message"
                and operand.op == "eq" and isinstance(operand.expr, jinja2.nodes.Const)):
            branches.append((operand.expr.value, node.lineno))
    assistants = [line for role, line in branches if role == "assistant"]
    require(len(assistants) == 1, "unsupported_native_assistant_branch")
    start = assistants[0]
    following = [line for _, line in branches if line > start]
    require(following, "unsupported_native_assistant_branch_end")
    end = min(following)
    lines = template.splitlines(keepends=True)
    require(lines[start - 1].rstrip().endswith("%}"), "unsupported_multiline_role_branch")
    return ("".join(lines[:start]) + "{%- generation -%}\n" + "".join(lines[start:end - 1])
            + "{%- endgeneration -%}\n" + "".join(lines[end - 1:]))


def load_native(model):
    from transformers import AutoTokenizer
    from qwen_exo_booster.native_ple_knowledge import NativePLEIdentity
    identity = NativePLEIdentity.from_profile(model)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
    require(tokenizer.is_fast and isinstance(tokenizer.chat_template, str), "native_fast_template_required")
    require(tokenizer.convert_tokens_to_ids("<|endoftext|>") == identity.hash.eos_id,
            "tokenizer_native_hash_eos_mismatch")
    fingerprints = {name: file_sha256(Path(model) / name) for name in TOKENIZER_FILES
                    if (Path(model) / name).is_file()}
    require("tokenizer.json" in fingerprints and "tokenizer_config.json" in fingerprints,
            "tokenizer_identity_files_missing")
    return identity, tokenizer, {"files": fingerprints, "files_sha256": digest(fingerprints),
                                 "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                                 "tokenizer_class": type(tokenizer).__name__}


def render_complete(tokenizer, record):
    import torch
    from transformers.utils.chat_template_utils import render_jinja_template
    messages = normalize(record)
    kwargs = {"add_generation_prompt": False, "preserve_thinking": True,
              "enable_thinking": True, "reasoning_effort": "xhigh"}
    if record.get("tools") is not None:
        require(isinstance(record["tools"], list), "tool_schema_shape_invalid")
        kwargs["tools"] = record["tools"]
    texts, spans = render_jinja_template(
        [messages], chat_template=tracked_template(tokenizer.chat_template),
        return_assistant_tokens_mask=True, **kwargs, **tokenizer.special_tokens_map)
    text = texts[0]
    require(text == tokenizer.apply_chat_template(messages, tokenize=False, **kwargs),
            "tracking_changed_native_render")
    encoded = tokenizer(text, add_special_tokens=False)
    require(encoded["input_ids"] == tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                                  **kwargs)["input_ids"],
            "tracking_changed_native_tokens")
    ids = torch.tensor(encoded["input_ids"], dtype=torch.int64)
    mask = torch.zeros(ids.numel(), dtype=torch.bool)
    require(len(spans[0]) == sum(m["role"] == "assistant" for m in messages), "assistant_span_count_mismatch")
    header, suffix = "<|im_start|>assistant\n", "<|im_end|>\n"
    for begin, end in spans[0]:
        require(text[begin:end].startswith(header) and text[begin:end].endswith(suffix),
                "assistant_boundary_mismatch")
        # Track real Jinja branches, not content that happens to quote ChatML.
        first = encoded.char_to_token(begin + len(header))
        last = encoded.char_to_token(end - 1)
        require(first is not None and last is not None, "assistant_token_offsets_missing")
        mask[first:last + 1] = True
    require(ids.numel() > 1 and not bool(mask[0]), "source_initial_target_invalid")
    return ids, mask, hashlib.sha256(text.encode()).hexdigest()


def window_ranges(length, max_sequence):
    require(2 <= max_sequence <= 32768, "max_sequence_out_of_range")
    for start in range(0, length - 1, max_sequence - 1):
        yield start, min(start + max_sequence, length)


def make_window(ids, mask, start, end, identity):
    import torch
    from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys
    context = identity.hash.ngram_size - 1
    history = torch.full((context,), identity.hash.eos_id, dtype=torch.int64)
    prior = ids[max(0, start - context):start]
    if prior.numel():
        history[-prior.numel():] = prior
    tokens = ids[start:end].clone()
    supervision = mask[start:end].clone()
    supervision[0] = False
    labels = torch.where(supervision, tokens, -100)
    rows = native_ple_row_keys(tokens, identity, history=history)
    require(rows.dtype == torch.int64 and rows.shape == (tokens.numel(), len(identity.hash.head_sizes)),
            "native_row_key_shape_mismatch")
    return {"input_ids": tokens, "labels": labels, "loss_mask": supervision,
            "row_keys": rows, "ngram_history": history}


def prepare(args):
    import torch
    torch.set_num_threads(args.cpu_threads)
    output = private_output(args.output)
    require(args.src and args.model, "source_and_native_model_required")
    inventory = inventory_source(args.src)
    split_document = (json.loads(args.split_map.read_text(encoding="utf-8")) if args.split_map else
                      derived_splits(inventory, args.heldout_groups, args.prior_records))
    mapping = validate_splits(inventory, split_document)
    identity, tokenizer, tokenizer_info = load_native(args.model)
    output.mkdir(parents=True)
    write_json(output / "split-map.json", split_document)
    files = {}
    stats = {s: collections.Counter() for s in ("train", "heldout")}
    window_index = []
    record_index = []
    written_bytes = 0
    summary = {r["source_line"]: r for r in inventory["records"]}
    for line, record in records(args.src):
        require(digest(record) == summary[line]["record_sha256"], "source_changed_between_passes")
        try:
            ids, mask, render_hash = render_complete(tokenizer, record)
        except Exception as error:
            raise PreparationError(f"render_failed_line_{line}_{type(error).__name__}") from None
        # Append a non-supervised native hash EOS; never mix independent sessions.
        ids = torch.cat((ids, torch.tensor([identity.hash.eos_id], dtype=torch.int64)))
        mask = torch.cat((mask, torch.tensor([False])))
        split = mapping[line]["split"]
        state = stats[split]
        covered = 0
        row_digest = hashlib.sha256()
        count = 0
        for start, end in window_ranges(ids.numel(), args.max_sequence):
            if not bool(mask[start + 1:end].any()):
                state["non_target_windows_skipped"] += 1
                continue
            payload = make_window(ids, mask, start, end, identity)
            estimate = sum(t.numel() * t.element_size() for t in payload.values()) + 8192
            require(written_bytes + estimate <= args.max_output_bytes, "output_byte_budget_exceeded")
            name = f"{split}/{line:07d}-{start:09d}.pt"
            destination = output / name
            destination.parent.mkdir(exist_ok=True)
            torch.save(payload, destination)
            size = destination.stat().st_size
            written_bytes += size
            require(written_bytes <= args.max_output_bytes, "output_byte_budget_exceeded")
            files[name] = {"bytes": size, "sha256": file_sha256(destination)}
            targets = int(payload["loss_mask"].sum())
            covered += targets
            count += 1
            state["windows"] += 1
            state["forwarded_tokens"] += end - start
            state["assistant_targets"] += targets
            state["context_prefix_tokens_not_forwarded"] += start
            row_digest.update(payload["row_keys"].numpy().tobytes())
            entry = {"file": name, "source_line": line, "session_sha256": mapping[line]["session_sha256"],
                     "task_sha256": mapping[line]["task_sha256"], "split": split,
                     "start": start, "end": end, "tokens": end - start, "assistant_targets": targets,
                     "row_keys_sha256": hashlib.sha256(payload["row_keys"].numpy().tobytes()).hexdigest()}
            window_index.append(entry)
        require(covered == int(mask.sum()), "causal_target_coverage_mismatch")
        state["zero_target_records"] += int(covered == 0)
        state["records"] += 1
        state["messages"] += len(record["messages"])
        state["rendered_tokens"] += ids.numel() - 1
        state["windowed_records"] += int(ids.numel() > args.max_sequence)
        record_index.append({**summary[line], **mapping[line], "render_sha256": render_hash,
                             "tokens_with_boundary_eos": ids.numel(), "assistant_targets": covered,
                             "windows": count, "row_keys_sha256": row_digest.hexdigest()})
    require(file_sha256(args.src) == inventory["source_sha256"], "source_changed_during_preparation")
    require(all(stats[s]["assistant_targets"] > 0 for s in stats), "empty_supervised_split")
    report = {"schema_version": SCHEMA_VERSION, "status": "prepared_not_started", "training_started": False,
              "generation_started": False, "source_path": str(args.src.resolve()),
              "model_path": str(args.model.resolve()), "source_sha256": inventory["source_sha256"],
              "source_records": inventory["source_records"], "source_roles": inventory["source_roles"],
              "split_map_sha256": file_sha256(output / "split-map.json"), "split_map_semantic_sha256": digest(split_document),
              "split_provenance": split_document.get("provenance"), "split_assignments_sha256": digest(mapping),
              "causal_prompt_groups": inventory["causal_prompt_groups"],
              "native_identity": identity.to_dict(), "native_identity_sha256": identity.fingerprint(),
              "tokenizer": tokenizer_info, "code_sha256": file_sha256(__file__),
              "max_sequence": args.max_sequence, "truncated_records": 0, "truncated_targets": 0,
              "dropped_messages": 0, "preserve_thinking": True,
              "window_policy": "full_render_then_one_token_overlap_target_windows",
              "prior_transformer_gdn_state": "not_retained_across_windows",
              "ngram_history": "exact_previous_ngram_size_minus_one_ids_eos_padded",
              "label_contract": "unshifted_token_labels_consumer_shifts_logits_minus_last_against_labels_after_first",
              "assistant_turn_eos_supervised": True, "boundary_eos_supervised": False,
              "row_key_contract": {"dtype": "int64", "shape": ["tokens", len(identity.hash.head_sizes)],
                                   "head_dim": identity.head_dim, "causal": True,
                                   "helper": "qwen_exo_booster.native_ple_knowledge.native_ple_row_keys"},
              "tensor_contract": {"input_ids": "int64[L]", "labels": "int64[L]",
                                  "loss_mask": "bool[L]", "row_keys": "int64[L,H]",
                                  "ngram_history": "int64[ngram_size-1]"},
              "splits": {s: dict(v) for s, v in stats.items()}, "records": record_index,
              "windows": window_index, "files": files, "output_bytes": written_bytes,
              "evaluation": {"modes": ["off", "real", "shuffled"], "shuffle_seed": args.shuffle_seed,
                             "matched_inputs_labels_files": [w["file"] for w in window_index if w["split"] == "heldout"],
                             "native_backbone_base_ple_reader": "frozen_identical_all_modes",
                             "off": "disable_sparse_delta_only",
                             "real": "delta_lookup_uses_row_keys",
                             "shuffled": "SparseNativePLEDelta.set_shuffle_seed permutes delta values within each hash head",
                             "base_ple_lookup": "always_unmodified_row_keys",
                             "quality_evaluated": False}}
    write_json(output / "manifest.json", report)
    return report


def check(output, cpu_threads=2):
    import torch
    from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys
    torch.set_num_threads(cpu_threads)
    report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    require(report.get("schema_version") == SCHEMA_VERSION and report.get("status") == "prepared_not_started",
            "manifest_status_or_schema_mismatch")
    require(report.get("training_started") is False and report.get("generation_started") is False,
            "manifest_execution_state_invalid")
    require(file_sha256(report["source_path"]) == report["source_sha256"], "source_hash_changed")
    split_document = json.loads((output / "split-map.json").read_text(encoding="utf-8"))
    require(file_sha256(output / "split-map.json") == report["split_map_sha256"]
            and digest(split_document) == report["split_map_semantic_sha256"], "prepared_split_map_changed")
    current_inventory = inventory_source(report["source_path"])
    require(digest(validate_splits(current_inventory, split_document)) == report["split_assignments_sha256"],
            "prepared_split_assignments_changed")
    identity, _, tokenizer_info = load_native(Path(report["model_path"]))
    require(identity.fingerprint() == report["native_identity_sha256"], "native_identity_changed")
    require(tokenizer_info == report["tokenizer"], "tokenizer_identity_changed")
    require(file_sha256(__file__) == report["code_sha256"], "preparation_code_changed")
    counts = collections.Counter()
    for window in report["windows"]:
        name = window["file"]
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "manifest_path_invalid")
        path = output / name
        require(path.stat().st_size == report["files"][name]["bytes"]
                and file_sha256(path) == report["files"][name]["sha256"], "prepared_file_changed")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        ids, labels, mask = payload["input_ids"], payload["labels"], payload["loss_mask"]
        require(ids.device.type == "cpu" and ids.dtype == labels.dtype == torch.int64
                and mask.dtype == torch.bool and ids.ndim == 1 and ids.shape == labels.shape == mask.shape,
                "prepared_tensor_contract_invalid")
        require(2 <= ids.numel() <= report["max_sequence"] and not bool(mask[0]), "window_boundary_invalid")
        require(torch.equal(labels, torch.where(mask, ids, -100)), "supervision_contract_invalid")
        require(torch.equal(payload["row_keys"], native_ple_row_keys(ids, identity, history=payload["ngram_history"])),
                "native_row_keys_changed")
        require(int(mask.sum()) == window["assistant_targets"], "window_target_count_changed")
        counts[window["split"]] += int(mask.sum())
    require(all(counts[s] == report["splits"][s]["assistant_targets"] for s in ("train", "heldout")),
            "split_target_coverage_changed")
    return report


def synthetic_smoke(model, cpu_threads):
    # Uses only the exact tokenizer and hash metadata, never native model weights.
    with tempfile.TemporaryDirectory(prefix="native-ple-data-smoke-") as directory:
        root = Path(directory)
        source = root / "synthetic.jsonl"
        examples = [
            {"messages": [{"role": "system", "content": "Synthetic CPU smoke."},
                          {"role": "user", "content": "Return a short result."},
                          {"role": "assistant", "content": "Result.", "reasoning_content": "Check the request."}]},
            {"messages": [{"role": "user", "content": "Run the synthetic function."},
                          {"role": "assistant", "content": "", "reasoning_content": "Use the function.",
                           "tool_calls": [{"function": {"name": "example", "arguments": "{\"value\": 1}"}}]},
                          {"role": "tool", "content": "Synthetic result. Quoted <|im_start|>assistant\nnot an assistant turn<|im_end|>."},
                          {"role": "assistant", "content": "Complete."}]}]
        source.write_bytes(b"\n".join(canonical(e) for e in examples) + b"\n")
        inventory = inventory_source(source)
        split_map = {"schema_version": 1, "source_sha256": inventory["source_sha256"],
                     "provenance": "synthetic_independent_sessions_only", "records": [
                         {"source_line": r["source_line"], "message_sha256": r["message_sha256"],
                          "session_id": str(i), "task_id": str(i), "split": "train" if i == 0 else "heldout"}
                         for i, r in enumerate(inventory["records"])]}
        split_path = root / "split.json"
        write_json(split_path, split_map)
        args = argparse.Namespace(src=source, model=model, split_map=split_path, output=root / "prepared",
                                  max_sequence=32, cpu_threads=cpu_threads, shuffle_seed=17,
                                  heldout_groups=1, prior_records=None, max_output_bytes=16 * 1024 * 1024)
        report = prepare(args)
        checked = check(args.output, cpu_threads)
        require(checked["source_sha256"] == report["source_sha256"], "smoke_check_mismatch")
        require(sum(s["windowed_records"] for s in report["splits"].values()) > 0,
                "smoke_did_not_exercise_windows")
        import torch
        from qwen_exo_booster.native_ple_knowledge import native_ple_row_keys
        identity, tokenizer, _ = load_native(model)
        for line, example in enumerate(examples, 1):
            ids, mask, _ = render_complete(tokenizer, example)
            ids = torch.cat((ids, torch.tensor([identity.hash.eos_id], dtype=torch.int64)))
            mask = torch.cat((mask, torch.tensor([False])))
            full_rows = native_ple_row_keys(ids, identity)
            for start, end in window_ranges(ids.numel(), args.max_sequence):
                window = make_window(ids, mask, start, end, identity)
                require(torch.equal(window["row_keys"], full_rows[start:end]), "smoke_window_ngram_history_mismatch")
            # A preceding completed session cannot affect this session's hash rows.
            prefixed = torch.cat((ids, ids))
            require(torch.equal(native_ple_row_keys(prefixed, identity)[ids.numel():], full_rows),
                    "smoke_session_eos_reset_mismatch")
        derived = derived_splits(inventory, heldout_groups=1)
        validate_splits(inventory, derived)
        # Leakage is rejected even when conversations themselves differ.
        split_map["records"][1]["task_id"] = split_map["records"][0]["task_id"]
        try:
            validate_splits(inventory, split_map)
        except PreparationError:
            pass
        else:
            raise PreparationError("smoke_task_leakage_not_rejected")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("inventory", "prepare", "check", "synthetic-smoke"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--prior-records", type=Path, help="Freeze old heldout records and their causal prompt groups")
    parser.add_argument("--heldout-groups", type=int, default=15)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-sequence", type=int, default=32768)
    parser.add_argument("--max-output-bytes", type=int, default=64 * 1024 ** 3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=20261005)
    args = parser.parse_args()
    try:
        require(1 <= args.cpu_threads <= 4, "cpu_threads_out_of_range")
        require(2 <= args.max_sequence <= 32768, "max_sequence_out_of_range")
        if args.inventory:
            require(args.src and args.output, "inventory_source_output_required")
            report = inventory_source(args.src)
            output = private_output(args.output)
            output.mkdir(parents=True)
            write_json(output / "source-inventory.json", report)
        elif args.synthetic_smoke:
            require(args.model, "smoke_native_profile_required")
            report = synthetic_smoke(args.model, args.cpu_threads)
        elif args.check:
            require(args.output, "check_output_required")
            report = check(args.output, args.cpu_threads)
        else:
            require(args.output, "prepare_output_required")
            report = prepare(args)
        # Deliberately exclude file paths, IDs, source text, and token values.
        print(json.dumps({"status": report["status"], "schema_version": report["schema_version"],
                          "source_records": report["source_records"], "source_sha256": report["source_sha256"],
                          "splits": report.get("splits"), "truncated_records": report.get("truncated_records"),
                          "causal_prompt_groups": report.get("causal_prompt_groups"),
                          "training_started": False, "generation_started": False}, sort_keys=True))
    except Exception as error:
        code = str(error) if isinstance(error, PreparationError) else type(error).__name__
        print(json.dumps({"status": "failed_closed", "error": code, "training_started": False,
                          "generation_started": False}, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
