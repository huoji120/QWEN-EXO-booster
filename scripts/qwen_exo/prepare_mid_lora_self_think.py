"""Prepare private full-prefix Claude answer-only/self-Think examples on CPU.

Reads only the explicitly supplied repaired JSONL. It neither generates Think,
loads model weights, performs optimization, nor changes a running service.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]


def _jsonline(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def prepare(profile, source, output, *, total_tokens=32768, max_think_tokens=8192,
            min_think_tokens=1, heldout_fraction=0.1, seed="mid-lora-self-think-v1",
            train_limit=None, heldout_limit=None, verify_code=False):
    from qwen_exo_booster.native_mid_lora_self_think import (
        example_to_record, file_sha256, group_source_records, iter_source_records,
        load_profile_tokenizer, prepare_example, verify_self_think_dataflow,
    )
    source, profile, output = Path(source).resolve(), Path(profile).resolve(), Path(output).resolve()
    if output.is_relative_to(REPO):
        raise ValueError("Private training artifacts must remain outside the source tree")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing preparation directory")
    if not source.is_file():
        raise FileNotFoundError("Explicit repaired source JSONL does not exist")
    if not 2 <= total_tokens <= 32768 or not 1 <= min_think_tokens <= max_think_tokens:
        raise ValueError("Invalid total/Think token budgets")
    if any(limit is not None and limit < 1 for limit in (train_limit, heldout_limit)):
        raise ValueError("Experiment sample limits must be positive or omitted")
    source_hash = file_sha256(source)
    tokenizer, binding = load_profile_tokenizer(profile)
    cpu_gate = verify_self_think_dataflow(tokenizer) if verify_code else None
    groups = group_source_records(source, heldout_fraction=heldout_fraction, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    counts, excluded, seen, selected = Counter(), Counter(), set(), {"train": [], "heldout": []}
    limits = {"train": train_limit, "heldout": heldout_limit}
    sampling = train_limit is not None or heldout_limit is not None
    prepared_partial = output / "prepared.jsonl.partial"
    index_partial = output / "index.jsonl.partial"
    exclusions_partial = output / "excluded.jsonl.partial"
    experiment_partial = output / "experiment.jsonl.partial"
    # The full prepared stream is never capped by experimental sample limits.
    with prepared_partial.open("x", encoding="utf-8") as prepared, \
            index_partial.open("x", encoding="utf-8") as index, \
            exclusions_partial.open("x", encoding="utf-8") as exclusions:
        for line_number, record in iter_source_records(source):
            counts["source_rows"] += 1
            group, split = groups[line_number]
            messages = record["messages"]
            for target_index, target in enumerate(messages):
                if not isinstance(target, dict) or target.get("role") != "assistant":
                    continue
                counts["assistant_targets"] += 1
                provenance = {"source_sha256": source_hash, "source_line": line_number,
                              "task_group": group, "split": split, "split_seed": seed,
                              "profile_binding": binding}
                try:
                    example = prepare_example(tokenizer, messages, target_index,
                        total_tokens=total_tokens, max_think_tokens=max_think_tokens,
                        min_think_tokens=min_think_tokens, provenance=provenance)
                except ValueError as exc:
                    # Do not echo source text or third-party template error payloads.
                    reason = str(exc)
                    if not reason or len(reason) > 100 or any(char not in "abcdefghijklmnopqrstuvwxyz_" for char in reason):
                        reason = "template_or_source_validation_failed"
                    excluded[reason] += 1
                    _jsonline(exclusions, {"source_line": line_number, "target_index": target_index,
                                          "task_group": group, "reason": reason})
                    if reason in {"answer_exceeds_total_budget", "complete_prefix_insufficient_think_budget", "ordinary_complete_prefix_overflow"}:
                        # With preserve_thinking=True and the same system template,
                        # later complete prefixes include this entire answer plus
                        # additional message boundaries. They cannot fit again.
                        for later_index in range(target_index + 1, len(messages)):
                            if messages[later_index].get("role") != "assistant":
                                continue
                            counts["assistant_targets"] += 1
                            excluded["complete_prefix_overflow_after_budget_boundary"] += 1
                            _jsonline(exclusions, {"source_line": line_number, "target_index": later_index,
                                "task_group": group, "reason": "complete_prefix_overflow_after_budget_boundary",
                                "after_target_index": target_index, "budget_boundary_reason": reason})
                        break
                    continue
                sample_hash = example.provenance["sample_hash"]
                if sample_hash in seen:
                    excluded["duplicate_source_target"] += 1
                    _jsonline(exclusions, {"source_line": line_number, "target_index": target_index,
                                          "sample_hash": sample_hash, "reason": "duplicate_source_target"})
                    continue
                seen.add(sample_hash)
                counts["prepared"] += 1
                counts[split] += 1
                serialized = example_to_record(example)
                _jsonline(prepared, serialized)
                _jsonline(index, {"prepared_line": counts["prepared"], **example.provenance,
                                 "allowed_think_tokens": example.allowed_think_tokens})
                if sampling and limits[split] is not None:
                    rank = int(hashlib.sha256((seed + sample_hash).encode()).hexdigest(), 16)
                    heap = selected[split]
                    heapq.heappush(heap, (-rank, counts["prepared"], serialized))
                    if len(heap) > limits[split]:
                        heapq.heappop(heap)
    if file_sha256(source) != source_hash:
        raise ValueError("Source changed during preparation; partial outputs are unusable")
    if not counts["prepared"]:
        raise ValueError("No complete-prefix examples fit; inspect excluded.jsonl.partial")
    sampled_counts = Counter()
    if sampling:
        kept = {row[2]["provenance"]["sample_hash"] for heap in selected.values() for row in heap}
        with prepared_partial.open(encoding="utf-8") as prepared, experiment_partial.open("x", encoding="utf-8") as experiment:
            for line in prepared:
                record = json.loads(line)
                split = record["provenance"]["split"]
                if limits[split] is None or record["provenance"]["sample_hash"] in kept:
                    experiment.write(line)
                    sampled_counts[split] += 1
    files = {"prepared": prepared_partial, "index": index_partial, "excluded": exclusions_partial}
    if sampling:
        files["experiment"] = experiment_partial
    artifacts = {}
    for name, partial in files.items():
        final = partial.with_suffix("")
        partial.rename(final)
        artifacts[name] = {"path": str(final), "sha256": file_sha256(final)}
    group_counts = Counter(split for group, split in set(groups.values()))
    report = {
        "schema": 1, "status": "data_prepared_gpu_training_not_started",
        "source_path": str(source), "source_sha256": source_hash,
        "profile_path": str(profile), "profile_binding": binding,
        "total_tokens": total_tokens, "max_think_tokens": max_think_tokens,
        "min_think_tokens": min_think_tokens, "heldout_fraction": heldout_fraction,
        "split_seed": seed, "group_counts": dict(group_counts),
        "counts": dict(counts), "exclusions": dict(excluded),
        "experimental_limits": limits, "experimental_counts": dict(sampled_counts),
        "artifacts": artifacts, "full_index_capped": False,
        "source_think_supervised": False, "gold_visible_to_generator": False,
        "cpu_synthetic_gate": cpu_gate, "training_started": False,
        "gpu_used": False, "live_service_changes": False,
        "warnings": (["No heldout examples fit; do not claim heldout evaluation"] if not counts["heldout"] else [])
                    + (["No train examples fit; do not start optimization"] if not counts["train"] else []),
    }
    with (output / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Parent-verified repaired Claude messages JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New private directory outside source tree")
    parser.add_argument("--total-tokens", type=int, default=32768)
    parser.add_argument("--max-think-tokens", type=int, default=8192)
    parser.add_argument("--min-think-tokens", type=int, default=1)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="mid-lora-self-think-v1")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--heldout-limit", type=int)
    parser.add_argument("--verify-code", action="store_true", help="Explicit synthetic CPU tokenizer/dataflow checks")
    parser.add_argument("--verify-only", action="store_true", help="Run synthetic checks without opening any corpus")
    args = parser.parse_args()
    if args.verify_only:
        from qwen_exo_booster.native_mid_lora_self_think import load_profile_tokenizer, verify_self_think_dataflow
        if args.source is not None:
            parser.error("--verify-only must not receive a source corpus")
        destination = args.output.resolve()
        if destination.is_relative_to(REPO):
            parser.error("Private verification receipt must remain outside the source tree")
        tokenizer, binding = load_profile_tokenizer(args.profile)
        result = {"profile_binding": binding, "gate": verify_self_think_dataflow(tokenizer)}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(destination), "cpu_synthetic_only": True}))
        return
    if args.source is None:
        parser.error("--source is required unless --verify-only is selected")
    result = prepare(args.profile, args.source, args.output,
        total_tokens=args.total_tokens, max_think_tokens=args.max_think_tokens,
        min_think_tokens=args.min_think_tokens, heldout_fraction=args.heldout_fraction,
        seed=args.seed, train_limit=args.train_limit, heldout_limit=args.heldout_limit,
        verify_code=args.verify_code)
    print(json.dumps({key: result[key] for key in ("status", "counts", "exclusions", "warnings")}, sort_keys=True))


if __name__ == "__main__":
    main()
