#!/usr/bin/env python3
"""Evaluate a trained activation editor on the live server via /generate logprobs."""

from __future__ import annotations

import argparse
import json
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from qwen_exo_booster.activation_training import pack_trajectory_context


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("non-object response")
    return value


def build_eval_samples(
    trajectory_path: Path,
    tokenizer: Any,
    max_context_tokens: int,
    max_target_tokens: int,
    holdout_ratio: float,
) -> list[dict[str, Any]]:
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    messages = trajectory["session"]["messages"]
    samples = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index < 2:
            continue
        content = message["content"]
        if not isinstance(content, str) or len(content.strip()) < 20:
            continue
        context_ids = pack_trajectory_context(
            messages[:index], tokenizer, max_context_tokens
        )
        target_ids = tokenizer.encode(content, add_special_tokens=False)[
            :max_target_tokens
        ]
        if len(target_ids) < 4:
            continue
        samples.append(
            {"index": index, "context_ids": context_ids, "target_ids": target_ids}
        )
    split = max(1, int(len(samples) * (1 - holdout_ratio)))
    return samples[split:] or samples[-1:]


def target_nll(
    base_url: str,
    sample: dict[str, Any],
    editor: str | None,
    timeout: float,
    strength: float | None = None,
) -> tuple[float, int]:
    ids = sample["context_ids"] + sample["target_ids"]
    custom_params: dict[str, Any] = {}
    if editor is not None:
        spec: dict[str, Any] = {
            "editor": editor,
            "tail_offset": len(sample["target_ids"]),
        }
        if strength is not None:
            spec["strength"] = strength
        custom_params["qwen_exo_activation_editor"] = spec
    result = post(
        f"{base_url}/generate",
        {
            "input_ids": ids,
            "rid": f"editor-eval-{uuid.uuid4().hex}",
            "extra_key": f"qwen-exo:editor-eval:{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 1,
                "skip_special_tokens": True,
                "custom_params": custom_params,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
            "stream": False,
            "no_logs": True,
        },
        timeout,
    )
    meta = result.get("meta_info") or {}
    rows = meta.get("input_token_logprobs") or []
    if rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        rows = rows[0]
    context_end = len(sample["context_ids"])
    values = []
    for row in rows[context_end:]:
        if isinstance(row, (list, tuple)) and row and row[0] is not None:
            values.append(float(row[0]))
    count = len(values)
    return (-sum(values), count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--editor", required=True)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=1024)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples = build_eval_samples(
        args.trajectory,
        tokenizer,
        args.max_context_tokens,
        args.max_target_tokens,
        args.holdout_ratio,
    )[: args.limit]
    print(f"eval_samples={len(samples)}", flush=True)
    totals = {"baseline": [0.0, 0], "editor": [0.0, 0]}
    for sample in samples:
        base_nll, base_count = target_nll(args.base_url, sample, None, args.timeout)
        edit_nll, edit_count = target_nll(
            args.base_url, sample, args.editor, args.timeout, args.strength
        )
        totals["baseline"][0] += base_nll
        totals["baseline"][1] += base_count
        totals["editor"][0] += edit_nll
        totals["editor"][1] += edit_count
        print(
            f"msg={sample['index']:>3} base={base_nll / max(base_count, 1):.4f} "
            f"edit={edit_nll / max(edit_count, 1):.4f}",
            flush=True,
        )
    report = {
        "editor": args.editor,
        "samples": len(samples),
        "baseline_nll": totals["baseline"][0] / max(totals["baseline"][1], 1),
        "editor_nll": totals["editor"][0] / max(totals["editor"][1], 1),
        "improvement": (totals["baseline"][0] - totals["editor"][0])
        / max(totals["baseline"][1], 1),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
