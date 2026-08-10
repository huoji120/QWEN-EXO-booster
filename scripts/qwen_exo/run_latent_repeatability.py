#!/usr/bin/env python3
"""Measure greedy output repeatability before attributing latent A/B changes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from transformers import AutoTokenizer

from run_latent_matrix import run_case
from run_latent_prompt_matrix import load_prompts, prompt_ids


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [str(row.get("output") or "") for row in rows]
    first_tokens = [row.get("distribution", {}).get("token_id") for row in rows]
    return {
        "repeats": len(rows),
        "unique_outputs": len(set(outputs)),
        "output_counts": dict(Counter(outputs)),
        "first_token_ids": first_tokens,
        "first_token_id_unique": len(set(first_tokens)),
        "logprobs": [row.get("distribution", {}).get("logprob") for row in rows],
        "outputs": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--artifact", default="identity-cognition-smoke")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--token-windows", nargs="+", type=int, default=[1, 16, 32])
    parser.add_argument("--prompts-file")
    parser.add_argument("--prompt-names", nargs="+")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = load_prompts(args.prompts_file)
    if args.prompt_names:
        selected = set(args.prompt_names)
        prompts = [prompt for prompt in prompts if prompt["name"] in selected]
        if len(prompts) != len(selected):
            raise ValueError("unknown prompt name")
    reports: list[dict[str, Any]] = []
    for prompt in prompts:
        input_ids = prompt_ids(tokenizer, prompt["prompt"])
        baseline_rows = [
            run_case(
                args.base_url,
                input_ids,
                artifact=None,
                strength=0.0,
                timeout=args.timeout,
                max_new_tokens=args.max_new_tokens,
            )
            for _ in range(args.repeats)
        ]
        conditions: list[dict[str, Any]] = [
            {"mode": "baseline", "summary": summarize(baseline_rows)}
        ]
        for token_window in args.token_windows:
            for strength in args.strengths:
                active_rows = [
                    run_case(
                        args.base_url,
                        input_ids,
                        artifact=args.artifact,
                        strength=strength,
                        timeout=args.timeout,
                        max_new_tokens=args.max_new_tokens,
                        token_window=token_window,
                    )
                    for _ in range(args.repeats)
                ]
                conditions.append(
                    {
                        "mode": "active",
                        "artifact": args.artifact,
                        "strength": strength,
                        "token_window": token_window,
                        "summary": summarize(active_rows),
                    }
                )
        reports.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(input_ids),
                "conditions": conditions,
            }
        )
    print(
        json.dumps(
            {
                "artifact": args.artifact,
                "strengths": args.strengths,
                "token_windows": args.token_windows,
                "repeats": args.repeats,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
