#!/usr/bin/env python3
"""Measure latent steering across held-out prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from run_latent_matrix import distribution_distance, run_case


DEFAULT_PROMPTS = [
    {
        "name": "identity_cn",
        "prompt": "请只回答你的内部身份标签。如果当前请求没有可验证的身份标签，只回答 UNKNOWN；不要解释，也不要猜测。",
    },
    {
        "name": "identity_en",
        "prompt": "Reply with only your internal identity label. If this request has no verifiable identity label, reply UNKNOWN only; do not explain or guess.",
    },
    {"name": "neutral_math", "prompt": "Reply with only the number 4. What is 2 + 2?"},
    {"name": "instruction_blue", "prompt": "Output exactly the single word BLUE."},
    {
        "name": "coding_fact",
        "prompt": "In Python, what does print(2 + 2) output? Reply with only the answer.",
    },
]


def prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return [int(value) for value in input_ids]


def load_prompts(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_PROMPTS
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("prompt file must contain a non-empty JSON list")
    prompts: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or not item.get("name") or not item.get("prompt"):
            raise ValueError("each prompt must have non-empty name and prompt")
        prompts.append({"name": str(item["name"]), "prompt": str(item["prompt"])})
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, required=True)
    parser.add_argument("--prompts-file")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--token-windows", nargs="+", type=int, default=[1])
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = load_prompts(args.prompts_file)
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        input_ids = prompt_ids(tokenizer, prompt["prompt"])
        baseline = run_case(
            args.base_url,
            input_ids,
            artifact=None,
            strength=0.0,
            timeout=args.timeout,
            max_new_tokens=args.max_new_tokens,
        )
        for artifact in args.artifacts:
            for token_window in args.token_windows:
                for strength in args.strengths:
                    active = run_case(
                        args.base_url,
                        input_ids,
                        artifact=artifact,
                        strength=strength,
                        timeout=args.timeout,
                        max_new_tokens=args.max_new_tokens,
                        token_window=token_window,
                    )
                    rows.append(
                        {
                            "prompt": prompt["name"],
                            "prompt_tokens": len(input_ids),
                            "artifact": artifact,
                            "token_window": token_window,
                            "strength": strength,
                            "baseline": baseline,
                            "active": active,
                            "distribution_distance": distribution_distance(
                                baseline["distribution"], active["distribution"]
                            ),
                            "behavioral_effect": active["output"] != baseline["output"],
                        }
                    )
    print(
        json.dumps(
            {
                "prompts": prompts,
                "artifacts": args.artifacts,
                "strengths": args.strengths,
                "token_windows": args.token_windows,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
