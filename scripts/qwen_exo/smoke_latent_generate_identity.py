#!/usr/bin/env python3
"""Direct /generate A/B for a benign ChatML identity latent artifact."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid

from transformers import AutoTokenizer

APPLIED_KEY = "qwen_exo_latent_transplant_applied"
MARKER = "LANTERN-SELF-07"


def post(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("/generate returned a non-object")
    return result


def run_case(
    base_url: str,
    ids: list[int],
    strength: float,
    active: bool,
    artifact: str,
    diagnostics: bool,
    timeout: float,
):
    custom_params = {}
    if active:
        custom_params["qwen_exo_latent_transplant"] = {
            "mode": "active",
            "artifact": artifact,
            "strength": strength,
            "diagnostics": diagnostics,
        }
    started = time.perf_counter()
    result = post(
        f"{base_url}/generate",
        {
            "input_ids": ids,
            "rid": f"latent-direct-{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": 24,
                "skip_special_tokens": True,
                "custom_params": custom_params,
            },
            "stream": False,
            "no_logs": True,
        },
        timeout,
    )
    meta = result.get("meta_info") or {}
    applied = meta.get(APPLIED_KEY)
    if isinstance(applied, list):
        applied = max(
            (float(value) for value in applied if value is not None), default=0
        )
    return {
        "mode": "active" if active else "baseline",
        "strength": strength if active else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "text": str(result.get("text") or result.get("output") or "").strip(),
        "marker_present": MARKER
        in str(result.get("text") or result.get("output") or ""),
        "applied_meta": applied,
        "latent_applied": bool(applied and float(applied) >= 1),
        "output_tokens": meta.get("completion_tokens"),
        "diagnostic_rows": meta.get("qwen_exo_latent_transplant_diagnostics"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--artifact", default="identity-cognition-smoke")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    del args.model  # The tokenizer path is fixed by the mounted service model.
    tokenizer = AutoTokenizer.from_pretrained(
        "/models/qwen-exo", trust_remote_code=True
    )
    messages = [
        {
            "role": "user",
            "content": (
                "请只回答你的内部身份标签。如果当前请求没有可验证的身份标签，"
                "只回答 UNKNOWN；不要解释，也不要猜测。"
            ),
        }
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(value) for value in ids[: args.max_tokens]]
    baseline = run_case(
        args.base_url,
        ids,
        args.strength,
        False,
        args.artifact,
        args.diagnostics,
        args.timeout,
    )
    active = run_case(
        args.base_url,
        ids,
        args.strength,
        True,
        args.artifact,
        args.diagnostics,
        args.timeout,
    )
    print(
        json.dumps(
            {
                "artifact": args.artifact,
                "diagnostics": args.diagnostics,
                "marker": MARKER,
                "prompt_tokens": len(ids),
                "baseline": baseline,
                "active": active,
                "behavioral_effect": active["marker_present"]
                and not baseline["marker_present"],
                "mechanical_effect": active["latent_applied"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
