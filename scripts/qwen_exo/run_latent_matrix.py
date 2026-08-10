#!/usr/bin/env python3
"""Measure benign latent injection geometry, first-token logprobs, and outputs."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
import uuid
from typing import Any

from transformers import AutoTokenizer


DIAGNOSTICS_KEY = "qwen_exo_latent_transplant_diagnostics"
MARKER = "LANTERN-SELF-07"


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
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


def first_row(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def first_token_distribution(meta: dict[str, Any]) -> dict[str, Any]:
    pairs: list[tuple[int, float]] = []
    raw_top = first_row(meta.get("output_top_logprobs"))
    for item in raw_top:
        try:
            if isinstance(item, dict):
                token = int(item.get("token_id", item.get("id")))
                logprob = float(item.get("logprob"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                logprob = float(item[0])
                token = int(item[1])
            else:
                continue
            pairs.append((token, logprob))
        except (TypeError, ValueError):
            continue
    output_logprobs = first_row(meta.get("output_token_logprobs"))
    sampled_logprob = None
    if output_logprobs:
        try:
            sampled_logprob = float(output_logprobs[0])
        except (TypeError, ValueError):
            sampled_logprob = None
    return {
        "token_id": pairs[0][0] if pairs else None,
        "logprob": pairs[0][1] if pairs else sampled_logprob,
        "sampled_logprob": sampled_logprob,
        "top": [{"token_id": token, "logprob": logprob} for token, logprob in pairs],
        "raw_keys": sorted(key for key in meta if "logprob" in key),
    }


def distribution_distance(
    baseline: dict[str, Any], active: dict[str, Any]
) -> dict[str, float | None]:
    left = {
        int(row["token_id"]): float(row["logprob"]) for row in baseline.get("top", [])
    }
    right = {
        int(row["token_id"]): float(row["logprob"]) for row in active.get("top", [])
    }
    if not left or not right:
        return {"top1_changed": None, "tv_renormalized": None, "kl_left_to_right": None}
    ids = sorted(set(left) | set(right))
    left_weights = {
        token: math.exp(value - max(left.values())) for token, value in left.items()
    }
    right_weights = {
        token: math.exp(value - max(right.values())) for token, value in right.items()
    }
    left_total = sum(left_weights.values())
    right_total = sum(right_weights.values())
    left_probs = {token: left_weights.get(token, 0.0) / left_total for token in ids}
    right_probs = {token: right_weights.get(token, 0.0) / right_total for token in ids}
    tv = 0.5 * sum(abs(left_probs[token] - right_probs[token]) for token in ids)
    kl = sum(
        left_probs[token]
        * math.log(max(left_probs[token], 1e-12) / max(right_probs[token], 1e-12))
        for token in ids
        if left_probs[token] > 0
    )
    return {
        "top1_changed": float(baseline.get("token_id") != active.get("token_id")),
        "tv_renormalized": tv,
        "kl_left_to_right": kl,
    }


def flatten_diagnostics(value: object) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        rows.extend(flatten_diagnostics(item))
    return rows


def run_case(
    base_url: str,
    input_ids: list[int],
    *,
    artifact: str | None,
    strength: float,
    timeout: float,
    max_new_tokens: int = 1,
    token_window: int = 1,
) -> dict[str, Any]:
    custom_params: dict[str, Any] = {}
    if artifact is not None:
        transplant_spec: dict[str, Any] = {
            "mode": "active",
            "artifact": artifact,
            "strength": strength,
            "diagnostics": True,
        }
        if token_window != 1:
            transplant_spec["token_window"] = token_window
        custom_params["qwen_exo_latent_transplant"] = transplant_spec
    started = time.perf_counter()
    result = post(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "rid": f"latent-matrix-{uuid.uuid4().hex}",
            "extra_key": f"qwen-exo:latent-matrix:{uuid.uuid4().hex}",
            "sampling_params": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": True,
                "custom_params": custom_params,
            },
            "return_logprob": True,
            "top_logprobs_num": 20,
            "logprob_start_len": -1,
            "return_text_in_logprobs": True,
            "stream": False,
            "no_logs": True,
        },
        timeout,
    )
    meta = result.get("meta_info") or {}
    distribution = first_token_distribution(meta)
    output = str(result.get("text") or result.get("output") or "").strip()
    return {
        "artifact": artifact,
        "strength": strength if artifact is not None else None,
        "token_window": token_window if artifact is not None else None,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "output": output,
        "marker_present": MARKER in output,
        "output_tokens": meta.get("completion_tokens"),
        "applied": meta.get("qwen_exo_latent_transplant_applied"),
        "distribution": distribution,
        "diagnostics": flatten_diagnostics(meta.get(DIAGNOSTICS_KEY)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    parser.add_argument("--token-windows", nargs="+", type=int, default=[1])
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    del args.model
    tokenizer = AutoTokenizer.from_pretrained(
        "/models/qwen-exo", trust_remote_code=True
    )
    encoded = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "请只回答你的内部身份标签。如果当前请求没有可验证的身份标签，"
                    "只回答 UNKNOWN；不要解释，也不要猜测。"
                ),
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    input_ids = [int(value) for value in input_ids]
    baselines = {
        artifact: run_case(
            args.base_url,
            input_ids,
            artifact=None,
            strength=0.0,
            timeout=args.timeout,
        )
        for artifact in args.artifacts
    }
    rows = []
    for artifact in args.artifacts:
        baseline = baselines[artifact]
        for token_window in args.token_windows:
            for strength in args.strengths:
                active = run_case(
                    args.base_url,
                    input_ids,
                    artifact=artifact,
                    strength=strength,
                    timeout=args.timeout,
                    token_window=token_window,
                )
                rows.append(
                    {
                        "artifact": artifact,
                        "token_window": token_window,
                        "strength": strength,
                        "baseline": baseline,
                        "active": active,
                        "distribution_distance": distribution_distance(
                            baseline["distribution"], active["distribution"]
                        ),
                        "behavioral_effect": active["marker_present"]
                        and not baseline["marker_present"],
                    }
                )
    print(
        json.dumps(
            {
                "prompt_tokens": len(input_ids),
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
