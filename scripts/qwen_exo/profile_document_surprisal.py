#!/usr/bin/env python3
"""Profile per-token surprisal of a long document through /generate logprobs."""

from __future__ import annotations

import argparse
import json
import urllib.request
import uuid
from typing import Any

from transformers import AutoTokenizer

CHUNK = 60000
OVERLAP = 4096


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


def first_row(value: object) -> list[Any]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/models/qwen-exo")
    parser.add_argument("--document", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = open(args.document, encoding="utf-8").read()
    ids = tokenizer.encode(text, add_special_tokens=False)
    total = len(ids)
    surprisals: list[float | None] = [None] * total
    start = 0
    while start < total:
        end = min(total, start + CHUNK)
        window = ids[max(0, start - OVERLAP) : end] if start else ids[:end]
        result = post(
            f"{args.base_url}/generate",
            {
                "input_ids": window,
                "rid": f"surprisal-{uuid.uuid4().hex}",
                "extra_key": f"qwen-exo:surprisal:{uuid.uuid4().hex}",
                "sampling_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "max_new_tokens": 1,
                    "skip_special_tokens": True,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "stream": False,
                "no_logs": True,
            },
            args.timeout,
        )
        meta = result.get("meta_info") or {}
        logprobs = meta.get("input_token_logprobs") or []
        if logprobs and isinstance(logprobs[0], list) and logprobs[0] and isinstance(
            logprobs[0][0], list
        ):
            logprobs = logprobs[0]
        values: list[float | None] = []
        for item in logprobs:
            if isinstance(item, (list, tuple)) and item:
                try:
                    values.append(float(item[0]))
                except (TypeError, ValueError):
                    values.append(None)
            elif item is None:
                values.append(None)
            else:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    values.append(None)
        base = max(0, start - OVERLAP) if start else 0
        scored_start = len(window) - len(values)
        for local_index, value in enumerate(values):
            absolute = base + scored_start + local_index
            if absolute >= total:
                break
            if start <= absolute < end and value is not None:
                surprisals[absolute] = -value
        print(f"scored {start}..{end} of {total}", flush=True)
        if end >= total:
            break
        start = end

    scored = [value for value in surprisals if value is not None]
    scored_sorted = sorted(scored)
    quantiles = {}
    for q in (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        index = min(len(scored_sorted) - 1, int(q * len(scored_sorted)))
        quantiles[str(q)] = scored_sorted[index]
    thresholds = {}
    for threshold in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0):
        anchors = sum(1 for value in scored if value >= threshold)
        thresholds[str(threshold)] = {
            "anchors": anchors,
            "estimated_span_tokens_16": anchors * 16,
        }
    window_profile = []
    width = 4096
    for offset in range(0, total, width):
        segment = [v for v in surprisals[offset : offset + width] if v is not None]
        if segment:
            window_profile.append(
                {
                    "start": offset,
                    "depth": round(offset / total, 3),
                    "mean": sum(segment) / len(segment),
                    "max": max(segment),
                    "anchors_ge_8": sum(1 for v in segment if v >= 8.0),
                }
            )
    report = {
        "document_tokens": total,
        "scored_tokens": len(scored),
        "quantiles": quantiles,
        "thresholds": thresholds,
        "window_profile": window_profile,
        "surprisals": [
            round(value, 4) if value is not None else None for value in surprisals
        ],
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"tokens": total, "quantiles": quantiles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
