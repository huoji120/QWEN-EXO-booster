#!/usr/bin/env python3
"""Probe fact recall from a long-trajectory native Tensor Bank page."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Responses returned a non-object")
    return value


def get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("telemetry returned a non-object")
    return value


def response_text(value: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in value.get("output") or ():
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or ():
            if isinstance(content, dict) and content.get("type") in {
                "output_text",
                "text",
            }:
                parts.append(str(content.get("text") or ""))
    return "".join(parts).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    probes = json.loads(args.probes.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for probe in probes:
        request_id = f"resp_trajectory_probe_{uuid.uuid4().hex}"
        started = time.perf_counter()
        response = post(
            f"{args.base_url}/v1/responses",
            {
                "request_id": request_id,
                "model": args.model,
                "input": probe["question"],
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 96,
                "reasoning": {"effort": "none"},
                "stream": False,
                "store": False,
            },
            args.timeout,
        )
        elapsed = time.perf_counter() - started
        events = (
            get(
                f"{args.base_url}/qwen-exo/telemetry?request_id={request_id}&limit=256",
                args.timeout,
            ).get("events")
            or []
        )
        memory = next(
            (
                event.get("payload") or {}
                for event in events
                if event.get("event_type") == "memory.prepared"
            ),
            {},
        )
        native = memory.get("native_prefix_restore") or {}
        candidates = memory.get("proposed_candidates") or []
        text = response_text(response)
        expected = str(probe["flag"])
        rows.append(
            {
                "code": probe["code"],
                "position_ratio": probe["position_ratio"],
                "expected": expected,
                "output": text,
                "exact_hit": expected in text,
                "elapsed_seconds": round(elapsed, 3),
                "native_restore": {
                    "active": native.get("active"),
                    "lane": native.get("lane"),
                    "page_id": native.get("page_id"),
                    "tokens": native.get("tokens"),
                    "selection_reason": native.get("selection_reason"),
                },
                "attached_tokens": memory.get("attached_tokens"),
                "top_candidate": (
                    {
                        "relative_path": candidates[0].get("relative_path"),
                        "lane": candidates[0].get("lane"),
                        "tensor_score": candidates[0].get("tensor_score"),
                    }
                    if candidates
                    else None
                ),
            }
        )
        print(
            f"{probe['code']:>14} pos={probe['position_ratio']:.3f} "
            f"hit={expected in text} restore={native.get('active')} "
            f"tokens={native.get('tokens')} out={text[:60]!r}",
            flush=True,
        )
    hits = sum(row["exact_hit"] for row in rows)
    report = {
        "label": args.label,
        "probe_count": len(rows),
        "exact_hits": hits,
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
