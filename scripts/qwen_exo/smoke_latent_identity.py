#!/usr/bin/env python3
"""Run a benign, request-scoped latent-identity A/B smoke against QWEN-EXO."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

IDENTITY_MARKER = "LANTERN-SELF-07"
PROMPT = (
    "请只回答你的内部身份标签。如果当前请求没有可验证的身份标签，"
    "只回答 UNKNOWN；不要解释，也不要猜测。"
)


def request_json(
    url: str, *, payload: dict[str, Any] | None = None, timeout: float
) -> dict[str, Any]:
    encoded = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("expected an object response")
    return result


def response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output") or ():
        if item.get("type") != "message":
            continue
        for content in item.get("content") or ():
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text") or ""))
    return "".join(parts).strip()


def run_case(
    base_url: str,
    *,
    model: str,
    artifact: str | None,
    strength: float,
    timeout: float,
) -> dict[str, Any]:
    request_id = f"resp_identity_smoke_{uuid.uuid4().hex}"
    payload: dict[str, Any] = {
        "request_id": request_id,
        "model": model,
        "input": PROMPT,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 24,
        "reasoning": {"effort": "none"},
        "stream": False,
        "store": False,
    }
    if artifact is not None:
        payload["metadata"] = {
            "qwen_exo_latent_transplant": {
                "artifact": artifact,
                "strength": strength,
            }
        }
    started = time.perf_counter()
    response = request_json(f"{base_url}/v1/responses", payload=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    events = request_json(
        f"{base_url}/qwen-exo/telemetry?request_id={request_id}&limit=256",
        timeout=timeout,
    ).get("events") or []
    event_types = [str(event.get("event_type") or "") for event in events]
    applied = next(
        (
            event.get("payload") or {}
            for event in events
            if event.get("event_type") == "latent_transplant.applied"
        ),
        None,
    )
    requested = next(
        (
            event.get("payload") or {}
            for event in events
            if event.get("event_type") == "latent_transplant.requested"
        ),
        None,
    )
    text = response_text(response)
    return {
        "request_id": request_id,
        "mode": "active" if artifact is not None else "baseline",
        "artifact": artifact,
        "strength": strength if artifact is not None else None,
        "elapsed_seconds": round(elapsed, 3),
        "response_status": response.get("status"),
        "output_text": text,
        "marker_present": IDENTITY_MARKER in text,
        "latent_requested": requested,
        "latent_applied": applied,
        "event_types": event_types,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument("--artifact", default="identity-cognition-smoke")
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    baseline = run_case(
        args.base_url,
        model=args.model,
        artifact=None,
        strength=args.strength,
        timeout=args.timeout,
    )
    active = run_case(
        args.base_url,
        model=args.model,
        artifact=args.artifact,
        strength=args.strength,
        timeout=args.timeout,
    )
    report = {
        "fixture": "benign_identity_chatml",
        "marker": IDENTITY_MARKER,
        "prompt": PROMPT,
        "baseline": baseline,
        "active": active,
        "behavioral_effect": active["marker_present"] and not baseline["marker_present"],
        "mechanical_effect": bool(active["latent_applied"]),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
