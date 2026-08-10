#!/usr/bin/env python3
"""Compare native Cognition against request-scoped latent steering on Responses."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


MARKER = "LANTERN-SELF-07"


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
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


def get_json(url: str, timeout: float) -> dict[str, Any]:
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


def compact_telemetry(events: list[dict[str, Any]]) -> dict[str, Any]:
    memory = next(
        (
            event.get("payload") or {}
            for event in events
            if event.get("event_type") == "memory.prepared"
        ),
        {},
    )
    requested = next(
        (
            event.get("payload") or {}
            for event in events
            if event.get("event_type") == "latent_transplant.requested"
        ),
        None,
    )
    applied = next(
        (
            event.get("payload") or {}
            for event in events
            if event.get("event_type") == "latent_transplant.applied"
        ),
        None,
    )
    return {
        "event_types": [str(event.get("event_type") or "") for event in events],
        "cognition": (memory.get("cognition") or {}),
        "native_prefix_restore": (memory.get("native_prefix_restore") or {}),
        "latent_requested": requested,
        "latent_applied": applied,
    }


def run_case(
    base_url: str,
    prompt: str,
    *,
    model: str,
    artifact: str | None,
    strength: float,
    token_window: int,
    timeout: float,
) -> dict[str, Any]:
    request_id = f"resp_latent_cognition_{uuid.uuid4().hex}"
    payload: dict[str, Any] = {
        "request_id": request_id,
        "model": model,
        "input": prompt,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 64,
        "reasoning": {"effort": "none"},
        "stream": False,
        "store": False,
    }
    if artifact is not None:
        spec: dict[str, Any] = {"artifact": artifact, "strength": strength}
        if token_window != 1:
            spec["token_window"] = token_window
        payload["metadata"] = {"qwen_exo_latent_transplant": spec}
    started = time.perf_counter()
    response = request_json(f"{base_url}/v1/responses", payload, timeout)
    events = (
        get_json(
            f"{base_url}/qwen-exo/telemetry?request_id={request_id}&limit=256", timeout
        ).get("events")
        or []
    )
    text = response_text(response)
    return {
        "request_id": request_id,
        "mode": "active" if artifact is not None else "baseline",
        "artifact": artifact,
        "strength": strength if artifact is not None else None,
        "token_window": token_window if artifact is not None else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "status": response.get("status"),
        "text": text,
        "marker_present": MARKER in text,
        "telemetry": compact_telemetry(events),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--strengths", nargs="+", type=float, required=True)
    parser.add_argument("--token-windows", nargs="+", type=int, default=[1])
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts file must contain a non-empty list")
    rows: list[dict[str, Any]] = []
    for item in prompts:
        if not isinstance(item, dict) or not item.get("name") or not item.get("prompt"):
            raise ValueError("each prompt needs name and prompt")
        prompt = str(item["prompt"])
        baseline = run_case(
            args.base_url,
            prompt,
            model=args.model,
            artifact=None,
            strength=0.0,
            token_window=1,
            timeout=args.timeout,
        )
        for artifact in args.artifacts:
            for token_window in args.token_windows:
                for strength in args.strengths:
                    active = run_case(
                        args.base_url,
                        prompt,
                        model=args.model,
                        artifact=artifact,
                        strength=strength,
                        token_window=token_window,
                        timeout=args.timeout,
                    )
                    rows.append(
                        {
                            "prompt": item["name"],
                            "prompt_text": prompt,
                            "artifact": artifact,
                            "token_window": token_window,
                            "strength": strength,
                            "baseline": baseline,
                            "active": active,
                            "behavioral_effect": active["text"] != baseline["text"],
                        }
                    )
    print(
        json.dumps(
            {
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
