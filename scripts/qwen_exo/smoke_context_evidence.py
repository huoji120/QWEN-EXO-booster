#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

from smoke_contracts import response_text, retrying_http_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused QWEN-EXO post-tool Context Evidence smoke test"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def events_of(payload: dict[str, Any], event_type: str) -> list[dict[str, Any]]:
    return [
        event
        for event in payload.get("events") or ()
        if event.get("event_type") == event_type
    ]


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    status_code, runtime_status = retrying_http_json(
        f"{base_url}/qwen-exo/status", timeout=args.timeout
    )
    marker = f"QWEN_EXO_CONTEXT_READY_{uuid.uuid4().hex[:12].upper()}"
    initial_request_id = f"resp_{uuid.uuid4().hex}"
    initial_status, initial_response = retrying_http_json(
        f"{base_url}/v1/responses",
        method="POST",
        payload={
            "request_id": initial_request_id,
            "model": args.model,
            "input": (
                "Call inspect_context_marker with the supplied marker. After the "
                "tool result arrives, report exactly which marker the direct "
                f"observation verified. Marker: {marker}"
            ),
            "temperature": 0,
            "max_output_tokens": 256,
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "inspect_context_marker",
                    "description": "Inspect and return the supplied marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {"marker": {"type": "string"}},
                        "required": ["marker"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
        timeout=args.timeout,
    )
    calls = [
        item
        for item in initial_response.get("output") or ()
        if item.get("type") in {"function_call", "tool_call"}
    ]
    call = calls[0] if calls else {}
    call_id = str(call.get("call_id") or call.get("id") or "")

    continuation_request_id = f"resp_{uuid.uuid4().hex}"
    continuation_status = 0
    continuation_response: dict[str, Any] = {}
    if initial_status == 200 and call_id:
        continuation_status, continuation_response = retrying_http_json(
            f"{base_url}/v1/responses",
            method="POST",
            payload={
                "request_id": continuation_request_id,
                "model": args.model,
                "previous_response_id": initial_response.get("id"),
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": (
                            "Direct verification result: exact comparison observed "
                            f"and verified marker `{marker}`."
                        ),
                    }
                ],
                "temperature": 0,
                "max_output_tokens": 512,
            },
            timeout=args.timeout,
        )

    telemetry_status = 0
    telemetry: dict[str, Any] = {}
    deadline = time.monotonic() + min(args.timeout, 30.0)
    while continuation_status == 200 and time.monotonic() < deadline:
        telemetry_status, telemetry = retrying_http_json(
            f"{base_url}/qwen-exo/telemetry?"
            + urllib.parse.urlencode(
                {"request_id": continuation_request_id, "limit": 256}
            ),
            timeout=args.timeout,
        )
        if events_of(telemetry, "post_tool_recall.completed") and events_of(
            telemetry, "self_ask.think_context_committed"
        ):
            break
        time.sleep(0.25)

    context_events = events_of(telemetry, "context_evidence.completed")
    recall_events = events_of(telemetry, "post_tool_recall.completed")
    refresh_events = events_of(telemetry, "refresh.completed")
    commit_events = events_of(telemetry, "self_ask.think_context_committed")
    context = context_events[-1].get("payload", {}) if context_events else {}
    recall = recall_events[-1].get("payload", {}) if recall_events else {}
    refresh = refresh_events[-1].get("payload", {}) if refresh_events else {}
    committed = commit_events[-1].get("payload", {}) if commit_events else {}
    output_text = response_text(continuation_response).strip()
    passed = bool(
        status_code == 200
        and runtime_status.get("runtime_state") == "ready"
        and runtime_status.get("context_evidence_mode") == "active"
        and initial_status == 200
        and call_id
        and continuation_status == 200
        and telemetry_status == 200
        and context.get("status") == "eligible"
        and context.get("eligible_count", 0) > 0
        and context.get("answer", {}).get("bytes", 10_000) <= 192
        and recall.get("status") == "context_evidence_ready"
        and recall.get("admitted") is True
        and recall.get("reference_status") == "no_eligible_reference"
        and recall.get("context_status") == "eligible"
        and refresh.get("selected_document_ids") == []
        and refresh.get("selected_lanes") == ["context"]
        and refresh.get("semantic_injection", {}).get("bytes", 10_000) <= 256
        and committed.get("purpose") == "post_tool"
        and 0 < committed.get("token_count", 0) <= 96
        and marker in output_text
        and "<qwen_exo" not in output_text
        and "GAP=" not in output_text
    )
    report = {
        "passed": passed,
        "base_url": base_url,
        "marker": marker,
        "runtime": {
            "status_code": status_code,
            "runtime_state": runtime_status.get("runtime_state"),
            "context_evidence_mode": runtime_status.get("context_evidence_mode"),
        },
        "initial": {
            "status_code": initial_status,
            "response_id": initial_response.get("id"),
            "call_id": call_id,
            "call": call,
        },
        "continuation": {
            "status_code": continuation_status,
            "response_id": continuation_response.get("id"),
            "output_text": output_text,
        },
        "context_evidence": context,
        "post_tool_recall": recall,
        "refresh": refresh,
        "think_context_committed": committed,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
