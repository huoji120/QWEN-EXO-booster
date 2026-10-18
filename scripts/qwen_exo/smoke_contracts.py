#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    encoded = None
    request_headers = {"Accept": "application/json", **(headers or {})}
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=encoded, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"body": body}
        return exc.code, parsed
    except (OSError, TimeoutError) as exc:
        return 599, {"error_type": type(exc).__name__, "error": str(exc)}


def retrying_http_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + min(timeout, 30.0)
    while True:
        status_code, response = http_json(
            url,
            method=method,
            payload=payload,
            timeout=timeout,
            headers=headers,
        )
        if status_code != 429 or time.monotonic() >= deadline:
            return status_code, response
        time.sleep(1.0)


def response_text(payload: dict[str, Any]) -> str:
    text = []
    for item in payload.get("output") or ():
        for content in item.get("content") or ():
            if content.get("type") in {"output_text", "text"}:
                text.append(str(content.get("text") or ""))
    return "".join(text)


def stage(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QWEN-EXO memory, tool-call, and cancellation smoke test"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    report: dict[str, Any] = {"base_url": base_url, "stages": []}
    status_code, health = http_json(f"{base_url}/qwen-exo/health", timeout=args.timeout)
    report["stages"].append(
        stage(
            "health",
            status_code == 200 and health.get("runtime_state") == "ready",
            status_code=status_code,
            response=health,
        )
    )

    thinking_marker = "QWEN_EXO_THINKING_READY"
    thinking_status, thinking_response = retrying_http_json(
        f"{base_url}/v1/responses",
        method="POST",
        payload={
            "model": args.model,
            "input": (
                "This is a trivial verification task. Reason briefly about copying "
                f"a fixed marker, then give exactly {thinking_marker} as the final answer."
            ),
            "temperature": 0,
            "max_output_tokens": 1536,
        },
        timeout=args.timeout,
    )
    reasoning_items = [
        item
        for item in thinking_response.get("output") or ()
        if item.get("type") == "reasoning"
    ]
    thinking_text = response_text(thinking_response).strip()
    report["stages"].append(
        stage(
            "thinking_semantics",
            thinking_status == 200
            and bool(reasoning_items)
            and thinking_marker in thinking_text,
            status_code=thinking_status,
            reasoning_items=reasoning_items,
            output_text=thinking_text,
            reasoning_source="server_default_chat_template_kwargs",
        )
    )

    fixture_run_id = uuid.uuid4().hex
    fixture_path = f"smoke/qwen-exo-memory-{fixture_run_id}.md"
    fixture_url = f"{base_url}/qwen-exo/knowledge/{fixture_path}"
    marker = "QWEN_EXO_MEMORY_READY_7B31"
    policy_list_status, policy_listing = http_json(
        f"{base_url}/qwen-exo/policydata", timeout=args.timeout
    )
    policy_documents = (
        tuple(policy_listing.get("documents") or ())
        if policy_list_status == 200
        else ()
    )
    policy_document = policy_documents[0] if len(policy_documents) == 1 else None
    policy_document_path = str((policy_document or {}).get("relative_path") or "")
    policy_document_url = (
        f"{base_url}/qwen-exo/policydata/{policy_document_path}"
        if policy_document_path
        else None
    )
    policy_document_status, policy_document_detail = (
        http_json(policy_document_url, timeout=args.timeout)
        if policy_document_url is not None
        else (409, {})
    )
    policy_content = str(policy_document_detail.get("content") or "")
    policy_document_ready = (
        policy_list_status == 200
        and len(policy_documents) == 1
        and policy_document_status == 200
        and policy_document_detail.get("source_kind") == "coding_agent_execution_policy"
        and "stable operational identity is GPT" in policy_content
    )
    cleanup_urls: list[str] = []

    def cleanup_fixtures() -> None:
        # PolicyData is operator-managed identity state. Smoke tests never mutate it;
        # a process or server crash therefore cannot replace the live personality.
        for cleanup_url in tuple(cleanup_urls):
            http_json(
                cleanup_url,
                method="DELETE",
                timeout=min(args.timeout, 30.0),
            )

    atexit.register(cleanup_fixtures)
    policy_identity = "GPT"
    fixture_records = "\n".join(
        f"Validation record {index}: the exact answer remains `{marker}`."
        for index in range(64)
    )
    fixture = (
        "---\ntitle: QWEN EXO memory validation\n"
        "tags: [qwen_exo_memory_validation]\n---\n"
        "# QWEN EXO memory validation\n\n"
        f"The exact qwen_exo_memory_validation answer is `{marker}`.\n\n"
        f"{fixture_records}\n"
    )
    put_status, put_result = http_json(
        fixture_url,
        method="PUT",
        payload={"content": fixture},
        timeout=args.timeout,
    )
    if put_status == 200:
        cleanup_urls.append(fixture_url)
    report["stages"].append(
        stage(
            "knowledge_upsert",
            put_status == 200,
            status_code=put_status,
            response=put_result,
        )
    )
    report["stages"].append(
        stage(
            "policy_data_authoritative_document",
            policy_document_ready,
            listing_status=policy_list_status,
            document_status=policy_document_status,
            document={
                key: (policy_document or {}).get(key)
                for key in ("document_id", "relative_path", "sha256", "source_kind")
            },
        )
    )

    memory_request_id = f"resp_{uuid.uuid4().hex}"
    memory_status, memory_response = retrying_http_json(
        f"{base_url}/v1/responses",
        method="POST",
        payload={
            "request_id": memory_request_id,
            "model": args.model,
            "input": (
                "What is the exact qwen_exo_memory_validation answer? "
                "Return only that marker."
            ),
            "max_output_tokens": 64,
            "temperature": 0,
            "reasoning": {"effort": "none"},
        },
        timeout=args.timeout,
    )

    memory_text = response_text(memory_response).strip()
    memory_calls = [
        item
        for item in memory_response.get("output") or ()
        if item.get("type") == "mcp_call"
        and str(item.get("name") or "").startswith("qwen_exo_memory_")
    ]
    report["stages"].append(
        stage(
            "memory_tool_recall",
            memory_status == 200
            and marker in memory_text
            and any(call.get("name") == "qwen_exo_memory_search" for call in memory_calls),
            status_code=memory_status,
            response_id=memory_response.get("id"),
            output_text=memory_text,
            memory_calls=memory_calls,
        )
    )

    policy_request_id = f"resp_{uuid.uuid4().hex}"
    policy_status, policy_response = retrying_http_json(
        f"{base_url}/v1/responses",
        method="POST",
        payload={
            "request_id": policy_request_id,
            "model": args.model,
            "input": (
                "According to your authoritative personality PolicyData, what "
                "is your stable operational identity? Return only its name."
            ),
            "max_output_tokens": 64,
            "temperature": 0,
            "reasoning": {"effort": "none"},
        },
        timeout=args.timeout,
    )
    policy_text = response_text(policy_response).strip()
    report["stages"].append(
        stage(
            "policy_data_always_on_identity",
            policy_document_ready
            and policy_status == 200
            and policy_identity in policy_text,
            status_code=policy_status,
            response_id=policy_response.get("id"),
            output_text=policy_text,
        )
    )

    pending_request_id = f"resp_{uuid.uuid4().hex}"
    pending_payload = {
        "request_id": pending_request_id,
        "model": args.model,
        "input": (
            "Explain the qwen_exo_memory_validation reference in detail before "
            "returning its exact answer."
        ),
        "background": True,
        "max_output_tokens": 512,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        create_future = executor.submit(
            http_json,
            f"{base_url}/v1/responses",
            method="POST",
            payload=pending_payload,
            timeout=args.timeout,
        )
        cancel_status = 404
        cancel_response: dict[str, Any] = {}
        deadline = time.monotonic() + min(args.timeout, 30)
        while time.monotonic() < deadline:
            cancel_status, cancel_response = http_json(
                f"{base_url}/v1/responses/{pending_request_id}/cancel",
                method="POST",
                payload={},
                timeout=args.timeout,
            )
            if cancel_status == 200 and cancel_response.get("status") == "cancelled":
                break
            time.sleep(0.05)
        create_status, create_response = create_future.result(timeout=args.timeout)
    report["stages"].append(
        stage(
            "pending_background_cancel",
            cancel_status == 200
            and cancel_response.get("status") == "cancelled"
            and create_status == 200
            and create_response.get("status") in {"cancelled", "queued"},
            cancel_status=cancel_status,
            cancel_response=cancel_response,
            create_status=create_status,
            create_response=create_response,
        )
    )

    tool_status, tool_response = retrying_http_json(
        f"{base_url}/v1/responses",
        method="POST",
        payload={
            "model": args.model,
            "input": "Call return_marker with marker QWEN_EXO_TOOL_READY. Do not answer in text.",
            "temperature": 0,
            "max_output_tokens": 128,
            "reasoning": {"effort": "none"},
            "tool_choice": "required",
            "tools": [
                {
                    "type": "function",
                    "name": "return_marker",
                    "description": "Return the requested verification marker.",
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
        for item in tool_response.get("output") or ()
        if item.get("type") in {"function_call", "tool_call"}
    ]
    report["stages"].append(
        stage(
            "structured_tool_call",
            tool_status == 200
            and any(
                call.get("name") == "return_marker"
                and "QWEN_EXO_TOOL_READY" in str(call.get("arguments") or "")
                for call in calls
            ),
            status_code=tool_status,
            calls=calls,
            response=tool_response if not calls else None,
        )
    )

    delete_status, delete_result = http_json(
        fixture_url,
        method="DELETE",
        timeout=args.timeout,
    )
    if delete_status == 200 and fixture_url in cleanup_urls:
        cleanup_urls.remove(fixture_url)
    report["stages"].append(
        stage(
            "knowledge_cleanup",
            delete_status == 200,
            status_code=delete_status,
            response=delete_result,
        )
    )
    policy_after_status, policy_after = (
        http_json(policy_document_url, timeout=args.timeout)
        if policy_document_url is not None
        else (409, {})
    )
    report["stages"].append(
        stage(
            "policy_data_unchanged",
            policy_after_status == 200
            and policy_after.get("sha256") == (policy_document or {}).get("sha256"),
            status_code=policy_after_status,
            before_sha256=(policy_document or {}).get("sha256"),
            after_sha256=policy_after.get("sha256"),
        )
    )

    if not cleanup_urls:
        atexit.unregister(cleanup_fixtures)
    report["passed"] = all(item["passed"] for item in report["stages"])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
