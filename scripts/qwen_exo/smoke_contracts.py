#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import json
import time
import urllib.error
import urllib.parse
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
    policy_restore: dict[str, Any] = {}
    if not policy_documents:
        policy_fixture_path = f"smoke/qwen-exo-policy-{fixture_run_id}.md"
        policy_fixture_url: str | None = (
            f"{base_url}/qwen-exo/policydata/{policy_fixture_path}"
        )
    elif len(policy_documents) == 1:
        policy_fixture_path = str(policy_documents[0].get("relative_path") or "")
        policy_fixture_url = (
            f"{base_url}/qwen-exo/policydata/{policy_fixture_path}"
            if policy_fixture_path
            else None
        )
        original_status, original_policy = (
            http_json(policy_fixture_url, timeout=args.timeout)
            if policy_fixture_url is not None
            else (409, {})
        )
        original_content = original_policy.get("content")
        if original_status != 200 or not isinstance(original_content, str):
            policy_fixture_url = None
        else:
            policy_restore.update(
                url=policy_fixture_url, content=original_content, pending=False
            )
    else:
        policy_fixture_url = None
    cleanup_urls: list[str] = []

    def cleanup_fixtures() -> None:
        changed = False
        for cleanup_url in tuple(cleanup_urls):
            cleanup_status, _cleanup_result = http_json(
                cleanup_url,
                method="DELETE",
                timeout=min(args.timeout, 30.0),
            )
            changed = changed or cleanup_status == 200
        if policy_restore.get("pending"):
            restore_status, _restore_result = http_json(
                str(policy_restore["url"]),
                method="PUT",
                payload={"content": policy_restore["content"]},
                timeout=min(args.timeout, 30.0),
            )
            changed = changed or restore_status == 200
        if changed:
            http_json(
                f"{base_url}/qwen-exo/tensor-bank/reindex",
                method="POST",
                payload={},
                timeout=min(args.timeout, 30.0),
            )

    atexit.register(cleanup_fixtures)
    policy_marker = "QWEN_EXO_POLICY_READY_41C9"
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
    policy_fixture = (
        "---\ntitle: QWEN EXO policy validation\n"
        "tags: [qwen_exo_policy_validation]\n---\n"
        "# QWEN EXO policy validation\n\n"
        "For every QWEN EXO code change, qwen_exo_policy_validation requires "
        f"regression verification code `{policy_marker}` before delivery.\n"
    )
    if policy_fixture_url is None:
        policy_put_status, policy_put_result = (
            409,
            {
                "error": "PolicyData fixture requires zero or one readable existing document"
            },
        )
    else:
        policy_put_status, policy_put_result = http_json(
            policy_fixture_url,
            method="PUT",
            payload={"content": policy_fixture},
            timeout=args.timeout,
        )
    if policy_put_status == 200:
        if policy_restore:
            policy_restore["pending"] = True
        else:
            cleanup_urls.append(policy_fixture_url)
    report["stages"].append(
        stage(
            "policy_data_upsert",
            policy_put_status == 200,
            status_code=policy_put_status,
            response=policy_put_result,
        )
    )

    bank_status, bank_result = http_json(
        f"{base_url}/qwen-exo/tensor-bank/reindex",
        method="POST",
        payload={},
        timeout=args.timeout,
    )
    bank_documents = int(bank_result.get("document_state_count") or 0)
    report["stages"].append(
        stage(
            "native_tensor_bank_reindex",
            bank_status == 200
            and bank_documents > 0
            and bank_result.get("model_native_documents") == bank_documents
            and bank_result.get("complete_gdn_document_states") == bank_documents
            and bank_result.get("document_level") is True
            and bank_result.get("storage_dtype") == "float8_e4m3fn",
            status_code=bank_status,
            response=bank_result,
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
    telemetry_status, telemetry = http_json(
        f"{base_url}/qwen-exo/telemetry?"
        + urllib.parse.urlencode({"request_id": memory_request_id}),
        timeout=args.timeout,
    )
    memory_events = [
        event
        for event in telemetry.get("events") or ()
        if event.get("event_type") == "memory.prepared"
    ]
    memory_payload = memory_events[-1].get("payload", {}) if memory_events else {}
    memory_text = response_text(memory_response).strip()
    memory_native_restore = memory_payload.get("native_prefix_restore") or {}
    report["stages"].append(
        stage(
            "memory_judge_and_attachment",
            memory_status == 200
            and memory_text == marker
            and telemetry_status == 200
            and memory_payload.get("knowledge_admission_mode") == "semantic_eligibility"
            and memory_native_restore.get("active") is True
            and memory_native_restore.get("lane") == "knowledge"
            and memory_native_restore.get("tokens", 0) > 0
            and any(
                decision.get("status") in {"true", "eligible"}
                for decision in memory_payload.get("semantic_decisions") or ()
            )
            and any(
                candidate.get("lane") == "knowledge"
                for candidate in memory_payload.get("proposed_candidates") or ()
            ),
            status_code=memory_status,
            response_id=memory_response.get("id"),
            output_text=memory_text,
            memory=memory_payload,
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
                "A QWEN EXO code change is ready for delivery. According to "
                "qwen_exo_policy_validation, what exact regression verification "
                "code is required? Return only that code."
            ),
            "max_output_tokens": 64,
            "temperature": 0,
            "reasoning": {"effort": "none"},
        },
        timeout=args.timeout,
    )
    policy_telemetry_status, policy_telemetry = http_json(
        f"{base_url}/qwen-exo/telemetry?"
        + urllib.parse.urlencode({"request_id": policy_request_id}),
        timeout=args.timeout,
    )
    policy_events = [
        event
        for event in policy_telemetry.get("events") or ()
        if event.get("event_type") == "memory.prepared"
    ]
    policy_payload = policy_events[-1].get("payload", {}) if policy_events else {}
    policy_text = response_text(policy_response).strip()
    report["stages"].append(
        stage(
            "policy_data_judge_and_native_state",
            policy_status == 200
            and policy_text == policy_marker
            and policy_telemetry_status == 200
            and policy_payload.get("policy_data", {}).get("attached_tokens", 0) > 0
            and policy_payload.get("policy_data", {}).get("text_attached") is False
            and policy_payload.get("policy_data", {}).get("injection_mode")
            == "native_full_attention_salient_kv_and_gdn_document_state"
            and policy_payload.get("policy_data", {})
            .get("native_state", {})
            .get("tokens", 0)
            > 0
            and policy_payload.get("native_prefix_restore", {}).get("active") is True
            and policy_payload.get("native_prefix_restore", {}).get("lane")
            == "policydata"
            and policy_put_result.get("document_id")
            in policy_payload.get("policy_data", {}).get("document_ids", ()),
            status_code=policy_status,
            response_id=policy_response.get("id"),
            output_text=policy_text,
            memory=policy_payload,
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
    if policy_restore.get("pending"):
        policy_cleanup_status, policy_cleanup_result = http_json(
            str(policy_restore["url"]),
            method="PUT",
            payload={"content": policy_restore["content"]},
            timeout=args.timeout,
        )
        if policy_cleanup_status == 200:
            policy_restore["pending"] = False
    elif policy_fixture_url is not None:
        policy_cleanup_status, policy_cleanup_result = http_json(
            policy_fixture_url,
            method="DELETE",
            timeout=args.timeout,
        )
        if policy_cleanup_status == 200 and policy_fixture_url in cleanup_urls:
            cleanup_urls.remove(policy_fixture_url)
    else:
        policy_cleanup_status, policy_cleanup_result = (
            409,
            {"error": "PolicyData fixture was not installed"},
        )
    report["stages"].append(
        stage(
            "policy_data_cleanup",
            policy_cleanup_status == 200,
            status_code=policy_cleanup_status,
            response=policy_cleanup_result,
        )
    )

    cleanup_bank_status, cleanup_bank_result = http_json(
        f"{base_url}/qwen-exo/tensor-bank/reindex",
        method="POST",
        payload={},
        timeout=args.timeout,
    )
    report["stages"].append(
        stage(
            "native_tensor_bank_cleanup_reindex",
            cleanup_bank_status == 200
            and cleanup_bank_result.get("source_digest")
            != bank_result.get("source_digest"),
            status_code=cleanup_bank_status,
            response=cleanup_bank_result,
        )
    )

    if not cleanup_urls and not policy_restore.get("pending"):
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
