#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def request_json(
    url: str, *, payload: dict[str, Any] | None = None, timeout: float
) -> dict[str, Any]:
    encoded = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def gpu_memory_snapshot(
    expected_gpu_count: int | None = None,
) -> list[dict[str, float | int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    devices = []
    for raw_line in completed.stdout.splitlines():
        values = [value.strip() for value in raw_line.split(",")]
        if len(values) != 3:
            raise RuntimeError(f"Invalid nvidia-smi memory row: {raw_line!r}")
        index, used_mib, total_mib = (int(value) for value in values)
        devices.append(
            {
                "index": index,
                "used_mib": used_mib,
                "total_mib": total_mib,
                "used_fraction": used_mib / total_mib,
            }
        )
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPU memory rows")
    if expected_gpu_count is not None and len(devices) != expected_gpu_count:
        raise RuntimeError(
            f"Expected memory data for {expected_gpu_count} GPUs, found {len(devices)}"
        )
    return devices


def attach_memory_evidence(
    result: dict[str, Any],
    *,
    max_fraction: float,
    expected_gpu_count: int | None = None,
) -> None:
    devices = gpu_memory_snapshot(expected_gpu_count)
    bounded = all(float(device["used_fraction"]) <= max_fraction for device in devices)
    result["gpu_memory"] = devices
    result["max_gpu_memory_fraction"] = max_fraction
    result["memory_bounded"] = bounded
    result["passed"] = bool(result.get("passed")) and bounded


def response_text(payload: dict[str, Any]) -> str:
    parts = []
    for item in payload.get("output") or ():
        if item.get("type") != "message":
            continue
        for content in item.get("content") or ():
            if content.get("type") in {"output_text", "text"}:
                parts.append(str(content.get("text") or ""))
    return "".join(parts)


def make_prompt(target_tokens: int, marker: str) -> str:
    filler_count = max(1, target_tokens - 80)
    filler = "state " * filler_count
    return (
        f"The exact verification marker is {marker}. Keep it unchanged.\n"
        f"Context filler begins: {filler}\n"
        "Return only the exact verification marker, with no punctuation."
    )


def run_stage(
    base_url: str,
    model: str,
    target_tokens: int,
    timeout: float,
    require_marker: bool = True,
) -> dict[str, Any]:
    marker = f"QWEN_EXO_{target_tokens}_READY"
    prompt = make_prompt(target_tokens, marker)
    started = time.perf_counter()
    response = request_json(
        f"{base_url}/v1/responses",
        payload={
            "model": model,
            "input": prompt,
            "temperature": 0,
            "max_output_tokens": 32,
            "reasoning": {"effort": "none"},
            "stream": False,
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    text = response_text(response).strip()
    usage = response.get("usage") or {}
    return {
        "target_tokens": target_tokens,
        "passed": response.get("status") == "completed"
        and (not require_marker or text == marker),
        "elapsed_seconds": elapsed,
        "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "completion_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "response_id": response.get("id"),
        "finish_status": response.get("status"),
        "output_text": text,
    }


def run_stage_safely(
    base_url: str,
    model: str,
    target_tokens: int,
    timeout: float,
    require_marker: bool = True,
) -> dict[str, Any]:
    try:
        return run_stage(
            base_url,
            model,
            target_tokens,
            timeout,
            require_marker=require_marker,
        )
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", "replace")
        try:
            error_body = json.loads(raw_body) if raw_body else None
        except json.JSONDecodeError:
            error_body = raw_body
        return {
            "target_tokens": target_tokens,
            "passed": False,
            "status_code": exc.code,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "error_body": error_body,
        }
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {
            "target_tokens": target_tokens,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


_ACTIVE_ADAPTIVE_PHASES = frozenset(
    {
        "observing",
        "restored",
        "triggered",
        "refreshing",
        "semantic_ready",
        "replay_scoring",
        "post_tool_refreshing",
    }
)


def wait_for_adaptive_idle(base_url: str, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + timeout
    polls = 0
    while True:
        polls += 1
        try:
            status = request_json(f"{base_url}/qwen-exo/status", timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return {
                "passed": False,
                "elapsed_seconds": time.perf_counter() - started,
                "polls": polls,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        phase_counts = (status.get("adaptive_retrieval") or {}).get(
            "phase_counts"
        ) or {}
        active = {
            phase: int(count)
            for phase, count in phase_counts.items()
            if phase in _ACTIVE_ADAPTIVE_PHASES and int(count) > 0
        }
        if not active:
            return {
                "passed": True,
                "elapsed_seconds": time.perf_counter() - started,
                "polls": polls,
                "phase_counts": phase_counts,
            }
        if time.perf_counter() >= deadline:
            return {
                "passed": False,
                "elapsed_seconds": time.perf_counter() - started,
                "polls": polls,
                "phase_counts": phase_counts,
                "active_phase_counts": active,
                "error": "adaptive retrieval did not become idle before timeout",
            }
        time.sleep(0.5)


def run_overload_attempt(
    base_url: str,
    model: str,
    target_tokens: int,
    timeout: float,
    attempt: int,
) -> dict[str, Any]:
    marker = f"QWEN_EXO_OVERLOAD_{attempt}_READY"
    request = urllib.request.Request(
        f"{base_url}/v1/responses",
        data=json.dumps(
            {
                "request_id": f"resp_overload_{uuid.uuid4().hex}",
                "model": model,
                "input": make_prompt(target_tokens, marker),
                "temperature": 0,
                "max_output_tokens": 32,
                "reasoning": {"effort": "none"},
                "stream": False,
            }
        ).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {
                "status_code": response.status,
                "passed": payload.get("status") == "completed"
                and response_text(payload).strip() == marker,
                "response_status": payload.get("status"),
            }
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            payload = {"raw_body": raw_body}
        error = payload.get("error") if isinstance(payload, dict) else None
        return {
            "status_code": exc.code,
            "passed": exc.code == 429
            and isinstance(error, dict)
            and error.get("type") == "rate_limit_exceeded"
            and error.get("code") == 429,
            "retry_after": exc.headers.get("Retry-After"),
            "error": error,
        }


def runtime_contract(
    status: dict[str, Any], *, expected_tp_size: int = 2, expected_model: str = "auto"
) -> dict[str, Any]:
    hybrid = status.get("hybrid_state") or {}
    model = status.get("model") or {}
    dense = (
        model.get("architecture") == "Qwen3_5ForConditionalGeneration"
        and model.get("layer_count") == 64
        and model.get("full_attention_layers") == 16
        and model.get("linear_attention_layers") == 48
    )
    moe = (
        model.get("architecture") == "Qwen3_5MoeForConditionalGeneration"
        and model.get("layer_count") == 40
        and model.get("full_attention_layers") == 10
        and model.get("linear_attention_layers") == 30
    )
    model_ok = {"auto": dense or moe, "dense": dense, "moe": moe}.get(expected_model)
    if model_ok is None:
        raise ValueError(f"unknown expected model variant: {expected_model!r}")
    checks = {
        "runtime_ready": status.get("runtime_state") == "ready",
        "tp_size": status.get("tp_size") == expected_tp_size
        and hybrid.get("tp_size") == expected_tp_size,
        "bf16": hybrid.get("dtype") == "bfloat16"
        and hybrid.get("mamba_state_dtype") == "bfloat16",
        "page_size": hybrid.get("page_size") == 64,
        "atomic_hybrid_lifecycle": hybrid.get("atomic_full_gdn_lifecycle") is True,
        "scheduler_native_jobs": status.get("scheduler_native_internal_jobs") is True,
        "verified_qwen35_layout": model_ok,
    }
    return {"passed": all(checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged QWEN-EXO Responses smoke test")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--expected-tp-size", type=int, default=2)
    parser.add_argument(
        "--expected-model", choices=("auto", "dense", "moe"), default="auto"
    )
    parser.add_argument("--expected-gpu-count", type=int)
    parser.add_argument("--model", default="qwen-exo")
    parser.add_argument(
        "--stages",
        default="1024",
        help="Comma-separated target prompt sizes, e.g. 1024,32768,100000",
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run an additional concurrent batch at the first stage size",
    )
    parser.add_argument(
        "--overload-concurrency",
        type=int,
        default=8,
        help="Concurrent requests used to prove structured admission 429 behavior",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use the legacy Demo health surface instead of QWEN-EXO status",
    )
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Require a completed response but do not require the marker text",
    )
    parser.add_argument(
        "--max-gpu-memory-fraction",
        type=float,
        default=0.95,
        help="Fail if either target GPU exceeds this used-memory fraction",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    stages = tuple(int(value) for value in args.stages.split(",") if value.strip())
    if not stages or any(value < 128 for value in stages):
        raise ValueError("Smoke stages must be at least 128 tokens")
    if args.concurrency < 1:
        raise ValueError("Concurrency must be at least one")
    if not 0 < args.max_gpu_memory_fraction < 1:
        raise ValueError("Maximum GPU memory fraction must be between zero and one")
    if args.overload_concurrency < 0:
        raise ValueError("Overload concurrency cannot be negative")

    report: dict[str, Any] = {
        "base_url": base_url,
        "model": args.model,
        "created_at": time.time(),
        "health": request_json(
            f"{base_url}/health" if args.baseline else f"{base_url}/qwen-exo/health",
            timeout=args.timeout,
        ),
        "status": (
            {}
            if args.baseline
            else request_json(f"{base_url}/qwen-exo/status", timeout=args.timeout)
        ),
        "stages": [],
        "gpu_memory_before": gpu_memory_snapshot(args.expected_gpu_count),
    }
    report["runtime_contract"] = (
        {"passed": True, "checks": {"legacy_baseline": True}}
        if args.baseline
        else runtime_contract(
            report["status"],
            expected_tp_size=args.expected_tp_size,
            expected_model=args.expected_model,
        )
    )
    exit_code = 0 if report["runtime_contract"]["passed"] else 1
    for target_tokens in stages:
        if exit_code != 0:
            break
        stage = run_stage_safely(
            base_url,
            args.model,
            target_tokens,
            args.timeout,
            require_marker=not args.latency_only,
        )
        if stage.get("passed"):
            attach_memory_evidence(
                stage,
                max_fraction=args.max_gpu_memory_fraction,
                expected_gpu_count=args.expected_gpu_count,
            )
        report["stages"].append(stage)
        if not stage["passed"]:
            exit_code = 1
            break

    if exit_code == 0 and not args.baseline and args.concurrency > 1:
        report["pre_concurrency_quiescence"] = wait_for_adaptive_idle(
            base_url, args.timeout
        )
        if not report["pre_concurrency_quiescence"]["passed"]:
            exit_code = 1

    if exit_code == 0 and args.concurrency > 1:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    run_stage_safely,
                    base_url,
                    args.model,
                    stages[0],
                    args.timeout,
                    not args.latency_only,
                )
                for _ in range(args.concurrency)
            ]
            concurrent_stages = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        report["concurrency"] = {
            "requests": args.concurrency,
            "target_tokens": stages[0],
            "passed": all(item["passed"] for item in concurrent_stages),
            "wall_seconds": wall_seconds,
            "sum_request_seconds": sum(
                float(item.get("elapsed_seconds") or 0) for item in concurrent_stages
            ),
            "overlap_factor": (
                sum(
                    float(item.get("elapsed_seconds") or 0)
                    for item in concurrent_stages
                )
                / wall_seconds
                if wall_seconds > 0
                else 0
            ),
            "stages": concurrent_stages,
        }
        attach_memory_evidence(
            report["concurrency"],
            max_fraction=args.max_gpu_memory_fraction,
            expected_gpu_count=args.expected_gpu_count,
        )
        if not report["concurrency"]["passed"]:
            exit_code = 1

    if exit_code == 0 and not args.baseline and args.overload_concurrency > 0:
        report["pre_overload_quiescence"] = wait_for_adaptive_idle(
            base_url, args.timeout
        )
        if not report["pre_overload_quiescence"]["passed"]:
            exit_code = 1

    if exit_code == 0 and not args.baseline and args.overload_concurrency > 0:
        max_running_requests = int(report["status"].get("max_running_requests") or 4)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.overload_concurrency
        ) as executor:
            overload_attempts = list(
                executor.map(
                    lambda attempt: run_overload_attempt(
                        base_url,
                        args.model,
                        stages[0],
                        args.timeout,
                        attempt,
                    ),
                    range(args.overload_concurrency),
                )
            )
        accepted_count = sum(
            attempt["status_code"] == 200 for attempt in overload_attempts
        )
        rejected_count = sum(
            attempt["status_code"] == 429 for attempt in overload_attempts
        )
        health_after = request_json(f"{base_url}/qwen-exo/health", timeout=args.timeout)
        report["overload"] = {
            "requests": args.overload_concurrency,
            "max_running_requests": max_running_requests,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "attempts": overload_attempts,
            "health_after": health_after,
            "passed": rejected_count >= 1
            and accepted_count <= max_running_requests
            and all(attempt["passed"] for attempt in overload_attempts)
            and health_after.get("runtime_state") == "ready",
        }
        attach_memory_evidence(
            report["overload"],
            max_fraction=args.max_gpu_memory_fraction,
            expected_gpu_count=args.expected_gpu_count,
        )
        if not report["overload"]["passed"]:
            exit_code = 1

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
