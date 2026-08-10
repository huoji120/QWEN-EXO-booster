from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DeepSWE failures up to K times, stop each task at its first "
            "binary success, and persist pass@K statistics."
        )
    )
    parser.add_argument("--dataset", required=True, help="DeepSWE dataset directory")
    parser.add_argument(
        "--first-pass-dir",
        required=True,
        help="Completed or running one-attempt Pier job directory",
    )
    parser.add_argument("--jobs-dir", required=True, help="Directory for retry jobs")
    parser.add_argument("--job-prefix", required=True)
    parser.add_argument(
        "--runner",
        default="/data/QWEN-EXO-booster/scripts/qwen_exo/run_deep_swe_compressed.py",
    )
    parser.add_argument(
        "--config",
        default=(
            "/data/QWEN-EXO-booster/scripts/qwen_exo/"
            "mini_swe_agent_qwen_exo_code.yaml"
        ),
    )
    parser.add_argument("--model", default="openai/qwen-exo")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--first-pass-pid",
        type=int,
        help="Wait for this first-pass Pier process before starting retries",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--trace-path",
        default="/data/qwen-exo-booster/state/trace.jsonl",
    )
    parser.add_argument(
        "--summary",
        help="Summary JSON path; defaults to <jobs-dir>/pass-at-k-summary.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest(dataset: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json(dataset / "manifest.json")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Manifest has no task list: {dataset / 'manifest.json'}")
    return manifest, tasks


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_first_pass(pid: int, poll_seconds: float) -> None:
    while _process_is_running(pid):
        print(f"[pass-at-k] waiting for first-pass pid={pid}", flush=True)
        time.sleep(poll_seconds)


def _task_id_from_config(config: dict[str, Any]) -> str | None:
    task = config.get("task")
    if not isinstance(task, dict):
        return None
    task_path = task.get("path")
    if not task_path:
        return None
    return Path(str(task_path)).name


def _result_record(
    trial_dir: Path,
    *,
    attempt: int,
    source: str,
    runner_exit_code: int | None = None,
) -> dict[str, Any]:
    result_path = trial_dir / "result.json"
    exception_path = trial_dir / "exception.txt"
    record: dict[str, Any] = {
        "attempt": attempt,
        "source": source,
        "trial_dir": str(trial_dir),
        "runner_exit_code": runner_exit_code,
        "passed": False,
        "reward": 0,
    }
    if result_path.exists():
        result = _load_json(result_path)
        rewards = result.get("verifier_result", {}).get("rewards", {})
        reward = rewards.get("reward", 0)
        record.update(
            {
                "result_path": str(result_path),
                "reward": reward,
                "passed": reward == 1,
                "f2p": rewards.get("f2p"),
                "f2p_passed": rewards.get("f2p_passed"),
                "f2p_total": rewards.get("f2p_total"),
                "p2p": rewards.get("p2p"),
                "p2p_passed": rewards.get("p2p_passed"),
                "p2p_total": rewards.get("p2p_total"),
                "partial": rewards.get("partial"),
                "started_at": result.get("started_at"),
                "finished_at": result.get("finished_at"),
                "exception_info": result.get("exception_info"),
            }
        )
    elif exception_path.exists():
        record.update(
            {
                "exception_path": str(exception_path),
                "exception": exception_path.read_text(
                    encoding="utf-8", errors="replace"
                )[:4000],
            }
        )
    else:
        record["exception"] = "No result.json or exception.txt was produced"
    return record


def _scan_first_pass(first_pass_dir: Path) -> dict[str, dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    for config_path in sorted(first_pass_dir.glob("*/config.json")):
        config = _load_json(config_path)
        task_id = _task_id_from_config(config)
        if task_id is None:
            continue
        attempts[task_id] = _result_record(
            config_path.parent,
            attempt=1,
            source="first_pass",
        )
    return attempts


def _find_retry_trial(job_dir: Path, task_id: str) -> Path | None:
    for config_path in sorted(job_dir.glob("*/config.json")):
        if _task_id_from_config(_load_json(config_path)) == task_id:
            return config_path.parent
    return None


def _copy_trace(trace_path: Path, trial_dir: Path) -> str | None:
    if not trace_path.is_file():
        return None
    destination = trial_dir / "qwen-exo.trace.jsonl"
    shutil.copy2(trace_path, destination)
    return str(destination)


def _refresh_task_stats(task: dict[str, Any], max_attempts: int) -> None:
    attempts = task["attempts"]
    successes = sum(1 for attempt in attempts if attempt.get("passed"))
    first_success = next(
        (attempt["attempt"] for attempt in attempts if attempt.get("passed")),
        None,
    )
    task.update(
        {
            "attempts_run": len(attempts),
            "successes": successes,
            "success_rate": successes / len(attempts) if attempts else 0.0,
            "first_success_attempt": first_success,
            "passed": first_success is not None,
            "pass_at_k": first_success is not None and first_success <= max_attempts,
            "exhausted": first_success is None and len(attempts) >= max_attempts,
        }
    )


def _refresh_aggregate(summary: dict[str, Any]) -> None:
    tasks = list(summary["tasks"].values())
    total_attempts = sum(task["attempts_run"] for task in tasks)
    total_successes = sum(task["successes"] for task in tasks)
    pass_count = sum(1 for task in tasks if task["pass_at_k"])
    completed_count = sum(1 for task in tasks if task["pass_at_k"] or task["exhausted"])
    task_count = len(tasks)
    summary["aggregate"] = {
        "task_count": task_count,
        "completed_tasks": completed_count,
        "remaining_tasks": task_count - completed_count,
        "pass_at_k_count": pass_count,
        "pass_at_k_rate": pass_count / task_count if task_count else 0.0,
        "completed_pass_at_k_rate": (
            pass_count / completed_count if completed_count else 0.0
        ),
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "attempt_success_rate": (
            total_successes / total_attempts if total_attempts else 0.0
        ),
    }
    summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_summary(
    *,
    manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
    first_pass_dir: Path,
    max_attempts: int,
) -> dict[str, Any]:
    summary_tasks: dict[str, Any] = {}
    for item in tasks:
        task_id = str(item["task_id"])
        summary_tasks[task_id] = {
            "task_id": task_id,
            "display_title": item.get("display_title"),
            "repo": item.get("repo"),
            "language": item.get("language"),
            "attempts": [],
        }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset": manifest.get("dataset"),
        "max_attempts": max_attempts,
        "early_stop_on_success": True,
        "first_pass_dir": str(first_pass_dir),
        "tasks": summary_tasks,
    }


def _merge_first_pass(
    summary: dict[str, Any], first_pass: dict[str, dict[str, Any]]
) -> None:
    for task_id, attempt in first_pass.items():
        task = summary["tasks"].get(task_id)
        if task is None:
            continue
        existing_numbers = {item["attempt"] for item in task["attempts"]}
        if 1 not in existing_numbers:
            task["attempts"].append(attempt)
            task["attempts"].sort(key=lambda item: item["attempt"])


def _run_attempt(
    *,
    args: argparse.Namespace,
    task_id: str,
    attempt: int,
    jobs_dir: Path,
) -> dict[str, Any]:
    job_name = f"{args.job_prefix}-{task_id}-a{attempt}"
    job_dir = jobs_dir / job_name
    command = [
        "python3",
        args.runner,
        "--task",
        str(Path(args.dataset) / task_id),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--model",
        args.model,
        "--config",
        args.config,
        "--agent-timeout-multiplier",
        str(args.agent_timeout_multiplier),
    ]
    print(
        f"[pass-at-k] task={task_id} attempt={attempt}/{args.max_attempts}",
        flush=True,
    )
    completed = subprocess.run(command, check=False)
    trial_dir = _find_retry_trial(job_dir, task_id)
    if trial_dir is None:
        trial_dir = job_dir
    record = _result_record(
        trial_dir,
        attempt=attempt,
        source="retry",
        runner_exit_code=completed.returncode,
    )
    trace_copy = _copy_trace(Path(args.trace_path), trial_dir)
    if trace_copy is not None:
        record["trace_path"] = trace_copy
    return record


def main() -> int:
    args = _parser().parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    dataset = Path(args.dataset)
    first_pass_dir = Path(args.first_pass_dir)
    jobs_dir = Path(args.jobs_dir)
    summary_path = Path(args.summary) if args.summary else jobs_dir / "pass-at-k-summary.json"
    for path in (dataset, first_pass_dir, Path(args.runner), Path(args.config)):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest, manifest_tasks = _load_manifest(dataset)
    if summary_path.exists():
        summary = _load_json(summary_path)
        if summary.get("max_attempts") != args.max_attempts:
            raise ValueError("Existing summary uses a different max_attempts")
    else:
        summary = _new_summary(
            manifest=manifest,
            tasks=manifest_tasks,
            first_pass_dir=first_pass_dir,
            max_attempts=args.max_attempts,
        )

    if args.dry_run:
        _merge_first_pass(summary, _scan_first_pass(first_pass_dir))
        for task in summary["tasks"].values():
            _refresh_task_stats(task, args.max_attempts)
        _refresh_aggregate(summary)
        print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
        return 0

    if args.first_pass_pid is not None:
        _wait_for_first_pass(args.first_pass_pid, args.poll_seconds)

    _merge_first_pass(summary, _scan_first_pass(first_pass_dir))
    for task in summary["tasks"].values():
        _refresh_task_stats(task, args.max_attempts)
    _refresh_aggregate(summary)
    _write_json_atomic(summary_path, summary)

    jobs_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest_tasks:
        task_id = str(item["task_id"])
        task = summary["tasks"][task_id]
        _refresh_task_stats(task, args.max_attempts)
        while not task["passed"] and len(task["attempts"]) < args.max_attempts:
            attempt_number = max(
                (attempt["attempt"] for attempt in task["attempts"]),
                default=0,
            ) + 1
            record = _run_attempt(
                args=args,
                task_id=task_id,
                attempt=attempt_number,
                jobs_dir=jobs_dir,
            )
            task["attempts"].append(record)
            task["attempts"].sort(key=lambda attempt: attempt["attempt"])
            _refresh_task_stats(task, args.max_attempts)
            _refresh_aggregate(summary)
            _write_json_atomic(summary_path, summary)
            if task["passed"]:
                print(
                    f"[pass-at-k] PASS task={task_id} "
                    f"attempt={task['first_success_attempt']}",
                    flush=True,
                )

    _refresh_aggregate(summary)
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
