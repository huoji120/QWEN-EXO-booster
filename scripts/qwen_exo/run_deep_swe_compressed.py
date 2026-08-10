from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(url: str, method: str, *, timeout: float = 30) -> str:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _wait_for_runtime(base: str, *, timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _request(f"{base}/health", "GET", timeout=10)
            return
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"QWEN-EXO runtime did not become ready: {base}") from last_error


def _reset_runtime(runtime_url: str) -> None:
    base = runtime_url.rstrip("/")
    _wait_for_runtime(base)
    print(
        _request(f"{base}/flush_cache", "POST", timeout=120).strip(),
        flush=True,
    )
    print(
        _request(f"{base}/qwen-exo/recall-trace", "DELETE").strip(),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a DeepSWE task or dataset with QWEN-EXO context compaction."
    )
    parser.add_argument(
        "--task", required=True, help="Frozen task or dataset directory"
    )
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--model", default="openai/qwen-exo")
    parser.add_argument("--pier", default="/root/.local/bin/pier")
    parser.add_argument(
        "--config",
        default=(
            "/data/QWEN-EXO-booster/scripts/qwen_exo/" "mini_swe_agent_qwen_exo.yaml"
        ),
    )
    parser.add_argument(
        "--agent-source",
        default="/data/QWEN-EXO-booster/scripts/qwen_exo/agents",
    )
    parser.add_argument("--api-base", default="http://172.17.0.1:30001/v1")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=1.5,
        help="Pier agent timeout multiplier; defaults to a bounded 1.5x task limit",
    )
    parser.add_argument("--no-reset-runtime", action="store_true")
    parser.add_argument(
        "--n-tasks",
        type=int,
        help="Deterministically sample this many tasks from a dataset directory",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Pier dataset sampling seed used with --n-tasks",
    )
    parser.add_argument(
        "--disable-compaction",
        action="store_true",
        help="Run the identical agent with context compaction disabled for A/B tests",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--skip-verification", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.pier,
        "run",
        "--path",
        args.task,
        "--agent-import-path",
        "qwen_exo_pier_agent:QwenExoMiniSweAgent",
        "--model",
        args.model,
        "--jobs-dir",
        args.jobs_dir,
        "--job-name",
        args.job_name,
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--agent-timeout-multiplier",
        str(args.agent_timeout_multiplier),
        "--yes",
        "--debug",
        "--ak",
        f"config_file={args.config}",
        "--ae",
        f"OPENAI_BASE_URL={args.api_base}",
        "--ae",
        f"OPENAI_API_BASE={args.api_base}",
        "--ae",
        "OPENAI_API_KEY=qwen-exo-local",
        "--ae",
        "PYTHONPATH=/tmp/qwen-exo-agent",
    ]
    if args.n_tasks is not None:
        if args.n_tasks < 1:
            raise ValueError("--n-tasks must be positive")
        command.extend(("--n-tasks", str(args.n_tasks)))
        command.extend(("--sample-seed", str(args.sample_seed)))
    if args.skip_verification:
        command.append("--disable-verification")
    if args.disable_compaction:
        command.extend(("--ae", "QWEN_EXO_COMPACTION_ENABLED=0"))
    command.append("--no-delete" if args.keep_container else "--delete")
    return command


def main() -> int:
    args = _parser().parse_args()
    for path in (args.task, args.config, args.agent_source):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0

    if not args.no_reset_runtime:
        _reset_runtime(args.runtime_url)
    pier_env = os.environ.copy()
    existing_python_path = pier_env.get("PYTHONPATH")
    pier_env["PYTHONPATH"] = str(Path(args.agent_source)) + (
        os.pathsep + existing_python_path if existing_python_path else ""
    )
    return subprocess.run(command, check=False, env=pier_env).returncode


if __name__ == "__main__":
    sys.exit(main())
