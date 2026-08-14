from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = SOURCE_ROOT / "scripts" / "qwen_exo" / "launch_mlx.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def test_mlx_launcher_dry_run_freezes_safe_single_gpu_profile(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uname",
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  -s) printf "Darwin\\n" ;;\n'
        '  -m) printf "arm64\\n" ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n",
    )

    invocation_log = tmp_path / "python-invocations.log"
    fake_python = fake_bin / "python"
    _write_executable(
        fake_python,
        "#!/bin/sh\n"
        'printf "%s | %s\\n" "$QWEN_EXO_SERVICE_CONFIG" "$*" '
        '>> "$QWEN_EXO_TEST_LOG"\n',
    )

    model_path = tmp_path / "model"
    data_path = tmp_path / "data"
    model_path.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "QWEN_EXO_SOURCE_PATH": str(SOURCE_ROOT),
            "QWEN_EXO_PYTHON": str(fake_python),
            "QWEN_EXO_MODEL_PATH": str(model_path),
            "QWEN_EXO_DATA_PATH": str(data_path),
            "QWEN_EXO_TEST_LOG": str(invocation_log),
            "QWEN_EXO_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    command = shlex.split(result.stdout.strip())
    assert command[0] == "SGLANG_USE_MLX=1"
    assert command[1:4] == [
        str(fake_python),
        "-m",
        "qwen_exo_booster.service_launcher",
    ]
    assert command[4] == "--"
    arguments = command[5:]
    assert arguments[arguments.index("--device") + 1] == "mps"
    assert arguments[arguments.index("--tp-size") + 1] == "1"
    assert arguments[arguments.index("--context-length") + 1] == "102400"
    assert arguments[arguments.index("--max-total-tokens") + 1] == "102400"
    assert arguments[arguments.index("--max-running-requests") + 1] == "64"
    assert arguments[arguments.index("--max-prefill-tokens") + 1] == "65536"
    assert arguments[arguments.index("--mem-fraction-static") + 1] == "0.80"
    assert arguments[arguments.index("--qwen-exo-max-internal-fanout") + 1] == "32"
    assert (
        arguments[
            arguments.index("--qwen-exo-reflection-memory-max-history-tokens") + 1
        ]
        == "92160"
    )
    assert arguments[arguments.index("--quantization") + 1] == "mlx_q4"
    assert arguments[arguments.index("--kv-cache-dtype") + 1] == "mxfp8"
    assert arguments[arguments.index("--default-chat-template-kwargs") + 1] == (
        '{"enable_thinking": false, "preserve_thinking": false}'
    )
    assert arguments[arguments.index("--mamba-radix-cache-strategy") + 1] == (
        "no_buffer"
    )
    assert "--disable-cuda-graph" in arguments
    assert "--disable-overlap-schedule" in arguments
    assert "--enable-qwen-exo" in arguments
    assert arguments[arguments.index("--qwen-exo-state-dir") + 1].endswith(
        "/state-mlx-tp1-mlx_q4-mxfp8"
    )

    invocations = invocation_log.read_text(encoding="utf-8")
    assert str(data_path / "service-config-mlx.json") in invocations
    assert "scripts/qwen_exo/check_mlx.py" in invocations
    assert "python/qwen_exo_booster/fingerprint.py" in invocations


def test_mlx_launcher_accepts_mxfp8_profile(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uname",
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  -s) printf "Darwin\\n" ;;\n'
        '  -m) printf "arm64\\n" ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n",
    )
    fake_python = fake_bin / "python"
    _write_executable(fake_python, "#!/bin/sh\nexit 0\n")
    model_path = tmp_path / "model"
    data_path = tmp_path / "data"
    model_path.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "QWEN_EXO_SOURCE_PATH": str(SOURCE_ROOT),
            "QWEN_EXO_PYTHON": str(fake_python),
            "QWEN_EXO_MODEL_PATH": str(model_path),
            "QWEN_EXO_DATA_PATH": str(data_path),
            "QWEN_EXO_QUANTIZATION": "mlx_mxfp8",
            "QWEN_EXO_DRY_RUN": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    arguments = shlex.split(result.stdout.strip())[5:]
    assert arguments[arguments.index("--quantization") + 1] == "mlx_mxfp8"
    assert arguments[arguments.index("--qwen-exo-state-dir") + 1].endswith(
        "/state-mlx-tp1-mlx_mxfp8-mxfp8"
    )
