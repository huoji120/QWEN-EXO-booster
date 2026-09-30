"""Run the native launcher against an isolated executable service fixture."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(
    BASH is None or os.name == "nt", reason="POSIX bash is required"
)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def launch_environment(tmp_path):
    source = tmp_path / "source with spaces"
    scripts = source / "scripts" / "qwen_exo"
    scripts.mkdir(parents=True)
    for name in ("launch_js4090.sh", "launch_native_js4090.sh"):
        shutil.copyfile(ROOT / "scripts" / "qwen_exo" / name, scripts / name)
    engine = source / "python" / "sglang" / "srt"
    engine.mkdir(parents=True)
    (engine / "server_args.py").touch()
    model = tmp_path / "catalog" / "model"
    model.mkdir(parents=True)
    (model / "sentinel").write_text("model-data")
    data = tmp_path / "runtime"
    (data / "policydata").mkdir(parents=True)
    (data / "policydata" / "operator.md").write_text("operator policy")
    pre = tmp_path / "separate-pre-complete"
    pre.mkdir()
    (pre / "sentinel").write_text("pre-complete-data")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\nexit 0\n")
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_EXO_")}
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
        "QWEN_EXO_SOURCE_PATH": str(source),
        "QWEN_EXO_MODEL_PATH": str(model),
        "QWEN_EXO_DATA_PATH": str(data),
        "QWEN_EXO_PRE_COMPLETE_PATH": str(pre),
        "QWEN_EXO_PYTHON": sys.executable,
        "QWEN_EXO_ENABLED": "0",
    })
    return scripts / "launch_native_js4090.sh", source, data, env


def test_native_service_reads_host_paths_and_propagates_exit(launch_environment):
    launcher, source, data, env = launch_environment
    package = source / "python" / "qwen_exo_booster"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "service_launcher.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "model = Path(args[args.index('--model-path') + 1])\n"
        "assert (model / 'sentinel').read_text() == 'model-data'\n"
        "pre = Path(os.environ['QWEN_EXO_PRE_COMPLETE_KNOWLEDGE_DIR'])\n"
        "assert (pre / 'sentinel').read_text() == 'pre-complete-data'\n"
        "runtime = Path(os.environ['QWEN_EXO_MODEL_DATA_ROOT'])\n"
        "assert (runtime / 'policydata/operator.md').read_text() == 'operator policy'\n"
        "(runtime / 'service-result').write_text('host paths accessible')\n"
        "sys.exit(23)\n"
    )
    result = subprocess.run([BASH, str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    assert (data / "service-result").read_text() == "host paths accessible"


@pytest.mark.parametrize("python_path", [None, "/nonexistent/qwen-python"])
def test_native_rejects_missing_interpreter(launch_environment, python_path):
    launcher, _, data, env = launch_environment
    if python_path is None:
        env.pop("QWEN_EXO_PYTHON")
    else:
        env["QWEN_EXO_PYTHON"] = python_path
    result = subprocess.run([BASH, str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (data / "service-result").exists()
