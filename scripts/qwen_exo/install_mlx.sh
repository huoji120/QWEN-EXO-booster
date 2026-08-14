#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_EXO_SOURCE_PATH:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${QWEN_EXO_VENV:=${QWEN_EXO_SOURCE_PATH}/.venv}"
: "${QWEN_EXO_PYTHON:=python3}"
: "${QWEN_EXO_INSTALL_TEST_DEPS:=0}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "QWEN-EXO MLX installation requires an Apple Silicon Mac." >&2
  exit 1
fi
if [[ ! -f "${QWEN_EXO_SOURCE_PATH}/python/pyproject_other.toml" ]]; then
  echo "QWEN-EXO source tree not found: ${QWEN_EXO_SOURCE_PATH}" >&2
  exit 1
fi

"${QWEN_EXO_PYTHON}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("QWEN-EXO MLX installer requires Python 3.11 or newer")
PY

if [[ ! -x "${QWEN_EXO_VENV}/bin/python" ]]; then
  "${QWEN_EXO_PYTHON}" -m venv "${QWEN_EXO_VENV}"
fi

requirements_file="$(mktemp "${TMPDIR:-/tmp}/qwen-exo-mlx-requirements.XXXXXX")"
cleanup() {
  rm -f -- "${requirements_file}"
}
trap cleanup EXIT

"${QWEN_EXO_VENV}/bin/python" - \
  "${QWEN_EXO_SOURCE_PATH}/python/pyproject_other.toml" \
  "${requirements_file}" \
  "${QWEN_EXO_INSTALL_TEST_DEPS}" <<'PY'
from pathlib import Path
import sys
import tomllib

source = Path(sys.argv[1])
target = Path(sys.argv[2])
include_tests = sys.argv[3] == "1"
project = tomllib.loads(source.read_text(encoding="utf-8"))["project"]
extras = project["optional-dependencies"]
requirements = list(extras["runtime_common"])
requirements.extend(
    item for item in extras["srt_mps"] if not item.startswith("sglang[")
)
if include_tests:
    requirements.extend(extras["test"])
target.write_text("\n".join(dict.fromkeys(requirements)) + "\n", encoding="utf-8")
PY

"${QWEN_EXO_VENV}/bin/python" -m pip install --upgrade pip
"${QWEN_EXO_VENV}/bin/python" -m pip install -r "${requirements_file}"
SGLANG_BUILD_RUST_EXTS=none \
  "${QWEN_EXO_VENV}/bin/python" -m pip install --no-deps -e \
  "${QWEN_EXO_SOURCE_PATH}/python"

PYTHONPATH="${QWEN_EXO_SOURCE_PATH}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${QWEN_EXO_VENV}/bin/python" \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/check_mlx.py"

printf 'QWEN-EXO MLX environment ready: %s\n' "${QWEN_EXO_VENV}"
