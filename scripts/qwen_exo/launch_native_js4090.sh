#!/usr/bin/env bash
set -euo pipefail

# Native fallback for js4090 hosts without Docker. The shared launcher owns
# validation, environment construction, path translation, and server argv.
REPO=${QWEN_EXO_SOURCE_PATH:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
: "${QWEN_EXO_PYTHON:?QWEN_EXO_PYTHON must name the native SGLang Python executable}"
if [[ ! -x "${QWEN_EXO_PYTHON}" ]]; then
  echo "Native Python executable not found or not executable: ${QWEN_EXO_PYTHON}" >&2
  exit 1
fi

export QWEN_EXO_SOURCE_PATH="${REPO}"
export QWEN_EXO_NATIVE=1
export PATH="$(dirname -- "${QWEN_EXO_PYTHON}"):${PATH:-}"
export PYTHONPATH="${REPO}/python${PYTHONPATH:+:${PYTHONPATH}}"

exec bash "${REPO}/scripts/qwen_exo/launch_js4090.sh"
