#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_EXO_IMAGE:=qwen-exo-booster:sglang-v0.5.16-cu129}"
: "${QWEN_EXO_CONTAINER:=qwen-exo-booster}"
: "${QWEN_EXO_MODEL_PATH:=/data500/models/2026_7_18_memit_test_finally}"
: "${QWEN_EXO_DATA_PATH:=/data/qwen-exo-booster}"
: "${QWEN_EXO_CONTEXT_LENGTH:=102400}"
: "${QWEN_EXO_MEM_FRACTION_STATIC:=0.80}"
: "${QWEN_EXO_MAX_RUNNING_REQUESTS:=4}"
: "${QWEN_EXO_PORT:=30000}"
: "${QWEN_EXO_MAMBA_STRATEGY:=extra_buffer}"

if [[ ! -d "${QWEN_EXO_MODEL_PATH}" ]]; then
  echo "Model directory not found: ${QWEN_EXO_MODEL_PATH}" >&2
  exit 1
fi

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | sort -u)"
if [[ -n "${active_pids}" ]]; then
  echo "Refusing to start while GPU compute PIDs are active: ${active_pids}" >&2
  exit 1
fi

mkdir -p \
  "${QWEN_EXO_DATA_PATH}/knowledge" \
  "${QWEN_EXO_DATA_PATH}/state" \
  "${QWEN_EXO_DATA_PATH}/logs"

docker rm -f "${QWEN_EXO_CONTAINER}" >/dev/null 2>&1 || true

exec docker run --rm \
  --name "${QWEN_EXO_CONTAINER}" \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_SHM_DISABLE=0 \
  -e SGLANG_MAMBA_SSM_DTYPE=bfloat16 \
  -v "${QWEN_EXO_MODEL_PATH}:/models/qwen-exo-27b:ro" \
  -v "${QWEN_EXO_DATA_PATH}:/data/qwen-exo" \
  "${QWEN_EXO_IMAGE}" \
  python3 -m sglang.launch_server \
    --model-path /models/qwen-exo-27b \
    --served-model-name duckgpt \
    --tp-size 2 \
    --dtype bfloat16 \
    --context-length "${QWEN_EXO_CONTEXT_LENGTH}" \
    --mem-fraction-static "${QWEN_EXO_MEM_FRACTION_STATIC}" \
    --max-running-requests "${QWEN_EXO_MAX_RUNNING_REQUESTS}" \
    --disable-custom-all-reduce \
    --enable-priority-scheduling \
    --mamba-radix-cache-strategy "${QWEN_EXO_MAMBA_STRATEGY}" \
    --page-size 64 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --watchdog-timeout 1200 \
    --enable-qwen-exo \
    --qwen-exo-state-dir /data/qwen-exo/state \
    --qwen-exo-knowledge-dir /data/qwen-exo/knowledge \
    --host 127.0.0.1 \
    --port "${QWEN_EXO_PORT}"
