#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_EXO_ENABLED:=1}"
: "${QWEN_EXO_SOURCE_PATH:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${QWEN_EXO_PYTHON:=${QWEN_EXO_SOURCE_PATH}/.venv/bin/python}"
: "${QWEN_EXO_CONTEXT_LENGTH:=102400}"
: "${QWEN_EXO_DTYPE:=float16}"
: "${QWEN_EXO_QUANTIZATION:=mlx_q4}"
: "${QWEN_EXO_KV_CACHE_DTYPE:=mxfp8}"
: "${QWEN_EXO_STATE_DIRECTORY_NAME:=state-mlx-tp1-${QWEN_EXO_QUANTIZATION}-${QWEN_EXO_KV_CACHE_DTYPE}}"
: "${QWEN_EXO_MEM_FRACTION_STATIC:=0.80}"
: "${QWEN_EXO_MAX_RUNNING_REQUESTS:=64}"
# MLX uses one shared unified-memory KV budget. Keep a full single-request
# context without multiplying the allocation by the concurrency admission cap.
: "${QWEN_EXO_MAX_TOTAL_TOKENS:=${QWEN_EXO_CONTEXT_LENGTH}}"
: "${QWEN_EXO_MAX_PREFILL_TOKENS:=65536}"
: "${QWEN_EXO_MAX_INTERNAL_FANOUT:=32}"
: "${QWEN_EXO_MAX_INTERNAL_TOKENS:=12288}"
: "${QWEN_EXO_MAX_OUTPUT_TOKENS:=8192}"
: "${QWEN_EXO_MAX_REASONING_TOKENS:=3072}"
: "${QWEN_EXO_TENSOR_BANK_MAX_DOCUMENT_TOKENS:=$((QWEN_EXO_CONTEXT_LENGTH - 2048))}"
: "${QWEN_EXO_TENSOR_BANK_SALIENT_TOKEN_BUDGET:=4096}"
: "${QWEN_EXO_TENSOR_BANK_SURPRISAL_THRESHOLD:=6.0}"
: "${QWEN_EXO_TENSOR_BANK_SPAN_TOKENS:=16}"
: "${QWEN_EXO_OBSERVER_MODE:=active}"
: "${QWEN_EXO_ENABLE_ADAPTIVE_REFRESH:=1}"
: "${QWEN_EXO_CONTEXT_EVIDENCE_MODE:=active}"
: "${QWEN_EXO_CONTEXT_INTEGRITY_MODE:=active}"
: "${QWEN_EXO_CONTEXT_INTEGRITY_CONTEXT_DIVISOR:=3}"
: "${QWEN_EXO_REFLECTION_MEMORY_MODE:=active}"
: "${QWEN_EXO_REFLECTION_MEMORY_MAX_HISTORY_TOKENS:=92160}"
: "${QWEN_EXO_RESPONSE_COMPACTION_MODE:=active}"
: "${QWEN_EXO_QK_RECALL_PRESET:=balanced}"
: "${QWEN_EXO_QK_PREFILTER_MODE:=active}"
: "${QWEN_EXO_QK_EXPANSION_MARGIN:=0.02}"
: "${QWEN_EXO_SCORE_BIAS_MODE:=trajectory_active}"
: "${QWEN_EXO_CONSOLE_TRACE_DEFAULT_SCOPE:=activity}"
: "${QWEN_EXO_PORT:=30000}"
: "${QWEN_EXO_DRY_RUN:=0}"
: "${SGLANG_MLX_CLEAR_CACHE_STEPS:=1}"
: "${SGLANG_MLX_CACHE_LIMIT_GIB:=2}"

case "${QWEN_EXO_ENABLED}" in
  0|1) ;;
  *)
    printf 'Invalid QWEN_EXO_ENABLED=%s; expected 0 or 1.\n' "${QWEN_EXO_ENABLED}" >&2
    exit 1
    ;;
esac
case "${QWEN_EXO_QUANTIZATION}" in
  none|mlx_q4|mlx_q8|mlx_mxfp8) ;;
  *)
    printf 'Invalid QWEN_EXO_QUANTIZATION=%s; expected none, mlx_q4, mlx_q8, or mlx_mxfp8.\n' "${QWEN_EXO_QUANTIZATION}" >&2
    exit 1
    ;;
esac
case "${QWEN_EXO_KV_CACHE_DTYPE}" in
  auto|bf16|bfloat16|mxfp8) ;;
  *)
    printf 'Invalid QWEN_EXO_KV_CACHE_DTYPE=%s; expected auto, bf16, bfloat16, or mxfp8.\n' "${QWEN_EXO_KV_CACHE_DTYPE}" >&2
    exit 1
    ;;
esac
case "${QWEN_EXO_DRY_RUN}" in
  0|1) ;;
  *)
    printf 'Invalid QWEN_EXO_DRY_RUN=%s; expected 0 or 1.\n' "${QWEN_EXO_DRY_RUN}" >&2
    exit 1
    ;;
esac

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "QWEN-EXO MLX requires an Apple Silicon Mac." >&2
  exit 1
fi
if [[ -z "${QWEN_EXO_MODEL_PATH:-}" ]]; then
  echo "QWEN_EXO_MODEL_PATH is required; set it to a local checkpoint directory." >&2
  exit 1
fi
if [[ -z "${QWEN_EXO_DATA_PATH:-}" ]]; then
  echo "QWEN_EXO_DATA_PATH is required; set it to a persistent runtime directory." >&2
  exit 1
fi
if [[ ! -d "${QWEN_EXO_MODEL_PATH}" ]]; then
  echo "Model directory not found: ${QWEN_EXO_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -x "${QWEN_EXO_PYTHON}" ]]; then
  echo "MLX Python environment not found: ${QWEN_EXO_PYTHON}" >&2
  echo "Run: bash scripts/qwen_exo/install_mlx.sh" >&2
  exit 1
fi
if [[ ! -f "${QWEN_EXO_SOURCE_PATH}/python/sglang/srt/server_args.py" ]]; then
  echo "QWEN-EXO source tree not found: ${QWEN_EXO_SOURCE_PATH}" >&2
  exit 1
fi
if (( QWEN_EXO_CONTEXT_LENGTH < 4096 )); then
  echo "QWEN_EXO_CONTEXT_LENGTH must be at least 4096." >&2
  exit 1
fi
if (( QWEN_EXO_MAX_PREFILL_TOKENS > QWEN_EXO_CONTEXT_LENGTH )); then
  echo "QWEN_EXO_MAX_PREFILL_TOKENS cannot exceed QWEN_EXO_CONTEXT_LENGTH." >&2
  exit 1
fi

export PYTHONPATH="${QWEN_EXO_SOURCE_PATH}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_USE_MLX=1
export SGLANG_LOGPROB_CHUNK_SIZE=512
export SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB=512
export SGLANG_MLX_CLEAR_CACHE_STEPS
export SGLANG_MLX_CACHE_LIMIT_GIB
export QWEN_EXO_SERVICE_CONFIG="${QWEN_EXO_DATA_PATH}/service-config-mlx.json"

"${QWEN_EXO_PYTHON}" "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/check_mlx.py"
if [[ "${QWEN_EXO_ENABLED}" == "1" ]]; then
  "${QWEN_EXO_PYTHON}" \
    "${QWEN_EXO_SOURCE_PATH}/python/qwen_exo_booster/fingerprint.py" \
    "${QWEN_EXO_MODEL_PATH}"
fi

seed_corpus() {
  local source_directory="$1"
  local target_directory="$2"
  if [[ -d "${source_directory}" ]]; then
    mkdir -p "${target_directory}"
    cp -a "${source_directory}/." "${target_directory}/"
  fi
}

mkdir -p \
  "${QWEN_EXO_DATA_PATH}/knowledge" \
  "${QWEN_EXO_DATA_PATH}/policydata" \
  "${QWEN_EXO_DATA_PATH}/cognition" \
  "${QWEN_EXO_DATA_PATH}/${QWEN_EXO_STATE_DIRECTORY_NAME}" \
  "${QWEN_EXO_DATA_PATH}/logs"
seed_corpus \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/knowledge" \
  "${QWEN_EXO_DATA_PATH}/knowledge"
seed_corpus \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/policydata" \
  "${QWEN_EXO_DATA_PATH}/policydata"
rm -f "${QWEN_EXO_DATA_PATH}/cognition/gpt-identity-card.md"
seed_corpus \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/cognition" \
  "${QWEN_EXO_DATA_PATH}/cognition"

server_args=(
  --model-path "${QWEN_EXO_MODEL_PATH}"
  --served-model-name duckgpt
  --device mps
  --tp-size 1
  --dtype "${QWEN_EXO_DTYPE}"
  --kv-cache-dtype "${QWEN_EXO_KV_CACHE_DTYPE}"
  --context-length "${QWEN_EXO_CONTEXT_LENGTH}"
  --mem-fraction-static "${QWEN_EXO_MEM_FRACTION_STATIC}"
  --max-running-requests "${QWEN_EXO_MAX_RUNNING_REQUESTS}"
  --max-total-tokens "${QWEN_EXO_MAX_TOTAL_TOKENS}"
  --max-prefill-tokens "${QWEN_EXO_MAX_PREFILL_TOKENS}"
  --disable-cuda-graph
  --disable-overlap-schedule
  --mamba-radix-cache-strategy no_buffer
  --page-size 1
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs
  '{"enable_thinking": false, "preserve_thinking": false}'
  --watchdog-timeout 1200
  --host 127.0.0.1
  --port "${QWEN_EXO_PORT}"
)
if [[ "${QWEN_EXO_QUANTIZATION}" != "none" ]]; then
  server_args+=( --quantization "${QWEN_EXO_QUANTIZATION}" )
fi
if [[ "${QWEN_EXO_ENABLED}" == "1" ]]; then
  server_args+=(
    --enable-qwen-exo
    --qwen-exo-state-dir "${QWEN_EXO_DATA_PATH}/${QWEN_EXO_STATE_DIRECTORY_NAME}"
    --qwen-exo-knowledge-dir "${QWEN_EXO_DATA_PATH}/knowledge"
    --qwen-exo-policy-data-dir "${QWEN_EXO_DATA_PATH}/policydata"
    --qwen-exo-cognition-dir "${QWEN_EXO_DATA_PATH}/cognition"
    --qwen-exo-max-internal-fanout "${QWEN_EXO_MAX_INTERNAL_FANOUT}"
    --qwen-exo-max-internal-tokens "${QWEN_EXO_MAX_INTERNAL_TOKENS}"
    --qwen-exo-max-output-tokens "${QWEN_EXO_MAX_OUTPUT_TOKENS}"
    --qwen-exo-max-reasoning-tokens "${QWEN_EXO_MAX_REASONING_TOKENS}"
    --qwen-exo-tensor-bank-max-document-tokens "${QWEN_EXO_TENSOR_BANK_MAX_DOCUMENT_TOKENS}"
    --qwen-exo-tensor-bank-salient-token-budget "${QWEN_EXO_TENSOR_BANK_SALIENT_TOKEN_BUDGET}"
    --qwen-exo-tensor-bank-surprisal-threshold "${QWEN_EXO_TENSOR_BANK_SURPRISAL_THRESHOLD}"
    --qwen-exo-tensor-bank-span-tokens "${QWEN_EXO_TENSOR_BANK_SPAN_TOKENS}"
    --qwen-exo-observer-mode "${QWEN_EXO_OBSERVER_MODE}"
    --qwen-exo-qk-recall-preset "${QWEN_EXO_QK_RECALL_PRESET}"
    --qwen-exo-qk-prefilter-mode "${QWEN_EXO_QK_PREFILTER_MODE}"
    --qwen-exo-qk-expansion-margin "${QWEN_EXO_QK_EXPANSION_MARGIN}"
    --qwen-exo-console-trace-default-scope "${QWEN_EXO_CONSOLE_TRACE_DEFAULT_SCOPE}"
    --qwen-exo-context-evidence-mode "${QWEN_EXO_CONTEXT_EVIDENCE_MODE}"
    --qwen-exo-context-integrity-mode "${QWEN_EXO_CONTEXT_INTEGRITY_MODE}"
    --qwen-exo-context-integrity-context-divisor "${QWEN_EXO_CONTEXT_INTEGRITY_CONTEXT_DIVISOR}"
    --qwen-exo-reflection-memory-mode "${QWEN_EXO_REFLECTION_MEMORY_MODE}"
    --qwen-exo-reflection-memory-max-history-tokens "${QWEN_EXO_REFLECTION_MEMORY_MAX_HISTORY_TOKENS}"
    --qwen-exo-response-compaction-mode "${QWEN_EXO_RESPONSE_COMPACTION_MODE}"
    --qwen-exo-score-bias-mode "${QWEN_EXO_SCORE_BIAS_MODE}"
  )
  if [[ "${QWEN_EXO_ENABLE_ADAPTIVE_REFRESH}" == "1" ]]; then
    if [[ "${QWEN_EXO_OBSERVER_MODE}" != "active" ]]; then
      echo "Adaptive refresh requires QWEN_EXO_OBSERVER_MODE=active" >&2
      exit 1
    fi
    server_args+=( --qwen-exo-enable-adaptive-refresh )
  fi
fi

if [[ "${QWEN_EXO_DRY_RUN}" == "1" ]]; then
  printf 'SGLANG_USE_MLX=1'
  printf ' %q' "${QWEN_EXO_PYTHON}" -m qwen_exo_booster.service_launcher -- \
    "${server_args[@]}"
  printf '\n'
  exit 0
fi

exec "${QWEN_EXO_PYTHON}" -m qwen_exo_booster.service_launcher -- \
  "${server_args[@]}"
