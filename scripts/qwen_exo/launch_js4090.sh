#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_EXO_IMAGE:=qwen-exo-booster:sglang-v0.5.16-driver550}"
: "${QWEN_EXO_ENABLED:=1}"
: "${QWEN_EXO_CONTAINER:=qwen-exo-booster}"
: "${QWEN_EXO_SOURCE_PATH:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${QWEN_EXO_CONTEXT_LENGTH:=102400}"
: "${QWEN_EXO_TP_SIZE:=2}"
: "${QWEN_EXO_DOCKER_GPUS:=all}"
: "${QWEN_EXO_DTYPE:=bfloat16}"
: "${QWEN_EXO_QUANTIZATION:=}"
: "${QWEN_EXO_KV_CACHE_DTYPE:=fp8_e4m3}"
if [[ -n "${QWEN_EXO_QUANTIZATION}" ]]; then
  QWEN_EXO_QUANTIZATION_LABEL="${QWEN_EXO_QUANTIZATION}"
else
  QWEN_EXO_QUANTIZATION_LABEL=unquant
fi
: "${QWEN_EXO_STATE_DIRECTORY_NAME:=state-cuda-tp${QWEN_EXO_TP_SIZE}-${QWEN_EXO_QUANTIZATION_LABEL}-${QWEN_EXO_KV_CACHE_DTYPE}}"
: "${QWEN_EXO_USE_JIT_FP8_QUANT:=1}"
: "${QWEN_EXO_LOGPROB_CHUNK_SIZE:=512}"
: "${QWEN_EXO_MEM_FRACTION_STATIC:=0.80}"
: "${QWEN_EXO_MAX_RUNNING_REQUESTS:=64}"
: "${QWEN_EXO_CPU_OFFLOAD_GB:=0}"
: "${QWEN_EXO_CUDA_GRAPH_MAX_BS:=8}"
: "${QWEN_EXO_PORT:=30000}"
: "${QWEN_EXO_MAX_INTERNAL_FANOUT:=32}"
: "${QWEN_EXO_MAX_INTERNAL_TOKENS:=12288}"
: "${QWEN_EXO_MAX_PREFILL_TOKENS:=65536}"
: "${QWEN_EXO_MAX_OUTPUT_TOKENS:=8192}"
: "${QWEN_EXO_MAX_REASONING_TOKENS:=3072}"
: "${QWEN_EXO_TENSOR_BANK_MAX_DOCUMENT_TOKENS:=$((QWEN_EXO_CONTEXT_LENGTH - 2048))}"
: "${QWEN_EXO_TENSOR_BANK_SALIENT_TOKEN_BUDGET:=4096}"
: "${QWEN_EXO_TENSOR_BANK_SURPRISAL_THRESHOLD:=6.0}"
: "${QWEN_EXO_TENSOR_BANK_SPAN_TOKENS:=16}"
: "${QWEN_EXO_MAMBA_STRATEGY:=extra_buffer}"
: "${QWEN_EXO_OBSERVER_MODE:=active}"
: "${QWEN_EXO_SURPRISAL_THRESHOLD:=0.8}"
: "${QWEN_EXO_SURPRISAL_WINDOW:=8}"
: "${QWEN_EXO_SURPRISAL_MARGIN:=0.2}"
: "${QWEN_EXO_Q_DRIFT_THRESHOLD:=0.35}"
: "${QWEN_EXO_Q_PRE_TOKENS:=8}"
: "${QWEN_EXO_Q_POST_TOKENS:=4}"
: "${QWEN_EXO_RECOVERY_TOKENS:=8}"
: "${QWEN_EXO_REPLAY_OBSERVATION_TOKENS:=8}"
: "${QWEN_EXO_REPLAY_PREFIX_TOKENS:=1024}"
: "${QWEN_EXO_REPLAY_MAX_CANDIDATES:=2}"
: "${QWEN_EXO_REPLAY_REFERENCE_TOKENS:=128}"
: "${QWEN_EXO_REPLAY_MINIMUM_GAIN:=0.02}"
: "${QWEN_EXO_REPLAY_SWITCH_MARGIN:=0.05}"
: "${QWEN_EXO_REPLAY_MAYBE_KL_CAP:=4.0}"
: "${QWEN_EXO_IMMEDIATE_UNCERTAINTY_RETRIEVAL:=1}"
: "${QWEN_EXO_ENABLE_ADAPTIVE_REFRESH:=1}"
: "${QWEN_EXO_QK_ONLY_KNOWLEDGE:=0}"
: "${QWEN_EXO_QK_EXPANSION_MARGIN:=0.02}"
: "${QWEN_EXO_QK_RECALL_PRESET:=balanced}"
: "${QWEN_EXO_QK_PREFILTER_MODE:=active}"
: "${QWEN_EXO_MOE_TOP_K:=}"
: "${QWEN_EXO_MOE_EXTRA_EXPERTS:=0}"
: "${QWEN_EXO_ENABLE_RETURN_ROUTED_EXPERTS:=0}"
: "${QWEN_EXO_CONSOLE_TRACE_DEFAULT_SCOPE:=activity}"
: "${QWEN_EXO_CONTEXT_EVIDENCE_MODE:=active}"
: "${QWEN_EXO_CONTEXT_INTEGRITY_MODE:=active}"
: "${QWEN_EXO_CONTEXT_INTEGRITY_CONTEXT_DIVISOR:=3}"
: "${QWEN_EXO_REFLECTION_MEMORY_MODE:=active}"
: "${QWEN_EXO_REFLECTION_MEMORY_IDLE_SECONDS:=600}"
: "${QWEN_EXO_REFLECTION_MEMORY_MIN_EVENTS:=3}"
: "${QWEN_EXO_REFLECTION_MEMORY_MIN_TOKENS:=256}"
: "${QWEN_EXO_REFLECTION_MEMORY_MAX_ATTEMPTS:=3}"
: "${QWEN_EXO_REFLECTION_MEMORY_MAX_OUTPUT_TOKENS:=4096}"
: "${QWEN_EXO_REFLECTION_MEMORY_MAX_HISTORY_TOKENS:=92160}"
: "${QWEN_EXO_RESPONSE_COMPACTION_MODE:=active}"
: "${QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB:=512}"
: "${QWEN_EXO_TELEMETRY_INCLUDE_TEXT:=0}"
: "${QWEN_EXO_SCORE_BIAS_MODE:=trajectory_active}"
: "${QWEN_EXO_SCORE_BIAS_MIN_SURPRISAL:=0.8}"
: "${QWEN_EXO_SCORE_BIAS_MAX:=0.25}"
: "${QWEN_EXO_SCORE_BIAS_HALF_LIFE_STEPS:=4.0}"
: "${QWEN_EXO_SCORE_BIAS_MAX_BLOCKS:=8}"
: "${QWEN_EXO_SCORE_BIAS_MIN_AGE_STEPS:=2}"
: "${QWEN_EXO_SCORE_BIAS_MAX_AGE_STEPS:=16}"
: "${QWEN_EXO_SCORE_BIAS_TAIL_TOKENS:=4096}"
: "${QWEN_EXO_SCORE_BIAS_TAIL_RATIO:=0.15}"
: "${QWEN_EXO_SCORE_BIAS_SELECTED_BLOCKS:=2}"

case "${QWEN_EXO_ENABLED}" in
  0|1) ;;
  *)
    echo "Invalid QWEN_EXO_ENABLED=${QWEN_EXO_ENABLED@Q}; expected 0 or 1." >&2
    exit 1
    ;;
esac

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
QWEN_EXO_MODEL_CATALOG_PATH="${QWEN_EXO_MODEL_CATALOG_PATH:-$(dirname -- "${QWEN_EXO_MODEL_PATH}")}"
if [[ ! -d "${QWEN_EXO_MODEL_CATALOG_PATH}" ]]; then
  echo "Model catalog directory not found: ${QWEN_EXO_MODEL_CATALOG_PATH}" >&2
  exit 1
fi
catalog_host_roots=("${QWEN_EXO_MODEL_CATALOG_PATH}")
if [[ -n "${QWEN_EXO_MODEL_CATALOG_EXTRA_PATHS:-}" ]]; then
  IFS=':' read -r -a extra_catalog_roots <<< "${QWEN_EXO_MODEL_CATALOG_EXTRA_PATHS}"
  for root in "${extra_catalog_roots[@]}"; do
    [[ -z "${root}" ]] && continue
    if [[ ! -d "${root}" ]]; then
      echo "Additional model catalog directory not found: ${root}" >&2
      exit 1
    fi
    catalog_host_roots+=("${root}")
  done
fi
catalog_container_roots=()
for index in "${!catalog_host_roots[@]}"; do
  catalog_container_roots+=("/models/catalog-${index}")
done
QWEN_EXO_MODEL_CATALOG_ROOTS="$(IFS=:; echo "${catalog_container_roots[*]}")"
: "${QWEN_EXO_MODEL_CATALOG_CONFIG:=/data/qwen-exo/model-catalog.json}"
: "${QWEN_EXO_MODEL_DATA_ROOT:=/data/qwen-exo}"
: "${QWEN_EXO_PRE_COMPLETE_PATH:=${QWEN_EXO_DATA_PATH}/pre-complete}"
export QWEN_EXO_MODEL_PATH QWEN_EXO_MODEL_CATALOG_PATH QWEN_EXO_DATA_PATH
export QWEN_EXO_MODEL_CATALOG_ROOTS QWEN_EXO_MODEL_CATALOG_CONFIG QWEN_EXO_MODEL_DATA_ROOT
export QWEN_EXO_PRE_COMPLETE_PATH

if [[ ! -f "${QWEN_EXO_SOURCE_PATH}/python/sglang/srt/server_args.py" ]]; then
  echo "QWEN-EXO source tree not found: ${QWEN_EXO_SOURCE_PATH}" >&2
  exit 1
fi

if [[ "${QWEN_EXO_ENABLED}" == "1" ]] && ! python3 \
  "${QWEN_EXO_SOURCE_PATH}/python/qwen_exo_booster/fingerprint.py" \
  "${QWEN_EXO_MODEL_PATH}"; then
  echo "QWEN-EXO startup aborted before Docker launch." >&2
  printf '%s\n' \
    "Directory names and marketing labels are never trusted." \
    "Set QWEN_EXO_MODEL_PATH to a Qwen-series checkpoint with one of the exact" \
    "verified Dense 27B, MoE 35B-A3B, or MoE 122B-A10B Qwen3_5* runtime structures." >&2
  exit 2
fi

active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | sort -u)"
if [[ -n "${active_pids}" ]]; then
  echo "Refusing to start while GPU compute PIDs are active: ${active_pids}" >&2
  exit 1
fi

# Seed only reviewed precompile sources. Existing canonical files are refreshed,
# while unrelated user uploads and nested directories remain untouched. Compiled
# Tensor Bank and Native Bank artifacts are always generated in the runtime data
# root and are never copied from the repository.
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
  "${QWEN_EXO_PRE_COMPLETE_PATH}" \
  "${QWEN_EXO_DATA_PATH}/logs"
seed_corpus \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/knowledge" \
  "${QWEN_EXO_DATA_PATH}/knowledge"
# PolicyData is operator-managed and must remain a single document. Seed it only
# when the persistent lane is empty; never re-expand a live profile on restart.
if ! find "${QWEN_EXO_DATA_PATH}/policydata" -maxdepth 1 -type f \( -name '*.md' -o -name '*.markdown' \) -print -quit | grep -q .; then
  install -m 0644 \
    "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/policydata/coding-agent-execution-policy.md" \
    "${QWEN_EXO_DATA_PATH}/policydata/coding-agent-execution-policy.md"
fi
seed_corpus \
  "${QWEN_EXO_SOURCE_PATH}/scripts/qwen_exo/corpus/cognition" \
  "${QWEN_EXO_DATA_PATH}/cognition"


docker rm -f "${QWEN_EXO_CONTAINER}" >/dev/null 2>&1 || true

docker_args=(
  --restart unless-stopped
  --name "${QWEN_EXO_CONTAINER}"
  --gpus "${QWEN_EXO_DOCKER_GPUS}"
  --ipc=host
  --network=host
  --ulimit memlock=-1
  -e NCCL_P2P_DISABLE=1
  -e NCCL_SHM_DISABLE=0
  -e SGLANG_MAMBA_SSM_DTYPE=bfloat16
  -e "SGLANG_OPT_USE_JIT_PER_TOKEN_GROUP_QUANT=${QWEN_EXO_USE_JIT_FP8_QUANT}"
  -e "SGLANG_LOGPROB_CHUNK_SIZE=${QWEN_EXO_LOGPROB_CHUNK_SIZE}"
  -e "SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB=${QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB}"
  -e QWEN_EXO_SERVICE_CONFIG=/data/qwen-exo/service-config.json
  -e QWEN_EXO_API_KEY_STORE=/data/qwen-exo/api-keys.json
  -e QWEN_EXO_MANAGED_RESTART=1
  -e "QWEN_EXO_DEFAULT_ACTIVATION_EDITOR=${QWEN_EXO_DEFAULT_ACTIVATION_EDITOR:-}"
  -e "QWEN_EXO_DEFAULT_ACTIVATION_EDITOR_STRENGTH=${QWEN_EXO_DEFAULT_ACTIVATION_EDITOR_STRENGTH:-}"
  -e "QWEN_EXO_MODEL_CATALOG_ROOTS=${QWEN_EXO_MODEL_CATALOG_ROOTS}"
  -e "QWEN_EXO_MODEL_CATALOG_CONFIG=${QWEN_EXO_MODEL_CATALOG_CONFIG}"
  -e "QWEN_EXO_MODEL_DATA_ROOT=${QWEN_EXO_MODEL_DATA_ROOT}"
  -e QWEN_EXO_PRE_COMPLETE_KNOWLEDGE_DIR=/data/qwen-exo-pre-complete
  -v "${QWEN_EXO_DATA_PATH}:/data/qwen-exo"
  -v "${QWEN_EXO_PRE_COMPLETE_PATH}:/data/qwen-exo-pre-complete"
  -v "${QWEN_EXO_SOURCE_PATH}/python:/sgl-workspace/sglang/python:ro"
)
for index in "${!catalog_host_roots[@]}"; do
  docker_args+=( -v "${catalog_host_roots[index]}:${catalog_container_roots[index]}:ro" )
done


for debug_env in \
  CUDA_LAUNCH_BLOCKING \
  NCCL_DEBUG \
  NCCL_DEBUG_SUBSYS \
  SGLANG_KERNEL_API_LOGLEVEL \
  SGLANG_KERNEL_API_LOGDEST \
  SGLANG_KERNEL_API_DUMP_DIR \
  SGLANG_KERNEL_API_DUMP_INCLUDE \
  SGLANG_KERNEL_API_DUMP_EXCLUDE; do
  if [[ -n "${!debug_env:-}" ]]; then
    docker_args+=( -e "${debug_env}=${!debug_env}" )
  fi
done
unset debug_env

server_args=(
  --model-path "${catalog_container_roots[0]}/$(basename -- "${QWEN_EXO_MODEL_PATH}")"
  --served-model-name duckgpt
  --tp-size "${QWEN_EXO_TP_SIZE}"
  --dtype "${QWEN_EXO_DTYPE}"
  --kv-cache-dtype "${QWEN_EXO_KV_CACHE_DTYPE}"
  --context-length "${QWEN_EXO_CONTEXT_LENGTH}"
  --mem-fraction-static "${QWEN_EXO_MEM_FRACTION_STATIC}"
  --max-running-requests "${QWEN_EXO_MAX_RUNNING_REQUESTS}"
  --max-prefill-tokens "${QWEN_EXO_MAX_PREFILL_TOKENS}"
  --attention-backend triton
  --sampling-backend pytorch
  --disable-custom-all-reduce
  --cuda-graph-backend-decode full
  --cuda-graph-backend-prefill disabled
  --cuda-graph-max-bs-decode "${QWEN_EXO_CUDA_GRAPH_MAX_BS}"
  --enable-priority-scheduling
  --mamba-radix-cache-strategy "${QWEN_EXO_MAMBA_STRATEGY}"
  --page-size 64
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --default-chat-template-kwargs
  '{"enable_thinking": false, "preserve_thinking": false}'
  --watchdog-timeout 1200
  --host 127.0.0.1
  --port "${QWEN_EXO_PORT}"
)
if [[ -n "${QWEN_EXO_QUANTIZATION}" ]]; then
  server_args+=( --quantization "${QWEN_EXO_QUANTIZATION}" )
fi
if [[ "${QWEN_EXO_ENABLE_RETURN_ROUTED_EXPERTS}" == "1" ]]; then
  server_args+=( --enable-return-routed-experts )
fi
if [[ "${QWEN_EXO_CPU_OFFLOAD_GB}" != "0" ]]; then
  server_args+=( --cpu-offload-gb "${QWEN_EXO_CPU_OFFLOAD_GB}" )
fi
if [[ "${QWEN_EXO_ENABLED}" == "1" ]]; then
  server_args+=(
    --enable-qwen-exo
    --qwen-exo-state-dir "/data/qwen-exo/${QWEN_EXO_STATE_DIRECTORY_NAME}"
    --qwen-exo-api-key-store /data/qwen-exo/api-keys.json
    --qwen-exo-knowledge-dir /data/qwen-exo/knowledge
    --qwen-exo-enable-policy-data
    --qwen-exo-policy-data-dir /data/qwen-exo/policydata
    --qwen-exo-cognition-dir /data/qwen-exo/cognition
    --qwen-exo-max-internal-fanout "${QWEN_EXO_MAX_INTERNAL_FANOUT}"
    --qwen-exo-max-internal-tokens "${QWEN_EXO_MAX_INTERNAL_TOKENS}"
    --qwen-exo-max-output-tokens "${QWEN_EXO_MAX_OUTPUT_TOKENS}"
    --qwen-exo-max-reasoning-tokens "${QWEN_EXO_MAX_REASONING_TOKENS}"
    --qwen-exo-tensor-bank-max-document-tokens "${QWEN_EXO_TENSOR_BANK_MAX_DOCUMENT_TOKENS}"
    --qwen-exo-tensor-bank-salient-token-budget "${QWEN_EXO_TENSOR_BANK_SALIENT_TOKEN_BUDGET}"
    --qwen-exo-tensor-bank-surprisal-threshold "${QWEN_EXO_TENSOR_BANK_SURPRISAL_THRESHOLD}"
    --qwen-exo-tensor-bank-span-tokens "${QWEN_EXO_TENSOR_BANK_SPAN_TOKENS}"
    --qwen-exo-observer-mode "${QWEN_EXO_OBSERVER_MODE}"
    --qwen-exo-context-evidence-mode "${QWEN_EXO_CONTEXT_EVIDENCE_MODE}"
    --qwen-exo-observer-surprisal-threshold "${QWEN_EXO_SURPRISAL_THRESHOLD}"
    --qwen-exo-observer-surprisal-window "${QWEN_EXO_SURPRISAL_WINDOW}"
    --qwen-exo-observer-surprisal-margin "${QWEN_EXO_SURPRISAL_MARGIN}"
    --qwen-exo-observer-q-drift-threshold "${QWEN_EXO_Q_DRIFT_THRESHOLD}"
    --qwen-exo-observer-q-pre-tokens "${QWEN_EXO_Q_PRE_TOKENS}"
    --qwen-exo-observer-q-post-tokens "${QWEN_EXO_Q_POST_TOKENS}"
    --qwen-exo-observer-recovery-tokens "${QWEN_EXO_RECOVERY_TOKENS}"
    --qwen-exo-replay-observation-tokens "${QWEN_EXO_REPLAY_OBSERVATION_TOKENS}"
    --qwen-exo-replay-prefix-tokens "${QWEN_EXO_REPLAY_PREFIX_TOKENS}"
    --qwen-exo-replay-max-candidates "${QWEN_EXO_REPLAY_MAX_CANDIDATES}"
    --qwen-exo-replay-reference-tokens "${QWEN_EXO_REPLAY_REFERENCE_TOKENS}"
    --qwen-exo-replay-minimum-gain "${QWEN_EXO_REPLAY_MINIMUM_GAIN}"
    --qwen-exo-replay-switch-margin "${QWEN_EXO_REPLAY_SWITCH_MARGIN}"
    --qwen-exo-replay-maybe-kl-cap "${QWEN_EXO_REPLAY_MAYBE_KL_CAP}"
    --qwen-exo-qk-expansion-margin "${QWEN_EXO_QK_EXPANSION_MARGIN}"
    --qwen-exo-qk-recall-preset "${QWEN_EXO_QK_RECALL_PRESET}"
    --qwen-exo-console-trace-default-scope "${QWEN_EXO_CONSOLE_TRACE_DEFAULT_SCOPE}"
    --qwen-exo-qk-prefilter-mode "${QWEN_EXO_QK_PREFILTER_MODE}"
    --qwen-exo-context-integrity-mode "${QWEN_EXO_CONTEXT_INTEGRITY_MODE}"
    --qwen-exo-context-integrity-context-divisor "${QWEN_EXO_CONTEXT_INTEGRITY_CONTEXT_DIVISOR}"
    --qwen-exo-reflection-memory-mode "${QWEN_EXO_REFLECTION_MEMORY_MODE}"
    --qwen-exo-reflection-memory-idle-seconds "${QWEN_EXO_REFLECTION_MEMORY_IDLE_SECONDS}"
    --qwen-exo-reflection-memory-min-events "${QWEN_EXO_REFLECTION_MEMORY_MIN_EVENTS}"
    --qwen-exo-reflection-memory-min-tokens "${QWEN_EXO_REFLECTION_MEMORY_MIN_TOKENS}"
    --qwen-exo-reflection-memory-max-attempts "${QWEN_EXO_REFLECTION_MEMORY_MAX_ATTEMPTS}"
    --qwen-exo-reflection-memory-max-output-tokens "${QWEN_EXO_REFLECTION_MEMORY_MAX_OUTPUT_TOKENS}"
    --qwen-exo-reflection-memory-max-history-tokens "${QWEN_EXO_REFLECTION_MEMORY_MAX_HISTORY_TOKENS}"
    --qwen-exo-response-compaction-mode "${QWEN_EXO_RESPONSE_COMPACTION_MODE}"
  )
  if [[ -n "${QWEN_EXO_MOE_TOP_K}" ]]; then
    server_args+=( --qwen-exo-moe-top-k "${QWEN_EXO_MOE_TOP_K}" )
  fi
  if [[ "${QWEN_EXO_MOE_EXTRA_EXPERTS}" != "0" ]]; then
    server_args+=( --qwen-exo-moe-extra-experts "${QWEN_EXO_MOE_EXTRA_EXPERTS}" )
  fi
  if [[ "${QWEN_EXO_IMMEDIATE_UNCERTAINTY_RETRIEVAL}" == "1" ]]; then
    server_args+=( --qwen-exo-immediate-uncertainty-retrieval )
  fi
  if [[ "${QWEN_EXO_TELEMETRY_INCLUDE_TEXT}" == "1" ]]; then
    server_args+=( --qwen-exo-telemetry-include-text )
  fi
  if [[ "${QWEN_EXO_ENABLE_ADAPTIVE_REFRESH}" == "1" ]]; then
    if [[ "${QWEN_EXO_OBSERVER_MODE}" != "active" ]]; then
      echo "Adaptive refresh requires QWEN_EXO_OBSERVER_MODE=active" >&2
      exit 1
    fi
    server_args+=( --qwen-exo-enable-adaptive-refresh )
  fi
  if [[ "${QWEN_EXO_QK_ONLY_KNOWLEDGE}" == "1" ]]; then
    server_args+=( --qwen-exo-qk-only-knowledge )
  fi
  if [[ "${QWEN_EXO_SCORE_BIAS_MODE}" != "off" ]]; then
    server_args+=(
      --qwen-exo-score-bias-mode "${QWEN_EXO_SCORE_BIAS_MODE}"
      --qwen-exo-score-bias-min-surprisal "${QWEN_EXO_SCORE_BIAS_MIN_SURPRISAL}"
      --qwen-exo-score-bias-max "${QWEN_EXO_SCORE_BIAS_MAX}"
      --qwen-exo-score-bias-half-life-steps "${QWEN_EXO_SCORE_BIAS_HALF_LIFE_STEPS}"
      --qwen-exo-score-bias-max-blocks "${QWEN_EXO_SCORE_BIAS_MAX_BLOCKS}"
      --qwen-exo-score-bias-min-age-steps "${QWEN_EXO_SCORE_BIAS_MIN_AGE_STEPS}"
      --qwen-exo-score-bias-max-age-steps "${QWEN_EXO_SCORE_BIAS_MAX_AGE_STEPS}"
      --qwen-exo-score-bias-tail-tokens "${QWEN_EXO_SCORE_BIAS_TAIL_TOKENS}"
      --qwen-exo-score-bias-tail-ratio "${QWEN_EXO_SCORE_BIAS_TAIL_RATIO}"
      --qwen-exo-score-bias-selected-blocks "${QWEN_EXO_SCORE_BIAS_SELECTED_BLOCKS}"
    )
  fi
fi

exec docker run "${docker_args[@]}" \
  "${QWEN_EXO_IMAGE}" \
  python3 -m qwen_exo_booster.service_launcher -- "${server_args[@]}"
