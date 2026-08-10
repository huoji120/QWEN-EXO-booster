#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

: "${QWEN_EXO_IMAGE:=qwen-exo-booster:sglang-v0.5.16-driver550}"
: "${QWEN_EXO_BASE_IMAGE:=lmsysorg/sglang@sha256:30d09acc893b5647ea69fb63d5b30302e3f2199ac57c42d2e5c784cb6f2efdaf}"

revision="$(git rev-parse HEAD)"
docker build \
  --file docker/QWEN-EXO-booster.Dockerfile \
  --build-arg "SGLANG_BASE_IMAGE=${QWEN_EXO_BASE_IMAGE}" \
  --build-arg "QWEN_EXO_REVISION=${revision}" \
  --tag "${QWEN_EXO_IMAGE}" \
  .

docker run --rm --gpus all \
  "${QWEN_EXO_IMAGE}" \
  python3 /sgl-workspace/sglang/scripts/qwen_exo/check_cuda.py

docker run --rm --gpus all \
  "${QWEN_EXO_IMAGE}" \
  python3 /sgl-workspace/sglang/scripts/qwen_exo/check_imports.py

docker run --rm --gpus all \
  "${QWEN_EXO_IMAGE}" \
  python3 /sgl-workspace/sglang/scripts/qwen_exo/check_kernels.py
