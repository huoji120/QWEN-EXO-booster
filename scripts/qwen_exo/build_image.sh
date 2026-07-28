#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}"

: "${QWEN_EXO_IMAGE:=qwen-exo-booster:sglang-v0.5.16-cu129}"
: "${QWEN_EXO_BASE_IMAGE:=lmsysorg/sglang@sha256:435dd550e0b891a6d624ec124b577a1a8eadea13c4ebfa47ea07717e522ca72b}"

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
