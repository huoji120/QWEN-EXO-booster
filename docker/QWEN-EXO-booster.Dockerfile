ARG SGLANG_BASE_IMAGE=lmsysorg/sglang@sha256:435dd550e0b891a6d624ec124b577a1a8eadea13c4ebfa47ea07717e522ca72b
FROM ${SGLANG_BASE_IMAGE}

ARG QWEN_EXO_REVISION=local
LABEL org.opencontainers.image.title="QWEN-EXO-booster" \
      org.opencontainers.image.description="SGLang Qwen3.5 hybrid-memory inference runtime" \
      org.opencontainers.image.revision="${QWEN_EXO_REVISION}" \
      ai.qwen-exo.upstream.commit="fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1" \
      ai.qwen-exo.cuda.family="12.9"

ENV QWEN_EXO_REVISION=${QWEN_EXO_REVISION} \
    SGLANG_MAMBA_SSM_DTYPE=bfloat16

# The release image already has an editable install rooted at this workspace.
# Overlaying the pinned fork preserves its compiled CUDA/Rust artifacts while
# replacing the Python source with the reviewed QWEN-EXO fork.
WORKDIR /sgl-workspace/sglang
COPY . /sgl-workspace/sglang

RUN python3 -m compileall -q \
      python/qwen_exo_booster \
      python/sglang/srt/server_args.py \
      python/sglang/srt/entrypoints/http_server.py

CMD ["/bin/bash"]
