ARG SGLANG_BASE_IMAGE=lmsysorg/sglang@sha256:30d09acc893b5647ea69fb63d5b30302e3f2199ac57c42d2e5c784cb6f2efdaf
FROM ${SGLANG_BASE_IMAGE}

ARG QWEN_EXO_REVISION=local
LABEL org.opencontainers.image.title="QWEN-EXO-booster" \
      org.opencontainers.image.description="SGLang Qwen hybrid-memory inference runtime" \
      org.opencontainers.image.revision="${QWEN_EXO_REVISION}" \
      ai.qwen-exo.upstream.commit="fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1" \
      ai.qwen-exo.cuda.family="12.6-mvc-driver550"

ARG QWEN_EXO_TRITON_VERSION=3.6.0

ENV QWEN_EXO_REVISION=${QWEN_EXO_REVISION} \
    SGLANG_MAMBA_SSM_DTYPE=bfloat16 \
    PYTHONPATH=/sgl-workspace/sglang/python

# The pinned image supplies CUDA 12.6/Torch 2.7 binary artifacts. The reviewed
# fork is selected explicitly through PYTHONPATH; the image's installed SGLang
# package must never shadow this source overlay.
WORKDIR /sgl-workspace/sglang
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      pybase64==1.4.3 \
      transformers==5.12.1 \
      xgrammar==0.2.1 \
      gguf==0.19.0 \
      torch_memory_saver==0.0.9.post1 \
      openai==2.6.1 \
      openai-harmony==0.0.4 && \
    python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      flashinfer-python==0.6.14 \
      sglang-kernel==0.4.5

# Current SGLang FLA kernels use tl.extra.cuda.gdc_wait, which is absent from
# the base image's Triton 3.3.1. This combination is GPU-smoke-tested on 550.78.
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
      "triton==${QWEN_EXO_TRITON_VERSION}" \
      bitsandbytes==0.50.0 \
      accelerate==1.14.0

COPY . /sgl-workspace/sglang

RUN python3 -m compileall -q \
      python/qwen_exo_booster \
      python/sglang/srt/server_args.py \
      python/sglang/srt/entrypoints/http_server.py \
      python/sglang/srt/entrypoints/openai/serving_responses.py \
      python/sglang/srt/managers/scheduler.py \
      python/sglang/srt/managers/schedule_batch.py \
      python/sglang/srt/managers/tokenizer_manager.py \
      python/sglang/srt/managers/scheduler_components/batch_result_processor.py \
      python/sglang/srt/model_executor/forward_batch_info.py \
      python/sglang/srt/distributed/device_communicators/pynccl_allocator.py \
      python/sglang/srt/layers/attention/vision.py \
      python/sglang/srt/models/qwen3_5.py \
      python/sglang/srt/models/qwen3_vl.py

RUN python3 scripts/qwen_exo/check_imports.py --basic

CMD ["/bin/bash"]
