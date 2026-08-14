# Apple Silicon MLX deployment

This profile runs QWEN-EXO directly on an Apple Silicon GPU through SGLang's
native MLX model runner. It does not use Docker, CUDA, NCCL, or tensor
parallelism. The HTTP API, scheduler-native internal jobs, Tensor Bank,
Attention-Q signals, Observer, Score Bias, and the control console stay on the
same QWEN-EXO code path as the CUDA profile.

## Reviewed boundary

The MLX implementation is native rather than a PyTorch-MPS fallback:

- `MlxModelRunner` loads the checkpoint with `mlx-lm`;
- Full-Attention KV is owned by the MLX attention cache pool;
- Gated DeltaNet cache trees are snapshotted as MLX auxiliary state;
- native Bank artifacts store complete Full-Attention K/V and auxiliary state;
- Q/K observation, Score Bias, latent controls, and activation editors execute
  with `mlx.core` arrays;
- model-native artifacts use an `mlx-tp1-*` topology namespace and cannot be
  reused by the CUDA profile.

The checkpoint compatibility gate is unchanged. Only the exact verified Dense
27B and MoE 35B-A3B Qwen3.5 hybrid structures are accepted. A generic
`mlx-community` model is not accepted merely because MLX can load it.

## Requirements

- Apple Silicon Mac (`arm64`), macOS;
- Python 3.11 or newer;
- local Qwen checkpoint in Hugging Face layout, including `config.json` and
  `model.safetensors.index.json`;
- enough unified memory for weights, the selected context, KV, GDN state, and
  temporary activations.

The installer pins `mlx>=0.31.2,<0.33` and `mlx-lm>=0.31.2,<0.32`. The lower
bound includes Qwen3.5 Dense/MoE support and the Qwen3.5 cache-advance fix. The
launcher defaults to 4-bit load-time quantization and retains the release
baseline of a 102,400-token context. This preserves the original parameter
baseline; memory-constrained Macs should explicitly select a smaller context,
and the default is not a claim that every Mac can serve it under every workload.

## Install

From the repository root:

```bash
bash scripts/qwen_exo/install_mlx.sh
```

The script creates `.venv`, installs the MPS/MLX dependency set, installs this
checkout in editable mode without CUDA or Rust extensions, and runs a real
Metal matrix operation. It also verifies imports for both
`mlx_lm.models.qwen3_5` and `mlx_lm.models.qwen3_5_moe`.

To use another environment:

```bash
QWEN_EXO_VENV=/path/to/venv \
QWEN_EXO_PYTHON=python3.12 \
bash scripts/qwen_exo/install_mlx.sh
```

## Launch

Keep runtime state outside the checkout:

```bash
export QWEN_EXO_MODEL_PATH=/path/to/Qwen3.5-27B
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime

bash scripts/qwen_exo/launch_mlx.sh
```

The launcher performs the MLX/Metal and checkpoint-structure preflights before
loading model weights. Its QWEN-EXO behavioral defaults retain the release
baseline (`context_length=102400`, `max_prefill_tokens=65536`,
`max_running_requests=64`, and `mem_fraction_static=0.80`). The important
fixed backend parameters are:

```text
SGLANG_USE_MLX=1
device=mps
tp_size=1
dtype=float16
quantization=mlx_q4
kv_cache_dtype=mxfp8
page_size=1
mamba_radix_cache_strategy=no_buffer
overlap_schedule=disabled
cuda_graph=disabled
```

Useful resource overrides:

```bash
export QWEN_EXO_CONTEXT_LENGTH=16384
export QWEN_EXO_MAX_PREFILL_TOKENS=4096
export QWEN_EXO_MAX_RUNNING_REQUESTS=4
export QWEN_EXO_MAX_TOTAL_TOKENS=65536
export QWEN_EXO_MEM_FRACTION_STATIC=0.65
export QWEN_EXO_QUANTIZATION=mlx_q4  # none, mlx_q4, mlx_q8, or mlx_mxfp8
export QWEN_EXO_KV_CACHE_DTYPE=mxfp8 # auto, bf16, bfloat16, or mxfp8
export SGLANG_MLX_CLEAR_CACHE_STEPS=1
export SGLANG_MLX_CACHE_LIMIT_GIB=2

bash scripts/qwen_exo/launch_mlx.sh
```

When using a pre-quantized MLX checkpoint, keep `mlx_q4` if its bit width is
compatible or set `QWEN_EXO_QUANTIZATION=none`. `mlx-lm` detects quantization
metadata in the checkpoint and does not quantize it a second time.

For a true floating-point 8-bit checkpoint, use an MLX `mxfp8` model and set
`QWEN_EXO_QUANTIZATION=mlx_mxfp8`. This is E4M3 weight storage with an E8M0
shared scale per 32-value group; it is distinct from affine integer
`mlx_q8`. MXFP8 uses roughly twice the weight memory of `mlx_q4`.

The MLX launcher defaults the attention KV pool to `mxfp8` independently of
weight quantization. This keeps Q4 weights while storing KV as E4M3 values
with E8M0 group scales. It preserves `QWEN_EXO_CONTEXT_LENGTH` and
`QWEN_EXO_MAX_TOTAL_TOKENS`; the tradeoff is an extra dequantization step at
attention time.

Unless overridden, `QWEN_EXO_MAX_TOTAL_TOKENS` equals the context length. MLX
uses one shared unified-memory KV budget, so the concurrency value remains an
admission cap rather than multiplying the persistent KV allocation by 64.

## Verify

```bash
curl -f http://127.0.0.1:30000/qwen-exo/health
curl -s http://127.0.0.1:30000/qwen-exo/status
curl -s http://127.0.0.1:30000/v1/models
```

`/qwen-exo/status` reports the backend and topology, for example:

```json
{
  "hybrid_state": {
    "backend": "mlx",
    "topology_key": "mlx-tp1-float16-mlx-q4-auto"
  }
}
```

The console remains loopback-only at `http://127.0.0.1:30000/qwen-exo/`.

Run the backend-independent and MLX policy regressions with:

```bash
PYTHONPATH=python .venv/bin/python -m pytest \
  test/registered/qwen_exo/test_mlx_preflight.py \
  test/registered/qwen_exo/test_mlx_launcher.py \
  test/registered/qwen_exo/test_hybrid_state.py \
  test/registered/qwen_exo/test_config_runtime.py -q
```

## Current verification status

On 2026-08-12, the environment and a real Dense 27B service were verified on an
Apple M5 Max with 128 GiB unified memory using MLX 0.32.0, MLX-LM 0.31.3, and
PyTorch 2.11.0. The pre-quantized `mlx-community/Qwen3.5-27B-4bit` checkpoint
loaded through the native MLX runner in 1.31 seconds and occupied 14.09 GB of
MLX weight memory. The final launch allocated 262,144 KV tokens (about 16 GiB)
for eight running requests at a 32,768-token context.

The loopback service reached `ready`, loaded 10 Knowledge documents and one
PolicyData document, and exposed topology `mlx-tp1-float16-mlx-q4-auto`. A real
OpenAI Responses request completed with HTTP 200 and restored 1,856 cached
native-policy prefix tokens. A direct generation returned `MLX_READY` in 9
output tokens and 0.729 seconds end to end. The complete QWEN-EXO suite passed
503 tests with 2 platform-conditioned skips while the service remained ready.

This evidence verifies the Dense 27B model-load, cache restoration, scheduler,
and served-generation path on this host. It does not establish MoE 35B-A3B
execution, cross-backend numerical equivalence, or the worst-case memory margin
for eight simultaneous 32K requests.

## Failure boundaries

- A CUDA Tensor Bank cannot be restored by MLX, and an MLX Bank cannot be
  restored by CUDA. Recompile after switching backend, dtype, or quantization.
- MLX uses one process and `tp_size=1`; the CUDA `tp_size=2` performance and
  capacity evidence does not transfer to Mac.
- `page_size=1` and `no_buffer` are correctness requirements for the current
  MLX auxiliary-state radix implementation.
- Reduce context, prefill budget, and concurrency first if unified-memory
  pressure appears. A process that starts successfully has not yet proven the
  requested worst-case context and concurrency combination.
