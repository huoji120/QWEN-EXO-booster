# QWEN-EXO-booster

<p align="center">
  <a href="README.md">简体中文</a> · <strong>English</strong>
</p>

QWEN-EXO-booster is a **second-stage SGLang development project** for Qwen hybrid-attention inference. It targets Qwen3.5-style Hybrid Full-Attention / Gated DeltaNet models and extends the OpenAI-compatible Responses API with model-native state, knowledge recall, long-context execution, tool calls, and runtime observability in one inference pipeline.

> This is not a conventional RAG demo or a script that merely splits model layers across GPUs. The default deployment uses real SGLang Tensor Parallelism (`TP=2`) and maintains a consistent lifecycle across Full-Attention K/V and Gated DeltaNet recurrent/conv state.

## One-sentence installation

Give this repository README to an LLM and ask it to run the following instruction on a Linux host with **Docker, NVIDIA Container Toolkit, two RTX 4090 GPUs, and NVIDIA driver 550+**:

```text
Read README_EN.md and docs/qwen_exo/SERVER_27B_DEPLOYMENT.md. Verify that QWEN_EXO_MODEL_PATH points to a compatible Qwen Hybrid checkpoint and QWEN_EXO_DATA_PATH points to a separate runtime-data directory. Then run bash scripts/qwen_exo/build_image.sh followed by bash scripts/qwen_exo/launch_js4090.sh. Wait until http://127.0.0.1:30000/qwen-exo/health reports runtime_state=ready, and access the console through an SSH tunnel.
```

This is not an unconditional cloud one-click installer. Model weights, GPUs, the NVIDIA driver, and Docker must already be available. The launcher validates the checkpoint structure before occupying the GPUs and rejects unsupported models.

Apple Silicon Macs use the repository's native MLX execution path and do not
require Docker or CUDA:

```bash
bash scripts/qwen_exo/install_mlx.sh
export QWEN_EXO_MODEL_PATH=/path/to/Qwen3.5-27B
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime
bash scripts/qwen_exo/launch_mlx.sh
```

See [Apple Silicon MLX deployment](docs/qwen_exo/APPLE_SILICON_MLX_DEPLOYMENT.md)
for dependencies, fixed backend parameters, and verification boundaries.

## What this project adds

Upstream SGLang already provides high-performance inference, continuous batching, Radix Cache, Tensor Parallelism, and OpenAI-compatible APIs. QWEN-EXO adds capabilities for Qwen Hybrid models and long-running agents:

- **Long-context inference**: the default target is 102,400 tokens.
- **Hybrid-state restoration**: Full-Attention KV and Gated DeltaNet recurrent/conv state are managed together, preventing KV-only restoration from losing linear-attention state.
- **Model-native knowledge recall**: Markdown knowledge is compiled into Tensor Bank K/V plus complete GDN state. Eligible requests can restore native state instead of concatenating every document into the prompt.
- **Attention-Q × Tensor-Bank-K retrieval**: candidate ranking uses the model's final Full-Attention query heads and persisted Bank K heads. Knowledge and PolicyData use physically separate lanes.
- **Semantic Judge as the final gate**: Q×K only generates candidates. Every candidate must pass the constrained Reference Judge; invalid, stale, rejected, or unjudged candidates fail closed.
- **In-Flight Observer**: selected-token surprisal, Q signals, and local uncertainty are monitored during decode to trigger Self-Ask and bounded refresh when needed.
- **Causal Replay / Maybe gate**: baseline and candidate branches are scored against the same future tokens. NLL gain, switching margin, and KL decide whether a candidate may be scheduled for next-turn restoration.
- **Execution Capsule**: coarse execution state can be retained across turns for long-running tasks.
- **Tool calls and Responses semantics**: Qwen thinking, structured tool calls, SSE/Responses events, and cancellation propagation remain supported.
- **Unified resource admission**: KV tokens, request slots, Mamba slots, and temporary logprob workspace are estimated before scheduler-native work, with atomic admission across TP ranks.
- **Operations console**: health, Q×K candidates, Judge decisions, native restore, Self-Ask, Causal Replay, Maybe, and raw telemetry are visible from one console.

## Core technologies

| Technology | Purpose |
|---|---|
| **SGLang** | Inference scheduling, continuous batching, Radix Cache, scheduler-native internal jobs, and OpenAI-compatible HTTP serving |
| **PyTorch / CUDA / NCCL** | TP=2 execution, GPU state, cross-rank communication, and model forward passes |
| **MLX / Metal / MLX-LM** | Single-process native execution, Full-Attention KV, and GDN auxiliary state on Apple Silicon |
| **Qwen Hybrid Attention** | Full-Attention layers provide KV; Gated DeltaNet layers provide recurrent/conv state |
| **Tensor Parallelism** | True two-GPU model parallelism through `--tp-size 2` |
| **FP8 KV Cache / BF16 State** | Reduces KV memory while retaining the reviewed BF16 state baseline |
| **Tensor Bank** | Persists document-native K/V, salient token positions, and complete GDN state |
| **Raw Attention-Q × K** | Performs retrieval using model-native attention signals without an external embedding service |
| **16-token local windows** | Requires support inside a contiguous local window to reduce isolated-token extrema |
| **Median/MAD relative evidence** | Records robust per-query background, relative scores, and margins as shadow audit evidence |
| **Reference Judge** | Produces the final semantic-admission decision through constrained JSON output |
| **Scheduler-native internal jobs** | Judge, Self-Ask, Self-Answer, Capsule, and Replay do not recursively call HTTP |
| **Observer / Self-Ask** | Detects persistent uncertainty during decode and creates an internal question |
| **Causal Replay / Maybe** | Compares baseline and candidate branches on shared future tokens; fails closed and never rewrites emitted tokens |
| **FastAPI / Uvicorn** | QWEN-EXO control plane, health, knowledge management, and telemetry APIs |
| **React / Vite / Tailwind** | Operations console and recall-trace visualization |
| **Docker** | Pins the reviewed SGLang/CUDA/Torch runtime baseline |

## Requirements

Default verified profile:

- Linux x86_64
- Docker and NVIDIA Container Toolkit
- 2 × NVIDIA RTX 4090, approximately 48 GiB per GPU
- NVIDIA driver 550.78 or a deployment-validated compatible version
- CUDA 12.6 base image
- Qwen Hybrid Dense 27B or MoE 35B-A3B structure
- Runtime service at `127.0.0.1:30000`
- Served model ID: `duckgpt`

Compatibility is determined from the checkpoint's `config.json`, not its directory name, marketing version, or container alias. Startup rejects unsupported structures.

An Apple Silicon MLX profile is also available: macOS arm64, `tp_size=1`,
`page_size=1`, and the `no_buffer` GDN state-cache strategy. MLX and CUDA model
artifacts use different topology namespaces and cannot be cross-restored. The
MLX launcher keeps the release 102400-token QWEN-EXO baseline while using MLX
Q4 weights and an MXFP8 KV cache; this is not a capacity guarantee for every
Mac.

## Installation and startup

### 1. Clone the repository

```bash
git clone https://github.com/huoji120/QWEN-EXO-booster.git
cd QWEN-EXO-booster
```

If you use an internal remote or already have a checkout, enter that repository instead.

### 2. Set model and runtime-data paths

Runtime data must live outside the checkout. Do not place weights, Tensor Bank artifacts, telemetry, request traces, or training outputs inside the Git worktree.

```bash
export QWEN_EXO_MODEL_PATH=/data/models/Qwen3.5-27B
export QWEN_EXO_DATA_PATH=/data/qwen-exo-runtime
export QWEN_EXO_IMAGE=qwen-exo-booster:sglang-v0.5.16-driver550

mkdir -p "$QWEN_EXO_DATA_PATH"
```

### 3. Build the image and run preflight checks

```bash
bash scripts/qwen_exo/build_image.sh
```

The build script performs:

1. CUDA and GPU checks;
2. SGLang and QWEN-EXO import checks;
3. GPU kernel checks;
4. Docker image creation using the pinned base image and current Git revision.

### 4. Launch the service

```bash
bash scripts/qwen_exo/launch_js4090.sh
```

Key defaults:

```text
TP=2
weights dtype=BF16
quantization=FP8
KV cache=FP8 E4M3
context length=102400
page size=64
observer=active
adaptive refresh=enabled
```

The launcher checks for active GPU compute PIDs before Docker startup. Do not run two inference backends on the same GPUs.

### 5. Wait for readiness

```bash
curl -f http://127.0.0.1:30000/qwen-exo/health
curl -s http://127.0.0.1:30000/qwen-exo/status
curl -s http://127.0.0.1:30000/v1/models
```

The service is ready only when `/qwen-exo/health` returns:

```json
{
  "status": "ok",
  "runtime_state": "ready"
}
```

## Accessing the console

The control plane is bound to `127.0.0.1` by default and must not be exposed directly to the public Internet. Use SSH local port forwarding for remote access.

### Verify the service on the GPU host

```bash
ssh <gpu-host> 'curl -f http://127.0.0.1:30000/qwen-exo/health'
```

### Create the SSH tunnel locally

```bash
ssh -N -L 30000:127.0.0.1:30000 <gpu-user>@<gpu-host>
```

Keep that terminal open and visit:

```text
http://127.0.0.1:30000/qwen-exo/
```

### Console routes

- `/qwen-exo/`: user workspace, chat, and standard operations;
- `/qwen-exo/admin`: operations console;
- **Recall Trace** in the console: request candidates, Q×K, Semantic Judge, native restore, Self-Ask, Causal Replay, Maybe, and raw events;
- `/qwen-exo/recall-trace`: compatible recall-trace route.

The console can inspect and manage:

- Knowledge Markdown;
- PolicyData;
- Tensor Bank compilation state;
- service configuration and healthy revision state;
- request telemetry and Recall Trace;
- Reflection Memory jobs;
- Observer, Adaptive Refresh, Score Bias, and Causal Replay state.

Knowledge and PolicyData mutations are loopback control-plane operations. Access them only through an SSH tunnel or trusted operator-controlled reverse proxy.

## API quick start

### Responses inference

```bash
curl --no-buffer http://127.0.0.1:30000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "duckgpt",
    "input": "Explain how this service restores hybrid attention state.",
    "stream": true,
    "max_output_tokens": 256
  }'
```

### Knowledge metadata

```bash
curl http://127.0.0.1:30000/qwen-exo/knowledge
```

### Recall Trace

```bash
curl 'http://127.0.0.1:30000/qwen-exo/recall-trace?limit=10'
```

### Telemetry

```bash
curl 'http://127.0.0.1:30000/qwen-exo/telemetry?limit=100'
```

Telemetry is redacted by default: prompts, outputs, reasoning, tool arguments, references, and secrets are not written verbatim. See [API, telemetry, security, and console](docs/qwen_exo/API.md) for the full contract.

## Local verification

Run Python regression tests without loading the production model:

```bash
PYTHONPATH=python python -m pytest test/registered/qwen_exo -q
```

Build the console:

```bash
cd frontend/qwen-exo
npm ci
npm run build
```

GPU deployment checks:

```bash
python3 scripts/qwen_exo/check_cuda.py
python3 scripts/qwen_exo/check_imports.py
python3 scripts/qwen_exo/check_kernels.py
python3 scripts/qwen_exo/smoke_contracts.py
```

Apple Silicon MLX checks:

```bash
.venv/bin/python scripts/qwen_exo/check_mlx.py
PYTHONPATH=python .venv/bin/python -m pytest \
  test/registered/qwen_exo/test_mlx_preflight.py \
  test/registered/qwen_exo/test_mlx_launcher.py \
  test/registered/qwen_exo/test_hybrid_state.py \
  test/registered/qwen_exo/test_config_runtime.py -q
```

## Repository layout

```text
python/qwen_exo_booster/       QWEN-EXO runtime, Memory Pipeline, Judge, Observer, APIs
python/sglang/                 SGLang fork and model/scheduler integration
scripts/qwen_exo/              Build, launch, preflight, smoke, evaluation, and Bank tools
docker/                        QWEN-EXO Dockerfile and deployment configuration
frontend/qwen-exo/             React/Vite operations console
docs/qwen_exo/                 Architecture, API, deployment, and verification documents
test/registered/qwen_exo/      Registered regression tests
scripts/qwen_exo/corpus/       Versioned Knowledge, PolicyData, and optional Cognition sources
```

## Security boundaries

- Never commit model weights, Tensor Bank artifacts, runtime telemetry, request traces, training data, or editor weights.
- The control plane is loopback-only by default. Do not expose write routes such as `/qwen-exo/knowledge` or `/qwen-exo/policydata` directly to the Internet.
- `causal_replay` compares candidate branches and never rewrites already emitted tokens.
- Judge, native-state binding, and resource admission all fail closed.
- The project does not call an external LLM and performs no implicit external learning.
- Do not infer model compatibility from a checkpoint directory name.
- Do not run the legacy backend and QWEN-EXO on the same GPUs simultaneously.

## Further reading

1. [Architecture and state contracts](docs/qwen_exo/ARCHITECTURE.md)
2. [Dual RTX 4090 deployment](docs/qwen_exo/SERVER_27B_DEPLOYMENT.md)
3. [Apple Silicon MLX deployment](docs/qwen_exo/APPLE_SILICON_MLX_DEPLOYMENT.md)
4. [API, telemetry, security, and console](docs/qwen_exo/API.md)
5. [Implementation progress and verification evidence](docs/qwen_exo/IMPLEMENTATION_PROGRESS.md)
6. [Demo-to-runtime migration matrix](docs/qwen_exo/DEMO_MIGRATION_MATRIX.md)

## Project scope

QWEN-EXO-booster is an SGLang-based model-runtime development project focused on **model-native state management for Qwen Hybrid models, long-context memory recall, internal task scheduling, agent continuity, and verifiable operations**. Any claim of better model capability or accuracy must be supported by controlled evaluation with fixed model, context, concurrency, and output length; a single demonstration is not sufficient evidence.
