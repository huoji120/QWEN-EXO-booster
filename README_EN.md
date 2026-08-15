# QWEN-EXO-booster

<p align="center">
  <a href="README.md">简体中文</a> · <strong>English</strong>
</p>

![img](/banner.png)

A Qwen hybrid-attention inference backend built as an **SGLang fork**, designed to substantially extend the capabilities of Qwen-family models.

> Supports macOS and Linux. SGLang does not support native Windows deployment; use WSL on Windows. Supports Qwen3.5 through Qwen3.8, including derivative checkpoints and MoE models. We recommend Qwen3.8-27B.

## Why QWEN-EXO

### Model-native knowledge injection

Unlike RAG, QWEN-EXO provides an attention-based knowledge-recall mechanism that lets the model retrieve relevant knowledge without depending on a conventional RAG architecture. This substantially expands the knowledge available to Qwen-family models.

Adding knowledge is simple and does not require fine-tuning: write knowledge documents and ingest them into the knowledge base.

![img](/images/1.png)

### Reflection Memory

QWEN-EXO includes server-side Reflection Memory. Successful and failed task trajectories can be reviewed and distilled into reusable memories, allowing the system to improve from prior execution evidence.

![img](/images/2.png)

### Observability

You can inspect the knowledge recalled and injected for each request, helping detect irrelevant recall or behavioral drift.

![img](/images/3.png)

## One-sentence installation

Give this repository README to an LLM and ask it to execute the following instruction on a Linux host with **Docker, NVIDIA Container Toolkit, two RTX 4090 GPUs, and NVIDIA driver 550+**:

```text
Read README_EN.md and docs/qwen_exo/SERVER_27B_DEPLOYMENT.md. Verify that QWEN_EXO_MODEL_PATH points to a compatible Qwen Hybrid checkpoint and QWEN_EXO_DATA_PATH points to a separate runtime-data directory. Then run bash scripts/qwen_exo/build_image.sh followed by bash scripts/qwen_exo/launch_js4090.sh. Wait until http://127.0.0.1:30000/qwen-exo/health reports runtime_state=ready, and access the console through an SSH tunnel.
```

## Installation and startup

### Open any agent terminal

Enter: `Help me deploy this QWEN-EXO SGLang fork:`

```bash
git clone https://github.com/huoji120/QWEN-EXO-booster.git
cd QWEN-EXO-booster
```

## Accessing the web console

The console is bound to `127.0.0.1` by default and must not be exposed directly to the public Internet. Use SSH local port forwarding for remote access.

### Verify the service on the GPU host

```bash
ssh <gpu-host> 'curl -f http://127.0.0.1:30000/qwen-exo/health'
```

### Create an SSH tunnel locally

```bash
ssh -N -L 30000:127.0.0.1:30000 <gpu-user>@<gpu-host>
```

Keep that terminal open, then visit:

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

Knowledge and PolicyData mutations are loopback control-plane operations. Access them only through an SSH tunnel or a trusted operator-controlled reverse proxy.

## API quick start

### Responses inference

```bash
curl --no-buffer http://127.0.0.1:30000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "duckgpt",
    "input": "Explain how this service restores hybrid-attention state.",
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

Telemetry is redacted by default: prompts, outputs, reasoning, tool arguments, references, and secrets are not written verbatim. See [API, telemetry, security, and console](docs/qwen_exo/API.md) for the detailed contract.

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

## Repository layout

```text
python/qwen_exo_booster/       QWEN-EXO runtime, Memory Pipeline, Judge, Observer, APIs
python/sglang/                 SGLang fork and model/scheduler integration
scripts/qwen_exo/              Build, launch, preflight, smoke, and evaluation tools
scripts/qwen_exo/corpus/knowledge/  Unified knowledge precompile sources: factual knowledge and Reflection Memory
scripts/qwen_exo/corpus/policydata/  Versioned PolicyData source
scripts/qwen_exo/corpus/cognition/   Optional Cognition source
docker/                        QWEN-EXO Dockerfile and deployment configuration
frontend/qwen-exo/             React/Vite web console
docs/qwen_exo/                 Architecture, API, deployment, and verification documents
test/registered/qwen_exo/      Registered regression tests
```

Runtime Markdown and JSON sources are not duplicated per model. A model switch changes only the checkpoint path and `model-profiles/<model-fingerprint>/state-*`. The selected model recompiles its Tensor Bank and Native Bank from the unified precompile sources, so models can use different tokenization, K/V, and GDN states without content forks. Generated Banks are local to that model, quantization, and topology and are not portable release assets.

The model catalog fields `checkpoint_quantization` and `runtime_quantization` must be interpreted separately from the runtime `kv_cache_dtype`. Dense 27B GPTQ is W4A16, while FP8 applies only to the Full-Attention KV cache. GDN/Mamba recurrent and convolution states are managed separately by the runtime; `--quantization fp8` is not an additional compression layer for a GPTQ checkpoint.

## Important safety boundaries

## Further reading

1. [Architecture and state contracts](docs/qwen_exo/ARCHITECTURE.md)
2. [Dual RTX 4090 deployment](docs/qwen_exo/SERVER_27B_DEPLOYMENT.md)
3. [API, telemetry, security, and console](docs/qwen_exo/API.md)
4. [Implementation progress and verification evidence](docs/qwen_exo/IMPLEMENTATION_PROGRESS.md)
5. [Demo-to-runtime migration matrix](docs/qwen_exo/DEMO_MIGRATION_MATRIX.md)

## Project scope

QWEN-EXO-booster is an SGLang-based model-runtime development project focused on **model-native state management for Qwen Hybrid models, long-context memory recall, internal task scheduling, agent continuity, and verifiable operations**. Any claim of improved model capability or accuracy must be supported by controlled evaluation with a fixed model, context, concurrency, and output length; a single demonstration is not sufficient evidence.
