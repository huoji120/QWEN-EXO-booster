# QWEN-EXO-booster documentation

QWEN-EXO-booster is the SGLang-based production rewrite of the
`in-flight-tensor-retrieval` semantic demo. The default profile uses the
verified Dense 27B Qwen hybrid layout (`Qwen3_5ForConditionalGeneration`).
Qwen release labels are not used as a compatibility signal: several Qwen
generations share the `Qwen3_5*` runtime shell, so startup validates the actual
model structure.

## Read in this order

1. [Architecture and contracts](ARCHITECTURE.md)
2. [Apple Silicon MLX deployment](APPLE_SILICON_MLX_DEPLOYMENT.md) or
   [Dual RTX 4090 deployment](SERVER_27B_DEPLOYMENT.md)
3. [API, telemetry, security, and console](API.md)
4. [Implementation progress and evidence](IMPLEMENTATION_PROGRESS.md)
5. [Demo-to-runtime migration matrix](DEMO_MIGRATION_MATRIX.md)

## Quick start

```bash
export QWEN_EXO_MODEL_PATH=/path/to/qwen-checkpoint
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime

# Optional explicit preflight; launch_js4090.sh runs it when QWEN_EXO_ENABLED=1.
python3 python/qwen_exo_booster/fingerprint.py "$QWEN_EXO_MODEL_PATH"
bash scripts/qwen_exo/build_image.sh
bash scripts/qwen_exo/launch_js4090.sh
```

Apple Silicon uses the native MLX path instead of the Docker/CUDA profile:

```bash
bash scripts/qwen_exo/install_mlx.sh
bash scripts/qwen_exo/launch_mlx.sh
```

The preflight reads `config.json`; it does not infer compatibility from the
directory name. Only the structurally verified Dense 27B and MoE 35B-A3B
`Qwen3_5*` runtime layouts are accepted.

## Quick local verification

```bash
PYTHONPATH=python python -m pytest test/registered/qwen_exo -q
python -m ruff check --select F401,F821,UP037 python/qwen_exo_booster
cd frontend/qwen-exo && npm ci && npm run build
```


## Unified knowledge precompile sources

The single publishable knowledge source tree is
`scripts/qwen_exo/corpus/knowledge/`. Factual references live at its root and
reviewed, reusable execution memories live under `reflection-memory/`. The
launcher copies this tree into the shared runtime Knowledge lane before startup.
PolicyData and optional Cognition remain separate typed lanes under
`scripts/qwen_exo/corpus/`.

All configured models read the same reviewed Markdown. Each selected model then
compiles its own topology-scoped Tensor Bank and Native Bank under
`model-profiles/<model-fingerprint>/`, using that checkpoint's tokenizer,
fingerprint, quantization, and TP layout. Precompiled Banks are deliberately not
published because they are not portable across those boundaries; every
deployment compiles its own on first startup or reindex.

Generated Bank state, caches, telemetry, request traces, training jobs, editor
weights, raw trajectories, and smoke outputs are runtime artifacts and are
excluded from Git. Challenge-specific and task-specific trajectory corpora are
not part of the published memory set.

For the Dense 27B GPTQ profile, use the catalog-derived `gptq_marlin` runtime loader with `dtype=float16`, FP16 GDN/Mamba state, and `kv_cache_dtype=fp8_e4m3`; this is W4A16 plus FP8 Full-Attention KV cache, not FP8 weights layered on top of GPTQ. Other CUDA profiles retain the BF16 model/state baseline.

## Runtime entry points

| Purpose | Entry point |
|---|---|
| Build image | `scripts/qwen_exo/build_image.sh` |
| Launch dual-4090 server | `scripts/qwen_exo/launch_js4090.sh` |
| Install Apple Silicon environment | `scripts/qwen_exo/install_mlx.sh` |
| Launch Apple Silicon MLX server | `scripts/qwen_exo/launch_mlx.sh` |
| MLX/Metal check | `scripts/qwen_exo/check_mlx.py` |
| CUDA/device check | `scripts/qwen_exo/check_cuda.py` |
| GPU kernel check | `scripts/qwen_exo/check_kernels.py` |
| Behavioral contract smoke | `scripts/qwen_exo/smoke_contracts.py` |
| Context/concurrency smoke | `scripts/qwen_exo/smoke_responses.py` |
| Responses API | `POST /v1/responses` |
| Runtime status | `GET /qwen-exo/status` |
| Chinese user workspace | `GET /qwen-exo/` |
| Operations console | `GET /qwen-exo/admin` |

## Safety defaults

- observer: `active`;
- adaptive refresh: enabled;
- raw trace text: disabled;
- source mutations: directly available on the loopback-bound control plane;
- external learning: absent;
- invalid semantic decisions: ineligible;
- CUDA graph: full Decode graph with eager Prefill (`prefill` graph disabled).
