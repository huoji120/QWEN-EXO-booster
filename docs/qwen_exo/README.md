# QWEN-EXO-booster documentation

QWEN-EXO-booster is the SGLang-based production rewrite of the
`in-flight-tensor-retrieval` semantic demo. The default profile uses the
verified Dense 27B Qwen hybrid layout (`Qwen3_5ForConditionalGeneration`).
Qwen release labels are not used as a compatibility signal: several Qwen
generations share the `Qwen3_5*` runtime shell, so startup validates the actual
model structure.

## Read in this order

1. [Architecture and contracts](ARCHITECTURE.md)
2. [Dual RTX 4090 deployment](SERVER_27B_DEPLOYMENT.md)
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

The preflight reads `config.json`; it does not infer compatibility from the
directory name. Only the structurally verified Dense 27B and MoE 35B-A3B
`Qwen3_5*` runtime layouts are accepted.

## Quick local verification

```bash
PYTHONPATH=python python -m pytest test/registered/qwen_exo -q
python -m ruff check --select F401,F821,UP037 python/qwen_exo_booster
cd frontend/qwen-exo && npm ci && npm run build
```


## Versioned knowledge and memory

Reviewed source material is committed under `scripts/qwen_exo/corpus/` and is
copied into the runtime data directory by the launcher:

- `knowledge/`: factual reference documents;
- `knowledge/reflection-memory/`: curated, reusable execution memories;
- `policydata/`: authoritative execution policy;
- `cognition/`: optional reviewed cognition documents when present.

Generated Tensor Bank state, caches, telemetry, request traces, training jobs,
editor weights, raw trajectories, and smoke outputs are runtime artifacts and
are intentionally excluded from Git. Challenge-specific and task-specific test
corpora are not part of the published memory set.

## Runtime entry points

| Purpose | Entry point |
|---|---|
| Build image | `scripts/qwen_exo/build_image.sh` |
| Launch dual-4090 server | `scripts/qwen_exo/launch_js4090.sh` |
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
