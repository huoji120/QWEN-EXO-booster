# Dual RTX 4090 deployment

## Supported baseline

| Item | Value |
|---|---|
| GPUs | 2 × RTX 4090, SM89, 48 GiB each |
| Host driver | NVIDIA 550.78 or a deployment-validated compatible driver |
| Model compatibility | Verified Qwen-series Dense 27B or MoE 35B-A3B hybrid structure |
| Host model path | Set with `QWEN_EXO_MODEL_PATH` |
| Repository path | Resolved automatically, or set with `QWEN_EXO_SOURCE_PATH` |
| Runtime data | Set with `QWEN_EXO_DATA_PATH` |
| Container model alias | `/models/qwen-exo-27b` (alias only; never a compatibility signal) |
| Service | `127.0.0.1:30000` |
| Served model | `duckgpt` |
| Context target | 102400 tokens |

Qwen release labels such as 3.5, 3.6, and 3.8 may share the `Qwen3_5*`
runtime architecture shell. The launcher reads `config.json` and validates the
complete supported structure; it does not trust the host path, container alias,
or marketing version. Unsupported architectures and unverified shapes are
rejected before GPU admission and Docker startup.

The deployment uses this immutable base image:

```text
lmsysorg/sglang@sha256:30d09acc893b5647ea69fb63d5b30302e3f2199ac57c42d2e5c784cb6f2efdaf
```

The base supplies CUDA 12.6 and Torch 2.7.1. The reviewed fork is copied to
`/sgl-workspace/sglang` and selected through `PYTHONPATH`; the base image is a
binary substrate, not the source of truth.

## Build

Use a clean checkout on the deployment host. Generated caches, runtime state,
telemetry, editor artifacts, and experiment outputs are not part of the build
context.

```bash
cd /path/to/QWEN-EXO-booster
export QWEN_EXO_IMAGE=qwen-exo-booster:sglang-v0.5.16-driver550
bash ./scripts/qwen_exo/build_image.sh
```

The build runs CUDA/device, import, and kernel preflights. It is accepted only
when all three checks succeed against the pinned image and the checked-out
source tree.

Default image tag:

```text
qwen-exo-booster:sglang-v0.5.16-driver550
```

A build is accepted only when:

- `check_cuda.py` sees exactly CUDA 12.6, two SM89/BF16 devices with at least 48 GB each, and completes a finite BF16 matrix operation;
- `check_imports.py` loads SGLang from `/sgl-workspace/sglang` and imports every modified integration module;
- `check_kernels.py` completes gated RMSNorm and eager vision rotary GPU operations.

## Cutover guard

The launch script refuses to start while any GPU compute PID is present:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

Stop the legacy service through its supervisor or send `TERM` to its current worker processes. Do not copy stale PIDs into an operations runbook, and never run both backends on these GPUs at once.

Set a persistent runtime directory outside the checkout:

```bash
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime
mkdir -p \
  "$QWEN_EXO_DATA_PATH/knowledge" \
  "$QWEN_EXO_DATA_PATH/policydata" \
  "$QWEN_EXO_DATA_PATH/cognition" \
  "$QWEN_EXO_DATA_PATH/logs"
```

The launcher refreshes reviewed Markdown from the unified precompile source tree
`scripts/qwen_exo/corpus/knowledge/` into the shared runtime Knowledge lane before
startup. Factual references live at that directory's root and reusable reflection
memory lives under `reflection-memory/`. PolicyData and optional Cognition remain
separate typed lanes under `scripts/qwen_exo/corpus/`. Canonical filenames may be
overwritten by their reviewed source versions; unrelated user uploads and nested
directories are never removed or overwritten. The retired
`cognition/gpt-identity-card.md` text-injection file is intentionally removed;
Cognition is native-only.

The repository never ships a precompiled Tensor Bank or Native Bank. On first
startup, each deployment compiles the reviewed Markdown with its active tokenizer,
model fingerprint, quantization, and TP topology. Generated Banks, telemetry,
request traces, editor weights, and training jobs stay under
`QWEN_EXO_DATA_PATH` and are not committed to Git.


Source mutations remain available through the loopback-only control plane.
Keep the server bound to `127.0.0.1`, expose it only through an
operator-controlled tunnel, and admit only reviewed Markdown documents.

Required launch variables:

```bash
cd /path/to/QWEN-EXO-booster
export QWEN_EXO_MODEL_PATH=/path/to/qwen-checkpoint
export QWEN_EXO_DATA_PATH=/path/to/qwen-exo-runtime
bash ./scripts/qwen_exo/launch_js4090.sh
```

When `QWEN_EXO_ENABLED=1`, before checking GPU occupancy the launcher executes
the dependency-free model preflight in
`python/qwen_exo_booster/fingerprint.py`. A false-compatible path name cannot
bypass the structural check. The service launcher repeats the same guard inside
the container, and `ServerArgs` validates it again when `--enable-qwen-exo` is
resolved. With `QWEN_EXO_ENABLED=0`, all three QWEN-EXO structural guards are
bypassed and the launcher preserves upstream SGLang model compatibility.

Resolved defaults:

```text
--tp-size 2
--dtype bfloat16
--quantization fp8
--kv-cache-dtype fp8_e4m3
--context-length 102400
--page-size 64
--mamba-radix-cache-strategy extra_buffer
--mem-fraction-static 0.80
--max-running-requests 64
--max-prefill-tokens 65536
--cuda-graph-backend-decode full
--cuda-graph-backend-prefill disabled
--cuda-graph-max-bs-decode 8
--enable-priority-scheduling
--reasoning-parser qwen3
--tool-call-parser qwen3_coder
--enable-qwen-exo
--qwen-exo-observer-mode active
--qwen-exo-enable-adaptive-refresh
--qwen-exo-max-internal-fanout 32
--qwen-exo-max-internal-tokens 12288
```

Override these through the documented `QWEN_EXO_*` launcher variables rather
than editing the script or relying on a model directory name.

### Experimental absolute-probability expert addition

The experiment is disabled by default. When explicitly enabled, it preserves
the native Top-8 output and adds the next experts using their absolute
probability from the full router:

```text
QWEN_EXO_MOE_TOP_K=                 # preserve checkpoint Top-8
QWEN_EXO_MOE_EXTRA_EXPERTS=8        # explicit experiment: add ranks 9-16
QWEN_EXO_ENABLE_RETURN_ROUTED_EXPERTS=0
```

The native Top-8 is dispatched through the unmodified checkpoint route. The
full router softmax supplies absolute weights for the disjoint extra experts;
those outputs are accumulated as an additive path without renormalizing or
scaling down the native output. Extra experts execute in native-width Top-8
chunks, so peak fused-MoE temporary width stays unchanged. Set
`QWEN_EXO_MOE_EXTRA_EXPERTS=0` for native behavior. The result is an inference
experiment, not a trained checkpoint, and must be evaluated against native
Top-8 on the same prompts.

Score Bias defaults to `trajectory_active` with a bounded maximum bias. Select
another mode explicitly when required:

```bash
export QWEN_EXO_SCORE_BIAS_MODE=off                # disabled
export QWEN_EXO_SCORE_BIAS_MODE=trajectory_shadow  # score only
export QWEN_EXO_SCORE_BIAS_MODE=trajectory_active  # apply bounded bias
bash ./scripts/qwen_exo/launch_js4090.sh
```

The runtime aligns matched historical spans to fixed blocks, caps the positive
logit bias at `QWEN_EXO_SCORE_BIAS_MAX`, and decays it by the configured
agent-turn half-life. Missing or unalignable prompt evidence fails closed.

Historical blocks are scored from the next turn's exact `input_token_logprobs`.
If prompt logprobs are missing or cannot be aligned to the tool-output tokens,
that block is not admitted; no proxy surprisal is used.

The launcher passes the following workspace defaults into the scheduler before
model-worker initialization:

```text
QWEN_EXO_LOGPROB_CHUNK_SIZE=512             -> SGLANG_LOGPROB_CHUNK_SIZE
QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB=512   -> SGLANG_QWEN_EXO_WORKSPACE_SAFETY_RESERVE_MIB
QWEN_EXO_MAX_INTERNAL_FANOUT=32             -> global and per-parent child-job slots
QWEN_EXO_MAX_INTERNAL_TOKENS=12288          -> cumulative child tokens per parent
```

Startup fails if the logprob chunk or workspace reserve violates the reviewed
bounds. Resource admission still limits real concurrent work from live
rank-local memory; `max_running_requests=64` is a scheduler ceiling, not a
promise that every long request can run simultaneously.

On startup, the runtime performs the one-time model-native Bank prefill before reporting `ready`. The document compilation ceiling must not exceed `context_length - 2048`; the reviewed 102,400-token service therefore uses 100,352 tokens and fails closed when the configured ceiling violates that reserve. Each document preserves one complete full-document GDN recurrent/conv boundary state and up to 4,096 aligned exact Full-Attention K/V tokens selected from its salient spans. Rank-local FP8 artifacts live under the topology-scoped `model-native/<topology>/native-bank/<source-digest>/` directory and remain bound to the model fingerprint, TP topology, storage dtype, source digest, compiler threshold, span width, document ceiling, and Full-Attention budget. After a Knowledge or PolicyData change, `POST /qwen-exo/tensor-bank/reindex` eagerly rebuilds the snapshot; the service does not report `ready` until all document artifacts validate.

Do not raise `QWEN_EXO_MEM_FRACTION_STATIC` above the reviewed `0.80` default on this host. A `0.95` stress launch left only 2.29 GiB free per rank and later OOMed on an 864 MiB logits allocation during concurrent long agent traffic; `0.99` cannot preserve the required workspace. The accepted `0.80` launch leaves 9.30 GiB after pool construction and passed four concurrent 20,019-token active-observer requests.

Active Observer uses eager Prefill and full Decode CUDA Graph. Graph replay is bounded to captured batch sizes; unsupported sizes fall back to eager execution.

The launcher now defaults to active adaptive refresh. Decode pauses at the native
`</think>` boundary, commits an admitted `Self-question` / `Self-answer` pair
inside the private reasoning turn, and resumes the final answer from that exact
token prefix. Post-tool recall uses the latest assistant/tool trajectory and is
queued for the same native Think path; it is never rewritten into developer or
user instructions.

After all external references are rejected, active Context Evidence Check
judges bounded chunks of the latest direct post-tool observation. Only an
admitted observation can produce a request-local Self-Answer; assistant
reasoning is never evidence, and the result is not persisted or replayed. Set
`QWEN_EXO_CONTEXT_EVIDENCE_MODE=shadow` to collect decisions without injection,
or `off` to skip the fallback.

The committed private context contains only `Self-question: ...` and
`Self-answer: ...`. It has no XML wrapper and never emits the former
GAP/EVIDENCE/QUESTION policy-reflection record.

SGLang server warmup remains enabled so one-time CUDA compilation cannot consume
the first hidden job's deadline. Treat the service as deployment-ready only when
SGLang `/health` returns 200 (or the log reports `ready to roll`); the QWEN-EXO
runtime health route can become reachable while server warmup is still running.

### RTX 4090 MoE kernel profile

The BF16 TP=2 MoE resolves to `E=256,N=256` per rank. The checked-in
Triton 3.6.0 profile is:

```text
python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_6_0/
  E=256,N=256,device_name=NVIDIA_GeForce_RTX_4090.json
```

The target-host tuner evaluated 89 curated configurations at 18 token counts
from 1 through 4096. No separate down-projection/TMA profile was promoted;
SGLang logs that it loads the tuned profile on both TP ranks and intentionally
reuses it for the down projection.

A matched direct `/generate` protocol used one 64-token warmup followed by two
256-token, `temperature=0`, `ignore_eos=true` requests. Mean wall throughput
changed from `9.7297` to `9.9358 token/s` (`+2.12%`). A subsequent tuned-only
four-run steady sample averaged `10.4862 token/s`; it is supporting evidence,
not an A/B result. This is a modest kernel improvement, not a claim that the
remaining eager Observer, TP communication, or GDN costs were removed.

For observer-only diagnostics, explicitly opt out without editing the launcher:

```bash
export QWEN_EXO_OBSERVER_MODE=shadow
export QWEN_EXO_ENABLE_ADAPTIVE_REFRESH=0
export QWEN_EXO_CONTEXT_EVIDENCE_MODE=off
bash ./scripts/qwen_exo/launch_js4090.sh
```

## Health, API, and console

```bash
curl -f http://127.0.0.1:30000/qwen-exo/health
curl -s http://127.0.0.1:30000/qwen-exo/status
curl -s http://127.0.0.1:30000/v1/models
curl -s http://127.0.0.1:30000/qwen-exo/policydata
curl -s http://127.0.0.1:30000/qwen-exo/knowledge
curl -s http://127.0.0.1:30000/qwen-exo/recall-trace
```

Chinese user workspace and operations console:

```text
http://127.0.0.1:30000/qwen-exo/
http://127.0.0.1:30000/qwen-exo/admin
```

The console exposes separate PolicyData and Knowledge editors, direct
Markdown import, atomic save-and-compile, explicit reindex actions, and the
current-route Recall Trace. The direct compatibility page is:

```text
http://127.0.0.1:30000/recall-trace
```

From another machine, tunnel the loopback-only service:

```bash
ssh -L 30000:127.0.0.1:30000 <gpu-host>
```

## Verification sequence

Run the behavioral contract smoke first. It creates and removes its own PolicyData and Knowledge fixtures:

```bash
python3 scripts/qwen_exo/smoke_contracts.py \
  --output /data/qwen-exo-booster/logs/contracts-smoke.json
```

It verifies health, Qwen reasoning, direct PolicyData/Knowledge mutation, a model-native Tensor Bank rebuild (FP8 raw K/V plus complete Section Delta pages), independent retrieval and batch judge, Knowledge private text, PolicyData native state with `text_attached=false`, Recall telemetry, pending-background cancellation, structured function tool calls, authoritative-source rebuild, and fixture cleanup.

Run the focused post-tool Context Evidence smoke separately:

```bash
python3 scripts/qwen_exo/smoke_context_evidence.py \
  --output /data/qwen-exo-booster/logs/context-evidence-smoke.json
```

Acceptance: `passed=true`, `context_evidence.status=eligible`,
`post_tool_recall.status=context_evidence_ready`,
`post_tool_recall.reference_status=no_eligible_reference`, a direct answer no
larger than 192 bytes, and `0 < think_context_committed.token_count <= 96`.
This smoke does not run a SWE task.

Run the focused native-path probe inside the service container. It obtains a live last-Full-Attention Q sketch, performs token-level raw-K selection, then requires a first-load and second-hit transition for the same 128-token hybrid prefix:

```bash
docker cp scripts/qwen_exo/probe_native_bank.py \
  qwen-exo-booster:/tmp/probe_native_bank.py
docker exec -e PYTHONPATH=/sgl-workspace/sglang/python qwen-exo-booster \
  python3 /tmp/probe_native_bank.py
```

Acceptance: `passed=true`, `first_cache_status=loaded`, `second_cache_status=hit`, and `cached_tokens=128` for both requests.

Then run staged context and continuous-batching smoke:

```bash
python3 scripts/qwen_exo/smoke_responses.py \
  --stages 1024,32768,100000 \
  --concurrency 4 \
  --overload-concurrency 8 \
  --max-gpu-memory-fraction 0.95 \
  --output /data/qwen-exo-booster/logs/context-smoke.json
```

Acceptance:

- exact marker at every stage;
- both TP ranks remain healthy;
- no OOM, rank divergence, or watchdog timeout;
- 100K leaves bounded GPU memory;
- four concurrent requests all complete;
- overload is a structured 429, never a rank-local allocator failure.
- four concurrent 20K active-observer requests fit the transient FP32 workspace and leave the health endpoints responsive.

For eviction/refill coverage, run the 100K stage twice and then rerun `smoke_contracts.py`. The verified target completed this sequence and logged a 99,968-token native prefix match without corrupting subsequent judge, cancellation, or tool-call requests.

Measured results and baseline caveats are recorded in [IMPLEMENTATION_PROGRESS.md](IMPLEMENTATION_PROGRESS.md).

## Logs, state, and rollback

```bash
tail -f /data/qwen-exo-booster/logs/server.log
docker stop qwen-exo-booster
```

The container is ephemeral. Authoritative PolicyData, Knowledge, execution capsules, JSONL traces, smoke reports, and locally derived rank-local Bank artifacts live under `/data/qwen-exo-booster` and survive restart. Only reviewed Markdown under `scripts/qwen_exo/corpus/knowledge/` is published; every deployment builds its own Tensor Bank and Native Bank.


Rollback is clean: stop the QWEN-EXO container and restart the unchanged legacy checkout and virtual environment. The new runtime contains no external-learning dependency or credential.
