# Implementation progress and evidence

Updated: 2026-08-10

## Status legend

- **Implemented**: code and focused behavior tests pass.
- **Integrated**: wired into SGLang's request, model, allocator, and scheduler paths.
- **Server-verified**: exercised with the real Dense 27B and MoE 35B-A3B checkpoints on both RTX 4090s.
- **Pending**: no claim until the named evaluation is run.

## Capability matrix

| Capability | State | Evidence |
|---|---|---|
| Qwen-series model structure and fingerprint | Server-verified | compatibility is derived from `config.json`, not release labels or paths; verified layouts are Dense 27B (`Qwen3_5ForConditionalGeneration`, 64 layers: 16 Full Attention + 48 Gated DeltaNet) and MoE 35B-A3B (`Qwen3_5MoeForConditionalGeneration`, 40 layers: 10 + 30, 256 experts with 8 active) |
| True TP=2, BF16, 64-token pages | Server-verified | two scheduler ranks and two GPU compute processes; runtime status reports TP=2/BF16/page=64 |
| Atomic Full-Attention/GDN lifecycle | Server-verified | native UnifiedRadixCache stress, eviction/refill recovery, repeated 100K requests, and exact native-path recovery for stale split metadata under batched internal judges |
| Scheduler-native internal jobs | Server-verified | reference judge and capsule jobs executed through `TokenizerManager`; pending background cancellation passed |
| Strict batch reference judge | Server-verified | relevant reference admitted as `true`; malformed, stale, changed-digest, timeout, and long-question fail-closed tests pass |
| Independent PolicyData + Knowledge memory | Server-verified | separate loopback-only sources → independent rank → shared-prefix fail-closed judge → lane-specific private attachment → exact answer → cleanup |
| Model-native FP8 Tensor Bank | Server-verified | MoE/TP=2 cold startup exported 9 model-native pages with raw K/V plus complete GDN Section Delta; Dense 27B `loaded → hit` evidence remains archived |
| Next-turn eligible-memory restoration | Integrated, automated-verified | current-turn re-judge selects a source-bound native Bank prefix and does not re-prefill the admitted reference text |
| In-flight surprisal, Attention-Q drift, memory energy | Server-verified | request-correlated decode summaries populated in the live telemetry console |
| Self-Ask/Self-Answer and safe replay | Integrated, automated-verified | eligible-only replay branches load native Bank state, preserve source/model/token identity, and fail closed on invalid answers/artifacts |
| Execution capsule update/restore | Server-verified | terminal update jobs ran after real generations; private restore and context-capacity rollback tests pass |
| Cross-rank admission and structured overload | Server-verified baseline; native Bank automated-verified | peer consensus reserves KV, request, workspace, and both cached+active Mamba slots for a first native Bank load; structured 429 replaces allocator/OOM failure |
| Bounded JSONL/SSE telemetry | Server-verified | redaction is the default; the current diagnostic deployment explicitly enables full-text telemetry and reports `redacted=false` |
| Direct source APIs | Server-verified | real PolicyData and Knowledge PUT/DELETE without a token prompt; deployment remains loopback-only and public metadata remains content-free |
| Operations and Recall consoles | Server-verified | separate source editors, current-route Recall inspector, byte-regressed legacy-compatible page, and 1440×1000 no-overflow browser QA |
| Qwen thinking and structured tool calls | Server-verified | explicit reasoning item plus final answer; `return_marker` function call parsed with exact arguments |
| 1K / 32K / 100K execution | Server-verified | exact marker returned at every stage; 100K prompt measured at 99,972 input tokens |
| Continuous batching | Server-verified | four concurrent 1K requests all completed in 1.934 s wall time, overlap factor 3.975 |
| Accuracy uplift | Measured, not established | expanded DeepSWE diagnostics produced binary reward 0 on every grader-bearing run; the best short-spec submission passed 61/66 feature tests while preserving 30,014/30,014 prior tests |

## 27-deliverable acceptance ledger

This ledger audits every implementation/evaluation slice; “measured, no uplift” is a completed negative result, not a quality claim.

| ID | Deliverable | Acceptance | Primary evidence |
|---:|---|---|---|
| 01 | Frozen SGLang/container baseline | Server-verified | immutable base digest, source bind mount, CUDA/import/kernel preflights |
| 02 | Qwen3.5 model identity and hashes | Server-verified | runtime fingerprint, architecture/layer split, 55,562,855,904 weight bytes |
| 03 | True TP=2 BF16 model and Mamba state | Server-verified | two scheduler ranks/processes and runtime hybrid-state contract |
| 04 | 1K/32K/100K native execution | Server-verified | exact markers; 99,972-token measured 100K request |
| 05 | Typed per-sequence hybrid handle | Automated-verified | `contracts.py`, `hybrid_state.py`, lifecycle/state-identity tests |
| 06 | Atomic Full-Attention KV + GDN lifecycle | Server-verified | UnifiedRadixCache eviction/refill and stale-prefix recovery |
| 07 | Prefix namespace and exact memory-span identity | Automated-verified | `memory_span.py`, namespace/digest tests, native attachment metadata |
| 08 | Scheduler-native hidden jobs | Server-verified | direct `GenerateReqInput` judge/capsule jobs; no recursive HTTP |
| 09 | Parent deadline/priority/token budget/cancellation | Server-verified | pending-background cancellation smoke plus internal-job tests |
| 10 | Shared-prefix batch reference judge | Server-verified | independent Knowledge/PolicyData batch decisions in real contract smoke |
| 11 | Strict fail-closed judge parsing | Automated-verified | malformed, partial, stale digest, timeout, and long-question tests |
| 12 | One typed semantic-eligibility gate | Automated-verified | pipeline/refresh/replay/restoration tests use the same decision binding |
| 13 | Independent Knowledge repository | Server-verified | direct Markdown PUT/rank/attach/DELETE and separate source digest |
| 14 | Independent PolicyData repository | Server-verified | direct Markdown PUT/rank/attach/DELETE and separate source digest |
| 15 | Eligible-only private attachment | Server-verified | lane-specific exact answers; rejected candidates never attached |
| 16 | Next-turn restore with current-turn re-judge | Automated-verified | stale/missing/changed-source restoration tests |
| 17 | Self-Ask/Self-Answer refresh controller | Automated-verified | off/shadow/active gates, cooldown, trigger cap, eligible-only answer tests |
| 18 | Safe causal replay/Maybe decision | Automated-verified | source binding, gain/switch/KL thresholds, invalid-answer fail-closed tests |
| 19 | Execution capsule update and restore | Server-verified | terminal hidden update jobs, private restore, capacity rollback tests |
| 20 | Selected-token surprisal | Server-verified | request-correlated live decode telemetry |
| 21 | Last-Full-Attention-layer Q summaries | Server-verified | eager Q hook and live Q magnitude/drift telemetry |
| 22 | External-memory energy proxy | Server-verified | exact-span K centroid and Q-alignment telemetry, explicitly not softmax mass |
| 23 | Unified TP-consensus resource admission | Server-verified | KV/request/Mamba/workspace reservation, structured 429, 4 × 20K stress |
| 24 | Responses thinking/tool/stream/cancel semantics | Server-verified | visible reasoning/final separation, exact structured call, cancellation smoke |
| 25 | Redacted persistent REST/SSE telemetry | Server-verified | text-sealed JSONL, bounded retention, restart and stream-resume tests |
| 26 | Loopback control plane and Chinese consoles | Server-verified | direct editors/reindex/Recall trace; 1440 × 1000 browser overflow QA |
| 27 | Task-quality probe | Measured, no uplift | short, medium, large, repeated, and long-loop DeepSWE diagnostics all had binary reward 0 or were cancelled without a submission; see the evidence table below |

Audit result: all 27 slices have implementation or measured evidence. Twenty-six runtime/operations slices are accepted at their stated level; slice 27 is a completed negative evaluation and blocks any accuracy-uplift claim.

Native Bank cutover addendum: the rank-local raw-K/V + complete Section Delta path is covered by focused export, FP8 round-trip, source-token binding, Qwen3.5 MRoPE inversion, RoPE rebasing, scheduler materialization, TP-shape mismatch, unified-radix cache hit, cache namespace, causal replay, next-turn restore, and operations-reindex tests. It is also revalidated on the real 27B checkpoint with TP=2; the live probe observed `loaded → hit` for the same 128-token Q-selected native prefix.

## Automated verification

Current workstation verification on 2026-08-10:

```text
PYTHONPATH=python python -m pytest -q test/registered/qwen_exo
418 passed, 31 skipped
```

Historical 2026-07-31 evidence: the earlier workstation run skipped 24 Linux-only tests that imported the native SGLang scheduler/runtime (`resource`, CUDA/TP process groups, and allocator integration). Its synchronized 182-test suite was rerun inside the live Linux service container with `pytest-asyncio`; all 182 tests passed with no skip.

The workstation warnings are the pre-existing NumPy/Torch binary mismatch and TypedStorage deprecation.

```text
PYTHONPATH=/sgl-workspace/sglang/python python3 -m pytest -q /tmp/qwen_exo_tests
182 passed in 12.83s
```

Static and packaging checks:

```text
ruff: QWEN-EXO sources/scripts/tests clean
ruff format: clean
undefined-name checks on modified SGLang integration files: clean
compileall: clean
node --check admin.js: clean
legacy Recall renderer SHA-256 regression: `487e84d694420dfefef6c8bcd49bfe4cc313cae76be4c10bc80bf55ef7c98f9a`
```

Independent final review reported no remaining Blocker or High finding.

## Native Tensor Bank live verification

The final `js4090` run used the real 55.56 GB checkpoint, BF16, TP=2, 64-token allocator pages, and the release `0.80` memory watermark:

- Cold startup built 18 logical 512-token Bank pages and 36 mmap-loadable rank artifacts. Runtime status reported `full_attention_kv_pages=18`, `complete_gdn_boundary_pages=18`, `storage_dtype=float8_e4m3fn`, and `selectable_prefix_tokens=128`.
- A warm restart loaded and validated the persisted snapshot without model-prefilling the sources again; application startup completed in 9 seconds after model/pool initialization.
- `probe_native_bank.py` obtained a real 32-dimensional last-Full-Attention Q sketch, selected page 4 with score `0.6267538667`, chose 128 source tokens, and observed `first_cache_status=loaded`, `second_cache_status=hit`, and `cached_tokens=128` on both requests.
- The complete Responses smoke passed thinking, structured tool call, cancellation, Knowledge text attachment, PolicyData native-state injection, native Bank reindex, cleanup, and authoritative-source rebuild.
- Final server log contained no scheduler exception, traceback, or native Bank export failure; `/qwen-exo/health` remained `ready` after the probe.

## Native PolicyData state cutover

The 2026-07-31 server cutover removed the PolicyData text compiler and all
request-side PolicyData reflection insertion. After semantic eligibility, one
highest-ranked policy page owns the request's single recurrent-state slot.
SGLang restores its selected Full-Attention raw K/V and complete GDN
recurrent/conv Section Delta under one aligned radix identity; Knowledge can
still use its separate untrusted text path. Stale, missing, unaligned, or
over-budget policy state has no text fallback.

The real TP=2 27B smoke rebuilt schema-4 Bank artifacts, selected a short
PolicyData fixture as page 7, padded its compiler-only state to 64 tokens, and
reported `native_policy_full_attention_kv_and_gdn_state`,
`text_attached=false`, and `complete_section_delta_plus_selected_raw_kv`. The
model returned the exact hidden policy marker `QWEN_EXO_POLICY_READY_41C9` with
HTTP 200. Cleanup rebuilt the authoritative Bank from 14 fixture pages to 9
pages. The client-agent compactor, system prompt, and action policy were not
changed for this cutover.

## Target-server baseline

| Item | Observed value |
|---|---|
| Host | `js4090` |
| Driver | 550.78 |
| GPUs | 2 × RTX 4090, SM89, 48 GiB each |
| Container CUDA | 12.6 under driver minor-version compatibility |
| Torch / Triton | 2.7.1+cu126 / 3.6.0 |
| Model fingerprint | `e6acdcc448e1b440bff5372dde93172fdb16cb63f10e13f7e72ba7e70b27d9d2` |
| Checkpoint weight bytes | 55,562,855,904 |
| KV pool | 205,248 BF16 tokens per rank |
| Mamba pool | 154 slots per rank |
| Post-pool free memory | 9.30 GiB per rank |

`check_cuda.py`, full source imports, gated RMSNorm, and eager vision rotary kernel preflights passed on both target GPUs. The source/base mismatch found during deployment was fixed explicitly: current FLA kernels require Triton 3.6, while Torch 2.7's Inductor path is bypassed only for the QWEN-EXO vision rotary call.

Historical 2026-07-31 operational evidence: container `qwen-exo-booster` ran on `127.0.0.1:30000` with TP=2, observer `shadow`, adaptive refresh disabled, `mem_fraction_static=0.80`, logprob chunk 512, and workspace safety reserve 512 MiB; both health endpoints passed after a real Responses request returned HTTP 200/completed with exact output `READY`. The current release launcher instead defaults to active Observer, adaptive refresh enabled, eager Prefill, and a full Decode CUDA Graph. This documentation update does not claim a new server deployment or measurement.

## RTX 4090 MoE kernel tuning

The live Qwen3.6-35B-A3B BF16 TP=2 path now loads a Triton 3.6.0 profile for
`E=256,N=256` on both RTX 4090 ranks. The tuner selected 18 batch-size entries
from 89 curated candidates. A matched direct-generation wall benchmark moved
from `9.7297` to `9.9358 token/s` (`+2.12%`); a later four-run tuned-only sample
averaged `10.4862 token/s`. The Responses correctness smoke completed with the
exact final output `TUNED_MOE_READY`, and QWEN-EXO health, persistent telemetry,
active Observer, Score Bias, and the model-fingerprint-matched native Bank all
remained healthy. The down projection reuses the tuned up-projection profile
without TMA and is still a separate optimization opportunity.
The focused config-loader test ran in a CPU-only container on `js4090` and passed `2/2`; no local model or CUDA process was started.

## GPU workspace admission and release watermark

The requested high-watermark test found a real non-pool allocation that KV/Mamba accounting alone did not cover. At `mem_fraction_static=0.95`, each rank started with 326,976 KV tokens and 2.29 GiB free, then concurrent DeepSWE traffic failed when a rank attempted an additional 864 MiB FP32 logits allocation with only 827.56 MiB free. `0.99` was therefore not promotable, and `0.95` is also rejected for this workload.

The source fix is fail-closed rather than a launch-only workaround:

- QWEN-EXO constrains `SGLANG_LOGPROB_CHUNK_SIZE` to at most 512 rows;
- every scheduler admission charges `2 × min(prompt_tokens, 512) × padded_vocab_size × sizeof(FP32)` for overlapping raw and indexed logits;
- each TP rank reads live CUDA free memory, subtracts a minimum 512 MiB safety reserve, and participates in the existing atomic consensus;
- active reservations use the maximum concurrent scratch requirement because the processor owns one shared chunk workspace, while KV, request, and Mamba reservations remain additive.

The release default remains `mem_fraction_static=0.80`: 205,248 KV tokens, 154 Mamba slots, and 9.30 GiB post-pool free memory per rank. With active Self-Ask enabled, four independent 20,019-token requests completed `200/completed` together in 37.043 s; post-response health returned 200, the log contained no OOM/traceback, and observed device usage was 43,518/43,079 MiB. This is the accepted capacity point, not a claim that a higher static fraction is safe.

## End-to-end contract smoke

The real 27B service passed, in one run:

1. health and runtime identity;
2. visible Qwen reasoning plus a final marker;
3. direct loopback Knowledge fixture upsert;
4. direct loopback PolicyData fixture upsert;
5. independent Knowledge retrieval, scheduler-native judge, private attachment, and exact answer;
6. independent PolicyData retrieval, scheduler-native judge, native K/V + GDN state, no request text, and exact answer;
7. persisted `inflight-memory-visualization-v2` traces for both admitted lanes and decode observation;
8. cancellation while a background request was still in judge preparation;
9. structured function tool call;
10. direct cleanup of both fixtures.

The same contract run passed after two consecutive 99,972-token requests forced radix eviction/refill. This specifically covers the recovered native-prefix metadata path that previously exposed a scheduler crash.

A Chinese post-tool smoke after the Self-Ask cutover to the strict
`submit_self_question` JSON schema completed `200/completed` in 46.951 s. The
visible reasoning contained a Chinese `Self-question`; request-scoped telemetry
reported `policy_reflection_ready`, `admitted=true`, `think_context_ready=true`,
`text_injected=false`, and a 90-token `self_ask.think_context_committed` event.
This replaces the language-sensitive `QUESTION|...` parser.

## Context and concurrency results

All measurements are end-to-end Responses API wall time on the same dual-4090 host. The staged prompt used deterministic filler and an exact marker; QWEN-EXO used `reasoning.effort=none` for the context benchmark.

| Backend | Input target | Result | Wall time |
|---|---:|---|---:|
| QWEN-EXO / SGLang | 1K | exact marker | 1.325 s |
| Legacy Transformers demo | 1K | completed, but all 32 output tokens were hidden thinking and no final marker | 19.963 s |
| QWEN-EXO / SGLang | 32K | exact marker | 14.629 s |
| Legacy Transformers demo | 32K | completed, but all 32 output tokens were hidden thinking and no final marker | 44.137 s |
| QWEN-EXO / SGLang | 100K | exact marker, 99,972 measured input tokens | 51.691 s |
| Legacy Transformers demo | 100K | unsupported: configured maximum is 91,517 | — |

Observed end-to-end ratios are 15.07× at 1K and 3.02× at 32K. These are behavioral service measurements, not normalized kernel-only numbers: the legacy endpoint ignored the requested no-thinking mode and consumed its full 32-token ceiling, while QWEN-EXO returned the final marker in 12–14 tokens.

Four concurrent cached-prefix 1K requests:

| Backend | All requests | Wall time | Overlap factor |
|---|---|---:|---:|
| QWEN-EXO / SGLang | 4/4 completed with exact markers | 1.934 s | 3.975 |
| Legacy Transformers demo | 4/4 completed, no visible final markers | 16.369 s | 2.495 |

The QWEN-EXO batch wall time was 8.46× lower. A repeated 100K stress request was admitted after cache pressure, completed in 42.064 s, and scheduler logs recorded a 99,968-token native prefix match during the same lifecycle.

Final active-observer workspace stress used four independent 20,019-token prompts with 64 output tokens. All four returned HTTP 200 with completed Responses objects in 37.043 s wall time. This run exercised the release `0.80` memory watermark after the scheduler workspace fix; the service remained healthy with no CUDA OOM.

## Expanded DeepSWE diagnostics

These are end-to-end agent diagnostics against frozen tasks and hidden graders, not a statistically controlled comparison against another model. Historical passes used several agent revisions, so only like-for-like rows should be compared directly.

| Task | Scale | Model calls / compactions | Maximum observed prompt | Feature tests | Prior tests | Result |
|---|---|---:|---:|---:|---:|---|
| `mashumaro-flattened-dataclass-fields` | shortest task statement in the 113-task set, 79 words | 132 / 0 | 69,851 | 61/66 | 30,014/30,014 | reward 0; submitted in 40m49s |
| `mashumaro-flattened-dataclass-fields` quality-guard candidate | same frozen task; durable compaction + semantic guard + periodic contract reminders | 236 / 2 | 71,462 | 53/66 | 30,014/30,014 | reward 0; submitted in 1h33m31s |
| `cattrs-partial-structuring-recovery` | medium, 178 words | 59 / 0 | 41,101 | 53/69 | 7/7 | reward 0; best clean recorded submission |
| `fastapi-deprecation-response-headers` | large, 423 words and 20 explicit behaviors plus inheritance rules | 136 / 1 | 70,463 | 40/137 | 3,134/3,134 | reward 0; submitted in 45m42s |
| `arktype-json-schema-refs-dependencies` | recursive parser/schema task | 205 / 2 | 71,478 | not graded | not graded | cancelled after a non-progress loop |

The Mashumaro result explains why the model can appear successful on a small task: its own focused tests passed and 92.4% of hidden feature tests passed, but five explicit edge contracts were still wrong. Four failures share one cause: the implementation flattened raw child attributes instead of reusing the child's serialization/deserialization path, so child aliases, omit-none behavior, and nested dataclasses were lost. The fifth missed alias-aware collision detection. No context compaction occurred.

The same-task quality-guard candidate did not improve correctness: feature coverage fell from 61/66 to 53/66 while calls rose from 132 to 236 and wall time from 40m49s to 1h33m31s. Its semantic stagnation guard never fired, while periodic contract reminders were injected 12 times and also caused the trajectory converter warning `Message at index 2 has no preceding agent step`. This stochastic two-run comparison cannot attribute the score drop to one component, but it provides no evidence to ship periodic reminders; that mechanism was removed. The action-policy templates remain upstream-owned.

The FastAPI submission similarly preserved every prior test and implemented much of the mechanical API/OpenAPI propagation, but runtime headers and inheritance semantics were mostly wrong. At model call 73, with a 54,364-token prompt and before the only compaction at call 94, the model explicitly identified route-handler injection as the requested architecture and then chose a manually added middleware because it seemed cleaner. Its new tests were changed to install that middleware, narrowing the contract. This is a design and verification failure that predates compaction, not missing access to the task text.

The cattrs runs also failed without compaction. Across nine recorded cattrs attempts, no run earned binary reward; clean submissions passed between 37 and 53 of 69 feature tests while retaining 7/7 prior tests. Eight graded historical `fastapi-implicit-head-options` attempts likewise earned reward 0; the only run reaching 19/43 feature tests regressed 280 prior tests, while clean submissions remained at 0/43.

The original ArkType trajectory exposed two controller defects: repeated semantically identical searches and lossy recursive summaries. Follow-up controller experiments were diagnostic only and were rejected because they did not establish task-accuracy uplift.

The shipped DeepSWE controller remains the paper-faithful QWEN-EXO compactor. It has no periodic task reminder, semantic stagnation guard, or system/action-template override. The only retained runner control is the explicit finite Pier timeout multiplier of 1.5. Historical focused controller evidence recorded `187 passed, 25 skipped`; the agent-specific compaction suite recorded `7 passed`, with Ruff check and format clean.

The coding-agent requests did not exercise QWEN-EXO next-turn state restoration. Across the 132-call Mashumaro, 136-call FastAPI, and 205-call ArkType traces, every response had `previous_response_id` absent, `next_turn_restoration.status=not_requested`, and native-prefix restore inactive. Static Knowledge and trusted PolicyData were successfully selected every turn, but there was no linked task-specific long-term state. Retrieval availability therefore did not guarantee policy compliance.

The serving runtime itself remains separately verified: the exact-marker smoke passed at 99,972 input tokens, and the diagnostic prompts stayed below the configured 102,400-token limit. The evidence supports a narrow conclusion: physical long-context execution works; autonomous repository-task correctness does not. Context dilution and lossy repeated compaction worsen long runs, but the dominant observed failures are authoritative-path selection, exhaustive contract tracking, test-oracle quality, and tool-loop control.

Raw results are retained on `js4090` under `/data/deep-swe/qwen-exo-reflection-eval/`, `/data/deep-swe/qwen-exo-pass-eval/`, `/data/deep-swe/qwen-exo-full-eval/`, and `/data/deep-swe/qwen-exo-focused-eval/`. Combined `partial` scores near 1.0 are dominated by large prior-test suites and must not be reported as feature accuracy; binary reward and feature-test pass rate remain the promotion gates.

No correctness-uplift percentage is claimed. The evaluated MoE service is accepted as a serving/runtime implementation, not as an autonomous complex-coding agent. A promotion test requires a frozen multi-task sample, bounded semantic-loop guards, a task-state ledger, fresh-context review, identical sampling and agent policy, and a same-task stronger-model baseline.
