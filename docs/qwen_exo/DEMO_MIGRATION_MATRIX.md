# Demo-to-runtime migration matrix

The source demo is a semantic oracle, not a service implementation to copy.
This matrix records the clean cutover.

| Demo concern | Demo location/pattern | QWEN-EXO implementation | Cutover rule |
|---|---|---|---|
| Model execution | `inference.py` `ModelWorker`, Transformers `generate` | native SGLang tokenizer manager, scheduler, TP workers | no Accelerate device-map worker |
| GPU placement | one serialized worker/replica queue | true `TP=2` plus continuous batching | no layer-by-layer pseudo-TP |
| Full-Attention KV | process-local tensors | rank-local FP8 raw-K/V Bank artifacts; selected K/V is re-RoPE'd into native SGLang paged KV/radix slots | allocator/cache owns lifetime; no Python past-key-values |
| GDN recurrent/conv state | ad-hoc retained tensors | complete rank-local FP8 Section Delta loaded into the native hybrid Mamba pool | selected K/V and the nonlinear recurrent/conv state transition together |
| Candidate proposal | model-native recall/Tensor Bank | TP-synchronized raw Attention-Q against per-token raw-K sketches, selecting a 64/128-token aligned prefix from 512-token pages | proposal never implies admission |
| Shadow reference check | serial shadow forward | shared-prefix constrained `ReferenceJudge` jobs | malformed output is ineligible |
| Self-Ask | recursive/serial model work | `SelfAskRefreshService` sibling jobs | no recursive HTTP or recursive job depth |
| Self-Answer/replay | direct injection into current loop | eligible-only record, next-safe-turn restore and re-judge | no mid-decode DeltaNet swap |
| Semantic eligibility | distributed boolean checks | typed `EligibilityDecision` with question/reference/model digests | one gate for prefill, refresh, replay, restore |
| Execution trajectory | `trajectory_memory.py` | constrained `ExecutionCapsuleService` and atomic store | update on meaningful terminal response |
| Memory mixing | `memory_mixture.py` | native Bank prefix for the primary admitted page; bounded private instruction compiler for non-Bank references | an imported native candidate is not re-prefilled as reference text |
| Support Markdown | `support_runtime.py` | authoritative `KnowledgeRepository` | `.md`/`.markdown`, safe paths, atomic writes |
| Observer | decode hooks in demo | selected-token logprob + exact Attention-Q/custom info channel | state keyed by request ID, not batch row |
| Memory energy | demo attention diagnostics | pre-RoPE Q-to-selected-native-K centroid proxy registered at Bank load | explicitly not exact softmax mass |
| Capacity | local free-memory checks | TP-consensus scheduler reservation plus native allocation | 429 before rank-local OOM |
| Telemetry | demo metadata/visualization | redacted JSONL, REST, SSE, operations console | raw text is opt-in only |
| API | custom demo server | upstream `/v1/responses` plus `/qwen-exo/*` control plane | keep OpenAI event/tool semantics |
| External learning | optional network completion | none | no hidden external model dependency |

## Behavior preserved

- thinking-capable Responses requests;
- structured function tools through upstream parsers;
- decode-time uncertainty from selected-token surprisal;
- candidate proposal followed by semantic rejection/admission;
- fail-closed protection against irrelevant WFP/CTF references;
- long-task coarse state continuity through execution capsules;
- next-turn memory restoration;
- one-time model-native Bank export, mmap reload, radix hit reuse, and source-token-bound restore;
- request-correlated diagnostics.

## Behavior intentionally removed

- serial shadow forwards in the user decode loop;
- recursive calls to the HTTP server;
- one complete model replica per configured GPU;
- storing model prompts/reasoning/references in telemetry by default;
- accepting candidate rank as proof of semantic relevance;
- restoring stale source bytes after a document digest changes;
- pretending that an in-memory prototype proves 100K, concurrency, latency, or
  quality.

## Golden cutover gate

Before replacing the demo in production, run the same public fixtures through
both systems and compare observable contracts:

1. ordinary text response and stream event order;
2. thinking summary boundaries;
3. structured tool call name/arguments;
4. relevant reference admitted;
5. irrelevant WFP/CTF reference rejected;
6. no-judge and malformed-judge fail closed;
7. next-turn source is re-judged;
8. cancellation removes owned hidden jobs;
9. trace output contains no raw sensitive fields;
10. capacity exhaustion is structured, not an OOM or TP hang.
11. native Bank artifacts bind model/source/page/token identities and cache-hit telemetry;
12. first load reserves selected Full-Attention K/V plus cached and active GDN slots atomically across TP ranks.

Hidden benchmark prompts, solutions, and verifier content must never be copied
into knowledge sources or fixtures.
