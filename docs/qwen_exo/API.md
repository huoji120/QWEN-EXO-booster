# QWEN-EXO API and operations console

Base URL in the dual-4090 deployment: `http://127.0.0.1:30000`.

## Responses API

QWEN-EXO extends SGLang's existing OpenAI-compatible Responses API; it does not
introduce a second generation protocol.

```bash
curl --no-buffer http://127.0.0.1:30000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "duckgpt",
    "input": "Explain the exact WFP layer used for outbound connect authorization.",
    "stream": true,
    "max_output_tokens": 256
  }'
```

Continue a response trajectory:

```bash
curl http://127.0.0.1:30000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "duckgpt",
    "previous_response_id": "resp_...",
    "input": "Continue from the verified state and state the next action."
  }'
```

OpenCode does not send `previous_response_id`. Both its native Responses path
and default AI SDK path resend the prepared full history and send
`prompt_cache_key` equal to the OpenCode session ID. `ResponsesRequest` accepts
this field, and QWEN-EXO uses a non-empty value as the primary stateless
conversation identity:

```json
{
  "model": "duckgpt",
  "prompt_cache_key": "<OpenCode session ID>",
  "input": ["<full prepared message history>"]
}
```

Without `prompt_cache_key`, QWEN-EXO falls back to versioned canonical bytes for
effective `instructions`, role-separated system/developer content, and the
first real user task. The key retains a CRC32 prefix and a full SHA-256 payload
digest, so equal CRC32 values for different payloads remain isolated. Without
an explicit key, byte-identical canonical payloads cannot be distinguished as
independent conversations and intentionally share the fallback identity.

Call IDs never create a new conversation. They are accepted only as bounded
learned aliases to a previously established prompt-cache or canonical identity;
unknown or ambiguous tool-only requests fail closed. QWEN-EXO may restore the
latest successfully finalized MemoryPipeline/native-attractor state internally,
but `request.started.parent_response_id`, API `previous_response_id`, response
store lineage, and execution-capsule parents remain null unless the client sent
an explicit parent or a verified compaction envelope supplied one.

Calls that omit `reasoning` use the server default chat-template kwargs
`{"enable_thinking": true, "preserve_thinking": true}`. Passing
`{"reasoning":{"effort":"none"}}` is an explicit per-request opt-out and
overrides the default for that request.

This Qwen chat template exposes a boolean thinking switch, not distinct
`low`/`medium`/`high` depth levels: every non-`none` effort only means enabled.
Responses clients that render OpenAI reasoning-summary events should send
`{"reasoning":{"summary":"auto"}}` without an effort tier. For OpenCode, use
`@ai-sdk/openai`, mark the model `reasoning: true`, and set the model option
`reasoningSummary: "auto"`; `@ai-sdk/openai-compatible` selects Chat
Completions instead of the Responses endpoint.

Thinking summaries, structured function tools, streaming events, response
retrieval, and background cancellation remain the upstream Responses API
contract. QWEN-EXO private memory and capsule text are not added to user-visible
response items.

## Runtime endpoints

### `GET /qwen-exo/health`

Returns HTTP 200 only when the QWEN-EXO runtime is ready; otherwise HTTP 503.

### `GET /qwen-exo/status`

Returns the resolved runtime contract, including:

- model fingerprint and TP geometry;
- Full-Attention/GDN state policy;
- knowledge source digest and document count;
- enabled internal services;
- observer mode and trace event count;
- privacy and source-repository posture.

## Telemetry

### `GET /qwen-exo/telemetry`

Query parameters:

| Parameter | Meaning |
|---|---|
| `request_id` | Optional exact request filter |
| `limit` | 1–1000, default 256 |

```bash
curl 'http://127.0.0.1:30000/qwen-exo/telemetry?limit=100'
```

Each event has `event_id`, per-request `sequence`, `request_id`, `event_type`,
Unix `timestamp`, and a typed `payload`.

### `GET /qwen-exo/telemetry/stream`

Server-Sent Events stream. Use `after=<event_id>` for an incremental resume or
send the standard `Last-Event-ID` header. Events use type `trace`; idle
connections receive SSE comments as keepalives.

Default telemetry is redacted. Sensitive values are replaced with:

```json
{
  "redacted": true,
  "sha256": "...",
  "bytes": 123
}
```

## Knowledge API

Knowledge sources are UTF-8 `.md` or `.markdown` files rooted below the
configured knowledge directory. Path traversal and non-Markdown suffixes are
rejected.

### Public metadata

```bash
curl http://127.0.0.1:30000/qwen-exo/knowledge
```

Returns the source digest and document metadata without content.

### Direct control-plane access

Content reads and mutations are available directly from the same control-plane
origin; they do not require a bearer token. Keep the service bound to loopback
and expose it only through an operator-controlled SSH tunnel or trusted reverse
proxy.

### Read content

```bash
curl http://127.0.0.1:30000/qwen-exo/knowledge/runbooks/network.md
```

### Create or replace atomically

```bash
curl -X PUT http://127.0.0.1:30000/qwen-exo/knowledge/runbooks/network.md \
  -H 'Content-Type: application/json' \
  -d '{"content":"# Network runbook\n\nVerified operational content."}'
```

The repository writes through a temporary file, `fsync`s it, atomically
replaces the target, and refreshes the source snapshot.

### Rebuild an independent source index

```bash
curl -X POST http://127.0.0.1:30000/qwen-exo/knowledge/reindex
curl -X POST http://127.0.0.1:30000/qwen-exo/policydata/reindex
```

The two routes rebuild separate snapshots; neither route mixes PolicyData with
Knowledge.

### Rebuild the model-native Tensor Bank

After an authoritative Knowledge or PolicyData change, rebuild the persisted rank artifacts and atomically publish the replacement snapshot:

```bash
curl -X POST http://127.0.0.1:30000/qwen-exo/tensor-bank/reindex
```

The response reports the source/model digests, logical page count, FP8 storage type, physical page size, selectable prefix width, Full-Attention K/V coverage, and complete GDN boundary coverage. The route fails closed if any TP rank artifact is absent, stale, or incomplete.

For an eligible PolicyData candidate, the request metadata reports
`policy_data.injection_mode=native_full_attention_kv_and_gdn_section_delta`,
`policy_data.text_attached=false`, and the exact native page identity. The
server does not append PolicyData source or compiler padding to `instructions`.
The aligned hidden prefix is a radix-cache identity for atomically restored K/V
and GDN state. If the artifact is stale, unavailable, over budget, or cannot be
aligned, PolicyData fails closed instead of falling back to text.


### Delete

```bash
curl -X DELETE http://127.0.0.1:30000/qwen-exo/knowledge/runbooks/network.md
```

## User workspace

Open the Chinese Responses workspace:

```text
http://127.0.0.1:30000/qwen-exo/
```

It streams answer and reasoning events from `POST /v1/responses`, continues a
local conversation with `previous_response_id`, and renders the matching
PolicyData, Knowledge, and observer recall trace after each completed turn.


## Operations console

Open:

```text
http://127.0.0.1:30000/qwen-exo/admin
```

The Chinese console provides:

- runtime/model/hybrid-state posture;
- live redacted SSE trace rail and request inspector;
- selected-token surprisal, Q drift, and memory-energy proxy chart/readouts;
- separate PolicyData and Knowledge metadata, direct Markdown import, content
  editing, atomic save-and-compile, reindex, and delete actions;
- resolved system contract and feature flags.

The console opens directly without a token prompt.

## Capacity errors

A cross-rank admission failure returns an abort reason compatible with HTTP 429:

```json
{
  "type": "abort",
  "status_code": 429,
  "code": "qwen_exo_capacity_exhausted",
  "message": "QWEN-EXO capacity admission failed: kv_capacity",
  "retry_after": 1
}
```

The message suffix identifies `kv_capacity`, `request_slots`, `mamba_slots`, `workspace_capacity`, or `peer_rank_capacity`. `workspace_capacity` includes the live per-rank safety reserve for transient FP32 logprob tensors.

Clients should apply bounded backoff and retry the entire request. They must not
retry hidden child jobs directly.
