---
canonical: true
quality: 0.95
source_kind: curated_reflection_memory
document_group: reflection_memory
tags: [reflection-memory, retrieval, admission, telemetry]
---

# Keep retrieval, review, and injection evidence separate

A retrieved candidate is not an admitted memory. Treat the recall pipeline as distinct evidence boundaries:

1. Query capture proves only that a usable request representation exists.
2. Tensor or lexical ranking proves only that candidates were proposed.
3. Semantic review proves whether each presented candidate was eligible, rejected, or invalid.
4. Selection proves which eligible document, if any, won the request-level admission decision.
5. Native-prefix restoration or private text attachment proves what actually entered model execution.
6. A next-turn attractor is future state and is not evidence of current-turn injection.

Report counts at their owning boundary. Never label proposed candidates as injected, never interpret a top score as semantic approval, and never combine PolicyData, Knowledge, Cognition, Self-Ask, or future-turn restoration into one ambiguous “memory used” number.

When a review input exceeds its bounded context, preserve the original task and latest evidence, record original and reviewed token counts, and execute the review on that bounded representation. Do not silently convert an overlong question into a zero-execution result.
