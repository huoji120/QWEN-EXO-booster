---
canonical: true
quality: 0.9
source_kind: curated_reflection_memory
document_group: reflection_memory
tags: [reflection-memory, tooling, failure-analysis]
---

# Separate tool failures from domain evidence

A tool invocation, transport, serialization, permission, or environment failure does not prove that the domain hypothesis is wrong. First classify the earliest failing boundary, repair the invocation, and repeat the same minimal probe.

Decision rules:

- A malformed argument or rejected payload is evidence about the call contract, not the target behavior.
- A connection or process failure is evidence about reachability or runtime state, not semantic correctness.
- A valid response with the expected request applied is domain evidence and may confirm or falsify the hypothesis.
- A timeout is unresolved unless independent evidence identifies what continued running and where.

Preserve the original question while debugging the tool path. Do not pivot to a different target or strategy after one formatting error. Pivot only after a correctly executed observation makes the current explanation less likely or proves the required capability unavailable.
