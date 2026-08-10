---
canonical: true
quality: 0.95
source_kind: software_engineering_reference
---
# Repository Evidence and Change Verification

This reference maps repository observations to the claims they can support and connects them to safe incremental change mechanics. Recall it when an implementation location, a failure explanation, or the strength of verification evidence is uncertain.

## Evidence strength

Strong behavioral evidence includes a focused regression that fails before a fix and passes after it, a compiler or type result, and a direct request or command that observes the public contract with its real exit status.

Structural evidence includes definitions, references, exports, wrappers, generated paths, configuration parsing, call chains, and persisted schemas. It locates ownership and propagation surfaces but does not prove runtime behavior by itself.

Weak evidence includes filenames without inspected content, generic comments, a clean working tree, a planned change, and repeated broad searches. These cannot replace a behavioral observation.

## Change and verification mechanics

- Identify the public entry point, authoritative owner, affected callers, and nearest behavior before editing. Generated behavior should be changed at its generator or factory source.
- Prefer a localized patch that preserves untouched bytes and established error identity. Broad replacement is justified only when the current representation cannot express the required behavior.
- A temporary reproduction is useful only when it observes the requested public behavior. It does not replace integration or the repository's normal regression location.
- Placeholder methods, unconditional fallbacks, alternate entry-point exceptions, and unwired exports are incomplete when the public contract requires real behavior.
- Run focused verification after the first coherent edit. Expand to nearby regression coverage only after the target behavior works.
- Preserve actual selection and exit status. Empty output, a timeout, an unrelated suite, or a missing target remains unresolved.
- Branches, commits, patches, submissions, and reports are observable delivery artifacts when requested; test success does not produce them automatically.

For multi-requirement work, keep a small map from each public value, default or precedence rule, failure mode, generated path, concurrency boundary, and delivery obligation to one observation that could disprove it. The earliest failing boundary is usually more diagnostic than the largest available suite.
