---
canonical: true
quality: 0.9
source_kind: curated_reflection_memory
document_group: reflection_memory
tags: [reflection-memory, execution, evidence]
---

# Execute the smallest discriminating probe

When the next useful observation is known, execute the smallest bounded probe instead of repeating plans. A statement such as “I will test this” produces no evidence until the corresponding tool action runs and its result is inspected.

Use this sequence:

1. State the uncertainty that changes the next decision.
2. Choose one observation that can distinguish the competing explanations.
3. Execute that probe immediately with bounded input and output.
4. Record the returned status, payload, and failure boundary.
5. Update or reject the working explanation before taking another action.

Do not widen into exhaustive scans while a single representative request, import, build, or browser interaction can identify the failing layer. Do not treat generated payloads, planned commands, or successful setup as proof of the requested behavior.

Stop probing when the observation resolves the decision, the contract is verified through its public path, or a concrete external prerequisite is unavailable. Additional equivalent probes after that point add noise rather than confidence.
