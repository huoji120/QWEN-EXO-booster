---
canonical: true
quality: 0.95
source_kind: software_engineering_reference
---
# Python API Propagation Surfaces

This reference maps how one Python behavior propagates through definitions, public exports, type declarations, wrappers, constructors, decorators, registries, serializers, generated factories, sync and async variants, defaults, and cache keys. Recall it when an implementation works in one function but may be missing from another Python API surface.

Common surfaces include:

- the authoritative class, function, or generated implementation;
- public package exports and type declarations;
- convenience functions and wrapper methods;
- constructors, decorators, registration functions, and factory helpers;
- serialization or schema generation;
- sync and async variants;
- generated-code factories and their cache keys;
- focused tests plus compatibility tests.

A reliable navigation sequence is to locate the public symbol, inspect its definition and references, identify the nearest behavior tests, and then enumerate wrappers that reproduce its signature or forward its arguments. Existing code generation should normally be extended at its source rather than patched after generation.

When a feature changes defaults or precedence, each layer that can supply a value must distinguish an omitted value from an explicit value. Repeated inclusion, nested composition, and explicit overrides are useful boundary cases because they reveal accidental early resolution.

When a feature changes partial or failure behavior, inspect how defaults, required fields, nested objects, collection atomicity, validation detail, and public error types are represented before choosing a data model.
