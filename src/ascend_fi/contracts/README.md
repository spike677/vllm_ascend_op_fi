# Shared contracts v0

These dataclasses are JSON-serializable handoff objects shared by capability
research and evaluation/discovery work:

- `Capability`: validated or experimental low-level implementation metadata;
- `Candidate`: an observed, environment-bound target discovered for a
  capability;
- `ValidationResult`: the independent lossless gate result for a candidate;
- `EvaluationResult`: baseline/fault/recovery mechanism and model-effect data.

The contracts use only the Python standard library. Unknown JSON fields are
preserved in `extra` and emitted again during serialization, allowing additive
schema evolution.

They are data contracts only. They do not register primitives, expose a
production CLI, select product policy, install hooks, or arm faults. DemonCAT
continues to own `PrimitiveSpec`, `RuntimeBinding`, `FaultHandler`, controller
lifecycle, and the production interface.

An unknown candidate is not eligible for automatic destructive injection.
`ValidationResult.destructive_injection_allowed` becomes true only when the
candidate is resolved, the lossless attempt passed, exact restoration is
confirmed, and status is `LOSSLESS_VALIDATED`.
