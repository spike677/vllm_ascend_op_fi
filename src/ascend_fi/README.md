# ascend_fi product-facing integration scaffold

This directory establishes the product-facing boundary for reusable Ascend fault-injection
backends, strategies, observers, discovery helpers, evaluation helpers, and shared
handoff contracts. It is an integration scaffold; no native implementation has been migrated here yet.

It intentionally contains no `Fault20`, `Fault26`, `Fault39`, or `Fault41`
package names. Fault numbers identify validation cases; they are not product
runtime abstractions.

The current native implementation remains the single canonical source under
`fault20_fast_validation/source/` until each component has an isolated build,
ABI test, and model-level regression in this structure. Capability manifests
under `profiles/capabilities/` point to those canonical sources during the
transition, so code is not duplicated and allowed to drift.

The current canonical development clone is D:\project\demoncatpfi. The
C:\Users\admin\Documents\demoncatpfi copy is a legacy/local investigation
clone and must not be used as a second source of truth.

## Ownership boundary

This repository owns:

- Ascend execution-plane backends;
- mutation and observation implementations;
- discovery and evaluation utilities;
- capability descriptors, tests, and validation evidence.

The isolated DemonCAT product owns:

- the production `position + primitive + target + params` interface;
- `PrimitiveSpec`, `RuntimeBinding`, and the production registry;
- controller lifecycle, policy enforcement, and the production CLI.

This repository must not introduce a competing production registry,
controller, or injection CLI.

## Planned package roles

```text
ascend_fi/
├── backends/       # execution-plane adapters and native bindings
├── strategies/     # data/value mutation primitives
├── observers/      # read-only evidence collection
├── discovery/      # logical candidate to physical binding discovery
├── validation/     # independent lossless target gate
├── evaluation/     # baseline/fault/recovery measurement
├── profiles/       # validated portable model capability output
└── contracts/      # JSON-serializable capability, candidate, validation, evaluation DTOs
```

`execution_plane` describes where an implementation binds (`aclnn`,
`torch_npu`, `vllm_ascend_backend`, `atb`, `hccl`, or `state`). It does not
replace DemonCAT's semantic `position`.

`contracts/` is a shared data boundary, not a runtime registry or dispatcher.
It carries evidence between discovery and evaluation while DemonCAT retains
ownership of product policy, binding selection, controller lifecycle, and the
production CLI.
