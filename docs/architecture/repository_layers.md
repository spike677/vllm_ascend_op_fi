# 已迁移：仓库分层

> 此路径为兼容旧链接保留。中文 canonical 文档已迁移至 [docs/overview/repository_layers.md](../overview/repository_layers.md)。本页不再作为独立的正式版本维护。

# Repository layer transition

## Decision

This repository is the NPU/backend research and validation source. The isolated
DemonCAT project remains the product framework and production control plane.

The canonical development clone for the current host is
`D:\project\demoncatpfi`. `C:\Users\admin\Documents\demoncatpfi` is a
legacy/local investigation clone. The current repository `HEAD` is
`31d8e365`; the architecture scaffold and manifests are working-tree changes,
not contents of that commit.

The repository transitions from Fault-number-centered organization to reusable
backend/strategy/observer organization without rewriting or moving the
validated native runtime in one step.

```text
Phase 1 architecture boundary complete
Physical migration NOT complete
```

```text
vllm_ascend_op_fi
├── src/ascend_fi/           reusable implementation/integration boundary
├── profiles/capabilities/   reusable capability metadata
├── experiments/             historical-case index and freeze policy
├── docs/                    architecture and integration contracts
├── fault15_scale_granularity/
└── fault20_fast_validation/ current canonical legacy source/evidence tree
```

## Layer ownership

```text
src/ascend_fi
= reusable implementation and integration boundary

experiments/faultXX
= historical experiment, configuration, and evidence

profiles/capabilities
= reusable capability metadata

closed DemonCAT
= final PrimitiveSpec, RuntimeBinding, controller, and product CLI owner
```

No `Fault20`, `Fault26`, `Fault39`, or `Fault41` package becomes a product
source hierarchy under `src/ascend_fi/`. This repository does not implement a
second position registry, primitive registry, RuntimeBinding registry,
production injection CLI, or controller lifecycle.

## Semantic position versus execution plane

`position` describes where the fault exists in the model/runtime semantics.
`execution_plane` describes where a concrete implementation binds.

```text
semantic position: kv_cache
        │
        ▼
primitive: kv_update_misdirect
        │
        ▼
mechanism: kv_cache_method_patch
        │
        ▼
execution plane: vllm_backend
        │
        ▼
downstream: torch_npu._npu_reshape_and_cache
```

A `kv_cache` semantic position may be implemented by a `vllm_backend`,
`torch_npu`, or `op_api` backend. Changing the execution plane does not change
the user-facing semantic fault. Sharing a downstream library does not make two
semantic positions the same primitive.

## Stable versus environment-bound fields

Stable integration metadata includes capability id, candidate semantic
positions, execution plane, mechanism, observation/injection/restoration
support, and validated environment constraints.

Physical fields such as symbol address, rank-local storage pointer, tensor
slot, call index, stream, allocation generation, and shape-specific cache key
belong to discovery or `RuntimeBinding` resolution. They are not promoted to a
stable user ABI merely because one experiment used them.

## Incremental migration gates

For each component:

1. record the existing canonical source and evidence in a capability manifest;
2. isolate backend, strategy, observer, and lifecycle responsibilities;
3. move one source of truth into `src/ascend_fi/`;
4. keep a compatibility build path from the historical experiment;
5. reproduce ABI checks and baseline/fault/recovery E2E;
6. update the manifest to reference the new canonical product source;
7. reduce the historical Fault directory to experiment parameters, evidence,
   hashes, and a pointer to the reusable implementation.

No mass move is performed before these gates pass. Physical migration starts
only after evaluation/discovery V0 consumes at least one historical capability
through candidate resolution, lossless validation, injection, and evaluation.

## Experiment placement

Fault41 is not yet a committed capability. Once its runtime and evidence are
frozen, its curated source belongs under:

```text
experiments/fault41_state_cache_corruption/
├── README.md
├── source/
├── harness/
├── evidence/
└── reports/
```

This keeps the historical case under `experiments/` instead of reintroducing a
Fault-number directory at repository root. Nested worktrees and temporary
scripts remain external/local state until explicitly curated.

## Shared contracts v0

`src/ascend_fi/contracts/` defines JSON-serializable DTOs for capability
metadata, discovered candidates, lossless validation, and baseline/fault/recovery
evaluation. They preserve unknown fields for additive evolution and are safe to
exchange between the evaluation/discovery and capability-research branches.

These DTOs are evidence handoff objects only. They do not register primitives,
select a `RuntimeBinding`, arm a backend, or establish a second controller
lifecycle. The product boundary remains owned by DemonCAT.
