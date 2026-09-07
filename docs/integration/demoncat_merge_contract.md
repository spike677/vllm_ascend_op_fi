# DemonCAT merge contract

## Purpose

This contract keeps the isolated DemonCAT product framework and this Ascend
backend research repository composable without creating parallel registries,
controllers, target schemas, or production CLIs.

## Product-owned responsibilities

DemonCAT owns:

- the user-facing `position + primitive + target + params` contract;
- `PrimitiveSpec` and the semantic primitive catalog;
- target normalization and validation exposed to users;
- `RuntimeBinding` selection and environment policy;
- `FaultHandler` dispatch, controller lifecycle, authorization, and recovery
  policy;
- the production CLI and service API.

These concepts must not be reimplemented as an independent production stack in
`vllm_ascend_op_fi`.

## Backend-repository responsibilities

This repository owns:

- execution-plane backend implementations and native interposers;
- mutation strategies that operate on already bound targets;
- read-only observers and event formats;
- discovery/fingerprinting utilities for environment-specific physical
  bindings;
- baseline/fault/recovery evaluation helpers;
- capability manifests, build/ABI checks, tests, and scoped validation
  evidence.

Development CLIs may trace, probe, validate, or benchmark. They must not become
a competing production injection controller.

## Capability handoff

`profiles/capabilities/*.json` is the intermediate protocol. A manifest states:

```text
semantic compatibility
  + execution_plane
  + mechanism
  + observe/inject/restore support
  + physical selector capabilities
  + canonical implementation/build references
  + validation boundary and limitations
```

DemonCAT may translate a manifest into one or more `RuntimeBinding` entries. It
retains authority over whether a capability is exposed, which parameters are
public, and how request policy is enforced.

The manifest is descriptive metadata, not a live registry. Loading a manifest
must not arm or execute a fault.

## Binding boundary

DemonCAT provides a logical request and policy. The selected backend resolves
timing-local physical state:

```text
PrimitiveSpec / normalized target       (DemonCAT)
                │
                ▼
RuntimeBinding and policy               (DemonCAT)
                │
                ▼
logical capability request              (contract boundary)
                │
                ▼
symbol/tensor/storage/stream binding     (Ascend backend)
                │
                ▼
mutation or observation                 (Ascend strategy/observer)
                │
                ▼
evidence/result and recovery status      (contract boundary)
```

Raw call indices, addresses, cache pointers, and allocation generations are
backend evidence and binding inputs. They do not become stable product target
fields unless DemonCAT explicitly adopts a normalized semantic form.

## Lifecycle and safety requirements

Any injectable backend integrated into DemonCAT must:

- remain disabled by default and require the product's developer/test gate;
- validate ABI, library version, architecture, symbol, tensor metadata, and
  target relation before mutation;
- fail closed on ambiguous binding, unsupported combinations, copy/sync error,
  re-entry, or stale allocation state;
- expose bounded hit policy and deterministic disarm behavior;
- report mutation, consumer ordering where applicable, and exact recovery;
- never infer a stronger root-cause claim than the recorded evidence supports.

## Compatibility and versioning

- Capability manifest `schema_version` versions metadata syntax.
- Native shared-control and metadata ABIs retain their own magic/version/size
  handshakes.
- DemonCAT bindings pin accepted capability ids and ABI ranges.
- A CANN/vLLM/torch_npu/model combination is unsupported until its manifest or
  profile records a passed validation boundary.
- Legacy experiment sources remain canonical until the migration gates in
  `docs/architecture/repository_layers.md` pass.

## Merge acceptance checklist

Before registering a backend in DemonCAT:

1. the capability manifest points to one canonical implementation;
2. build and ABI verification pass in the target inference image;
3. disabled/no-hit behavior is output transparent;
4. bounded fault behavior has auditable hit evidence;
5. recovery is exact or an explicit restart requirement is reported;
6. observer-only code cannot change selection or mutation decisions;
7. model, device, CANN, vLLM, topology, and request limitations are recorded;
8. the integration adds no second production CLI, primitive registry, or
   controller lifecycle.
