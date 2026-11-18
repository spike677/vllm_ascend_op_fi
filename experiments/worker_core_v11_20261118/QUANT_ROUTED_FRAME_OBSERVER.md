# Bounded routed-call frame observation

`quant_route_frames.py` implements the source-bound portion of
`QUANT_ROUTED_CHAIN_NEXT_STEP.md`. It observes existing calls and does not bind
request rows, admit faults, call a model twice, or read tensor values. The
existing integer observer still owns its separately counted integer copies.
This change does not add another index-only probe.

The observer is enabled by the existing controlled inventory option
`--group-abi --capture-integer-routes`. Ordinary C1/C2/C3 admission is unchanged.
The new record scope is
`PYTHON_FRAME_OBJECT_SCALAR_EPOCH_OBSERVATION_ONLY`; even
`PYTHON_OBJECT_CHAIN_COMPLETE_NATIVE_UNVERIFIED` keeps
`native_registration_proven`, `routed_output_row_proof` and `native_qualified`
false. It is not an alternate row-proof authority.

## Evidence and refusal

Each actual DeepSeek MoE frame receives a bounded record. It retains current
typed prepare/dispatch/builder/MLP objects, source/code/globals identity, actual
quantization flags, and observed return objects. The current production profile
accepts the BaseDeviceAdaptor W8A8 fusion branch for observation; A5/MXFP,
dynamic EPLB, replacement payload classes, group-list substitutions, missing
frames and nonidentity prepare/finalize remain explicitly refused.

GMM1, GMM2 and unpermute observations are joined to the integer record by actual
parent sequence, forward identity, record hash and the enclosing source frame's
actual operator sequence list. Descriptor equality never substitutes for an
unobserved call. Consumed routed temporaries may be retired by the original
model after consumption; they are not reread as live producers later.

For the actual post-route `mul_` and `add_`, the observer records current schema
arguments and their positional/keyword/default origins. The scaling operand
must be an observed finite Python scalar belonging to the current unchanged
model configuration. A tensor scalar remains refused and is never read back.
The addition operand must be the actual shared return, with the actual `alpha`
recorded. The existing conservative write invalidation remains active. Reports
retain all affected storage epochs and require the target's observed epoch to
advance exactly once with unchanged extent and exactly one original call.
These checks do not grant either operand a request-row proof. The last actual
view must connect the same routed target to the model return.

Limits are 32 parent calls per forward, 48 source frame pairs per parent and
512 KiB of charged metadata per parent, including return packets. Budget refusal
retains the parent/overflow count and lets each original call run once. No
temporary profile owner, dispatch mode, stream or device allocation is added.
Strong tensor/payload references last only through the enclosing model frame;
normal and exception unwind release them. The existing source observer refuses
to end while a routed frame remains live. Serving reports now expose
`owned_routed_frame_parents`, `owned_routed_frame_tensor_holds` and
`owned_routed_frame_payload_holds` separately. Retained JSON evidence is not
called zero host memory.

The observer records its callback duration. That timing excludes shared profile
and graph overhead and must not be added to overlapping timing categories or
represented as an observer-free baseline.

## Exact source closure and deployment

The added production source bundle is
`a7dc26af878b8a86f5c9c58ae8a886c5cd58e6a0c6f346c562fa4d7b0f6d0c9d`
(24,469 bytes). Its three text hashes were checked locally:

| Source | SHA256 |
| --- | --- |
| `vllm_ascend/device/device_op.py` | `fe5262b76b3c4841d735b6c06c0fe02035aeecc8562f81e78b07d0e691c6cd17` |
| `fused_moe/moe_stage_contracts.py` | `f9ae3e9d5d55e7c172fb7d7a3f93856447514ebc9bf60c2213eb77805357cdd8` |
| `fused_moe/moe_stage_params.py` | `77faab96fc89478528b18c12ba18b044d16c2d2e5bd16b0ecf0946373fcf2555` |

The prior MLP/argument/finalizer and DeepSeek source hashes remain pinned in
`REVIEWED`. Typed payloads require the current exact frozen slotted classes.
The existing compiled-code check now also resolves classmethod functions and
property getters; their code and globals checks are preserved.

Source-only exporters must include the new deployment module
`demoncat.runtime.vllm_ascend.worker_extension.quant_route_frames`. Both quant
probe setup source lists now hash that loaded module. The other changed runtime
files are `quant_source_rows.py` and
`serving_shared_scale_group_inventory.py`. No Native source, ABI or SDK bundle
changed. Root must freeze/install the new Python source before a fresh run.

The existing root-controlled case can use the unchanged reviewed Native build:

```text
python -B /source/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py --group-abi --capture-integer-routes --build-sha256 b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac
```

It retains the same model/prompts, TP1, two B8 first-decode observations, health
checks and complete 135-projection denominator. No fault is scheduled. This
command is a handoff only; no container or NPU was started by this change.
The source87 offline audit must continue using `--source-root` pointing to its
frozen source87 runtime, rather than this changed checkout.

## Remaining Native gates

The actual loaded implementations of routing, GMM1, GMM2 and unpermute still
need authenticated ownership and reviewed semantics. In particular, cumulative
versus count grouping, row-local GMM/SwiGLU/quantization, and inverse-index plus
probability-reduction ordering remain separate acceptance conditions. The
current active dispatcher dump and the newly reviewed Python delegates do not
prove those facts. Every record carries these unresolved gates; no mutable or
unknown path is converted into a supported Native target.

CPU contracts use real PyTorch dispatch/profile callbacks and synthetic tensors.
They cover complete object/scalar/epoch chains, same-value replacement, payload
type substitution, current classmethod replacement, inference alias writes
without version counters, wrong scalar/config/add operand, final-view mismatch,
missing source frames, frame/byte/parent budgets, original exceptions, detached
reports and two inference forwards. They establish observer behavior only.
