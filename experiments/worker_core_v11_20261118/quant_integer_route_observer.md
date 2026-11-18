# Bounded integer routing observations

This change adds an opt-in, read-only census to the existing natural B8 decode
observer. It does not bind a routed tensor to request rows, qualify a Native
target, grant C1/C2/C3, or change the shared-scale fault. All 135 projection
candidates and their existing unresolved reasons remain in the census.

The production source bundle SHA256 is
`90f151c5fbc830748b0d8cdc04aab32339810b8f48c61d2a08bb08cf3672e264`.
Its 153 embedded files were independently checked against their recorded
SHA256 values. The observed Python sources are pinned in
`quant_token_routes.REVIEWED`; their loaded code/globals are checked by the
existing `SourceBoundRows` implementation.

The reviewed `moe_custom_common.h` declares GATHER=0 and SCATTER=1.
`moe_custom_full_load_base.h:385` and
`moe_custom_row_idx_gather.h:185` write the inverse expert-sort mapping for
row_idx_type=0: an original expanded slot indexes its sorted destination.
`moe_custom_gather_dynamic_quant.h` divides the original slot by K to find
its input token. The full-range, no-drop profile therefore checks the actual
inverse permutation, expert IDs, counts, and cumulative counts. It permits
different orders within one expert; it does not assume a stable sort.

The source `TokenDispatcherWithAllGather.token_dispatch` requests count mode
and `token_combine` passes `abs(expanded_row_idx)` to unpermute. Negative values
can be sentinels in other routing branches. This profile rejects every negative
value; taking its absolute value does not make it valid. It requires the actual
current call to have B8/K6/E64, active_num=48, row_idx_type=0, quant_mode=1,
count mode, full expert range [0,64), no padding/drop, no input scale/offset,
and the checked untransformed AllGather source frame. Actual scalar arguments
and whether they were positional, keyword, or omitted current-schema defaults
are recorded. No historical default is inferred from shape.

The bundle does not authenticate the loaded custom Native registration, nor
contain the grouped-GMM/SwiGLU and unpermute Native semantics. Complete raw
current schemas and dispatch dumps are saved, and a shadowed execution-key
registration is refused. `native_registration_proven` remains false even when
the current dump contains no shadow. `TokenRouteProof` is an immutable
arithmetic result, not an admission capability. Its 48 entries identify
source-token indices and route slots; they are not 48 requests. The routed
output remains unproven until a separately reviewed actual full chain can
establish that fact.

## Capture and lifetime

For each observed parent, five integer-only snapshots are permitted:

| Snapshot | Layout | Bytes |
| --- | --- | ---: |
| Top-k expert IDs | int32[8,6] | 192 |
| Original expanded row indices | int32[48] | 192 |
| Expert counts | int64[64] | 512 |
| Actual cumulative counts | int64[64] | 512 |
| Actual absolute indices | int32[48] | 192 |

The limit is 1600 integer bytes per parent and 32 parents per forward. An
overflow is reported without dropping it from the parent counter. There is no
float activation readback. Scalar operator options and tensor descriptors are
metadata, not activation values.

Snapshots occur on the producer's current stack, before its consumer. Each
consumer must receive the exact retained object, unchanged descriptor/version
and observed write epoch. The first GMM consumption is completed before the
retired routed temporary is released; the actual later `set_(empty)` is not
joined by a reused address. All retained tensor references are cleared on
parent return or exception. A failed helper-mode exit retains its handle and
integer reference until exact-owner cleanup succeeds. Cleanup follows the
actual Torch mode stack; it never removes a foreign mode and is retryable after
the foreign owner removes its own mode.

The blocking `aten._to_copy` to independent CPU integer storage, and any CPU
`resolve_conj`/`resolve_neg` helpers used to read it, have a separate bounded
raw helper ledger. They are not projection or Native DynamicQuant parents.
Each helper records its actual original-call count, descriptors, failure,
cleanup, and elapsed time. The observer reports copy bytes, retained integer
payload bytes, peak owned CPU copy size, total/overflow helper dispatches,
source-profile callback time, and observer-hook time. Those times overlap
(readback is inside the observer hook); they must not be summed as disjoint
durations. Synchronizing the integer copy can change timing and allocation
lifetime. This does not promise identical ordering to an unobserved request.

## Root queue handoff

No new Native build is required. Keep the previously checked Native bundle
manifest SHA256
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`,
its exact existing model/image/cards/serial gate and fresh source/packages.
Append one opt-in flag to the existing zero-F two-healthy census command:

```text
python /source/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py --group-abi --capture-integer-routes --build-sha256 b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac
```

The existing reviewed case mounts `/model`, `/quant`, `/installed`, `/source`
read-only and only `/evidence` writable. Root owns its exact queue manifest,
lease, environment and invocation. This agent did not run SSH, a container,
SDK build, or NPU. The probe runs two actual B8 first-decode observations, no
fault owner and no statistics collector, then same-Worker healthy comparison
and complete cleanup. It is a controlled observation entry, not a public RPC
C2/C3 acceptance run.

The source-only exporter must add this one runtime dependency:

```text
src/demoncat/runtime/vllm_ascend/worker_extension/quant_token_routes.py
```

The changed existing closure is `quant_source_rows.py`,
`serving_shared_scale_group_inventory.py`, and the inventory/C1 probe source
hash lists. Capture defaults to false in `SourceBoundRows.from_runtime`, the
group inventory session/installer, and the probe. The flag is rejected before
framework/model import unless `--group-abi` is also present. C1 does not enable
it, so its statistics and fault path do not acquire these new helper calls.

Raw route records live under each forward's
`row_provenance.source_boundaries.integer_routes` (the exact existing outer
row-trace nesting is retained). Each parent graph event has an index/hash
reference to its detached route record. Results with missing calls, wrong
objects/epochs, unknown sources/options, unproven registration, failed copies,
or budget overflow retain those facts and cannot upgrade routed row coverage.

## CPU contracts

The tests use actual CPU Torch Library operators, dispatcher modes, checked
Python source frames, and inference-mode tensors. They verify 1600-byte
positive capture, tie-order independence, retired-storage behavior, negative
indices, wrong objects, versionless writes, invalid counts/digests/forward IDs,
copy and original-call failures, real Library override, parent overflow,
training refusal, immutable report snapshots, and failed/foreign mode cleanup
with same-owner retry. They do not emulate or certify CANN numeric results.

Local validation: 191 focused quant/source/MLA/MoE/C1/cleanup/probe tests passed.
The 24 new integer-route tests are CPU fixtures only. After the final helper
ledger/accounting changes, all 61 route/shared-MoE/source-boundary/probe tests
passed again. Production Torch 2.9 and natural NPU readback acceptance remain
pending root's fresh-source run.
