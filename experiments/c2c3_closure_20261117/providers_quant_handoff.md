# Shared-scale all-native inventory and remaining row gate

The new V2 native observer records every actual typed GWS entry/return and
execution entry/return before selected owner/arm logic. Its 32768 fixed records
are separate from the old selected-call counters. Overflow, orphan execution,
executor registration collision, foreign thread and missing details are explicit
errors. An executor address reused after a completed call receives a different
registration lifetime. No registration is treated as a persistent cache.
The original C1 arm/finish API and numerical fault are unchanged.

`SharedScaleInventory` joins every native call to the actual public dynamic
quant input, Q and scale addresses, complete geometry, dtype, format, device,
thread and one original GWS/execute, then verifies the actual Q/scale storage
reaches the same module's QMM and projection result. All public and native calls,
including missing/unowned calls, remain in the denominator. The Worker probe
performs two identical B8 generations, observes each actual first pure decode
after a completed public prefill engine step, validates real scheduler rows and
full B8 vocabulary outputs, then removes owned hooks and generates again.

The existing real embedding→RMSNorm→same-tensor projection lineage can identify
request rows for projections sharing that input. Other projections remain
`ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN`; their shape is never used
as a request count. Healthy per-target input and row hashes are auxiliary
inventory evidence, not a numerical fault or E5 result. Qualified targets and
C1/C2/C3 remain zero/false. This explicitly improves the former public-only
135-module inventory without claiming the unresolved all-target contract.

Root-only no-device SDK build, using a fresh output directory:

```sh
docker run --rm --name c2c3-quant-inventory-build-a1 --network none --memory 8g --cpus 4 \
  --mount type=bind,src="$SOURCE/fault_injection/native_adapters",dst=/input,readonly \
  --mount type=bind,src="$SOURCE/experiments/p3p4_closure_20261117/build_quant_v2.py",dst=/build_quant.py,readonly \
  --mount type=bind,src="$BUILD",dst=/build \
  quay.io/ascend/vllm-ascend:v0.18.0 bash -lc \
  'source /usr/local/Ascend/cann-8.5.1/set_env.sh; exec python3 /build_quant.py'
```

The output bundle is `$BUILD/quant-v2`. No driver or device mount is needed.
The build preserves exact V2 header declarations, checks V1/V3/V4 are not
exported, verifies new inventory symbols and hashes the actual source,
`batch_scale_inventory.h`, controller, DSO and real opapi. The verified loader
requires matching header and build identities. Existing old DSO receipts are
not replaced with the new source hash. Native host contract passed at
`/home/17-fault-c2c3-20261117/providers-quant-cpu-a1/receipt.json` with zero
devices/NPU calls and Event/State sizes 128/128. The independent Python tests
use real CPU modules and explicit native mocks, never NPU evidence.

After the new SDK build and installed packages are reviewed:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --installed "$INSTALLED" --quant "$BUILD/quant-v2" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py" \
  --source-commit "$COMMIT" --image-id "$IMAGE_ID" --card 2 --out "$CASE"
```

Only root runs this case through the global resource queue. The manifest pins
the model, CANN8.5.1/Ascend910B4, package/source inputs, actual new build SHA,
rank/card, 1800s/80GiB/16CPU bounds, and cleanup. Inputs are read-only; raw
evidence remains on 124. No native arm function is called.

The next C2 implementation must first prove the remaining actual row-preserving
model/operator chains. It must then bind a `BoundaryTargetIdentity` to the live
model/module/site/method/build/process and geometry, and bind each fresh native
executor registration to that individual call lease. It needs explicit
owner/generation, per-target lossless/effect, unchanged original input and all
target OFF/drain/release. Missing row proof is an unresolved candidate, not a
supported exclusion or a reason to reduce the denominator. C3 requires the live
successful C2 proof in the same Worker.

## Actual row-preserving operator observation

The Python observer now starts an owned `TorchDispatchMode` only during each
actual first decode. The existing scheduler→embedding→RMSNorm identity supplies
the first seed. It binds each proven value to its weak tensor reference, actual
Storage identity/address/extent, offset, dtype, device, shape, strides and real
PyTorch version. Inference tensors explicitly report version `null`; observed
writes independently invalidate every overlapping physical storage range,
including aliases with distinct Tensor/Storage objects and independent version
counters. No activation is kept alive solely for this graph.

Only exact loaded ATen overload objects with captured actual schemas can
propagate row labels. Supported edges are explicit row-preserving copies,
feature-only reshapes/views/splits/concatenations and elementwise operations.
For binary operations, every row-shaped data operand must already prove the
same ordered request mapping. Unknown `[B,H]`, `[1,H]` or `[H]` tensors cannot
borrow row labels. A scalar has no request axis; a frozen parameter may broadcast
only while its actual module ownership, tensor identity, version/write epoch
and non-request broadcasting geometry still match. Buffers are not automatically
treated as static parameters. Unknown in-place writes, indexing, row `roll`,
cross-row transpose, matrix operations and opaque Attention/MoE outputs remain
UNVERIFIED, even when shapes and every value happen to be equal.

After the actual dynamic-quant Native input/Q/scale/executor binding and owned
public QMM/projection-output checks succeed, an existing input row proof may
continue to that exact projection output. This is a specific verified public/
Native chain, not a generic matrix multiplication whitelist. The next actual
projection may then use that output through supported operators. All Native and
public calls still remain in the independent denominator. An overflow or row
observer error adds an explicit scope failure rather than discarding calls.
The row operator trace includes the observer's read-only tensor readback/copy
operations on that thread; its total is not relabelled as a model-op count.
The separate Native/public census remains the projection denominator.

Thirty-two focused CPU contracts passed, including actual CPU model forwards
with explicit Native/scheduler fixtures. Two successive projections now have
four observed row mappings across two CPU decodes. Inserting an actual
same-shape, same-value row permutation preserves all four Native calls while
leaving only the two first-projection row proofs. These are fixture results;
they do not prove any natural additional target. Other contracts cover unknown
binary row data, conflicting request order, parameter mutation, aliases with
different Storage objects, missing inference version counters, stale versions,
object reuse, capacity loss and refusal to pop another dispatcher owner.
Row mappings and evidence are copied deeply at binding time. Every subsequent
lookup checks the frozen row digest and full node digest before reusing the
tensor proof. `query()` and `report()` return independent nested snapshots, so
mutating the original scheduler mapping or returned rows, metadata, node hashes
or parameter metadata cannot relabel an existing proof. Four additional
contracts cover these API boundaries and digest corruption detection; this is
not a promise to sandbox arbitrary code that edits private Python internals.

This increment changes only Python observation. The existing matched Native
bundle is `/home/17-fault-c2c3-20261117/build-quant-inventory-a1/quant-v2`, with
`runtime/build.json` SHA256
`086dbdf473a3e98e02870adb913f0386585873248489983da2d4d65fd6dcfc1e`.
Read-only inspection of the reviewed vLLM environment recorded immutable image
ID `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
The queue generator now requires an explicit immutable runtime image ID. It
does not rewrite or claim missing provenance in the older Native build receipt.
After publishing the new Python package, root can prepare the next read-only
case without launching anything:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --installed "$INSTALLED" \
  --quant /home/17-fault-c2c3-20261117/build-quant-inventory-a1/quant-v2 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py" \
  --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --card 2 --out /home/17-fault-c2c3-20261117/providers-quant-row-inventory-a1-cases.json
```

Natural all-projection row qualification is still blocked by actual data.
Attention needs the exact query-row→output-row relationship through its actual
layout/slot boundary. MoE needs full actual routing/permutation index values,
the expert row→request mapping and the matching restore relation, not only
producer object identities or equal row counts. The reviewed `moe_census.py`
and `moe_restore.py` joins, and
`/home/17-fault-c2c3-20261117/natural-moe8-a1/natural-moe8-census/probe/result.json`,
provide useful producer roles but do not yet establish those complete index
values for this quant scope. No Attention/MoE exemption is introduced.

There is also an explicit remaining Native execution gap after row proof:
`dcat_dq_arm/finish` carries TLS ownership and selected counters, not a frozen
group owner/generation/target/call lease or a per-target restoration receipt.
The C1 healthy-input equality and finite-input guards remain unchanged. A new
group ABI must bind each actual prepare/execute registration to its per-call
lease, preserve C1, distinguish healthy qualification from propagation inputs,
report real Q/scale effect and unchanged input, retain failed resources for OFF,
and support whole-group stop/drain/release plus same-Worker health before the
shared live C2 proof. The row observer neither calls arm nor grants that missing
authority; quant C2/C3 remain false until every unresolved row and this Native
contract are actually closed.
