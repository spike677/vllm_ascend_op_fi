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
  --source-commit "$COMMIT" --card 2 --out "$CASE"
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
