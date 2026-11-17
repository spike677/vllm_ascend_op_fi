# Fault26 independent native writer census

This is a new read-only inventory, not a repetition of the selected-writer C1
probe and not C2/C3 acceptance. The two supported cases are DeepSeek W8A8 QMM
writers and SmolLM2 FIA destinations. Each observes two natural, identical,
single-request fresh prefills in one actual standard Worker, with fixed 64 KV
blocks. It makes no decode or continuous-batch coverage claim.

The bounded native trace records all OpCache runs before selected-owner or fault
admission, typed QMM registration and writer execution, FIA before-prepare/GWS/
execution, and actual AddRms producer/consumer executions. Python retains the
actual public residual tensor until its next real consumer, joins complete
native/public shape, stride and byte geometry, and records actual owning module,
request, process lifetime, cache registration, and relevant allocator segments.
Every native writer remains in the report if it lacks a callback, public
producer, adjacent live allocation or actual consumer. Counter overflow or
missing detail is an error, never a smaller denominator. No native arm flag is
set and every callback continues once. Two whole-vocabulary hashes must match;
all hooks/callbacks/references must be removed before same-process control
generation. Native-qualified targets and C1/C2/C3 remain zero/false.

Prerequisites: root must build the changed trace/proxy sources plus
`fault26_inventory.h` through the existing native build profiles. The native
commits `d7d1357` and `bf293ca` are one dependency set. Verify these additional
trace DSO symbols using `nm -D`: `dcat26_inventory_state_v1`,
`dcat26_inventory_read_v1`, `dcat26_inventory_fia_before_v1`. Event ABI is 416
bytes; State ABI is 160 bytes. The fixed capacity is 32768 events, with explicit
overflow/error counters. Native host-only contract passed on 124 in
`/home/17-fault-c2c3-20261117/providers26-cpu-a1/receipt.json`; this is not NPU
evidence. Python CPU tests use explicit native mocks and real CPU modules.

Root staging and review commands (replace variables with immutable actual
source/build/packages paths; do not reuse old native bundles):

```sh
python3 "$SOURCE/experiments/engineering_20261117/stage_native_bundle.py" \
  --source "$SOURCE" --build "$BUILD26" --out "$BUNDLE26" --kind 26
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_inventory_case.py" \
  --installed "$INSTALLED" --native "$BUNDLE26" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_inventory_probe.py" \
  --source-commit "$COMMIT" --mechanism 26.residual_stomp --card 2 --out "$QMM_CASE"
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_inventory_case.py" \
  --installed "$INSTALLED" --native "$BUNDLE26_FIA" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_inventory_probe.py" \
  --source-commit "$COMMIT" --mechanism 26.fia_destination --card 5 --out "$FIA_CASE"
```

Only root acquires the global resource queue and starts either case. All source,
model, package and native mounts are read-only. Evidence remains on 124. The case
contains actual `build.json` SHA, model, CANN8.5.1/Ascend910B4, cards, rank,
1800-second/80GiB/16CPU budget and recovery steps. The Worker verifies all staged
native file hashes before attaching the already-loaded observer ABI. Result
summaries expose actual writers and completely joined lifetimes separately.

The next C2 adapter must add native per-target leases binding owner/generation,
thread/device, exact cache registration and actual output/residual allocations,
with one-shot continuation tokens, lossless qualification, budget and stale
rejection. At the real next AddRms consumer it must validate the same residual
lifetime, synchronize, restore/read back every changed span, then release. A
shared all-stop/all-drain barrier must precede any group release. Targets whose
fresh-request allocations differ cannot reuse an old address or producer
sequence as a lease. The actual census determines which identities can be
requalified per request and which remain unresolved; no structural module count
or broad `ALL_HITS` environment flag supplies that proof.
