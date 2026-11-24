# Fault33 arch22 S2 build candidate and live page binding

This increment prepares an independent `tnd_page16_arch22_s2_256` package. It
does not open the public Worker's F gate. No Native build or device execution
was performed by this agent. It grants no C1, C2, C3, or E5 qualification.

The source88 normal pair is retained as the reason to develop this route:
its independent report SHA is
`4c111ce4d6a91aa9df7a64ccc456ef8ec32b395d56dabbf12d9c37fd5649868e`,
with 281 checks, zero issues, and 435 exact cross-arm complete payload pairs.
It is not a normal certificate for this new package. Neither the old
67-element FAI FD difference nor the source87 Attention28 census is reused as
an admission proof.

## Exact implementation change

`apply_tnd33_arch22_s2.py` starts from pristine official commit
`c77c0829f1f0f0b8f882db1d9b051a758b605472`. It retains the reviewed normal
arch22 route preference. In `FiaTilingNonQuant::CalcInnerSize`, selection occurs
immediately before the original `sInnerLoopTimes_` calculation. Only an owned
`arch22_s2_256` call selects S2=256 instead of the observed S2=512.

The subsequent original tail/alignment, M selection, `CreateSplitInput`,
`SplitCore`, `SetSplitOutput`, FD distribution, workspace sizing,
`GenTilingKey`, and serializer remain intact. TND M remains 512. There is no
kernel arithmetic edit, copied Q/K/V operand, output mutation, fake launch
key, forced block dimension, or old FAInfer FD ABI.

The Native selector requires all of:

- A 32-hex call token that matches the independently loaded observer's actual
  TLS scope through `dcat_attention_observer_scope_matches_v1`.
- A 32-hex owner, positive generation, current PID and OS native TID, matching
  phase/mode, and target `model.layers.0.self_attn.attn`.
- The actual reviewed nonquantized FP16 TND Q `[1,16,128]`, flat page16 KV,
  Hq=16/Hkv=8/D=128, int8 mask `[2048,2048]`, and int32 page table.
- Actual KV length 1025–2048, page-rounded native capacity, native S2=512,
  AIC count 16–26 and AIV/AIC ratio 2; no optional quant/rope/prefix/padding,
  sliding-window or sink path.
- Actual parser-owned sequence tensor joined by identity to the context,
  and canonical native sparse3 window `(INT_MAX,0)`.

`arch22_s2_off` permits only baseline/baseline_repeat/off/recovery_off phases
and keeps S2=512. Unscoped calls use the fixed normal route without selecting
256. The selector rechecks the owner and actual TLS scope when the original
host serializer is captured. Environment variables alone cannot enter the
selected path without the observer's matching live scope. Trusted execution
code must set and remove these fields around exactly one original call; an
RPC never accepts environment fields or arbitrary callbacks.

Eleven actual cached source files were SHA-verified locally, including the
attention parser, info/index headers, split-core implementation, and both
kernel consumers. Applying the recipe in memory changed exactly the two
nonquant tiler files and the existing route-preference file. The full fresh
source tree is additionally pinned by its pristine Git commit. None of the
cached raw source/host/result files is committed.

## CPU gate in the fresh build

`compile_arch22_s2_matrix.py` compiles the *original* pinned `split_core.cpp`
and header with only `OP_LOGI/OP_LOGD` diagnostic macros disabled. It calls
the real `SplitCore` for every AIC count 16–26, actual KV length 1025–2048,
and S2 in `{512,256}`: 22,528 cases. It loads no SDK/DSO/device.

The separate Python checker reconstructs complete head/tile coverage from
the returned outer arrays, verifies every FD producer part and workspace
slot, every FD consumer endpoint and inactive arrays, and independently
calculates required workspace element counts. Matching those counts to a
real final Native POD remains an execution check. The matrix is a CPU
host-planning check; it is not a
device kernel correctness or real Worker platform receipt. AIC=20 in the old
host record was inferred from workspace arithmetic. This new host capture
records actual `aic`, `aiv`, and `libapi_workspace` instead.

Local validation: 18 new CPU contracts and 32 relevant existing normal,
source-build, key-registration and TND contracts passed (50 total, no skips).
They cover page relocation, actual object/storage joins, clone/slot/page/KV
negative cases, owner/thread/generation, exactly one original invocation,
failed-return retention, source closure and producer/consumer span failures.
The Windows environment has no C++ compiler. The actual 22,528-case source
execution and fresh SDK/Native compilation remain mandatory root build gates;
they have not been reported as completed.

## Root-only fresh build

After freezing a new source containing this commit, use a new output:

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$NEW_SOURCE/experiments/c2c3_closure_20261117/build_tnd33_124.py" \
  --source "$NEW_SOURCE" \
  --out "$R/tnd33-arch22-s2-build-a1" \
  --interfaces "$R/attention-preflight-a4/cann851/attention-preflight/sdk_interfaces.json" \
  --strategy arch22-s2
```

Use the exact interface-manifest bytes and path recorded as the readonly
`/sdk_interfaces.json` mount in the prior normal build's `controller.json`;
if that recorded path differs from the command above, use the recorded path.
The controller pins the existing CANN8.5.1 Worker image, clones pristine c77
source, uses the separately pinned dependency sources, and creates a
device-free build container. No prior vendor binary is an input.

Expected outputs under `tnd33-arch22-s2-build-a1/build/run`:

- `evidence/arch22_s2_matrix/{receipt.json,split_matrix.jsonl}` with the actual
  original-source CPU matrix and independent checks.
- `evidence/arch22_schema/{fia_abi_schema.json,compile_receipt.json}` from the
  new build's public serializer declarations.
- `stage/vendors/dcat_f33_arch22_s2_transformer`, its fresh host libraries,
  kernel metadata/ELF, and complete binary registration.
- `evidence/completion.json` with source/build/schema/matrix/ELF hashes and
  `fault_execution_ready=false`, `stock_private_exact_proven=false`.

The entry set retains the ten normal arch22/FAI warmup entries. It excludes
the old 590/600 FAI fault entries. A matching key never proves an ABI or
algorithm match by itself.

## Live logical input contract and remaining execution work

`paged33_logical_binding.LogicalPagedInput` owns references only. It does not
install hooks, start requests, load a library or acquire a public Core lease.
The trusted adapter supplies its actual Worker/model/module, forward context,
metadata, scheduler row mapping, and observed writer receipt. The helper
checks the context's exact module and metadata objects, the actual 3D input
views of the 4D caches, storage identity/offset/stride, page uniqueness/bounds,
writer/consumer slot, full query/mask/table, and complete valid logical K/V.
It verifies the same live input before and after invoking the original
closure once and preserves the original return object.

Comparison accepts two completed live objects under the same owner,
generation, thread, Worker/model/module and persistent caches. It records
`raw_input_exact` separately from `logical_input_equivalent`. Normalization
permits only validated used-page IDs, their full-table hash, and the derived
current writer/consumer slot. Unused table values, supplied-vs-omitted
attributes, query/mask/scalars, geometry and complete logical K/V must still
match. Different physical cache contents outside the valid logical prefix
are not asserted equal. Hash readbacks are source-bound; complete input bins
are not saved by this helper. It neither proves a Native launch nor authenticates
a serialized report as a live object.

Failures keep the first error and strong references until explicit same-owner
release. Foreign threads and in-flight release are rejected. Release touches
only these references; it cannot remove a Native owner, wrapper, or hook.

Next execution increment must bind this fresh package, exact source/version,
SDK/observer, actual final POD and computed split arrays to a new normal/OFF
pair before considering F. The existing normal adapter intentionally rejects
the new build variant, so its old certificate cannot silently admit it.
The eventual public F/OFF route must bind a fresh normal result to the exact
live first-layer logical input and owner/generation, preserve all raw input
differences, verify actual selected launch/complete output changes and OFF
recovery, and retain uncertain Native cleanup ownership. This commit changes
no public registry, Runtime, Host, exporter, existing proof or F gate.
