# Quant group managed binding and legacy isolation

This increment adds the verified low-level Python binding. It does not install
Serving hooks, authorize a CoverageSession, qualify a natural projection, or
enable an NPU run. Unknown Attention/MoE row lineage continues to block the
complete candidate set. `native_qualified_targets=0`, `c2=false`, `c3=false`.

The initial binding matched Native source `072561ef`. Root compiled it
against actual CANN 8.5.1 in `native-quant-group-v1-a3`, with manifest SHA256
`9c243c468fe42c12cc2af091dfb8642f4c9249f6b842ee2810e3306916654f0e`.
The bundle is `/home/17-fault-c2c3-20261117/native-quant-group-v1-a3`.
This is SDK compilation and CPU contract evidence, not numerical or NPU
acceptance. The later empty-owner cancellation increment requires a fresh
bundle; see [its root build handoff](providers_quant_group_cancel.md).

## Verified loading

`QuantGroupNative.load()` checks the approved manifest digest, every bundle
file, the three production source hashes against the installed package,
actual SDK opapi path/hash, private group DSO, and 15 ctypes/actual DSO/SDK
structure sizes. It verifies the actual mapped symbol owners against the
selected library handle. No group is installed as a side effect of loading.
`query()` and `raw()` return detached snapshots. Callback failures return an
explicit error status; bounded diagnostic rows and overflow counts remain
visible, including exception-formatting failures.

The existing `quantization._LEASE` serializes managed mapping checks and
loading/arm operations. `quant_native_isolation` enumerates actual executable
mappings and uses `RTLD_NOLOAD`, `dlsym` and `dladdr` to verify symbol owner,
address range, file device/inode and content hash. It does not identify a mode
by basename or an environment toggle. Deleted/replaced executable mappings
are rejected. The new group rejects mapped legacy Native code. All three old
managed loaders reject group code before constructing a backend or loading a
DSO; old managed session admission and every arm recheck the exclusion.
The original C1 Native sources, libraries, numerical behavior and builders
remain unchanged.

This is exclusion at managed public entry points. It cannot prevent arbitrary
manual `dlopen` or direct legacy ABI calls. The Native group also performs its
own exclusion checks. Releasing a Python lock does not unload a DSO: a process
that loaded one mode cannot switch to the other; use a separate Worker.

## Ownership and recovery

The binding retains installed callbacks and their owner in a process-global
registry until Native removal succeeds. A failed removal retains those strong
references. A public invocation holds its actual input strongly, passes only
actual contiguous zero-offset NPU tensor metadata, and invokes the original
public operation once. The Native helper and effect calls stay separately
visible in the raw census. Public return validates the actual returned Q/scale
storage; it does not manufacture ownership of outputs before they return.

The input remains strongly held if any Native parent, prepare, executor,
quarantine, exception or orphan descriptor is unresolved. Python never asks
for late recovery when a public operation fails. The Native execute stack is
the only recovery window for borrowed output storage. A nonzero execute result
or a prepared-but-unexecuted executor cannot become a successful drain just
because the Python call returned. Exported recovery accepts no caller-made
lifetime witness. An unrecoverable returned-call failure retains resources
and requires Worker disposal; it cannot claim same-Worker healthy OFF.

Only a later provider may construct target/row identities and connect the
original live C2 proof to CoverageSession. These low-level control methods
provide no independent C2/C3 authority. The explicit empty-owner cancellation
API stops and drains only an owner that admitted no target. It cannot replace
the whole-group recovery barrier for an admitted group. Passive discovery can
also use the uninstalled raw observer.

## CPU validation and root handoff

Local tests cover all SDK layout sizes, callback retention and exceptions,
failed-public-call input retention with no late recovery, exactly one original
public invocation, detached receipts, pending removal, CPU-tensor rejection,
legacy/group exclusion, concurrent check/load and failed-loader lock release.
These tests use explicit Python doubles; they do not prove real Native effects.
The focused local run completed 61 unittest cases: 60 passed and the actual
Linux ELF case was platform-skipped on Windows. The Linux case was separately
run on the host as described below.

The Linux host-only test compiled tiny marker ELF fixtures with arbitrary
`.data` filenames and loaded them locally in fresh child processes. All four
cases passed: old library before group load, group before old managed load,
an old object armed after group load, and a deleted mapped library. The receipt
is under `/home/17-fault-c2c3-20261117/providers-quant-guard-cpu-a1/receipt.json`.
Those libraries are test marker fixtures, not CANN or the production group DSO;
no device, container, model or quantization operation was started.

After review and the fresh cancellation SDK build, root can run the following
in the same immutable image used for SDK a3, with current source and matching
bundle mounted read-only at `/source` and `/bundle`,
only `/result` writable, `runc`, no device/driver mounts and no network. The
runner must record the actual image ID and preserve its owned-container
cleanup receipt. The script itself starts no container. It loads the reviewed
DSO and reads ABI/state only; it rejects exposed Ascend device nodes and any
model/helper activity, installed owner or resource count.

```sh
export PYTHONPATH=/source
export VLLM_SERVER_DEV_MODE=1 DCAT_DQ_GROUP_V1_ALLOWED=1 DCAT_DQ_V2_ENABLE=0
export DCAT_REAL_OPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
python3 /source/experiments/c2c3_closure_20261117/providers_quant_group_load_probe.py \
  --bundle /bundle \
  --approved-build-sha256 "$APPROVED_GROUP_BUILD_SHA256" \
  --out /result/load-probe
```

Expected status is `MANAGED_DSO_LOAD_AND_ZERO_STATE_PASS_NOT_NATIVE_ACCEPTANCE`.
The reviewed image ID is
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
This command is prepared for root and has not been run by the provider agent.
