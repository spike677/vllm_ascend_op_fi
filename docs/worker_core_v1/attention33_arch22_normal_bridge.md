# Attention33 arch22 normal bridge candidate

This is a separate, normal-only build and execution route. It is not a
stock/private equivalence certificate, C1/C2/C3 qualification, E5 result, or
permission to run F. The public provider's F gate is unchanged. The adapter
also rejects `fault` before invoking the original API or entering Native scope.

The preserved source53 first-boundary records have identical complete Q and
logical K/V hashes, mask/table/scalars, and stable repeated outputs in each
arm. The first immediate output differs in 67/2048 FP16 elements (maximum
absolute difference 0.00006103515625). Later mutation checks were exact.
The stock runtime registered ELF SHA is
`c182034ed30e94e277feeb4ae3ac859027758efc49f5418dcbaabecea236f11a`,
and its actually bound key `103000000010300003` has distinct AIC/AIV entries.
Private FD instead launched key `5100000000010200103`, block 17, from ELF
`1e8604d4e453f4d6ba4d1cb9c821935b8d6ddfb2e0aad3c21f0eea692c84b7d8`.
These identify different implementations; they do not decode the stock ABI
or prove that route selection explains every output difference.

The actual c77 source explains the private route: `CheckSpecConditions`
accepts TND, sparse mode 3, GQA inner precision 0, 3D paged KV, D128 and
page16. `RouteToFia` then returns false, preferring FAI SplitFuse. The new
variant changes only this preference for the narrowly matched real call.
The original v3 parser, checker, tiler, `GenTilingKey`, serializer and kernel
remain unchanged. No key, block count, S2 factor, Q/K/V, mask, table, output or
forward is rewritten. The new manifest explicitly records the route change.

The normal package applies that fixed implementation to matching normal calls
even without an observer. Otherwise post-unregister calls would revert to the
known-different private route. Owner-scoped PID/thread/token capture only reads
the native serializer. It never chooses the arithmetic. Nonmatching inputs
retain the original route, and normal FAI warmup entries remain compiled.
No 590/600 fault entry is included.

Source53's actual mask is `int8 [2048,2048]`; the table is
`int32 [1,80]`. The new guards retain those dtype constraints and refuse all
other optional tensors using the complete pinned public host input enum.
Python `pre_tokens`/`next_tokens` were omitted; their declared defaults are
2147483647. They remain omitted in the original call. The final Native POD is
checked separately and recorded as an actual observation. It is not filled
from the declaration.

## Build and run, by the root operator only

Use a new pristine checkout of
`c77c0829f1f0f0b8f882db1d9b051a758b605472`, with the existing independently
pinned dependencies and exact CANN 8.5.1 interface probe. Do not patch or
overwrite the old FD build, stage or evidence. This change performs no build.

The existing root-only controller now supports `--strategy arch22-normal-bridge`:

```sh
python3 -B NEW_FROZEN_SOURCE/experiments/c2c3_closure_20261117/build_tnd33_124.py \
  --source NEW_FROZEN_SOURCE \
  --out /home/17-fault-c2c3-20261117/tnd33-arch22-normal-build-a1 \
  --interfaces EXACT_EXISTING_CANN851_INTERFACE_MANIFEST \
  --strategy arch22-normal-bridge
```

It clones source from
`/home/17-fault-engineering-20261117/ops-transformer-pristine-a1` and uses
dependency sources at
`/home/haoran/experiments/fault28/fault28-fia-graph-20261106/dependencies_a4`.
The exact interface-manifest path is the readonly `/sdk_interfaces.json`
mount source in the preserved
`/home/17-fault-c2c3-20261117/tnd33-fai-fd-build-a1/controller.json`; reuse its
actual path and bytes, without inventing a new SDK declaration. The controller
pins the existing Worker image `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`
and creates a device-free build container. It is not executed by this agent.
The detailed inner commands are:

```sh
python3 -B fault_injection/native_adapters/attention28/fia/apply_tnd33_arch22_bridge.py \
  --source NEW_PRISTINE_C77_SOURCE --output NEW_POLICY_RECEIPT_DIR

python3 -B fault_injection/native_adapters/attention28/fia/build_fresh.py \
  --source NEW_PATCHED_C77_SOURCE \
  --policy-receipt NEW_POLICY_RECEIPT_DIR/policy_source_manifest.json \
  --dependencies PINNED_DEPENDENCY_SOURCE_ROOT --work NEW_BUILD_ROOT \
  --sdk /usr/local/Ascend/cann-8.5.1 \
  --sdk-interface-manifest EXACT_SDK_PROBE --sdk-interface-sha256 EXACT_SDK_PROBE_SHA \
  --opdef-compat ascend910b-cann851
```

The distinct vendor is `stage/vendors/dcat_f33_arch22_normal_transformer`.
The schema comes from `evidence/arch22_schema/fia_abi_schema.json`, compiled
from this build's real arch22 host entry. The final observer must be freshly
built with the owned-scope readback ABI. Runtime verification requires all
new artifact hashes, SDK identity, original stock Norm DSO coexistence and
actual fresh ELF registration/final arguments.

Use the same new source/wheel/driver/observer for two fresh cases, sequentially:

```sh
python3 -B experiments/c2c3_closure_20261117/attention33_worker_probe.py \
  --phase stock --first-boundary-diagnostic --arch22-normal-bridge \
  --experimental-qualification --model /model --out /evidence/run \
  --observer-receipt OBSERVER_RECEIPT --observer-receipt-sha256 OBSERVER_SHA

python3 -B experiments/c2c3_closure_20261117/attention33_worker_probe.py \
  --phase baseline --first-boundary-diagnostic --arch22-normal-bridge \
  --experimental-qualification --model /model --out /evidence/run \
  --completion /native/evidence/completion.json --completion-sha256 NEW_COMPLETION_SHA \
  --vendor /native/stage/vendors/dcat_f33_arch22_normal_transformer \
  --public-schema /native/evidence/arch22_schema/fia_abi_schema.json --schema-sha256 NEW_SCHEMA_SHA \
  --observer-receipt OBSERVER_RECEIPT --observer-receipt-sha256 OBSERVER_SHA
```

Each case retains the existing five complete natural phases, all 87 payloads
per phase, first-boundary Q/immediate output bins, logical K/V read-before
hashes, later readbacks, original runtime records, and cleanup. The private
arm observes 28 real decode calls in each of its first four phases. Its last
healthy phase has no Native observation scope and therefore also checks the
package's normal behavior after observer cleanup. Do not reuse the old frozen
source53 reviewer or the FD-specific F admission certificate for this variant.
The original stock runtime gap and selected-ACLNN-owner coverage limit remain
in the old artifact; new offline resolutions must remain separately labeled.

Trusted public Core deployment may construct `Arch22NormalNativeCall` and
pass that exact object to `install_attention_serving_binding`. No RPC accepts
paths, callbacks or arbitrary factories. Existing `33.paged_decode` registry
mapping stays `PagedDecodeServingProvider`; no registry/exporter change is
needed. Public baseline captures only its selected natural target. Complete
stock/private normal equivalence still needs the independent two-case probe.

## CPU evidence and remaining gates

The five cached actual source SHA pins passed; applying the patch in memory
changed exactly three source files. Unit contracts exercise real CPU tensor
storage identity, actual original-call/return preservation, complete optional
rejection, compiled-key set identity, final-POD disagreements, foreign
owner/thread rejection and before/after Native-end retry behavior. Native
callbacks in those contracts are CPU fixtures, not device evidence.

Fresh CANN compilation and actual device execution are pending. The narrower
route may still differ numerically from stock; key equality does not prove
implementation equality. Require complete byte-exact new stock/private
payloads, actual final launch/POD identities and all recovery gates before
considering any F implementation. If those fail, retain the exact difference
and continue investigation. Do not relabel a missing bridge as inapplicability.
