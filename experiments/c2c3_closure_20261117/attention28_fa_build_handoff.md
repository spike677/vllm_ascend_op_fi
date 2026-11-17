# Fresh FA source and build boundary

This inventory does not grant new qualification to `28.fa.legal_tiling`.

The frozen import is commit `6dad85c480255829b0a4e74657a7e2ff2ce390b3`;
`fault_injection/native_adapters/attention28/source_import.json` pins each imported
file. The official operator checkout remains
`c77c0829f1f0f0b8f882db1d9b051a758b605472`. Reusable source is
`native_execution_strategy/native/{apply_fa_bridge.py,fa_bridge.h,fa_plan.h,fa_pod_decoder.cpp}`
and the plan/scope CPU tests. Their historical binaries, registry certificate,
vendor, decoder executable and numerical results are not inputs.

The source-level route is FA `FlashAttentionScoreTilingS1s2Bn2gs1`, BNSD FP16,
B=1 or 8, Hq/Hkv=16/8, Q/KV=2048/2048, D=128, causal sparse mode 2,
keep_prob=1, inner_precise=0. Typed policy selects M=64 or 128 and nRatio=4 or 8;
S2 base remains 128. The selection precedes the existing native template,
multicore, sparse, tensor-size and matmul/workspace calculations. The new
controller reuses those exact four patched host files and preserves official
kernel source bytes. Kernel key `22619644976` is a build request, not launch
evidence or proof that every selection is legal on the current device.

Read-only inspection of `/home/native-execution-strategy-20261030` found:

* `build/evidence_a1/build_completion.json`: historical host/package CPU build
  passed; source frozen, NPU not executed by that build. Its build log SHA is
  `1cd95beeaf990b04fddd2aa2160aaedfebca61f9063e5ad24c209b52b87dc3db`.
* `fa_pod_selftest.json`: historical public POD roundtrip 1464 bytes, 205 fields,
  65 array elements, `stock_abi_validated=false`.
* The historical vendor contains host/JIT sources; it is not a newly compiled
  and registered matching OPC kernel for this attempt. Its old HF model runs
  and old collector results do not establish current standard Worker C2/C3.
* Official generated FP16 input descriptor identity is
  `FlashAttentionScore_d510be03e39ccdbb4abc4f0db5af9547`; CANN 9.1's immutable SDK
  matcher independently contains that descriptor. The new build must generate
  it again, and must compile its own `.o` and metadata. No old descriptor or
  binary is copied.

## Root-owned CPU command

After integrating this change into a fresh source snapshot, root may run:

```sh
python3 -B /home/17-fault-c2c3-20261117/source-NEW/experiments/c2c3_closure_20261117/attention28_fa_build_124.py \
  --source /home/17-fault-c2c3-20261117/source-NEW \
  --out /home/17-fault-c2c3-20261117/fa28-fresh-build-a1
```

This creates a new local source clone, reapplies the fixed source patch and
uses the exact CANN 9.1 SDK interface pins already in the FIA source build
contract. Shared builder hooks preserve the existing FIA policy, checks and
operator by default; that contract JSON is unchanged. FA has its own receipt
validator, CPU checks, FP16 kernel compiler and binary registry validator.
The controller resolves and records the image ID and uses `--runtime=runc`,
`--network=none`, `ASCEND_VISIBLE_DEVICES=void`, no device mounts, four CPUs,
16 GiB, and a bash entrypoint. Only its new build directory is writable.
The agent has not started this container or called an NPU.

Expected new files are under `build/run/evidence/`: full host/package logs,
`fa_public_schema.json`, compiled decoder and all its header hashes,
plan/scope/roundtrip logs, matching OPC kernel log, registration receipt and
`completion.json`. Final status is
`CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION`; loading remains false. A compiler
or registry mismatch is retained as failure, never resolved by copying an old
kernel or inventing metadata. CPU tests cover exact four-file patch identity,
alternate-SDK/FIA receipt refusal, duplicate/wrong keys, wrong binary digest,
and compiler-derived summary representation.

## Exact remaining runtime work

The new generation still needs actual SDK/compiler completion; complete new
host/kernel/vendor binding; observed actual function-to-ELF/key/launch arguments;
public FA POD interpretation of the actual final tiling bytes; block/schedule,
workspace and original versus controlled factor audit; exact OFF/bridge and
complete output repeats; independent new B/F raw collector evidence; and a
natural eligible model/Worker boundary. None follows from constants in source
or a passing CPU roundtrip. The current observer's FIA argument slots and
1576-byte schema cannot be reused as FA argument slots or its 1464-byte schema.

For both FIA mechanisms, the qualified small-module input contract remains
BNSD prefill. The currently observed Qwen standard Worker uses TND/page16 and
CANN 8.5.1. This is ineligible for the CANN 9.1 BNSD mechanisms; the independent
33 FAI TND strategy does not fill that gap. Worker preparation must retain the
actual API's original Q/K/V, mask, metadata and return object, refuse a layout
or SDK mismatch, and cannot silently repack tensors or force another route.
