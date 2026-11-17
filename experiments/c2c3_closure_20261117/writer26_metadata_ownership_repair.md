# Fault26 v2 metadata ownership and dependency rebuild

This repair applies to the current v2 QMM and FIA writer experiment, not the
omitted legacy writer DSOs. `writer26_group_runtime.cpp::descriptor_matches`
and the actually loaded `fault26_trace_preload.cpp::capture_tensor` formerly
leaked SDK view/storage/stride arrays. The loaded main/split opapi dependencies
also use the repaired Fault20 metadata captures. All three paths therefore
require fresh builds; replacing only `libdcat26_group_v2.so` is insufficient.

The two current Fault26 functions now use the same noncopyable array owner as
20/39. Success, every later getter failure, partially assigned arrays, invalid
geometry and exception unwind all release their actual `new[]` arrays. There
are separate additive read-only counter exports from the v2 shim and tracer:
`dcat26_group_metadata_arrays_v1` and `dcat26_trace_metadata_arrays_v1`.
The ABI and existing writer lifecycle remain separate from those exports.

The root ran `test_metadata_ownership.py --writer26` on 124, source-a40:
`/home/17-fault-c2c3-20261117/metadata-ownership-writer26-cpu-a1/receipt.json`.
The exact production-body CPU mock passed 214 scenarios with 279 actual
`new[]` and 279 actual `delete[]`, zero live arrays. This is a host-memory
contract, with no SDK/NPU calls or natural Worker qualification.

Fresh trace builds run that contract and write `metadata-array-cpu.json` plus
`metadata-array-ownership.json`, binding the actual input source hashes, the
balanced allocation counters, and defined exports/hashes of the tracer and
both metadata proxies. `providers26_build_group.py` verifies that evidence
against current source **before** copying a parent or starting its build
container. Old parents have no ownership receipt and explicitly refuse with
`WRITER26_METADATA_OLD_PARENT_REBUILD_REQUIRED`. Evidence files are copied
without changing bytes. The group builder separately reruns the contract,
compiles the v2 shim and requires its counter export. All test dependencies
are included in the native recipe and group source fingerprints.

At actual Worker attachment, the runtime verifies its immutable group manifest
and all four already-loaded images: v2 shim, tracer, main opapi and split
opapi. Every counter function must originate from that exact manifest image;
SDK aliases additionally require identical mapped inode/device and bytes.
No late vendor loading or counter reset occurs. Initial, per-request,
recovery and Worker health reports preserve raw counters. All must be
quiescent, have equal acquisitions/releases, zero getter/capture/ownership
errors, stable independent instances and monotonic counts. Final C2/C3 health
also requires actual array activity during this workload. An unused image may
remain zero; missing image evidence may not.

Root validation order:

1. Run `python3 -B tests/native/test_metadata_arrays_elf.py --writer26` on a
   Linux CPU host. This checks four independent real ELF counter instances,
   an SDK-style same-inode alias, and refusal of a foreign same-byte copy.
2. Rebuild the trace parent with the normal exact-SDK recipe and stage a fresh
   parent bundle. Historical build outputs are not edited.
3. Run `providers26_build_group.py` with that fresh parent and its exact
   manifest hash. The old source-a21 parent cannot pass the new gate.
4. Use the existing root-owned queue and one-device resource gate for a fresh
   natural inference census, lossless/B/F/OFF, C2 then bounded live C3. Verify
   all four loaded readbacks and all existing buffers/descriptors/workspace,
   hooks, leases, native denominator and exact OFF checks together.

Local Python contracts cover parent refusal, stale dependency bytes, failed
allocation accounting, runtime orchestration and failed no-activity health.
The local Windows authoring environment cannot execute the Linux loader
contract. These source changes and CPU results confer no new C2/C3 status and
do not alter historical statistics or numerical evidence.
