# Independent CPU build and scope audit

The new Native26 group-v2 and fresh FA builds pass the recorded-artifact audit.
**This is CPU build/contract evidence only. No NPU launch, numerical effect,
natural model coverage, or C2/C3 qualification is granted.** The existing root
checkpoint remains3/17 C2 and3/17 C3 in its previously accepted scopes.

The audit used only read-only source/receipt/image inspection and explicit
host-memory/control test programs. It did not create or start a container,
load either fault provider into a Worker, or call an NPU. Original build and
acceptance receipts were not edited.

| Recomputed evidence | Result |
| --- | --- |
| source-a29, commit1076720aca23ad55decaf2fe42e174176c0e0a5f | All578 manifest files match |
| Native26 deployment | All9 actual artifact hashes and6 declared source hashes match |
| Native26 image/SDK | Exact arm64 image ID, two actual SDK header hashes,8.5.1 installation metadata match |
| Native26 ABI/control |18 actual exported function names/types/addresses and ELF symbol-version requirements match; CPU contracts pass |
| FA driver source-a28, commitc6af00601535e007eae8f88e74cafd7788bf50a0 | All562 manifest files and7 imported policy/decoder sources match |
| FA fresh vendor | All851 actual artifact hashes match; host installer, kernel, metadata and POD decoder match their named hashes |
| FA official source and SDK | Official HEADc77c0829f1f0f0b8f882db1d9b051a758b605472, four patched files and five actual9.1 SDK interface files match |
| Existing Worker18/20/39 result files | Original bytes remain unchanged; no scope was enlarged by rewriting these receipts |

## Native26 contract and runtime limits

Deployment manifest SHA256 is
`5a6adbf86bd0fdbb5a18ab16953714db6a292ff315bdc0dce28c1a59654b7653`;
build receipt SHA256 is
`ec4fe00b0da1d68ac97545d84b5da06708f1fed431e20d9bca67659de3d17ef0`.
The new `libdcat26_group_v2.so` SHA256 is
`e7c21c25eb228ee126b17c6ec76717861ba6942d043bd18302466fd5cd32174a`.
The image is pinned to
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
The two legacy writer DSOs are absent from the deployment directory; their
legacy install exports are absent from the new DSO. The independent trace and
proxy libraries retain the build receipt's exact parent-file hashes.

The SDK control program proves the compiled enums and structure sizes:
FP16=1, BF16=27, ND=2; Config72, Call144, Lease248, Receipt296, State136,
RuntimeState240 bytes. Its invalid/null control calls refuse. The host-memory
simulation exercises the QMM/FIA transaction engine, including failed
copy/synchronization/free and retryable recovery. These simulated memory
operations are not actual ACL device operations or a real consumer chain.

The host's default libstdc++ could not start the SDK-control executable because
it lacks `GLIBCXX_3.4.29`. That failed attempt is preserved. The independent
audit then invoked the already-present pinned image's CPU loader and C/C++
libraries directly, without a container, and obtained the same
`PASS_SDK_ABI_CONTROL_ONLY` result as the build. The image-library byte hashes
are in the summary. Native26 therefore remains bound to the matched image
runtime; this is not proof that it loads on an arbitrary host or8.x/9.x SDK.

Host and image `nm` render symbol order/version suffixes differently. The audit
retains that byte inequality, compares actual function symbol identities and
addresses, and separately checks ELF symbol-version requirements. These include
GLIBCXX_3.4.29 and GLIBC_2.34; the discrepancy was not hidden by accepting
unversioned names alone. Actual Native callback resolution, writer/residual/
consumer identity, NPU readback and group recovery still require real testing.

## FA build limits

Completion SHA256 is
`7189c077dad9411e53240d6a0cc9190477d00433d5b1325833c0fcb26c7cac5d`.
Its `source_commit` is the **official operator repository commit**, not the
separate source-a28 driver commit. This distinction is explicit in the summary.
The image is pinned to
`sha256:9008234fbe4948ce9e0383c34972044b369e42dc7711d44d5878f52267b8614a`.
The host-package hash identifies the fresh `.run` installer in`build_out`, not
an installed vendor `.so`.

The build includes key22619644976 and records compiled core typeMIX with
task ratio`tilingKey`. Public POD CPU roundtrip covers1464 bytes,205 fields
and65 array elements. It explicitly reports `stock_abi_validated=false` and
`actual_serialized_pod_verified=false`. Neither compiled metadata nor a CPU
roundtrip proves actual launch arguments, serialized runtime tiling, or stock
bridge equivalence. The five SDK interface files are individually verified;
`sdk_interface_manifest_sha256` remains null and is not presented as a complete
SDK manifest. B/F/OFF, independent raw statistics and natural model/Worker
coverage remain outstanding.

## Old scopes do not authorize new shared code

Worker18's accepted implementation fingerprint remains
`4fed69214272bb3d5117ba377f988000693410b134728e2307352dad461c98be`.
The actual source-a29 shared tensor/MoE/Conv3d provider fingerprint is
`09ff290a635f59d67359802d6a13fed6538e916b3f385d321890038ba7a7aeca`.
The current fingerprint includes all13 participating source files, including
the new MoE/Conv3d and serving adapters. **The old result does not qualify this
new implementation.** A fresh runtime qualification is required for the new
source scope even when reusing mechanism18.

`coverage_session.py` is byte-identical to source-a20, SHA256
`0e3ee343340d9d27bd1ccaaae52c3225497ffb9d37f7aea9218d26c794b221c1`.
Its C3 gate still requires a live issuer proof, exact scope equality, the same
owner/target lifetimes, and a later generation. A saved report cannot recreate
that proof. MODEL_LOCAL17 explicitly refuses a Worker identity; prepared
MoE8 and read-only22 census do not inherit Worker18 qualification. Root's
current matrix retains these pending/not-established rows. This audit changes
neither the17-mechanism count nor the27-node ledger.

## Deliverables

[agent7_cpu_builds_a29_summary.json](agent7_cpu_builds_a29_summary.json) contains
only the small audit summary, source/artifact/image/runtime hashes and the
bounded CPU outputs. Its SHA256 is
`0ffe7856a42e913345ddc44a5d0bc63e8efe563d693e605a8d316d2739aaccd4`.
The reproducible reader is
[audit_cpu_builds_124.py](audit_cpu_builds_124.py), invoked with `--root`, a
fresh`--out`, and optionally`--rerun-cpu-contracts`.
The final remote sidecar is
`/home/17-fault-c2c3-20261117/agent7-cpu-builds-a29-a5`.
Earlier reader attempts are retained: a dict/list schema assumption, host
runtime/tool-format differences, and the host-installer path were corrected
without editing source evidence. No binaries or original statistics are
included. Twenty-three focused local audit tests pass; they are CPU checks,
not additional mechanism acceptance runs.
