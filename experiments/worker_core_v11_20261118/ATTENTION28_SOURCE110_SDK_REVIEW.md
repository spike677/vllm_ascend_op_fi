# Source110: fresh TND28 normal SDK build completed

The fresh `tnd28-prefill-normal-build-a2` completed all18 device-free build
steps on124 in the pinned CANN8.5.1/A2 image. It compiled the host package,
actual public serializer and matching kernel. The owned container exited0
without OOM and was removed. All174 preceding production-image CPU contracts
passed, with zero skips.

This repairs the source109 build refusal without changing the failed source109
originals. The actual schema SHA is identical to source109: no schema bytes
were removed or rewritten to pass. The verifier reconstructs63 semantic fields
plus exactly two SDK-generated padding fields and verifies all65 offsets,
widths and counts across the full1576 bytes. Its compiler dependency must be
the pinned SDK `register/tilingdata_base.h`, not a same-named stub.

Root independently rehashed all1163 vendor artifacts, all17 reconstructed
source files and the CPU/schema/macro/registration evidence. The schema covers
the full serializer with no overlaps or holes; all18 command return codes are0.
The compiled key set includes the observed TND key `5000000000000200103`.
This is a build fact, not proof that a Worker loaded or launched this package.

The compiler retained a warning that it used the default API SoC version when
it did not find a tiling SO. The new binary/registration and explicit build
target were verified by the build contracts; actual launch qualification still
requires the separate NPU evidence below. The warning was not suppressed.

## Scope still withheld

The build is **normal-only**. `loading_allowed`, `fault_execution_ready`,
stock/private equivalence, actual launch and C1/C2/C3 remain false. No NPU
computation was executed. Remaining gates are:

- Actual new-package function ownership and complete raw/POD/launch/stream join.
- Original noncontiguous V and prepared-pointer/copy/stride lineage, plus actual
  int8 causal-mask values. A public API journal alone cannot prove internal
  prepared tensors or Slice operands.
- Complete stock/private normal-repeat equality and same-Worker OFF/health
  with zero owned resources.

No private normal output, fault propagation or statistics E5 is claimed.
Fault33's separate Q1/page16 build cannot satisfy these TND2048 gates.

## Original identities

Originals: `/home/17-fault-c2c3-20261117/tnd28-prefill-normal-build-a2`.

| Item | SHA256 or commit |
| --- | --- |
| Frozen source | `823e4b59697a4dfdb21890bfa4082e69b61ea7f5` |
| Source manifest | `583c558bfa57ba3a410a3599d664861467cf303d92338f3eb720ffb7af7349b3` |
| CPU test log | `97abc50d52053cc3c40ec14517773bc2931ac4bb7291f29760c430faca0abf11` |
| Build completion | `a93bdaa7e045e2f35b0e1537caae089c4cbaec75beeae2da151235dd563c5a0a` |
| Device-free controller | `a6f7c00f4ff36c622d527c007b8f3256c376d7c1250029ceee81196501b17e75` |
| Actual serializer schema | `20dd6a361142aa9acd73f0787673e24660daef46846e5b362eb9306b15fe035c` |
| Macro dependency proof | `75b181d347b41657572fa79cd269ab2d9de75409961ee84d11872ced8bb1dca6` |
| Kernel binary | `dbda70fee42e4326293960d23a273789f695742ede78c58ded342c6824935657` |
| Kernel metadata | `999b08df18c5fdf981758072445769acd3d0bfbd89754cdbc24bfb0efd91c149` |
| Private op-api library | `120e0e037ffcda37debcf48ed5bfc10b6bda32e879508802c36a6c7759ddc263` |
| Independent rehash review | `613f1de2b4158cfce06cc77c5347d14f316a7d90aad326d4b3a2112d1d5ef4b9` |

Only reviewed source, tests and summaries are published. The actual SDK
headers, vendor binaries, private tensors and raw statistics are not uploaded.
