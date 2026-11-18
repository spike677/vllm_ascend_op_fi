# GMM1 fixed-wheel lookup boundary

The current GMM1 typed branch is **UNKNOWN**. The fixed 64 KiB text window
did not locate a reference to either candidate GMM1 API name. This is a
bounded miss, not evidence that the operator or a candidate branch is absent.
No new Native hook, operator execution, row binding, or qualification follows
from this review. The full 135-projection denominator and the existing 10/135
request-row proofs remain unchanged; C2/C3 remain false.

## Checked local originals

| Original | SHA256 |
| --- | --- |
| `v11-quant-gmm1-elf-static-a1-partial.tar.gz` | `8ebcbe25fd4a47cf7fcd8f4bea5e179a787ee4fdb0d1c7d6f0f28ea4dc19a059` |
| `v11-quant-gmm1-local-window-a2.tar.gz` | `64bdac1df293a12a92ede4cccb503e3579a649439c16f6c85c0cdba83b06ac43` |
| a2 disassembly | `4ef4aa9b04a36e3bfcea0575d1e95eb4a6faa904fe7066f5a3ae23031cc9a4a1` |
| a2 actual `torch_npu/__init__.py` | `967690b75576c1dfb76ed3cef3bd8a876fe94ec3195e4afa3116ef7d31e9b160` |

Both archive hashes and all nine manifest-listed artifact sizes/hashes were
recomputed locally. No original package or ELF is committed. The extracts
bind `libtorch_npu.so` SHA256
`dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041`,
58,590,337 bytes, from image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
The root rehashed that ELF; this local review has its extracted metadata.

The a1 whole-library disassembly attempt stopped at its 90-second bound. Its
manifest explicitly says `complete_xref_scan=false`; its selected-symbol file
contains only a newline and supplies no GMM1 function address. The partial
attempt is not upgraded to a completed search.

The a2 range is exactly `[0x175bc3c, 0x176bc3c)`, with 16,384 addressed
four-byte instruction slots. Parsing was checked against the actual objdump
format. There are zero `adrp` references to a `0x2f0*` page and zero textual
`0x2f0b*` references. The range was a bounded neighboring-code hypothesis,
not an identified GMM1 function. Nearest exported template labels are not
function-owner evidence. No new window is inferred from this miss.

## What is known about the public entry

The captured public operator is
`npu::npu_grouped_matmul_swiglu_quant(Tensor x, Tensor weight, Tensor group_list,
Tensor weight_scale, Tensor x_scale, *, Tensor? bias=None, Tensor? offset=None)
-> (Tensor, Tensor, Tensor)`. The actual installed Python initialization imports
`torch_npu._C` at line 66 and exports each `torch.ops.npu` name into its globals
at lines 112–118. The captured `_C` ELF needs `libtorch_npu.so`. This establishes
the Python-to-dispatcher route, not the current PrivateUse1 implementation
address or its selected ACLNN API.

The fixed ELF contains both old `GroupedMatmulSwigluQuantWeightNZ` and
`GroupedMatmulSwigluQuantWeightNzV2` names and adjacent C++ lambda type names.
String proximity and the public single-Tensor schema do not select a branch.
Source107's 26 GMM1 source windows contain zero events from the four container
creator/destroy interfaces. Thus there is no actual GMM1 caller PC to reuse
from those events, and no basis to borrow a GMM2 caller PC or assume a
TensorList conversion/cache outcome.

## Smallest next metadata collection

1. Read the ELF's existing `.gnu_debuglink`: its file offset is `0x373c98c`
   and size is `0x1c` in the checked section table. The same ELF's Build-ID is
   `f784d0d51702c5ee243716c8945650872ff18f61`. For example,
   `readelf -x .gnu_debuglink "$ELF"` reads this section without disassembling
   `.text`. Check only the named file's standard local debug locations and
   `/usr/lib/debug/.build-id/f7/84d0d51702c5ee243716c8945650872ff18f61.debug`.
   A file is usable only after the full Build-ID/debuglink CRC and identity
   checks match. If present, query only the two public GMM1/v2 wrappers and
   `op_api` symbols, then request their exact function/FDE spans. A missing
   debug file stays missing; a nearby release's symbols are not a substitute.
2. Read the installed `torch_npu/version.py` and its distribution metadata.
   The locally available version string is only
   `2.9.0.post1+gitee7ba04`; the full upstream and `op-plugin` submodule commits
   are not established. Preserve any full commit/build identity and hash the
   files. Public source is usable only after those exact identities match;
   neither the release name nor the shortened suffix is enough.
3. In the same fixed image, a device-free, no-operator-call probe may record
   the two GMM1 operators' actual schemas, `_dispatch_dump`, and
   `_dispatch_dump_table`, together with canonical loaded `_C`/`libtorch_npu`
   file identities. These APIs describe registration keys, signatures, source
   labels, and fallback state. They do not establish an ACLNN owner or prove
   that a kernel/API executed. Existing same-image RMSNorm/SwiGLU originals
   illustrate the limitation: their PrivateUse1 source is the generic
   `build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413`, not a kernel
   function address. Source107's saved dispatch dumps concern
   `vllm::moe_forward_shared`, not GMM1.
4. If the new dump remains generic, inspect the *installed production*
   `Dispatcher.h`, `KernelFunction.h`, `BoxedKernel.h`, and relevant Python
   dispatcher declarations before proposing an address probe. The local
   development headers are Torch 2.10, so they cannot authorize a Torch 2.9
   layout. Do not read private kernel fields using guessed offsets or cast a
   Python object address to a Native entry.

The stop condition is no exact matching source/symbol identity and no supported
current-kernel-address observation. In that case the typed GMM1 branch remains
unknown, and no GMM1 hook is written. A later exact address permits only its
bounded function/FDE and resolver closure to be reviewed. Actual loaded
ACLNN owner, current arguments, and routed token/request semantics still need
their own evidence before row admission; dispatcher registration alone is
insufficient. This step does not repeat GMM2 or the old integer/frame census.
