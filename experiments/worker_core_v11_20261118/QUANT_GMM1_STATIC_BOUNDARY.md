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
| `v11-gmm1-dispatch-metadata-a1.tar.gz` | `17ed7828f71abc3fbd6d59d42ecbe5d8e599bfef59cc3cda912b502a2d39ff2c` |
| `v11-gmm1-dispatch-metadata-a2.tar.gz` | `41382a968a642295a2463b68abb197839043445479584ebdff8fdce92af8fa73` |
| `v11-gmm1-debuglink-a1.json` | `e26afe26a1bb64e886e8027721e07c1b24ee08b62961e5861d177d5beca0cc71` |

Both static archive hashes and all nine manifest-listed artifact sizes/hashes
were recomputed locally. Each installed-metadata archive's five declared files
were also checked. No original package or ELF is committed. The extracts
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

## Installed metadata follow-up

The additional device-free metadata packages supply five source/header files.
The exact installed `torch_npu/version.py` is SHA256
`d07f96f22289c55c3ab70ef6a735b8458b775ec4de7b4dce352c3d33a3c39b2f`
and declares full `git_version=ee7ba047d3021288b2957294ffdcdb1e0d2cea65`.
Its distribution `direct_url.json` records wheel SHA256
`73d2e4fb88acccfc032dee8d5e540168098b96c2375ca5a5d3860d6c4c4bc6fb`.
`METADATA`, `direct_url.json`, and `version.py` also match the wheel's selected
`RECORD` entries, including byte counts and base64 SHA256 values.
The `op-plugin` submodule/build identity is not yet established.

Package a1 **did not obtain GMM1 dispatcher registrations**. Its result has
`errors=[RuntimeError('Failed to load the backend extension: torch_npu...')]`
and no operator/schema/dump fields. The controller's exit code 0 means the
probe saved its result; it does not turn the import failure into a successful
registration audit. `requested_operator_calls=0` is retained. A retry must
show the actual explicit `torch_npu` import result rather than treating
disabled automatic loading as successful NPU registration.

Package a2 sourced the fixed CANN environment, disabled automatic backend
loading, and explicitly imported `torch_npu`. It records `errors=[]`, zero
requested operator calls, and the expected mapped `_C` and `libtorch_npu`
hashes. Its two schemas distinguish the old single-Tensor weight operator
with three Tensor returns from the v2 TensorList weight/scale operator with
two Tensor returns. Both current PrivateUse1 registrations report the same
generic `compiler_depend.ts:8413` location; neither dump supplies a kernel
address or selected ACLNN name. This is successful schema/registration
observation in that CPU process, not a replay of source107's Native GMM1 call.

The supplied *production Torch 2.9* declarations close the public-address
interface question for these headers. `Dispatcher.h` line 490 returns a
`SafeKernelFunction`; its public operations are `callBoxed`, `debug`, and
`opHandle`. `KernelFunction.h` lines 279–295 keep the unboxed pointers
private, and `BoxedKernel.h` lines 183–208 keep `getFnPtr` private. The Python
stub's `_SafeKernelFunction` has only `call_boxed` and `op_handle`; the
operator handle has `schema`, `debug`, and `redispatch_boxed`. None of these
declared inspection methods exposes a kernel function pointer. The call
methods will not be used to manufacture an address observation. This is an
inspection of the supplied interfaces, not a claim about uninspected internal
implementation hooks.

## Debug and exact-source checks

The existing `.gnu_debuglink` at file offset `0x373c98c`, size `0x1c`, names
`libtorch_npu.so.debug` and records CRC32 `0x456e1e00`; the 28 bytes were
independently decoded. The ELF's Build-ID is
`f784d0d51702c5ee243716c8945650872ff18f61`. All four recorded standard paths
are absent: the library directory, its `.debug` subdirectory, the corresponding
`/usr/lib/debug` path, and the full Build-ID path. No broader filesystem or
text-section scan was performed. A nearby release's debug symbols are not a
substitute.

The exact full commit was then queried through the GitHub connector and
bounded official HTTPS reads. These are access results, not proof that no
vendor/private/historical source exists:

| Exact-commit endpoint | Observed result |
| --- | --- |
| [GitHub commit object](https://api.github.com/repos/Ascend/pytorch/git/commits/ee7ba047d3021288b2957294ffdcdb1e0d2cea65) | HTTP 404 |
| [GitHub `.gitmodules`](https://api.github.com/repos/Ascend/pytorch/contents/.gitmodules?ref=ee7ba047d3021288b2957294ffdcdb1e0d2cea65) | HTTP 404, no commit found for ref |
| [Gitee exact commit](https://gitee.com/api/v5/repos/ascend/pytorch/commits/ee7ba047d3021288b2957294ffdcdb1e0d2cea65) | HTTP 404, `Commit not found` |
| [GitCode exact commit](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/commits/ee7ba047d3021288b2957294ffdcdb1e0d2cea65) | HTTP 404, `404 Not Found Commit` |

The Gitee and GitCode exact-commit raw `.gitmodules` reads also returned 404.
No branch HEAD or approximate version was substituted. Consequently no
`op-plugin` gitlink or exact GMM1 implementation was obtained in this step.

## Remaining minimum evidence

Obtain either the matching debug artifact or the exact wheel's source/build
manifest, including the full `op-plugin` commit and any applied patches.
For debug data, require full Build-ID/debuglink CRC and file identity before
querying only the two GMM1/v2 wrappers and `op_api` function spans. For source,
first bind the declared submodule tree; then inspect only the two implementations
and their registration configuration. Full wheel identity remains separate
from a source-tree hypothesis. Successful dispatcher metadata cannot fill
this missing source/address evidence. Do not repeat its CPU probe merely to
refresh the same generic source label.

The stop condition is no exact matching source/symbol identity and no supported
current-kernel-address observation. In that case the typed GMM1 branch remains
unknown, and no GMM1 hook is written. A later exact address permits only its
bounded function/FDE and resolver closure to be reviewed. Actual loaded
ACLNN owner, current arguments, and routed token/request semantics still need
their own evidence before row admission; dispatcher registration alone is
insufficient. This step does not repeat GMM2 or the old integer/frame census.
