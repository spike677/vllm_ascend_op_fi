# Fixed-wheel GMM2 typed-entry review

The source102 container observation's actual stock caller PC `0x1755174`
selects a compiled branch for `aclnnGroupedMatmulWeightNzGetWorkspaceSize`
and `aclnnGroupedMatmulWeightNz`. The conclusion comes from the observed PC,
the fixed ELF's string-to-cache data flow, its 23-argument indirect call, and
the matching CANN header. It does not come from the count of 14 TensorLists.

This is `STATIC_BRANCH_AND_TYPED_ABI_IDENTIFIED`, **not** an observation of the
current GWS's resolved function owner or actual argument values. The public
operator remains `npu.npu_grouped_matmul.default`; no operator is renamed.
The current result stays zero F, 10/135 request-row-proven projections, with
the other 125 candidates retained and C2/C3 false.

## Immutable inputs and verification

The already observed caller belongs to `libtorch_npu.so`, 58,590,337 bytes,
SHA256 `dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041`.
The three private static packages were extracted by the root from that exact
staged ELF without loading it or executing a driver/operator:

| Package | SHA256 |
| --- | --- |
| `v11-quant-gmm2-elf-static-a1.tar.gz` | `c72341e0b42547da91783d561fa616704ac34b1d7cc075db7a92d4be71b22331` |
| `v11-quant-gmm2-elf-static-a2.tar.gz` | `5e17ae3f6292cba5c95e123964f1adacc43e52ab1ad92e4b689034ad37c2f178` |
| `v11-quant-gmm2-elf-static-a3.tar.gz` | `4e967720b40e40c43505a6477a83fad727f0f7b98a61409f880b859abc6a1bd2` |

All archive hashes and every manifest-listed member's size and SHA were
recomputed locally. The packages retain readelf/objdump command arguments and
output hashes, selected FDEs, and the original rodata bytes. Large original
artifacts and ELF bytes are not committed. This review has the static text
extracts; the root performed the staged ELF byte rehash.

The original actual-call audit and its separate local/124 byte availability
limits remain in [QUANT_SOURCE102_CONTAINER_REVIEW.md](QUANT_SOURCE102_CONTAINER_REVIEW.md).
The root's full a5 re-audit summary SHA is
`64aed87b5bb3a5e4db55765a19d2871e294007a71b5bb1c67b52b9b987d145f3`;
it includes the actual DSO rehash and all 19 CPU-contract log hashes.

## Static call chain

Addresses below are ELF virtual addresses relative to the loaded image base.
The inspected text/rodata PT_LOADs happen to have equal virtual and file
offsets; that equality is verified in the packages, not assumed generally.
The stripped dynamic symbol table does not name these local functions. Its
nearest exported C++ template symbol is not treated as the function owner.

| Evidence | Exact relationship |
| --- | --- |
| Actual caller `0x1755174` | Return PC after the IntArray conversion helper at `0x1755170`; within FDE `0x1754060..0x175b790` |
| `0x17541b8..0x17541bc` | Defines the static cache-block base as `0x37b5000 + 0x8e8` |
| `0x175817c..0x1758194` | Passes rodata `0x2f0a1d0`, exactly `aclnnGroupedMatmulWeightNzGetWorkspaceSize`, to string-keyed lookup helper `0x8f4230`; stores the returned pointer at block offset 1872 |
| `0x1754bf0..0x1754bfc` | Checks both selected GWS and execute cache pointers before this branch proceeds |
| `0x17595d4..0x17595dc` | Copies the selected GWS pointer from block offset 1872 to the invocation cache at offset 1936 |
| `0x1755244`, `0x1755284` | Loads that invocation-cache pointer and passes it, with the current converted argument tuple, to helper `0x175bb60` |
| `0x175bb60..0x175bc3c` | Expands the tuple into 8 register arguments plus 15 stack arguments; the actual indirect call instruction is `0x175bc0c: blr x8` |
| `0x1758158..0x1758170` | Passes rodata `0x2f09e50`, exactly `aclnnGroupedMatmulWeightNz`, to the same lookup helper and stores the returned pointer at block offset 88 |
| `0x175532c..0x1755414` | Creates the execution closure used by this converted-argument branch; its invocation thunk is `0x17484e0` |
| `0x17484e0..0x1748528` | Invocation thunk tail-jumps to closure body `0x1747c64` |
| `0x1747c9c..0x1747cb0` | Loads the selected execute pointer at the same cache-block offset 88 and calls it with four arguments |

V4 and V5 strings, caches, and other branches are present in this same ELF.
They are not selected merely because the public schema is shared. This review
also does not assert what address the lookup helper returned during the
historical process: it follows the compiled symbol-name and cache relationship.

## Typed contract to observe

The exact installed SDK header is
`/usr/local/Ascend/cann-8.5.1/include/aclnnop/aclnn_grouped_matmul_weight_nz.h`,
SHA256 `f41308d5814aa82ba92ff77aa8e258b15ca3ae72d2c0c61fadc2cdd95de6d7fd`.
Its GWS contract has the following ordered 23 slots. The future translation
unit must include this actual header and check the export's function type;
this table is not a hand-written substitute for the SDK declaration.

| Zero-based slot | Declared argument |
| --- | --- |
| 0–7 | `const aclTensorList*`: x, weight, biasOptional, scaleOptional, offsetOptional, antiquantScaleOptional, antiquantOffsetOptional, perTokenScaleOptional |
| 8 | `const aclTensor* groupListOptional` |
| 9–11 | `const aclTensorList*`: activationInputOptional, activationQuantScaleOptional, activationQuantOffsetOptional |
| 12–15 | `int64_t`: splitItem, groupType, groupListType, actType |
| 16 | `aclIntArray* tuningConfigOptional` |
| 17 | `int64_t quantGroupSize` |
| 18–20 | `aclTensorList*`: out, activationFeatureOutOptional, dynQuantScaleOutOptional |
| 21 | `uint64_t* workspaceSize` |
| 22 | `aclOpExecutor** executor` |

Execute is `(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
aclrtStream stream)`. The stock libopapi disk candidate, SHA256
`b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c`, exports
these two names at ELF offsets `0x1ba8bd0` and `0x1b8a730`. These are disk
candidates, not a substitute for the actual loaded/resolved owner.

The 23-slot helper distinguishes this branch from V4's 21 slots and V5's 22.
In particular, `quantGroupSize` has no corresponding public Torch schema
argument. Its actual value is still unknown and must be read at GWS entry;
neither an assumed zero nor a Python-derived value is allowed.

## Next bounded zero-F observation

Extend only the existing private exact-owner proxy with this one typed GWS/
execute pair, loaded before Worker startup and recording within the existing
actual GMM2 scope. Do not intercept global dlsym, patch the stock library, add
fault modes, or bind routed rows yet. The extension requires a new source
fingerprint, SDK build, and root review before its one B8 decode run.

Record these facts at the actual boundaries, while each object is valid:

1. Actual resolved GWS/execute address, loaded-image identity and executable
   segment, original-call count, caller PC, current scope token, thread,
   device, and sequence. Check the exact pair; a basename or disk symbol is
   insufficient. The observed GWS caller is expected to be the helper return
   PC `0x175bc10`; any new path must be retained and reviewed, not relabelled.
2. Each of the 14 **named** TensorList argument handles and the named IntArray
   handle, including explicit nulls, joined to their current creator
   registration and its copied-member snapshot. Validate public size getters
   against that registration while the actual GWS holds the handle. Do not
   recover named slots from construction order, use a retired/reused handle,
   or reinterpret opaque layouts. The individually supplied groupList Tensor
   requires a current full typed getter snapshot with owned-array release.
3. All five current int64 scalars, including quantGroupSize; actual GWS status,
   workspace requirement, and returned executor with a fresh registration.
   Then the matching actual execute's executor, workspace pointer/size,
   stream, status, and original-call count. A failed GWS with a non-null
   executor or a prepared executor with no execute remains unresolved; it
   cannot be erased as drained or freed by the observer.
4. Lifecycle ordering through the existing destroy records and end-of-request
   cleanup. No late dereference of saved member descriptor pointers, no
   floating activation export, and no replacement or mutation of any public
   tensor. Current container registration alone must not be represented as
   independently proved descriptor immutability or routed row semantics.

The full 135-projection public/Native denominator stays in the same request.
Successful interface observation requires all 26 actual GMM2 windows to have
one correctly owned GWS/execute pair and complete named-slot/lifetime joins,
without overflow, dropped calls, exceptions, unresolved prepared executors,
or cleanup/health failures. A version/owner/geometry mismatch, missing pair,
ambiguous reused registration, or inaccessible current metadata ends this
hypothesis with an explicit refusal. Cache hits that bypass GWS stay visible
and do not count as observed GWS pairs. Bounds and observation overhead must
be recorded; no claim of unchanged timing or zero observer cost is made.

Even that success would establish only this Native interface join. Current
routing integer producers, GMM1, and unpermute must still join by actual
object/lifetime and semantics before a routed token-to-request proof can be
admitted. The existing 52-frame/260-integer audits are not rerun as a new
qualification claim and cannot authorize a later process. All unknown
candidates remain; public C2/C3 requires the eventual complete qualified set
and the original live C2 proof in the same Worker.

No Native implementation or runtime/admission change is included in this
static-review commit. Validation was local archive/hash checking and manual
instruction/ABI data-flow review only; no SSH, SDK build, container, or NPU
was run by the reviewing agent.
