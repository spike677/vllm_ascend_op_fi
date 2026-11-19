# TND28 prepared executor: cancellation and the next single-target change

Read-only source/ELF review, 2026-11-19. This document adds no executable gate,
does not enable F, and does not qualify a normal run. The previous denominator
candidate completed its a2 SDK build; the new compiled-input observer requires
its own successful build and actual normal connection. Its build failures and
source113's zero journal remain failures, not evidence of execution.

## Exact evidence and types

The generated inner unit is pinned by `tnd28_input_compile.py` to SHA
`d35f8b60d1ad50d97b7cf0d2050be5c23fd0d53eba77a58c053c184d96af4fe2`.
Its IntArray GWS creates a parent with `NnopbaseGetExecutor`, obtains a
Contiguous child, stores it with `NnopbaseSetUnContExecutor`, and performs FIA
host tiling. Execute runs Contiguous first, then `NnopbaseRunWithWorkspace`.
There is no ViewCopy call in this exact path. A parent is a pooled
`NnopbaseExecutor` carried as `aclOpExecutor*`; a child is a separate ordinary
`aclOpExecutor` or a cache wrapper. They are not interchangeable lifetimes.

Read-only reference sources are under the retained
`tnd28-prefill-normal-build-a2/build/run/dependencies/opbase` tree:

| Source and relevant lines | SHA256 |
| --- | --- |
| `src/nnopbase/common/api/acl_op_api.cpp:491`, public destroy dispatch | `f4419e64b924aad1e484b2b914807514af6f8e214ab1b339be0f16313a4e7317` |
| `src/nnopbase/composite_op/aclnn_engine/op_executor.cpp:1399`, CommonOpExecutorRun | `22cdc0008900d6e5a96c89f616307414b9993e6726ab03d9de67dad3db25a48d` |
| `src/nnopbase/individual_op/executor/indv_executor.cpp:417`, child reset; `:438`, parent reset | `d2e20768a962d0eb8e6cafb4d704d778c8ac265355c3a3b0ae4baf85439bbd16` |
| `src/nnopbase/individual_op/executor/indv_executor_ext.cpp:369`, clear; `:1177`, child setter | `b75869d495669af2186251704a7e7232bd84f3b49015393f50f6b81cad55b951` |

These dependency sources explain the control flow; they do not establish that
their newer private C++ layouts match production. That distinction matters:
the dependency `common_types.h` uses `gert::TensorV2`. The actual fixed-image
CANN 8.5.1 headers are under `include/aclnn`, with different bytes:

| Actual SDK header | SHA256 |
| --- | --- |
| `opdev/common_types.h` | `4e365e2b0805beb6f97e1cbcd686724aff778110797d17ffdb8724330bbbf620` |
| `opdev/op_executor.h` | `c79a9b705fc0e0b795bc66bc5e5c1fee15b0ca79b7ad48db5c6b6e16bc615817` |
| `acl_meta.h` | `be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed` |

Compile against those SDK headers and verify actual dependency-file paths,
including `fast_vector.h` and the shape headers. Do not add the dependency
opbase headers to make compilation succeed. `op_executor.h` itself includes
`opdev/*`, so the exact SDK `include/aclnn` directory must be available.

The staged production ELFs were rehashed, then inspected with `nm`/`objdump`;
neither was loaded or executed. They are from the fixed production image:

* `libopapi.so`, 37,858,936 bytes, SHA
  `b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c`:
  `aclnnContiguous` at `0xda1458` calls CommonOpExecutorRun at `0xda14d4`.
* `libnnopbase.so`, 4,000,424 bytes, SHA
  `6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99`:
  CommonOpExecutorRun is at `0x1c9f08`. Its nonrepeat path invokes the deleting
  destructor at `0x1ca3bc..0x1ca3c8` before returning, also on a returned error.
  Cache-wrapper and repeatable paths differ and must not be inferred from a
  successful return alone. Public `aclDestroyAclOpExecutor` at `0x10be30`
  dispatches the parent magic to `NnopbaseResetExecutor` at `0x2911c8`.
  `NnopbaseSetUnContExecutor`/`GetUnContExecutor` and
  `NnopbaseGetViewCopyExecutor` are actual exports. The child setter writes
  null/zero without dereferencing a null child. Parent reset then clears args
  and marks the pool slot idle; it does not free the parent allocation.

Source and ELF both show that parent reset may inspect/delete the child and
then clear the parent even when child cleanup reports an error. Therefore a
copy-returned parent cannot be passed directly to public destroy while it
still contains the consumed child's address. A scalar return error is also
insufficient evidence that no resource was released.

## Two-stage ABI and cancellation

Do not implement the existing `current_input_v1` as a weaker pending boolean:
its current contract claims the prepared-input join already happened, which
cannot be true before Slice executes. A new versioned interface must separate:

1. **Prepare plan**, immediately before original FIA GWS. Bind the actual
   public objects, typed descriptors/attributes, returned child and offsets,
   parent, owner/generation, module object, request, PID/start/native TID and
   current observed scope. A private live record permits only the pending
   host key plan; it makes no copy-completed or launch claim. Saved JSON and
   environment strings alone cannot construct this record.
2. **Execute admission**, after the one original Contiguous call returned and
   immediately before original FIA execute. Revalidate the same live record,
   actual parent/workspace/stream, child-offset snapshot, current public-input
   lifetime and unchanged values. Require the original copy return and its
   observed current-scope launch. Only this state may enter the replay FIA
   call. Later FIA launch/POD/output bytes corroborate it; they do not authorize
   the earlier host decision. Consume admission once, including failed calls.

Before executing the child, use the actual SDK getters to establish ordinary
child magic, `IsRepeatable()==false`, and no ViewCopy child. Unknown/cache or
repeatable types refuse F; do not change repeatability to simplify ownership.
After a normal return, retire the child handle as consumed without reading it
again. For a rejected post-copy admission, the required cancellation sequence
is actual same-stream drain, parent-child detach with
`NnopbaseSetUnContExecutor(parent,nullptr,0)`, getter readback of null/zero and
no ViewCopy, then `aclDestroyAclOpExecutor(parent)` and parent getter readback.
Record each completed step before receipt I/O. Successful reset plus the
source/ELF control flow establishes pool release; getters alone do not expose
the private `isWork` flag and must not be described as doing so.

Before-copy cancellation owns a still-live child, so it uses the appropriate
public destruction path once, not the consumed-child path. No generated
reentry or pool-address reuse may satisfy the old owner's pending record.
After known parent release, never retry destruction through the retired
address, even if later reporting fails. Unknown C++ unwind, foreign owner,
failed drain or uncertain reset keeps the first error and pending owner; no
new F or healthy/released assertion is allowed. Pointer records do not retain
allocator storage: drain/cancellation must finish within the actual original
Execute invocation's borrowed workspace lifetime, or F must be refused.
Public input/output strong references do not by themselves own that workspace.

## Minimal implementation in the existing Attention path

Keep the existing Worker, Host/Runtime and controller. Add an exact trusted TND
adapter to the existing `AttentionDeployment`; specialize only
`FiaDenominatorReplayServingProvider` and its execution adapter. The current
`attention28_prefill.inspect_prefill_input` accepts BNSD, so TND must reuse the
actual `validate_tnd_prefill`/prepared-input evidence, not relax or relabel that
matcher. Preserve all 28 natural calls, but activate Native capture/intervention
only for the one held module object/occurrence chosen from the actual normal
call. No user layer name and no second full Native census are required.

For the same new package, establish the actual stock/private normal comparison
and M128/S2=128, eight-scheduling-core control comparison on this target; the
old normal report or key equality grants no authority. In the active Worker,
neither new comparison is presumed successful; a required exact-normal
mismatch stops F. In that same live Worker,
the bounded progression is normal, control, qualification OFF, one F, recovery
OFF, then ordinary post-unregister token health. F uses the current live normal
and control records, actual unchanged input and two-stage admission. The sum
recurrence remains the existing second-tile denominator replay; no output
scaling, BNSD conversion, new kernel algorithm or other mechanism is added.

Provider recovery must preserve the session needed for real OFF until native
drain/cancellation and OFF complete; the generic cleanup cannot simply remove
hooks and erase a pending F. Reuse `needs_off`, exact binding reservations and
the public disable/unregister/detach path. Query stays read-only. The existing
probe/preparer and independent reader need only these additional phase and
receipt checks, not a new Worker/controller or generic capability framework.

Before enabling F, CPU contracts must cover wrong/expired current objects,
foreign thread, parent reuse, replayed admission, before/after child return,
repeat/cache rejection, drain failure, detach failure/readback mismatch,
reset before/after release, reporting failure after release and real public
OFF/health recovery. Actual new normal/control and numerical F/OFF remain
device evidence requirements. Immediate Attention effect and downstream
hidden/logits propagation are reported separately; tokens alone are not a
numerical test. This is one target only: C2, C3 and E5 stay ungranted.

## Current public-input component and Execute borrow

`attention28_replay_binding.ReplayInputBinding` now reads the existing
`AttentionExecutionAdapter`'s actual synchronous API-call ledger. The ledger
holds the original callable, positional/keyword argument objects and current
call row; it is not supplied by RPC. The plan holds the actual Worker,
deployment reservation, discovered module, metadata, request and Q/K/V/mask
objects, and re-reads their values and geometry before use. Equal-value clones,
changed inference tensors, another call, owner or thread do not satisfy it.
Returned summaries are copies; a serialized/copy/replayed plan is not authority.
Local public-reference release is explicitly not a Native abort receipt. Its
F method still refuses because the two-stage Native bridge is not bound.

The Native bridge needs a **borrow within the original synchronous Execute
invocation**, not a new cross-request allocator owner. The real API caller
supplies a valid workspace span for that call. The generated shim may borrow
that span with its exact parent executor, stream, PID/native thread and plan,
provided it completes same-stream drain before returning or unwinding from the
borrow. It must neither infer allocation ownership from an address nor claim
that public tensor references own the workspace. The installed torch_npu
headers declare `unsafe_empty_workspace` and `RunOpApiV1/V2/V3` but do not expose
their bodies; no caller error-path allocation lifetime beyond this invocation
has been proven or is needed for a successfully drained borrow.

On rejected post-copy admission, cancellation is performed while that borrow
is live: drain, detach the consumed child, verify null/zero child and absent
ViewCopy, reset the parent, then record reset return/readback. Steps are never
repeated after their actual completion. A failed drain or unknown parent reset
at the end of this call is terminal uncertainty for this attempt. Preserve the
first error and diagnostic identities, prohibit further F, and erase every
operational raw pointer when leaving the invocation. An outside cleanup RPC
must not retry operations on that old workspace or executor address or claim
zero Native resources. This restriction does not require keeping the allocator
alive across requests; it prevents falsely asserting that it was kept alive.

The next Native increment must be called by these existing generated Prepare/
Execute wrappers and the existing Attention driver, with no alternate Worker
or controller. Its CPU contracts do not replace new-package normal/control,
actual prepared-copy/launch, failure/drain and numerical F/OFF device evidence.
