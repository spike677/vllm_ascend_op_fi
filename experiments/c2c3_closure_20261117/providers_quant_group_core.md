# Quant group core, ABI draft v1

Status: CPU memory contracts only. This change adds an independent group core
and fixed-field ABI payloads; it does not load a DSO, register a provider, call
CANN, or grant a target C2/C3 qualification. The legacy C1 source, DSO, build,
and numerical mechanism are unchanged. Unknown Attention/MoE row provenance
continues to block the entire group.

Files:

- `fault_injection/native_adapters/batch_scale_group_core.h`
- `fault_injection/native_adapters/test_batch_scale_group_core.cpp`

The first file is a C++17 header with an injectable `Ops` device boundary. It is
not yet an exported C ABI or a production shim. `Call`, `Config`, `Lease`,
`NativeEvent`, `ChildPlan`, `ChildReceipt`, `Receipt`, `State`, and `LiveWitness`
define the data that the subsequent separate group DSO must expose. Any change
to their layout requires a matching source/build fingerprint and bindings.

## Model calls and actual Native calls

One model invocation gets one independently observed parent token and one
coverage ticket. `Fault` requires **two real DynamicQuantV2 executions** on the
same current input:

| Role | Actual descriptor domain | Output storage | Census |
| --- | --- | --- | --- |
| Correct reference helper | `[M,H]`, per-row scale `[M]` | Owned Q and scale | Its own GWS enter/exit and execute enter/exit |
| Model effect | `[1,M*H]`, scale `[1]` | Separate owned Q and scale capacity `[M]` | Its own GWS enter/exit and execute enter/exit |

The helper executes and is synchronized/read back before the effect executes.
The effect scalar scale is copied to all M owned scale entries, preserving the
existing flatten-domain fault mechanism. The core compares the full actual Q
and scale bytes from these two calls. Both the reference and effect must use
the current input's full extent, and that input must remain byte-identical
before/after both executions and publication. The current F input may differ
from healthy B after upstream propagation. Healthy B is never used as the
numerical oracle for F; the CPU fixture supplies explicit output bytes and
implements no quantization or rounding formula.

No Q or scale is published until both real calls are accounted for, drained,
read back, and different. Equal Q and scale returns `NoEffect`, publishes
nothing, and retains the transaction for recovery. Lossless uses one correct
Native execution into owned outputs. B and OFF use the original public output
storage and one correct execution. Lossless and OFF must equal that target's
frozen healthy B, including its full input. A repeated B can only revalidate
the same bytes; it cannot replace them or inherit old qualification for new
bytes. B is prohibited once the set is sealed.

`Journal` receives each actual Native event before owner/role classification.
An admitted F has one parent event plus eight Native events, two GWS calls,
two executions, and one explicitly classified helper. Raw helper events are
never removed from the Native census or relabeled as a second model target.
The eventual `CoverageSession.finish(..., original_calls=1)` describes only the
single completed public model invocation; its artifact must also expose the
two Native child executions and all eight events.

Each child records parent, child lease, owner/generation, thread/device,
registration, executor, actual views, stream, workspace, statuses, and actual
original-call counts. Required ordering is
`parent < GWS-enter < GWS-exit < execute-enter < execute-exit`. Reusing an
executor address requires a new registration. Old/cross-parent/cross-role
tickets are rejected. Raw counts, errors, preparing/pending/executing calls,
unclassified events and overflow remain visible even when admission fails.
Actual calls that touch retained resources must have the exact matching core
callbacks; a completed but unbound raw call cannot make release safe.

## Ownership, budgets and recovery

The group owns Q/scale temporary buffers, helper workspace, alias descriptors,
per-call host readbacks/backups, and per-target healthy references. The model
effect borrows the actual framework workspace; its pointer/capacity is bound
to the actual execution and is never freed by the group. GWS-required sizes,
owned workspace, borrowed workspace, and execute pointer/capacity are explicit
receipt fields. The device budget includes owned buffers/workspace plus the
borrowed workspace capacity while the call is active. `State.device_bytes`
reports owned allocations and `borrowed_workspace_bytes` reports the retained
external workspace lease separately. The latter is not evidence that the
framework allocation was deallocated.

Allocation overlap with public input/output, other owned buffers or helper
workspace is rejected. An allocation returning an unsafe alias is quarantined
and cannot be freed as newly owned memory. Partial allocations, failed frees,
and failed descriptor destruction remain tracked. The fixed host budget
covers scoped references/readbacks; the independent Journal has its own
bounded event capacity and retains its census after group release. Those
retained audit records are not reported as outstanding target leases.

The core supports one active model transaction at a time. Configured bounds
include target count, admitted model-call count, owned/borrowed device extent,
scoped host extent, and independent Journal capacity. Admission consumes the
call budget before allocation. A failed transaction remains active and cannot
silently rearm because it has zero published hits. The common session must
add its deadline/concurrency/once/every-match admission policy; this core is
not a substitute for that authorization.

Before recovery releases any retained transaction or target reference:

1. Every target must have received stop.
2. The raw observer must have no GWS in progress, pending executor, executing
   call, overflow, observer error, or unaccounted event touching resources.
3. Every target and actual child stream must pass real synchronization.
4. The whole-group barrier must pass. Stopping one target also blocks the
   normal acknowledge path from releasing buffers before all targets drain.
5. Any public input read or output restoration requires the same call's live
   token and exact views. Partial publication is restored byte-for-byte and
   read back before release. First failure and latest recovery error remain
   separate receipt fields.

`LiveWitness.lifetime_token` is an association to a future shim-owned strong
Tensor/allocator-lifetime registration. The shim must verify that registration
is still live; accepting a caller-supplied integer would not prove liveness.
Same address/shape/descriptor with a new lifetime token is rejected. CPU tests
verify this association only, not real Tensor ownership. A byte rollback is
explicitly distinct from a healthy same-Worker OFF request.

A GWS error with a nonnull executor, prepared-but-not-executed child, missing
raw callback, or unresolved observer error retains resources. No nonexistent
executor cancel/destructor is assumed. Retry is possible only when real
missing completion/synchronization/ownership evidence becomes available;
stream, copy and release failures have retry contracts. Returning from
execute is not a drain proof. A nonzero real execute status now retains its
raw pending registration: there is no verified CANN contract establishing
executor consumption on that error path, and later successful stream sync
cannot supply it. Input changes are
reported and never repaired by overwriting the model input.

## CPU validation

From a host with g++:

```text
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  fault_injection/native_adapters/test_batch_scale_group_core.cpp \
  -o /tmp/test_batch_scale_group_core
/tmp/test_batch_scale_group_core
```

The test executable uses host memory and explicit simulated Native outputs.
It does not call CANN, launch a container, or access an NPU. Contracts cover
complete two-target qualification, current-input helper/effect comparison,
the exact 1-parent/2-Native-execution census, no-effect rejection, last-byte
input mutation, wrong role/thread/registration, executor address reuse,
GWS/execute/copy/sync/release failures, actual workspace binding/budget/alias,
GWS and callback omissions, repeated B, unresolved/incomplete group refusal,
stop/ack bypass, whole-group drain failure, live-token reuse, and overflow.

Host attempt `providers-quant-group-core-cpu-a1` passed the initial core and
contracts. Attempt `providers-quant-group-core-cpu-a2` passed the first core
commit's 19 test functions with the strict g++ command above. Its header SHA256 is
`18592e6ca6521dcc28cb7175e4d0fc263ac3687e36bc7e8ca5d3c815696e7f32`,
and test-source SHA256 is
`5e9d2b5a33f52af9fe81fd355b6fff766a72e99f56b9cdd4389a4b40d85605ce`.
The optional ASan/UBSan build in that same attempt failed at link time because
the host has neither `-lasan` nor `-lubsan`; no sanitizer binary ran. Both the
successful CPU execution and failed optional build are retained in the
receipt. No dependencies were installed. Raw CPU receipts remain under
`/home/17-fault-c2c3-20261117/` and are not copied into Git.

## Required second-stage integration

These gates are pending, not claimed by the CPU core:

- A separate SDK-built group DSO with ABI exports, actual `dlsym` GWS/execute
  call sites, complete independent raw observation before classification,
  truthful helper/workspace accounting, and source/ELF/build verification.
- A serialized, thread-safe Native boundary. Journal/Engine are deliberately
  not internally synchronized; the shim must hold the reviewed mutex for all
  transitions, obtain actual current thread/device, reject overlap, and keep
  raw observer events rather than discard a concurrent call. A numeric
  `max_concurrency=1` check is not a mutex.
- Strong references to the real model input/Q/scale and framework workspace
  lifetime until success or full recovery, and real validation of every live
  token before reading/restoring public memory.
- Managed Python loader/admission/arm mutual exclusion with old C1, using
  `quantization._LEASE`, before `CDLL`/`BackendControl` and on each arm. The new
  group must reject mapped old C1 DSOs and continue checking actual mappings.
  Long-lived mappings keep the other mode ineligible after a Python lock is
  released. This is scoped to managed entry points; arbitrary manual
  `dlopen`/direct legacy ABI calls are not claimed to be blocked.
- Two complete real inventories, stable process/Worker/module/site/method/
  build identity, actual row-provenance proof, and B/lossless/OFF for every
  eligible target. Unknown Attention/MoE edges remain UNVERIFIED in the
  denominator and block the complete group. Executor addresses stay per-call
  child identities rather than becoming persistent target identities.
- Shared `CoverageSession` parent tickets, complete raw/public denominators,
  per-target effect artifacts, full stop/drain/recover hooks, and same-Worker
  healthy OFF plus zero owned leases/device/host/inflight evidence. Native
  stop failures must raise, not rely on an ignored callback return value.
- C3 authorization using only the original same-process live C2 proof object,
  unchanged owner/scope/target set and incremented generation; copied or
  serialized proof reports cannot authorize F admission.

No new natural target set, C2, C3, real numerical calibration, live library
isolation, or same-Worker healthy OFF result follows from this core commit.
E5 still requires the separate raw B/F statistics comparison with fixed
`rtol=1e-5, atol=1e-8`; the exact-byte comparisons here are mechanism/recovery
contracts rather than E5 first-difference selection.

The later typed-shim increment and executor-error retention update are
documented in `providers_quant_group_shim.md`. Their fresh source hashes and
CPU/SDK attempts supersede the initial core source for the next group bundle;
the initial receipt is not relabeled as a validation of those newer files.
