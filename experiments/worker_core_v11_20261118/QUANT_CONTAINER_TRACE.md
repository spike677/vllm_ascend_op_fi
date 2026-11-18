# Four-container diagnostic implementation

This implements only the next hypothesis in `QUANT_NATIVE_ROUTE_GATE.md`:
observe actual `aclCreateTensorList`, `aclDestroyTensorList`,
`aclCreateIntArray`, and `aclDestroyIntArray` calls. It does not implement a
GWS/execute proxy, fault, graph admission or routed-row certification.
The previous 10/135 request-row scope remains unchanged.

The shim returns the original pointer/status and calls each real function
once. It records current function/caller PC and exact reviewed owner,
entry/return timing, copied host integer or public descriptor metadata, and a
fresh registration for each returned container. A failed destroy retains that
registration; release refuses until all recorded live containers retire.
Destroying null is preserved as a real call but is not an object lifetime.
Pointer reuse after retirement gets a new registration. Metadata exceptions
poison the diagnostic, release any SDK getter arrays, and do not suppress the
original call. The shim never allocates/frees a model tensor, reads activation
bytes, dereferences saved descriptor pointers later, or calls a numerical API.

The new private custom proxy depends on the original custom library by its
fixed absolute image path and on actual SDK `libnnopbase.so`. The original
library's bytes and SONAME remain unchanged. The existing exact-image runtime
identity helper checks the real NN functions and original custom executable
segments. Caller classification independently checks the current PC against
the fixed stock `libtorch_npu` or custom extension image; unknown callers
remain explicit. No global `dlsym` interposition or PTA-cache change is used.

Budgets are 4096 invocations, 8192 entry/return events, 128 members/integers per
creator and 8 MiB copied metadata. Event buffers have separately reported host
capacity. Overflow never changes original execution and permanently invalidates
the diagnostic. One installed diagnostic can be released but not reinstalled
in that process. Snapshot queries are read-only; stopping does not erase
records. After stop, a real destroy of a retained registration can still retire
it. Release frees only observer records and copied metadata. The preloaded
interposer and exact-image mapping pins remain resident until Worker exit;
zero scoped buffers/registrations/getter arrays does not mean an unloaded DSO
or zero process-wide host memory. The snapshot labels this resource scope.

## Build handoff

Only root executes this command on 124, with a fresh output directory:

```text
python3 <frozen-source>/experiments/worker_core_v11_20261118/build_quant_container_trace.py \
  --source <frozen-source> --source-commit <40-digit-frozen-commit> \
  --out /home/17-fault-c2c3-20261117/quant-container-trace-sdk-a3
```

The builder pins production image `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`,
NN/custom/stock/custom-extension bytes, public SDK metadata header and source
closure. It uses an owned, uniquely named, no-network `runc` CPU container with
no device mounts, 4 CPUs, 4 GiB and a 300-second bound. Timeout and exceptions
preserve a receipt and clean only that exact owned container. The source mount
is read-only. A successful manifest is published only after compiler, exported
ABI, 19 actual-shim CPU-double scenarios and marker-only actual DSO load pass.

The first root SDK attempt stopped before C++ compilation because changing the
original SONAME also changed its executable-segment profile. That failure is
preserved. The replacement build edits only the newly compiled wrapper to add
an absolute DT_NEEDED. It first runs a real CPU ELF contract with equal
original/wrapper SONAMEs: basename lookup must select the wrapper, absolute
NOLOAD must select the original, the delegated anchor must belong to that
original mapping, and original SHA must remain unchanged. No segment masks or
identity exceptions are used.

Root's second build passed the real same-SONAME ELF contract, then stopped on
two shared-header warnings treated as errors before the C++ scenarios ran.
The next source confines an indentation diagnostic suppression to the existing
byte-pinned identity header include; our TU and fixtures remain under strict
warnings. Actual metadata capture exceptions now call the shared exception
counter before rethrowing to the ABI guard, with array cleanup still tested.
Both failed build originals remain preserved.

Local validation so far: command/receipt contracts passed; no Docker or
SDK compiler was run by this agent. The 19 C++ scenarios are **not yet
executed**. A successful SDK/CPU receipt will still not prove the stock resolver
actually calls these functions in inference. The B8 diagnostic and its exact
loaded-owner/current-source results remain separate required evidence.

`libcust_opapi.so` must be preloaded before Worker startup; loading it during an
existing Worker session would not replace cached function pointers. Keep the
existing quant vendor first in `ASCEND_CUSTOM_OPP_PATH`, then this new vendor,
then the original vLLM custom vendor. Putting this proxy first could resolve
DynamicQuant through its real dependency and bypass the existing complete
Native/public census. The managed binding must attach by `RTLD_NOLOAD` only.

## One B8 decode handoff

The managed binding now verifies the approved bundle, actual owned mapping,
all control-function owners, profile and ABI sizes before attaching by NOLOAD.
The opt-in source observer binds current reviewed routing/GMM1/GMM2/unpermute
frames to exact current public schemas, arguments, input objects and write
epochs. It emits no graph row bindings. It records only descriptor metadata
and scalar arguments; the integer route snapshot helper is not instantiated.
Scope-end failures retain the handle and original failure, and record each
retry return code. A source-owner/profile error leaves the Native raw census
unclassified and rejects the diagnostic.

`probe_quant_containers.py` performs healthy-before, one fresh B8 prefill and
first decode, explicit collect, observer removal/release, and same-Worker
healthy-after. It requires all 135 structural candidates, 135 actual projection
and Native prepare/execute calls, 270 public API calls, complete public/Native
bindings and unchanged healthy tokens. It does **not** change the old two-pass
inventory's `complete_healthy_replays`/`recovered` qualification fields to true
after one pass. Those remain historical scope fields; the new result separately
reports this one-pass diagnostic's census, cleanup and health. No fault owner
is installed.

The raw Native record is saved before release. Every cleanup retry saves a new
immutable `cleanup-observation-NNN.json`, so a release failure does not prevent
retry or overwrite the first observation. Failure during model construction
also attempts a same-process NOLOAD state read; it never reloads the library.

After the fresh SDK build and fresh installed source package pass root review:

```text
python3 <frozen-source>/experiments/worker_core_v11_20261118/prepare_quant_container_case.py \
  --installed <fresh-installed-package> --quant <existing-reviewed-quant-group-bundle> \
  --trace <fresh-container-trace-sdk-bundle> --source-commit <frozen-worker-source-commit> \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --model /home/haoran/models/DeepSeek-V2-Lite-W8A8 --card 2 \
  --probe <frozen-source>/experiments/worker_core_v11_20261118/probe_quant_containers.py \
  --out <new-reviewed-case.json>
```

This only prepares the JSON; root's existing queue and device gate own launch.
All inputs mount read-only, evidence stays on 124, and the outer 1800-second /
80-GiB / 16-CPU case budget stays unchanged. Do not reuse the failed SDK a1/a2.

New source-only closure: `fault_injection.quant_container_native` (depends on
existing `fault_injection.metadata_arrays`) and
`demoncat.runtime.vllm_ascend.worker_extension.quant_container_scopes`. The
existing `quant_source_rows` and `serving_shared_scale_group_inventory` have
opt-in glue; all default paths remain unchanged. Probe setup records hashes
for the loaded observer/Native-binding modules. Root owns the common exporter
and must include these new modules in the frozen package closure.

Local validation passed all 328 quant tests, including 21 new
binding/scope/cleanup/manifest tests. The subsequent shared-header compile fix
passed all 8 build command/receipt contracts. These are not real stock Native
entry evidence.
Stop before F on missing actual entries, current caller/scope ambiguity,
unmatched/nonretired registrations, source errors, truncation/overflow,
incomplete 135 census, or changed health. Even a successful diagnostic only
answers creator/destroy observability; GWS/member/executor use and routed-row
semantics remain unproven.
