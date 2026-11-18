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
it. Release frees only observer records and copied metadata.

## Build handoff

Only root executes this command on 124, with a fresh output directory:

```text
python3 <frozen-source>/experiments/worker_core_v11_20261118/build_quant_container_trace.py \
  --source <frozen-source> --source-commit <40-digit-frozen-commit> \
  --out /home/17-fault-c2c3-20261117/quant-container-trace-sdk-a1
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

Local validation so far: command/receipt contracts passed; no Docker or
SDK compiler was run by this agent. The 19 C++ scenarios and new real ELF
contract are prepared, **not yet executed**. A successful SDK/CPU receipt will still not prove the stock resolver
actually calls these functions in inference. The B8 diagnostic and its exact
loaded-owner/current-source results remain separate required evidence.

`libcust_opapi.so` must be preloaded before Worker startup; loading it during an
existing Worker session would not replace cached function pointers. Keep the
existing quant vendor first in `ASCEND_CUSTOM_OPP_PATH`, then this new vendor,
then the original vLLM custom vendor. Putting this proxy first could resolve
DynamicQuant through its real dependency and bypass the existing complete
Native/public census. The managed binding must attach by `RTLD_NOLOAD` only.
