# Fault18 ordinary single-target task

This product route uses the existing `WorkerFaultRuntime`, owner/generation,
`WorkerFaultClient`, normal serving enqueue and actual root/logits hooks. It
does not import an experiment, create a Worker, call `model.forward` from RPC,
run `CoverageSession`, perform a full-layer census, or enable a collector/dump.

The trusted deployment installs `install_rms18_task_binding(worker, ...)` from
`serving_tensor_binding.py`. The public mechanism remains `18.api_half`.
`prepare/register/check/enable` installs idle owned hooks. `advance(request_id)`
sets an expected actual scheduler ID; the host then submits/completes its
ordinary request through its existing serving engine. Four matching fresh
prefills have roles `normal`, `fault`, `off`, `health`. RPC does no inference.
`finish/unregister/detach` completes the same Runtime lease.

The normal request observes only the first supported actual RMS return, then
requires one real numerical consumer. No later RMS return is copied or given
an event. After that first consumer, operator observation becomes passthrough.
If that first return is unconsumed, the bounded task refuses; it does not scan
all layers searching for a better target. The existing structural/actual
native-weight/return-lineage adapter identifies the site; callers give no
layer name. The supported normal input is one TP1 serial eager fresh prefill
with a discoverable real logits boundary. Other shapes/schedulers remain
unadapted rather than silently generating a replay.

For F, the live module and callable, operator/schema, complete target input,
original return, weight bytes/storage and complete root input must match this
task's normal. Exactly that return uses the original `scale_value(value, .5)`
action. This remains a module-return half-value proxy, not a native RMS reduce
error. Normal and all unselected returns retain their original objects through
the new optional passthrough mode; its default is false so historical group
behavior is unchanged. OFF/health install no RMS observer or action.

Only normal/F read back the selected return; all four requests retain complete
hidden/logits hashes for matched OFF and health recovery. Small bounded effect
summaries use `actual_request_effect`. They are not msprobe statistics and grant
no C1/C2/C3/E5 qualification. `current_request_effect` is unverified while the
next request is pending; `last_fault_effect` remains explicitly historical.
Failed F cannot be retried. It requires actual OFF and health; attempts and the
first failure are never cleared. A new task needs a new Runtime owner/provider.
The old explicit `install_tensor_serving_binding(first_live_validation=True)`
continues to select the unchanged ten-request group validation path.

CPU contracts cover the real public Host/Runtime and real torch operations;
Worker metadata and scheduler are explicitly simulated. The changed ordinary
route passed Source126 blue-zone normal/F/OFF/health validation and independent
original-record review; see [the exact scope and pins](review_rms18_ordinary_source126.md).
Yellow-zone main-project wiring remains pending. Four same-prompt Qwen requests
showed one automatically selected return-half effect and actual consumer, exact
full hidden/logits OFF and health, and zero owned hooks/modes/leases after finish.
This is not a ten-request or all-target coverage re-run. Current Worker query
validation fields do not inherit this prior blue-zone result.

The separate acceptance-only entry is
`experiments/worker_core_v11_20261118/prepare_rms18_single_task.py`.
It reads `--source`, `--installed`, `--source-manifest-sha256`, `--card 2|5`,
and a fresh `--out` case-manifest path, verifies this probe's own source list
against installed bytes, and prepares one case for the existing root queue.
There is no native library/build argument. Its probe enqueues four real requests,
keeps their external/internal mapping, and writes `probe/result.json` plus the
original `probe/host_receipt.json`. Failure preserves the first error and may
enqueue at most two recovery requests; it never retries F. Eight CPU driver
contracts passed; preparation and CPU success are not NPU acceptance.

## Source105 / Fault33 audit boundary

`Paged33S2ServingProvider` still binds its Source105 normal/reference/lossless/
qualification-OFF/F/recovery-OFF phases to the native live normal gate. Merely
renaming its driver ordinary does not remove those dependencies. The minimal
future split is an ordinary single-site owner task around the existing native
execution/input/restore adapter, with current live normal admission distinct
from the historical C1 report. Removing phases without replacing that live gate
would weaken admission. No Fault33 mechanism, gate or evidence is changed here.
