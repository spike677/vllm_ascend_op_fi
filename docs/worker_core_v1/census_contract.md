# Worker Core V1 natural-call census

Status: **CPU_only**. This change has no Native build, NPU run, deployed-model
acceptance, C1/C2/C3 result, or NPU recovery qualification. Existing experiment
evidence is neither rerun nor reclassified.

`WorkerCallCensus` in
`src/demoncat/runtime/vllm_ascend/worker_extension/call_census.py` is a trusted
deployment helper for an already loaded Worker. It reuses
`engineering_closure.path_discovery.observe_execution_path`,
`worker_request_probe`, `NaturalTensorObserver`, and the typed `NativeObserver`.
It never starts a Worker, calls `forward`, enqueues a request, initializes or
selects a device, compiles/loads a Native library, or writes a tensor. The host
owns normal serving requests and the runtime owns authentication and exclusion
against other census/fault owners.

## Runtime connection

```python
# Under the existing Worker runtime owner lock, with its current validated
# IdentityTracker.read(worker) result. Do not call the identity RPC recursively.
census = WorkerCallCensus(
    worker, context.worker, phase="prefill",
    expected={"internal_request_id": actual_enqueued_id,
              "prompt_token_ids": frozen_prompt_tokens},
    max_events=10000,
)
owners[owner] = census                 # retain BEFORE any hook installation
census.start()                        # does not run the model
# The serving host executes its existing normal request independently.
census.stop(worker, current_identity)  # nonblocking: STOPPED or DRAINING
census.cleanup(worker, current_identity)
receipt = census.query(worker, current_identity)
if receipt["releasable"]:
    save_receipt(receipt)              # preserve failure history too
    del owners[owner]
```

`start`, `stop`, and `cleanup` may raise. The caller must retain the helper and
query it after an exception; no `finally: del owners[owner]`. Retry cleanup on
the same owner thread and original Worker/runner/model identity. `query` exposes
state even when a supplied identity or thread differs, with `identity_valid`
and `releasable` false; it does not perform scheduler/device queries.

The constructor binds objects but installs nothing. Its identity is supplied by
the trusted runtime, then checked against PID, rank, configured device and live
Worker/runner/model objects. Each RPC supplies the current complete identity;
generation or model replacement must be refused, with the original helper
retained for owner-scoped recovery.

The host may provide `native_observer_factory` directly in trusted deployment
code. It returns a dictionary from exact dispatcher names to existing, exact
`NativeObserver` instances. This is **never** an RPC value, import path, source
snippet, arbitrary callback, or configuration factory. The returned readers
must already own their reviewed read-only ABI; the factory must not create
device/Worker lifetimes. Native windows exist only within synchronous dispatch
scopes; the existing typed observer releases its per-source lock in `finally`.
Native read failures remain in each window and make observation status FAILED.

## Scope and evidence

V1 binds one actual request. Prefill requires the existing reviewed single
fresh-prefill scheduler adapter with capacity 1. First decode accepts one
expected request at reviewed capacity 1 or 8:

```python
expected = {actual_enqueued_id: {"prompt": frozen_prompt_tokens,
                                 "logical_id": host_logical_id}}
```

`logical_id` may be a nonempty string or integer, as in the existing scheduler
fixtures. Multiple expected IDs are rejected before installing hooks. Additional
actual IDs, mixed/unreviewed scheduler contexts, wrong input lineage, or wrong
request phases are retained as `request_failures` and NOT_VISIBLE request edges.
They do not become qualified targets.

The existing report retains actual module/callable/tensor object identities,
request object evidence, process/Worker/model generation, rank/device, every
observed dispatch, all raw Native windows, dropped-call counts and unobserved
Native/tiling/kernel edges. No model name, module name, or matching shape is
promoted to a Native call or kernel identity. Kernel edges remain NOT_VISIBLE
unless independently supplied evidence explicitly supports a different status.
`qualification_granted` is always false. Native absence is not inferred from an
unconfigured observer.

## Release conditions

| Condition | Observation / owner result |
| --- | --- |
| Running or draining forward/dispatch | `releasable=false`; no forced unwinding |
| Original Worker, runner, model, generation, or owner thread differs | `identity_valid=false`, `releasable=false` |
| Owned hook replaced by foreign callback | Preserve foreign hook and original handle; `releasable=false` |
| Hook remove raises or silently leaves its hook installed | Keep handles, mark cleanup failed, retry actual removal |
| Partial start followed by failed cleanup | Keep helper, original installation error and retryable cleanup handle |
| Actual request binding fails or includes another request | `observation_status=FAILED`, `request_binding_valid=false` |
| Stopped; original identity confirmed; no in-flight call; all owned hooks and metadata/dispatch scopes removed | `resource_cleanup_confirmed=true`, `releasable=true` |

The final resource condition can release an owner after a failed observation.
It does not erase errors, validate request evidence, or grant healthy-output,
Native, target, C2/C3, or recovery acceptance. Empty captures are similarly
failed observations with independently recoverable resources. Complete cleanup
allows subsequent ordinary calls on the same Worker/model; the test verifies
unchanged CPU output and that subsequent calls are outside the closed census.

The older `worker_path_finish` now removes its Worker capture attribute only
after cleanup is confirmed. Failed generator `__exit__` calls are not retried:
Python exhausts those generators even when cleanup raises. A separately retained
cleanup callback retries the real handles and checks that removal occurred.

## CPU verification

Run with the provided CPU environment and `PYTHONPATH=src;fault39/src;.;tests;tests/worker_extension`:

```text
python -B -m unittest worker_extension.test_core_census engineering_closure.engineering_tests.cpu.test_execution_path -v
```

The new contracts use real CPU Torch dispatch and the existing synthetic
scheduler/device metadata fixtures. Device APIs are guarded against use;
`torch_npu` is unavailable during these tests. Tests cover natural capture,
same-Worker normal output after cleanup, exact identity/thread checks, actual
multi-request refusal, in-flight drain, cleanup exception, no-op removal,
foreign hook retention, partial installation recovery, typed Native readback
failure, and the legacy lost-owner regression. These fixtures prove lifecycle
and observation contracts only, not NPU scheduling or Native execution.
