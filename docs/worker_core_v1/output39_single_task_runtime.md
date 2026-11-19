# Output39 ordinary single-task execution

This implementation separates the ordinary `39.output_proxy` task from the
retained six-stage / complete-set experiments. It is ready for integration
review, with CPU contracts only. The inner main project is unavailable here;
neither main-project wiring nor this new NPU request sequence is accepted yet.
Historical C1/C2 and failed cumulative C3 records are unchanged.

| Item | Current result / boundary |
| --- | --- |
| Reused execution | Existing Native BF16 postwrite open/apply/finite-original/restore/release, shared with the old 39 and 41 paths |
| Automatic target | Current actual weight operand owner, supported original projection return and observed value consumer; no user layer/address |
| Bounded selection | At most 16 actual projections in one normal request; only one pending candidate; no full-model census |
| Fault budget | One independent task, at most one target in its one fault request; spent objects are never reset |
| Controls | Four ordinary scheduled requests: normal, fault, OFF, health; matched fixed input and fresh prefill scope |
| Display | Same-return BF16 byte changes, finite→Inf/NaN counts, at most four bit-pattern examples; current request and prior fault fields separated |
| Recovery | Exact owned Native restoration and zero leases, plus separately matched OFF/health hidden+logits hashes |
| Collection | No default dump, files, msprobe, global profiler, full target inventory or control-RPC model forward |
| Open deployment boundary | Serial eager TP1, one fresh request, bounded exclusive contiguous ND BF16 return ≤1 MiB; existing build/source-specific custom operator checks remain |
| Main project / NPU | Implementation ready; actual wiring and new sequence not verified; all C1/C2/C3/E5 flags remain false |

## Execution and unchanged guards

`fault_injection/deployments/output39_execution.py` does not call `model.forward`.
The new public provider/session observe a request which the existing host has
enqueued. Existing scheduler, root-final-input, logits-row and held Worker /
model / thread checks stay in use. Unknown active dispatch modes are refused;
only the task's own mode is paused for diagnostic readbacks. Between root
return and logits entry, unrelated host tensor operations cannot become a
claimed model consumer.

During normal execution, a pending projection is observed at its actual return,
before any consumer. A bounded Native `read_span` records its original bytes in
memory. A new projection may displace an unconsumed pending candidate; an
unselected write invalidates that candidate. Neither is an all-target coverage
claim. The first supported real value consumer fixes the target. The consumer
must return without modifying the pending span: the complete bytes must still
equal the pre-consumer readback. Only then is the unchanged Native mode 0
lossless action performed and verified. This new **consumer-then-lossless**
normal order is explicitly different from the old C1 experiment and needs its
own NPU check.

The fault request binds a newly returned live tensor. It matches the normal
request's input hash, actual model/functions/weight objects, return geometry,
producer position and **pre-consumer** return hash. The unchanged Native guard
then validates finite original data and writes the selected prefix before the
actual consumer. The complete same-return before/after bytes prove that only
the chosen elements changed. Consumer API/schema, logical operand read views,
scalars, root/logits boundary and actual dispatch position must match normal.
Failure does not switch targets, weaken finite checks or turn a refusal into a
successful injection.

No CPU temporary is the mutation target. The target itself stays alive until
the original Native restore/release confirms recovery after real logits return.
The output fingerprint includes full hidden states and logits; arrays are not
saved. Changes to logits, Top-1 or text are not required: a proven internal
same-return change and real consumer are an explicit supported phenomenon.

One actual task may contain at most eight requests including bounded recovery.
Normal/F/OFF/health are the ordinary four. Failure keeps its first error and
owned resources. `recover` drains the exact window/lease and, after a fault
attempt, requires actual OFF and health requests before owner release.
`disable`/`clean`/`unregister` refuse premature release when that recovery is
incomplete. Successful cleanup never clears `attempted` or `_fault_attempted`.
The next task obtains a new Runtime owner, provider and normal binding.

## Integration seam

The trusted host's existing Worker setup uses:

```python
install_fault39_output_serving_binding(
    actual_worker, transport=verified_native_postwrite,
    prompt_token_ids=actual_task_prompt, runtime_only=True,
)
```

The exact `Output39ServingBinding` still has `stage='c1'` and
`max_live_leases=1`, but `runtime_only=True` and `folder=None`. These are trusted
deployment details, not additions to the public three-field `FaultConfig`.
The integration-owned `create_single_request_provider` guard should select
`providers.output39_task.Output39TaskProvider` for this explicit binding;
historical bindings retain their historical adapter. This change deliberately
does not edit the root agent's registry/guard work.

Ordinary `HostAdapter` / `WorkerFaultClient` actions remain prepare, register,
check, enable, advance(actual enqueued id), query, finish, unregister, detach.
Query performs no forward or recovery write. No experiment directory is
imported by the new runtime. The normal request is an actual workload control,
not a hidden forward inside prepare/register/check/query.

The single-return open/apply implementation was extracted into
`PostwriteProvider._apply_single_return`; both old and new paths use that one
implementation. Existing Native recovery is reused. No Native source, ABI,
finite guard, shared Runtime or historical experiment protocol was changed.

## CPU validation and minimal 124 check

The focused command is:

```powershell
$env:PYTHONPATH="$PWD;$PWD/src;$PWD/tests;$PWD/tests/worker_extension"
python -B -m unittest test_output39_task test_postwrite_providers test_postwrite_native_recovery test_core_output39 test_core_output39_custom test_serving_session_lifecycle -q
```

80 tests passed, zero skips: 15 new task contracts plus 65 affected existing
contracts. They use real CPU Torch operations, hooks and public Host/Runtime;
Native, scheduler and custom-op/source registrations are explicit CPU doubles.
They cover no-census/no-files, changed inputs/weight, nonfinite normal data,
absent consumer, an earlier unconsumed candidate, actual custom GEMM with RMS
and SwiGLU, unrelated host reads, foreign mode, exit-before/after-pop failure,
Native restore failure and fresh ownership for a later task. They do not prove
an NPU hit or main-project integration.

Root's next bounded check can reuse the verified Source101/Source68 postwrite
DSO and existing fixed image/Qwen3-0.6B deployment; no Native ABI rebuild is
introduced here. Recheck the actual mapped DSO/SHA/read-span ABI, current
`vllm.unquantized_gemm` and consumer registrations. The implementation supports
the existing source-pinned custom GEMM plus reviewed `npu_rms_norm` /
`npu_swiglu`, not just aten. New versions fail those existing checks explicitly.

Use the root's resource gate/lock and a single short acceptance driver, clearly
labelled **blue-zone public Host acceptance, not inner-main-project wiring**:

1. Standard vLLM initialization only: fixed image, real Qwen weights, the public
   Worker extension, TP1/eager, serial fresh prefill, prefix caching disabled.
2. Install the verified transport with the runtime-only binding, then the
   existing public client actions. No protocol stages or model calls inside RPC.
3. Enqueue the identical prompt four times with fresh internal ids; advance
   before each actual scheduler step. Check normal target, actual custom API /
   consumer, one Native F effect, exact owned restore, then exact OFF/health.
4. Finish, unregister, detach and remove binding. Save only bounded task/result
   metadata and required cleanup/failure evidence; no statistics/dump request.

Stop at the first unsupported source/geometry, no eligible consumer within the
bound, request or semantic mismatch, failed native finite/readback, or uncertain
recovery. Do not fall back to all-target research, reset spent flags or repeat a
fault to chase text changes. An actual failure stays a failure, with owner kept
until the existing recovery path succeeds. Once this one deployment passes,
the remaining external blocker is the unavailable inner main-project source.
