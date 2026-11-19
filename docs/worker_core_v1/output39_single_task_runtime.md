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

### Independent four-request acceptance driver

`experiments/worker_core_v11_20261118/output39_single_task_probe.py` is the
blue-zone validation entry, and `prepare_output39_single_task.py` only writes
one queue case. Neither is imported by the runtime/provider. The case fixes the
reviewed image `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`
and `/home/gyl/models/Qwen3-0.6B`; root selects an existing reviewed postwrite DSO
and rechecks its bytes. The recorded build key is explicitly root-provided
reviewed-build identity, not a newly performed build audit.

Normal must identify the actual `vllm.unquantized_gemm.default` return and a
supported consumer within 16 observed projections, with an original Native
mode-0 receipt. Otherwise F is never enqueued. Nominal execution enqueues
normal/F/OFF/health exactly once each, using the same prompt and distinct actual
scheduler ids. It separately checks the Native mode-1 finite-to-positive-Inf
effect, actual consumer, exact span restoration, OFF/health hidden-plus-logits
hashes, generated tokens/text and owned-resource cleanup. Text or logits need
not change for the local consumer effect to be observed. A failed request keeps
its actual mapping and first error; recovery may add only OFF and health, never
another F. The maximum is six real requests, not a reduced cumulative protocol.

The default case has a 480-second soft deadline and a 600-second queue limit,
including at most 120 seconds for recovery. Source/installed module identity,
Native SHA and model artifacts are checked before F. A refused boundary remains
failed, with retained ownership reported when recovery is uncertain. Results
and the complete bounded Host receipt are private metadata; no full census,
statistics or tensor dump is requested. Each metadata file has a 16-MiB limit.

Root deploys a fresh source bundle containing the runtime registry routing and
this driver, installs matching modules, and prepares a fresh case file. These
paths and hashes are deployment parameters, not end-user fault configuration:

```bash
PYTHONPATH="$SOURCE:$SOURCE/src" python3 -B -m experiments.worker_core_v11_20261118.prepare_output39_single_task \
  --source "$SOURCE" --installed "$INSTALLED" --library "$REVIEWED_POSTWRITE_SO" \
  --sha256 "$DSO_SHA256" --build-fingerprint "$REVIEWED_BUILD_KEY" \
  --source-manifest-sha256 "$SOURCE_MANIFEST_SHA256" \
  --card 5 --timeout 600 --out "$CASES"

python3 "$SOURCE/experiments/c2c3_closure_20261117/run_queue_124.py" \
  --cases "$CASES" --attempt "$FRESH_ATTEMPT" --cards 5
```

Only root runs the second command after the existing per-case resource lock and
device gates permit it. The preparer adds no duplicate collector mount, does
not execute a subprocess and never claims an NPU result. The existing queue
owns the temporary container/Worker lifetime. Its cleanup report is separate
from the in-Worker resources checked above.

Eight additional CPU contracts test the actual public driver ordering, original
Native receipt checks using an explicit ABI double, incomplete request mapping,
lost advance reply, no F retry, before-F rejection, primary error preservation,
queue bounds and source/installed mismatch. Run them with
`python -B -m unittest test_output39_single_task_probe -v` under the same
`PYTHONPATH` as above. CPU success is named `CPU_OUTPUT39_TASK_DRIVER_CONTRACT_ONLY`;
it does not grant NPU, main-project, C1/C2/C3 or E5 acceptance.

### Source114 normal refusal and limited callable adaptation

The first Source114 acceptance attempt that passed the device gate used commit
`0b677c7e4ba14e955602a65157cbc38d8ca765f1`, source manifest
`6b15fe311d4f3a2a358b6b98b3a6f783217f032efc132ce2469a9e31420e4445`.
Its first normal request refused with
`UNREVIEWED_TENSOR_CALLABLE_CLOSURE:forward`. It never reached F. The owned
pending request was aborted, Runtime ownership was released and the deployment
binding removed. This is cleanup after a rejected normal request, **not** a
successful numerical OFF/health comparison. The original failed result and
Host receipt remain unchanged and are not published as raw data here.

The actual trace goes through `AscendQKVParallelLinear.forward`, whose reviewed
image source calls `super().forward(input_)`. Python creates a `__class__` cell
for that method. The generic TensorCallableGuard rejects all closures. The
fixed-image files copied read-only from an already running, verified no-device
build container were `vllm_ascend/ops/linear.py` (SHA
`fcf2520f8818cb02e6b27a7b9a926745c5c36f1af08a01d9be0627ce2c87316e`)
and `vllm/model_executor/layers/linear.py` (SHA
`b8be54d8f4849818da7b1ca97315854f5e60fcb3b5c6837cd13501589e47f6c1`).
This source check did not instantiate a model or call a device.

The adaptation is local to an already reviewed custom-GEMM return. It allows
only the implicit `__class__` cell pointing to the exact declaring class in the
current module MRO. The existing `_SourceCache`/`_FrameBinding` pure identity
helpers compile and compare the loaded source, bind the original function,
code, globals, defaults and super-chain owners, and verify actual producer-time
frames with the same `self`. MRO changes, parent-forward replacement, wrong
class cells, foreign closures, loaded-code changes or wrong-frame owners refuse.
The generic TensorCallableGuard, Native finite guard and original operation
remain unchanged. No discovery observer, profiler or full census is created;
the helper's source identity is added to the acceptance driver's pins.

Seven CPU regressions cover the supported super-chain and these refusal cases,
including all four actual public-Host requests with the explicit CPU Native
and scheduler doubles. A new root-scheduled attempt is still required to
establish this deployment's normal/F/OFF/health behavior.
