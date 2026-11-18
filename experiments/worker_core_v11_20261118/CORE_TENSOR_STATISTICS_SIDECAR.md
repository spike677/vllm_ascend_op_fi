# Experimental public Worker18 statistics sidecar

This supplies an executable, CPU-tested collection entry for the **current
public `ActualTensorRequest`**, without modifying Core, providers, registry,
exporter or the injection mechanism. It has not run on NPU and grants no E5 or
new C2/C3 qualification. Historical source-a37 originals remain separate.

## Actual entry and capture boundary

The existing `natural_tensor_worker_core_probe.drive` still drives the complete
ten-request public protocol through `WorkerFaultClient`: native baseline,
baseline, baseline repeat, lossless, C2 fault/OFF, two C3 faults/OFF, and the
ordinary cleanup-health request. Its failure path still owns recovery OFF,
health, unregister and detach. No synthetic forward replaces an enqueue.

The new experiment entry installs the sidecar after successful public enable,
joining the exact live Worker and Runtime ticket. A same-thread profile
observer watches only the real original `_begin_request` return on that exact
`ActualTensorRequest` object. A queued RPC alone starts no collection. The
original function, compiled code and frozen source files are checked; no
Runtime callable or provider factory is patched. The experiment launcher
composes only the original **probe's** drive entry and restores it on exit.

One fresh, disposable TP1 Worker supplies three windows:

| Original directory | Actual public phase | Meaning |
| --- | --- | --- |
| `statistics/observation/step0/rank0` | `baseline` | B |
| `statistics/observation/step1/rank0` | `baseline_repeat` | independent B repeat |
| `statistics/observation/step2/rank0` | `c2_fault` | F |

These are three real prefill requests, **not decode token steps**. F still uses
the current public group's admission, complete-target accounting and effects;
the sidecar does not set `_statistics_arm`, change a target, consume an old
proof, or issue any qualification.

Capture begins after real `_begin_request` has installed the owned
adapter/observer, before model computation. The unchanged `collect()` closes
it at the real logits-processor return. It therefore covers native API and
submodule input/output records inside that model/logits window; it does **not**
claim preceding scheduler preparation or the following sampler.

The inspected, byte-pinned `PytorchService._register_module_hook` uses default
`recursive=True`; `ModuleProcessor.register_module_hook` consequently skips
the outer root module. Its missing root-module record is disclosed. **No
root-return compatibility shim is shipped.** Every existing root hook/flag and
its order must remain identical; any new root hook refuses. All other native
module/API records are retained without an operator filter or synthetic
first-difference record. The collector's normal output-hook order is retained.

## Collector ownership and recovery

The source96 NPU attempt failed before the first capture window: automatic
custom-op registration replaced an `OpOverloadPacket` by a generated function
without its `default` metadata. TorchDispatchMode then failed at the actual
`_C_ascend.npu_add_rms_norm_bias` call. The failed original and owned container
cleanup are retained; CPU protocol tests had not exercised a custom packet.

The compatibility helper now preserves the **original OpOverload** as the
`default` attribute on the exact collector-generated wrapper, matching the
same collector's normal `API_ATTR_LIST` handling. It verifies registry source,
generated code, closure-bound packet/name/namespace and current identity before
adding missing metadata; foreign attributes or later changes refuse. It does
not replace a callable, alter inputs/outputs, filter operators or rewrite raw
statistics. Collector identity records every affected packet and helper hash.
Direct `.default` calls are not newly instrumented by this metadata repair;
that pre-existing collector visibility limit is retained. The attributes,
like the stopped global wrappers, end with the disposable process.

The unedited package's public `start/stop/step` controls are reused. The
sidecar attaches only to the existing `session.statistics_capture` observer
slot, so owned diagnostic snapshots and return-proxy arithmetic pause capture
symmetrically. Real model APIs remain enabled. An original collector exception
is never converted to success.

Every window records full before/start/restore hook inventories, callable
code/source identities, and exact original-forward identities. Newly observed
hooks must be the source-pinned ModuleProcessor hooks through its actual
`ThreadSafe.synchronized` closure and the same live processor. Unknown source,
code, globals, decorator owner or hook rejects. Existing hooks cannot be
removed, replaced or reordered.

After stop, only verified new collector module hooks, flags, attributes and
the pinned forward wrappers are restored. The stopped service's
`hooked_modules` must contain exactly the same owned model; that one cache
entry is removed so the next window can register fresh hooks. No foreign
model cache or API counter is cleared. `step()` performs the native step
transition. This cache rearm is explicitly disclosed in collector identity.

On failure, an outstanding Core completion hook must first be resolved by
**Core's existing recovery**, not deleted by the sidecar. Failed or partial
windows remain failed; unknown/changed resources retain evidence rather than
being called cleaned. Profile ownership, session attachment, full hook
inventory and cleanup failure are included in the separate sidecar receipt.

msprobe's process-wide API wrappers are left in their stopped state until the
temporary Worker exits. `global_api_wrappers_restored=false` is explicit; this
does not claim full process-wide collector unloading. Run only in an isolated,
disposable Worker, not an existing service. Injection resource qualification is
not expanded by the sidecar's local cleanup result.

## Source-only preparation and execution

Root must freeze a new source manifest containing this implementation and use
its exact SHA. The actual collector is mounted read-only from the reviewed
`/home/batch-scale-dynamic-msprobe-20261023/python_packages`. The helper verifies
the historical package controls plus the exact synchronized wrapper,
registration utility and PyTorch service sources. Changed code refuses; there
is no version-name-only fallback.

From the source checkout, **prepare without launching**:

```sh
python -B -m experiments.worker_core_v11_20261118.prepare_core_tensor_statistics \
  --source /home/17-fault-c2c3-20261117/source-NEW \
  --source-manifest-sha256 FROZEN_MANIFEST_SHA256 \
  --card 5 --image REVIEWED_IMAGE_DIGEST --out reviewed-worker18-statistics.json
```

The case contains one temporary Worker and the existing ten-request protocol,
with read-only source/model/collector mounts. Root must apply its current
package, device gate, resource lock and queue controller. The preparer never
starts a container or changes shared device ownership. The source manifest
must be available at `/source/source_manifest.json`.

The case's executable command is:

```sh
python3 -u /source/experiments/worker_core_v11_20261118/core_tensor_statistics_probe.py \
  --source-manifest /source/source_manifest.json \
  --source-manifest-sha256 FROZEN_MANIFEST_SHA256 \
  --mechanism 18.api_half --model /model --out /evidence/probe --memory .35
```

After the attempt, run the **separate CPU raw-only review**:

```sh
python -B -m experiments.worker_core_v11_20261118.review_core_tensor_statistics \
  --statistics ATTEMPT/probe/raw_model_capture/statistics \
  --out ATTEMPT/independent-raw-review-new
```

The reader opens only raw triads, configuration and collector identity. It
does not read the sidecar receipt, Runtime logs, hits, tensors or an expected
first location. B self, B/repeat, B/F and repeat/F use the frozen comparator
SHA `d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`
and `abs(F-B) > 1e-8 + 1e-5*abs(B)`. Nulls, NaNs, structural mismatch, no
difference and earlier differences are preserved. Missing or extra step/rank
originals refuse; no actual step is relabeled. B repeat is not F-repeat proof.

## Source/export dependency closure

No shared exporter edit is included. Keep these new experiment files in the
frozen source archive (they are not additions to the production Runtime API):

```text
experiments/worker_core_v11_20261118/core_tensor_statistics_capture.py
experiments/worker_core_v11_20261118/collector_packet_compat.py
experiments/worker_core_v11_20261118/core_tensor_statistics_probe.py
experiments/worker_core_v11_20261118/prepare_core_tensor_statistics.py
experiments/worker_core_v11_20261118/review_core_tensor_statistics.py
```

They reuse the existing `natural_tensor_worker_core_probe.py`,
`prepare_tensor_worker_core.py`, `tensor_statistics_capture.py` and
`audit_statistics_series.py` under `experiments/c2c3_closure_20261117`, the
unchanged `fault_injection/statistics_audit.py`, and the already exported
public Worker18 Runtime/Host/provider/coverage closure. Their normal third
party dependencies are the existing Torch, vLLM and the pinned msprobe
package. No new native library or weight is built or exported.

The four new test modules are `test_core_tensor_statistics_capture`,
`test_core_tensor_statistics_probe`, `test_core_tensor_statistics_raw_review`
and `test_core_tensor_statistics_msprobe`. They reuse existing
`test_core_tensor_serving` and `test_tensor_worker_core_probe` fixtures, which
remain explicitly CPU simulations. With `src;fault39/src;.;tests;tests/worker_extension`
on `PYTHONPATH`:

```text
python -B -m unittest test_core_tensor_statistics_capture test_core_tensor_statistics_probe test_core_tensor_statistics_raw_review test_core_tensor_statistics_msprobe test_core_tensor_serving test_tensor_worker_core_probe test_statistics_audit test_tensor_statistics_window -q
```

Local result: **77 tests ran, 76 passed, one skipped**. The skip is the real
pinned-msprobe CPU integration because that package is not installed locally;
synthetic JSON/hook fixtures are not msprobe evidence. Root should run
`test_core_tensor_statistics_msprobe` with the actual reviewed package in its
device-free production CPU image before a device attempt. An installed but
incompatible package fails rather than silently skipping. No actual new B/F
statistics, device acceptance, first location or E5 success is claimed here.
