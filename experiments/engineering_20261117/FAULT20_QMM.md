# Fault20 natural intermediate QMM serving qualification

This addition implements an intermediate projection boundary for
`20.buffer_substitute`. It does not reuse the final floating LogitsProcessor
head as evidence of QMM. The current status is **CPU verified; actual QMM
injection and NPU serving acceptance pending**.

## Selection and invocation ownership

The adapter inventories actual INT8 dynamic W8A8 projection objects through
their quantization-method chains and TP metadata. In a real fresh prefill it
observes the input embedding that consumes the root's actual token storage,
the first RMSNorm that consumes that embedding, and the first actual dynamic
projection consuming that normalizer output. That execution-order policy is
fixed before discovery. There are no user-supplied layer names.

The selected projection and native facts must repeat in two complete healthy
requests. `plan_adaptation` then requires a unique stable target **within that
selected projection scope**. Other projection candidates are not declared
covered or natively qualified. This is C1 selection, not C2/C3 coverage.

The quantization method still calls its original implementation once. Owned
pass-through observers verify that real `npu_dynamic_quant` outputs are the
same storage objects passed as QMM INT8 input and per-token scale. The QMM
weight and weight scale belong to the actual projection. Every present native
formal is matched against current typed GWS/Run metadata, including address,
slot, shape, stride, offset, storage size, element size, and actual format.
The floating projection input is recorded separately and is never called the
native INT8 input. Unknown signatures, absent lineage and extra formals refuse.

The inspected vLLM-Ascend `w8a8_dynamic.py` implementation performs real
dynamic quantization, passes `layer.weight`, `layer.weight_scale`, per-token
scale and bias to `npu_quant_matmul`, and requests the input's floating dtype.
Its weight-loading path transposes the logical matrix to K×N before NZ
conversion. This adapter supports the observed two-dimensional prefill path;
the optional three-dimensional squeeze/unsqueeze path is not qualified.

## Transaction and acceptance

Eight ordinary host-enqueued requests execute discovery 1, discovery 2,
reference, lossless substitution, qualification OFF, fault, observed OFF,
and disabled OFF. Actual scheduler IDs and complete token/position bytes bind
each request. The native callback is resumed only around the selected QMM
and paused as that call returns. The real root then finishes, and the actual
LogitsProcessor supplies complete global logits for the end gate. No
Worker/runner execution method or logits output is replaced.

Native source/ABI/build/profile, exact reference input hashes, lossless output
equality, and both OFF checks remain the existing gates. Private reports retain
actual projection ownership, complete-logits hashes, native receipts and
selection evidence. CPU fixtures do not establish any NPU hit or acceptance.

`statistics/mix` uses the unchanged historical collector at real request
steps 2 (reference) and 5 (comparison). A separate baseline-control process
preserves all phases but uses reference mode at step 5. Public evidence remains
B/F only. Runtime receipts cannot choose an independent statistical first
difference; that comparison must use raw collector/config/stack/module data.

## Implementation and execution

- `fault_injection/serving_qualification.py`: distinct
  `ProjectionQualifiedNativeSession`, sharing the existing native gates.
- `src/demoncat/runtime/vllm_ascend/worker_extension/serving20_qmm.py`:
  root/embedding/norm/quantization/QMM ownership and real logits completion.
- `providers/fault20.py`: exact reviewed session/backend pair dispatch.
- `experiments/worker_positive_20261117/serving_probe.py`: QMM selects the new
  intermediate projection deployment automatically by the actual operator.
- `prepare_qmm20_cases.py`: emits one bounded standard-Worker trial for the
  existing resource-lock/idle-gate controller. It neither starts devices nor
  rebuilds native libraries. Use a fresh installed wheel and reviewed native
  bundle, including `/native/serving_review.json`, and the current probe.

CPU validation: 31 existing/new Fault20 tests pass, including eight actual CPU
module-forward phases, complete logits recovery, exact provider dispatch,
native scale-address rejection, cloned quantization-output rejection, absent
natural QMM rejection, and unknown public-signature rejection. The CPU native
runtime is explicitly mocked. DeepSeek-V2-Lite-W8A8 on 124 remains the next
actual target; other models require the same observed lineage and native gates.

## First actual deployment attempt

The first 124 DeepSeek attempt completed its initial healthy generation but
refused before installing the Fault20 observer, at
`LOADED_WORKER_ENVIRONMENT_UNPROVEN`. It did not enter `discovery_1`, execute
lossless substitution, or inject a fault. Both native operators reported
matching builds and no live callback/lease.

The saved actual environment had `maps_status=OBSERVED`. All seven proxy and
real implementation files under `/native` were verified. Eight image libraries
failed the strict mapped-device/inode versus opened-file identity check:
`libacl_rt.so`, `libascendcl.so`, `libnnopbase.so`, `libopapi_cv.so`,
`libopapi_oam.so`, `libopapi_transformer.so`, `libtorch_npu.so`, and
`libcust_opapi.so`. This is an environment-identity blocker, not QMM acceptance.

The second deployment can use the existing CPU-only
`snapshot_mapped_libraries_124.py` to extract identical bytes from the fixed
acceptance image, check the build-profile hashes where available, and supply
its manifest to `prepare_qmm20_cases.py --mapped-libraries`. The generator
rechecks each file's size and SHA, pins the Docker image digest, and mounts
the same bytes read-only at the observed paths. Runtime mapping, inode,
regular-file, ELF, SHA and stability checks remain mandatory and unchanged.
A snapshot is not an environment certificate. All loaded identities must pass
on the actual next Worker before native discovery is allowed.

Environment refusals now preserve a private `environment_refusal.json` and
summarize missing or unverified libraries in qualification status. This adds
diagnostics without excluding extra loaded libraries or weakening any gate.
