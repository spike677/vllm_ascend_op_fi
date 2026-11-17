# Fault20 natural intermediate QMM serving qualification

This addition implements an intermediate projection boundary for
`20.buffer_substitute`. It does not reuse the final floating LogitsProcessor
head as evidence of QMM. The third 124 attempt **passed actual DeepSeek C1
Worker execution, one native substitution, complete-logits change and OFF**.
The independent statistics comparison finds the first observable candidate at
QMM output, but baseline NaN prevents whole-dump E5 acceptance. The new scope
does not qualify every projection or a model family.

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
native INT8 input. The actual empty optional formals are verified separately
from nonempty Run slots. Unknown signatures, absent lineage and extra formals refuse.

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

CPU validation: 33 existing/new Fault20 tests pass, including eight actual CPU
module-forward phases, complete logits recovery, exact provider dispatch,
native scale-address rejection, cloned quantization-output rejection, absent
natural QMM rejection, and unknown public-signature rejection. The CPU native
runtime is explicitly mocked. Actual DeepSeek-V2-Lite-W8A8 acceptance below is
separate; other models require the same observed lineage and native gates.

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

The second attempt passed environment and MLA prefill checks and observed the
natural `model.layers.0.self_attn.kv_a_proj_with_mqa` QMM. It refused before
the original native execution in discovery: the Python verifier incorrectly
equated a present formal tensor with a nonempty Run storage slot. The saved
typed observation showed empty `[0]` `y_scale`, `x1_offset` and `y_offset`
formals, all with zero bytes/addresses and Run slot `-1`. Actual nonempty slots
were x1=0, x2=1, x1_scale=2, x2_scale=3, out=7.

The verifier now matches all ten formal names/order, admits only these three
observed empty placeholders with their complete zero-storage, geometry,
format and output-dtype contract, and still rejects unknown or nonempty
unbound formals. Actual public INT8/weight/scale storage matching is unchanged.
No native core was modified. Failed formal checks now retain both raw typed
observations and the public transaction before refusal. The second attempt
is not an injection or OFF pass; the third used this precise Python fix with
the same previously built native core.

## Third actual attempt: limited Worker acceptance

The run is `/home/17-fault-engineering-20261117/qmm20-a3/fault20-qmm-fault`,
using `source-a8` and the independently installed `packages-a8`. The actual
environment is Ascend910B4/A2, CANN8.5.1, TP1, eager, BF16, and pretrained
DeepSeek-V2-Lite-W8A8. Each of the eight host-enqueued requests performs the
same fresh 14-token prefill. These are qualification request phases, not eight
decode tokens or a continuous-generation acceptance test.

Automatic actual execution selected
`model.layers.0.self_attn.kv_a_proj_with_mqa`: INT8 x1 `[14,2048]`, INT8 NZ x2
`[2048,576]`, FP32 per-token scale `[14]`, BF16 weight scale `[576]`, BF16 output
`[14,576]`. The native nonempty Run slots are 0/1/2/3/7. The two healthy actual
requests selected the same object and native identity. There are 135 structural
projection candidates; that number is not an all-eligible execution denominator.
The generic canonical-role field remains `UNRESOLVED`; the separately observed
embedding→normalizer→dynamic quantization→projection lineage establishes this
limited selection, not complete logical-role adaptation.

The fault maps the selected INT8 input to an INT4-equivalent grid:
`16 * clamp(round_nearest_ties_away_from_zero(x1 / 16), -8, 7)`. It writes a
separate temporary device buffer, verifies its contents by device readback,
and substitutes only the native x1 Run address. The INT8 ABI, original Python
input, x2 weight and both scales remain unchanged. The original native QMM is
executed once; there is no output rewrite. Lossless control instead copies the
original x1 bytes to independent storage. Thus msprobe can observe unchanged
public QMM inputs and the first numerical change at its output.

| Check | Observed result |
| --- | --- |
| Eight phases | discovery 1/2, reference, lossless, qualification OFF, fault, observed OFF, disabled OFF all completed |
| Selected native boundaries | One per active phase, seven total; disabled OFF has no callback observation and runs the normal model |
| Fault | One x1 substitution, one changed slot, one original native call, temporary readback and release confirmed |
| Protected input bytes | Four complete native input hashes identical before/after every audited execution and across phases |
| Complete QMM output | Fault SHA changes; lossless and observed OFF exactly match reference |
| Complete global logits | `[1,102400]`, finite in all eight phases; fault SHA changes, both OFF phases exactly restore reference |
| Generated result | All eight Top-1 IDs remain 185; healthy checks before/after both return newline |
| Cleanup | Registered/enabled/native-owned false; inflight/leases zero; detached and deployment removed |

Baseline global-logits SHA256 is
`7f0b59420ae5f51c1ffc6c47cdb18bfc6fd91fa46ad1660c75ce806606d6f262`;
fault is `a1770b841f4ff5b38657f6b87a1fa1374c05d5c3e7dc9c746558e39e5b186ac4`.
Hashes prove exact change/recovery; no magnitude or task-quality claim is derived
from them. No TP2, decode, other shape, other CANN version or A3 acceptance is
inherited. The adapter does not create Workers or replace execution methods.

## Independent statistics and retained failure

The unfiltered raw `statistics/mix` groups are actual `step2/rank0` (reference)
and `step5/rank0` (fault), with the same configuration and collector. A separate
CPU `analyse_pair.py` read only their raw triplets/configuration/collector metadata,
using baseline-reference `rtol=1e-5`, `atol=1e-8`. It did not read native events,
the selected path, full tensors or this acceptance summary.

Its first observable candidate is record index 83:

```text
NPU.npu_quant_matmul.0.forward/output/0
Max: 25.75 -> 15.6875
absolute difference: 10.0625
relative to baseline: 39.0776699%
```

There are 172 unavailable statistics fields before that candidate. Across each
complete dump, 10,306 fields are null and 6 are nonfinite. B self-comparison
refuses at index 599, `Tensor.__setitem__.8.forward/input_args/0`, Max `NaN`
versus `NaN`. Therefore `baseline_self_comparable=false`,
`whole_dump_comparable=false`, and **E5_NOT_PASSED** remain the final strict
statistics outcome. This is a valid observable-prefix location consistent with
input-buffer substitution, not a clean whole-model first-difference acceptance.
The comparator also retains `UNPROVEN_REQUEST_MAPPING`; the independent statistics
do not borrow scheduler mapping from the separately checked runtime receipts.

The read-only audit matched four actual imported module hashes against their
installed package files and `source-a8` source bytes, and checked all 28 referenced
native proof hashes. It performed no NPU calls. Raw statistics, complete tensors,
native binaries and original failed attempts stay on 124.
[`fault20_qmm_worker_summary.json`](fault20_qmm_worker_summary.json) contains the
scope, phase results, cleanup, independent candidate and refusal, version/build
identity, and SHA256 references. This adds a QMM scope to mechanism 20; it does
not increase the count of distinct public Worker mechanisms.
