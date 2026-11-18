# Public Worker output39 original-evidence review

`review39_worker_core_result.py` is a read-only, standard-library reviewer for
the new six-request public `39.output_proxy` C1 protocol and its seventh ordinary
post-unregister request. It does not load the injector, a DSO, Torch, vLLM or a
device. It reuses the SHA-pinned source51 projection/consumer geometry checks
and Core41 JSON/artifact utilities; it does not reuse their qualification.

Keep these three Python files together in a new review directory:

- `review39_worker_core_result.py`
- `review39_output_source51.py`
- `review41_worker_core_result.py`

Copy matching files from the same reviewer commit. The default profile is
`source65-aten-v1`; its accepted operators and historical scope are unchanged.
The source51 helper now takes optional operator sets only when the separate
public custom profile explicitly supplies them. Its MODEL_LOCAL entry point
still uses the original aten-only sets, source pins and coverage requirements.

Run only on the original evidence host, using the new frozen source65 files and
the reviewed Native library. The manifest and commit arguments refer to the
source that produced the evidence, not to the later reviewer checkout.

```sh
python3 -I -B /review/review39_worker_core_result.py \
  --result /original/case/probe/result.json \
  --result-sha256 ORIGINAL_RESULT_SHA \
  --source-root /home/17-fault-c2c3-20261117/source65 \
  --source-manifest-sha256 ORIGINAL_SOURCE65_MANIFEST_SHA \
  --source-commit 37bbeb4f9612df550a37dafef06727d005111568 \
  --native-library /original/native/libdcat_postwrite.so \
  --serving-receipt /original/case/output39-runtime/RUNTIME_OWNER/serving_receipt.json \
  --serving-sha256 ORIGINAL_SERVING_SHA \
  > /new-review-directory/review.json
```

The serving file is not referenced by the probe's artifact manifest. Its exact
path and hash must therefore be supplied separately. It is optional only so a
run refused before any session file exists can still be diagnosed; missing
serving evidence cannot pass completed-request or C1 checks. All Host and raw
RPC files are resolved through the result's recorded paths and hashes. The
reviewer does not search for substitute files or rewrite originals.

Exit 0 means the submitted evidence and cleanup reconcile. It can represent a
correctly recorded refusal; it does **not** necessarily mean C1. Read
`source_bound_c1_receipts_reconciled` and `evidence.classification` separately:

- `SOURCE_BOUND_SINGLE_C1_RECEIPTS`: the complete six-request protocol, original
  RPC order, current selected producer/value-consumer, lossless/F/OFF readbacks,
  zero owned resources, unregister/detach and seventh same-Worker request join.
- `NO_ELIGIBLE_OBSERVED_PAIR`: the actual retained census has no eligible pair,
  selection did not occur and the original refusal is retained. A failed second
  request remains a partial census; it is not counted as a completed request.
- `PRE_C1_REFUSAL` or `REFUSED_OR_INCOMPLETE`: a prerequisite/request failed or
  the protocol is incomplete. Successful cleanup does not erase that failure.
- Exit 2 / `EVIDENCE_GAP_OR_CLEANUP_UNCONFIRMED`: a missing or contradictory
  artifact, identity, receipt, recovery or owner-release condition remains.

There is no fixed 140-site or model-name denominator. The reviewer reconstructs
all observed projection ordinals, operator counts and module/alias identities.
It verifies the first eligible pair in actual execution order and its current
original output storage and consuming operand view. Unsupported opaque GEMM
operators remain visible in the counts and do not become MM candidates.
Unselected cumulative transport refusals are compared with their observation
records, rather than mistaken for a failed selected intervention.

Actual enqueue IDs are joined to each advance, original query receipt and
completed Worker row. The separate serving receipt supplies the scheduler,
complete prompt/position metadata and full-vocabulary return contract. Input
digests are independently recomputed from the recorded complete token and
position lists and their tensor headers. The device equality of those values
remains the original root-hook readback receipt. Public Host owner, Runtime
owner, Native owner derivation and per-request Native generations are checked
separately. Native-loaded identity is not called callback registration.

The original complete hidden+logits equality check runs in the frozen Worker.
The evidence saves complete logits hashes but neither full hidden hashes nor
tensor bins. Consequently hidden lossless/OFF equality, Native exact readback,
synchronization and strong-reference duration remain source-bound gates and
receipts, not new device-byte observations. The seventh ordinary request proves
tokens and text, not a post-hook logits capture. The direct Native ABI exposes
no process-global owner table; only owned resources are reconciled. The build
fingerprint and external weights/model-source manifests are joined receipts;
no absent build receipt or external model bytes are claimed to be rehashed.

The review never grants C2, C3, E5, MODEL_LOCAL inheritance, `39.matmul_range`,
another model/device or general integration acceptance. It does not read the
probe status, `c1_evidence_complete` or hit totals as a substitute for raw joins.

CPU tests execute the real Host/Runtime and Torch hooks with explicit Native and
scheduler doubles. They cover a complete protocol, no eligible pair, current
input rejection, failed F plus real recovery, lost arm reply, failed restore,
foreign request/owner, altered RPC/Native receipts, cloned storage and retained hooks.
They establish no new NPU result. Independent review also reproduced and closed
the clean/unregister reservation and missing-deployment resolver defects in
`9c8d7ee6` (root integration `37bbeb4f`) before this source65 review.

## Explicit reviewed custom GEMM profile

For a newly frozen Worker run containing the reviewed custom adapter, current
dispatcher guard and per-lease receipts, also copy `review39_custom_gemm.py`.
Add `--projection-profile reviewed-custom-gemm-v1` to the command above and
replace every result, source-root, manifest, commit and serving hash with the
actual new run's pinned inputs. Source67 was scheduled for Core26 only; neither
its name nor an anticipated source68 label establishes a public39 result.

This profile requires the exact reviewed
`vllm::unquantized_gemm(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor`
schema, three pinned Python sources and the original module's weight object
with no bias. It checks the complete current dispatcher dump, one active
PrivateUse1 registration with no inactive/shadowed entry, the definition's same
origin, and the inspected `vllm_lib` construction at line 789 of the pinned
`vllm.utils.torch_utils` file. A stale Library ownership set or a boolean
`verified` field alone is insufficient. The guard is scoped to Torch
`2.9.0+cpu`; the separate Torch 2.9 no-device override experiment establishes
dispatcher behavior, not the actual Worker's current registration.

Both `postwrite_npu_consumers.py` and `gemm_lineage.py` must appear in the frozen
source manifest and the original Worker's loaded-source hash receipt. A file
present on disk is not accepted as proof that the Worker loaded it. No helper
imports either module while reviewing.

For every eligible actual inventory row, the reviewer joins the original
producer geometry and consumer metadata. The selected custom pair must reach
an actual reviewed `npu_rms_norm` or `npu_swiglu` input view containing selected
element zero. Exact no-write/no-alias schemas, the pinned torch_npu build/meta
source/current registration, actual BF16 inputs/outputs, RMS FP32 rstd, gamma,
scalars, ND formats and disjoint storages are checked. Meta functions and the
source65 schema export supply no actual shapes; geometry is derived only from
each new recorded operation's operands and returns.

Earlier refused calls retain their raw ordinal and remain in the complete
inventory. The selected key and every phase's full module/call denominator are
joined across two healthy observations, lossless, F and OFF; weight and library
object identities must remain the same. The sixth protocol request and the
ordinary post-unregister request remain separate health checks. No fixed
140-site MODEL_LOCAL count or qualification is imported.

Lossless and F also require the actual per-lease original geometry,
pointer/byte/device span, handle, owner and generation; successful open/apply,
restore/release returns; ordered original events; and the existing exact-ticket
post-recovery live readback. OFF opens no lease. These are saved ABI receipts,
not an offline memory read or fresh Native invocation. Historical reports that
lack this new audit field retain their historical evidence scope; absence does
not prove that a past Native operation failed.

The custom profile remains C1-only and source-bound. It never grants C2/C3,
E5, another custom operator, CANN 9.x, A3 or portable device qualification.
Its additional stdlib-only tests use explicitly constructed metadata doubles
for joins and negative cases, including override shadowing, wrong weight or
bias, hidden raw ordinals, disjoint consumer views, altered actual return
geometry, Native span/owner/release discrepancies and missing loaded sources.
Those metadata fixtures are not natural execution evidence.
