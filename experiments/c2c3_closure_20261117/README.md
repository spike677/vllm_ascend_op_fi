# C2/C3 closure — 2026-11-17

Frozen source: `91decf5efe48dfdf8c5494e52d258ccde788cc5d` (implementation
`cb5b0dc120ce6c80f1469bb8b7248da27a7bb258`). This is an execution workstream;
the previous engineering reports and all original 124 evidence remain intact.

The frozen starting point was **C2 0/17, C3 0/17**. The current strict result is
**C2 1/17, C3 1/17**, for the bounded Fault18 standard-Worker scope. A new source
and actual-build audit found unfreed SDK metadata arrays in the previously
accepted Fault20 and native Fault39 builds. Their numerical, coverage and OFF
receipts remain intact, but full-resource C2/C3 qualification is withdrawn
pending repaired builds and fresh acceptance. The separate HF Fault18 scope is
not counted twice. This is an in-progress checkpoint, not a claim that17 mechanisms or whole
model/version families are accepted. Existing scoped C1 receipts retain their
original source, build and workload. CPU contract tests do not grant qualification.
The [actual-build ownership audit](coverage_frozen_metadata_ownership_review.md)
binds this correction to the exact executed source and seven library hashes.

The [17-mechanism and27-node matrix](C2_C3_MATRIX.md) lists the accepted scopes,
natural-path gaps, queued experiments and version/hardware limits individually.

| Reviewed route | C2 | C3 | Independent statistics limitation |
| --- | --- | --- | --- |
|18.api_half, standard Qwen3-0.6B Worker, BF16 fresh prefill, TP1|113/113 targets|226/226 calls in two requests|Separate Worker capture pending; HF statistics cannot substitute|
|20.buffer_substitute, standard DeepSeek-V2-Lite-W8A8 Worker, BF16 fixed3-token fresh prefill, TP1|Requalification required;135/135 interventions observed|Requalification required;270/270 calls observed|First QMM output Max16.25→14.0;2 later NaN fields prevent whole-capture E5; host metadata arrays were not released|
|39.matmul_range, standard Qwen3-0.6B Worker, BF16 fresh prefill, TP1|Requalification required;113/113 interventions observed|Requalification required;226/226 calls observed|C2 first linear output5.1875→Inf with later NaNs; C3 raw capture missing; host metadata arrays were not released|
|18.api_half, natural HF model, separately reviewed|113/113 targets|226/226 calls|First consumer linear input1.734375→0.8671875; nulls retained|

The independent review is [AGENT7_ACTUAL_REVIEW.md](AGENT7_ACTUAL_REVIEW.md),
with exact scope, actual-file hashes and the different return-hook observation
semantics. All rows above are actual8.5.1/Ascend910B4(A2); none qualifies CANN9.x,
A3, concurrent requests, multiple Workers or another model automatically.
See [RESUME.md](RESUME.md) for the current running attempts; unreviewed runtime
successes do not silently change this table.

Root additionally reviewed [Native39 a6](AGENT7_NATIVE39_A6_REVIEW.md):342 C2 and
568 C3 references match original bytes. Each of113 targets changes in the once
request and in both every-match requests; all original allocations remain intact,
all target/group OFF and complete healthy logits restore.112 later input pairs
already contain propagated nonfinite values, so this is not113 isolated causal
tests. C3's missing raw triplet is an explicit E5 gap, not reconstructed data.
The older tracked-resource counters did not cover the SDK-allocated host arrays;
their zero values cannot establish complete resource release. This new finding
supersedes the earlier qualification decision without modifying those receipts.

Fresh standalone Fault28 B/F has [eight verified actual launches](attention28_cpu_fp64_launch_review.md).
This does not grant model or Worker C2/C3. The disclosed CPU-FP64 collector is a
new capture backend; older null statistics and historical results remain intact.
Its [independent raw-statistics review](AGENT7_NATIVE20_ATTENTION28_REVIEW.md)
finds no above-threshold legal-pair change; replay first differs at FIA output
Mean, with separate Norm118.30810159723288→116.27198217667856 (1.7210312676%).
Both pairs retain six nulls. These are standalone step0 results, not model steps.

Fresh FA now has a separate [CPU build review](attention28_fa_cpu_build_summary.json):
host, new matching kernel and registration completed; all851 vendor artifacts
match independently recomputed hashes. Subsequently, an [independent baseline
launch review](AGENT7_FA_A2_PARTIAL_REVIEW.md) verified its registered ELF,
actual argument slot, complete raw POD and all205 decoded fields on124. The run
then stopped at a Python JSON-envelope comparison bug, now corrected; its failed
receipt remains intact. F, repeat and OFF were not reached. Later attempts met
card5 initialization Resource_Busy and a card2 gate with another TP2 Worker.
A later [a5 independent review](AGENT7_FA_A5_AND_GATE_REVIEW.md) verified all four
actual B/repeat/F/OFF launches and24 complete output files: repeat/OFF are byte
exact, F differs. It remains standalone joint M/nRatio switching, with no stock
bridge or C2/C3. Fresh raw B/F statistics are collected separately. A new read-only FIA prefill reader records the actual Worker boundary and
explicitly rejects TND/page16 for the existing9.1 BNSD strategies.

Natural MoE group implementation now preserves original input objects and storage
on healthy, unselected and once-exhausted calls; only an admitted fault constructs
paired remapped indices/probabilities. This corrects a source-review finding and
still requires a fresh NPU group run. The new vision adapter prepares full
pretrained Qwen3-VL image prefill as MODEL_LOCAL; it does not inherit standard
Worker qualification. Native26 has a separate v2 recovery ABI and host-memory
failure contracts. Its SDK build passed and its conditional complete-set executor
now has a prepared same-Worker C2-to-C3 queue: any unresolved target, overlap or
budget overflow rejects the whole set. Actual natural group proof is still pending.
Changes to shared code never upgrade older receipts to the current HEAD: all
accepted scopes remain bound to the exact reviewed source/build/workload hashes.

The [live C2 proof audit](coverage_proof_authority_review.md) fixes a separate C3
admission defect: copying a frozen proof previously retained its private issuer
marker. Admission now requires the original issued object, unchanged fields and
the issuing process. Copies, reconstruction and fork-inherited proofs refuse.
Independent review of the exact frozen18/20/39 drivers found direct original-object
handoffs; their earlier results did not depend on the copy defect. That source
audit does not assert that the new registry ran in those historical processes.

The [independent build/scope audit](AGENT7_CPU_BUILDS_A29_REVIEW.md) confirms the
actual FA and Native26 artifacts, SDK/image bytes and CPU ABI contracts. Native26
requires the matched image C/C++ runtime; the default host lacks GLIBCXX_3.4.29.
Source-a29's main unittest run had729 cases, with2 skipped and no failures;
the separate Worker suite had194 passing cases. These sets overlap. Later focused
checks cover the new read-only RoPE and FA launch readers, not additional NPU runs.
After the Native26 complete-group integration, main discovery ran772 cases with
2 skips and no failures; separate Worker discovery passed210 cases. The subsequent
FA-envelope fix passed its50-case suite with1 skip. These overlapping CPU suites
verify contracts and regressions, not NPU qualification.

The subsequent quant row observer follows only actual reviewed row-preserving
operations and verified Native-DQ/owned-QMM chains. Unknown row-bearing operands,
Attention/MoE outputs and alias writes invalidate or withhold row evidence. Seed,
query and report mappings are detached snapshots with bound digests; mutating a
returned report cannot relabel a live tensor. This is read-only inventory, not
all135 natural projections or quant C2/C3. See [the exact limits](providers_quant_handoff.md).
The separate [quant group core](providers_quant_group_core.md) now has19 passing
host C++ contracts. It distinguishes one public model invocation from the
correct-domain helper and fault Native children, freezes healthy references,
and accounts for workspace, partial publication and full-group drain. Actual
SDK shim exports are now implemented in a separate library, with9 additional
host mock-runtime contracts and7 builder contracts. The actual8.5.1 SDK build-a2
passed after treating vendor includes as system headers; own-code warnings stay
strict. Further review found failed-executor ownership, exception-stack liveness
and getter-array cleanup gaps. They are being fixed before any Native use;
Tensor lifetime ownership and complete natural serving integration remain
unqualified. The optional sanitizer build failed because host ASan/UBSan
libraries are absent; no sanitizer result or Native qualification is claimed.
After this increment and the device-allocation gate, main discovery ran810 tests
with2 skips and no failures; separate Worker discovery passed228 and queue
contracts passed6. These suites overlap and are not added together.

The public Qwen3-VL-2B-Instruct files are now prepared on124 at pinned revision
`89644892e4d85e24eaac8bacfd4f463576704203`. All10 remote files match the official
metadata and the independent local verification:4,266,640,306 bytes total,
including the4,255,140,312-byte weight with SHA256
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
The transfer completed before verification and no-clobber publication; the
verified upload and final filename share one inode. No weights enter Git.

Vision CPU preflight-a3 passed the actual unchanged processor, producing
`input_ids[1,901]` and `pixel_values[3536,1536]` from the pinned existing image.
The container had no accelerator device nodes or network, and constructed no
model or loaded weights. Read-only driver libraries resolved the earlier import
failure. Imports still attempted runtime device discovery and logged failures;
the old receipt's `actual_npu_calls:0` does not mean zero runtime API calls.
Future probe output explicitly distinguishes compute invocations from possible
runtime discovery. Original a1/a2 failures and a3 bytes remain unchanged.
Natural full-model prefill and census subsequently completed with exact repeated
baseline. Its first group attempt saved positive C2/C3 phase records, but an
external Docker stop/kill interrupted statistics after85 seconds (exit137,
no OOM, no controller timeout). That interrupted whole run is not accepted.
A fresh complete coverage/recovery run is in progress on card2, with statistics
explicitly separated; no model-local result is called standard-Worker evidence.
The [independent preparation audit](AGENT7_VISION_PREPARATION_REVIEW.md)
re-read all10 final files and verified the executed source, image and processor
identities. Its [hash summary](vision_preparation_summary.json) contains no
model or tensor payloads;8 CPU negative/regression checks passed after integration.

The earlier [resource-owner lookup](resource_window_summary.json) remains an
unchanged pre-control snapshot. Under the user's prior authorization to stop
card2/5 tasks, root verified the exact single-node, one-replica AscendJob scope,
then applied a UID/resourceVersion-guarded1-to-0 replica change and normal
termination of its one remaining Pod. Private job backups stay on124. No host
Docker entry, daemon, checkpoint or cluster-wide policy was changed. The Job
subsequently returned404; root did not request its deletion and does not infer
the deletion actor or recreate the old object.

The retained checkpoint still names that terminated Pod. The
[live-owner review](coverage_stale_checkpoint_review.md) joined actual kubelet
socket identity, node-authenticated API, current PodResources and OCI records,
with a final full device gate. It found no live owner while retaining the
checkpoint evidence. It does not certify internal allocation clearance, driver
availability or exclusive reservation. A new card5 FA attempt then actually
completed four calls: B, repeat B, F and unscoped OFF. Repeat and OFF complete
outputs are exact; F differs. Independent launch/payload review passed,
and this synthetic standalone sequence gives no model/Worker C2/C3 certificate.
The first review's general default was replaced by an explicit one-attempt
UID/checkpoint scope. Its next attempted queue correctly refused a different
owner. Root separately verified and stopped that new single-node training Job
under the same prior user authorization, using guarded replica change and
normal Pod termination. No checkpoint edit or broad exemption was introduced.
The unchanged FA B/F statistics then completed, followed by real Qwen3-VL image
prefill census with exact repeated baseline. The earlier external Docker cleanup
source remains unidentified.
Raw-only fixed-tolerance comparison of the fresh FA statistics found4 exact
changes and0 exceedances, with10 null fields retained and no first exceedance.
This does not establish observable fault localization at the required tolerance.

Fault20 additionally has a [root review](native20_group_summary.json):408 C2 and
678 C3 evidence references match actual stored bytes; all targets have native
readback, unchanged original allocations, exact lossless/OFF and complete logits
recovery.133 later calls per fault request already receive propagated input
changes. These are135 verified interventions, not135 isolated causal tests.

C2 requires a complete independent observer denominator, lifetime-bound target
identities, all eligible targets hit once, no non-target interventions, per-target
effects, group drain/resource release and same-process healthy OFF recovery.
C3 requires that mechanism's matching C2 proof before a bounded every-match
session. New calls caused by propagation are recorded separately.

The root controller owns integration and all NPU scheduling. Seven independent
specialty worktrees use the same frozen base; at most three specialty agents run
concurrently due to the execution environment's limit. See `ownership.json`.

New raw statistics, private tensors and Native build products stay on 124 under
`/home/17-fault-c2c3-20261117`. Only reviewed source and summary/hash reports may
be published. E5 reads raw B/F statistics/mix, stack/construct, configuration and
collector identity only, with baseline reference, `rtol=1e-5`, `atol=1e-8`.
Null, NaN and alignment failures remain explicit; runtime hit logs cannot select
the first numerical difference.

`run_queue_124.py` accepts up to eight reviewed cases, obtains the shared
`/var/tmp/fault22_npu_2_5.lock`, checks device health, processes, open device handles
and dedicated reservations before each case, and cleans up only its own container.
The [Kubernetes allocation review](coverage_device_allocation_gate_review.md)
adds actual containerd state roots and kubelet-owned device-plugin checkpoint
discovery. On124, the checkpoint retains card2/5 allocations even when no compute
process is visible. Those assignments block execution; generic container device
permissions remain inventory only. Unknown active allocation sources also refuse.
The lock coordinates cooperating jobs; it is not host isolation. No expired
Docker entry-point isolation is reinstated. A finished container is not proof of
same-process recovery; that remains a separate mechanism receipt.
