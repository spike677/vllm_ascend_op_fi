# C2/C3 closure — 2026-11-17

Frozen source: `91decf5efe48dfdf8c5494e52d258ccde788cc5d` (implementation
`cb5b0dc120ce6c80f1469bb8b7248da27a7bb258`). This is an execution workstream;
the previous engineering reports and all original 124 evidence remain intact.

The frozen starting point was **C2 0/17, C3 0/17**. Root has now reviewed
**C2 3/17, C3 3/17** in bounded standard-Worker scopes: Fault18, Fault20 and native
Fault39 have C2 and C3. The separate HF Fault18 scope is not counted
twice. This is an in-progress checkpoint, not a claim that17 mechanisms or whole
model/version families are accepted. Existing scoped C1 receipts retain their
original source, build and workload. CPU contract tests do not grant qualification.

| Reviewed route | C2 | C3 | Independent statistics limitation |
| --- | --- | --- | --- |
|18.api_half, standard Qwen3-0.6B Worker, BF16 fresh prefill, TP1|113/113 targets|226/226 calls in two requests|Separate Worker capture pending; HF statistics cannot substitute|
|20.buffer_substitute, standard DeepSeek-V2-Lite-W8A8 Worker, BF16 fixed3-token fresh prefill, TP1|135/135 targets|270/270 calls in two requests|First QMM output Max16.25→14.0;2 later NaN fields prevent whole-capture E5|
|39.matmul_range, standard Qwen3-0.6B Worker, BF16 fresh prefill, TP1|113/113 targets|226/226 calls after fresh live C2|C2 first linear output5.1875→Inf with later NaNs; C3 raw capture missing|
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

Fresh standalone Fault28 B/F has [eight verified actual launches](attention28_cpu_fp64_launch_review.md).
This does not grant model or Worker C2/C3. The disclosed CPU-FP64 collector is a
new capture backend; older null statistics and historical results remain intact.

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
The lock coordinates cooperating jobs; it is not host isolation. No expired
Docker entry-point isolation is reinstated. A finished container is not proof of
same-process recovery; that remains a separate mechanism receipt.
