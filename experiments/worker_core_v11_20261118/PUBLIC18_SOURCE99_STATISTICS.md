# Public Worker18: actual statistics and retained control failure

The source99 standard Qwen3-0.6B Worker completed its original public protocol
on124/card5, TP1/BF16/CANN8.5.1/A2. Three actual prefill windows were collected:
step0 baseline, step1 repeated baseline and step2 C2 fault. They are not decode
steps. The [raw-only independent report](public18_source99_statistics_summary.json)
retains **E5=false** because the strict whole-window repeat control fails.

| Raw comparison | Result |
|---|---|
| Baseline self | 2,985 records /5,914 finite fields; zero exact or over-tolerance changes |
| Baseline/repeat | 60 over-tolerance changes; first at record50, cache slot_indices Max57 to73 |
| Baseline/fault | 4,403 over-tolerance fields; first at record13, `Functional.linear.0.forward/input_args/0`, Max1.734375 to0.8671875 |
| Repeat/fault | 4,399 over-tolerance fields; same first operator/field and values |
| First B/F magnitude | Absolute0.8671875; relative0.5 (50%); tolerance0.00001735375 |
| Unobservability | 26 null fields precede that first B/F difference;6,026 null fields per full capture; no NaN/Inf |

The fixed rule is `abs(F-B) > 1e-8 + 1e-5*abs(B)`. Original JSON digits are
preserved. Config, stack, construct and tensor structure align; no operator
filter, per-request synthesized tensor, runtime hit, expected location or
private array is an input to the independent comparator.

A complete supplementary enumeration finds56 repeat changes in28
`NPU._npu_reshape_and_cache.*` slot-index inputs and4 in the input statistics
of `Tensor.__setitem__.0/2`. All occur **after** the first B/F record13;
none occurs at or before it. The write-input statistics are not
assumed to be request-semantic values merely because their operator is a cache
write. These records are retained, not normalized, removed or called equal.
Consequently the whole-window control remains failed. The reproducible finding
is a **first observable changed linear input**, with an earlier unobservable
prefix. This neither uniquely identifies the RMS source nor establishes
whole-capture repeat stability. No additional identical NPU run is needed to
repeat this limitation.

The [separate runtime controls](public18_source99_runtime_controls.json) report
113 C2 and226 C3 hits, healthy token restoration, all three completed windows,
and released sidecar profile/hooks/session references. The full original RPC
chain has not been independently reaudited for this new scope; these counts
do not add a new mechanism qualification. The original source66 C2/C3 scope
remains intact. Global collector wrappers, with preserved original `default`
metadata, stay stopped until the disposable Worker exits. The native root
module and sampler are outside this capture. There is no F-repeat capture.

The independent raw report SHA256 is
`f1e0948ac793d1629cda7ab0c209049844f70fa5df1aa4852ec84a05f7fbbd34`.
It preserves each original triad/config/collector hash and the four full-review
hashes. Remote originals are under
`/home/17-fault-c2c3-20261117/v11-public18-statistics-source99-card5-a1/public-worker18-statistics-three-windows/probe`;
the separate review is `v11-public18-source99-raw-review-a1` in the same task
root. This publication contains summaries and implementation, not new raw
statistics, tensors, sidecar inventories or compiled libraries.

Source99 commit: `a07d360981c355ad7e5c45bdfcafe0433b676769`;
manifest SHA256 `91f65b63202e9e612396ee93c2fef58220c5f0f03e43170c82c65a152bcf3ff7`.
The actual pinned-msprobe production CPU preflight passed28 tests with no
skips, log SHA256 `cba031614dd353508e3a9b86520a1d82a6b1c4f8c4e64f2fbd2e9b0e90f01e34`.
Source96's original custom-wrapper `default` failure is preserved; sources97/98
also retain their CPU source-comparison/cached-dispatch test failures. The
compatibility repair changes only generated wrapper metadata and is disclosed
in the collector identity; it does not edit the collector package or statistics.
