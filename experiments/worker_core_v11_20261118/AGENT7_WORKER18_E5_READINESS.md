# Worker18 historical statistics readiness review

The existing source-a37 B / independent-B-repeat / F originals support a
reproducible **first observable change at the first linear consumer's input**.
They do not provide E5 for the current public Core lifecycle, and they do not
uniquely locate an internal RMS kernel error. No device run, remote command,
Runtime/provider change, raw-record edit, or new qualification was performed by
this review.

## Independent numerical result

`review_statistics_readiness.py` reads only each statistics root's original
`config.json`, `collector_identity.json`, and actual step/rank
`dump.json`, `stack.json`, `construct.json`. It takes no expected operator,
injection receipt, tensor payload, or target-hit argument. The unchanged
comparator is pinned to SHA256
`d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`.
It uses decimal values from the raw JSON and the fixed rule:

```text
abs(F - B) > 1e-8 + 1e-5 * abs(B)
```

| Comparison | Tensor records | Comparable statistic fields | Exact changes | Above tolerance |
| --- | ---: | ---: | ---: | ---: |
| B / itself | 3020 | 5974 | 0 | 0 |
| B / independent B repeat | 3020 | 5974 | 0 | 0 |
| B / F | 3020 | 5974 | 4347 | 4347 |
| Independent B repeat / F | 3020 | 5974 | 4347 | 4347 |

Both B/F comparisons independently select:

```text
step0 / rank0
Functional.linear.0.forward / input_args / 0
field: Max
B = 1.734375
F = 0.8671875
absolute difference = 0.8671875
tolerance = 0.00001735375
relative change = 0.5 (50%)
construct parent = Module.model.layers.0.self_attn.qkv_proj.AscendQKVParallelLinear.forward.0
```

Each full capture contains **6106 null fields**, including floating-point
Mean/Norm fields; **102 null fields precede the selected difference**. They
remain unobservable, never equal. There are no NaN, Inf, unsupported, or
incomparable fields in these supplied captures. The comparator's
`e5_statistics_comparable` and `whole_dump_comparable` flags mean that the
observable statistics align and can be compared; they do not mean every field
was observed. `full_numeric_prefix_observed` is false.

Only actual `step0/rank0` exists in each group. Config `step=[0,1,2]` is a
selection configuration, not evidence for three captured steps. There is one
F capture, so this review does not claim independent repeated-F stability or a
multi-step E5 result. No private tensor was read to choose the first difference.

## Why the RMS output statistics can be unchanged

After freezing the raw-only result, the source and capture boundaries were
reviewed separately. In the supplied raw data, the native
`NPU.npu_rms_norm.0.forward/output/0` and the first
`AscendRMSNorm` module output both retain Max `1.734375` and Min `-2.859375` in
B and F. The following linear input changes to Max `0.8671875` and Min
`-1.4296875`; its weight statistics are unchanged.

The exact historical `fault_injection/rmsnorm_return.py` applies a gain to the
module return **after the real RMS call returns**. It does not alter the RMS
kernel's internal reduction. The reviewed msprobe
`pytorch/dump/module_dump/module_processor.py` prepends its output hook
(`move_to_end(handle.id, last=False)`), so it observes the original module
return before the owned return-gain hook. The next real consumer receives the
effective changed return. This explains the observed boundary without
rewriting the raw record or demanding that the first difference be RMS output.

The local exact module-processor file matches the collector identity's SHA256
`4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7`.
The whole installed msprobe package was not independently rehashed by this
offline review; its file list remains the recorded collector identity. The
cause of all Mean/Norm nulls is not established here.

The historical collector uses the unedited package and public
`PrecisionDebugger.stop/start` to pause owned diagnostics/proxy arithmetic in
both arms. It does not filter model operators or synthesize a first-difference
tensor. Each capture records 1386 pauses and a completed, inactive window with
diagnostics depth zero. Raw stack inspection finds the legitimate dispatch
wrappers, but none of the reviewed diagnostic-body function names. This is
bounded corroboration, not an exhaustive purity proof from the absence of
stack frames. No record is removed by the new reader's stack check.

Two CPU tests exercise the actual return-gain adapter's hook order and
symmetric diagnostic-pause behavior. The real model's own later multiply
remains observable. These are **Torch CPU contracts**, not additional msprobe
or NPU observations.

## Source and request scope

Historical source-a37 is commit
`1e76ca0452683255ed52362835b05d24208ea1b5`; its 678-file source manifest hashes to
`0a71b5209274ceb16b28fbd302dae30a19c35b3dc3a8263ae15e09635d02fc1e`.
The probe, capture helper, return-gain adapter, comparator, scoped dispatch and
observer Git-blob bytes were checked against that manifest. All match.

The three case records name the same read-only source mount and image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
They have three distinct Worker identities and the same recorded model-weight
and model-source hashes. The model is Qwen3-0.6B, BF16, TP1, eager; the captured
request is one 10-token prefill, the fifth managed request / sixth driver
request after the initial healthy request. The whole scheduled-request window
includes input preparation, model APIs/modules, logits and sampler.

All three original raw-triad hashes agree with the references in their result
receipts. Scheduler token IDs and positions agree, and each driver request
mapping joins its recorded Worker request. The scheduler evidence explicitly
says its CPU mirror does **not** establish NPU input-byte equality. This review
does not use the runtime's `input_exact` or hit/success flags to upgrade that
claim. Reported cleanup has zero owned serving hooks, no active adapter or
observer, and the before/after healthy token result agrees; these records were
read only after independent numerical selection and do not grant lifecycle
qualification.

## Reproduce without a device

The original roots remain on 124 at:

```text
/home/17-fault-c2c3-20261117/worker18-statistics-card2-a2/
  natural-worker18-stats-b/probe/statistics
  natural-worker18-stats-repeat/probe/statistics
  natural-worker18-stats-f/probe/statistics
```

From a checkout containing this reader and the pinned comparator, run in an
existing CPU Python environment; this command does not load a model or execute
NPU operations:

```sh
R=/home/17-fault-c2c3-20261117
python -B -m experiments.worker_core_v11_20261118.review_statistics_readiness \
  --baseline "$R/worker18-statistics-card2-a2/natural-worker18-stats-b/probe/statistics" \
  --baseline-repeat "$R/worker18-statistics-card2-a2/natural-worker18-stats-repeat/probe/statistics" \
  --fault "$R/worker18-statistics-card2-a2/natural-worker18-stats-f/probe/statistics" \
  --comparator-sha256 d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2 \
  --out "$R/worker18-e5-readiness-new-attempt"
```

The output directory must not exist. Complete comparison details, including
null records, remain local. This commit contains only source, CPU tests, hashes
and the sanitized summary. It does not publish the old raw groups, runtime
identities, full report, private arrays, or a new accepted scope.

## Minimum remaining work for current public Core E5

Existing originals can be reused now for the historical source-a37 statistical
localization above. They cannot be reassigned to the accepted source66 public
Core group or later sources. In the reviewed current implementation,
`ActualTensorRequest.expect()` queues a request and `_begin_request()` starts
owned scopes inside the actual model pre-hook. It does not call the legacy
statistics-window start or set its statistics arm. The old
`ServingTensorGroup.configure_statistics()` also requires exactly four prior
requests and no active coverage group. It is not a drop-in public Core option.

A later targeted collector integration must preserve actual-request ownership,
root pre/post and logits hook order, symmetrical B/F diagnostics pauses,
unfiltered API/module capture and exception cleanup. It must explicitly state
which return boundary the module hook observes. It must not reorder hooks or
filter records merely to force an RMS-output first difference. CPU hook-order,
same-step append, empty/failed-window and cleanup contracts should precede a
new current-scope B/repeat/F capture. That targeted evidence need not rerun the
six already accepted functional groups solely to refresh their results.

The new reader adds missing-step, baseline instability, earlier-difference,
null, later-NaN, filtering, diagnostic-contamination and construct-drift
negative cases. Together with the existing comparator and capture-window
contracts, **28 CPU tests passed**. Production mechanisms, scopes and counters
are unchanged. All new live-qualification flags remain false.
