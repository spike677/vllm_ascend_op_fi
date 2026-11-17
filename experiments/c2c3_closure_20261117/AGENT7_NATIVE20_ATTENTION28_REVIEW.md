# Native20 and new Fault28 statistics — independent review

The source-reviewed Native20 auditor was independently re-run against the
original completed `providers20-group-a1` receipts. C2 has135 targets/hits and
C3 has270 matching calls/hits over two requests; all408/678 referenced artifacts
match their original bytes. Each actual QMM call has its own typed current
formals, original continuation, temporary readback and output proof. All-target
OFF and complete healthy logits restore exactly. The independently generated
review is byte-identical to root's earlier report, SHA256
`777e01d6837dba3fc4868d18e8253faac7abbda482166836a7d40ece29af7438`.

This supports root's existing **standard DeepSeek-V2-Lite-W8A8 Worker, TP1,
BF16, fixed3-token fresh-prefill, CANN8.5.1/A2** qualification. It does not add
a mechanism/node or qualify decode, concurrent requests, multi-worker use,
or another model/build. Later inputs include propagated changes; it is not135
isolated causal experiments. The rerun did not create a Worker or use an NPU.

## Raw statistics results

The comparisons below use only original dump/stack/construct, config and
collector identity. The independent comparator never reads Native hit logs,
expected first locations or full tensors. The threshold is unchanged:
`abs(F-B) > 1e-8 + 1e-5*abs(B)`.

| Actual capture pair | Independent result | Limitation |
| --- | --- | --- |
| Native20 C2 step5 baseline / step6 fault | First observable: `NPU.npu_quant_matmul.0.forward/output/0`, Max16.25→14.0,13.846153846% |132 preceding null fields; two later NaN statistics prevent whole-capture E5 |
| New Fault28 legal B/F, step0 | Two exact statistic changes, zero above threshold | Observable statistics cannot locate a first difference at the fixed tolerance |
| New Fault28 replay B/F, step0 | First observable: `NPU.npu_fused_infer_attention_score.0.forward/output/0`, Mean6.677826149825705e-7→−2.9812251511884824e-7 | Two preceding null fields; single standalone call, not a complete-model run |

Native20's raw baseline self-comparison is comparable. B and F each retain
10,266 null fields; the fault also retains two NaN fields. The valid first
observable QMM output change does not convert those missing/NaN fields into
equal data or uniquely prove the mechanism from statistics alone.

The new Fault28 replay first difference exceeds tolerance: absolute difference
`9.6590513010141874e-7`, threshold `1.0006677826149825705e-8`. Independently of
that Mean result, **Norm118.30810159723288→116.27198217667856**, a
**1.7210312675678527%** change, exceeds the0.1% amplitude goal. Max and Min are
unchanged. Mean is not used for amplitude acceptance.

For the legal pair, Norm118.30810448461165→118.30810159723288 changes by
only **0.0000024405587281%** and remains below the fixed tolerance. Its Mean
also changes exactly but is below tolerance. These are valid negative results,
not binary equality. They must not inherit the replay or historical results.

Both Fault28 pairs have18 finite statistic fields, six null fields, and no
NaN/Inf. The null fields are retained: the bool mask has unobservable Mean/Norm,
and the empty secondary FIA output has four null statistics. Two of these
nulls precede the first observable replay difference. `e5_statistics_comparable`
means the observable statistics can be compared, not that all preceding data
are observable/equal or that a unique causal root has been proved.

## Collector and scope

These are **new** captures with disclosed
`attention-msprobe-cpu-fp64-v1`: `PytorchDataProcessor.get_stat_info` was replaced
at runtime. The installed msprobe package files, serializer, record selection,
stack and construct remain unchanged. Complete supported tensors are copied
in their original dtype to CPU, then reduced in CPU FP64. Unsupported types
retain original handling, nulls and nonfinite values. This is not the unmodified
collector and does not reinterpret any historical dump.

B/F collector identities match. The actual source-a24 adapter matches recorded
SHA256 `1e66f220c1098a71eca9611bbc0cb582a088e7e47819269a5a8dbf3a8bb50b48`;
the original processor source is bound to
`e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011`.
The source review confirms complete copies/reductions, no sampling, no data
modification, no injector input, and no reads of historical dumps.

Actual `step0` is the **second identical standalone FIA invocation**, not a
decode token or full model. Q shape is[8,16,2048,128], K/V[8,8,2048,128], FP16,
BNSD. The observable replay result supports this bounded operator-level
capture only. It grants no model/Worker C2/C3, no all-layer/request coverage,
no three-step stability result, and no logits propagation conclusion.
Native launch verification remains a separate root review and is not used to
choose the statistics first difference.

## Artifacts and reproducibility

Only generated summaries and hashes are added here:

* [Raw comparison summaries](agent7_native20_attention28_statistics_summary.json),
  SHA256`10ab529f807972a695e85bd44ad0e1036ec29307d2bf604433191478e3406699`.
* [FIA output statistic amplitudes](agent7_attention28_statistics_amplitudes.json),
  SHA256`4b4fe671cee65ce8a1d168a9161233795895fc308d4bba1496f50f4b56e50f55`.
* [Native20 independent rerun](agent7_native20_independent_review.json), hash above.

The remote root is `/home/17-fault-c2c3-20261117`. Complete audit sidecars remain
in `agent7-28-native20-fixedtol-a1` and `agent7-native20-review-a1`. Native20
source-reviewed audit script is `review_native20_124_a2.py`, SHA256
`d4fd9ab4b26275b5112adc5525b6caf8751a74f4c36bf3613c67fb4b8b647aba`, the same
script published in root commit2a806b7. The raw comparator is source-a19
`fault_injection/statistics_audit.py`, SHA256
`d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`.

For each pair, `audit_statistics.py` accepts explicit baseline/fault triad,
config, collector and fresh output paths. New Fault28 inputs are
`reviewed-next8-a2/fia-{legal,replay}-cpu-fp64-{baseline,fault}/run`,
`observation/step0/rank0`. Native20 inputs are
`providers20-group-a1/providers20-deepseek-c2c3/c2/statistics`,
`observation/step5/rank0` and`step6/rank0`. Every exact input hash and full audit
hash is included in the summaries. Original raw statistics, private tensors
and native binaries remain on124 and are not committed.
