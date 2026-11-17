# Fault18: fresh standard-Worker raw statistics review

The fixed-tolerance first observable B/F difference is
`Functional.linear.0.forward/input_args/0`, field `Max`:
**1.734375 → 0.8671875**. The absolute difference is0.8671875, the
baseline-reference tolerance is0.00001735375, and the relative change is50%.
This locates the first observable consumer input. It does not uniquely prove
the upstream RMSNorm return boundary from statistics alone:102 preceding fields
are null and remain unobservable.

The comparison reads only the original `dump.json`, `stack.json`,
`construct.json`, configuration and collector identity. It does not read native
hits, provider effect records, complete tensors or an expected operator name.
It uses `abs(F-B) > 1e-8 + 1e-5*abs(B)`, retains the original JSON decimal
precision and never treats null as equal or NaN/NaN as valid equality.

| Pair | Alignment | Exact changes | Exceedances | First exceedance |
| --- | --- | --- | --- | --- |
| B self | Complete supplied structure/stack/construct/config/collector |0|0|None|
| Independent B repeat | Same; raw dump bytes also identical |0|0|None|
| B/F | Same |4347|4347|First linear input Max, as above|

Each capture contains3020 tensor records,5974 finite fields and6106 null fields;
there are no NaN, Inf or unsupported statistics. All supplied observable fields
are comparable. This is not a comparison of every tensor element or proof that
the6106 null fields are numerically equal. The full audit retains their exact
locations and the102 unobservable fields before the reported first difference.

The actual scope is Qwen3-0.6B, standard vLLM0.18 Worker, BF16, CANN8.5.1,
Ascend910B4/card2, TP1, fixed fresh prefill and one sampled token. B, independent
B repeat and F use three fresh temporary Workers. Each captures the same fifth
managed request after the same warmup and four qualification requests. Each
capture has one actual `step0`; it is not a three-step decode sequence. The
controller completed all three cases with exit0. The numerical comparison was
frozen separately from any later runtime/recovery review; it does not reissue
the earlier C2/C3 proof or enlarge its workload scope.

The collector is the recorded unedited msprobe package, configured
`statistics/mix`, `summary_mode=statistics`, `precision=high`, with no operator
filter. The shared capture controller uses public stop/start around owned
diagnostics and proxy arithmetic; actual model APIs, module returns, logits and
sampler remain in the capture. This exclusion is disclosed in collector
identity. No CPU-FP64 replacement collector is used in this experiment.

| Published summary | SHA256 |
| --- | --- |
| [B/F](worker18_statistics_bf_summary.json) | `1059d4f9154160a5cc47f77ef30fe844cf2ab7873655478aba20ec49e49c6c2a` |
| [B repeat](worker18_statistics_repeat_summary.json) | `8fdc30db2e234745fb76a2e43c478e631a30a0d3d72f9ac4a00ddded951c85de` |
| [B self](worker18_statistics_self_summary.json) | `e5eecfb1c7589d66e18ccde57450a4333ef40c10f08ff42b05a6df5b3626a940` |

The frozen comparator SHA256 is
`d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`.
The complete B/F audit SHA256 is
`ae655acf59b9e6184a3eb2fd93bc06e8058832b9c28a5bc550455d605ca4d09e`.
Original captures stay under
`/home/17-fault-c2c3-20261117/worker18-statistics-card2-a2/natural-worker18-stats-{b,repeat,f}/probe/statistics`;
full audit files stay under `worker18-statistics-raw-review-a1` on124. Only
these summaries and documentation are published. This new standard-Worker
evidence is separate from the older HF model capture.
