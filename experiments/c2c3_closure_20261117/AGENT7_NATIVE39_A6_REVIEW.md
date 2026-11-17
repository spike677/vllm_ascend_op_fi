# Independent Fault39 a6 review

The recorded Native39 a6 run supports **bounded standard-Worker C2 and C3** for
root review. This audit does not grant a runtime proof or public qualification.
The scope is the actual Qwen3-0.6B vLLM 0.18 Worker, TP1, BF16, fixed 14-token
fresh prefill, CANN 8.5.1, Ascend910B4/A2, physical card 5. It does not qualify
decode, request concurrency, TP2, another model, or another CANN/SoC build.

| Independent check | C2 once | C3 every-match |
| --- | --- | --- |
| Complete discovered target set | 113 | same 113 live target identities |
| Actual eligible calls / hits | 113 / 113 | 226 / 226 in two fresh requests |
| Coverage ledger including OFF | 339 calls | 452 calls |
| Referenced artifacts with matching original byte hashes | 342 | 568 |
| Every target Native effect and original storage preservation | passed | passed twice |
| Per-target OFF and complete model logits restoration | exact | exact |
| Fault repetition | one request only | every target output and whole logits exact |
| Group drain and same-Worker healthy generation | passed | passed |

Each phase observes 678 public calls, of which 113 execute eligible Native
MatMul and 565 are non-targets. Five Native cache registrations are reused across
113 actual module/call sites. The audit keys the **site plus cache identity and
epoch**, so cache reuse neither merges targets nor fabricates extra targets.
The real scheduler IDs join the driver mapping, the model prehook, the Native
observation, and the coverage ledger. C3 has the same scope, target lifetimes,
logical owner, and a later generation than its live C2.

The substituted inputs pass finite/device-readback/exact-cancellation checks.
All 113 target outputs change; 112 later original inputs already contain
nonfinite values from upstream propagation. This is not 113 isolated causal
experiments. Only the two designated input formals use temporary storage, the
original model storage hashes remain unchanged, and every synchronization/run/
free return code is zero. Native leases/inflight return to zero. Healthy
generation before and after both groups returns token `[576]`, text ` The`.

The installed code is from **source-a21 / packages-a5**, not the subsequently
built packages. Imported source bytes, the observer's three-file aggregate,
the recorded implementation/environment scope hashes, the frozen probe, and
all 12 Native build artifacts match. The controller exited zero with recorded
image ID `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.

## Independent statistics

Only original `dump.json`, `stack.json`, `construct.json`, config, and collector
identity are passed to the fixed-tolerance comparator. Runtime records above
do not select the first difference. The formula remains
`abs(F-B) > 1e-8 + 1e-5*abs(B)`; nonfinite transitions are explicitly classified.

* C2 `step2` reference versus `step6` fault: first observable difference is
  `Functional.linear.0.forward/output/0`, **Max 5.1875 → +Inf**. There are
  **126 preceding null fields**. Both captures have 6,602 null statistics, and
  F has 1,906 NaN statistics. Whole-capture E5 remains **false**. The first
  observable transition is valid evidence; preceding nulls are not equality,
  and the statistics alone do not uniquely prove the Native fault mechanism.
* Baseline self-comparison is valid; it is a self-check, not a repeat experiment.
* C3 has config and collector metadata but **no captured raw triad**. The audit
  reports `MISSING_RAW_CAPTURE`, E5 false. It neither synthesizes these files
  nor transfers C2's statistics to C3. Coverage and recovery are separately
  supported by the actual C3 records above.

## Reproducibility and retained boundaries

The compact, publishable result is
[agent7_native39_a6_summary.json](agent7_native39_a6_summary.json), SHA256
`96979970237b60ecba385b5c2af6a27b6a510176f667095d80010e9db2f47ad2`.
The original result remains on124 at
`/home/17-fault-c2c3-20261117/providers39-a6/providers39-qwen-c2c3/result.json`,
SHA256 `aa77add1286acd563b9a080399820aa3d088c35422c87f1c24f9066c5f1f58d0`.
Full independent sidecars remain in `agent7-native39-a6-audit-v3` under that
remote root. Earlier v1/v2 reader attempts remain unchanged; one stopped on
the missing C3 capture, and one used an incomplete reader-side observer-hash
formula. The final reader uses the actual source-a21 formula.

Run `review_native39_group_124.py` with `--attempt-case`, a fresh `--out`,
`--source .../source-a21`, and `--case-manifest .../providers39-a6-cases.json`.
The actual run used the independently reviewed raw comparator from source-a19
(including typed boolean-extrema handling), SHA256
`d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`.
No containers or NPU calls were started by this audit. Eighteen local tests
cover the raw/coverage auditors and cache-site, missing-capture, and immutable
sidecar review boundaries; they are CPU contract checks, not additional NPU
acceptance cases.

For the 17-mechanism rollup, this is one native Fault39 route, not an additional
mechanism for every model/node. If root grants this scoped C3, the existing
Fault18/20/39 checkpoint becomes **3/17 C2 and 3/17 C3**. The separate natural
HF Fault18 scope and Fault39 output proxy must not add another count. It does
not establish all27 nodes, blanket multi-model coverage, or full-capture E5.
