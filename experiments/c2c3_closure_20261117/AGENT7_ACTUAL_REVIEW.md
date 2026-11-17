# Independent review of actual 124 receipts

Reviewed on 2026-11-17. This review reads immutable saved evidence and source;
it does not run an NPU, start a container, or grant runtime/public qualification.
Root remains the qualification authority. The machine-readable review and all
input/report hashes are in `agent7_reviewed_scopes_summary.json`. New raw
statistics, canonical evidence objects, native artifacts and full reports remain
under `/home/17-fault-c2c3-20261117` on 124.

## Recommended bounded qualifications

| Mechanism and exact route | Reviewed actual coverage | Recommended scope | Not granted by this evidence |
| --- | --- | --- | --- |
| 39.matmul_range, Qwen3-0.6B standard vLLM Worker, BF16 fixed fresh prefill, TP1 | C2 113/113 targets; 342 referenced artifacts match actual bytes | C2 for this Worker/source/build/workload | C3; cross-model/version/TP claims; complete E5 |
| 18.api_half, Qwen3-0.6B natural HF eager prefill | C2 113/113; C3 226/226 matching calls in two forwards; 229 references per report match | Separate model-local C2/C3 | Standard Worker qualification from HF evidence |
| 18.api_half, Qwen3-0.6B standard vLLM Worker, BF16 fixed fresh prefill, TP1 | C2 113/113; C3 226/226; 229 C2 and 455 C3 references match | C2/C3 for this Worker/source/workload | Worker E5, decode/concurrent/multi-worker/general model qualification |
| 41.tail_column, standard Qwen Worker first pure decode | One native intervention in the first selected target, out of 28 observed candidates; 16 selected elements changed | Limited C1 | C2/C3; native out-of-bounds root cause; unrestricted cache shapes |
| 39.output_proxy, natural Qwen HF prefill | One selected returned `aten.matmul` storage, out of 140 eligible calls with consumers; one element changed | Limited model-local C1 | Standard Worker, C2/C3, arithmetic root cause, complete E5 |

All these reviewed runs use actual 124 Ascend910B4/A2. The Worker and HF rows
above use the pinned CANN 8.5.1/vLLM image, with exact source and image identity
recorded. They do not extend qualification to CANN 9.x or A3.

### Fault39 native target identity and recovery

The 113 independently observed eligible MatMul calls reuse **five** actual Native
cache identities. A target is the full module/call-site plus cache address/epoch
and Worker/model lifetime, not the cache address alone. Each of nine actual
requests contains 678 observed public calls: 113 eligible Native calls and 565
non-target public calls. The Native counter and public observer reconcile before
admission. Three lossless modes, target OFF hashes and the complete healthy logits
all match the reference exactly. Native synchronization/copy/run/free statuses
are zero, original storage remains unchanged, and group resources/hooks clear.

All 113 injected outputs change and become nonfinite. After the first target,
112 calls already receive propagated nonfinite original inputs. This is recorded
separately: each pair of **substituted** inputs remains finite, device readback is
verified, and the exact-cancellation construction is checked. Do not describe all
113 original model inputs as healthy or attribute each later difference solely to
its local intervention. C2 is coverage of those interventions, not 113 isolated
causal experiments.

`providers39-a5` subsequently refuses its first C3 discovery with
`LOGITS_HEAD_OR_BIAS_CHANGED`. Its prior completed C2 evidence is preserved; that
attempt provides no C3 qualification. Actual installed adapter/observer files
match source-a18, and all 12 recorded Native build artifacts match their byte
hashes. The top-level attempt failure is not relabeled as an overall pass.

### Fault18 scope and actual effects

The model-local and standard Worker receipts are independent runs. Both have
113 eligible RMSNorm return targets, actual consumers, exact substituted-return
readback, healthy repeated baselines, and same-process complete-output OFF
recovery. The Worker run additionally has nine real scheduler request IDs joined
to the public driver mapping, equal captured model inputs, and actual full hidden
and `[1,151936]` logits returns. Its C3 proof is obtained from the same live C2
session scope, not deserialized from the model-local report.

The HF provider implementation hash matches source-a17; its actual weight file
was independently hashed and matches the recorded 1,503,300,328-byte safetensors.
The Worker provider's seven-file hash matches source-a20. Both bind actual weights
and model-source manifests. The Worker has zero owned serving hooks, active
adapter and independent observer after cleanup; the healthy generation token
before and after remains identical.

Fault18 is an RMSNorm returned-value half proxy, including its reviewed actual
Worker return boundary. This review does not reinterpret it as an internal
Reduce arithmetic defect.

### Fault41 and Fault39 output proxy

Fault41 records the real cache writer, same live value-cache storage, actual slot,
legal bounded selected span and returned TND FIA consumer. Native readback,
restoration and release all pass; complete healthy model output is restored.
The observed single mutation changes 16 tail elements; it does not prove that all
28 candidates are injectable or restored as an all-target group. Source-a15's
recorded files and the native ACL library match their hashes. The proxy uses legal
addresses; no actual native overflow is claimed.

The natural Fault39 output proxy records original `aten.matmul` execution, its
returned storage, a real `aten.mul` consumer and one changed output element. The
model output develops 1,519,360 NaNs, then OFF restores exactly with zero leases.
This is a post-output proxy, distinct from Fault39's native input-substitution
arithmetic mechanism and its standard Worker coverage.

## Independently recomputed statistics

Only the original B/F triplets, configuration and collector identity enter this
stage. Fixed baseline-reference rule: `abs(F-B) > 1e-8 + 1e-5*abs(B)`. No target,
hit log, requested operator or full tensor selects the first difference.

| Actual capture | First observable exceedance | B → F | Earlier null fields | Whole-capture limitation |
| --- | --- | --- | ---: | --- |
| 18 HF, actual step0 vs step2 | `Functional.linear.0.forward/input_args/0`, Max | 1.734375 → 0.8671875 (50%) | 84 | 9,796 nulls retained; no NaN; baseline repeat has no observable difference |
| 41 Worker, B/F actual step0 | `Functional.linear.1.forward/output/0`, Max | 3.0625 → 2.96875 (3.0612%) | 138 | 6,072 nulls retained; downstream difference does not uniquely identify cache mutation |
| 39 output proxy, B/F actual step0 | `Torch.matmul.0.forward/output/0`, Max | 536.0 → +Inf | 228 | 9,796 nulls and 7,492 NaNs; whole-capture E5 fails |
| 39 native Worker C2, actual step2 vs step6 | `Functional.linear.0.forward/output/0`, Max | 5.1875 → +Inf | 126 | 6,602 nulls and 1,906 NaNs; whole-capture E5 fails |
| 28 FIA legal, B/F actual step0 | None | No observable exceedance | — | 10 finite fields equal; 14 nulls retained |
| 28 FIA denominator replay, B/F actual step0 | None | No observable exceedance | — | 10 finite fields equal; 14 nulls retained |

The Fault28 pairs are aligned and numerically comparable **where observed**;
this does not satisfy a first-difference localization requirement. Neither
inherits the historical 0.7968% Norm result. Null Mean/Norm values are not proof
of equality. Fault18 Worker statistics are explicitly
`SEPARATE_STANDARD_WORKER_CAPTURE_REQUIRED`; the HF result is not transferred.

## Reproduction and review provenance

`review_recorded_runs_124.py` performs the CPU-only review against explicit saved
paths and a specified immutable audit-source snapshot. The executed comparator
is source-a19, SHA
`d5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2`.
It includes the disclosed interpretation of declared `torch.bool` Max/Min;
ordinary floating statistics cannot use booleans to bypass comparison.

Successful final sidecars are `agent7-review-a2` and
`agent7-worker18-a1`. Initial `agent7-review-a1` mistakenly joined Native39 by
cache key alone; it remains saved, and the correction uses the full existing
target identity. `agent7-review-a3` exported the Worker18 references correctly
but stopped at an audit-reader field-name mismatch; its replacement reads the
actual scheduler `internal_request_id`. No test or original evidence was rerun
or modified to repair these offline reader errors.

Canonical JSON exports contain only exact recorded objects (or the exact
event-plus-output/discovery compositions hashed by the executed implementation).
Every export must match a digest already referenced in the original receipt;
otherwise it cannot satisfy the audit. These are evidence materializations,
not new observations. All qualification flags in the audit remain false for root
to make a separate, scoped acceptance decision.
