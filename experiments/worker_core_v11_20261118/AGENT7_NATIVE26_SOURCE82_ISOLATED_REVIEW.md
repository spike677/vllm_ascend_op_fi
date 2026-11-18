# Source82 Native26 isolated C2 evidence review

The three supplied records are consistent with a completed **isolated, one-target
F/OFF per fresh request** experiment for `26.fia_destination`. They do not confer
live authority on a saved JSON file. The root reviewer can record this completed
attempt as scope-limited isolated C2 evidence; this review grants no C2, C3 or E5
qualification and changes no production implementation.

## What was independently checked

This local, CPU-only review rehashed the supplied probe result, independent
reader output and controller receipt; recomputed their request, target, effect,
recovery and proof cross-links; and checked the current source gates. It made
**598 checks with no issues**. All 36 probe-recorded source hashes and eight
reader/helper hashes match the files inspected at `cc7fd07b`.

The source84 read-only reader had previously checked the original remote
artifacts: **69,520 checks, no issues**, including 273 JSON originals, 1,803 target
artifacts and 205 full stock/writer binaries. These categories overlap and must
not be summed. This local review did **not** download or reread those remote
artifacts; it cross-reviewed the pinned reader report and probe result. The
reader CLI enforces `--result-sha256`, but its output does not embed that argument.
The exact three inputs used here are consequently recorded together below and
in [the machine-readable summary](native26_source82_isolated_independent_summary.json).

| Input | SHA256 |
| --- | --- |
| Independent reader output | `3651278990a240f294f441d4ba66ea62fbfadf1b024578afbb3b8bace5c977f3` |
| Final probe result | `f84c1cc357ae5cce58b3b1d52d1e4a977e887de5acca883e8c0a9d08af654643` |
| Controller receipt | `77779e19ac4adbaa9443393d81f37eefb26762df1ec2061fea348f4af2c31319` |

## Actual scope and observed outcome

The controller identifies **physical card 5**, rank 0, SmolLM2-135M, CANN 8.5.1,
Ascend910B4/A2 and the standard `vllm_ascend.worker.worker.NPUWorker`. The image is
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
This is fixed-input fresh-prefill evidence, not decode, TP2, another model,
CANN 9.x or another hardware generation.

| Reconciled item | Actual result |
| --- | ---: |
| Calls retained in every group request | 30 |
| Eligible targets / retained unsupported target | 29 / 1 |
| Group requests / post-unregister health request | 64 / 1 |
| Independent F/OFF pairs | 29 |
| Owned Native executions: R, lossless, qualification OFF, F, pair OFF | 145 |
| Complete F/OFF coverage ledger | 1,740 calls |
| Eligible calls observed during F requests / actual F admissions | 841 / 29 |
| Original delta references / final merged original | 64 / 1 |
| RPC references | 206 |
| F complete-output runtime digests changed / returned tokens changed | 29 / 29 |
| Non-F group requests with exact healthy digest and returned output | 35 |

The unsupported ordinal 0 is retained with
`CURRENT_FRESH_FORWARD_HAS_NO_RESIDUAL_PRODUCER`. The inventory's other 29 rows
remain labelled `OBSERVED_NOT_QUALIFIED`; those preliminary labels are not used
as the acceptance oracle. The separate original R/lossless/OFF and F/consumer
records establish the actual per-target results.

Every F has one admission, a distinct ordinary enqueue and matching target in
the complete census order. Its current residual digest equals that target's
independent Reference and records zero preceding faults in that request. Its
following OFF is another fresh request. All 29 target recoveries, final Native
and public resource counters, one same-Worker health proof and the separate
post-unregister output agree on restoration. The controller exited 0 without
OOM; container removal is **not** used as proof of OFF recovery.

There are 3,985 typed records and 205 rehashed stock/writer binaries
(3,070,080 bytes). Complete residual/logits arrays were not independently
rehashable: their equality/change is the original runtime's full-output digest
observation. Token comparison is separately reproducible from the probe result.
Across 63 between-request gaps, the reader retains 819 additional non-target
calls under the strict paired-counter contract; it does not identify those
individual operators. The last typed snapshot is after the final actual health
request, not a separate post-cleanup typed snapshot. These limits remain visible.

## Source and live-proof boundary

Worker source82 is `8e95733d5464b65df61658cbb0bf881d9347d443`, manifest
`989dc35f6c171cae7d3f0b5e396d0150bcc2a4501bb9186f793d42fd30fe12e4`.
The Native compiler used source75
`f800a4884f22688737a4ab74a051f4f922ab4b03`, manifest
`62b700a85663d0955dbc4d99b318786ba75f12c740b0fc46b82666dee5a5f04d`.
The resulting Native build manifest is
`dc6150607edf68e3868d5c7d5c976c1ce5458dbba44190e9daf6d85ae779cfa0`.
Those source snapshots are deliberately distinguished. The reader rechecked
source/build binding; it did not reread external model weights/source or the
entire parent Native bundle.

The original coverage report records **one** C2 proof issuance for all 29 target
IDs, exact scope/owner/generation/session, with recovery hash matching the
complete restoration report. Its consumption is `null`. The issuance SHA is
`d3f16cf14672fda2988ffa71d1b464d9377552bdbfd3a88d6a7a4e64e68239ae`.
This establishes a historical recorded issuance, not a live object available to
this offline reviewer or a new Worker.

The inspected public gates explain that boundary:

- [`CoverageSession.c2_proof()`](../../fault_injection/coverage_session.py)
  requires the once contract, complete independent observation, every target's
  one hit, all outcomes and completed recovery; it registers the exact object
  identity and sealed fields in the current process.
- The every-match constructor requires that **same issued object**, matching
  scope, owner, candidate set and target IDs, with an advancing generation.
  A serialized proof or a proof from another process cannot satisfy it.
- [`CoreWriter26IsolatedCoverage`](../../src/demoncat/runtime/vllm_ascend/worker_extension/serving26_isolated.py)
  explicitly allows only `once` and adds its own source hash and isolated scope
  to the implementation identity.
- [`Fault26CoreProvider`](../../src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault26_core.py)
  starts the live C2→C3 transition only for the existing `c2_c3` stage. It does
  not perform that transition for `c2_isolated_once`.

No additional live step is missing merely to document this already completed
isolated C2 attempt. **Source77's cumulative C3 is a different source/execution
scope and cannot complete source82's isolated C3 claim.** The smallest next
development is an explicit same-Worker, same-scope isolated C2→C3 transition
which consumes the newly issued object. Its changed implementation needs a new
paired C2/C3 run for this mechanism; the retired source82 object cannot be
restored from JSON. The six unrelated accepted mechanisms need no rerun for
this step.

E5 remains `UNADAPTED_COLLECTOR_BOUNDARY`. No new `statistics/mix` was collected
here, and no independent statistical first difference is claimed. That is a
separate remaining acceptance task, unaffected by the changed tokens or Native
coverage results.
