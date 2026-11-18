# Native26 isolated-request original reviewer

Implemented against integration `8e95733d5464b65df61658cbb0bf881d9347d443`,
which contains the source82 isolated producer. This contribution changes only
offline reviewers, CPU audit tests and this explanation. It does not execute
SSH, start a container, load a Native library or run an NPU experiment.

`review26_group_originals.py` now dispatches the explicit
`stage=c2_isolated_once` to `review26_isolated_originals.py`. The existing
cumulative `c2_once` and `c2_c3` reviews retain their separate 8/9-request
contracts. Relabeling an isolated result as cumulative does not pass review.
CLI arguments are unchanged; all original result/source/build SHA pins remain
required.

## What is independently reconstructed

Each real enqueue/arm/query/collect sequence must join the corresponding
complete-request delta original. Every original is byte-rehashed, its
`previous_original_sha256` must point to the previous original, and its
start/stop offsets must identify exactly one completed request. The auditor
reconstructs cumulative rows, Native outcomes, coverage calls and serving
requests and compares their canonical JSON, including value types, with the
final full original. It also recreates both query and collect RPC compact
projections. The final cleanup may release stock host storage; it may not
replace the recorded stock artifacts or earlier observations.

Two healthy complete Native traces independently determine the frozen census,
eligibility and actual target order. The caller's `--expected-writers` must
match every request; unsupported calls remain in the census and ledger. For
the intended 30-call/29-eligible FIA shape, this implies 64 group requests,
29 separate F/OFF pairs, 1,740 ledger records and 145 owned executions. These
numbers are consequences of that complete census, not a substitute for it.
The reviewer supports other complete census sizes without hardcoded layers.

Every F must execute exactly the next independently discovered target. Its
current residual hash must equal that target's independently recorded
Reference residual before F; the original pre-fault receipt is checked against
those values. No second target may execute F in that request. Its following
fresh OFF must restore the same target and the complete ordinary output.
Reference, Lossless, qualification OFF, isolated F and recovery OFF provide
exactly five owned writer/consumer outcomes per eligible target. Saved
`isolated_pairs`, hit counts, success flags or C2 booleans are never the source
of the target set or the evidence of actual execution.

The complete observation ledger is rebuilt in actual callback order, including
already-attempted targets, concurrency-blocked later targets, unsupported
calls and all OFF passthrough. Target leases, Native outcomes, effect-file
references, per-target recovery, final healthy request, released resources,
public unregister and detach are checked. A separate ordinary request after
unregister must remain healthy. Counted non-target cache runs between requests
retain the existing strict boundary rule; target and typed event gaps fail.

The typed FIA reviewer has an explicit isolated mode. It checks all actual
typed calls and exact descriptors, owned execute counts, public return objects,
consumer bindings, descriptor release and every full stock/owned output file.
Its default cumulative mode is unchanged. The independent getter/build audit
is still required, as are the Worker implementation source and separately
compiled Native source snapshots. The isolated implementation hash explicitly
includes `serving26_isolated.py` and its scope; source82 is not relabeled as
the Native compiler75 snapshot.

## Scope and validation

The new tests execute the current public CPU driver, rather than fabricating
successful result receipts. The QMM fixture has three targets; the FIA fixture
has four complete calls and three eligible targets, preserving its unsupported
call. Both execute all 12 group requests plus ordinary post-unregister health.
An explicitly extended CPU ABI wire fixture rehashes all 23 FIA stock/owned
output files and 111 typed records. These are CPU contracts, not NPU evidence.

Negative cases cover altered original bytes, broken delta SHA chains, changed
offsets, missing/null deltas, final-merge differences, lost targets or outcomes,
extra F admissions, pre-F residual contamination, a different OFF, falsified
pair summaries, passthrough changes, resource gaps, typed record loss and
descriptor leaks. Boolean values cannot stand in for integer observations.
The actual pre-F budget-refusal driver preserves five completed controls and
zero F, and is reported as partial without qualification. A separate scope
test checks the current isolated implementation hash; it does not replace the
CLI's full source/compiler provenance review.

The final four-suite CPU regression passed 65 tests: the new isolated original
review, existing cumulative C2 and C2/C3 original reviews, and build-binding
review. CLI help and `git diff --check` also passed. These are one local
regression invocation; they are not 65 NPU requests or independent device cases.

All offline results retain `c2=false`, `c3=false`, `e5=false` and
`public_qualification_granted=false`. `group_completion_reconciled` means only
that complete saved originals agree. The public live proof is not recreated
from JSON. Residual and complete model-output equality use original runtime
digests; only FIA stock/owned tensor files are reread as complete bytes. Any
model-output/token changes are reported separately, and no statistics E5
conclusion is inferred. The 64-request source82 NPU attempt still needs its own
complete-original review on 124.
