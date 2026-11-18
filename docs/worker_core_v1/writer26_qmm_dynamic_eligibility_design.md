# QMM physical adjacency: retained gap and read-only next step

This is a design review, not an implementation of dynamic qualification. The
source68 public C1 evidence retains 135 actual calls per request: 2 stable
resolved, 106 proven nonadjacent and 27 unresolved. Of the latter, 3 lack a
current producer and 24 change physical eligibility between the two healthy
requests (18 lose adjacency, 6 gain it). Root's comparison found equal target
keys and all stable Native/formal fields for those 27 entries. Addresses are
already excluded from stable target identity; removing address fields cannot
repair these actual allocator transitions.

The committed source68 review has SHA256
`f1fac5eb3ead02b2000564820074922741ac86a9ca36ce0e86b27fc5d21abc1a`
and reports 17,286 checks, 1,080 actual calls, no review inconsistencies, and
only C1 evidence. None of this makes the 24 variable targets unsupported or
permits selecting the 2 stable targets as the complete C2 denominator.

## Mechanism predicate to preserve

`residual26.allocation_pair` requires two distinct currently active allocator
blocks in one actual segment, with the residual block ending exactly where
the output block starts. Both public tensor spans must lie in their blocks.
The underflow shift is actual padding plus 32 bytes, bounded by 4096 bytes.
`writer26_group_core.h` additionally checks complete span sizes. The mechanism
writes from the actual output address minus that shift, permits only the final
32 residual bytes to differ, and proves complete padding/output/residual
restoration. Actual writer and next consumer each execute once.

`writer26_experiment._allocation` distinguishes positively proven nonadjacency
from missing, ambiguous, same-block or unverified geometry. Those unknown cases
cannot be converted to exclusions. The experiment retains real output and
residual owners until the actual consumer completes; shortening those lifetimes
to influence allocator reuse would remove the recovery proof.

## First action: analyze existing full allocations without another F

Source68 already contains 924 relevant allocator records, 132 current-producer
calls across seven pre-uninstall requests. Root can rehash these existing raw
files and build a full 135-by-7 availability matrix. Keep the F request separate
from healthy/Reference/lossless/OFF controls. For each actual call retain:

- Request/phase, ordinal, target key and stable formal digest.
- Actual producer/writer/consumer call and thread IDs, original execution counts.
- Segment identity and each block's start/end/size/requested size/state/stream.
- Signed residual-to-output gap, padding/shift, complete tensor-span containment.
- Recomputed eligible, proven-nonadjacent or unknown result and original reason.

Join by actual site identity, never by address alone. Explain transitions as
different segments, reversed order, intervening free/active blocks or missing
evidence. Preserve the union, intersection and all earlier failures; do not
select a convenient final pair of healthy requests.

The existing workload already fixes prompt, eager TP1, single request, KV
allocation size and prefix-cache reset. A further warmup or preallocation policy
would need a reviewed actual public workspace API and the same policy for
healthy, R/L/OFF and F phases. It must prove the unchanged live allocator
predicate at every admission. Relocating tensors, manufacturing adjacent arenas,
per-layer padding allocations or altered baseline values would change the
mechanism and are outside this plan. No such stabilization is currently proven.

## Why conditional eligibility is a separate contract

The current CoverageSession treats a frozen eligible lease later presented as
ineligible as `DISCOVERED_TARGET_CLASSIFICATION_CHANGED`. The writer also retains
failed fresh-geometry opportunities in the denominator. Silently skipping the
24 changing targets would violate both contracts.

A future dynamic contract would require the full 135-site universe and all
potentially eligible sites, actual availability per call, fresh R/L/OFF whenever
needed for that physical predicate, and a bounded window that actually exercises
every required target. Any newly eligible unqualified call must prevent F and
remain visible; no timeout or failed candidate may disappear. Root has not
authorized implementing that changed qualification contract in this task.

The present implementation therefore keeps QMM whole-group refusal on these
24 transitions. The new isolated C2 execution profile changes request scheduling
only; it neither repairs nor bypasses dynamic physical eligibility.
