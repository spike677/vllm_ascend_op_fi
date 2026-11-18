# Output39 failed independent discovery originals

The source77 card2 a2 Worker completed its six C1 requests and then refused
the explicit collection with `POSTWRITE_INDEPENDENT_DISCOVERY_INCOMPLETE`.
The mapped environment observation had already succeeded. The 113 actual
projection calls and their supported/unsupported counts do not identify the
missing module boundary. The current evidence cannot establish that cause.

`ProjectionCoverage._new_observer` computes `discovery_diagnostics` before
raising this refusal. Previously the Provider reached `_capture('collect')`
only on success, so the complete in-memory failure diagnostics were absent
from the saved originals. This increment changes only failure recording and
readback; the observer, candidate statuses and coverage admission are unchanged.

Every failed explicit collection attempts an exclusive-create
`group-originals/NNN-collect_failed.json`. It retains the actual group report:
both independent forward inventories, all candidate statuses, structural
module classification, actual module calls, parameter-consumption records,
all six completed request records, and the original exception. Initial group
preparation failure is latched through the existing Provider failure path.
Neither cleanup nor a secondary artifact-write failure removes that reason.

`group_originals` references include the file SHA256/byte length and
`group_discovery_sha256`. Read-only query exposes `evidence.group_discovery`
with that diagnostic digest, completeness, eligibility counts, unresolved
candidates, module calls, structural classifications and matching original
references. Full forward/candidate/parameter records stay in the referenced
original, avoiding another large RPC copy. Query does not collect or run a
model/Native operation and still verifies every referenced file's bytes.

If artifact writing fails, the original collection error is returned and
`collection_failure_capture_pending` stays true. Cleanup restores Native
resources and removes hooks first, then retries the preserved group/request
metadata capture. A failed retry keeps the Runtime reservation. A successful
retry permits the ordinary unregister path; it does not grant C2/C3.

The next actual capture should be inspected in this order:

1. Join each `unobserved-module:<id>` candidate to
   `structural_noncall_inventory` and the original module alias table.
2. Compare both actual module-call inventories, including paths/object IDs.
3. For an uncalled logits-head owner, inspect `actual_parameter_consumption`:
   the existing exception requires the exact current head weight consumed by
   the exact held logits processor, an original return, and no claimed module
   forward. Parameter shape or a projection count is insufficient.
4. For a structural container, retain the existing exact ModuleList/source/
   unimplemented-forward and observed parent/children requirements. Every
   other unobserved executable module stays `NOT_OBSERVED`.

The normal view/geometry rejections and a projection lacking an owned weight
remain unchanged. No unseen module is reclassified as unsupported here.
New Python source/package hashes are required for the next run; Native ABI,
DSO, algorithms and queue parameters are unchanged. No SSH, container or NPU
was used to implement this fix, and no new production qualification is granted.

CPU contracts use the real Host/Runtime plus actual Torch projection calls
and an explicit CPU Native double. They cover complete failure originals,
same-worker cleanup and ordinary healthy use afterwards, detached query
snapshots, tampered-file refusal, and artifact failure/retry with Native
resources already clean and the Runtime owner retained.

Validation: 36 CPU tests passed across the four new failure-evidence
contracts, public group/custom-GEMM/probe routes, shared projection coverage
and collector-aware structural classification. No natural missing-module
cause is claimed from these CPU fixtures.
