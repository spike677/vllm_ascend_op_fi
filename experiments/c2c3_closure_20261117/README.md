# C2/C3 closure — 2026-11-17

Frozen source: `91decf5efe48dfdf8c5494e52d258ccde788cc5d` (implementation
`cb5b0dc120ce6c80f1469bb8b7248da27a7bb258`). This is an execution workstream;
the previous engineering reports and all original 124 evidence remain intact.

The initial qualification count is **C2 0/17, C3 0/17**. Existing scoped C1
receipts remain evidence only for their original source, build and workload.
An implementation or CPU contract test does not change either count.

C2 requires a complete independent observer denominator, lifetime-bound target
identities, all eligible targets hit once, no non-target interventions, per-target
effects, group drain/resource release and same-process healthy OFF recovery.
C3 requires that mechanism's matching C2 proof before a bounded every-match
session. New calls caused by propagation are recorded separately.

The root controller owns integration and all NPU scheduling. Seven independent
specialty worktrees use the same frozen base; at most three specialty agents run
concurrently due to the execution environment's limit. See `ownership.json`.

New raw statistics, private tensors and Native build products stay on 124 under
`/home/17-fault-c2c3-20261117`. Only reviewed source and summary/hash reports may
be published. E5 reads raw B/F statistics/mix, stack/construct, configuration and
collector identity only, with baseline reference, `rtol=1e-5`, `atol=1e-8`.
Null, NaN and alignment failures remain explicit; runtime hit logs cannot select
the first numerical difference.

`run_queue_124.py` accepts up to eight reviewed cases, obtains the shared
`/var/tmp/fault22_npu_2_5.lock`, checks device health, processes, open device handles
and dedicated reservations before each case, and cleans up only its own container.
The lock coordinates cooperating jobs; it is not host isolation. No expired
Docker entry-point isolation is reinstated. A finished container is not proof of
same-process recovery; that remains a separate mechanism receipt.
