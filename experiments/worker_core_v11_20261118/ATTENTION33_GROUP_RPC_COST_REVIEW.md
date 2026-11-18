# Fault33 group: public RPC cost remains a pre-F gate

Root and an independent reviewer found an additional cost-evidence gap in the
all-target group candidate. This does not invalidate source105's accepted
single-site result or any earlier named C2/C3 scope. It does not prove the new
case necessarily exceeds its budget. **Root will not schedule a group F until
this gap is closed.** Fresh zero-fault stock/private normal collection remains
useful and is a separate gate.

The five-phase normal driver invokes its collector through direct
`collective_rpc`. Its measured phase time and runtime-prefix indexing time do
not include the full public `WorkerFaultClient` receipt path used by the
60-request/189-RPC C2 protocol:

- `WorkerFaultClient._record` retains the reply in the history, and `_snapshot`
  makes a JSON copy of the complete history on each public action.
- Later replies contain cumulative evidence references. Repeated copying of
  the whole growing history can therefore add superlinear work.
- The group probe also writes new RPC originals, copies current provider state,
  rewrites the result, and finally writes the complete Host receipt.

The current resource plan forecasts normal phase and prefix-index costs, plus
recovery and `max(120 seconds, 15%)` margin. That margin is not a measured bound
on the additional public receipt path. Live elapsed-time checks account for
work already spent, but do not establish that future growing RPC receipts and
final persistence fit the remaining F/OFF/health reserve. An outer1800-second
container timeout is not a substitute for proving recovery admission.

The existing512 MiB runtime-log cap, per-request128 MiB capture limit,
1024-file/512 MiB other-artifact limits, live memory/disk checks, and the
1800-second case budget remain unchanged. No uncertain F is retried.

The next repair must either measure and enforce an explicit envelope for the
actual public path after its three real zero-fault controls, or introduce an
explicit incremental-receipt interface that preserves default Host behavior
and all189 original RPCs while avoiding repeated complete-history copies.
Either approach must include final persistence, OFF and health, and reject
growth outside the measured envelope. CPU cost calibration can support a
budget; it cannot serve as evidence of a model call or fault effect.

Separate CPU inspection also identified repeated linear identity searches in
the group's retained input records. An internal identity index may reduce
that work while preserving all strong references and cleanup semantics. It
does not by itself close the public RPC cost gap.

The production-image source112 CPU suite is allowed to finish unchanged. A
passing functional suite and a fresh SDK build will not automatically remove
this pre-F hold. No new device C2/C3 or E5 qualification is asserted here.
Publish implementation, contracts and summary documents only; raw statistics,
private arrays and compiled artifacts remain on124.
