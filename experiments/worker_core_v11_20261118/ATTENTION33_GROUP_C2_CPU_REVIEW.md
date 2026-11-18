# Fault33 group C2: corrected CPU protocol, no device qualification

The separate group candidate now connects all28 actual decode Attention
objects to the existing public Worker protocol. The60-request CPU fixture
exercises3360 original CPU API calls,1680 simulated Native receipts,28
isolated numerical effects,3136 coverage rows and189 original Host RPCs.
Its independent reader reconstructs the ledger and live CPU proof, and checks
finish, disable, unregister, detach and a subsequent ordinary56-call request.
Simulation is explicitly marked and cannot grant device C2.

Root's first integrated15-contract run passed in411.167 seconds. Independent
review nevertheless reproduced two uncovered defects. Both were repaired
before queuing any new device experiment:

1. Group collection inherited a path that omitted the normal collector's
   capacity checks. It now rejects captures over128 MiB before opening them,
   limits other request artifacts to1024 files/512 MiB, and checks directory,
   path and open-handle identity before and after bounded reading. A second
   inventory and equality with the current request precede completion. Failed
   originals and owner remain available for recovery.
2. Early disable after the three controls removed observers before detecting
   the incomplete protocol. Readiness is now checked before teardown. The
   rejected early disable retains hooks/input bindings for real same-Worker
   OFF and health; only then can unregister/detach release them. The first
   error remains, and this failed experiment never receives C2.

Root reran18 focused contracts, including those two actual Host/Runtime
recovery sequences and export closure:17 passed, with one Windows symlink
permission skip. The independent reviewer ran17 focused contracts:16 passed
with the same platform skip. The implementation agent then reran the complete
corrected60-request/189-RPC protocol once: PASS in308.331 seconds, zero skips.
These overlapping counts are not added together. Those Windows runs did not
cover the Linux symlink-negative gate.

Frozen source112 (`a243aeb04b2462488905aa47b1a3673941afa13f`) subsequently
completed the selected129 contracts in the real aarch64 production image:
129 passed, zero skipped,1369.648 seconds. This includes the corrected complete
60-request/189-RPC protocol, Linux symlink rejection, artifact limits, recovery,
normal-reader, resource and export contracts. The container had no NPU devices.
Its source-manifest SHA is
`cd135f6a27d735b9705cabcfd9784f7868c50c91f97d9c4eb3bfcf97685ee5eb`;
the complete CPU log SHA is
`ba83c35ae6d963fce97038350fee1b17ce0fd51adc1a2eec6ca8b0b0a834a096`.
This closes the Linux CPU gate, not any device qualification.

The earlier resource fix also remains: actual log growth increases reserved
parse memory; resource checks run before every F request and again before
Native admission. The default128 MiB log limit is unchanged. Only the exact
group scope may use512 MiB, and the case budget remains1800 seconds. CPU
success does not prove the real60-request experiment fits these bounds.

Source105 single-site acceptance and source108's externally interrupted
normal pair retain their original scope. This candidate must build its own
final frozen package, complete a fresh stock/private normal pair, independently
reconcile435 full payload pairs and140 target pairs, and pass measured resource
gates before any F. C3 and E5 remain separate and unqualified. The strict
overall totals remain C2 9/17 and C3 8/17 in their already accepted scopes.

Further independent review found an additional public-RPC cost gap. Actual
normal-phase measurements do not bound repeated complete-history copying and
final persistence by the public client. All group F remains held until a
measured/enforced repair covers that path and its recovery reserve; see the
[cost review](ATTENTION33_GROUP_RPC_COST_REVIEW.md). The129-test pass does not
override this separate admission requirement.

See [preparation and limits](../../docs/worker_core_v1/attention33_group_c2_preparation.md).
Publication contains source, tests and summaries only, without new raw
statistics, private tensors or compiled binaries.
