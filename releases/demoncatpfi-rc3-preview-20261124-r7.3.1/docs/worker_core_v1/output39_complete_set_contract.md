# Public 39.output_proxy complete-set C2/C3

The trusted existing-Worker binding accepts `stage='c2_once'` or `stage='c2_c3'`.
The default `c1` path is retained. Users still supply only
`WorkerFaultClient(HostAdapter(...), FaultConfig(39, 'output_proxy'))`.
The runtime factory now selects the group adapter only for that explicit
reviewed deployment. There is no model-local qualification inheritance, Worker
creation, runner replacement or RPC forward. This source requires a new actual
Worker run and independent review; CPU tests grant no C2/C3 or E5 acceptance.

## Existing algorithm, actual execution

`ProjectionCoverage` now exposes begin/returned/failure/end callbacks and
prepare/open-session operations. Its old synchronous `run` remains a
MODEL_LOCAL wrapper around those same operations. The new
`WorkerProjectionCoverage` supplies actual Worker identities, actual enqueued
request IDs and recovery health. `Serving39OutputSession` and
`PostwriteObservedForward` still own the real root/logits/dispatch observation.
All original operator implementations execute once. No model call is made by
prepare, advance, collect, query, finish, recover or unregister.

The public plan first repeats the existing six-request C1 protocol on this
same Worker. It then executes:

| Stage | Real requests | Meaning |
| --- | --- | --- |
| C2 | all-target lossless, N fault requests, OFF, health | Each stable eligible target once in an independent target replay; one live Native lease per fault request |
| C3, when requested | new all-target lossless, two fault requests, OFF, health | Every eligible call on both requests; multiple disjoint leases may remain live until their real consumers finish |

With N eligible targets this is N+9 requests for C2 only, or N+14 for C2+C3,
plus one ordinary post-unregister health request. N comes from complete
actual healthy inventories; the prior 28 actual eligible pairs are not a
constant in the implementation. The independent projection ordinal count,
all operator counts, actual module calls, every alias and structural noncalls
remain in original evidence. Unknown executable modules or changed order,
geometry, consumer classification, current methods or parameters refuse the
complete group. No first-N selection or fallback is permitted.

The actual logits call passes an lm_head object to LogitsProcessor and may
consume its weight without invoking lm_head.forward. This specific noncall
classification requires the exact head parameter object to appear as an
operand of a successfully returned real operator inside the actual processor,
on both healthy requests. The module must have no children. Its parameter
geometry, operator/schema, calling module, real logits binding and zero
module-forward count are recorded. Every group request must reproduce that
parameter consumption. Other unused executable modules stay unresolved.

## Current Native and consumer gates

All existing custom GEMM source/schema/current PrivateUse1 registration,
owned weight/no-bias, finite-original, original storage span, exact readback,
selected-element consumer and restoration gates remain mandatory. The group
retains metadata of the two healthy reviewed RMS/SwiGLU consumer registrations
after releasing their tensor observations, and revalidates those registrations
before opening any new lease. Actual consumer geometry and successful original
return must still be observed later. Diagnostic copy/hash operations do not
become model consumers or E5 statistics.

Native26 and the Native postwrite ABI/build are unchanged. The existing
postwrite transport lease capacity and 160 MiB maximum span capacity bound
the plan. The public target budget is at most 256 complete targets, time is
1200 seconds per group, and C3 requires capacity for the entire eligible set.
The live-lease limit is not a target selector. Propagated nonfinite input at a
later site can refuse actual C3: no finite gate is relaxed to obtain coverage.
Such a refusal is evidence of an incomplete attempt, not C3 acceptance.

Every normal selected Native lease retains the original read-only audit
ledger (open/apply/restore/release/live). The final live operation is the
existing recovery query. No added Native operation or query produces these
receipts. Current ordinary query does not mutate the ledger. Failed release
or restore retains its original receipt and owner; after-release failure
retry does not write to a handle whose absence was actually confirmed.

## Proof, recovery and originals

The Runtime provider owner stays constant. Its ticket generation, the
CoverageSession generation and actual per-request Native generations are
recorded separately. C2 issuance occurs only after actual OFF and healthy
return, original Native cleanup and owned-hook removal. C3 uses that exact
issued object with the same Worker/process/model/scope/targets and a greater
group generation. Copied/deserialized proof is rejected before C3 F. The
existing issued/consumed audit ledger preserves object binding and hashes;
serialized evidence cannot recreate proof authority.

`query` is pure readback. After each actual returned request,
`collect_replay()` explicitly writes an immutable full original. A new arm
requires collection. Group transitions and proof issuance happen on this
explicit action after the true logits return. A failed collection retains
owner and can explicitly retry the same completed health without another F.
Every raw RPC is saved once; the common Host history semantics are unchanged.
RPC summaries contain complete request indices, counts, original hashes and
the current Native readback. Private originals contain complete discoveries,
calls, Native lease receipts, per-target executions, health and live proof.

On failed execution, explicit recover closes the real observer and recovers
the same retained Native leases, then arms actual OFF and health requests.
Failed or missing OFF/health cannot release the logical group reservation or
Runtime owner. The failed attempt remains failed even after successful cleanup.
No retry silently runs F again. Foreign tickets/generations, foreign hooks,
uncertain Native state and unresolved cleanup preserve ownership.

## Root preparation

```bash
python3 "$SOURCE/experiments/c2c3_postwrite_20261117/prepare39_worker_core.py" \
  --source "$SOURCE" --native "$NATIVE" --sha256 "$NATIVE_SHA" \
  --build-fingerprint "$BUILD_SHA" --stage c2_c3 --max-live-leases 128 \
  --card 2 --out "$CASES"
```

Use `--stage c2_once` for the minimum C2 attempt. The existing verified
postwrite Native bundle is reused. The new frozen source/entry fingerprints,
actual installed sources, real registration and current geometry must be
retained and independently reviewed. Old source65 NO_ELIGIBLE and old
MODEL_LOCAL evidence are unchanged. Raw tensors/statistics/Native artifacts
stay on 124. `--statistics` still refuses before model startup; E5, yellow
integration, CANN 9.x and A3 remain separate unaccepted scopes.

Local validation: 91 CPU contracts across the public group, custom GEMM,
actual-enqueue driver, original C1, model-local coverage and Native recovery
passed. They include original-object/copy proof, strict finite refusal,
foreign tickets, unknown modules, no synchronous replay, pure query, real
custom consumer registration and per-lease audit joins, before-restore and
after-release failure recovery. An earlier invocation included a nonexistent
test module name (one loader error); the final run uses only existing modules.
Local Torch is 2.10.0+cpu with explicit fixture registration pins, not actual
Torch 2.9/PrivateUse1/NPU acceptance.

## Actual Worker process lifetime

Worker group target identities use the public Core identity's complete
`process_start_identity`, alongside its Worker/model lifetime and current PID.
For natural evidence, that identity must equal a fresh read of
`proc:<boot_id>:<pid>:<start_ticks>` through the existing Core identity helper.
Tick-only ModelLocal values, changed boot/process lifetimes and an unobserved
Linux lifetime are refused. `identity.py` is included in the group's source
fingerprint. The ModelLocal helper and the strict result auditor are unchanged.
CPU contracts may use the public Core's explicit local identity without gaining
natural Worker qualification. Earlier result originals are not rewritten or
promoted by this producer correction.
