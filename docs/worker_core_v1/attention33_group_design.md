# 33: proposed public all-target C2/C3 interface

This is a CPU design and a test of the existing coverage coordinator. It is
not a new Provider, Native package, launch permission or qualification. The
source105 first-layer C1 remains confined to its original scope. The new
`tnd_page16_arch22_all_normal` package remains normal-only. No edit to that
package may silently enable F. The source108 normal pair is still a root-run
experiment at the time of this design; its outcome is not assumed here.

## Existing boundaries and the smallest separation

| Existing file / interface | Restriction that must remain | Proposed separate group interface |
| --- | --- | --- |
| `paged33_s2_evidence.VerifiedNormalEvidence.match` | First-layer complete input signature, allowing only proved page relocation | `paged33_group_evidence.VerifiedNormalCatalog.bind_live(...)`, with one independently replayed normal entry per actual discovered object |
| `paged33_arch22_s2_effects.Arch22S2EffectsNativeCall.execute` | First layer, five phases, one original call, healthy input hash | `paged33_group_native.GroupS2Call.execute(..., binding, admission)`; exact target, occurrence, current input, owner and plan token |
| `paged33_s2_execution.Paged33S2ExecutionAdapter` | One held first-layer binding and five phase filenames | `paged33_group_execution.Paged33GroupExecutionAdapter`; full census plus a bounded ledger indexed by actual request and target |
| `providers/paged33_s2.Paged33S2ServingProvider` | Existing single-site RPC and report shape | `providers/paged33_group.Paged33GroupServingProvider`; same public actions, explicit separate trusted driver type |
| `coverage_session.CoverageSession` | Issued-object C2 gate, all-target reservation, original-once outcomes, common recovery | Reuse without changing Host, Runtime or the coverage coordinator |

The group execution adapter should reuse `AttentionExecutionAdapter`,
`NormalTargetScope`, `LogicalPagedInput`, the full-payload capture and the new
per-target normal reader. Extract small shared functions if needed; do not copy
the whole all-normal session or replace `execute_model`.

The future Native variant must have its own vendor, source/build receipt and
host schema, for example `tnd_page16_arch22_group_s2_256`. Its only numeric
policy remains the already defined S2=512 to S2=256 selection before the
original loop/SplitCore/workspace/serializer. Kernel arithmetic is unchanged.
Its host must bind the actual target-set digest, target ordinal/object ID,
request-phase token, owner/generation, PID/native TID and current observer
scope. A current-input admission digest and its control scope are echoed into
the host receipt. No arbitrary user callback or tensor is accepted by RPC.

Source108 all-normal bytes cannot qualify this different effect package.
Before F, that exact final package needs its own normal mode/raw comparison;
the existing two-arm preparer and reader can be parameterized for its explicit
new schema. A source105 certificate or a saved `normal_payloads_exact` flag is
never a substitute. Replaying original files creates a non-serializable
catalog of observations, not permission to execute.

## Stable targets, current calls and live admission

Discover all Attention/impl objects from the loaded model and both actual
healthy requests. The universe contains all 56 actual calls per request:
28 prefill calls outside the declared Q1/page16 mechanism, plus 28 matching
decode targets. Missing sources, changed objects, unobserved calls or missing
normal entries remain unresolved; they are not reclassified as unsupported.

Use a stable `BoundaryTargetIdentity` per object and call role. Its source/API
identity and geometry are stable; the current Q/KV hashes are per-call evidence,
not the target ID. The current `AttentionExecutionAdapter.target_key` includes
input hashes and therefore cannot be the C2/C3 group identity. Keep actual
cache storage/object lifetime in the live binding and provider proof. A new
cache or same-value clone cannot take over an old gate by matching an address,
shape or digest. Each `ObservedCall` carries its own real request/forward ID.

Proposed trusted interfaces (no implementations are enabled by this document):

```python
catalog = replay_all_normal_originals(frozen_inputs)  # read original bytes
scope = DiscoveredGroupScope(worker, actual_candidates, public_ticket)
controls = LiveControlLedger(scope, catalog, effect_build_fingerprint)

controls.observe_normal(binding, native_receipt, actual_request_result)
controls.seal_healthy_controls(reference, lossless, qualification_off)

ticket = observer.observe(actual_call, eligible=classified_eligibility)
decision = session.begin(ticket)
admission = controls.admit_current(
    binding, decision, request_plan, completed_prefix_receipts)
output, native_receipt = driver.execute(
    actual, boundary, invoke_original_once, binding=binding, admission=admission)
```

`LiveAdmission` must retain the exact live binding, owner/generation, current
request, target/occurrence, target-set digest, control kind, complete input
signature and selected S2. It is minted only inside the held Provider and is
consumed once before entering Native. The driver rechecks the live tensors,
versions/storage/offsets, page map, real writer, request mapping, thread and
Native state immediately before launch. Matching a serialized report or a
caller-provided object ID never mints it.

Healthy controls consist of three real R/L/qualification-OFF requests in the
current Worker. Each of the 28 objects must match its own replayed catalog
entry and produce S2=512 receipts, exact immediate output and all 87 complete
model payloads. Only proved page relocation is normalized. Completed control
tensors may be released after confirmed drains; the sealed live control
object retains the Worker/module/cache lifetime and the immutable receipt
identity. Releasing a tensor does not allow reconstructing a gate from JSON.

## Isolated C2 can use the existing coordinator

Create a `QualifiedTargetSet(selection='all_eligible', trigger='once',
concurrency=1)` and one `CoverageSession`. All 28 leases use one owner and one
coverage generation. `TargetLosslessProof` comes from the current controls,
with the actual before/after input hashes and whole-model reference/OFF hash.
No target has an inherited source105 lossless proof.

Run one ordinary natural request per selected target, followed by an ordinary
global OFF request before selecting the next target. The first not-yet-attempted
target is selected by actual call order. Keep its admitted `DispatchDecision`
active until the whole request returns and complete outputs are collected.
Then subsequent decode calls in that request remain eligible but pass through
with `CONCURRENCY_BUDGET_EXHAUSTED`; earlier targets pass through with
`TARGET_ALREADY_ATTEMPTED`. They still get their own observations/outcomes.
Finishing the decision immediately at API return would admit later targets
in the same request and would defeat isolation.

Only admitted calls require the healthy input signature. Later unselected
calls may receive propagated values: their actual bindings and outputs are
recorded, without pretending that their inputs are healthy or changing their
eligibility. The next target's F request starts from a confirmed global OFF
and must again match its own healthy control.

The bounded successful plan is 3 initial controls, 28 F/OFF pairs, and one
post-clean full-output health request: 60 natural protocol requests. The C2
coverage observer contains the 28 F and 28 OFF requests, 3,136 actual calls,
and exactly 28 target HITs. The controls and final health remain separate raw
artifacts, not omitted history. A HIT needs the selected S2=256 final POD and
an immediate attention numeric effect against its current S2=512 control.
Logit propagation is recorded separately and is not an extra HIT requirement.
A strategy selection with zero numeric effect is retained as such and cannot
be fabricated into a numerical HIT.

Use `GroupRecoveryHooks` for stop, real stream drain, per-target OFF and
whole-Worker health. No counter reset is a drain. Remove owned hooks before
the health proof is accepted. Only after `recovery.complete`, no pending
inputs/Native scope, and full output recovery may the Provider call
`c2_session.c2_proof()` and hold the returned original object. Failure retains
the session, target reservations, ticket and first error. Retry cleanup/OFF;
never repeat a target's uncertain F.

## C3 needs controls for propagated inputs, not a weaker hash check

The unchanged healthy matcher will correctly reject target 1 or later if an
upstream F changes its Q or current K/V. Neither changing the target hash nor
ignoring Q/KV hashes solves that admission problem. Same shape, same module,
an upstream anomaly flag or a saved C2 report is insufficient.

The least invasive candidate that preserves one original API call is a
**prefix-control ladder**, before the final two C3 requests. For target `i`,
the plan fixes actual targets `[0, i)` to F and target `i` to normal S2=512.
Those prefix targets may execute only using previously established live
controls for their respective prefixes. The requested root input, real
execution order and every prefix Native receipt must match exactly. Target 0
uses its unchanged healthy control; target 1 is calibrated after prefix 0;
and so forth. A target obtains its own propagated-input gate only after its
three returned prefix R/L/target-OFF controls agree, followed by a global OFF
that restores all 87 healthy payloads. "Target-OFF" here is not global OFF:
the earlier targets still receive the explicitly recorded prefix F.

All calibration F calls must be authorized by the held C2-complete Provider,
the original retained C2 object, the same Worker/build/target set, and the
already established prefix gates. This is a separate bounded control stage,
not a failed or shortened `every_match` session. No C3 counts or proof are
issued for it. Its complete 56-call requests and every F are published. The
conservative plan adds 27 x (3 controls + global OFF) = 108 real requests;
it is not hidden inside the later 56 C3 HITs. If the runtime budget cannot
support that plan, stop before F rather than skip controls.

For each final C3 call, require its current normalized signature to equal
that target's live prefix signature; require the actual completed prefix
target order, S2 receipts, geometry, output/effect signatures and request
mapping to equal the recorded prefix (tokens and physical page IDs are fresh
and independently joined). The healthy gate is retained unchanged for C2 and
global OFF. Unknown or nondeterministic changed input remains a refusal.

This ladder proves current-input **private S2=512 normal controls**, not fresh
stock/private equivalence for each propagated tensor. If the accepted normal
contract requires a stock/private pair for every changed tensor, C3 still has
a specific extra blocker: the current loader binds stock Norm and private FIA,
but does not independently bind a stock FIA GetWorkspace/execute route.
No current adapter can perform that comparison while preserving one public
original call. A separate reviewed Native reference path would be required,
with actual stock/private API ownership, owned shadow outputs, unchanged
input lifetimes and explicit launch roles. Do not invent it from the SONAME,
schema, key, source108 healthy samples or the private ABI. The ladder is a
proposed C3 contract boundary to settle before implementation, not permission
to weaken the current healthy matcher.

After controls, create a fresh `CoverageSession(trigger='every_match')` with
the **original** `c2_live_proof`, identical `CoverageScope`, candidates and
target IDs, the same owner and a greater coverage generation. Runtime ticket
generation and coverage generation are distinct namespaces and both must be
recorded in every Native admission. No serialized C2Proof, copied dataclass,
changed source/build scope, different Worker or new target set is acceptable.

The final C3 phase is two consecutive fresh natural F requests: each observes
56 calls, all 28 decode targets HIT once, then a global OFF and full-output
post-clean health. Prefill calls remain non-target originals. Changed values
of a discovered target do not make it `propagation_added`; that flag is for
actually new calls and cannot shrink or rename the original denominator.
Only the original CoverageSession issuance/use ledger proves live inheritance.

## Changes owned by the future implementation

New files should contain the group catalog/control/admission objects,
execution adapter, group coordinator/Provider, new Native recipe/header,
dedicated bounded public probe/preparer and independent raw reader. Keep the
source105 files as the existing single-site route. Reuse the all-normal
capture/reader helpers with explicit new variants and source pins.

Root-only shared changes are limited to the exact resolver mapping for the
new trusted driver type and exporter closure. `host.py`, `runtime.py` and
`coverage_session.py` need no planned behavioral changes. `advance` still
accepts only an enqueued request ID or explicit collection; `query` does no
collection, plan transition, proof issuance or resource mutation. Every new
health/post-detach claim must state whether full outputs or tokens/text were
actually observed.

`tests/test_paged33_group_design.py` checks only reuse of the real CPU coverage
coordinator: full denominator, one target per request, unresolved targets,
failed drain reservations, exact live C2 object and two every-match rounds.
It supplies synthetic CPU outcomes and grants no Worker/Native qualification.
Current-input/prefix, actual native geometry, uncertain F/OFF, foreign hook,
partial-start and final-output negative tests remain required with the future
implementation. This document does not schedule a new device experiment.
