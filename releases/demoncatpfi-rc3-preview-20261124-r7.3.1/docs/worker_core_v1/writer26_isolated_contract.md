# Explicit isolated C2 for Native26

`c2_isolated_once` is a new, explicit public Core stage. The user still supplies
only `FaultConfig(26, 'residual_stomp')` or `FaultConfig(26, 'fia_destination')`.
The trusted deployment chooses this reviewed execution profile, not a layer,
address, rank or target prefix. Existing C1, `c2_once` and `c2_c3` remain unchanged.

The source77 actual FIA result used **SAME_REQUEST_CUMULATIVE** execution:
29 C2 targets and 58 C3 matches had separate Native writer/consumer/recovery
receipts, but multiple faults occurred in the same request. Its original evidence
remains useful within that scope. It is not reclassified as isolated replay and
is not a qualification for this new source or stage.

## Plan and actual boundaries

Two complete healthy inventories freeze the full candidate denominator. Every
stable eligible target must pass the existing fresh Reference, lossless and OFF
qualification. Any unresolved candidate, allocator eligibility flip, alias or
formal change, empty/over-budget set, trace overflow or incomplete producer
proof blocks the entire group. `max_targets` never selects a prefix.

For N eligible targets, the plan then runs N pairs: one F in its own fresh
ordinary enqueue, followed by that target's OFF in a separate fresh enqueue.
The order is the complete actual census order. A missing or changed next target
fails; there is no alternate selection. Every other actual writer still runs
and is retained in every request's complete Native census.

The existing consumer performs Native restoration and release at the original
boundary. Only the admitted CoverageSession decision remains active until the
real root and logits complete. Its existing concurrency-one gate prevents a
second F in the same request. Later requests reuse that exact group and owner.
Each F also compares the already captured current residual digest with its
qualified healthy Reference before Native admission; it adds no tensor copy.
No RPC runs the model, no Native ABI changes, and no shared CoverageSession
semantics change. Each successful pair joins its F execution artifact, distinct
F/OFF request IDs, actual call IDs and target identity. Final recovery requires
every target's real OFF plus full-vocabulary same-Worker health.

Root/logits failure ends outstanding decisions using observed Native execution
records. Missing original execution evidence becomes FAILED. It never invents
OFF or health to release reservations. Native resources are restored only under
the existing live-storage/owner checks; logical owner retention remains visible
until real recovery can be confirmed. Late callbacks do not replay an original.
Foreign owner/generation and uncertain cleanup remain failures, with explicit
retry after repair.

## Evidence and bounded cost

The new stage has its own `demoncat.writer26.isolated_core_observations.v1`
format. Each explicit collect saves a complete request delta, its Native events,
candidates and outcomes, together with typed/public observations and the full
current Native state. Each original records the previous original's SHA256.
Successful cleanup additionally saves the complete merged report, full coverage
call ledger and original live C2 issuance receipt. Existing stock/full-byte and
allocator files are preserved. Tests reconstruct rows, outcomes and coverage
calls from the deltas and compare them with the final full report.

RPC projection contains indices, counts and original SHA256 references; every
RPC still contains its actual Native owner/pending/lease readback. Query only
reads and verifies saved originals. It neither collects nor saves nor advances.
The two mutable progress files explicitly identify themselves as indices, not
observation originals. This avoids repeatedly embedding all prior raw rows in
every Host history entry. Public Host history behavior is unchanged.

For FIA with 30 actual calls and 29 stable eligible targets:

| Resource | Plan | Existing available/limit |
| --- | ---: | ---: |
| Group requests | 2 B + 3 qualification + 58 F/OFF + 1 health = 64 | Budgeted before F |
| Ordinary post-unregister health | 1 | No Native owner |
| Native records after two B | 300 × 62 = 18,600 | 31,688 remaining in source77 |
| Typed records after two B | 60 × 62 + 29 × 5 = 3,865 | 32,588 remaining in source77 |
| Full group call denominator | 30 × 58 = 1,740 | 8,192 |
| Owned Native executions | 29 × (R/L/OFF/F/OFF) = 145 | Existing capacities unchanged |

Actual counters are recomputed for every new run; the source77 values are
planning inputs, not authority. The original request and session time limits
remain 1,800 and 1,200 seconds. Before the first F, the probe measures the full
ordinary-request plus arm/query/collect cycles for R/L/OFF. It forecasts
`2 × max(three measured cycles) × remaining requests + 120 seconds cleanup`,
plus a separate Python-control reserve. The full CPU driver measurement below
showed that early cycles alone underpredict growing evidence/Host history work.
The reserve is `ceil(360 × (group_requests/64)^3 × max(1, actual_call_count/30))`
seconds. It grows with the complete denominator, never only the selected count.
It also reserves 120 seconds outside the probe for controller/startup overhead.
Either limit exceeded means refusal before F and explicit resource cleanup.
This estimate is recorded and is not a guarantee or permission to raise limits.

Root's source77 budget metadata SHA256 is
`201dcec9b255dd4e78053c1050ce5e7eaf44c255906accd2c0f483482c50bd88`:
330.044 seconds, 17 group requests, 536,999,909 total evidence bytes. Its selected
17 cumulative request JSON originals alone total 120,139,758 bytes; its final
Host receipt is 77,202,912 bytes. Linear wall-time extrapolation is unsuitable
because that old format repeatedly serialized and hashed cumulative evidence.
The conservative disk reservation estimate is
`536,999,909 × 64/17 × 2 = 4,043,293,433 bytes` (about 3.77 GiB).
It is an estimate, not a measured bound on unclassified binary payloads.
All actual raw observations, tensors and Native binaries remain outside Git.

The full CPU driver measured 279.922 seconds and 957,530,550 evidence bytes for
64 group requests, one post-unregister health request and 204 separately saved
RPCs. It retained all 30 actual calls, 29 isolated target pairs and 1,740 group
call receipts, and detached with no retained owner. This is a CPU-double cost
measurement, not an NPU timing or qualification result. The 360-second control
reserve exceeds this measured total. With new actual qualification cycles no
slower than 6 seconds, the remaining estimate is `708 + 120 + 360 = 1,188`
seconds. With 180 seconds already elapsed and the outer 120-second margin, the
case estimate is 1,488 seconds. A slower new run must pass its own pre-F test;
these numbers do not promise the next hardware run will fit.

## Root-owned new attempt

Prepare a new frozen source and matching distributions, retain the reviewed
Native group ABI, and apply the already verified strict mapped-library mounts:

```bash
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" --native "$NATIVE" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py" \
  --source "$SOURCE" --source-commit "$COMMIT" --worker-core \
  --stages c2_isolated_once --mechanisms 26.fia_destination --max-targets 128 \
  --card 2 --out "$CASES"
```

No C3 is requested by this stage. A future C3 integration must consume the
original new live proof in the same Worker and exact scope; serialized audit
JSON cannot recreate it. CPU contracts grant no actual C1/C2/C3 or E5, and no
CANN 9.x or A3 qualification. E5 remains a separate statistics contract.

Final frozen local validation passed **94 CPU tests in 312.206 seconds** across
the retained Core C1/C2/C3/environment paths, writer experiment/coverage engines,
the isolated implementation and all corresponding public probes. This includes
the full 30-call/29-target/64-request case, real Torch root/logits failures,
missing calls, changed actual pre-F residual, foreign owner/generation, late
callbacks, original-file corruption, cleanup retry and pre-F budget refusal.
Earlier focused runs and the separate 279.922-second CPU cost measurement
overlap this coverage; their counts are not added to the 94.

## Explicit same-Worker isolated C2 to cumulative C3

`c2_isolated_c3` is a separate, FIA-only trusted deployment profile. It retains
the complete current candidate inventory, including the first fresh-forward
call without a live residual producer. It does not reuse source77/source82
qualification. `c2_isolated_once` still ends after its isolated C2; existing
`c2_once` and `c2_c3` keep their execution plans.

The new profile first completes the full isolated C2 above. Its explicit collect
then performs the existing whole-group recovery, same-Worker health and Native
release. Only that backend's newly issued **original Python C2 proof object**
is passed to the next backend. No RPC serializes or reconstructs authority.
The Worker, logical owner, process lifetime, source-bound scope and complete
target set remain the same; the Native generation increments once. The old
callbacks stay disabled. A failed recovery/install retains the original
failure, evidence and owner until explicit cleanup can establish zero resources.

C3 performs two new healthy inventories, complete Reference/lossless/OFF,
then two fresh requests in which **every matching eligible call** is admitted,
followed by real all-target OFF and health. C3 is cumulative within each request;
it is not described as isolated. The unchanged CoverageSession checks the exact
issued live object before fault sealing. New ambiguity, allocator change,
missing targets or copied proof blocks C3 without narrowing the set. Propagated
input or writer/consumer failures retain the unchanged Native safety gates.

For the source82-sized 30-call/29-eligible set the new plan has 64 isolated C2
requests plus 9 cumulative C3 requests, then one post-unregister healthy request.
Each generation retains all 30 calls per request. C2 has 29 F hits and C3 seeks
58 further hits; expected owned Native executions total `29*(5+6)=319`.
Before any C2 F, the actual trace capacity reserves the entire combined plan,
including both generations' typed records. C3 rechecks the remaining capacity.
Delta originals and final merged reports remain separate for both generations.
Neither collection nor query runs a model forward.

### Real cost evidence and clock boundaries

The new probe requires two fixed-SHA, read-only **cost originals**, kept on 124:

| Original | SHA256 |
| --- | --- |
| source82 result, all 64 complete cycle durations | `f84c1cc357ae5cce58b3b1d52d1e4a977e887de5acca883e8c0a9d08af654643` |
| source82 controller, actual start/end | `77779e19ac4adbaa9443393d81f37eefb26762df1ec2061fea348f4af2c31319` |

It validates both before model initialization. Missing/changed originals refuse
with a saved result; their success flags are never used. Actual source82 request
cycles total 736.386828 seconds, the largest late cycle is 25.441892 seconds,
and its controller interval is 921.375026 seconds. Faster early measurements
cannot erase those late costs. Each pre-F check applies the larger current
R/L/OFF-to-source82 comparable-cycle ratio (minimum one), complete-denominator
scaling and a 1.25 safety multiplier. C2 compares initial qualification cycles;
C3 compares its qualification cycles against source82's actual late maximum,
because C3 already carries the full prior C2 history. It does not multiply that
same accumulated history cost again by an early sub-second baseline.
Remaining C2 cost uses the actual 59 post-qualification
cycles; C3 uses the larger actual late-cycle cost, scaled for the full 73-request
history. The code records every input, calculation and refusal reason.

The existing 1200-second CoverageSession deadline begins at **each generation's
first fault group construction**. Neither generation may exceed that bound.
The 1800-second case remains one continuous clock; the probe starts its clock
before model initialization and never resets it at C3. Forecasts include 120
seconds for cleanup plus another 120 seconds outside the probe for controller
startup, exit and disk overhead. At the source82 measured speed and its actual
114.342905 seconds before F, the estimate is 1036.430113 seconds for the C2
generation and 285.502735 for C3; continuous projected case time is 1643.154172
seconds. These are estimates, not device acceptance or permission to extend
either deadline. A slower/late new run refuses before its first F. C3 repeats
the budget check after its own five healthy qualification requests, preserving
completed C2 evidence if no time remains for C3.

Prepare a fresh source/package with the unchanged reviewed Native bundle:

```bash
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" --native "$NATIVE" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py" \
  --source "$SOURCE" --source-commit "$COMMIT" --worker-core \
  --stages c2_isolated_c3 --mechanisms 26.fia_destination --max-targets 128 \
  --isolated-cost-result "$SOURCE82_RESULT" \
  --isolated-cost-controller "$SOURCE82_CONTROLLER" --card 2 --out "$CASES"
```

The preparer verifies and mounts those exact files at `/cost/source82-result.json`
and `/cost/source82-controller.json`, both read-only. It adds no SDK build or NPU
action. Root alone reviews and runs the case. This implementation and its CPU
doubles grant no new C1/C2/C3 or E5 qualification; actual new public-entry
evidence must be independently reviewed. Native ABI, allocator/liveness checks
and the fault's numerical mechanism are unchanged.

Local CPU validation covers 25 new contracts and 22 retained-entry regressions.
The full actual Host/Runtime fixture keeps 30 calls, 29 targets, 73 group
requests, 319 owned Native-double executions and the original live proof across
both generations, followed by ordinary post-unregister health. The public probe
fixture separately exercises its dynamic 19-request complete two-target plan,
read-only queries, explicit collection, per-generation SHA-linked originals,
cost refusal before either first F and failure preservation. Other negatives
cover copied proof, original proof under a changed implementation scope,
combined-capacity exhaustion, failed C2 health, stale callbacks, wrong mechanism,
missing/altered cost originals, incomplete/reordered cycles and continuous outer
deadline exhaustion. CPU fixtures do not establish device numerical effects.
