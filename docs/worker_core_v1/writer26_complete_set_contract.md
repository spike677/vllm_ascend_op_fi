# Native26 public Core complete-set C2

The trusted deployment may select `stage='c2_once'` in
`install_writer26_core_binding`. The user still supplies only
`FaultConfig(26, 'residual_stomp')` or `FaultConfig(26, 'fia_destination')`.
The binding consumes the existing Worker, fixed prompt, current Native-v2
bundle and evidence directory. It creates no Worker and runs no forward.
The default single-site C1 path, Native ABI and Native build are retained.
This source needs new actual Worker evidence; CPU contracts confer no C2/C3,
E5, yellow-framework, CANN 9.x or A3 acceptance.

`CoreWriter26Coverage` composes the existing guarded Core experiment with
`Writer26Coverage`, `QualifiedTargetSet`, `CoverageSession` and
`NativeWriter26Group`. It does not create a second ownership lifecycle.
The existing public protocol remains prepare/register/check/enable, ordinary
enqueue, arm, actual serving execution, read-only query, explicit collect,
finish/disable/unregister, ordinary health and detach. No RPC calls the model.

## Actual request and eligibility scope

Eight fresh-prefill requests are required:

1. Two complete healthy Native inventories with actual module identities and
   every registered alias retained.
2. Reference, Lossless and OFF for **every** stable resolved target, using
   three normal requests. Every target must qualify before the first F.
3. One request applying each qualified target once, a complete group OFF,
   then same-Worker full-vocabulary health after Native release.

This is **same-request cumulative multi-target execution**. It is not
independent per-target replay, decode coverage or C3 every-match. All original
producer/writer/consumer calls execute once. Every actual Native call remains
in the independent denominator, including unsupported calls and failures.
The existing model objects, current Native formals, public storage owners,
allocator adjacency, original call, exact next consumer and full restoration
receipts are rechecked on every admission. Current aliases cannot change.

`max_targets` is an explicit whole-set resource ceiling, never a selector.
Unverified candidates, two-healthy eligibility flips, overlapping lifetimes,
failed qualification or budget overflow block the whole group. The probe
preserves the complete census, executes its Native-release health and a
post-unregister ordinary health, and reports `...C2_QUALIFICATION_BLOCKED`.
It does not automatically choose another target set or retry F.

The prior actual QMM inventory has 135 calls: two stable resolved candidates,
106 proven nonadjacent calls, three initial unresolved producer lifetimes and
24 allocator eligibility flips. All 27 unresolved rows must be resolved by
evidence or continue blocking C2. The 24 flips have equal actual target keys
and equal stable Native formals; they cannot be repaired by removing addresses
from a key or silently labelling them unsupported. This implementation keeps
that refusal. These are workload observations, not hardcoded counts.

## Current-forward producer absence

An existing Native thread-local producer can outlive a fresh root request.
The new C2 classifier may prove a current-forward non-target only when:

- The complete independent trace has contiguous events, no overflow/errors,
  and all events belong to the actual callback thread.
- The callback is bound to this actual root request, the original continuation
  ran exactly once without admission, and the actual writer completed once.
- No producer GWS/enter/exit precedes this writer in the current request.
- The next real producer completed successfully and advances the inherited
  producer sequence by exactly one.
- The unresolved reasons are exclusively the absent current public
  producer/consumer lifetime, not a descriptor, execution or other failure.

The full current trace, native before/after states, inherited producer,
actual next producer and original unresolved reasons are preserved under
`current_forward_residual_absence`. Absence of an event alone is insufficient.
The actual FIA continuation and public typed return gates remain mandatory.
The first call stays in all 30 observed calls. A valid initial exclusion lets
the 29 stable FIA chains undergo their own fresh all-target qualification;
neither count is assumed by the code. The old C1 classifier is unchanged.

## Native budget and recovery

The independent trace and typed trace each keep their existing 32768-record
capacity. The plan estimates the full remaining request cost, not only
selected targets. Typed FIA additionally budgets one owned-execute record
per target per Reference/Lossless/OFF/F/OFF. From the old actual inventories:

| Mechanism | Remaining C2 Native records | Available after two healthy requests | Typed records |
| --- | ---: | ---: | ---: |
| QMM | 432 × 6 = 2592 | 31213 | not applicable |
| FIA | 300 × 6 = 1800 | 31688 | 60 × 6 + 29 × 5 = 505 of 32588 available |

These are planning estimates from prior observations; the new actual trace
must independently remain complete. No capacity is increased. Native device
and host limits stay 64 MiB and 512 MiB; the typed stock bridge keeps its
512 MiB limit. `QualifiedTargetSet` also checks the whole target resource sum.
FIA preserves two complete stock byte samples, original NCL format 47,
flat-storage typed descriptors, new Reference equality and every actual
public output/consumer receipt. No old Reference is reused.

Coverage recovery stops all targets and drains all targets before release.
Actual all-target OFF, Native removal, complete health and owned-hook removal
must precede the original in-process C2 proof. Logical group reservations are
reported separately during an active plan; they are not pending Native writes.
During failed cleanup they also prevent Runtime from declaring the provider
inert. Native-zero alone cannot release an incomplete group owner. Missing,
foreign or changed owner/generation, late callbacks and uncertain cleanup
retain ownership. An explicit retry may complete already-observed OFF/health
recovery without rerunning F; first cleanup errors and recovery readbacks stay
in `cleanup_failures`. Recovery cannot invent missing OFF/health receipts.

The existing `CoverageSession` records the original issued object's owner,
generation, scope, target IDs, process/object binding and hashes. JSON is
read-only audit evidence and cannot restore live proof authority. The C2
proof is issued once after finalization. C3 consumption remains a separate
future stage in this commit. Query neither collects nor saves; RPC carries
compact coverage hashes/counts and the complete current Native readback.
Immutable private observation files retain the full calls, candidate rows,
aliases, typed descriptors, output evidence, recovery and proof audit ledger.
The shared Host history semantics are unchanged.

## Root-owned preparation

Using a new frozen source snapshot, matching installed distributions, and
the existing reviewed Native group bundle, root may prepare one FIA case:

```bash
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py" \
  --installed "$INSTALLED" --native "$NATIVE" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py" \
  --source "$SOURCE" --source-commit "$COMMIT" --worker-core \
  --stages c2_once --mechanisms 26.fia_destination --max-targets 128 \
  --card 2 --out "$CASES"
```

Changing the mechanism to `26.residual_stomp` preserves the QMM full-set
refusal; it does not authorize an eligible-only census. Both probes save
every original RPC, eight request observations when applicable, final host
receipt and stdout/stderr on 124. The ordinary post-unregister request is a
ninth request. A successful actual C2 result still needs independent review.
`--statistics` explicitly refuses before model startup; E5 remains separate.

Local validation: 136 related CPU contracts passed, followed by 20 final
focused contracts after the last census-hash and case-generator assertions.
Those runs overlap; they are not 156 distinct tests. The real Host/Runtime
tests cover typed flat storage, all 30 calls/29 targets, current-forward
absence proof and missing/cross-thread evidence refusal, actual adjacency
flips, whole-set capacity refusal, foreign owner/generation, late callbacks,
read-only query and failed final health readback with an explicit cleanup
retry. The CPU proof remains `evidence_kind='cpu_contract'` and grants no
actual Worker C2.

## Loaded-environment refusal evidence

The source71 FIA attempt reached the five healthy/qualification requests, then
failed before F because its actual environment could not be fingerprinted. It
reported both duplicate `libcust_opapi.so` mappings and CANN/Torch-NPU mapped
identities that differed from the files opened at those paths. This is not a
missing SHA format exception to bypass, and the old attempt remains failed.

The observer now retains each executable mapping's device/inode/address/offset,
the file identities before and after hashing and from a fresh pathname stat,
and the exact comparison that failed. On an identity mismatch it additionally
records `/proc/self/map_files` readlink/fstat metadata or the access errno. It
does not hash that path as a substitute or grant authority from this diagnostic.
Every duplicate basename retains its underlying file-verification status and
all same-name mappings. No row is discarded and the default fingerprint still
rejects any unverified or ambiguous image. Resolving such a deployment requires
new actual mapping and bound-symbol evidence; installed-version labels or a
pathname hash are insufficient.

The writer group records the immutable `actual-environment.json` and
`environment-refusal.json` before raising `LOADED_WORKER_ENVIRONMENT_UNPROVEN`.
The public read-only report references that original SHA and exposes the missing
and unverified rows. It constructs no coverage group or live proof and requests
no F. An owner already registered is retained until explicit cleanup succeeds.
The four required libraries, all-extra-image check and `CoverageScope` validation
are unchanged. This adds diagnostic CPU contracts, not actual C2 acceptance.
