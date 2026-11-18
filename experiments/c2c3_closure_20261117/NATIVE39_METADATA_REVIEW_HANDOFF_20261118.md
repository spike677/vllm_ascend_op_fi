# Native39 recorded SDK-array ownership review

This is a CPU-only consistency review of existing evidence. It loads no native
library, starts no Worker/container and makes no NPU calls. Original artifacts
are read-only; each sidecar is created exclusively. Neither a passing review nor
its cross-group link grants C2/C3/E5 or proves numerical effects.

## Actual C2 result

The completed source42/packages16 C2 evidence passed the new independent check:

- 113 actual module/site targets reuse five distinct native signature/address/
  epoch keys. Every required key has a live registration capture witness.
- Each witness shows exactly three tensor captures, nine getters, nine arrays
  acquired and nine released; both before and after snapshots have no active
  captures/getters, live arrays, failures, exceptions or ownership errors.
- All 689 recorded counter snapshots, including 678 per-call snapshots, retain
  the same loaded proxy identities and monotonically nondecreasing counters.
  Every available native snapshot has the required live cache/epoch/witness
  registry and selects the same key as its independent native delta.
- Every expected phase is present. The final Worker health has no owner, enabled
  injector, in-flight calls, leases, hooks or native error; complete OFF equals
  baseline.
- The group's observed getter/array delta is zero because these are cached
  registrations. The proof comes from the exact registration captures, not a
  nonzero lifetime total. The split proxy reports 81 acquired/81 released and
  the main proxy reports zero. Unrelated earlier cache captures are not relabeled
  as this group's five required keys.

The original attempt remains:

`/home/17-fault-c2c3-20261117/native39-cache-witness-c2c3-card2-a2/providers39-qwen-c2c3/c2`

New sidecar:
`/home/17-fault-c2c3-20261117/review-native39-metadata-20261118-a2/c2.json`

SHA256 `8592421c05d6fa7bc992963638604b1a38db5a17bfb3ded99825313a2899de98`.
Its reviewer SHA is
`6a9156aa0e48da8bb92e844334f2b8f49427c58f9f3c46a2f7d4796e837aaddf`.
The sidecar records exact source paths, sizes and SHA256 values. No large
provider-state or independent-trace artifact was printed or republished.

## Independent gates and limits

`review_native39_metadata_arrays.py` recomputes the actual per-phase target
universe from `native_delta`; one required key cannot borrow another key's
capture, reuse a stale epoch or hide an intermediate error behind a clean final
state. Capture intervals from one counter instance must not overlap. It checks
all phase and available per-call snapshots, stable loaded image mappings/hashes/
counter instances, actual selected keys and final live registry. Missing fields
or incomplete phases refuse the review.

Use `--previous-stage` for C3 to independently verify the recorded same-Worker,
same-loaded-counter and same-cache-epoch/witness continuity from completed C2.
Counters must not reset across groups. This is a recorded continuity check and
does not reconstruct or replace the runtime's original live C2 capability.

The scope is **SDK getter-array ownership only**. Device workspaces, native
temporary buffers, runtime qualifications, actual effect proofs and raw
statistics remain subject to their existing independent reviewers. This script
does not decode device ABI payloads or retrospectively authenticate a recorded
native observer. C3 was still running when the C2 check completed; no C3 result
is asserted here.

## Exact implementation binding repair

The old main review expected a three-component implementation digest, while the
deployed repaired adapter binds four components:

```text
digest([adapter_sha, observer_sha, build_profile_sha, metadata_arrays_source_sha])
```

`review_native39_group_124.py:metadata_scope_binding` now reads both actual
installed `fault_injection/metadata_arrays.py` and the run's frozen source copy,
requires equal bytes, and includes the actual implementation SHA as the fourth
component. The exact scope comparison remains mandatory. This does not replace
the separate SDK-array reviewer above; both must pass their respective scopes.

Read-only verification confirmed:

```text
/home/17-fault-c2c3-20261117/packages-a16/installed/fault_injection/metadata_arrays.py
/home/17-fault-c2c3-20261117/source-a42/fault_injection/metadata_arrays.py
SHA256 = 9bf06627e266a968edc8d32fd16d98f908e47010450f128e0d2847326fe5ff4c
```

The full root review must use this actual source42, not the current review
checkout. Its case manifest is
`/home/17-fault-c2c3-20261117/native39-cache-witness-c2c3-card2-a2/reviewed_cases.json`.

## CPU tests and completion command

30 CPU tests passed: 24 SDK-array contracts and six existing/updated recorded
group-review contracts. Negative cases include missing/dead/copied witnesses,
foreign loaded instances, wrong epochs, getter ownership errors, counter reset,
missing phases/per-call snapshots, cross-group lifetime changes, old three-part
digests and changed installed/frozen source bytes. No runtime libraries are
imported by these tests.

```text
python -B -m unittest test_native39_metadata_review test_native39_recorded_review -v
```

After the actual C3 health artifact exists, run on CPU with a fresh sidecar:

```text
python review_native39_metadata_arrays.py --stage ATTEMPT_CASE/c3 --previous-stage ATTEMPT_CASE/c2 --out NEW_PAIR_REVIEW.json
```

The main full coverage/statistics/build review remains root-owned and has not
been run by this subtask. No original failure, phase file or runtime receipt was
edited; earlier audit sidecars are also retained.
