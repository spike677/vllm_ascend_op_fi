# Native metadata activity: cache lifetime versus group observation window

## Actual failure retained

Read-only review of the root-owned 124 attempt found that native39 had already
captured its metadata before the group's first array snapshot. The result at
`/home/17-fault-c2c3-20261117/native20-39-requalification-card2-a1/providers39-qwen-c2c3/result.json`
has SHA256 `1c37541197f922dd363f50f8d31b0975cecb6b057fc446459ce3866cfe51099d` and
remains `FAILED_WITH_EVIDENCE`. Recovery reported
`WORKER_RECOVERY_FAILED:METADATA_ARRAYS_NO_ACTUAL_WORKLOAD_ARRAY_ACTIVITY`.
No historical result or qualification is changed by this review.

The initial snapshot and all nine request snapshots have identical counters:

| Actual loaded proxy | Instance | Captures | Getters | Acquired / released | Live / active / errors |
| --- | ---: | ---: | ---: | ---: | --- |
| `/native/libopapi.so` | 281462286811864 | 0 | 0 | 0 / 0 | all 0 |
| `/native/libopapi_nn.so` | 281458926846680 | 27 | 81 | 81 / 81 | all 0 |

The split counter revision is 378 and peak live count is 3. Actual split-image
SHA256 is `98c36e4575df8dd771b4dee3921c5030df1fca37c2b42c8e77be1f26ff4d70d4`;
main-image SHA256 is
`bb01c50c8b172251d7f71a1282d528b52cf67bb78f8d31ba39c2c00de8a1402d`.
The build manifest is
`ab14a84f71de35b6012f05690f1f72f83e6eeb46c7052cbf56b2dc87df972a2b`.

Stages were discovery_1, discovery_2, reference, lossless_self, lossless_mat2,
lossless_dual, fault, off_observed, off. They completed 113 target matches per
request; the fault/observed-OFF/disabled-OFF denominator was 339. The 113 sites
reuse five independently observed cache addresses/epochs (epochs 109, 121,
128, 129, 130). The native typed metadata count was 9. These observations
support cache reuse; they cannot retrospectively distinguish LLM constructor
profiling from the healthy request before the group. The new witness is not
backfilled into these older records.

Selected raw evidence identities under that attempt's `c2/native` directory:

| File | SHA256 |
| --- | --- |
| `00001-metadata-arrays-initial.json` | `cdc62ed10bdeaa6827de9bc9a2d78fa32ad628c555121a3a055c3e1c47fa8dfe` |
| `00680-request.json` (discovery_1) | `03c7d8526e9695ba54fb2508fb01ad9d550e0a3c8caa4ad5309615dcdb94b3e3` |
| `01359-request.json` (discovery_2) | `33c33801f2cd1d195877afd8cc06a5ccbb949815adde8317af1c6980a071371b` |
| `05207-request.json` (fault) | `3ac21a1ad77a0510cb6ee933391491cf5491bc141b73038a2eb10646322749d2` |
| `06565-request.json` (off) | `d4692271c40effa457398193af0352bd92f389326855cde4e8a66cba2bda8162` |

## Corrected evidence boundary

The native39 proxy reads its actual private counter immediately before and
after the three real self/mat2/out SDK metadata captures. Each RAII destructor
has run before the second read. The additive `fault39_gws_commit_arrays_v1`
passes that interval to the native cache registration. Both endpoints must be
quiescent with no errors or live arrays; the deltas must be exactly 3 captures,
9 getters, 9 acquisitions and 9 releases. The native registry stores this
witness with the actual cache address/epoch. Cache replacement invalidates
that registration as before. Old bundles without this capability refuse.

After two independent natural discovery requests, the group binds witnesses
for exactly the cache keys used by its actual targets. A witness must match
an independently loaded counter instance whose current monotonic counters
contain its after endpoint. Image identity still requires the actual manifest,
SHA256, mapped inode/device and dladdr symbol origin. Later reads reject
counter reset, instance or image changes, errors and outstanding arrays.
Missing target, stale cache, foreign instance, zero-activity or future witness
cannot pass. A lifetime total of 81/81 with no bound witness still fails.

`metadata_array_activity` preserves both the observed group delta and the full
bound registration intervals. It explicitly records
`process_lifetime_nonzero_alone_accepted=false`. Resource evidence does not
replace the independent numeric first-difference comparison or the existing
native buffers, workspaces, hooks, leases and same-Worker OFF controls.

## Fault20 and short preflight

Fault20's reviewed path differs: `CurrentGwsScope` captures the real descriptor
on each typed `aclnnQuantMatmulWeightNzGetWorkspaceSize`, and current-call Run
admission requires that current descriptor. `capture_qmm_metadata` executes
unconditionally on each successful typed GWS with registry hooks present.
This supports expecting positive getter activity inside discovery, but is not
an NPU result. The Python group now requires positive activity after the first
discovery, before spending time on lossless/F/OFF. It does not accept native39
cache witnesses in place of Fault20 current-call evidence.

Both existing case preparers and Worker probes accept `--metadata-preflight`.
It cannot be combined with `--statistics` or `--c3`. The root controller still
owns device checks and resource locks. A new standard Worker runs the unchanged
healthy request plus two natural discovery requests, then releases the native
owner/callback and all owned hooks, removes the binding, and compares a healthy
control in the same Worker. It never executes fault phases and keeps C2/C3
false. Result status is `METADATA_PREFLIGHT_EVIDENCE_READY_FOR_REVIEW`; it is not
full mechanism acceptance.

Root sequence for a fresh immutable source/package/build is:

1. Run the production-body allocating CPU fixture, then the real Linux ELF
   identity/alias contract. Build fresh native39 with the new witness export;
   keep all manifest and source fingerprints.
2. Prepare the normal 20/39 queue cases with `--metadata-preflight`, omitting
   `--statistics --c3`. Inspect target-cache witness completeness for 39 and
   positive discovery-time acquisition/release deltas for 20, zero live/error
   counts, stable same-instance identity, and same-Worker cleanup/control.
3. Only then prepare fresh normal `--statistics --c3` attempts. Full C2/C3
   remains subject to every existing coverage, restoration and raw-statistics
   gate. Do not reuse the older failed attempt as acceptance.

CPU regression coverage includes real-new[] fixture capture/release intervals,
fresh captures, pre-existing warm caches with no new group getters, inactive
fake counters, incomplete targets, errors/live arrays, foreign instances,
counter resets, early discovery cleanup and incompatible preflight flags.
CPU success alone does not authorize an NPU qualification claim.
