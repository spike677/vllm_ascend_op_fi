# Source107 WeightNz original-evidence review

The independent reader reconciles the one zero-F B8 first-decode observation
on physical card 5. This is **typed Native interface observation**, not new
row authority, fault acceptance, or C1/C2/C3/E5 qualification.

The reader does not import `quant_gmm2_native`, its assessment function,
Torch, ctypes, or any production Runtime. It ignores saved assessment/PASS
results and reconstructs the relationships from the nine original JSON files.
Its four actual review-source SHA256 values are recorded in
[`quant_gmm2_source107_independent_review.json`](quant_gmm2_source107_independent_review.json).
Those are post-execution review sources, not files claimed to have executed
inside source107.

## Independently recounted observations

| Original evidence | Recomputed result |
| --- | --- |
| WeightNz GWS/execute | 26 pairs, 52 calls, 104 entry/return events; each original returns once |
| Named slots | 390 non-null registrations: 14 TensorLists and one IntArray per GWS; no roles inferred from creator order |
| Five signed scalars | `[2, 0, 1, 0, 0]` at all 26 GWS calls; the last value is the actual `quantGroupSize`, not a public-schema assumption |
| Named parameter/return association | x, weight, scale, perTokenScale, groupList and out match their current public window arguments/return |
| Four-container lifetimes | 1,241 calls, 2,482 events, 586 repeated addresses with distinct current registrations |
| Typed metadata ownership | 187 creator-member captures plus 26 current groupList captures; 639 arrays acquired and 639 released |
| Natural source windows | 26 groups of routing/GMM1/GMM2/unpermute, 104 windows with current input objects and write epochs |
| Full projection census | 135 structural/public/Native projections, 135 prepare and 135 execute calls; 270 public calls |
| Existing row-proof scope | The same ten independently checked sites; 125 candidates remain unproven; no new row authority |
| Cleanup and health | Original events/scopes unchanged at cleanup, journals released, no pending executor, same Worker; before/captured/after token arrays match for eight requests and two tokens each |

groupList's Native descriptor is compared with the actual public `group_list`
by storage address, dtype, offset, shape, stride and storage size. The source
window is independently joined to dispatch inputs and the recorded write
epoch. This does not read the integer payload or establish routed row semantics.
The named output is separately joined to the actual public return.

The process/start identity, PID/TID, Worker identity, logical device, five
startup-resolved function owners, exact caller offsets, current scope,
executor registration, workspace and stream are cross-checked. Startup
resolution is source-bound to the pre-LLM stage. If that resolution fails,
the probe must refuse before LLM construction: it is not correct to claim
that every startup observer failure still forwards the original operator.

## Sources and immutable inputs

| Item | SHA256 or commit |
| --- | --- |
| Runtime source107 | `ac20747717ed8dad7e50d35932d5bd942ce6b421` |
| Native compile source106 | `922ff7cb1d343a3e16f28d43776552b00123428d` |
| Nine-original package | `88107fcbb0cf12cb5516f9db78abb091d3d0f38de8e9ba9529bd5367a309735f` |
| Result original | `4423e740209b0a102f23df9bc5ac1fb09e4cb171813b4aee2e932c7fe6aaa17f` |
| Controller original | `0c4df4f1f4b4be9cdf345b6084b5f3dbe5d3320122676879c5c3c26fbbf23611` |
| SDK106 metadata package | `704e992a0d86aeda3bd4a23201c19daf51825bf08d29cd82f1810298ffe76653` |
| SDK106 build manifest | `6a2414e6e02a17d1abe06175a294b850f56908c24b31625b951708448996e620` |
| Recorded observer DSO | `51b0d92465232cbc26ad2d93416b028bd1601c3dcadc4282b25304512600526d` |

The SDK package's 62 metadata/log files were individually rehashed. Its
13 compile-source dependencies and all 19 existing plus 28 new CPU test logs
match. The local package excludes the DSO and CPU executables; this review
therefore records their local rehash as false. The recorded DSO identity is
not presented as a local binary rehash. Running the same reader against the
complete SDK directory on 124 additionally verifies available binary bytes.
CPU build results remain separate from the actual observation.

The review also rehashes 20 runtime/support modules and the frozen probe.
The read-only `/probe_quant_containers.py` support mount must have the exact
runtime bytes. No private tensors, original metadata, binaries or process
logs are committed by this review.

## Preserved limits and process warning

The original process log, SHA256
`3ceacc6005f0d0c37b255256af6a1bebca0f673a6f1f70ea176ab2eaa3639817`,
contains 17 occurrences of the forkserver `signal.signal` TypeError about an
invalid signal handler. Those traces are preserved. The observed main Worker
and controller completed, but this review does not infer that every background
process was healthy or identify the warning's root cause.

Health here compares token IDs only. It does not independently compare text,
logits or full numerical outputs. Execute returning is not device drain.
Release covers owned host journals and observed bookkeeping; the preloaded
original library mappings remain resident until process exit. There is no
statistics/mix capture or fault in this run.

## Re-running the independent review

Keep the reader and its three stdlib dependencies together:
`review_quant_container_observations.py`, `review_quant_routed_frames.py`,
and `audit_quant_source87_routes.py`. Run without importing the production
binding, using the frozen source roots and the original package:

```sh
python review_quant_gmm2_observations.py \
  --archive ORIGINALS.tar.gz \
  --archive-sha256 88107fcbb0cf12cb5516f9db78abb091d3d0f38de8e9ba9529bd5367a309735f \
  --native-build-root SDK106_BUILD \
  --native-build-sha256 6a2414e6e02a17d1abe06175a294b850f56908c24b31625b951708448996e620 \
  --native-source-root SOURCE106 --runtime-source-root SOURCE107 \
  --out NEW_REVIEW.json
```

The output is created exclusively. Missing, malformed or failed observations
produce a partial/refused diagnostic with all qualification flags false;
originals are never filled in, edited or relabelled. Tests include current
registration reuse, named slot order, signed scalar/boolean/null checks,
function/caller identity, workspace/executor/stream mismatches, missing calls,
metadata-array accounting, cleanup mutations and captured-token divergence.

Local validation: 33 CPU tests passed across the new independent reader and
the unchanged four-container reader. This reviewer made no SSH, container or
NPU call and changed no Native, Runtime or prior reader file.
