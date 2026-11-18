# Fault33 source105: public Worker single-site acceptance

The root accepts the completed **single-site functional and OFF experiment**
on124/card5, Qwen3-0.6B, TP1, FP16, CANN8.5.1/A2. The actual TND paged
first-decode call keeps Q=1, KV=1025, Hq/Hkv=16/8, D=128, page size16 and
M=512. The fault changes the real host-selected S2 grouping512→256 before
SplitCore. Native launch block dimension changes16→20. This is a controlled
S2 strategy variant, not evidence that historical blockSize-trigger selection
has been reproduced.

The source105 stock/private normal bridge first passed435 complete payload
pairs exactly. The same runtime package then completed reference, lossless,
qualification OFF, fault and recovery OFF through the public Worker client.
Each request performs a fresh1024-token prefill and first decode, retaining
all28 prefill and28 decode Attention calls. The sole injected call is the
actual first-layer decode object. All27 other kernel-input-matching decode
calls remain explicitly unqualified; they are not removed from the denominator.

The [summary](attention33_source105_single_summary.json) records the independent
raw-file review, including original RPCs, actual request identities, root input
hashes,64→65 logical pages, writer slots/cache storage, Native host/launch/ELF/POD,
435 complete output payloads and public cleanup. Eleven aggregate reader
checks passed with zero issues; this count is not a count of tensor elements.
The reader did not use qualification booleans as proof.

| Complete-array observation | Numeric elements changed | Maximum absolute difference |
|---|---:|---:|
| Immediate FIA output, shape `[1,16,128]` |72|0.0001220703125|
| Final logits, shape `[1,151936]` |135007|0.017578125|

Fault changes85 of87 saved payloads. The first recorded Attention output
change is layer0; its logical K/V remain exact, while subsequent Attention
outputs and the current decode cache rows change. Lossless, qualification OFF
and recovery OFF each restore all87 payloads exactly against reference.
All five protocol requests and post-unregister health return tokens
`[1102,13]` and text `" result."`. Numeric propagation is observed; changed
generated text is not. This is two emitted tokens, not continuous-generation
or task-accuracy acceptance. Post-unregister health observes tokens/text only.

The final public state has zero inflight calls, leases, owned hooks and API
wrappers; Native is disabled/unregistered. The reader also checks input-reference
release, deployment removal and same-Worker health. Its original results remain
unchanged, including the initial pending-review status.

## Preserved failures and exact identities

Source102 previously failed before protocol requests because the pinned observer
receipt file was passed to a reader expecting its build directory. The repaired
loader validates that exact receipt, derives its canonical root, and handles
uncertain preparation through public detach when no lease ticket exists. A
production-image CPU regression reproduced the old refusal and re-read all435
old normal pairs successfully. Source105 then ran its own fresh normal pair and
F/OFF; it does not inherit source102's execution qualification.

The first offline source105 review incorrectly expected the decode scheduler's
`request_ids` field in prefill records. A separate reader now checks each actual
schema and their request/token/page/storage joins. The old failed report is
retained. The second review reads the same original source105 results and bins;
there was no NPU rerun or modification of captured data.

| Artifact | SHA256 |
|---|---|
| Source105 manifest | `4a4bb617af27ecaca8232721782f8a1f3974a6e4fcb8081cb4c6e7c8018c2eb6` |
| Normal review | `1e113a94c04398facb0cd678c28d66fc02ca484d842b75fbef5ffc6cdad3349f` |
| Actual public result | `f87247ec8a78c8258489025216997c30deb3a61072603b90f0efd087147c2ecc` |
| Preserved first review | `24f11a6227a17637e112d41012ae00d6875981aca183372adc651c72de26a730` |
| Corrected reader source | `4deb3eedac4add1fd7f7bbd55c6f916ca80f8298621a54c6b5d594ccb25fcbc2` |
| Successful second review | `2e5d19cd91eb566e71a74429879f193ee8369fce00115f6556d7913daadf2111` |

Original runs remain under `/home/17-fault-c2c3-20261117/` in
`v11-33-s2-normal-source105-card5-a1`,
`v11-33-s2-public-source105-card5-a1`, and their independent-review directories.
Git contains implementation, tests and this summary, not complete arrays,
registered ELF files or new raw statistics.

## Acceptance limits

This accepts one actual target's function and recovery. **C2/C3 are not granted**;
the accepted totals stay9/17 and8/17. Saved reports cannot authorize a new run.
The actual scoped launch, registered binary and final POD were checked, but
selected ACLNN caller ownership remains unobserved. Before-read mask/logical-KV
evidence uses source-bound hashes; later complete matching KV bytes were read.

These numerical results come from complete-array comparisons. They are **not
an msprobe `statistics/mix` first-difference result**. E5 remains unassessed for
this new public Worker scope. Stock POD, other layers' per-target normal/effect
proof, broader shapes, CANN versions and hardware remain separate work.
