# Ordinary quant source118: independent evidence review

The 2026-11-19 card5 run supports a **blue-zone public Host/Runtime single-projection function and same-Worker OFF result**. It does not establish main-project integration, all-target coverage, C2/C3, or E5. The original `c1/c2/c3/qualification_granted=false` fields remain unchanged. No new device work, weights, Native build, or runtime changes were made for this review.

The immutable remote attempt is `product-quant-source118-card5-a1/quant-qwen06-ordinary-six-batches` under `/home/17-fault-c2c3-20261117`. Nine local JSON originals total 12,896,397 bytes; no originals are committed. [The compact JSON summary](review_quant_ordinary_source118.json) records all nine SHA-256/size pairs, per-phase full-readback hashes, four Native receipt digests, and the verification categories. Its SHA-256 is `410ca6f229c806b5159271d4deaacc80859e25ca18333a7fee3de7cad7bf673c` (UTF-8, LF).

The stdlib-only local reconciliation made 375 functional/provenance assertions plus two assertions documenting the pending-OFF display defect. It rehashed each original, matched all six phase byte counts and SHA references, recomputed the four complete Native packet digests, compared the six cumulative phase prefixes with the final record, and checked all 22 installed/Worker-loaded Python module hashes against Git blobs at the recorded runtime commit. It also reconciled every selected child GWS/execute event, lease, domain, workspace, stream and publication receipt. This is not a new NPU test.

| Identity | Recorded value |
| --- | --- |
| Runtime source | `51bdd40654648ffafcd3fbb203743f80ff4cf641` |
| Source manifest | `53a42f8ad28fbbd57e8ab6d9618d90b8e5ff7dbf96dc9b691f3ea788d7de5a36` |
| Image | `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0` |
| Existing Native build manifest | `b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac` |
| Separate Native compile source | `46cd8ad153ad07ce1057cfcbac73d7d9fb76e226` |
| Native library | `6eb6c19cff6006bc9cd0b1072504dc5d6b7db3a0d7b393af7c88f4db1c8bd2fd` |
| Result original | `be61dd097fb5472402cc0b65c69c148dd03dfb7d780109e347de80de8812ddc6` |
| Host original | `4a0a61952cb010262cc15de0900546f59b1c18f22633a937102a40046b0925ed` |
| Controller original | `0035ddc750a9fac33c0678d2bb9062b4507d2ea948dce76d3d10283758a00017` |

The runtime automatically selected `model.layers.0.self_attn.qkv_proj` in the ordinary Qwen3-0.6B dynamic-W8A8 a5 artifact. There are 84 structural projection candidates; only this actual single site is in scope. The input is BF16 `[8,1024]`, Q is INT8 `[8,1024]`, and per-token scale is FP32 `[8]`. TP1/rank0 uses logical device0 on physical card5. The six real B8 batches remain explicit current-scope calibration and execution: `normal → reference → lossless → calibration_off → fault → off`. Each has normally enqueued distinct request IDs, one prefill and one first pure-decode step; none invokes the model through an RPC replay.

The four Native admissions have modes `1,2,4,3` and distinct current-parent lifetime tokens. F contains one model invocation, one correct-domain reference helper, and one effect invocation: two actual GWS and two actual execute calls, all represented by the parent plus eight raw child events. The correct-domain helper completes before the effect execute, on the same stream, using the same unchanged current input and disjoint owned Q/scale buffers. The fault domain is `[1,8192]` with one FP32 scale; the reference retains `[8,1024]` and eight FP32 scales. Both outputs are exactly published according to the Native completion receipt.

| Observable | Normal → F → OFF |
| --- | --- |
| Q changed elements | `5565/8192`; 2627 unchanged; finite maximum absolute difference 58 |
| Scale changed elements | `7/8`; one unchanged; finite maximum absolute difference 0.009165847674012184 |
| Native changed bytes | Q 5565; scale 21 |
| Input full-readback hash | Identical in all six phases |
| Q/scale full-readback hashes | Both differ in F; reference, lossless, calibration OFF and final OFF equal normal |
| Decode hidden/logits hashes | Both differ in F; every non-F phase equals normal |
| Prefill hidden/logits hashes | Equal in all six phases |
| Returned second token | Logical row7: `374 → 653 → 374`; the other seven second tokens are unchanged |
| Returned first tokens | All eight unchanged throughout |

Each phase records exactly one actual QMM consumer and a BF16 `[8,4096]` result, at the selected module and pinned method source. The live adapter enforces current Q and scale tensor identity, immediately rereads their complete hashes, verifies the actual module-owned weight/scale/bias, then calls the original QMM once. **The consumer JSON does not contain every QMM argument object or that immediate reread separately.** This supports source-bound live execution evidence; it is not an independent offline reconstruction of every consumer argument.

OFF completed in the same process-start/Worker/model identity. All eight before/after healthy token pairs match; all recorded model file hashes are unchanged. Native fault-owned device/host/workspace/descriptor resources, pending calls and public leases are zero; all owned hooks, method/API patches, row modes and retained normal readbacks are zero. The 32-entry Host history completes `finish`, `unregister`, `detach`, and deployment removal without cleanup errors. No late public-pointer restoration was requested. Fixed Native metadata journal storage remains 10,682,368 bytes for diagnostics: the zero-resource conclusion concerns fault ownership and active resources, not all resident process memory. Container removal is recorded separately and is not used as OFF evidence. Runtime elapsed was 313.036 seconds and controller elapsed was 319.390 seconds, within the fixed 900-second case budget.

The actual Host original also confirms a display-only defect. At history entries 25 and 26, OFF is pending and five rows are completed, but `current_request_effect` still points to the prior F. `actual_fault_hits` stays one. After explicit OFF collect, the sixth row is present and the current effect is null. This is assigned to a separate display repair; the historical F receipt must remain unchanged.

The local package contains metadata, bounded numerical summaries and complete-readback hashes, not full tensor payloads, model weights or the Native DSO. Therefore the review reconciles recorded counts/hashes and frozen source enforcement; it does not independently recompute all tensor values or rehash the remote weight/DSO bytes. The runtime and Native compile-source identities are kept separate.

Source117 remains a failed, separate attempt: normal completed, reference refused before Native admission/F with `NATIVE_ROW_SOURCE_FRAME_CONTEXT_CHANGED`. Its previously reviewed result SHA is `9663072fd76dcfa367d576770a87122c5b6a11b8f7801f17b1389654c6a0dc02`, and normal-phase SHA is `473488af83b13cb80add2802b77593adfe7d50607d9aa8c9bb14a11a2a223dc0`. Source118 does not overwrite or relabel that failure. Main-project source/integration remains unavailable and unaccepted; the existing ABI still requires six current-scope batches, not a three-batch product API.
