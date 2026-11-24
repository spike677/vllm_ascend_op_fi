# Fault18 ordinary task: independent Source126 review

The original records support one automatic RMS return-half injection, its
actual consumer, and same-Worker OFF/health recovery. This was Qwen3-0.6B,
TP1, BF16 eager on 124 card 5: four fresh 14-token prefills, each generating
one token. The numerical effect reached hidden states and logits; all four
requests generated token **576**, text **` The`**. No generated-text change
or continuous-generation result is claimed.

This review read the original result, complete Host receipt, controller and
source manifest. It did not use the result's top-level `status` or `checks`
as acceptance authority. The machine summary is
[review_rms18_ordinary_source126.json](review_rms18_ordinary_source126.json).
Only these two summaries are published; original JSON and tensor data are not.

Source126 is `ce597d2dfb7a0c9c939d1a9164c83e3651f4abb0`. The manifest bytes
independently hash to
`c36875d564665c9163fd223bb9fc5eb1089b1b1dd4914a19dd713d6e110cec3b`.
All **31 participating source files** match both their Git blobs and manifest
entries; all **24 installed module fingerprints** join those pins. The
controller records the same manifest, commit and fixed image
`35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
This does not claim an offline rehash of unused manifest files, model-weight
arrays, or external vLLM source files. Their recorded identities are retained.

The 17 original Host records follow `attach → prepare → register → check →
enable → (advance → query) × 4 → finish → disable → unregister → detach`.
All replies retain the same public ticket, Worker/model identity and
generation. The separate provider-owner token is stable, as required by the
Runtime lease; it is not incorrectly equated with the public owner string.
Each queued internal ID joins its actual model-prehook snapshot and completed
query row. Complete input-ID and position hashes were recomputed from the
recorded prompt and agree in all four requests; computed-token counts are zero.

| Item | Independently reconciled record |
| --- | --- |
| Automatic target | `model.layers.0.input_layernorm`, ordinal 0 |
| Original operator | `npu.npu_rms_norm.default`, once in each observed normal/F call |
| Consumer | `vllm.unquantized_gemm.default` at `model.layers.0.self_attn.qkv_proj`, same normal/F lineage |
| Normal/F binding | Same live module, target input, weight object/storage/hash and original-return hash |
| Events, normal/F/OFF/health | `1 / 1 / 0 / 0` |
| Applied events | `0 / 1 / 0 / 0` |
| F readback summary | `[14,1024]` BF16; 14,336 of 14,336 values changed; gain 0.5 |
| Largest absolute change reported | 1.4296875; no nonfinite values reported |
| OFF and health | Both complete hidden and logits hashes equal normal |

All four retained numerical samples independently satisfy `after = before/2`;
for example `0.07568359375 → 0.037841796875`. The full element count and maximum
come from the running adapter's complete same-return CPU readback summary.
The offline review checks that summary's arithmetic, shape/count consistency
and exact association with the original event and Host effect; it does not
recompute unprovided full tensor arrays. Complete hidden/logits digest
comparisons demonstrate recorded numerical propagation and restoration,
without relying on whether the generated token changed.

OFF/health install no RMS observer. Their two public request-boundary hooks
remain until `finish`, where the actual readback reports zero owned hooks,
dispatch leases and inflight work. Pending requests expose no previous
`current_request_effect`; the one F effect remains explicitly historical.
After unregister/detach there is no pending request, replay or recovery,
registration, Host ticket/attachment, or retained ownership. The separate
binding-removal receipt agrees. Controller exit is 0, OOM is false and its
owned container was removed; container removal is not used as proof of
numerical recovery.

This remains a **module-return half-value proxy**, not an internal Native RMS
reduction fault. Acceptance covers this one target and prompt; it does not
grant all-layer, multi-model, C1/C2/C3 or E5 qualification. No statistics/mix
was collected in this run. The independent blue-zone acceptance driver is not
the inaccessible yellow-zone main project, whose integration remains unverified.
