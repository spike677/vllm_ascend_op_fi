# Agent7: Vision17 card2 recorded group review

The completed Qwen3-VL image-prefill run supports a **limited MODEL_LOCAL** actual-model padding intervention, repeated group coverage, and exact OFF restoration. It is not a standard vLLM Worker run. No statistics were collected in this attempt, so it provides no E5 or independent msprobe first-difference result. This audit grants no public qualification.

The scope is the real pretrained `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`, BF16, CANN 8.5.1, physical card2. The unchanged processor produces one image request with 901 text/visual tokens and 3,536 flattened image patches. The full returned logits have shape `[1,901,151936]`. This is prefill, not a continuous-generation or multi-model acceptance test.

| Check | Independent audit result |
|---|---|
| Structural and actual denominator | One unique stock `Conv3d`, `actual.model.visual.patch_embed.proj`; one actual call in every one of nine phases; complete independent census, zero dropped calls |
| Healthy controls | Native baseline, instrumented baseline, repeat, and lossless: identical complete input and output identities |
| C2 | One of one eligible target admitted once; one OFF call passed through |
| C3 | Same scope, target, process/model lifetime and owner; next generation; two of two eligible calls admitted; one OFF call passed through |
| Live C2 handoff | Frozen probe directly passes `provider.group.c2_proof()` to the C3 session; frozen authority checks issued object identity and process registry. Offline audit does not recreate a capability |
| Effect | All three fault phases record unchanged complete input/weight/bias, one native call, padding `(0,7,7)` instead of `(0,0,0)`, changed complete Conv3d output and changed full logits |
| Repeat and OFF | All three fault logits hashes equal; both OFF outputs equal the native baseline; all outputs finite |
| Owned cleanup | Every phase reports no owned hook, lease or in-flight call; group recovery and same-lifetime health agree. This is not a claim that model allocator memory or retained observer CPU copies are zero |
| Saved references | C2: 5/5; C3: 7/7; 11 unique original canonical objects exported with original JSON pointers and matching referenced bytes |

Complete baseline logits SHA256 is `951f864f0b7089168938ae2cb3ac81bda759da4591d89edb2dc3798942ec99d0`; all fault logits use `ecd51816a24c20f81a6205be858ff330306106a0d0b0aa4e7a8808c9b6bf9d0b`. The native Conv3d output SHA256 changes from `3120c7e1f19165725e7c51cfb2aef76f277ca8eefded20ce2c592c3db2317b70` to `d539ffb67cd068310adf775f81ff088c58d83123fc257f4930740beb17fe77b7`. These are complete-array hashes recorded by the executed observer; the offline audit did not reload private output payloads.

Two observation limits remain explicit:

1. The independent dispatch observer runs outside the adapter. Its recorded `conv_geometry.padding` is `(0,0,0)` in all nine phases. It proves the original call and denominator; the adapter's inner `func(**native)` source and event supply the `(0,7,7)` argument evidence. There is no independent observer after the adapter that verifies the final native arguments. The report does not relabel the outer census as such a trace.
2. The adapter reports 393 consumer records through storage identifiers. The independent census does not preserve the first `add` input/output for separate lineage replay. The fault Conv3d storage identifier later appears in layer17 norm inputs, illustrating allocation reuse. Consequently, 393 is a provider record count, not 393 independently established consumers. Only the reviewed provider-recorded first consumer and complete model output propagation are reported; independent complete consumer lineage remains false.

Source and environment bindings were reread on 124: all 625 files in source-a36, the 13-file provider fingerprint, nine model implementation files, the 4,255,140,312-byte model weight, original image/config hashes, and the exact read-only source mount and image ID. The source manifest SHA256 is `b50acc508d8fa15e51bae75dd4e219541262f4ec785abf5ef5595ece229e00c0`. Model source files inside the existing image were read without starting a container. No custom native kernel or build is asserted for this stock Conv3d mechanism.

Original evidence remains at `/home/17-fault-c2c3-20261117/natural-vision17-moe8-rope22-card2-a1/natural-vision17-group/probe/result.json`, SHA256 `98390141cbdf84ea1565693e5cd06e76d725b7fdd4acd451659a9a10c540aff8`. Fresh audit sidecars are at `/home/17-fault-c2c3-20261117/agent7-vision17-group-card2-review-a2`. The earlier card5 attempt interrupted during statistics remains a partial failed attempt; this completed run does not complete or supersede its statistics.

[Machine-readable summary](agent7_vision17_group_card2_summary.json), SHA256 `87fcd577a567929766cde461d0af80ca86343c2779f73edd9269cfcf141407dd`, is produced by [the read-only reviewer](review_vision17_group_124.py). Seven CPU audit regressions passed, including missing per-target effect, changed actual input, duplicate denominator, incomplete census, pending hook, and failed OFF restoration. The first audit attempt correctly rejected an initially incorrect reviewer assumption that the outer census contained modified padding; original execution receipts were not changed.
