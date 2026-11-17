# Agent7: actual MoE8 Worker group review

The recorded run supports a limited **standard vLLM Worker, TP1, eager, fresh-prefill** MoE8 C2/C3 candidate for root acceptance. It covers the actual 26 MoE layers in the loaded DeepSeek-V2-Lite-W8A8 model. This audit grants no qualification and does not change the repository's strict acceptance count. There is no statistics capture or E5 first-difference evidence in this run.

Nine separate requests were enqueued through the public vLLM interface and matched to the actual Worker model pre-hook and scheduler. They used the same 10 tokens, positions 0–9, one request per forward, zero prior computed tokens, and one generated token. The native baseline, instrumented baseline, repeated baseline, and lossless request returned identical complete hidden states and full-vocabulary logits. No model or router layer was synthesized or replaced for the test.

| Check | Independent result |
|---|---|
| Actual model structure | 27 layers, first dense, 26 actual MoE modules; 64 routed experts, top-k 6 |
| Per-forward denominator | Every one of nine requests has the same 26 unique structural targets and 26 actual `npu_moe_token_unpermute` calls: 234 calls total |
| C2 | All 26 targets hit once; all 26 OFF calls passed through and restored the complete output |
| C3 | Same Worker/model lifetime, scope and owner, next generation; two requests produced 52/52 hits; 26 OFF calls restored output |
| Independent actual pipeline | For every native call, the audit rejoins top-k probability/ID objects, routing, quantized expert MLP output, absolute-index producer, actual native inputs, and routed return objects |
| Applied arguments | Hidden expert output remains the original object/storage. In fault phases the real native consumer receives replacement index and probability tensors; healthy/OFF phases receive the original index/probability objects |
| Per-target value checks | All 78 fault events record paired-group readback, full permutation validation, exactly one native execution, changed output, and exact BF16-rounded complete-group CPU oracle agreement |
| Full-model propagation | Three fault hidden/logits hashes are identical and differ from baseline; both OFF states exactly restore baseline. All arrays are finite |
| Output token | Every recorded request returns token 185; full logits change does not imply a Top-1 change |
| Cleanup | Adapter hooks/leases/in-flight calls cleared, group recovery complete, owned serving hooks zero, observer/adapter absent; the same Worker generates token 185 again after cleanup |
| Reference bytes | C2 55/55 and C3 107/107 referenced artifacts verified; 161 unique original canonical objects retained in fresh remote audit files |

The mechanism remaps **paired token contribution groups** from `[0,1,2,3,4,5,6,7,8,9]` to `[1,2,3,4,5,6,7,8,9,9]` before the real native unpermute. It preserves the corresponding index/probability pairing and leaves expert output unchanged. It is not an elementwise address offset or a demonstrated out-of-bounds write. The native consumer is recorded after the remap through `consumed_arguments`, so this audit can separately inspect the applied arguments; it does not borrow Vision17's weaker outer-observer argument evidence.

Full hidden-state shape is `[10,2048]`; logits shape is `[1,102400]`. Healthy logits SHA256 is `aa2f5c4d524e29664d911d2be287051db2460cfded6c952b1f6cb418999bf6ee`; all three fault logits use `ce6fbe6b768cb622d661620dce1242404e09a40859dada19c6ba7e047b34e71f`. The whole-array hashes and exact value/oracle comparisons were produced during execution. Private tensor values were not replayed by this offline audit and are not published.

The four weight shards totaling 17,214,938,536 bytes, 11 model implementation sources, three MoE pipeline source files, all 625 source-a36 files, the 13-file provider fingerprint, and the original model/quantization configurations were independently rehashed. The actual container image is `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`; the controller binds source-a36 read-only, CANN 8.5.1, and physical card2. The frozen Worker entry creates the vLLM normally and uses actual requests; live C2 proof issuance and C3 consumption occur within the same Worker RPC session. Offline reference reconstruction does not reissue that proof.

The resource conclusion covers owned hooks, active adapters/observers, explicit leases and in-flight calls. It does not claim zero model allocator memory or zero observer CPU-copy memory. The current result also does not establish TP2, multi-request batching, decode, other models/versions, or independent msprobe localization.

Original receipt: `/home/17-fault-c2c3-20261117/natural-vision17-moe8-rope22-card2-a1/natural-moe8-group/probe/result.json`, SHA256 `0bc9d2c82134f41a5ffdb61d980fdc9d4c99d23621e00354f098870322ad4a5d`. Fresh audit sidecars: `/home/17-fault-c2c3-20261117/agent7-moe8-group-card2-review-a2`.

[Machine-readable summary](agent7_moe8_group_card2_summary.json), SHA256 `b8d3e169996b0438e56a3c30eed090d01c1b02a913c27f1f8f9bf8b2a880c6ed`, is generated by [the CPU-only reviewer](review_moe8_group_124.py). Seven audit tests passed, including changed expert output, unpaired probability remap, duplicate native execution, baseline copies instead of original arguments, shape changes, and the frozen probe's original JSON digest convention. The first audit attempt stopped on the reviewer's incorrect compact-JSON assumption for the environment digest; the corrected audit uses the exact frozen probe encoding, without modifying original evidence.
