# Online Softmax denominator replay: a new controlled reduction fault

This experiment preserves the previous legal-tiling experiment unchanged. It is
not a reproduction of the historical fault28 root cause. It intentionally counts
the first S2 block twice in the online Softmax denominator, once when that block
is computed and once when the second block updates the running denominator.
The score calculation, numerator/PV reduction, FP16 input/output, FP32 compute
type, causal mask, and final native normalization/copy remain unchanged.

B and F use the same private package and M128/S2=128 at B8. Only F selects a new
compiled template, key 119000000000400000. The healthy template remains
109000000000400000. The fault runs for every eligible call, every request, every
head, and each query with a second S2 block (q >= 128). Earlier queries cannot
trigger it. No returned tensor is edited. Host rejects unsupported keys/shapes.

For row q, let D be the correct denominator and D0 the contribution of keys
0..127, both expressed relative to the final row maximum. The expected fault is
O_F = numerator / (D + D0), versus O_B = numerator / D. For q < 128, O_F = O_B.
This follows the online Softmax recurrence, not an empirical output multiplier.
Reference for the correct algorithm: [FlashAttention](https://arxiv.org/abs/2205.14135).

## Frozen validation, before NPU execution

- One natural input only: byte-identical input_00 from the previous frozen bank;
  Qwen3-0.6B, B8, FP16, 2048-token prefill, 28-layer real FIA eager path.
- Fixed comparator: abs(F-B) > 1e-8 + 1e-5*abs(B), without epsilon or rounding.
- Independent first difference must be FIA output[0]. Max/Min/Norm at that
  output must change by at least 0.1%. Mean does not qualify magnitude.
- Full-model B/F: one warmup and three captured repeated prefills (step0–2),
  unfiltered statistics/mix API/module input and output, same CPU-FP64 collector.
  Repeat in independent processes, then disable only this error with M/S2 fixed.
- Select first difference using only dump/stack/construct/config/collector
  identity. Only afterwards inspect runtime, key, QKV, per-request logits and
  coverage. Preserve null and alignment failures.
- Native synthetic diagnostic: Q=K=0, V=1. B output is 1; F output is 1 before
  q128 and FP16((q+1)/(q+1+128)) afterwards. This is a separate operator fixture,
  not a replacement for natural model input or model acceptance.
- Fixed real-QKV B1/B8 controls: healthy fixed M128, fault fixed M128, and
  batch-gated fault. B1 batch-gated uses healthy M512; no replay is expected.
- Legal scores/PV keys remain covered once; denominator keys 0..127 are counted
  twice deliberately. No Reduce axis is split across cores. This distinction
  must be explicit in every result.

Only one idle physical NPU at a time, existing resource locks and per-case gate,
at most eight cases per batch. No host-wide Docker changes. Old private vendor,
statistics, input bank and search results are immutable. A separate vendor is
built and fingerprinted before any NPU case. Formal public collection is B/F
only; complete private arrays are auxiliary and are never publication inputs.

Possible conclusions include failure of the new implementation or statistical
target. A passing result establishes this new denominator-replay mechanism and
its model propagation, not historical legal-tiling precision error, FIA graph
mode, arbitrary geometry, continuous decoding or task accuracy.
