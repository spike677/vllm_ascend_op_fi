# Fault8 B/F msprobe statistics dumps — eight-token capture

Only the baseline (B) and systemic restore-offset fault (F) groups are included.
This is a new Fault8 artifact directory; existing Fault18 artifacts are unchanged.

Injection mechanism, scope, lifecycle and evidence boundaries:
[故障注入逻辑说明](INJECTION_LOGIC.md).

## Configuration and contents

- `task=statistics`, `summary_mode=statistics`, `level=mix`.
- Steps **0 through 7 inclusive**, ranks **0 and 1**.
- Each request uses `max_tokens=8`. Step0 covers the full 38-token prefill;
  steps1–7 cover subsequent decode forwards. Steps are not individual input tokens.
- Each archive contains 16 `dump.json`, 16 `stack.json`, 16 `construct.json`,
  `dump_config.json`, and `probe_config_ack.json`: **50 JSON files**.
- Roots are `B_ON/` and `F_ON/`. No tensor payloads, responses, service logs,
  runtime traces, binaries, source code, I/Z/R groups, or failed attempts.
- Archived config is the enabled request-time config. Runtime startup used
  dump-disabled configuration; `probe_config_ack.json` records activation.

| Archive | Group | Compressed bytes | Dump files |
|---|---|---:|---:|
| Fault8_Baseline_B_ON.zip | B: no substitution | 2,927,991 | 16 |
| Fault8_Fault_F_ON.zip | F: restore_offset=+1, duplicate_last | 2,933,471 | 16 |

ZIP integrity, exact membership, step/rank coverage and each member's SHA256
against the retained source archive were verified. Each dump contains 48
`NPU.npu_moe_token_unpermute.*.forward` entries: 768 per group.

## Associated runtime results

Model: Qwen3-VL-30B-A3B-Instruct-w8a8, text-only Q86, TP2/eager, A2/910B4,
CANN8.5.1, vLLM-Ascend `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`.

- B: `The marketplace roared to life at dawn` (8 tokens).
- F: eight newline tokens, all integer token ID198.
- Both finish with `length` because this capture deliberately caps generation at8.
- Each matches the corresponding no-msprobe historical response's first8 token IDs.
- F substitutes paired indices and top-k weights in all48 eligible prefill calls
  on each rank, with1776 changed layer/token group instances per rank. Native
  consumer calls occur with the probe active in step0. Single-token decode calls
  are ineligible for this mapping mutation but their downstream propagation is dumped.
- Owner/registry restoration and service shutdown passed. No additional recovery
  request was run in this B/F-only capture.

The fault is an actual mapping-input corruption proxy, not a demonstrated natural
padding defect. These dump-only ZIPs are not the full runtime forensic package:
hit/owner/response assertions above come from separately retained evidence.

## SHA256

```text
d62a732415b1f1da381aba54ca17604d57a63ec94936887ade38eef5da8803e8  Fault8_Baseline_B_ON.zip
fecc92130280f95e84f0aa06135b140c5e9fb01ec546f63a83f8fc2de0253bcd  Fault8_Fault_F_ON.zip
```

Source run: `/home/haoran/fault8_msprobe_bf8_20260916_v1/{B,F}`.
Local complete evidence: `fault8_17_readonly_audit/runtime_binding/msprobe_bf8/`
in the research working directory, separate from this Git delivery copy.
