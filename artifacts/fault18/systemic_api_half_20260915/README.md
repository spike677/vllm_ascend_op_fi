# Fault18 baseline / fault msprobe statistics dumps

Replaces the earlier rank0 step0/step1 explicit tensor samples at these same
archive paths. Current archives contain only B/F automatic statistics dumps
and configuration evidence from the validated composed-wrapper run.

## Configuration and contents

- `task=statistics`, `summary_mode=statistics`, `level=mix`.
- Steps **0 through 8 inclusive**; ranks **0 and 1**.
- Each archive has 18 `dump.json`, 18 `construct.json`, 18 `stack.json`,
  `dump_config.json`, and `probe_config_ack.json`: 56 JSON files, no tensors.
- Archive roots remain `B_ON/` and `F_ON/` for compatibility.
- `dump_config.json` deliberately starts with `dump_enable=false` to exclude
  warmup. `probe_config_ack.json` records activation before the formal request.
- No Z/R groups, service logs, runtime event JSONL, responses, binaries, source
  code, or intermediate failed attempts are included.

| Archive | Group | Files | Dump files | Compressed bytes |
| --- | --- | ---: | ---: | ---: |
| Fault18_Baseline_B_ON.zip | B: disabled | 56 | 18 | 5012373 |
| Fault18_Fault_F_ON.zip | F: alpha=-0.5, output gain=0.5 | 56 | 18 | 5759551 |

Every archived member was SHA256-compared with the downloaded source archive;
ZIP integrity, exact file counts and step/rank coverage were checked. Each
dump.json contains 97 `NPU.npu_rms_norm` call entries.

## Associated run results and boundaries

Model: Qwen3-VL-30B-A3B-Instruct-w8a8; text-only Q86, TP2, A2/910B4,
CANN8.5.1, vLLM-Ascend `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f`.

- B: 316 tokens, `finish_reason=stop`; text, integer token IDs and selected
  logprobs exactly matched the frozen baseline. 30,652 API/consumer calls/rank.
- F: 1,024 tokens, `finish_reason=length`; repeated `bustling`, first token
  difference at index0. 99,328 changed API outputs and matched consumer calls/rank.
- Actual finite changes were recorded inside dump-active step0 on both ranks.
- Owner removal, registry restoration and service shutdown passed. No R request
  was run after the user limited this delivery to B/F; no new-request recovery
  claim is made for this run.

This is a normalization-output effect proxy, not native reduction statistic
corruption or naturally reproduced nondeterminism. With `F -> W -> N`, msprobe's
RMSNorm API dump observes native output; the proxy's changed output is observed
downstream. These dump-only archives are not the full runtime evidence package.
Counts and response results above come from separately retained runtime logs
and responses, not from the ZIP contents alone.

Source evidence: `/home/haoran/fault18_determinism_20260915_composed_v4/cases/{B,F}`.
The wrapper composition / one-time module registration implementation is not
included in this dump-only update.

## Archive SHA256

```text
b83672b6f28d93d82a657588c0376904420a3376d51c11e79bcd8e18affd9a4a  Fault18_Baseline_B_ON.zip
96dfd443947c29ccb826f322290a87c91f9a1439a2ac40d7f45bb513fbf45a32  Fault18_Fault_F_ON.zip
```

Old tensor archives remain recoverable from Git history; no history is rewritten.
