# Qwen two-card Fault Effect Evaluation V1

This evaluator connects the existing
`op_api.quant_matmul.input_buffer_transform` capability to a real developer-mode
vLLM service. It does not implement tensor mutation. Native mutation remains in
the validated Fault20 v17 preload/controller runtime.

## Fixed phase order

The live suite runs these states serially:

1. `baseline` with the controller disarmed;
2. `lossless` using `arm-all-qmm-lossless`;
3. `mild` using the existing INT4-equivalent transform;
4. `strong` using the existing INT2-equivalent transform;
5. `recovery` with the controller disarmed.

The destructive phases cannot start until both lossless bindings pass:

- decode QMM input `[1, 5120]` through the shared `ValidationResult` gate;
- WikiText scoring QMM input `[8, 5120]` through exact NLL equivalence,
  successful native hits, temporary-buffer release, and a healthy service.

Every armed scope is protected by `finally: disarm()`. The final result uses the
existing `EvaluationResult`; phase details and per-rank records are placed in its
backward-compatible `extra` fields.

## Data and deterministic metrics

`eval_data/smoke_v1.jsonl` contains 20 fixed prompts with temperature zero,
fixed seed, and fixed generation limits. It covers simple math/facts, short
reasoning, exact formatting/newlines, longer generation, repetition, and
punctuation-sensitive output.

The V1 suite also supports:

- a fixed indexed subset of the official GSM8K test JSONL with deterministic
  numeric parsing (the `####` answer marker takes precedence);
- WikiText-2 raw text scored with OpenAI-compatible `echo` prompt logprobs;
- overlapping fixed 8-token windows. The one-token overlap means all tokens
  except the first document token receive a teacher-forced score while every
  targeted QMM has the same `M=8` selector. `8 × 5120 = 40960` bytes remains
  within the existing Fault20 v17 all-hit controller limit of 65536 bytes; no
  native limit or mechanism is changed;
- response equality/hash, token-piece comparison when integer IDs are absent,
  first/token divergence, output length ratio, repetition, punctuation spam,
  early-stop, crash/error, accuracy retention, NLL, and PPL ratio;
- three complete representative `arm -> fault -> disarm -> recovery` repeats.

Token pieces are never stored as token IDs. When the API does not expose integer
IDs, `token_ids` remains `null` and the comparison basis is recorded separately.
No LLM judge participates in PASS/FAIL.

## Rank-aware evidence

The runtime snapshot records the expected and observed worker count, worker PID
and liveness, native candidate/hit counters, evaluator-derived selector rejects,
and CANN/HCCL/OOM/worker-exit indicators. For the rank0-only modes, matching
rank1 calls are labeled `expected_reject`; they are not treated as mechanism
failure.

## CLI

Run the script on the model host so it can read `/proc`, the native JSONL trace,
and the POSIX shared-memory controller state:

```bash
PYTHONPATH=src python3 tools/run_live_qwen_fault_effect_evaluation.py \
  --base-url http://127.0.0.1:18031 \
  --model-name qwen3-32b-w8a8-eval \
  --control /path/to/fault20_buffer_ctl \
  --trace /path/to/buffer.jsonl \
  --service-log /path/to/server.log \
  --smoke eval_data/smoke_v1.jsonl \
  --gsm8k /data/gsm8k/test.jsonl \
  --gsm-limit 10 \
  --wikitext /data/wikitext-2-raw/test.txt \
  --wikitext-windows 8 \
  --wikitext-window-tokens 8 \
  --capabilities profiles/capabilities \
  --output /home/user/qwen_effect_v1 \
  --repeat 3
```

Use `--validate-only` to stop after the real candidate and decode lossless gate.

The outputs are:

- `candidate.json`;
- `lossless_validation.json` and `lossless_gate_runtime.json`;
- one raw phase JSON under each phase directory;
- `repeatability.json`;
- unified `evaluation_result.json`;
- `effect_evaluation_summary.md`.

## Evidence boundary

INT4/INT2-equivalent transforms retain INT8 storage. They are controlled input
precision-degradation proxies, not packed INT4/INT2 kernels. A successful V1
proves a repeatable, measurable, rank-aware, recoverable evaluation loop on the
local open-source model. It does not claim a healthy kernel naturally produced
the simulated fault.
