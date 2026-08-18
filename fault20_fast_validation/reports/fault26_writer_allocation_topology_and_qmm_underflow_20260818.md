# Fault26 writer allocation topology and QMM output-base underflow proxy

Date: 2026-08-18

## Result

Fault26 now has two deliberately separate operator-writer capabilities:

```text
Phase D
FIA wrong-destination output binding
  -> real FIA writes a live residual
  -> not an OOB write

Phase 27
QMM output base shifted across a proven allocation boundary
  -> real QMM writes 512 bytes into a live residual
  -> controlled allocation-boundary underflow proxy
```

The Phase 27 experiment passed with the following classification:

```text
FAULT26_DIRECT_ALLOCATION_OOB_CANDIDATE_FOUND
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN
REAL_QMM_WRITER_PROVEN
NEXT_ADDRMS_CONSUMER_MATCH_PROVEN
QMM_OUTPUT_RESTORED_EXACT
RESIDUAL_STORAGE_RESTORED_EXACT
RECOVERY_PASS
TOKEN_DIVERGENCE_NOT_OBSERVED_THIS_HIT
```

It must not be described as a naturally occurring QMM kernel OOB, the
historical FIA workspace defect, or proof of the original physical root cause.
The injected fault is a controlled output-base address error whose write is
performed by the real `aclnnQuantMatmulWeightNz` execution.

## Environment

```text
model             Qwen3-32B-w8a8-nopdmix
model type        dense W8A8
CANN              8.5.1
vLLM              0.18.0
TP                2
logical devices   0 and 1
mode              eager, VLLM_SERVER_DEV_MODE=1
container         fault26-qwen3-32b-readonly-inventory-
                  20260818_phase27_qmm_underflow_v2
port              18028, bound to 127.0.0.1
runtime           fault26_qmm_underflow_v2
```

The service was left running, healthy, and explicitly disarmed at handoff. It
was later manually terminated by the user; that normal exit is not a service
crash or delayed injection failure.

## Why Phase D is not OOB

The earlier Phase D mechanism changes the FIA output destination directly:

```text
normal FIA output address
  -> substitute live residual address
  -> real FIA writes within that supplied destination span
```

It proves a real-operator writer and residual propagation, but it does not
cross the writer's allocation boundary. Its correct name remains:

```text
FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY_PROVEN
```

Phase 27 instead uses a writer allocation boundary proven from the live NPU
caching allocator, then shifts the QMM output base backward across it.

## Read-only topology progression

The topology audit was staged before any boundary-crossing write:

1. Declared tensor spans proved that the live residual is earlier than the QMM
   output. A normal output-tail forward overrun could not reach it.
2. Visible writer topology ranked FIA, InplaceCopy, Matmul, and QMM in the
   residual lifetime. No healthy execution changed residual storage.
3. The first allocator snapshot was rejected because its residual metadata was
   stale from an earlier request.
4. A residual-generation gate then required a newly produced residual in the
   same request.
5. QMM-time allocator capture found an exact active-block adjacency candidate.
6. Phase 27 repeated that proof inside the same worker processes that later ran
   the controlled experiment.

The Phase 27 read-only request produced eight complete captures, four per
rank. One candidate per rank had the same topology:

| Field | Value |
|---|---:|
| residual generation | 3 |
| residual declared address | `0x12c1c235e200` |
| residual declared bytes / shape | 102400 / `[10,5120]` |
| residual allocation block | `[0x12c1c235e200, 0x12c1c2377400)` |
| residual block requested / actual | 102400 / 102912 bytes |
| QMM output address | `0x12c1c2377400` |
| QMM output bytes / shape | 512000 / `[10,25600]` |
| QMM output allocation block | `[0x12c1c2377400, 0x12c1c23f4600)` |
| QMM block requested / actual | 512000 / 512512 bytes |
| block relation | exactly adjacent |
| allocator padding after residual data | 512 bytes |
| minimum underflow to touch residual data | 513 bytes |

Thus:

```text
residual allocation end == QMM output allocation begin
0x12c1c2377400          == 0x12c1c2377400
```

This proves a plausible controlled boundary-crossing primitive. It does not
prove that the healthy kernel naturally calculates or uses such an address.

## Controlled primitive

The one-shot rank-0 configuration was:

```text
target API                 aclnnQuantMatmulWeightNz
target device              0
target QMM output           0x12c1c2377400
target QMM output bytes     512000
target residual             0x12c1c235e200
target residual bytes       102400
proven common boundary      0x12c1c2377400
output-base shift           1024 bytes backward
allocator padding crossed   512 bytes
residual overlap            512 bytes
max effective hits          1 per worker process
```

The runtime rejects a call unless device, addresses, sizes, rank-2 contiguous
metadata, output slot proof, residual producer, and allocation boundaries all
match exactly. Rank 1 receives the control-plane configuration but cannot hit
because its logical device is 1.

Execution order:

```text
backup current live residual
  -> replace QMM output base with output_base - 1024
  -> real QMM Run
  -> synchronize
  -> prove residual changed
  -> copy the complete shifted QMM result to its normal output storage
  -> verify normal QMM output byte-for-byte
  -> real next AddRms consumes the corrupted residual as x2
  -> restore the original residual
  -> verify restored residual byte-for-byte
```

No host-side mutation creates the residual corruption. The injection framework
changes the output address and performs backup/recovery; the real NPU QMM kernel
performs the unauthorized cross-allocation write.

## Runtime evidence

Only PID 772 / logical device 0 emitted underflow events. PID 773 emitted none.

| Evidence | Result |
|---|---:|
| QMM Run status | 0 |
| original output | `0x12c1c2377400` |
| replacement output | `0x12c1c2377000` |
| shift / overlap | 1024 / 512 bytes |
| residual changed bytes | 510 |
| residual hash before | `0x73350265b65514ae` |
| residual hash after QMM | `0x1a1890262ff75d34` |
| shifted output hash | `0x6832d9870071fbcd` |
| restored normal output hash | `0x6832d9870071fbcd` |
| output restored exact | true |
| actual AddRms consumer sequence | 4228 |
| consumer status | 0 |
| restored residual hash | `0x73350265b65514ae` |
| residual restored exact | true |
| host backup freed | status 0 |

The final runtime outcome was exactly:

```text
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN
```

The 512-byte overlap changed 510 bytes rather than all 512 because two written
bytes happened to equal their prior values. The success criterion is an actual
nonzero byte change plus real consumer and exact restoration, not a requirement
that every overlapped byte differ.

## E2E responses

All saved requests used:

```json
{
  "model": "qwen3-32b-w8a8-fault26",
  "prompt": "Explain why deterministic testing matters in one sentence.",
  "max_tokens": 8,
  "temperature": 0,
  "seed": 123,
  "return_token_ids": true
}
```

Baseline, read-only snapshot, pre-fault recovery, fault, and post-fault recovery
all returned HTTP 200 with the same result:

```text
text
" Deterministic testing ensures consistent and reliable results"

token IDs
[34907,4532,7497,25351,12966,323,14720,3059]
```

Therefore this hit proves the operator-level storage fault and propagation
window, but not token divergence. Output equality does not invalidate the
runtime proof because the corrupted 512-byte residual tail was consumed by the
real AddRms and then restored.

The disarmed post-fault request was token exact and `/health` returned 200.

## Failed attempt retained as a guard lesson

Phase 26 performed no write. Its context builder required the future AddRms GWS
record to exist before the QMM Run. Real execution order showed
`has_current_residual=true` but `has_next_add_rms=false`, so the guard rejected
every candidate.

The v2 correction does not guess the future consumer. It captures the current
producer residual at QMM time, then identifies the first successful future
AddRms by exact device, `x2` address, and byte size. The actual consumer sequence
is recorded only when that consumer executes.

The experiment also exposed that Python's `os.environ` cache does not observe a
native C `setenv()` auto-disarm. The worker extension now queries the process's
real C `getenv()` for this gate. Phase 27 was also explicitly disarmed after the
request, and the final query returned `armed=false` on both workers.

## Evidence boundary

Proven:

```text
live residual producer storage
real caching-allocator block boundaries
exact adjacent QMM output allocation
real QMM write across that allocation boundary
residual byte change during the QMM window
real next AddRms consumption
exact QMM-output and residual restoration
healthy E2E recovery
```

Not proven:

```text
healthy QMM naturally performs an OOB write
historical FIA workspace/root-cause reproduction
kernel-internal temporary-buffer layout
the original defect's tiling or size-calculation bug
token divergence for this single controlled hit
```

## Artifacts

Source:

```text
fault20_fast_validation/source/fault26_readonly_trace_v1/
  fault26_qmm_underflow_api.h
  fault26_qmm_output_underflow.cpp
  build_fault26_qmm_output_underflow.sh
  fault26_trace_preload.cpp
  fault26_worker_extension.py
  analyze_fault26_live_allocator.py
  create_fault26_inventory_container.sh
```

Remote evidence:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_qmm_underflow_v2/
  baseline_20260818_phase27.json
  readonly_snapshot_response_20260818_phase27.json
  pre_fault_recovery_20260818_phase27.json
  fault_response_20260818_phase27.json
  post_fault_recovery_20260818_phase27.json
  qmm_live_allocator_20260818_phase27.json
  qmm_underflow_20260818_phase27_qmm_underflow_v2.jsonl
```

The same compact evidence is archived under `.fault26_evidence/phase27/` and is
excluded from Git.
