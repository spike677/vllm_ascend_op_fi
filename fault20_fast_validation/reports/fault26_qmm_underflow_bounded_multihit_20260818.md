# Fault26 QMM allocation-boundary underflow bounded multi-hit validation

Date: 2026-08-18

## Result

The v3 runtime completed both bounded multi-hit stages on rank 0:

```text
Phase 28: same worker process, three requests, one hit per request
Phase 29: one inference request, three decode hits in that request
```

Both stages returned HTTP 200 and were text- and token-exact to their disarmed
baselines. Phase 29 proves single-request repeated-hit execution. It did not,
for this prompt and these three 512-byte residual-tail overlaps, produce a
token-level output disturbance.

Accepted classifications:

```text
QMM_OUTPUT_BASE_UNDERFLOW_BOUNDED_MULTI_HIT_PROVEN
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN x3
CROSS_REQUEST_MULTI_HIT_PROVEN
SINGLE_REQUEST_MULTI_HIT_PROVEN
TOKEN_DIVERGENCE_NOT_OBSERVED_FOR_SINGLE_REQUEST_THREE_HIT
PADDING_OUTPUT_RESIDUAL_EXACT_RECOVERY_PROVEN
```

## Environment

```text
model             Qwen3-32B-w8a8-nopdmix
served model      qwen3-32b-w8a8-fault26
CANN              8.5.1
vLLM              0.18.0
TP                2
mode              eager, VLLM_SERVER_DEV_MODE=1
container         fault26-qwen3-32b-readonly-inventory-
                  20260818_phase29_qmm_underflow_single_request_v3
port              18028, bound to 127.0.0.1
runtime           fault26_qmm_underflow_v3
target rank       logical device 0 / PID 772
non-target rank   logical device 1 / PID 773
```

The Phase 29 service remained running and healthy after the experiment. Both
worker gates were explicitly disarmed. Phase 28 was stopped normally to load
the filtered allocator probe; its evidence and container were retained.

The previous Phase 27 process was manually terminated by the user. Its normal
exit is therefore not evidence of an injection failure or service crash.

## v3 safety changes

Phase 27 shifted the QMM output base backward by 1,024 bytes across:

```text
512 bytes of residual allocator padding
+ 512 bytes of declared residual storage
```

The v3 runtime now backs up the complete crossed prefix before the real QMM
Run. After the Run it restores and readback-verifies the allocator padding
before any downstream consumer executes. It preserves only the declared
residual overlap until the exact next AddRms consumer completes, then restores
and readback-verifies the complete residual.

The runtime also adds a bounded `max_hits` control in `[1,16]`. The default is
still one. A new hit cannot be claimed while the prior hit is pending its
consumer and recovery. The gate automatically closes when the configured
limit is reached. The expected consumer must be the immediately following
AddRms sequence and must exactly match device, residual address, and byte size.

## Phase 28 prefill allocator proof

The candidate was not copied blindly from Phase 27. A read-only live allocator
snapshot was repeated inside the Phase 28 worker processes before arming. The
rank-0 candidate was:

| Field | Value |
|---|---:|
| residual address | `0x12c1c235e200` |
| residual declared bytes / shape | 102400 / `[10,5120]` |
| residual allocation | `[0x12c1c235e200, 0x12c1c2377400)` |
| residual requested / actual bytes | 102400 / 102912 |
| QMM output address | `0x12c1c2377400` |
| QMM output declared bytes / shape | 512000 / `[10,25600]` |
| QMM output allocation | `[0x12c1c2377400, 0x12c1c23f4600)` |
| QMM requested / actual bytes | 512000 / 512512 |
| allocator padding | 512 bytes |
| output-base shift | 1024 bytes backward |
| declared residual overlap | 512 bytes |

The two active allocation blocks shared the exact boundary
`0x12c1c2377400` on both ranks. The injection still targeted rank 0 only.

## Phase 28 cross-request runtime evidence

The underflow JSONL contained exactly nine events:

```text
prepared            3
applied             3
consumer_restored   3
```

Only PID 772 emitted events. The three completed records were:

| Hit | Run index | Producer -> consumer | Changed residual bytes |
|---:|---:|---:|---:|
| 1 | 257 | 4227 -> 4228 | 510 |
| 2 | 321 | 5251 -> 5252 | 510 |
| 3 | 385 | 6275 -> 6276 | 510 |

For all three hits:

```text
real QMM Run status                       0
all synchronization/copy/allocation APIs 0
output restored byte-exact                true
allocator padding restored byte-exact     true
residual restored byte-exact              true
consumer sequence == producer + 1         true
final outcome                             QMM_OUTPUT_BASE_UNDERFLOW_
                                          RESIDUAL_STOMP_PROXY_PROVEN
```

Rank 1 emitted no underflow event.

## Phase 29 single-request allocator proof

The first decode candidate was rejected because an idle allocator query showed
that its former addresses had already been freed and coalesced. The read-only
probe was then extended with exact byte-size filters and the service restarted.
In the new worker processes it captured only:

```text
residual storage bytes   10240, shape [1,5120]
QMM output bytes         51200, shape [1,25600]
writer                   QMM
```

Eight live captures per rank showed the same address pattern. The exact pair
below occurred at filtered hits 3, 5, and 7 in the same fixed request:

| Field | Value |
|---|---:|
| residual address | `0x12c1c231aa00` |
| residual declared span | `[0x12c1c231aa00, 0x12c1c231d200)` |
| residual active allocation | `[0x12c1c231aa00, 0x12c1c231d400)` |
| residual requested / actual bytes | 10240 / 10752 |
| QMM output address | `0x12c1c231d400` |
| QMM output active allocation | `[0x12c1c231d400, 0x12c1c2329e00)` |
| QMM requested / actual bytes | 51200 / 51712 |
| common active boundary | `0x12c1c231d400` |
| allocator padding | 512 bytes |
| output-base shift | 1024 bytes backward |
| declared residual overlap | 512 bytes |

Both blocks were `active_allocated` at all three selected occurrences on both
ranks. The injection remained restricted to logical device 0.

## Phase 29 single-request runtime evidence

The one fault request produced exactly nine underflow events, all from PID 772:

```text
prepared            3
applied             3
consumer_restored   3
```

| Hit | Run index | Producer -> consumer | Changed residual bytes |
|---:|---:|---:|---:|
| 1 | 898 | 2309 -> 2310 | 510 |
| 2 | 900 | 2313 -> 2314 | 434 |
| 3 | 902 | 2317 -> 2318 | 282 |

Every status field was zero. Each record had exact normal-output, allocator-
padding, and residual restoration, and each consumer sequence was exactly the
producer sequence plus one. Rank 1 emitted no underflow event.

## E2E responses

Every request used:

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

All Phase 28 baseline/hit/recovery responses and the Phase 29 baseline,
read-only allocator request, single-request three-hit response, and disarmed
recovery returned:

```text
text
" Deterministic testing ensures consistent and reliable results"

token IDs
[34907,4532,7497,25351,12966,323,14720,3059]
```

Thus response divergence was not observed in either stage. The negative token
result does not invalidate the storage-level hits: every real QMM changed live
residual bytes, the exact next real AddRms consumed that residual, and all
protected regions were recovered exactly.

## Interpretation boundary

The exact prefill pair occurs once per fixed request; Phase 28 therefore proves
only cumulative cross-request multi-hit. Phase 29 uses a separately proven
decode pair that recurs three times in one request and proves single-request
multi-hit.

Neither stage proves natural QMM OOB behavior, the original historical physical
root cause, or token divergence. The framework supplied a controlled output-
base error; the real QMM performed the cross-allocation write.

## Evidence

Remote evidence:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_qmm_underflow_v3/
  qmm_underflow_20260818_phase28_qmm_underflow_multihit_v3.jsonl
  evidence_phase28/
    baseline.json
    allocator_query_v2.json
    pre_fault.json
    fault_3hit.json
    fault_hit2.json
    fault_hit3.json
    recovery.json
    underflow_final_query.json

  qmm_underflow_20260818_phase29_qmm_underflow_single_request_v3.jsonl
  evidence_phase29/
    baseline.json
    decode_allocator_query.json
    fault_single_request_3hit.json
    recovery.json
    underflow_final_query.json
```

The corresponding local evidence directory is excluded from Git.
