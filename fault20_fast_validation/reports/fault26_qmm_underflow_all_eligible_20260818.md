# Fault26 QMM allocation-boundary all-eligible-hit validation

Date: 2026-08-18

## Result

Phase 30 proved a request-scoped all-hit mode for every decode QMM invocation
whose exact live allocation pair had been proven adjacent in the same worker
process:

```text
ALL_ELIGIBLE_QMM_HITS_PROVEN
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN x231
MODEL_RESPONSE_DIVERGENCE_PROVEN
POST_FAULT_EXACT_RECOVERY_PROVEN
```

One fixed request completed 231 real QMM boundary-crossing writes on rank 0.
The response diverged from baseline at completion token 7. After disarm, the
next request returned byte-for-byte to the baseline text and token IDs.

This is the persistent operator-fault experiment. It supersedes the earlier
single-hit and three-hit experiments for model-output observability, while
retaining their exact-address and live-allocation safety requirements.

## Environment

```text
model             Qwen3-32B-w8a8-nopdmix
served model      qwen3-32b-w8a8-fault26
CANN              8.5.1
vLLM              0.18.0
TP                2
mode              eager, VLLM_SERVER_DEV_MODE=1
container         fault26-qwen3-32b-readonly-inventory-
                  20260818_phase30_qmm_underflow_all_eligible_v4
port              18028, bound to 127.0.0.1
runtime           fault26_qmm_underflow_v4
target rank       logical device 0 / PID 772
non-target rank   logical device 1 / PID 773
```

The service remained running and healthy after recovery. Both worker gates
were explicitly disarmed.

## All-hit definition

`all-hit` does not mean writing through every unexamined QMM output address.
The validated scope was:

```text
API             aclnnQuantMatmulWeightNz
device          logical device 0
residual        BF16 [1,5120], 10240 bytes
QMM output      BF16 [1,25600], 51200 bytes
allocation      both blocks active and exactly adjacent
shift           output base - 1024 bytes
padding         512 bytes
residual overlap 512 bytes
request scope   arm immediately before one controlled request
termination     explicit disarm immediately after request
hard limit      4096 hits
```

All calls matching any proven address pair were claimed until the request
ended. The hard limit was not reached.

Calls with the same tensor sizes but non-adjacent active allocations were
excluded. Other QMM shapes and APIs were not silently reclassified as safe.

## Fresh live allocator proof

Phase 30 repeated the filtered read-only allocator capture after starting the
v4 worker processes. Eight captures per rank identified three exact adjacent
pairs and one non-adjacent combination.

Included pairs:

| Residual address | QMM output / common boundary | Residual block | Output block |
|---|---|---:|---:|
| `0x12c1c2315600` | `0x12c1c2318000` | 10752B | 51712B |
| `0x12c1c2318000` | `0x12c1c231aa00` | 10752B | 51712B |
| `0x12c1c231aa00` | `0x12c1c231d400` | 10752B | 51712B |

For each pair:

```text
residual requested size   10240 bytes
residual actual block     10752 bytes
QMM output requested      51200 bytes
QMM output actual block   51712 bytes
both block states         active_allocated
residual block end        == QMM output block begin
```

The observed non-adjacent pair
`residual=0x12c1c2318000, output=0x12c1c231d400` was excluded because its
residual block ended at `0x12c1c231aa00`, not at the output block begin.

## v4 control and recovery behavior

The worker RPC accepts at most eight fully specified target pairs. Every pair
must independently satisfy:

```text
output address == output allocation begin
residual allocation end == output allocation begin
residual declared span is inside the residual block
shift crosses allocator padding into declared residual storage
replacement output begins inside the proven residual span
```

The native runtime reparses and matches this list fail-closed on every QMM
candidate. A prior hit must complete its exact next-AddRms consumer and recovery
before another hit can be claimed.

Every hit performs:

```text
backup complete crossed prefix and residual
  -> real QMM executes with output base shifted backward
  -> restore and verify allocator padding
  -> restore and verify normal QMM output
  -> exact next AddRms consumes corrupted residual
  -> restore and verify residual
  -> free host backup
```

The default remains one hit. Phase 30 explicitly set `all_hits=true` and a
4096-hit hard limit.

## Runtime evidence

The underflow JSONL contained exactly 693 events:

```text
prepared            231
applied             231
consumer_restored   231
failed/aborted      0
```

Only PID 772 emitted underflow events. Address-pair hit counts were:

| Residual -> output | Hits |
|---|---:|
| `0x12c1c2315600 -> 0x12c1c2318000` | 7 |
| `0x12c1c2318000 -> 0x12c1c231aa00` | 7 |
| `0x12c1c231aa00 -> 0x12c1c231d400` | 217 |
| total | 231 |

Additional aggregate evidence:

```text
hit numbers                         1..231
QMM run indices                     896..1342
changed residual bytes per hit      313..512
sum of changed-byte observations    105480
all all_hits flags                  true
hard limit reached                  false
all ACL/copy/sync/free statuses     0
all output restorations exact       true
all padding restorations exact      true
all residual restorations exact     true
all consumers producer + 1          true
rank 1 hits                         0
```

## E2E response change

All requests used:

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

Baseline:

```text
" Deterministic testing ensures consistent and reliable results"
[34907,4532,7497,25351,12966,323,14720,3059]
```

All-eligible-hit response:

```text
" Deterministic testing ensures consistent and predictable outcomes"
[34907,4532,7497,25351,12966,323,44107,19554]
```

The first six completion tokens were identical. Divergence began at completion
token 7:

```text
baseline tail   [14720,3059] -> "reliable results"
fault tail      [44107,19554] -> "predictable outcomes"
```

The request returned HTTP 200 and the worker remained healthy.

After explicit disarm, recovery returned exactly to the baseline text and token
IDs. `/health` returned 200.

## Evidence boundary

Proven:

```text
all eligible, live-adjacent decode calls were repeatedly injected
231 real QMM boundary-crossing writes occurred in one request
the corrupted residuals reached their exact next real AddRms consumers
the persistent operator-level simulation changed model tokens and text
disarm restored exact deterministic behavior
```

Not proven:

```text
the healthy QMM implementation naturally performs an OOB write
all QMM calls of every shape have adjacent victim allocations
the historical physical defect had this exact address arithmetic
the excluded non-adjacent or unobserved calls are safe to underflow
```

## Evidence locations

Remote:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_qmm_underflow_v4/
  qmm_underflow_20260818_phase30_qmm_underflow_all_eligible_v4.jsonl
  evidence_phase30/
    baseline.json
    decode_allocator_query.json
    all_hit_arm.json
    fault_all_eligible.json
    all_hit_disarm.json
    all_hit_query.json
    recovery.json
```

Local validation evidence is under `.fault26_evidence/phase30/` and excluded
from Git.
