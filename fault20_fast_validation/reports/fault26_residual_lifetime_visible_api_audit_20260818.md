# Fault26 residual-lifetime visible-API audit

Date: 2026-08-18

## Result

The follow-up read-only audit passed for the storage that is actually used as
the Qwen3-32B residual in the tested path:

```text
FAULT26_RESIDUAL_LIFETIME_VISIBLE_API_AUDIT_PASS
FAULT26_FIXED_REQUEST_REDUCESUM_NOT_OBSERVED
FAULT26_REDUCESUM_TO_RESIDUAL_LINEAGE_NOT_PROVEN
```

No fault injection or device-memory write was performed. The result is scoped
to CANN 8.5.1, the current Qwen3-32B-w8a8-nopdmix service, and the three
requests below.

## Proven storage lineage

The typed v18 mapping proves this current path:

```text
previous aclnnAddRmsNormBias output.x
  == live residual storage
  == next aclnnAddRmsNormBias input.x2
```

It does not prove:

```text
ReduceSum output
  -> view / copy / cast
  -> residual storage
```

The reason is not merely a missing metadata getter. No external
`aclnnReduceSum` execute occurred inside any of the three armed request
sessions on either rank. The only ReduceSum execute events in the full process
trace occurred during early worker initialization, before the model request
sessions, and have no proven address/lifetime relation to the residual.

## Request sweep

All counts below are per TP rank. `FIA in live intervals` means a visible FIA
execute entry occurred between two adjacent synchronized residual-hash
checkpoints; it is not an individually bracketed FIA before/after pair.

| Session | Workload | Complete live generations | FIA in live intervals | ReduceSum in session | Changed live intervals |
|---|---|---:|---:|---:|---:|
| `2026081803` | 9 prompt tokens, 4 generated tokens | 508 / 512 | 252 | 0 | 0 |
| `2026081804` | 353 prompt tokens, 4 generated tokens | 508 / 512 | 252 | 0 | 0 |
| `2026081805` | two concurrent 10+8-token requests | 1778 / 1792 | 882 | 0 | 0 |

The requests all returned HTTP 200:

```text
2026081803  " - Page 2"
2026081804  "1. What is"
2026081805A "1.3.1.1."
2026081805B "1.1.1.1."
```

Each session was explicitly armed, then immediately disarmed. The final query
reported `armed=false`, `device_memory_modified=false` for worker namespace
PIDs 218 and 219.

## What the v18 checkpoints actually cover

The earlier report emphasized the individually bracketed QMM `P1 -> P2`
window. Re-analysis of the complete event stream shows that adjacent hashes
also bound the intervening visible L4 work:

```text
P0 -> P1  AscendQuantV3, QMM execute entry
P1 -> P2  real QMM Run
P2 -> P1  AscendQuantV3, FIA, optional InplaceCopy, next QMM entry
P2 -> P3  Matmul
P3 -> P4  AddRms execute boundary
```

For every complete generation in the table, all adjacent checkpoint hashes
were equal. Therefore the strongest valid statement is:

> No modification of the monitored live residual storage persisted across an
> adjacent synchronized checkpoint interval in these requests, including the
> grouped intervals that contained visible FIA executions.

This is broader than a QMM-only audit, but weaker than individually sampling
before and after every FIA or every L4 Run. In particular, the current spacing
cannot observe a transient modification that is both introduced and restored
inside the same interval.

## Evidence boundary

| Claim | Status |
|---|---|
| A live residual changed in these three no-injection sessions | Not observed |
| A persistent residual change crossed a grouped FIA-containing interval | Not observed |
| One specific FIA call was individually proven not to write the residual | Not proven by the grouped checkpoints |
| ReduceSum produced this residual storage | Not proven; external ReduceSum was absent from the requests |
| QMM or FIA was the historical illegal writer | Not proven |
| Wrong-pointer, allocator-padding, workspace, or kernel-temporary paths are impossible | Not proven |
| No shape/build/hardware combination can trigger the historical defect | Not proven |
| The historical defect is fixed | Not proven |

Because there is no first changing interval in these sessions, execution-order
bisection cannot identify an illegal writer from this data. A bisection becomes
meaningful only after a request naturally produces a changed live hash.

## Correct next gate

To investigate a historical ReduceSum victim, first select a model/build/input
whose forward request actually exposes external `aclnnReduceSum`. Then:

1. capture the typed ReduceSum GWS input/output metadata and prove the dynamic
   Run slots;
2. prove exact address/offset/stride aliasing through any view, copy, or cast to
   the later residual entry;
3. define the live interval from the last legal ReduceSum write to residual
   consumption;
4. sample coarse synchronized intervals and bisect only after the first
   unexpected change exists.

For the current Qwen3 path, a more granular audit can instead bracket each FIA
against the already proven AddRms residual storage. That would isolate FIA from
the other operations in `P2 -> P1`, but it would still audit AddRms-backed
residual storage, not a nonexistent request-time ReduceSum output.

A controlled `FIA_TRIGGERED_REDUCESUM_STOMP_PROXY` remains an effect-level
diagnostic primitive only when the ReduceSum victim is independently proven.
It cannot be used to manufacture the missing lineage or to claim a natural FIA
kernel out-of-bounds write.

## Runtime state and evidence

After all sessions:

```text
health endpoint       HTTP 200
server host PID       2978416
TP worker host PIDs   2982527 / 2982528
TP namespace PIDs     218 / 219
audit armed            false / false
device memory modified false / false
```

The vLLM process remains healthy on `127.0.0.1:18028`. The original Docker
object is no longer registered with the daemon, although the process remains
alive in its old Docker cgroup. Do not rely on the former container name for a
later stop/restart operation.

Local ignored evidence:

```text
.fault26_evidence/20260818_phase13_v18/
  inventory_after_shape_sweep_20260818.jsonl
  analysis_residual_lifetime_visible_apis_2026081803.json
  analysis_residual_lifetime_visible_apis_2026081804.json
  analysis_residual_lifetime_visible_apis_2026081805.json
```

Analyzer:

```text
fault20_fast_validation/source/fault26_readonly_trace_v1/
  analyze_fault26_residual_lifetime_coverage.py
```
