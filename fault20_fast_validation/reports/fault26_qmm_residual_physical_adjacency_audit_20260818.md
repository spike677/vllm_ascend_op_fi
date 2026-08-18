# Fault26 QMM/residual read-only address-interval audit

Date: 2026-08-18

## Result

The runtime audit separates two independent facts:

```text
QMM -> next AddRms.x2 lifecycle relation     proven
QMM output -> residual physical adjacency   not observed in declared spans
```

Final classification:

```text
FAULT26_QMM_RESIDUAL_DECLARED_INTERVAL_AUDIT_PASS
QMM_OUTPUT_RESIDUAL_NOT_ADJACENT_IN_DECLARED_SPANS
```

This was a read-only experiment. The v16c runtime contains no device-memory
copy, allocation, memset, or mutation path. It records typed metadata, live
`OpExecCache::Run` addresses, declared storage sizes, workspace spans, and the
real L4 execution order.

## Environment and workload

```text
model          Qwen3-32B-w8a8-nopdmix
CANN           8.5.1
vLLM           0.18.0
TP             2, logical devices 0 and 1
mode           eager, VLLM_SERVER_DEV_MODE=1
container      fault26-qwen3-32b-readonly-inventory-
               20260818_phase11_qmm_adjacency_v16c
port           18026
```

Two identical fixed baseline requests completed successfully. Both returned:

```text
text       "\nOkay, the"
token IDs  [198, 32313, 11, 279]
health     200 after both requests
```

The full audit window includes startup/warmup and those two requests.

## Capture chain

CANN 8.5.1 maps both a monolithic `libopapi` and the split
`libopapi_nn.so`. The QMM typed front end is resolved from the split library by
specific-handle lookup, so `LD_LIBRARY_PATH` or a global monolithic preload is
not sufficient. v16c mounts a same-SONAME, read-only metadata proxy at the
container's split-library path and keeps the real implementation as
`libopapi_nn_real.so`.

The accepted chain is:

```text
typed aclnnQuantMatmulWeightNzGetWorkspaceSize
  -> ten formal aclTensor metadata records
  -> AddParamToBuf formal-pointer order
  -> AddOpExecCache raw cache registration
  -> OpExecCache::Run live address vector
  -> exact compact-storage slot proof
  -> QMM output/input/workspace declared spans

typed AddRms GWS
  -> executor FIFO
  -> real AddRms execute order
  -> prior output.x == next input.x2 lifecycle gate
```

No cold activation address is treated as a live Run address.

## Coverage

Per rank:

```text
AddRms GWS / executes               1152 / 1152
QMM Runs                            1719
QMM output slot proven              1719 / 1719
QMM mapped to next AddRms           1719 / 1719
QMM after final consumer            0
safe residual-chain candidates      1707
forward-boundary rejected QMM       12
consumer QMM distribution           0:6, 1:573, 2:573
GWS/executor FIFO complete          true
```

The two ranks produced the same coverage and output-gap distribution.

## QMM output versus residual

For all 1,707 safe samples on each rank:

```text
ordering                 residual_end < qmm_output_begin
tail adjacency           0
reverse adjacency        0
overlap                  0
minimum separation       512 bytes
median separation        16,384 bytes
maximum separation       130,154,496 bytes
```

Therefore none of the observed declared intervals supports:

```text
QMM_output_end == residual_begin
```

The observed order is the reverse: residual storage ends before QMM output
storage begins. A normal forward overrun from the QMM output tail cannot reach
that earlier residual interval. The closest observed case still has a 512-byte
declared gap; it is not exact reverse adjacency either.

## Other QMM spans versus residual

Known declared spans also had zero exact adjacency and zero overlap:

| QMM span | Closest observed separation | Direction summary |
|---|---:|---|
| `x1` | 512 B | mixed; residual may be before or after x1 |
| `x2` / weight | 971,997,184 B | mixed, generally very distant |
| `x2_scale` | 6,042,624 B | scale ends before residual begins |
| `bias` | 6,145,536 B | bias ends before residual begins |
| workspace | 52,494,336 B | residual ends before workspace begins |

Optional absent formals (`x1_scale`, offsets, `y_scale`) do not provide a
comparable live span and are not counted as evidence.

## Evidence boundary

This audit proves the relation between live Run addresses and **declared**
tensor/workspace byte ranges. It does not expose the caching allocator's
reserved block size, padding, guard region, or internal QMM temporary buffers.
Consequently:

```text
declared output-tail adjacency hypothesis     rejected for observed calls
allocator-block adjacency                     unknown
kernel-internal temporary-buffer adjacency    unknown
real QMM out-of-bounds write                   not proven
historical physical root cause                 not reproduced
```

The v15 `QMM_TRIGGERED_RESIDUAL_STOMP_PROXY` remains correctly classified as a
controlled residual-corruption propagation proxy. It must not be described as
a physical output-tail OOB reproduction.

## Artifacts

Source:

```text
fault26_trace_preload.cpp
libopapi_metadata_proxy.cpp
analyze_fault26_qmm_residual_adjacency.py
analyze_fault26_qmm_residual_adjacency_offline.py
```

Remote evidence:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_qmm_adjacency_v16c/
  inventory_20260818_phase11_qmm_adjacency_v16c.jsonl
  qmm_20260818_phase11_qmm_adjacency_v16c.jsonl
  analysis.json
  analysis_offline.json
  response_1.json
  response_2.json
```

The same evidence is archived locally under
`.fault26_evidence/20260818_phase11_v16c/` and is intentionally excluded from
Git. The v16c service was left running and healthy on port 18026.
