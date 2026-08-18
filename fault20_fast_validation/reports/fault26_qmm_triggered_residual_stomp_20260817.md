# Fault26 unified QMM-triggered residual stomp handoff

Date: 2026-08-17

## Result

The implementation now has one fault-root classification and one injection
primitive:

```text
root cause model     libopapi QMM implementation fault
faulty API           aclnnQuantMatmulWeightNz
primitive            QMM_TRIGGERED_RESIDUAL_STOMP_PROXY
attention / MLP      context only; never a fault-type branch
```

The previous v13 implementation modified `AddRmsNormBias.x2` immediately before
the consumer. It proved the real residual victim and produced token divergence,
but was not QMM-triggered. v15 moves the trigger to the real QMM
`OpExecCache::Run` path and preserves the exact next-consumer lifecycle.

## Environment

```text
host              123.60.231.86:2125
image             quay.io/ascend/vllm-ascend:v0.18.0rc1
model             /home/haoran/models/Qwen3-32B-w8a8-nopdmix
served model      qwen3-32b-w8a8-fault26
port              18026
TP / devices      2 / logical 0,1 (host NPU 2,5)
mode              eager + VLLM_SERVER_DEV_MODE=1
docker socket     /run/fault26-docker-20260817.sock
live container    fault26-qwen3-32b-readonly-inventory-20260817t1530z_phase10_qmm_stomp_v15
```

The live container was left running, healthy, disarmed, and with
`pending_count=0`.

## Read-only relation proof (v14c)

The correct correlation clock is the real L4 executor thread. Python hooks can
run ahead and must not be paired by nearest wall-clock timestamp.

Per rank:

```text
typed AddRms calls                  512
L4 AddRms executes                 512
QMM Runs                            762
QMM mapped to a next consumer       762 / 762
QMM after final consumer            0
consumer distribution               0 QMM: 4, 1 QMM: 254, 2 QMM: 254
residual output.x -> next x2         508 / 511
forward-boundary transitions         3 (explicitly rejected)
safe QMM candidates                  758
typed/L4 x2 agreement                512 / 512
typed/L4 output.x agreement          512 / 512
GWS/execute FIFO agreement           512 / 512
```

The audit result is:

```text
FAULT26_QMM_TO_NEXT_RESIDUAL_RELATION_READONLY_PASS
```

Analyzer:

```text
fault20_fast_validation/source/fault26_readonly_trace_v1/
  analyze_fault26_qmm_residual_link.py
```

## v15 implementation

The runtime tracks AddRms GWS records in executor FIFO order. After a real
AddRms execute returns, its `output.x` metadata becomes the current residual
candidate. After a real QMM Run returns, the runtime peeks the next queued
AddRms and permits a hit only if:

```text
developer/QMM-stomp gates enabled
target logical device matches
no prior pending hit
current residual is BF16, ND, contiguous, [1,5120], 10240 bytes
next AddRms.x2 has the same metadata
current residual address == next AddRms.x2 address
target occurrence and one-hit control match
```

The write is applied after QMM completion. The exact next AddRms sequence must
consume that address. The runtime then synchronizes, restores the original
bytes, performs D2H readback, and verifies the original hash. A mismatch causes
fail-safe restoration and no further injection.

Source:

```text
fault26_qmm_residual_stomp_api.h
fault26_qmm_residual_stomp.cpp
fault26_qmm_residual_stomp_control.cpp
build_fault26_qmm_residual_stomp.sh
```

## Staged E2E

### Gate disabled baseline

```text
HTTP              200
4-token IDs       [198,32313,11,279]
enabled           0
pending/hit       0/0
```

### Generation 1: lossless

```text
target                   device 0, occurrence 1, [1,5120]
QMM cache                0xaaae546d9fc0
victim                   0x12c1c2312c00, 10240 bytes
expected consumer seq    819
candidate/hit/restore    1/1/1
consumer matched         true
all runtime statuses     0
hash before              0xa8eb8378763a667e
hash injected            0xa8eb8378763a667e
hash restored            0xa8eb8378763a667e
4-token IDs              [198,32313,11,279]
```

The disarmed recovery request was token-exact.

### Generation 2: numeric stomp

```text
mode                     xor_bf16_sign
range                    offset 0, length 256 bytes
changed                  128 BF16 elements
QMM cache                0xaaae546d9fc0
victim                   0x12c1c2312c00, 10240 bytes
expected consumer seq    8771
candidate/hit/restore    1/1/1
consumer matched         true
all runtime statuses     0
hash before              0x5b7972f684d912fb
hash injected            0xc62d55a7e6f8effb
hash restored            0x5b7972f684d912fb
HTTP / health            200 / 200
```

The 32-token baseline, fault, and recovery token IDs were all:

```text
[198,32313,11,279,1196,3208,330,5291,17,21,11762,8265,8375,3263,
 6771,752,7071,700,1246,311,5889,382,5338,11,358,1184,311,3535,
 1128,419,1943,3363]
```

The unchanged token sequence does not invalidate the mechanism: the injected
hash differed, the exact next AddRms consumed the victim address, and the
original hash was restored. It does mean that model-output divergence was not
observed for this particular hit.

Runtime analyzer result:

```text
FAULT26_QMM_TRIGGERED_RESIDUAL_STOMP_RUNTIME_PASS
```

Analyzer:

```text
fault20_fast_validation/source/fault26_readonly_trace_v1/
  analyze_fault26_qmm_residual_stomp.py
```

## Evidence boundary

Completed:

```text
FAULT26_QMM_TO_NEXT_RESIDUAL_RELATION_READONLY_PASS
FAULT26_QMM_TRIGGERED_RESIDUAL_STOMP_RUNTIME_PASS
QMM_TRIGGERED_RESIDUAL_NUMERIC_CHANGE_PROVEN
NEXT_ADDRMS_CONSUMER_MATCH_PROVEN
RESIDUAL_STORAGE_RESTORE_PASS
RECOVERY_PASS
```

Not established by v15 generation 2:

```text
MODEL_OUTPUT_DIVERGENCE_REPRODUCED
QMM_REAL_OUT_OF_BOUNDS_WRITE_PROVEN
FIA_REAL_OUT_OF_BOUNDS_WRITE_PROVEN
physical historical root cause reproduced
```

The exact output classification for this hit is:

```text
TOKEN_DIVERGENCE_NOT_OBSERVED_THIS_HIT
```

The QMM trigger and next-consumer relation establish semantic timing and
liveness only. They do not establish that the QMM output allocation is
physically adjacent to the residual allocation. Physical adjacency requires a
separate live address-interval audit.

## Operational notes

- `api_info` numeric suffixes and cache pointers are process-local context, not
  stable fault selectors across restarts.
- A high `no_next_count` means the next AddRms GWS record was not yet present in
  the FIFO at those QMM callbacks; those calls are fail-open skips and perform
  no write.
- Forward boundaries are guarded by live address equality, not call-position
  guesses.
- The implementation never allocates device memory and never writes outside the
  validated residual storage.
- The control plane defaults to disabled and numeric mode is fixed to one hit.
