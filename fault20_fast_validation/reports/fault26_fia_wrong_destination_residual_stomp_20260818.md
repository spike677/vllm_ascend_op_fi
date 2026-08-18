# Fault26 FIA wrong-destination residual stomp proxy validation

Date: 2026-08-18

## Result

`FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY_PROVEN`

The model-level experiment proved this chain:

```text
live residual produced by AddRmsNormBias
  -> FIA attentionOut binding changed to the residual address
  -> real aclnnFusedInferAttentionScoreV3 kernel wrote that address
  -> victim hash changed during the real FIA execute window
  -> the FIA result was copied to the original FIA output storage
  -> next AddRmsNormBias.input.x2 matched and consumed the victim
  -> model tokens diverged
  -> victim storage was restored byte-for-byte after the consumer
  -> disarmed request returned exactly to baseline
```

This is a bounded wrong-destination proxy. It does not claim that the FIA kernel
naturally computed an out-of-bounds extent, and it does not prove the historical
physical root cause.

## Environment

- Container:
  `fault26-qwen3-32b-readonly-inventory-20260818_phase18_fia_writer_v3`
- Service: `127.0.0.1:18028`
- Model: `qwen3-32b-w8a8-fault26`
- Model path: `/AIdata/haoran/Qwen3-32B-w8a8-nopdmix`
- Runtime: `fault26_fia_writer_proxy_v3`
- Hardware: two Ascend 910B4 devices, host NPUs 2 and 5
- Developer gate: enabled
- Dynamic default: disabled
- Target rank/device: 0
- Target FIA output: 81,920 bytes
- Target residual: 102,400 bytes
- Victim offset: 0
- Occurrence/max hits: 1/1

The output span was fully inside the declared residual allocation:

```text
0 + 81920 <= 102400
```

No physical OOB was performed.

## Fixed request and E2E responses

```json
{
  "model": "qwen3-32b-w8a8-fault26",
  "prompt": "Explain why deterministic testing matters in one sentence.",
  "max_tokens": 8,
  "temperature": 0,
  "seed": 123
}
```

Baseline and disarmed recovery:

```text
 Deterministic testing ensures consistent and reliable results
```

Fault request:

```text
 Determinant is a value that can be
```

All three requests returned HTTP 200 with eight completion tokens. The service
remained healthy and returned `/health` 200 after recovery.

## Real-writer evidence

Applied record:

```text
session                       2026081809
device                        0
matching index                1
original FIA output address   0x12c1c23ac200
victim residual address       0x12c1c232be00
FIA output bytes              81920
victim bytes                  102400
victim producer sequence      2178
real FIA execute status       0
pre/post stream sync          0 / 0
host backup/D2H               0 / 0
victim readback               0
FIA output copy/readback      0 / 0
replacement tensor destroy    0
victim hash before            0x5b69b135344ac7b2
victim hash after             0x6076d80e76ce4682
victim changed                true
output copy equal             true
real kernel wrote victim      true
applied pass                  true
```

Consumer and restore record:

```text
consumer matched              true
consumer executed             true
AddRms execute status         0
restore H2D/readback          0 / 0
host free                     0
restored victim hash          0x5b69b135344ac7b2
restore pass                  true
```

The log contained exactly one applied record and one restored record.

Remote evidence:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_fia_writer_proxy_v3/
  fia_writer_20260818_phase18_fia_writer_v3.jsonl
  inventory_20260818_phase18_fia_writer_v3.jsonl
```

## Why the writer attribution is valid

The injection framework did not copy or XOR the faulty value into the victim.
Before FIA it only:

1. selected a residual produced after the current arm epoch;
2. backed up its bytes;
3. built the FIA output tensor metadata over the victim address.

The victim changed between the synchronized pre-FIA and post-FIA observations,
and the real FIA execute returned success. The exact kernel-produced bytes were
then copied from the victim span to the original FIA output, preserving the
normal attention-output path while leaving the residual corrupted for its real
consumer. This separates the writer from the host-side residual mutation used
by earlier propagation-only proxies.

## Evidence boundary

Proven:

- the selected residual was live and produced after arm;
- the real FIA writable binding was redirected within the victim allocation;
- the real FIA kernel changed the victim during its execute window;
- the next real AddRms consumed the same victim storage;
- the fault caused token-level response divergence;
- the victim was restored exactly;
- disarm produced exact E2E recovery and a healthy service.

Not proven:

- a naturally occurring FIA bug changes the output pointer;
- the FIA kernel naturally writes beyond its declared output extent;
- allocator adjacency caused the historical issue;
- the historical Fault26 physical root cause has been reproduced.

The correct capability name is therefore
`FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY_PROVEN`, or at the generic layer
`OPERATOR_CAUSED_RESIDUAL_STOMP_PROXY_PROVEN`. `REAL_OOB_STOMP` remains a
separate small-harness objective.
