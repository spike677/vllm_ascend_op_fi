# Fault26 FIA writer Phase C: lossless writable-address substitution

Date: 2026-08-18

## Result

`FIA_LOSSLESS_WRITABLE_ADDRESS_SUBSTITUTION_PROVEN`

The experiment proved that the formal `attentionOut` tensor of the real
`aclnnFusedInferAttentionScoreV3` path can be rebound to a temporary device
buffer. The real FIA kernel wrote the temporary buffer, the wrapper copied the
result back to the original output storage, and the model response remained
exactly equal to the unarmed baseline.

This is Phase C only. No residual address was supplied to the writer library,
and no residual storage was modified.

## Scope and gates

- Container:
  `fault26-qwen3-32b-readonly-inventory-20260818_phase17_fia_lossless_v2`
- Service: `127.0.0.1:18028`
- Model: `qwen3-32b-w8a8-fault26`
- Runtime: `fault26_fia_writer_proxy_v2`
- Developer gate: `VLLM_SERVER_DEV_MODE=1`
- Writer gate: `FAULT26_FIA_WRITER_PROXY_ALLOWED=1`
- Dynamic default: disabled
- Target: device/rank 0, output bytes 81,920, occurrence 1, max hits 1
- API: `aclnnFusedInferAttentionScoreV3`
- Mode: `lossless_output_substitution`

The custom-opapi proxy was built in AddRms-only mode. It did not export FIA or
ReduceSum. This left a single FIA proxy layer; all 2,048 FIA execute events in
the unarmed request had `depth=1`.

## Fixed E2E request

```json
{
  "model": "qwen3-32b-w8a8-fault26",
  "prompt": "Explain why deterministic testing matters in one sentence.",
  "max_tokens": 8,
  "temperature": 0,
  "seed": 123
}
```

Unarmed baseline, armed lossless request, and disarmed recovery all returned:

```text
 Deterministic testing ensures consistent and reliable results
```

Each request returned HTTP 200 with eight completion tokens.

## L4 evidence

The single matching record reported:

```text
session                    2026081807
device                     0
output bytes               81920
dtype                      27
format                     47
original address           0x12c1c2393000
temporary address          0x12c100200000
real execute status        0
stream sync status         0
D2D copy status            0
temporary D2H status       0
original D2H status        0
destroy tensor status      0
device free status         0
temporary hash             0xaaa6c5ae06721aa0
original hash              0xaaa6c5ae06721aa0
readback equal             true
pass                       true
kernel wrote replacement   true
residual modified          false
```

The writer log contained exactly one substitution record and exactly one
`"pass":true` record. After the experiment both workers reported
`armed=false`, and `/health` returned 200.

Remote evidence:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_fia_writer_proxy_v2/
  fia_writer_20260818_phase17_fia_lossless_v2.jsonl
  inventory_20260818_phase17_fia_lossless_v2.jsonl
```

## Rejected phase14 result

The earlier phase14 armed response changed, but its writer log contained no
candidate, substitution, or PASS record. Investigation found two FIA proxy
layers: the same source had been exported by both `libopapi.so` and
`libcust_opapi.so`. The second GWS callback overwrote the first callback's
pending temporary record, so the real kernel could write the temporary without
the required copy-back evidence. That response is an instrumentation defect,
not a Fault26 result.

The v2 build fixed this by:

1. restricting the custom-opapi proxy to AddRms exports;
2. retaining FIA only in the main opapi proxy;
3. adding a fail-safe that does not overwrite an active GWS record;
4. recording enabled GWS candidates with real device and tensor metadata.

## Evidence boundary

Proven:

- FIA is a real writer-capable operator in the live model path;
- its real writable output binding can be substituted;
- the real kernel writes a distinct temporary device address;
- exact data and model behavior are preserved after copy-back;
- the mechanism is gated, single-hit, auditable, and recoverable.

Not proven by this phase:

- FIA wrote a live residual;
- a residual consumer observed an FIA-written corruption;
- a real FIA kernel performed an out-of-bounds write;
- the historical Fault26 physical root cause was reproduced.

Phase D may start only from this PASS and must retain the separate name
`FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY` until the real FIA write window,
victim change, AddRms consumption, model propagation, and recovery are all
independently demonstrated.
