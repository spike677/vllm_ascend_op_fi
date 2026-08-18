# Fault26 FIA actual-writer proxy

This developer-mode component provides two separately named validation modes
for the real `aclnnFusedInferAttentionScoreV3` path.

## Phase C: lossless writable-address substitution

```text
FIA attentionOut formal tensor
  -> same metadata, temporary device storage
  -> real FIA kernel execute
  -> D2D temporary output back to original output storage
  -> byte-for-byte D2H verification
```

This mode proves that the real FIA writable output binding can be substituted
without changing model behavior. It never accepts or modifies a residual.

## Phase D: bounded wrong-destination residual stomp proxy

```text
previous AddRmsNormBias.output.x
  -> tracked as a live residual
  -> arm establishes a new producer-sequence epoch
  -> a later FIA attentionOut is rebound inside that residual allocation
  -> residual is backed up before execute
  -> real FIA kernel writes the residual address
  -> FIA result is copied back to its original output storage
  -> next AddRmsNormBias.input.x2 must match the victim storage
  -> real AddRmsNormBias consumes the corrupted residual
  -> original residual bytes are restored and verified
```

This is an operator-caused, bounded wrong-destination proxy. The injection code
changes the writable binding, but the unauthorized data write itself is issued
by the real FIA kernel. It is not an out-of-bounds kernel and must not be named
`REAL_OOB_STOMP` or a reproduction of the historical physical root cause.

## Gates and fail-closed checks

The component is inert unless all of these are true:

- `VLLM_SERVER_DEV_MODE=1` at process start;
- `FAULT26_FIA_WRITER_PROXY_ALLOWED=1` at process start;
- `FAULT26_FIA_WRITER_PROXY_ENABLED=1` is explicitly armed later;
- mode, device, byte sizes, occurrence, and maximum bytes match;
- FIA output and residual are contiguous, offset-zero storage;
- output dtype equals residual dtype;
- the entire FIA output span fits inside the declared residual storage;
- the selected residual was produced after the current arm session began;
- the FIA output and residual do not already alias;
- the next AddRms `x2` exactly matches the victim storage;
- the next AddRms `x_out` does not alias the victim.

One matching call is claimed and the C++ writer gate automatically disarms.
Both workers must still be explicitly disarmed through the collective RPC after
the request because a non-target rank cannot claim the target-device hit.

The wrong-destination path records and checks:

- victim backup and pre-write hash;
- real FIA execute status;
- victim post-write hash and `victim_changed`;
- original FIA output copy-back and readback equality;
- exact next-AddRms consumer relation;
- consumer execute status;
- victim restoration and restored hash;
- single-hit and E2E recovery evidence.

## Control methods

The worker extension exposes:

- `fault26_arm_fia_lossless_output_substitution`
- `fault26_arm_fia_wrong_destination_residual_stomp`
- `fault26_disarm_fia_lossless_output_substitution`
- `fault26_query_fia_lossless_output_substitution`

The disarm/query names retain the Phase C name for compatibility; their returned
`mode` identifies the active configuration.

## Validated results

On 2026-08-18, Qwen3-32B-w8a8-nopdmix on two Ascend 910B4 devices proved:

- `FIA_LOSSLESS_WRITABLE_ADDRESS_SUBSTITUTION_PROVEN`
- `FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY_PROVEN`

See:

- `reports/fault26_fia_writer_lossless_phase_c_20260818.md`
- `reports/fault26_fia_wrong_destination_residual_stomp_20260818.md`
