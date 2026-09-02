# Fault Validation Manifest

This directory is historically named `fault20_fast_validation`, but it now
contains multiple local, developer-mode fault-injection implementations and
evidence sets.

## Fault20 — QMM buffer substitution / semantic slot proof

Status: PASS

Source:

- `source/l4_buffer_substitute_allhit_v1/`
- `source/l4_metadata_lifecycle_proxy/`

Evidence:

- `msprobe_dump/`
- `evidence/fault20_v17_msprobe_response_summary.md`

Core labels:

- `FAULT20_V17_BUNDLE_VERIFY_PASS`
- `FAULT20_V17_SEMANTIC_SLOT_MSPROBE_ADAPTATION_PASS`
- `MSPROBE_ACTIVE_SEMANTIC_INJECTION_PROVEN`

Boundary:

- Fault20 validates QMM input buffer substitution and effective-precision
  degradation under explicit developer-mode control.
- It is not a production runtime configuration.

## Fault26 — FIA wrong-destination residual stomp

Status: PASS

Source:

- `source/fault26_fia_writer_proxy_v1/`
- `source/fault26_readonly_trace_v1/`

Evidence:

- `evidence/fault26_evidence_20260818.tar.gz`
- `evidence/fault26_sustained_dump_20260902/`

Core labels:

- `FAULT26_PHASE_D_BOUNDED_RELATION_GATED_ALLHIT8_PASS`
- `FAULT26_SUSTAINED_ALLHIT_OUTPUT_DEGRADATION_PASS`
- `FAULT26_SUSTAINED_ALLHIT_EARLY_EOS_COLLAPSE_OBSERVED`
- `FAULT26_SUSTAINED_ALLHIT_RECOVERY_PASS`

Boundary:

- Fault26 demonstrates proxy-induced FIA wrong-destination residual stomp after
  topology/relation gating.
- It does not prove that the vendor kernel naturally performs spontaneous OOB
  writes in a healthy run.

## Fault39 — MatMul / QMM non-finite numerical behavior

Status: PASS for the documented proxy and finite/non-finite validation cases.

Source:

- Reuses the L4 buffer/proxy framework under `source/`.

Evidence and reports:

- `reports/fault39_qmm_overflow_design.md`
- `reports/fault39_qmm_overflow_validation.md`
- `reports/fault39_float_matmul_path_inventory.md`
- `reports/fault39_float_matmul_finite_nonfinite_validation.md`
- `reports/fault39_float_matmul_cancellation_validation.md`
- `reports/fault39_float_matmul_output_nonfinite_proxy_validation.md`
- `reports/fault39_matmul_fault_reproduction_summary.md`
- `reports/fault39_a3_validation_handoff.md`

Boundary:

- Fault39 reports distinguish real numerical non-finite behavior from explicit
  output non-finite proxy modes.

## Fault41 — state/cache tail corruption

Status: DESIGN FEASIBLE / IMPLEMENTATION BLOCKED / EXPERIMENTAL

Fault family:

- `state/cache tail corruption`

Planned proxy names:

- `OP_TRIGGERED_STATE_CACHE_TAIL_CORRUPTION_PROXY`
- `KV_CACHE_POST_WRITE_TAIL_CORRUPTION_PROXY` after a concrete KV-cache slot is
  proven.

Source:

- No Fault41 source directory is present in this GitHub worktree as of this
  manifest update.

Reports:

- `reports/fault41/fault41_tail_column_poison_smoke_20260901.md`
- `reports/fault41/fault41_state_cache_tail_corruption_roadmap.md`

Core label:

- `FAULT41_LD_PRELOAD_SYMBOL_LOADED_BUT_NOT_INTERPOSED`

Current boundary:

- The exploratory LD_PRELOAD library was loaded in the worker, but the same-name
  symbol did not interpose the torch_npu/ATB dispatcher path.
- Fault41 should reuse the common L4 framework, but keep an independent adapter,
  control command, report, and evidence directory.
- The next milestone is read-only proof of the real cache writer, cache
  storage/slot metadata, and later attention consumer relation.
- Fault41 must not be marked PASS until cache identity, writer-consumer relation,
  lossless writeback, and model-level E2E are demonstrated.
