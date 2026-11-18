# Source72 AddRmsNormBias rejection diagnostic

The two source72 healthy decodes completed with all 270 actual Native parents,
prepares and executes, zero faults, complete cleanup and three row-proven
projections per request. Inventory SHA256:
`ebba0a3a31d6161b61e293add84d5097eae1f19a6e998c43fd123e9d2057a469`.
The new embedding producer, residual clone and event291 no-chunk return are
observed and row-proven. Event292 AddRmsNormBias remains unproven.

Root's non-tensor extraction confirms that event292 x1/x2 match their current
node metadata, have row axis0 and equal request-row digests; no prior gamma
invalidation was recorded. The old trace does not contain frozen parameter
registration, `requires_grad` or a current owner check. It cannot determine
which existing gate rejected the call. No source72 field is retroactively
filled from defaults or guesses.

This increment adds **read-only diagnostics only** to that exact operator's
existing event. It does not alter `_lookup`, `_broadcast_parameter`, input
eligibility, the AddRms acceptance rule, numerical execution or Native calls.
Each event now contains:

```text
add_rms_row_diagnostic.before / .after
  complete, checks, failures
  first_input / second_input:
    current, record_present, checks, failures, node_id, row_axis, rows_sha256,
    actual/recorded_metadata, metadata_changed_fields,
    actual/recorded_write_epoch
  gamma / beta:
    registered, requires_grad, parameter_type,
    observed_model_parameter_owners, owner_object_id, owner_model_path,
    parameter_name, kind, actual/recorded_metadata,
    metadata_changed_fields, actual/recorded_write_epoch,
    broadcast_proven, checks, failures
add_rms_row_diagnostic.actual_rule_accepted_outputs
```

The diagnostics compute the current lookup/parameter predicates without
calling the evicting `_lookup`, registering a missing parameter, binding a
row, changing an epoch or retaining tensors. They preserve before/after
snapshots even when the unchanged rule subsequently evicts a stale proof.
Unsupported/failed diagnosis records `complete=false` and its error, while
the original operator still executes through the existing path exactly once.
Storage metadata and object identities are recorded; tensor values are not.

Nine new CPU contracts cover the unchanged positive path, missing frozen and
requires-grad parameter registrations, equal foreign gamma, replaced owner,
inference-mode alias write epoch, different request order, non-evicting reads
and detached report snapshots. Along with embedding and source-bound observer
regressions, 42 tests passed. This does not establish the production cause.

Root can freeze a new Python package and run the same controlled two-healthy,
zero-fault `providers_quant_inventory_probe.py --group-abi` case. Reuse the
unchanged source64 Native build `b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`;
no SDK rebuild is required. Extract the first AddRms event's before/after
`failures` and corresponding failed field only. Keep all 135 candidates per
request and every refusal. This diagnostic does not grant C1/C2/C3 and is not
public Core V1.1 acceptance; all new qualification flags remain false.
