# Discovery and evaluation V0

## Scope

V0 composes the existing shared contracts and native evidence formats:

```text
qmm_run_trace
  -> ObservedFact
  -> capability manifest matching
  -> ranked Candidate
  -> lossless ValidationResult
  -> existing backend adapter
  -> baseline / fault / disarm / recovery
  -> EvaluationResult
  -> portable model capability profile
```

It does not add a native injector, product registry, RuntimeBinding registry,
controller, or production CLI.

## Recorded fixture check

Run from the repository root:

```powershell
$env:PYTHONPATH = "src"
python tools/run_recorded_discovery_evaluation.py
```

The tool only reads committed fixtures. It does not connect to, arm, or modify
an inference service. Output contains the Candidate, ranking, ValidationResult,
EvaluationResult, and generated profile.

The QMM response text and SHA-256 values come from
`fault20_fast_validation/evidence/fault20_v17_msprobe_response_summary.md`.
Raw native `run_trace.jsonl` remains excluded from Git, so the fixture records
only one representative successful hit and does not claim the canonical
all-hit total. Committed evidence has token pieces but not integer token IDs;
therefore `token_ids` remains null and divergence explicitly uses
`token_comparison_basis=token_pieces`.

## Fail-closed gates

Automatic lossless validation requires:

- a manifest-declared symbol and execution plane;
- a proven input activation role and runtime slot;
- complete semantic-slot proof;
- a capability that declares lossless validation and restore support.

Destructive evaluation is refused until `ValidationResult` confirms a passed
lossless attempt and exact recovery. Once fault execution starts, the runner
attempts disarm even when the backend raises.

## Live adapter boundary

The Python protocols accept an existing developer/backend harness for lossless
and baseline/fault/recovery runs. A live adapter must still provide explicit
request boundaries, developer-mode authorization, arm/disarm, evidence
collection, and service health checks. Those actions are not inferred from
native call indices and are not implemented as a second product control plane.

Current V0 is therefore fixture-level E2E for the orchestration and contracts.
A separate A2/A3 run is required before declaring live
`DISCOVERY_EVALUATION_V0_E2E_PASS`.
