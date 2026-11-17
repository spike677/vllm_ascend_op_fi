# Independent evidence audit

These tools run on CPU and read existing evidence. They neither initialize an
NPU nor grant C1/C2/C3 or production qualification. All raw captures and complete
audit reports remain on 124; publish reviewed summaries and hashes only.

## Raw statistics

`audit_statistics.py` accepts only the original B/F `dump.json`, `stack.json`,
`construct.json`, capture configuration and collector identity. It has no Native
log, target, expected-operator, full-tensor or adjustable-tolerance argument.
Example (source root on `PYTHONPATH`):

```sh
python experiments/c2c3_closure_20261117/audit_statistics.py \
  --baseline BASELINE/observation/step0/rank0 \
  --fault FAULT/observation/step0/rank0 \
  --baseline-config BASELINE/config.json --fault-config FAULT/config.json \
  --baseline-collector BASELINE/collector_identity.json \
  --fault-collector FAULT/collector_identity.json --out NEW_ANALYSIS_DIRECTORY
```

The fixed rule is `abs(F-B) > 1e-8 + 1e-5*abs(B)`. Decimal input parsing and
arithmetic retain original JSON number precision; reported numerical values use
explicit `decimal_text` encoding. No epsilon is added to a zero denominator.
The report distinguishes first exact change and first threshold exceedance.
An earlier operator takes precedence even if a later FIA result also changes.

Full-capture structure, call stack, module relation, configuration and collector
must align. Only output paths in the configuration and package installation
paths in collector identity are excluded from identity comparison, following the
existing capture contract. No request, step, call or stack normalization occurs.
Actual step numbers must be passed as captured, not renamed into synthetic steps.

Null values are retained as unobservable, including before the first observed
exceedance. Paired NaNs are incomparable, and a NaN anywhere in either capture
prevents whole-capture E5 comparability even after a valid earlier difference.
All-null captures cannot pass. Signed infinities are recorded separately; a
finite-to-infinite transition is a difference without a finite relative error.
No observable exceedance is a valid negative result, not a reason to lower the
tolerance. Even a located first observable difference does not prove that an
earlier unobservable field was equal or that the operator caused the fault.

`independent_full.json` preserves all missing/nonfinite records; `summary.json`
retains counts, first values and the full-report SHA. Neither file rewrites raw
statistics. The module's SHA and every input file's byte hash are recorded.

## Saved C2/C3 coverage

`audit_coverage.py` independently recomputes call/target counts, joins identities
and decisions, rejects duplicate/missing outcomes, non-target writes, incomplete
OFF/resource recovery, and mismatched saved C2 scope for a C3 report. It verifies
referenced evidence bytes from explicitly supplied files/directories:

```sh
python experiments/c2c3_closure_20261117/audit_coverage.py \
  --report ORIGINAL_RECEIPT.json --pointer /coverage \
  --artifact-directory NATIVE_EVIDENCE_DIRECTORY --out NEW_AUDIT.json
```

`--pointer` is optional for a top-level coverage report. Directory scans include
only immediate JSON files, not private tensors or arbitrary recursive content.
For every-match reports, supply a top-level matching `--c2-report`; serialized
reports never replace the live-process C2 admission proof.

`report_consistent` and `referenced_artifact_bytes_verified` are separate results.
Neither authenticates a supplied observer universe, proves eligibility or Native
semantics, proves device synchronization, or confirms the loaded binary/build.
Those reviews remain explicit in `remaining_review`. Public/C2/C3 qualification
flags always remain false. Source/build/image/kernel audit and real NPU result
review are still required before the integrator changes any capability matrix.

## CPU verification

23 focused tests passed across saved coverage, natural model scope, raw-statistics
audit and the existing evidence manager. The coverage tests deliberately use
CPU fixtures; no reported NPU or model status is inferred from them. Regression
cases include old subthreshold FIA changes, earlier-op differences, nulls,
NaN-before/after-first-difference, exact tolerance boundaries, metadata mismatch,
counter tampering, non-target writes, incomplete recovery and C2 scope transfer.
