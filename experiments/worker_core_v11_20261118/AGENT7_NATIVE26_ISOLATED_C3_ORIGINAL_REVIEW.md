# Native26 isolated C2 to cumulative C3 original audit

This change adds an offline reader for the explicit `c2_isolated_c3` stage from producer commit `4331c110`. It does not reuse the source82 C2 proof or source77 cumulative C3 qualification. Both generations must have the new implementation scope, reconstructed from their actual source hashes. The probe source and Native compiler source remain separate provenance inputs.

`review26_group_originals.py` dispatches this stage to `review26_isolated_c3_originals.py`. Existing `c2_once`, `c2_c3` and `c2_isolated_once` defaults are preserved. No provider, Runtime, Native library or old receipt was changed by the audit commit.

The new reader checks:

- Both per-request delta SHA chains, actual arm/query/collect RPCs and original enqueue/output-processor mappings. The C2 health collect must append its final cleanup original before starting C3. The C3 chain starts independently with a null predecessor; it cannot continue or replace C2's chain.
- Every reconstructed row, outcome, ledger call and serving request against each generation's final complete original. Every later RPC must retain the same original `prior_c2` object. Driver summaries must match the original RPC projection but do not supply missing observations.
- C2's complete Native denominator and one eligible target per fresh F/OFF pair. Each F starts from its own previously qualified Reference residual, and the following OFF belongs to the same target. The existing independent writer/consumer, output, restoration and ledger audits are retained.
- C3's complete requalified denominator and two fresh requests that each inject every eligible target. It is explicitly **same-request cumulative**, not an independent target replay. Reference, Lossless, OFF, repeat output and final health are checked by the existing cumulative reader.
- The exact saved issuance/consumption record of the live C2 proof, object binding, same scope/owner/worker, new session and `native_generation + 1`. The public RPC generation stays fixed. Saved records never reconstruct a live proof capability.
- Full FIA typed records, every original stock/owned writer byte artifact, public-return/consumer identity, source-bound metadata and mapped-library environment, and final zero owned resources. No runtime hit summary chooses these conclusions.

For `E` independently reconstructed eligible targets and `N` complete Native calls per request, C2 requires `6 + 2E` group requests, `5E` owned executions and `2EN` ledger calls. C3 requires another 9 group requests, `6E` owned executions and `3N` ledger calls. A separate ordinary post-unregister health request is also required on a complete run. These are plan formulas, not measured NPU results in this commit.

If a later generation fails, its completed deltas and real cleanup are retained, and the already completed C2 is audited independently. A forced empty delta during model unwind is accepted only with the pinned failed-request original joined to the actual arm/phase/abort; it is not counted as a completed request. Missing or malformed later C3 data cannot erase completed C2, and cannot mark C3 complete. Cleanup data and failed partial Native observations do not substitute for missing F/OFF evidence.

Run the existing CLI with pinned original result, probe source, compiler source and Native build identities. The explicit stage in the original determines the reader; there is no flag that relabels old C2 as the new stage:

```text
python review26_group_originals.py \
  --evidence-root <original-evidence> --result-sha256 <result-sha256> \
  --source-root <executed-probe-source> --source-commit <commit> \
  --source-manifest-sha256 <manifest-sha256> \
  --native-root <actual-native-build> --native-build-sha256 <build-sha256> \
  --native-source-root <actual-compiler-source> --native-source-commit <compiler-commit> \
  --native-source-manifest-sha256 <compiler-manifest-sha256> \
  --mechanism 26.fia_destination --expected-writers <complete-native-count>
```

The reader runs without importing injection/model code or executing Native/device operations. On complete valid saved originals it reports consistency and historical stage reconciliation, while `c2`, `c3`, `e5` and `public_qualification_granted` remain false. A live authority cannot be issued by this CLI. E5 still requires its separate independent raw `statistics/mix` comparison.

Validation used the current public combined CPU driver, including a Native ABI double, and its emitted original files. Nineteen new tests cover both complete delta chains, source-scope separation, byte tampering, missing targets/deltas, foreign generations/owners/proofs, pre-F contamination, wrong OFF, partial budget refusal, actual model unwind and both generations' typed/full-output artifacts. Together with the existing isolated and cumulative readers, **61 CPU tests passed**. These CPU contracts do not grant NPU or standard-Worker qualification. No SSH, container or NPU operation was performed for this audit implementation, and no new private statistics or tensor artifacts are included.
