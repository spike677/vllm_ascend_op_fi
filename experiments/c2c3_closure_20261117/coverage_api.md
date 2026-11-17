# C2/C3 coverage contract: provider integration

Status: implemented and CPU contract tested. This does not grant any mechanism
C2, C3, E5, model, CANN or SoC qualification. Public C1 paths and unique native
configuration gates remain unchanged. The first clients are 39.matmul_range and
20.buffer_substitute; their real Worker evidence is a separate root-queued task.

Use `fault_injection.coverage_session`. The provider still owns every Native ABI,
input proof, lossless mode, effect artifact and storage restore operation.

1. Run an independent observer over the full actual healthy execution boundary,
   including calls declined by the injector. Capture two healthy forwards and
   every structural candidate's observed disposition. Use
   `runtime_discovery.RuntimeCandidateInventory(expected_modules=..., expected_workers=...)`,
   `observe(Candidate(...), forward_id=...)`, `finish_forward(..., healthy=True)`
   and `seal(observer, artifact_sha256=...)`, or call `observer.seal_discovery`
   with the equivalently complete independently collected inventory. An ambiguous
   native cache is unresolved, not an unsupported exclusion. An unexecuted module
   is NOT_OBSERVED, not a target. The eligible singleton case is supported.
2. Build one `TargetLease` per eligible lifetime identity and one
   `TargetLosslessProof` per lease and provider lossless mode. For native targets,
   include `TargetIdentity.process_start_identity` from the actual Worker.
   `QualifiedTargetSet(selection='all_eligible', trigger='once', ...)` must contain
   the exact complete eligible set. Resource limits cover all reserved leases.
3. Create the session; observe before deciding, then execute exactly once:

```python
observer = IndependentCallObserver(
    source='independent_native_trace', observer_id=observer_run_id,
    implementation_sha256=observer_source_sha, workload_sha256=workload_sha,
    max_calls=100000)
# Healthy observer calls seal_discovery before the fault run, as described above.
scope = CoverageScope(mechanism, workload_sha, environment_sha, source_sha,
                      evidence_kind='natural_worker')
hooks = GroupRecoveryHooks(stop, drain, recover, health)
session = CoverageSession(group=group, scope=scope, observer=observer,
    budget=SessionBudget(max_calls=50000, max_seconds=180, max_concurrency=1),
    recovery=hooks)
ticket = observer.observe(ObservedCall(target, call_id, forward_id, request_ids,
    Phase.PRE_FAULT_PROPAGATION), eligible=eligible)
decision = session.begin(ticket)
# Provider: if decision.admitted, apply its existing unique Native config.
# Always execute the original exactly once, and keep actual effect evidence.
session.finish(decision, Outcome.HIT if actual_hit else Outcome.PASSTHROUGH,
    native_applied=actual_hit, execution_sha256=execution_artifact_sha,
    effect_sha256=target_effect_artifact_sha, original_calls=1)
recovery_result = session.disable(timeout_seconds=30)
observer.seal(artifact_sha256=independent_trace_sha, complete=True)
report = session.report()
# Every observed call, including passthrough and OFF, requires a finish receipt.
```

Stop hooks first disable all owned targets. `drain(lease, remaining_seconds)` must
synchronize actual streams/ranks and obey the timeout; return exactly `True` only
when drained. The session invokes no restore or release until **every** target's
drain succeeds within the deadline. A later target failure or timeout retains all
reservations and restores nothing; retry repeats the entire incomplete drain
barrier. `recover(lease)` restores and releases its resources and returns a
`RecoveryReport` bound to that exact lease, proof SHA and baseline-matching OFF
output SHA. `health(worker_tuple, owner, generation)` returns `WorkerHealthProof`
with equal full healthy baseline/OFF output hashes in that same Worker lifetime
and zero active leases, device bytes, host bytes, in-flight calls and owned hooks.
Failure retains group ownership; retry `disable` on that same session. A failed
session must not be abandoned to permit a new owner to reuse its targets.

For C3, obtain `proof = c2_session.c2_proof()` only after the complete C2 run and
confirmed recovery. Construct a new group generation with `trigger='every_match'`
and pass `c2_proof=proof`; workload, implementation, environment, candidate set,
target lifetimes and owner must match. The live receipt cannot be recreated from
serialized JSON. Admission requires the original internally issued object, its
unchanged field digest and the actual issuing process ID. `copy`, `deepcopy`,
`dataclasses.replace`, reconstruction and fork-inherited objects do not preserve
authority. Keep the original object on the owning Worker; do not rewrite its
owner/scope to fit a later run. CPU/constructed scope cannot open a natural Worker C3 run.
Providers may serialize separate native generations within the logical group,
recording exact native generation/cache epoch in each execution artifact.

Natural pretrained inference outside a standard Worker (for example a VAE) uses
`evidence_kind='natural_pretrained_model'`, with `model_weights_sha256` and
`model_source_sha256` referencing the actual weight manifest and model source.
Only NPU `BoundaryTargetIdentity(scope_kind='MODEL_LOCAL', ...)` targets are
accepted in this scope. Its report sets `real_pretrained_model_evidence=True`
and `real_worker_evidence=False`. These scope fields also bind the live C2 proof;
changing weights, model source or scope cannot reuse it to open C3. Hashes remain
references to artifacts for independent review, not authentication of their
contents or automatic public qualification.

New post-anomaly targets use `phase=POST_FIRST_ANOMALY, propagation_added=True`.
They have their own report section and pass through pending new discovery. An
existing discovered target cannot evade every-match accounting by this flag.

The adapter supplies authentic artifacts and the scheduler supplies an authentic
module/rank universe. Hashes are artifact references, not a cryptographic verifier
of the underlying Native facts. The report always retains
`public_qualification_granted=False` for the main integrator's independent review.

CPU command (Windows, repository root):

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests'
& '<unified-env>/Scripts/python.exe' -B -m unittest test_coverage_session test_coverage_planning test_boundary_targets test_auto_adaptation -v
```

Verification including the group drain barrier: 50 tests passed, 0 skipped.
No NPU/container action was run.
The suite covers C1 regression, same-shape distinct lifetimes, incomplete/ambiguous
denominators, missing outcomes, every-match omissions, budgets, exact once original
execution, non-target writes, effect proof, in-flight disable, retained ownership,
retry, zero-resource health, and propagation additions.

The subsequent live-proof fix and frozen 18/20/39 driver review are recorded in
[coverage_proof_authority_review.md](coverage_proof_authority_review.md). These CPU
checks and source review do not replace any historical NPU evidence or extend it
to the new coverage implementation.
