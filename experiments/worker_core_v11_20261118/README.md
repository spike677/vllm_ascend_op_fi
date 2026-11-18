# Worker Core V1.1 + 17 Fault Provider Completion

Base: `027916b3a2e021d107e41cc063d29e66d879ee6b`. Preserve the six accepted
C2/C3 scopes and all earlier source snapshots, failures and raw evidence.
This workstream does not reset or transfer their qualification.

## Ownership and execution

Root owns shared contracts, registry integration, device queue and qualification
review. Three existing CPU agents use separate worktrees: writers (26 then
39.output C2/C3), quant (actual request-row lineage), attention (four separate
28/33 routes). No agent may SSH, start a container or acquire an NPU. Public
runtime/host files are not rewritten by those mechanism branches.

[Execution Adapter contract](../../docs/contracts/execution_adapter_contract.md)
freezes the common responsibilities while allowing mechanism-specific adapters.
The serving runtime keeps Worker and inference ownership. No generic
execute_model hook, unverified factory, late Native load or implicit replay is
introduced to make the inventory appear complete.

Root read-only inventory on124 found card2/5 with no compute processes and one
existing gyl_test container. Prior source67/68 controllers were complete with
owned cleanup. This observation is not a reservation: each new case still
requires the existing resource lock, fresh process/allocation gate and a new
attempt/source scope. No NPU case has started for this new milestone yet.

## Work ledger

| Mechanisms | Preserved evidence | Next concrete work |
|---|---|---|
|8,18,22|Bounded natural Worker C2/C3, including source66 public8/18 and source50 public22|Keep qualification; add uncontaminated E5/device payload and only genuinely new fused/model paths.|
|20,39.matmul,41|Repaired Native builds or public group C2/C3|Keep original recovery proof; extend E5/actual new API or environment separately.|
|26.residual,26.fia|Public C1 source68/source67,135/30 observed calls and2/29 resolved chains|Qualify all stable candidates without truncation, then C2 and C3 with actual writer/consumer completion and full group OFF.|
|quant.batch_shared_scale|Public C1 source64;2 proven row mappings out of135|Classify the other133 through actual row transformations; unresolved lineage remains unverified; then coverage.|
|39.output|Public C1 source68;113 observed projections/28 eligible pairs|Reuse actual GEMM/consumer guard and Native receipts for multi-target C2/C3.|
|28.fa.legal,28.fia.legal,28.fia.replay|Distinct standalone/kernel and historical model scopes|Public request-scoped execution adapters for real layouts/launches, no empty factory.|
|33.paged|Actual TND bridge has67 differing output elements|Resolve normal stock/private bridge first; F is prohibited while it fails.|
|15.alias|Healthy NZ constructed mechanism; observed natural VAE format30 does not match|Natural private-base/view/write/base-consumer chain or explicit non-applicability for the inspected path, not a global model claim.|
|17.spatial|Natural Qwen3-VL MODEL_LOCAL census|Standard multimodal request identity and actual padding/consumer proof.|
|25.reduce_scatter|Six bounded8.5.1/9.1 calibration cases have no replay sample|Review actual collective/backend evidence once; no repeated identical calibration or manufactured fault.|

Hardware/versions: current Worker evidence is exact8.5.1/A2. Other8.x/9.x
Workers, A3 and yellow-zone wiring require separate evidence. E5 is independent
and must not block otherwise safe functional development. New raw statistics,
private arrays and Native binaries stay124; publish implementation, tests,
reviewers and summaries only.

## Shared increment and original-file findings

The optional execution contract is documented. `WorkerEndpointSet` now provides
explicit host-side rank sets with separate existing Core clients/leases. It
starts no Worker and makes no atomic group claim. The related36 CPU contracts
and all219 `test_core*.py` contracts passed locally (Torch2.10 CPU). Multi-rank
NPU qualification is not inferred from these tests.

Read-only review of source64's first healthy request found the first MLA row
proof break at event187: a view changes `[8,16,64]`, stride`[3072,192,1]`,
offset128 into `[8,16,1,64]`, stride`[3072,192,64,1]`, preserving storage.
Event197 similarly inserts a unit dimension into a noncontiguous query.
The observed `o_proj` and downstream133 projection mappings remain unverified
in the original. A narrow stride-aware repair requires a new healthy census;
the old evidence and qualification remain unchanged.

For26 QMM, the27 unresolved aggregate candidates have stable target keys and
stable Native cache/registration/geometry descriptors across both healthy
requests. Three first-layer calls lack current residual-producer proof. The
other24 change allocator adjacency eligibility between requests (18 lose
adjacency,6 gain it). These are not module-name or target-key mismatches.
They cannot be silently excluded to grant C2 to the remaining two candidates.
