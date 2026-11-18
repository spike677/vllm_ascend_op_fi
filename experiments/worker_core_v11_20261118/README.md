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
attempt/source scope. New milestone attempts are listed below; each uses a new source and attempt scope.

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

Source69 (`d58251c0`) built three fresh wheels in the production image without
an NPU. Its137 CPU preflight tests had one failure in an existing snapshot
isolation test: two queries intentionally produce different monotonic times,
but the test compared the whole dictionary for equality. The test now advances
a controlled clock100→200 and still compares every remaining field, including
row/tensor identity and mutated nested-prefix isolation. Production behavior
is unchanged. Source69 has not run the NPU census; a new attempt must use the
corrected preflight before proceeding.


## Current V1.1 checkpoints (2026-11-18)

Source70 (`0044484b`, source tar SHA256
`3df8fcdab9a112ce6a496bad7cddff77b1e2a36f8f16697e8e55516eb58cbd19`)
passed137 production-image CPU contracts and completed two fresh B8 healthy
DeepSeek quant inventories on124/card2. Both retain135 actual projection
calls and complete Native GWS/execute denominators. Row proofs improved from
2 to3 per request (including layer0 o_proj);132 remain unproved per request.
There were zero fault injections and zero Native-qualified targets. This is
controlled census progress, not public C2/C3 qualification. The first remaining
break follows the original embedding residual clone into `maybe_chunk_residual`
and `npu_add_rms_norm_bias`. The new source-bound embedding/TP1 producer adapter
requires a fresh census; it does not retroactively relabel source70.

Remote originals remain under
`/home/17-fault-c2c3-20261117/v11-quant-lineage-source70-card2-a1`.
The original result SHA256 is
`6d7dc75f05bd15bd153859ffc147de053ae53ebd7fe5df40dcd6a002b8054fc3`;
the full inventory SHA256 is
`f68a3b2296ca3aae2d579c3252eee89a088fa1d6346c20da0e128a1e1bf5b114`.
These bytes are not included in this source-only publication.

Source71 (`16fc8789`) passed112 production-image contracts. Its26 FIA complete-set
attempt completed both healthy inventories and Reference/Lossless/OFF stages,
then refused the group fault before execution because the loaded environment
fingerprint was unavailable. Two mapped `libcust_opapi.so` paths and several
CANN/Torch path-vs-mapping identity failures remain explicit. The attempt
`v11-core26-fia-source71-card2-a1` ended with no C2/C3 grant; the reported owner
was released and deployment removed. No library-identity check was relaxed.
Independent audit and corrected environment observation are still required.

The optional17 multimodal census uses the actual scheduler feature,
placeholder, processed pixels and visual call, followed by actual model/logits
objects. Cache-skipped encoders remain unobserved. This route performs no F and
does not grant C1/C2/C3/Native/E5.39.output has public complete-set protocol
code; device acceptance remains pending. Shared registry/export integration
passed89 local CPU tests with one platform skip (88 executed), including the
new quant embedding,39 group and17 observer contracts. CPU success is not NPU
qualification. Six prior C2/C3 mechanism scopes remain unchanged.


Source71 Attention public observation completed two cases.33 records112 actual
FIA calls: two requests, each28 prefill and28 decode calls;56 decode calls match
the current input contract.28 denominator replay records56 prefill calls across
two requests; all use actual TND layout and none matches the current native
fault template. Both retain all calls, invoke the original API once per call,
leave inputs unchanged, and finish with zero owned hooks, wrappers, inflight
calls or leases. These are Python/API observations, not an independent Native
launch denominator, exact bridge or C1/C2/C3 qualification. New F remains closed.

The device-free environment diagnostic reproduced the image overlay difference:
loaded mappings use device253:2 while pathname stat uses the overlay device,
with matching inode values. `map_files` access is denied. The diagnostic does
not assume matching inode or file hash proves mapping identity. Seven unchanged
public image libraries were copied to an isolated host stage and mounted
read-only at their original paths in a new no-device container. This made the
five loaded staged libraries pass the original strict identity check. No host
or existing container library changed; new NPU workers still need their own
mapping and actual function-pointer proof, especially for duplicate custom-library
basenames. The unstaged no-device driver stub remains unverified.


Source72 (`7646e878`, manifest SHA256
`a976dba82911e87864ebc4f6039c7cd3a12d79b0644c5fd23349a29342d3db49`)
passed100 production-image CPU contracts. Two fresh B8 healthy quant requests
completed on124/card2. Each actual embedding producer and TP1 same-object
communication return is now observed; clone and residual no-chunk edges are
proven. The next AddRmsNormBias edge is still unresolved, so complete projection
row proofs remain3/135 per request. All270 actual Native preparations/executions
are bound; zero targets are qualified and no F is executed. Original result
SHA256:`52d09b36e433b4ea6f14246e1bc29bc9eb7ccc2a1daba7ced1ef12f2bbaff1b4`;
inventory:`ebba0a3a31d6161b61e293add84d5097eae1f19a6e998c43fd123e9d2057a469`.
The owner was never installed, owned API/method hooks were removed, both
scheduler tensor references cleared, and before/after healthy tokens match.
These two attempts improve evidence coverage without granting C2/C3.
