# Worker Core V1 — integration checkpoint

The serving runtime creates, loads and schedules Workers. DemonCATPFI attaches
to the existing Worker, discovers its actual model and delegates fault actions
to the existing provider lifecycle. It does not restart Workers, reload global
weights, edit vendor code, or translate a catalog entry into qualification.

```python
from demoncat.runtime.vllm_ascend.worker_extension import (
    FaultConfig, HostAdapter, WorkerFaultClient,
)

client = WorkerFaultClient(
    HostAdapter(existing_executor, prepare_binding=deployment_binding),
    FaultConfig(fault_id=20, variant="buffer_substitute"),
)
structure = client.discover()
```

`deployment_binding` is trusted host integration code. It prepares the reviewed,
inert provider binding and deployment settings; the end user supplies no layer,
rank, device, operator, address or library path. On a multi-Worker executor,
trusted routing must choose one *returned actual identity*. V1 does not infer
rank zero or route by response position. Incompatible or missing deployments
remain explicit refusals.

## Shared components and their boundaries

| Capability | Implementation | Current boundary |
|---|---|---|
| Identity, generation, tenants, ownership | existing `identity.py` / `runtime.py` | one actual Worker; uncertain actions retain owner |
| Public configuration and routing | `config.py` / `host.py` | all17 catalog IDs resolve; this does not qualify their factories |
| Provider inventory | `registry.py`, exposed in capabilities/discovery |13 callable RPC routes;4 entries explicitly awaiting RPC adaptation; factory availability grants no qualification |
| Structural CandidateSet | `discovery.py` using existing model discovery | all aliases and unknown roles retained; no model forward or device load |
| Natural call census | `call_census.py` using existing path/request/native observers | one naturally scheduled prefill or first-decode request; bounded capture |
| Native build | existing `fault_injection.native_build.NativeBuildManager` | explicit recipes, source/compiler/ABI/library fingerprints; no universal8.x/9.x promise |
| Fault execution/recovery | existing provider dispatch/readback | independent native/Python resource evidence; no alternate lifecycle |
| Coverage | existing `CoverageSession` and provider group adapters | V2 support and acceptance remain mechanism/workload specific |

Start a census only after the serving host has actually enqueued its request:
`client.census_start(phase="prefill", expected={"internal_request_id": actual_id,
"prompt_token_ids": actual_tokens})`. The host runs its normal scheduler; then
call `census_cleanup()`. The census itself does not invoke model execution.
`prepare/register/check/execute/advance/finish/unregister` use the same provider
state machine as before. See [the host contract](host_contract.md) and
[the census contract](census_contract.md).

The new [output39 C1 contract and remaining-route audit](output39_host_contract.md)
keeps its MODEL_LOCAL evidence separate and leaves all three 28 variants plus
33 pending. Only real standard enqueue requests can advance its single-target
discovery/lossless/F/OFF/health sequence; C2/C3/E5 are not provided by that route.

The [natural tensor/MoE contract](tensor_group_host_contract.md) now routes 8/18
through actual enqueued requests. RPC actions arm or collect the existing group
session; they do not invoke a replay forward. This new public entrance requires
its own source-bound review and does not inherit the old natural-group result.
That review now reconciles source66:113/226 RMS targets/hits and26/52 MoE
targets/hits, exact group OFF receipts and final same-Worker health. See
[the actual source66 scope and limits](../../experiments/c2c3_closure_20261117/WORKER_INCREMENT_20261118.md#public-818-source66-natural-scheduled-requests-independently-reconciled).
The separate [nested-mode recovery repair](moe_nested_mode_recovery.md) has new
CPU failure-path contracts; it does not relabel the old snapshot's exception path.
The other legacy tensor factories still require trusted synchronous replay.
Thirteen callable factories must not be described as thirteen naturally
scheduled, verified public Worker routes.

The [Native26 public contract](writer26_host_contract.md) uses the same
prepare/register/advance/finish lifecycle for its QMM and typed FIA variants.
It keeps the complete natural inventory while selecting one proven site for
C1; it does not provide C2/C3 or turn diagnostic runtime captures into E5.
Actual source67/68 runs now reconcile both public C1 chains. The separate
source68 output39 custom-GEMM run also reconciles its current producer/consumer,
one actual Native intervention, exact OFF and owned cleanup. These runs use
pretrained models and normally scheduled inference; they introduce no training
path. See [the delivery scope](../../experiments/c2c3_closure_20261117/WORKER_CORE_DELIVERY_20261118.md).

An ordinary census cannot observe all internal kernels. Native or fused edges
without a matched observer remain `NOT_VISIBLE`/unverified. A failed request
mapping is a failed observation; if its owned hooks/mode are demonstrably gone,
it may release ownership. Inflight calls, wrong identity, foreign wrappers or
unknown cleanup retain ownership and block new fault work. Cleanup performs a
fresh readback instead of trusting its action's success response.

## Validation and unchanged evidence

The initial integration passes91 focused CPU contracts: public host/discovery,
extension, natural CPU dispatch census, RPC failure isolation and cleanup. These
tests do not grant NPU, model, C1/C2/C3 or E5 qualification. The reference NPU
entry is `providers20_worker_probe.py --worker-core`; it uses the same frozen
group qualification and recovery logic, plus public discovery/census/client.
Its real result now passes in the bounded source46 scope; see
[actual acceptance](../../experiments/c2c3_closure_20261117/WORKER_CORE_V1_ACCEPTANCE_20261118.md).

The existing Native39 source42/package16 run remains frozen. Independent review
accepted113 C2 targets and226 C3 calls with exact OFF and SDK release. Fault20's
source44 short preflight observed135 QMM calls twice and released15012 SDK arrays,
but armed no fault. These are not results for the new Core snapshot.

The initial V1 qualification was6/17 for both C2 and C3: the existing
Fault18, MoE8, repaired Native39/Native20 scopes, public Core22 C2/C3, and
public Core41 C2/C3. Core22 source50 passes9567 independent actual-file conditions,
including original live-proof issuance/use,29 RPCs and post-cleanup same-Worker
health. See [its reviewer](../../experiments/c2c3_closure_20261117/worker_core_rope_review.md).
Core41 source60 adds6796 reconciled artifact/provenance/evidence conditions,
43 original RPCs and original live C2 issuance/use. Its28 targets/56 calls and
whole-group OFF are source-bound; final ordinary health captures tokens/text,
not complete logits after observer removal. See
[the original-file summary](../../experiments/c2c3_postwrite_20261117/core41_source60_review.json).
Keep C0/C1/C2/C3, Recovery and E5 separate. In
particular, Native39's C2 Inf/NaN statistics and absent C3 raw capture remain
independent E5 limitations. Do not rerun accepted cases merely to refresh counts.

The [V1.1 matrix](../../experiments/c2c3_closure_20261117/C2_C3_MATRIX.md)
now records C2 **8/17**, C3 **7/17**. Source84 adds bounded output39 isolated
C2, with its cumulative C3 failure retained. Source89 adds FIA26 isolated C2
and same-Worker cumulative C3; see [acceptance and limits](../../experiments/worker_core_v11_20261118/NATIVE26_SOURCE89_ACCEPTANCE.md).
Those new scopes preserve all six original qualifications and do not grant E5.

The new runtime includes the `engineering_closure` observer package in the root
wheel. Install all three distributions (root, `src`, `fault39`) from the same
reviewed snapshot; a source-tree import is not proof of installed-wheel closure.
The blue-zone implementation is independent of yellow-zone catalog/controller
class names. Actual yellow-zone integration, new9.x builds and A3 execution
remain separate acceptance work.
