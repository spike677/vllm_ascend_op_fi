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
| Provider inventory | `registry.py`, exposed in capabilities/discovery |11 existing RPC providers;6 routes explicitly awaiting RPC adaptation |
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

Strict current qualification is6/17 for C2 and5/17 for C3: the existing
Fault18, MoE8, repaired Native39/Native20 scopes, public Core22 C2/C3, and
KV41 C2. Core22 source50 passes9567 independent actual-file conditions,
including original live-proof issuance/use,29 RPCs and post-cleanup same-Worker
health. See [its reviewer](../../experiments/c2c3_closure_20261117/worker_core_rope_review.md).
KV41's direct live-proof receipt gap remains. Keep C0/C1/C2/C3, Recovery and E5 separate. In
particular, Native39's C2 Inf/NaN statistics and absent C3 raw capture remain
independent E5 limitations. Do not rerun accepted cases merely to refresh counts.

The new runtime includes the `engineering_closure` observer package in the root
wheel. Install all three distributions (root, `src`, `fault39`) from the same
reviewed snapshot; a source-tree import is not proof of installed-wheel closure.
The blue-zone implementation is independent of yellow-zone catalog/controller
class names. Actual yellow-zone integration, new9.x builds and A3 execution
remain separate acceptance work.
