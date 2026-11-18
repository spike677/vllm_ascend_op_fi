# Explicit multi-Worker endpoints

`WorkerEndpointSet` extends host routing over an existing executor. Deployment
supplies the complete expected rank set; no rank is selected by default and RPC
response order is irrelevant. Each endpoint pins the full live process/model
identity. A fresh census must still match when its client first attaches.

```python
from demoncat.runtime.vllm_ascend.worker_extension import (
    FaultConfig, WorkerEndpointSet,
)

routes = WorkerEndpointSet(existing_executor, expected_ranks=[0, 1])
rank0 = routes.client_for(0, FaultConfig(39, "matmul_range"))
rank1 = routes.client_for(1, FaultConfig(39, "matmul_range"))
# Use the existing explicit prepare/register/check/execute and recovery protocol
# on each client, and the serving scheduler for actual requests.
report = routes.snapshot()
```

The example demonstrates routing only. A Provider still needs deployment and
actual target qualification on each rank. Host configuration is trusted
deployment policy; user fault configuration gains no rank, layer or address
selector. Binding preparation remains inert and Native preload precedes Worker
startup. Neither construction nor `client_for` executes inference.

Each rank has its own client, tenant, owner, nonce, ticket and generation.
Repeated `client_for` access returns that client, including uncertain leases;
it does not retry or replace it. Configuration changes on an already-bound rank
are rejected. Recovery is explicit through the existing client. Snapshots retain
every endpoint, including failures and unattempted endpoints. They do not grant
an aggregate C2/C3 or distributed atomic recovery claim.

If a Worker restarts, its old lease cannot transfer to a replacement even when
the rank number is unchanged. The Runtime continues to enforce identity on
every action. Finish recovery/accounting of the old set before discovering a
new one. Other ranks' retained resources must not be dropped from the report.

Validation: CPU Extension/Runtime tests cover reversed responses, missing and
duplicate ranks, changed identity before attach, per-rank tickets, a lost enable
reply, retained recovery ownership and independent detach. They do not qualify
multi-rank NPU execution or collective fault semantics.
