# Worker Core V1 host entry

This is the blue-zone, CPU-tested host/RPC contract. It does not claim an
integration with the unavailable inner-framework Registry or Controller.
`fault_injection.integration.HostContract` and
`demoncat_fault39.contracts.HostContract` already define catalog/binding
registration. They remain unchanged. `HostAdapter` adds only the missing
existing-Worker RPC route; `WorkerFaultRuntime` remains the sole fault lease
and lifecycle authority. Fault39 keeps its own backend and native ABI.

## Public selection

```python
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient

# `executor` was created and initialized by vLLM/vLLM-Ascend.
# Its deployment has already installed a reviewed, inert provider binding.
host = HostAdapter(executor)
fault = WorkerFaultClient(host, FaultConfig(fault_id=20, variant="buffer_substitute"))
prepared = fault.prepare()
```

The configuration selects an existing canonical catalog mechanism. Fault39
uses `FaultConfig(39, "matmul_range")`, and the unnumbered quantization family
uses `FaultConfig("quant", "batch_shared_scale")`. The configuration accepts
only `fault_id`, `variant`, and `implementation_version`. It accepts no
layer, rank, device, operator, tensor address, build path, or retry option.
The old model-local `FaultConfig` remains separate and continues to reject
Fault39 fault arming.

Every subsequent action is explicit, and callers inspect each receipt before
continuing: `register()`, `check()`, `enable()` or `execute()`, then provider-
specific serving `advance(actual_request_id)` / `finish()`, and finally
`disable()`, `unregister()`, `detach()`. These call the existing RPC methods;
they do not create a parallel provider state machine. `prepare()` does not
automatically register or enable a fault. A host-driven serving replay still
needs the existing scheduler's workload; an RPC request ID alone proves no
request-to-target binding.

## Trusted host responsibilities

`HostAdapter(executor, select_worker=None, prepare_binding=None, timeout=30)`
consumes an existing executor with
`collective_rpc(method, timeout=..., args=...)`. It does not create, start,
stop, reload, or replace a Worker or model.

The adapter reads `demoncat_worker_identity` and retains the complete identity,
including process start, Worker start and model instance. With one ready
Worker it routes automatically. With multiple Workers a trusted deployment
policy `select_worker(config, tuple_of_actual_identities)` must return one of
those identities. There is no implicit rank-zero selection and no client rank
selector. Incomplete identities or duplicate ranks reject before mutation:
the existing broadcast RPC distinguishes non-target ranks but cannot safely
address two live recipients sharing one rank. A deployment needing that
topology must supply a scoped executor; V1 does not implement multi-Worker
scheduling.

Every targeted RPC carries the original Worker identity and exact attachment
or lease ticket. Broadcast response order does not determine routing.
`RPC_NOT_TARGETED` receipts are excluded; exactly one targeted receipt must
remain. A changed Worker/model identity is never replaced with a newly
discovered identity during recovery.

An optional trusted `prepare_binding(identity, config)` callback supplies the
existing deployment binding before provider prepare. It may return
`{"prepare": {"operator": "aclnnQuantMatmulWeightNz"}, "register": {}}`, for
example, after automatic deployment inspection. The callback must only
provision an inert reviewed binding and return JSON settings; it must not
arm native callbacks, acquire fault leases, or modify an unrelated binding.
The deployment owns binding installation/removal. Providers remain responsible
for verifying actual runtime/build/target eligibility. Omission means use an
already installed binding, not that a binding or runtime is qualified.

## Ownership and uncertain replies

The client generates its own tenant and owner, records intent before sending
mutations, and retains the Runtime-issued attach nonce and lease generation.
It rejects mismatched owner, generation, attachment, and Worker receipts.
Snapshot copies cannot mutate the live ticket. There are no automatic retries
and no implicit cleanup on exception or context exit.

A failed or ambiguous operation records `Recovery.status=REQUIRED` and keeps
the original owner/ticket. Non-recovery mutation is blocked locally; explicit
owner-scoped `recover`, `disable`, `abort`, `clean`, `unregister`, or `detach`
still uses the existing Runtime. A successful `recover` may require scheduled
recovery requests before `finish`; it does not imply release or health proof.

If a prepare reply is lost, its generation may be unknown. The known attachment
can explicitly `detach()`: Runtime unregisters only this tenant's leases and
returns their recovery receipts. Failed detach preserves the owner and any
recovered ticket. No foreign tenant is drained. If the attach reply itself is
lost, the nonce is unknown; the client cannot invent one or reattach. It keeps
an uncertain intent for trusted host reconciliation. Similarly, a lost final
detach reply does not justify claiming release. V1 intentionally has no
unauthenticated ticket-recovery or force-release RPC.

## Evidence boundaries

`capability_enumeration` is inventory, not execution eligibility.
`discover()` invokes the optional attachment-scoped `demoncat_discover` C0
inventory; it never enables a fault. `provider_check=COMPLETED` means only that
the existing provider's check action completed. Some serving providers defer
lossless qualification to their scheduler-driven sequence.

Receipts keep raw provider replies and host failure history, while independent
top-level fields remain separate:

| Field | Meaning |
| --- | --- |
| C2 | `NOT_ASSESSED`; needs independent natural-call denominator and target hits |
| C3 | `NOT_ASSESSED`; needs accepted C2 and every-match evidence |
| Recovery | `REQUIRED`, `PENDING`, `NOT_ASSESSED`, or `LEASE_RELEASE_CONFIRMED` |
| Recovery.healthy_output | `NOT_ASSESSED`; a resource-release receipt alone is insufficient |
| E5 | `NOT_ASSESSED`; needs the independent numerical comparator |

Neither a changed model output, provider statistics, nor a successful resource
release promotes C2/C3/E5 or healthy-output recovery. Provider-specific evidence
can be assessed separately without changing the ownership contract. These CPU
tests exercise the real Worker RPC state machine with readback fixtures; they
are not native launch, NPU acceptance, or inner-framework integration evidence.

## Focused CPU validation

Run `python -B -m unittest discover -s tests/worker_extension -p test_core_host.py`
with repository root, `src`, and `fault39/src` on `PYTHONPATH`. Contracts cover
actual identity routing, two Workers, response reordering, duplicate-rank
refusal, generation/owner retention, lost prepare/enable/attach replies,
foreign-tenant isolation during recovery, unchanged Fault39 compatibility,
trusted QMM selection, and independent Recovery/C2/C3/E5 reporting.

Recorded 2026-11-18 on base `649c4f88ce5dc792d42c4aa1a521ad1b08c9311f`:
29 focused CPU tests passed, with no skips. No Native build, device/container
query, NPU execution, or serving lifecycle operation was performed for this
host change. Actual Fault20/39 deployment qualification remains with the
single NPU controller.
