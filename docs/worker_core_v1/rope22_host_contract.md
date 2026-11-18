# Fault22 serving group through Worker Core

`WorkerFaultClient(HostAdapter(existing_executor), FaultConfig(22, 'wrong_row'))`
uses the existing WorkerFaultRuntime lease and the existing ServingRoPEGroup /
RoPEGroupProvider. It does not create a Worker, run a model from an RPC, reload a
model, select a user-supplied layer/device/address, or retry a fault.

## Trusted deployment

The host's reviewed deployment operation installs
`serving22.install_fault22_serving_binding` **on the selected existing Worker**.
Its inputs are `prompt_token_ids`, the reviewed `weight_manifest`,
`workload_sha256`, `environment_sha256`, and explicit
`first_live_validation=True`. `HostAdapter.prepare_binding` may invoke this
operation through the serving runtime's existing trusted deployment facility,
or consume a previously installed binding. Installation is inert: no hooks,
dispatch modes, process profile callback, coverage lease, forward, or native
load is created. Its return value says `BOUND_NOT_QUALIFIED`.

The public RPC accepts no deployment factory, trace factory, operator, proof,
target list, or user settings beyond the actual enqueue ID at `advance`.
The private CPU seams require `simulation=True`; their CoverageScope remains
`cpu_contract`. Multi-Worker selection still requires an explicit trusted host
policy selecting a complete identity returned by the actual Workers.

The existing reviewed RoPE source/dispatch/owner/cache checks remain mandatory.
Only the observed fused outer call and its independently joined real reader /
launcher may select `22.wrong_row.fused_positions`. A missing fused binding or
call cannot fall back to the older `index_select` lookup proxy. The variant
changes the last **legal** positions input to the first legal position in a new
allocation. It is not uint32 overflow, a tiling intervention, or proof of a raw
device launch payload. Existing direct legacy/model-local adapters remain
separate implementations and do not supply this public serving qualification.

## Existing RPC lifecycle and real enqueue

Call `prepare`, `register`, `check`, then `enable` (or `execute`) once. `check`
reports local binding readiness only. Four real healthy requests must still
establish source-bound discovery and lossless behavior before fault admission.

For each row below, the existing serving host enqueues the reviewed fresh
prefill, obtains its real internal request ID, calls `client.advance(id)`, and
lets the existing scheduler execute that request to completion. No synthetic
request ID or direct model replay in the client can complete a row.

| Natural request | Existing group phase |
| --- | --- |
| 1–4 | Native baseline, identity baseline, repeated baseline, lossless |
| 5–6 | C2 once across all independently observed sites, then complete OFF |
| 7–9 | C3 every-match across two requests using the original live C2 proof, then OFF |
| 10 | Same Worker healthy request after group/fault-resource cleanup |

The next `advance` finalizes the previous completed row and selects the next
phase. After request 10, call `finish`, `disable`, `unregister`, and `detach`.
Every result must be inspected. A missing request, failed root/logits callback,
changed held identity/source/cache, unknown site, lost reply, or cleanup failure
retains the existing owner. No successful transport reply implies C2/C3 or E5.

`advance` only records the request ID. The owned root pre-hook starts the
existing trace/adapter on the actual root call's current args/kwargs. PyTorch
snapshots pre-hooks before entering that hook, so this reviewed seam enters
only its newly installed root hooks, in their real registration order, exactly
once; it never re-runs the model or operator. Foreign root pre-hooks whose
ordering is unreviewed are refused and preserved. After the actual logits
return, collection closes the per-request modes/profile/temp lease. Between
RPCs, the idle scheduler reservation is reported as `replay_pending`; active
dispatch modes and temporary leases are zero. With no owner, ordinary serving
has no managed-request requirement.

The final completion callback captures the actual processor return again,
after any preceding return-transforming hooks. An earlier logits observer
snapshot cannot stand in for that final value. An OFF return mismatch prevents
live proof issuance even when earlier observer values appear unchanged.

Runtime supplies the already validated identity to the provider. Its local
guard rechecks held Worker/runner/model objects, process, thread, rank and
configured/observed logical device. It never recursively calls
`demoncat_worker_identity` under the Runtime lock. The original direct
ServingRoPEGroup path retains its identity RPC behavior.

## Proof, recovery and separate assessments

`CoverageSession.report()['live_c2_proof']` contains detached issuance and
consumption audit receipts: full scope, owner, source/consumer generations,
target IDs, candidate/discovery/observation/recovery/report hashes, proof hash,
and process-local Python object identity. The public provider retains the
original live proof privately and exposes only these JSON audit receipts in
`provider_state.coverage.C2` / `.C3`. The issued and consumed object binding and
proof hash must agree. These receipts cannot be deserialized into authority;
copies, reconstructed objects, changed scope/owner/generation and changed
targets still fail the existing exact-object gate.

The group owner is Runtime's provider owner; the public client owner remains in
the existing RPC ticket. `coverage.deployment` joins that provider owner and RPC
generation to the source hashes of this binding/bridge. C2 and C3 use increasing
CoverageSession generations within that one public RPC lease.

The proof's `report_sha256` covers the coverage report **without** the
`live_c2_proof` audit ledger, avoiding a self-referential issuance hash. The
ledger is detached/read-only evidence. A caller holding a pre-issuance report
must read `report()` again after `c2_proof()` to include issuance; the public
provider does this before saving its C2 result.

Explicit `recover` first stops and removes owned scopes. If the current group
has no complete observed OFF, it allows only a new real `recovery_off` request;
it never retries a fault. A failed drain/health cleanup can instead retry
against the same already observed OFF and sealed denominator. Hook removal is
read back, and each failed removal retains its handle for retry. Foreign hooks
are not removed. An incomplete failed attempt remains a failed coverage result
even if its resources and same-Worker health are subsequently recovered.

If removal/pop actually completes and then raises, the failed response is
retained while confirmed-absent handles are retired using the actual hook map
or dispatch-mode stack. A before-removal failure retains ownership. Deployment
binding occupancy survives `clean` until the existing RPC `unregister` releases
its ticket; unbinding cannot strand a still-owned ticket.

The final cleanup/healthy request, C2, C3, resource Recovery and E5 are separately
reported. E5 requires its independent raw-statistics workflow and is not
assessed here. CPU fixture success grants no natural-Worker qualification.
This integration is not a 17/17 statement, yellow-zone in-source adoption, or
CANN 9.x / A3 acceptance; those deployments still need independent acceptance.

## CPU verification

From the reviewed source root with all three distributions on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
python -B -m unittest -v test_core_rope22 test_coverage_proof_authority
```

`test_core_rope22` uses real HostAdapter, WorkerFaultRuntime, torch module hook
iteration, natural fixture forwards and the existing RoPE group/observer/
injector. It checks full shared-site denominators, exactly-once roots/operators,
original proof use, failed root/logits and uncertain-reply recovery, owner
isolation, identity drift, partial/silent cleanup failure and ordinary
post-cleanup serving. Its self-contained `rope22_fixture.py` imports no
experiment history; both are included in the source-only integration bundle.

## Root-scheduled standard Worker probe

`experiments/c2c3_closure_20261117/prepare_rope22_worker_core.py` generates a
reviewable case without launching it. It reuses the existing fixed
Qwen3-0.6B pretrained weights, prompt, TP1 eager Worker and bounded resource
configuration. For example, the root scheduler can prepare its new reviewed
source snapshot with:

```sh
python3 experiments/c2c3_closure_20261117/prepare_rope22_worker_core.py \
  --source /absolute/reviewed/frozen-source --card 2 --out worker-core22-case.json
```

The generated case calls `natural_rope_worker_core_probe.py` with
`--model /model --out /evidence/probe --memory .35`. That entrypoint constructs
the normal serving host, obtains its actual Worker identity through
`HostAdapter`, and installs the inert reviewed binding on that same Worker.
All ten managed requests then use public enqueue, actual output-processor ID
mapping, `client.advance`, scheduler completion and `client.query`. Every
root/operator runs through the serving scheduler; no fault RPC invokes a model.

The full server-local `result.json` retains the nine original phase rows in
`requests`, with every reader, outer/native call, observation and scheduler
mapping. Request 10 is separate in `health_request`. `cleanup` records the
group cleanup preceding that healthy request; `worker_core.final_cleanup`
records the final public cleanup. `c2` and `c3` are the complete original
CoverageSession/group reports, including `live_c2_proof`; `c2_open` and
`c3_open` join their actual owner, generation, target denominator and trigger.
Neither phase rows nor proof ledgers are reduced to a summary.

`worker_core_rpc/*.json` contains each original public Host action/query
receipt once, indexed with its SHA256. `host_receipt.json` retains the final
client scope and entire original history. The generated case also retains
stdout and stderr under `/evidence`. These raw artifacts belong on the serving
host (server 124 in the current deployment), not in public Git or the
source-only integration bundle. A failed attempt stays failed even if one
explicit real recovery OFF permits owner cleanup. Uncertain fault requests
are never retried; failed cleanup retains the existing owner and its receipts.

CPU tests use the real public Host/Runtime and torch model hooks with a small
scheduler fixture. The fixture supplies actual enqueue-state mappings and
executes its model only from `wait_for_completion`. Run from a full reviewed
checkout:

```powershell
python -B -m unittest -v test_rope_worker_core_probe
```

This experiment code and its scheduler test are deliberately outside the
source-only bundle. A new frozen source and actual Worker validation are
required before deployment acceptance. The pinned source46 reviewer still
describes source46; its missing proof evidence is not retroactively repaired
by the new probe. New independent review must verify full scope/owner/
generation/targets and recompute the original Coverage report hash after
removing its audit ledger and the six group report decoration fields:
`execution_boundary`, `mechanism_semantics`, `phase_rows`,
`scheduler_requests_claimed`, `raw_device_launch_payload_observed`, and
`independent_statistics`. Serialized proof receipts remain evidence only.
