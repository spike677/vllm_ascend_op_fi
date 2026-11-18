# Native26 public Worker single-site C1

This adapter starts from `37bbeb4f`. It adds an explicit
`Writer26CoreBinding` branch to the existing `fault26.py` providers. The
Native26 ABI, `NativeWriter26Group`, `Writer26Experiment`, deterministic
single-site policy and `Serving26GroupExperiment` remain the implementation.
The new entry needs its own frozen-source standard-Worker run and independent
review. Direct experiment results from source54/source62 do not qualify it.

```python
client = WorkerFaultClient(
    HostAdapter(existing_executor, prepare_binding=reviewed_deployment_binding),
    FaultConfig(26, 'residual_stomp'),  # or 'fia_destination'
)
```

The user supplies only the fault ID and variant. The trusted host policy routes
to a complete identity returned by the existing executor; multiple Workers
require explicit trusted routing. The deployment calls
`install_writer26_core_binding` on that actual Worker with its fixed prompt,
private evidence directory and reviewed existing group bundle SHA. It checks
already loaded Native bytes/ABI/readback and a serial eager TP1 eval context.
Binding installs no hooks, Native owner or Worker, and invokes no model.
Unknown deployment refuses before Runtime creates a lease. FIA requires the
current typed stock bridge; it cannot use the legacy FIA descriptor path.

The existing Runtime creates the public ticket and its private provider owner
namespace. That provider owner and the original generation are passed to the
existing Native group constructor. Full Worker/runner/model, PID/thread/device,
binding reservation and Native owner checks run at actions and real callbacks.
They never call identity RPC recursively under Runtime's lock. Raw RPC replies
bind public ticket, provider owner and current Native owner words/generation;
`public_core.owner` is the Runtime-assigned **provider owner namespace**, not
the host's display owner string. No JSON record grants runtime authority.

## Request protocol

`prepare -> register -> check -> enable` only prepares the existing observation
and effect machinery. Each ordinary host enqueue supplies its actual internal
request ID to `advance`, followed by the serving engine's normal execution.
RPC code does not call forward, enqueue, restart, reload or replace serving
methods. Root/logits hooks bind the current args/kwargs and must be the final
local hooks, so earlier or later return replacements cannot be mistaken for
the actual output. Every original producer/writer/consumer call remains once.

Eight complete fresh-prefill requests retain the existing plan:

1. Two independent healthy inventories, retaining every call and alias.
2. A fresh Reference, Lossless and OFF for the deterministic minimum key among
   actual twice-healthy resolved candidates.
3. Exactly one F, followed by actual OFF and full-vocabulary health after Native
   removal in the same Worker.

The single-site plan has one target; it does not truncate the inventory or use
`max_targets` as a selector. All 135 QMM calls, or all 30 FIA calls including an
unproduced/unresolved first call when observed, remain in the actual denominator.
Those counts describe tested workloads, not hardcoded admission rules. Current
objects, storage, allocator adjacency, typed descriptors and the actual next
consumer are checked for every admission. FIA retains both complete stock byte
samples, new Reference equality, original NCL format 47 and full current typed
snapshots; no old Reference or ND storage relaxation is used.

After each actual request, `query` provides read-only current Native state and
marks collection pending. `client.collect_replay()` explicitly uses the
existing `advance` action with `collect=true`; it closes completed observers
and writes a new complete immutable observation file. A new arm without this
collection fails and enters the original recovery gate. There is no automatic
fault retry or candidate fallback. `finish` requires all eight requests and
recovery, removes owned hooks and records the zero-resource C1 report.
`disable -> unregister` releases the reservation. The probe then enqueues one
ordinary uninstrumented same-Worker health request and detaches.

## Readback, failure and evidence

Query reads Native owner/generation, pending/executing, descriptor, host/device
bytes and callback counters at that moment. It checks owned hook tables and
original-file hashes. It never collects, saves, restores or advances a request.
RPC evidence contains the complete candidate/request indices and original-file
path, byte count and SHA256. Full Native events, all candidate rows, aliases,
call packets, typed records, public outputs and failures remain in private
`core-evidence/*.json` originals, as well as the existing writer inventory and
typed stock artifacts. Pending or later changed state is explicitly marked as
not captured by the previous collection. Changed/missing original bytes refuse
readback instead of being silently replaced. The common Host history is unchanged.

Clean makes the provider inert but keeps its exact binding reservation until
unregister. Missing/foreign tickets, changed generation/model/storage, uncertain
Native state and failed cleanup retain Runtime ownership. A rejected zero-resource
prepare can unregister. Hook removal failures are reconciled only by actual hook
table identity. A Native close exception after actual removal is acknowledged
only when the original owner/generation/thread's readback proves every resource
zero; the original error and RPC owner remain until explicit retry. Before-remove,
foreign or unavailable readback keeps the resource ledger.

Failure uses the existing stop/drain/current-storage restore/readback/release
path. It retains the first failure and original candidate; cleanup does not
turn a failed attempt into C1 success. C2/C3 are always false here. E5 remains
separate; `--statistics` refuses before model creation because diagnostic
copy/hash reads have no reviewed mix-statistics isolation in this entry.

The old `FiaServingBinding`/`ResidualServingBinding` compatibility paths remain
distinct. Their provider query no longer implicitly calls collect; both in-repo
legacy Worker probe callers now issue the explicit collect action. Their existing
mechanism gates and evidence scopes do not gain new public-Core qualification.

## Root-owned deployment command

The case generator only prepares a reviewed case; it does not run a device.
Use the current source snapshot and existing reviewed Native-v2 bundle:

```sh
python3 SOURCE/experiments/c2c3_closure_20261117/providers26_prepare_group_cases.py \
  --installed PACKAGES --native NATIVE --source SOURCE \
  --probe SOURCE/experiments/c2c3_closure_20261117/providers26_worker_core_probe.py \
  --source-commit COMMIT --worker-core --stages c1_single_effects \
  --mechanisms 26.fia_destination --max-targets 1 --card 2 --out CASES.json
```

For QMM select `26.residual_stomp`; the existing generator keeps DeepSeek QMM
and Smol typed-FIA workloads, same prompt/weights, one old warmup and fixed event
capacity. It mounts the source snapshot read-only, verifies the source manifest
and actual loaded `/installed` module SHA values, and records every original
RPC, final Host receipt, private observations, stdout/stderr and post-unregister
health. No old direct-entry receipt is relabeled. A missing group-v2 build or
typed profile refuses; the legacy NativeBuildManager 26 recipe does not itself
build this group-v2 bundle.

CPU verification uses real Torch hooks and the actual Host/Runtime with explicit
Native/scheduler doubles. It grants no CANN, A2/A3, NPU or downstream-framework
acceptance. The source-only exporter includes these CPU fixture dependencies
and contracts, not deployment experiment history, tensors, statistics or binaries.
