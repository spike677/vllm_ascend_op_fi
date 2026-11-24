# Fault41 live cache group through Worker Core

`WorkerFaultClient(HostAdapter(existing_executor), FaultConfig(41, 'tail_column'))`
uses the existing WorkerFaultRuntime ticket, Postwrite41WorkerProvider,
KVGroupCoverage, CoverageSession and NativePostwriteLibrary. Factory discovery
means that the bridge is available; it grants no C1, C2, C3 or E5 qualification.
This public path needs its own reviewed source snapshot and natural Worker
acceptance. Previously accepted direct-group results do not qualify it.

## Trusted existing Worker binding

The reviewed serving host installs `serving41.install_fault41_serving_binding`
on the actual Worker selected by HostAdapter. The inputs are the existing
verified `NativePostwriteLibrary` transport, fixed prompt token IDs, complete
`CoverageScope`, bounded `max_seconds`, and explicit
`first_live_validation=True`. The scope must name `41.tail_column`, include the
reviewed workload/environment/implementation/model-weight/model-source hashes,
and have `evidence_kind='natural_worker'`. Loading/building that transport stays
in the existing reviewed deployment and NativeBuildManager workflow.

Installation only binds the held Worker/runner/model and loaded transport.
It opens no Native lease, installs no hook, starts no Worker, executes no model,
and returns `SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED`. Missing bindings are refused
by the inert provider factory before Runtime can create an unusable Native
owner. Multi-Worker selection still requires an explicit trusted host identity
policy; no default rank selection is introduced.

Public RPC settings cannot provide a layer, target, device, pointer, transport,
factory or serialized proof. Only `advance` accepts the actual enqueued request
ID. Private CPU seams require `_simulation=True`, explicit constructor and ABI
readback doubles, and `evidence_kind='cpu_contract'`.

The original adapter discovers the actual attention modules, aliases, K/V
cache objects, writer calls and FIA consumers. Its source fingerprints, TP1
eager single-request scheduler requirements, live slot metadata and Native
BF16 contiguous-span gates remain mandatory. There is no target fallback or
user layer selection. The mechanism changes the last two legal V-slot columns
per head after the real cache writer and before the real consumer. It remains
a controlled legal-address proxy, not native out-of-bounds corruption.

## Public actions and real serving requests

Call `prepare`, `register`, `check`, and `enable` (or `execute`) once. No action
runs a forward. For each row, the host enqueues the fixed reviewed fresh
prefill, obtains its real internal ID, calls `client.advance(id)` before
execution, and lets the existing scheduler complete the request. Each request
must reach the original adapter's first pure decode boundary. Direct model
replay from an RPC is unsupported.

| Natural requests | Existing protocol and gate |
| --- | --- |
| 1–2 | Repeated C1 healthy observation of actual writer/cache/consumer inventory |
| 3–5 | C1 lossless, one actual fault with downstream effect, complete OFF |
| 6–7 | Repeated group observation; freeze the entire actual eligible target set |
| 8–12 | All-target lossless; C2 once over two requests; OFF; normal health after Native/API/hook uninstall |
| 13–17 | All-target lossless; C3 every-match over two requests with the original live C2 object; OFF; normal health after uninstall |

The next action consumes only the previous request that actually returned.
`query` never consumes a request, advances a phase, restores a tensor or removes
a hook. Missing execution cannot manufacture a hit or healthy output. Root
and health callbacks must observe the final arguments and return. A later
input- or return-transforming hook is refused and preserved. Original writers, consumers
and model roots execute once per actual call; the bridge never replays them.

After request 17, call `finish`, inspect the complete C2/C3/Recovery evidence,
then `disable`, `unregister`, and `detach`. Execute a separate ordinary serving
request after public unregister and check its same-Worker output. This last
request is a host-owned acceptance check; `detach` is not a health certificate.
Check every action and retain its original response. No uncertain action is
automatically retried and a fault is never retried.

The provider uses the current identity already verified by Runtime and checks
the held Worker/runner/model, PID, thread, rank, configured/observed device and
visible-device mapping. It does not recursively invoke an identity RPC under
Runtime's lock. Callback ownership also includes the exact provider install
epoch, so retained callbacks from an earlier C2 install cannot enter C3 or a
new owner. Existing direct Postwrite41WorkerProvider behavior remains separate.

## Readback and original proof receipts

The Native implementation has direct `open/apply/restore/release/live` ABI
leases, not a callback registration API. Readback therefore reports
`native_registered=false` with
`native_registration='NOT_APPLICABLE_DIRECT_ABI_LEASES'`. `native_loaded` is
based on the exact held library object, actual ABI, reviewed file SHA, loaded
mapping and available span reader. It is not inferred from Python registration.

`native_lease` reads the exact Native handle with its actual Native owner and
generation. `leases` counts retained current Native wrappers;
`coverage_held_leases` separately reports original CoverageSession reservations.
`owned_hooks` reads actual hook maps. `owned_api_state` walks the existing owned
API/collector chain in read-only mode. It never calls its edit operations.
These distinct resources are not reported as an invented Native registry count.
Uncertain or foreign readback retains the public Runtime owner.

The group owner is Runtime's provider owner. Its C2/C3 generations advance
inside the one public RPC generation. The direct Native adapter keeps its own
original owner/generation scheme; `coverage.deployment` joins these identifiers
to the public binding, transport SHA and bridge source hashes. It does not
replace the Native ownership checks.

`coverage.C2.live_c2_proof.issued` and
`coverage.C3.live_c2_proof.consumed` retain the original CoverageSession audit
receipts: scope, target IDs, owner, generations, discovery/observation/recovery/
report hashes, proof SHA and exact process-local object identity. C3 privately
consumes the original issued object; copied objects and JSON cannot become
authority. `report_sha256` is the undecorated CoverageSession report excluding
only its live-proof audit ledger. Group-report decorations are not part of that
original CoverageSession hash. The bridge keeps complete reports and actual
request observations rather than reconstructing a smaller qualification summary.

The complete actual denominator and full valid K/V prefixes remain mandatory.
Even C2's once-exhausted pass-through must match the complete prefixes and
consumer control. C3 requires every eligible match. A healthy root value alone
cannot hide an altered cache, alias, call count or incomplete target set.

## Explicit recovery

An actual root failure, foreign callback, changed current object/storage,
uncertain ABI response, cleanup failure or health mismatch retains ownership.
`recover` uses the original Native and group recovery implementation. A failed
C1 attempt requires a new real C1 OFF request; a group failure requires its real
full-prefix OFF and same-Worker healthy request before all reservations release.
If OFF already succeeded, recovery retries cleanup and schedules only another
real health request. No recovery action executes a model or retries a fault.
Failure history and failed request evidence remain in the public report even
after resources recover. Recovery is separate from successful full coverage.

Hook removal is verified against its actual owner map. A before-remove failure
retains the handle; a confirmed removal followed by an exception retires only
that absent handle while preserving the failed response. Foreign hooks/API links
are never overwritten. Native release follows the same rule using the existing
`live(handle, owner, generation)` ABI: only exact zero confirms that one lease
is absent. Other leases do not authorize this conclusion. Failed readback or a
foreign owner retains the wrapper. An explicit retry first resolves uncertain
release, so it never restores a handle already confirmed absent. The original
exception and Runtime owner survive the first uncertain action.

Binding removal refuses any public Runtime owner still retained, including
when host registration was removed but final Native readback was unavailable.
With no owner and all resources removed, normal serving has no managed-request
requirement. E5 still needs its independent raw-statistics workflow.

## CPU verification and source-only export

From the reviewed source root:

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
$env:TORCH_DEVICE_BACKEND_AUTOLOAD='0'
python -B -m unittest test_core_kv41 tests.test_postwrite_native_recovery
```

The public tests use real HostAdapter, Runtime, Torch hooks and the existing
group/Native lease wrapper, with explicit CPU scheduler/source/ABI doubles.
They cover the complete 17-request protocol, proof object binding, exactly-once
calls, full prefix changes, health mismatch, foreign owner/generation, late
callbacks, no-forward RPCs, and recoverable cleanup failures. ABI boundary
tests include restore/release failures before and after the operation and
readback failure after actual release. These tests grant no NPU qualification.

The source-only integration exporter includes this contract, the public tests
and their existing CPU fixtures when it includes the provider. It ships the
original three distributions once, with source paths and SHA256 manifests; no
experiment history, Native binaries, model weights, arrays or raw statistics are
added. Inner-source adoption, CANN 9.x, A3 and 17/17 acceptance remain unclaimed.

## Root-scheduled public standard Worker case

`prepare41_worker_core.py` reuses the existing fixed prompt, Qwen3-0.6B
safetensors loader, TP1 eager first-decode workload and bounded resource plan.
It only writes one case; it neither executes the command nor queries a device.
Supply the reviewed Native directory and its actual build receipt:

```sh
python3 experiments/c2c3_postwrite_20261117/prepare41_worker_core.py \
  --source "$SOURCE_ROOT" --native "$NATIVE_DIR" \
  --sha256 "$LIB_SHA256" --build-fingerprint "$BUILD_FINGERPRINT" \
  --model /home/gyl/models/Qwen3-0.6B --card 2 \
  --out "$CASE_JSON"
```

The directory must contain the reviewed `libdcat_postwrite.so`. No layer or
expected layer count is accepted by the new probe. The actual complete
writer/cache/consumer inventory determines the group denominator.
`natural41_worker_core_probe.py` uses public actions for all 17 managed requests,
then enqueues one ordinary request after unregister/detach. It verifies the
same actual Worker identity and healthy generated tokens/text. Both group
health requests additionally retain the existing full root-output equality.

Full query/action replies are stored once under `/evidence/probe/worker_core_rpc`,
indexed by SHA in `result.json`; `host_receipt.json` retains the complete client
history. C2/C3 reports include the original live-proof issuance/consumption
ledger, and each request keeps its entire actual writer/cache/consumer record.
The case also retains stdout/stderr in `/evidence`. These private artifacts stay
on the serving host. They are not bundled into public source history.

Failed execution keeps its original error/receipts and permits at most one
explicit real recovery OFF followed by health. It never retries a fault or an
uncertain fault advance, and recovery cannot turn the run status into success.
Failed cleanup keeps the existing Runtime owner. E5 statistics remain separate
from this public lifecycle acceptance. The CPU orchestration contracts run with
`python -B -m unittest tests.test_kv41_worker_core_probe`; the new case requires
independent actual Worker review before any public Core41 acceptance claim.
