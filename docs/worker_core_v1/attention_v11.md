# V1.1 Attention: actual-call public entry, first CPU increment

This implements the owned actual-call boundary described by
[ExecutionAdapter contract](../contracts/execution_adapter_contract.md). It does
not replace the common Runtime/Host, registry, exporter or CoverageSession.

## Registry integration

The four constructors live in
`demoncat.runtime.vllm_ascend.worker_extension.providers.attention_serving`:

| Mechanism | Constructor | Actual API/Native semantic |
|---|---|---|
| `28.fa.legal_tiling` | `FaLegalTilingServingProvider` | Actual FA v3; FA S1/S2 `m`/`n_ratio`, separate ABI |
| `28.fia.legal_tiling` | `FiaLegalTilingServingProvider` | Actual FIA; healthy coupled M512/M128 templates |
| `28.fia.denominator_replay` | `FiaDenominatorReplayServingProvider` | Actual FIA; M128 reduction replay, separate build |
| `33.paged_decode` | `PagedDecodeServingProvider` | Actual TND/page16 first decode; FAI split/fuse S2 |

Each constructor takes `(worker, model, identity)` and implementation version
`1`. Publish four distinct capability rows, with execution
`trusted_actual_attention_census_and_matched_native_baseline`, attachment
`owner_scoped_actual_model_module_api`, and `natural_model_verified=false`.
The held deployment chooses the honest execution boundary: `python_hook` when
only observing, `native` only after the exact reviewed Native object is present.
Registry availability is not F admission or C1/C2/C3 qualification.

Trusted code calls `install_attention_serving_binding` with the loaded Worker,
actual prompt and provenance. Omitting `native` is useful zero-F observation; it
does not load a DSO or claim `native_loaded`. The optional concrete Native
adapters are `FaNativeCall`, `FiaNativeCall`, and `PagedNativeCall` (the latter
wraps the existing `TNDNativeAdapter`). No RPC accepts a callback, path, QKV,
module selector, strategy or evidence flag.

## What is executable now

`prepare/register/check/enable` install owned model/module hooks and public API
wrappers. `advance(request_id)` queues an expected actual scheduler ID; it does
not run inference. A normal enqueue supplies the model call. `query` reads
state only; `collect_replay` explicitly collects the finished request.

The bounded sequence is `observe`, `observe_repeat`, `baseline`,
`baseline_repeat`, then a **closed stock/private bridge gate**. Discovery-only
deployments can finish after the first two requests. Every actual observed
Attention call, including unsupported APIs/layouts/options, remains in the
denominator. The first increment can observe all four routes and execute a
matched zero-F Native baseline. It cannot grant or enter F. Further bridge
authority must be implemented/reviewed against new actual evidence; serialized
booleans and historical qualifications do not bypass the gate.

Root/module/logits completion is checked against the actual last hook, current
thread, Worker/model and scheduler objects. Inputs and original outputs are
hashed completely; no raw tensor arrays are persisted by this first probe.
This is source-bound evidence, not an independent full-array review or E5.
Original arguments and returned objects are preserved and the original API is
called exactly once. The code never calls `execute_model`/`model.forward`,
transposes QKV, constructs a replacement mask or repacks KV for eligibility.

The common observer uses the already reviewed exact vLLM Attention/Ascend
implementation source pins. A new source/class fails explicitly. Structural
aliases do not become actual calls; an ambiguous shared logical parent remains
unadapted. Existing unrelated hooks are not removed.

Recovery retains partially installed hooks/wrappers, Native scopes and binding
tickets. A clean operation retains the ticket until unregister. Setter/remove
exceptions retire only resources whose actual state is confirmed restored.
Foreign wrapper/thread/model changes refuse cleanup; an ordinary failed
request with proved zero resources does not permanently lock the owner.

## Native build boundary

The new public adapter requires fresh observer exports
`dcat_attention_observer_scope_active_v1` and
`dcat_attention_observer_scope_matches_v1`. They read only the current thread's
TLS token. A failed end retains ownership; an after-end exception can retire the
resource only after actual zero readback. A replaced foreign token is never
ended. Old observer artifacts do not acquire these guarantees retroactively.

FIA additionally requires the newly source-bound optional
`FAULT28_FIA_GRAPH_TID` guard: public calls must match both the current PID and
native thread, rather than using the old ancestry-only process scope. The
constructor checks the fresh policy receipt and exact patched header bytes.
Historical standalone callers without this variable retain their old contract.
This change needs an independent fresh Native build/CPU compile check; no NPU
acceptance is claimed by the current Windows contracts.

The FA/FIA/33 helpers reuse fresh artifact, SDK, ABI, host and final-launch
verifiers. The old FiaBackend `.run()` path is deliberately not called: it would
materialize contiguous QKV, construct a mask and invoke a new API. FA never
decodes a FIA entry; stock33 never uses a private ABI as a guess.

## Fault33 known bridge failure

Source53's independently read original normal arms had equal first-decode
query bytes (4096 B), logical K/V hashes (2,099,200 B each), mask, block table
and scalars. Their **immediate original API output** already differed by
67/2048 FP16 elements (maximum absolute difference 6.103515625e-5). Each arm's
repetitions were stable. Later mutation is not an explanation for that first
difference.

Actual stock launch: `aclrtLaunchKernelWithHostArgs`, key
`103000000010300003`, block16, schedule1, args2032. Private launch: key
`5100000000010200103`, block17, schedule1, args2960. Their registrations and
current-call joins were resolved offline; the original reported observation
gap remains preserved. Selected ACLNN library ownership was not observed.

Therefore the Native route/reduction/rounding boundary is the next adapter
target, not a Python output adjustment. Required next evidence is a fresh,
source-bound stock launch/selected API-owner trace and a private **normal**
route matching that actual reduction schedule/rounding, followed by complete
stock/private output comparison. CPU geometry/ABI checks can reject invalid
routes but cannot prove NPU arithmetic equality. F remains blocked. The
current implementation does not change tolerances, raw data or the old bridge.

## Short zero-F entry

After the root integrates the registry mapping, run only through the root's
existing serialized NPU queue, in the reviewed standard Worker environment:

```sh
python -B experiments/c2c3_closure_20261117/natural_attention_worker_core_probe.py \
  --mechanism 28.fia.legal_tiling --model /model --out /evidence/attention-census
```

The same CLI accepts the other three explicit mechanisms. It uses two measured
normal requests and one ordinary request after finish/disable/unregister/detach
(plus a warm healthy request). Fault33 uses a 1024-token prefill and first
decode; 28 uses a 2048-token fresh prefill. Requests, scheduler mappings,
complete source-bound hashes, original RPC files, host receipt and actual
cleanup are preserved. Post-unregister health is token-only. The probe loads
no custom Native adapter and enters no fault scope. No test in this increment
opens SSH, creates a container or starts an NPU run.
