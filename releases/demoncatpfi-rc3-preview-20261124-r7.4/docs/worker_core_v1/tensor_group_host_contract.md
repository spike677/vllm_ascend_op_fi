# Natural Fault8/18 public Worker entry

`FaultConfig(8, 'restore_offset')` and `FaultConfig(18, 'api_half')` now select
one `TensorServingProvider` through the existing WorkerFaultClient and Runtime.
The default routes require an inert `TensorServingBinding` before Runtime
creates a fault owner. Installing that binding does not acquire hooks, modes,
profile callbacks or target leases. Runtime remains the public owner authority.

Trusted deployment supplies the existing Worker, actual prompt token IDs,
weight manifest, workload/environment hashes and an explicit first-live-review
flag through `install_tensor_serving_binding`. There are no public layer,
operator, address, rank, row, build or proof selectors. The provider verifies
the held Worker/runner/model/process/thread/device, uses Runtime's actual
identity without recursive RPC, and reserves the binding at register.

`prepare`, `register`, `check`, `enable` do not invoke the model. After each
actual host `enqueue`, call `client.advance(actual_internal_request_id)` and
then the serving host's normal completion method. The expected request is
validated against the real fresh-prefill scheduler context. Hooks and dispatch
or profile scopes enter inside the actual root forward; actual logits return
closes them and retains the observed row. `query()` reads detached state only.
`collect_replay()` explicitly consumes the completed request and advances the
qualification/coverage protocol. It does not invoke inference.

The protocol uses the existing ServingTensorGroup and ServingMoEGroup algorithms:

1. Four actual requests: native baseline, observed baseline, repeat and lossless.
2. Complete independently qualified target set, `all_eligible` once, then real OFF.
3. Only the original live C2 proof object authorizes two every-match requests,
   followed by real OFF. The new source-bound public scope cannot reuse old
   JSON reports or historical natural/model-local certificates.
4. Owned fault/observer resources are removed before a final same-Worker healthy
   request. Full input, hidden and logits snapshots must equal the healthy
   reference, in addition to the outer host's token check.

The previous natural counts (113/226 RMS returns, 26/52 MoE calls) remain
historical expectations. They are not configured target counts or new-entry
qualification. The observed call objects, independent denominators, numerical
fault mechanisms, complete MoE permutation oracle, consumer checks, and common
CoverageSession proof/recovery authority are unchanged. No C1, C2, C3, Native
or E5 status is inherited. C1 is not separately claimed by this group adapter;
E5 requires its separate raw B/F comparator and cannot block safe recovery.

Failure and uncertain RPC replies retain the original owner and reservation.
Cleanup cannot pop active, foreign or wrong-thread scopes. A failure after a
group opens needs an explicit `recover`, at most one new `recovery_off` request,
then one `recovery_health`; fault requests are never retried. Partial hook
removal can be retried without losing already-retired handles. `clean` keeps
the exact binding reservation; only successful `unregister` removes it.
`remove_tensor_serving_binding` refuses a reserved binding. The caller then
detaches the existing Runtime ticket; no Worker lifecycle method is replaced.

The direct `TensorModuleProvider`/`install_tensor_replay` APIs remain available
for their explicit legacy `constructed_module`/`pretrained_model` scope.
NaturalTensorGroupProvider also retains its explicit model-local deployment.
The default 8/18 Worker routes never fall back to them when binding is absent.

CPU contracts run the real Host/Runtime and both existing Torch algorithms,
with explicit CPU scheduler/identity seams. They test complete groups/live
proofs, query purity, no model RPC, owner/clean/unregister semantics, missing
binding, empty prepare failure, busy/foreign scopes, lost replies, partial
cleanup, and actual OFF/health after a failed fault. They are not NPU evidence.

Root prepares two cases without executing them:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/prepare_tensor_worker_core.py" \
  --source "$SOURCE" --card 2 --out /home/17-fault-c2c3-20261117/tensor-core-cases.json
```

The cases use Qwen3-0.6B for Fault18 and DeepSeek-V2-Lite-W8A8 for Fault8,
CANN 8.5.1, one eager Worker, physical card 2 or 5, 1800 seconds maximum,
64 GiB/8 CPUs and read-only source/model mounts. The single root queue must
apply its current image, idle/live-resource, source/package and recovery
gates. No new Native DSO or SDK build is required. Full original RPC/phase
receipts remain in each `/evidence/probe`, including failed attempts.

Source-only export includes this contract, both current probe/preparation
scripts, and the finite existing CPU fixture dependency closure. It includes
no weights, tensors, actual queue JSON, statistics or historical receipts.
