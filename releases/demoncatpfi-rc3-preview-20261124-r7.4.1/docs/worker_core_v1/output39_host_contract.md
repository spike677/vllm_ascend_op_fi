# Output39 public Worker C1 and remaining route audit

The default C1 route below is retained. The separately selected
[complete-set C2/C3 route](output39_complete_set_contract.md) needs its own
new-source actual Worker acceptance and does not inherit earlier qualification.

This audit starts from `dc40a87d`, where five catalog entries still had
`WORKER_RPC_ADAPTATION_PENDING`. The following are separate mechanism and
deployment boundaries. Native/operator evidence does not grant a Worker route
or transfer qualification between variants.

| Entry | Existing real boundary and host driver | Current evidence and smallest missing contract |
| --- | --- | --- |
| `28.fa.legal_tiling` | `native_execution_strategy.StrategySession`; `experiments/c2c3_closure_20261117/attention28_fa_launch_probe.py` | Fresh standalone FA launch/argument path is described in `attention28_fa_launch_handoff.md`. It uses deterministic Q/K/V and mask, not a naturally scheduled Worker. A naturally observed FA call, current stock/private full-output bridge, source-bound route and owned scoped OFF are still needed before RPC adaptation. |
| `28.fia.legal_tiling` | `fault_injection.fia`; `worker_extension/attention28_prefill.py:FiaPrefillAudit` and `attention28_prefill_worker_probe.py` | Existing natural prefill adapter is read-only. BNSD metadata is only a candidate; current Qwen observations are TND, and the standalone policy build is CANN 9.1 rather than the inspected Worker 8.5.1 path. `strategy_request` does not enter Native scope. Actual compatible Native loading, complete values/launch, stock bridge and OFF remain missing. |
| `28.fia.denominator_replay` | Same read-only prefill auditor, separate coupled replay policy and key | Its denominator policy cannot inherit the legal-tiling arm or convert TND into BNSD. The same actual-call/SDK/stock bridge gaps remain, with its own lossless/F/OFF receipts required. |
| `33.paged_decode` | `paged33_tnd.py`, `paged33_boundary_probe.py`; `attention33_worker_probe.py` drives real prefill/first decode | The natural source48/53 family preserves actual query/KV and private launch records. The stock/private full-byte bridge remains a refusal, even when within-run repeats/OFF agree. See `attention33_baseline_difference_diagnostic.md` and `REVIEW_ATTENTION33_SOURCE53.md`. Do not turn same-vendor OFF into stock equivalence or admit F through a placeholder factory. |
| `39.output_proxy` | `PostwriteProvider` and `ProjectionCoverage` in `fault_injection/deployments`; previous `natural39_coverage_probe.py` is HF `MODEL_LOCAL` | Source51 review reconciles its MODEL_LOCAL boundary and limitations (`REVIEW39_SOURCE51.md`); it is distinct from standard-Worker `39.matmul_range`. Its existing actual projection-return/value-consumer/Native lease boundary can be reused for a new naturally scheduled **single-target C1** entry. This change implements that route without adapting MODEL_LOCAL coverage into Worker C2/C3. |

The registry now exposes 13 callable provider routes and leaves the four 28/33
entries pending. Enumeration never grants C1/C2/C3, E5, CANN 9.x/A3 or private
downstream integration acceptance. This new public output39 route requires its
own frozen-source device run and independent review.

## Public configuration and trusted deployment

```python
client = WorkerFaultClient(
    HostAdapter(existing_executor, prepare_binding=reviewed_host_binding),
    FaultConfig(39, 'output_proxy'),
)
```

The host binding invokes `install_fault39_output_serving_binding` on the actual
Worker selected from its returned identity. It supplies the existing verified
`NativePostwriteLibrary`, fixed prompt and private evidence directory. Binding
is inert: no Worker creation, hooks, Native lease or model invocation. Missing
bindings, wrong loaded Native bytes/ABI/inode and unsupported deployment contexts
refuse. `_simulation` and its readback double are explicit CPU contracts and
always retain `qualification_granted=false`.

The user supplies no layer, target, rank, address, loading policy or hook. The
existing Worker Runtime owns attachment, public owner and generation. The new
provider reserves only its binding ticket. Each action and real callback checks
the held Worker/runner/model, PID/thread/device and current owner; it does not
recursively call Worker identity RPC while Runtime's lock is held.

## Existing mechanism, real requests

`PostwriteProvider._observed_result`, `_qualify_observed_pair`, `_begin_run` and
`_complete_run` are extracted from its original `_replay/qualify/run` methods.
Those MODEL_LOCAL drivers still use the same code and keep their original scope.
`PostwriteObservedForward` only enters the existing `_ObserveDispatch` around
already executing root/logits calls; it never invokes a model. The original
Native finite-input checks, prefix selection, current storage/geometry, full
restore/readback/release and failure-retained leases are unchanged.

Six standard requests run in order: healthy observation, healthy repeat,
lossless, one F, actual OFF, healthy after the injection observer/Native lease
has been removed. The last request retains only the owned scheduler/root/logits
health hooks. Public finish removes those hooks. In the probe, unregister is
followed by one ordinary uninstrumented same-Worker request, then detach.

The first stable actual eligible projection/value-consumer pair is selected by
the original policy. There is no layer count, first-layer name or failed-target
fallback. All actual projection records, complete operator/module inventories,
aliases and unsupported calls are retained; this is not an all-eligible claim.
The original `aten.mm`, `aten.linear` and `aten.matmul` producer contracts remain.
The inspected `vllm.unquantized_gemm.default` has its own explicit return-storage
adapter; it is never renamed as MM. Other unreviewed lowerings, including
`npu_linear`, remain unsupported. If no actual eligible pair is visible, the two
healthy inventories end in an explicit refusal with zero F.

### Inspected custom GEMM and read-only consumers

The custom GEMM adapter requires its exact schema
`vllm::unquantized_gemm(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor`,
the current module's exact weight object, and no bias. The existing
`ReviewedGemmBinding` verifies these source SHA256 values on each actual call,
the loaded function code/globals, actual registration Library ownership and
PrivateUse1 dispatch. The current-registration guard is pinned to Torch
`2.9.0+cpu`: the complete PrivateUse1 list must contain just the original active
registration, with no inactive/overridden entries. The active implementation and
schema definition origins must agree with the exact file/line of the reviewed
`vllm_lib` construction. The raw dump, construction lines, current origin and
guard profile are recorded; the Library's historical `_op_impls` set alone is
not authority. New-Library and same-Library `allow_override=True` registrations
refuse. Matching a model name, tensor shape or weight values does not provide
this ownership. Arbitrary private C++/Python state manipulation is not a sandbox
isolation claim.

| Source | SHA256 |
| --- | --- |
| `vllm_ascend.patch.worker.patch_unquantized_gemm` | `0317441758008fa74133b2ee7d3e3b2588d632d6522acfb6a8c98ee21554e5bf` |
| `vllm.model_executor.layers.utils` | `1d908a5cf4ff519fdd987b7a8798070f57510d9031f8f9c2e013149cdb060cad` |
| `vllm.utils.torch_utils` | `b8c0a763d52c0eabb7b34c14b648b6f5e43f67d806f2819755b0ea23f4250221` |

The actual BF16 return must have the reviewed linear geometry and exclusive
storage. The Native transport still validates its original ND span and applies
the existing prefix proxy directly. This proves no native MatMul cache identity.

Two additional consumer adapters are limited to torch `2.9.0+cpu` and torch_npu
`2.9.0.post1+gitee7ba04`, as inspected without device execution in image
`35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
Both require the actual OpOverload, exact schema with no argument or return
alias/write annotation, and the inspected PrivateUse1 registration. Their Meta
shape source must have SHA256
`3e3997c2b99dcff7a4ee4d693bd7e2365f6cf7e906dee5e08585081c7ea13df4`.

* `npu.npu_rms_norm.default`: contiguous BF16 ND rank-two input, separate
  BF16 last-dimension gamma, finite positive epsilon; original returns are
  same-shape BF16 output and FP32 `[rows, 1]` rstd in separate ND allocations.
* `npu.npu_swiglu.default`: contiguous BF16 ND rank-two input, actual `dim=-1`,
  even last dimension; original ND BF16 return has exactly half that dimension.

Only the actual `input` object is a value-read operand. The existing selected
index proof checks its current storage, offset and stride, so disjoint slices
of the same allocation do not qualify; gamma aliases do not supply input
lineage. Actual source/schema/dispatch and return geometry are recorded per
call. Every custom GEMM remains in the independent inventory even when refused.
The original operation runs once; these checks execute no helper model or
consumer. Opaque mutating attention is not a consumer, and the existing schema
write check still rejects any write to a live selected lease before that call.

These are source/CPU contracts. They require a new standard-Worker run to grant
any C1; historical MODEL_LOCAL or matmul_range results grant no qualification.

Root and logits pre/post hooks must be the final local hooks at those boundaries.
They bind current positional/keyword arguments and the final actual return,
corroborated by `snapshot_fresh_prefill`, real enqueue IDs, complete prompt bytes
and the original logits source binding. The current root is entered manually
once inside its already-running pre hook; no newly installed root pre hook is
assumed to participate in that iteration. Every original call executes once.

Query is read-only. It checks owned hook tables, dispatch-stack presence and
the exact Native lease owner/generation/live receipt, and never restores,
collects, completes a request or runs forward. This direct Native ABI has no
callback registration table or process-wide lease enumeration; it does not
claim a fabricated global zero. Uncertain cleanup, foreign mode/hook ownership,
changed storage or incomplete real OFF retains the Runtime owner. Before/after
hook-removal and dispatch-pop failures support explicit cleanup retry after
actual resource readback. The first failure and partial inventory remain.

`clean` makes the provider inert (`registered=false`) while retaining its exact
deployment reservation until `unregister`. Repeated clean and later unregister
must preserve that owner relationship; a missing/foreign ticket or replaced
binding refuses without writing Native state. Missing deployment/Native
dependencies refuse during read-only resolver admission, before Runtime creates
an owner. An existing verified binding with a rejected prepare argument can
still unregister its zero-resource Runtime ticket. No fabricated Native-loaded
state or relaxation of Runtime's inert-state check is used for either path.

After a failed F, `recover` can restore only still-current owned storage, then
arms actual OFF and health requests. It never replays F. Resource release alone
does not turn the failed attempt into C1 success. C2/C3 and E5 remain false or
unassessed; no live C2 proof is manufactured or read from JSON.

## Root preparation and evidence

```sh
python3 -B "$SOURCE/experiments/c2c3_postwrite_20261117/prepare39_worker_core.py" \
  --source "$SOURCE" --native "$NATIVE_DIR" --sha256 "$LIB_SHA256" \
  --build-fingerprint "$BUILD_FINGERPRINT" --model /home/gyl/models/Qwen3-0.6B \
  --card 2 --out "$CASE_JSON"
```

Use a new reviewed source snapshot and matching packages. No Native ABI/vendor
change is required; the actual existing postwrite library still needs its exact
SHA and mapped-file verification. The case preserves the standard eager TP1
fixed fresh-prefill workload. A model name is not an eligibility certificate.
The probe's original loaded-file SHA checks include `gemm_lineage.py` and
`postwrite_npu_consumers.py`. Before the new registration profile is used in the
pinned image, root can run the device-free
`check_torch_dispatch_override_cpu.py --expected-torch 2.9.0+cpu` contract. It
registers and removes CPU-owned dispatcher metadata only and executes no tensor
operator. Local fixture runs on a different Torch version do not validate the
production version's dispatcher behavior.

`natural39_worker_core_probe.py` saves every original collective RPC response
individually, including identity/capabilities and exceptions. It performs only
the required actions/readbacks and one extra final public snapshot. The common
Host history semantics are unchanged. Complete request/partial-failure rows,
Host receipt, stdout/stderr and loaded-source hashes remain in `/evidence`;
public Git contains source, contracts and summaries only.

Baseline and F reports now carry `native_lease_receipt` with schema
`demoncat.postwrite.lease_receipt.v1`. It records the actual open address,
byte span and device; exact owner/generation/handle and tensor geometry;
apply mode, selected indices and changed count; restore/release return codes;
and the existing recovery readback for that same Native handle. The ordered
`events` use `operation`, `sequence`, `returned`, `return_code` and `error`.
Normal completion is open, apply, restore, release, live(false). OFF carries
null because it opens no lease. All snapshots are detached metadata and
perform no additional Native query or Tensor operation.

Failure evidence retains `last_native_lease_receipt` and its first failure.
A release exception followed by an exact absent-handle readback can safely
close the lease under the existing recovery rule, but its original exception
and missing return code remain recorded; retry cannot rewrite it as a
successful call. These originals let review join the selected actual output
and Native span without changing the ABI, mutation, or recovery decisions.
Their absence in an older profile is an evidence limit, not evidence that an
older device run failed.

`--statistics` refuses explicitly **before importing vLLM or creating a model**.
The current root/logits complete-value binding performs diagnostic Tensor
copies/hashes between the two boundaries; wrapping that window in mix statistics
would pollute the API record. The existing MODEL_LOCAL `model_observation` stop
point cannot simply be transplanted across both real Worker calls. A future
review can reuse `NativePostwriteLibrary.snapshot`/`postwrite_evidence` byte
readbacks to bind exact current ND spans without diagnostic Torch operators,
then stop the unchanged full API/module collector after final logits and before
CPU formatting/comparison. That requires its own value, collector-transparency
and recovery contracts. This change neither filters statistics nor chooses a
first-difference hint.

CPU tests use real Torch hooks/dispatch and the real Host/Runtime, with explicit
Native, scheduler and deployment doubles. They exercise six real requests and
post-unregister health, actual root-pre order, query without execution, complete
denominators, wrong owner/generation, changed model/storage, unknown deployment,
late callbacks, failed restore, uncertain fault-arm response and cleanup retry.
They establish no new Native or standard-Worker acceptance.
