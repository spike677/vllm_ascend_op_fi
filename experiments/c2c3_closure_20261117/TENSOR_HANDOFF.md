# Tensor workflow handoff, 2026-11-17

All NPU/container execution is root-queued. This workflow performs source work,
CPU contracts and read-only inspection. All new raw statistics stay on 124.

## Reviewed handoffs already sent to root

- `89faba5`: independent eager tensor census and real HF/Worker/VAE loaders.
- `ba5dd46`: canonical eager/native-lineage RMSNorm return proxy, real advanced
  RoPE lookup/materialization lineage, independent bounded model-local groups.
  CPU 43 tests passed including old C1 contracts and common coverage checks.
- `50a822b`: actual B/B-repeat/F capture at a common callsite, complete OFF,
  exact reviewed unmodified msprobe public start/stop pauses around provider
  diagnostics/auxiliary proxy arithmetic, and the raw-only strict comparator.
  CPU 21 tests passed. No E5 claim before the root-queued capture is inspected.
- Shared commits were cherry-picked for development only: 6363182, 7361d44,
  8f940be and 28a3ca7. Root already owns them; do not re-pick their local copies.

Root's planned HF18 command for the reviewed handoff is:

```
python experiments/c2c3_closure_20261117/natural_tensor_group_probe.py \
  --mechanism 18.api_half --model /model --out /evidence/probe \
  --c3-forwards 2 --statistics
```

This is natural pretrained HF inference, MODEL_LOCAL, not a standard Worker.
The live C2/C3 proof binds actual weight-manifest and model-source hashes. The
proxy halves the actual RMSNorm returned value, never a Reduce internal value.
No public qualification count was changed by this workflow.

## Actual HF18 model-local result and observable boundary

Root completed `natural18-group-a2/natural-group-18/probe/result.json`: 113 actual
RMSNorm targets, C2 113/113, C3 226/226 across two forwards, exact full-model OFF,
and no owned hooks, leases or in-flight calls after recovery. This remains
natural pretrained MODEL_LOCAL evidence, not standard Worker evidence.

The a1 capture retained a construct mismatch from observer-owned Tensor.stride
queries. Commit `88b403c` put those metadata queries in the same reviewed public
collector pause as the provider diagnostics. Root recaptured a2; no existing raw
JSON was edited and no construct field was ignored. Its independent B/B-repeat
dump, stack and construct files are byte-identical. Root's independent Decimal
audit at `/home/17-fault-c2c3-20261117/natural18-group-a2-baseline-fault` reports:

- first observable difference: `Functional.linear.0.forward/input_args/0.Max`,
  1.734375 -> 0.8671875, relative difference 0.5;
- 84 preceding null cells, 9796 null cells in all, zero incomparable cells;
- E5 observable under unchanged rtol=1e-5, atol=1e-8. Null is not equality.

This is an actual downstream consumer input, not localization inside the RMSNorm
operator. The reviewed msprobe `module_processor.py`, SHA256
`4876e89f7f726cd552fc1bf0e7b7745a898461deec2cb2abf33b02deba7329d7`, inserts its
module forward hook at the front of `_forward_hooks` (lines 236–240). The owned
RMSNorm return-gain hook runs afterwards. Thus the collector's module-output
record can still contain the original return, while the actual next consumer
receives the half-scaled value. The proxy arithmetic is an excluded owned
diagnostic operation; no artificial tensor was added to become the first
difference. The result supports output-half propagation only, never native
Reduce-internal corruption or an earlier RMS operator first-difference claim.

The original a1 remains preserved with its failed E5 alignment. Root separately
audited its C2/C3 counts and all 229 canonical-object SHA references at
`/home/17-fault-c2c3-20261117/natural18-group-a1-coverage-audit`.

## Actual natural observations supplied by root

Worker result:
`/home/17-fault-c2c3-20261117/natural-census-a2/natural-census-worker/probe/result.json`.
Two exact healthy forwards. Per forward: 57 `npu_rms_norm`, 56
`_C_ascend.npu_add_rms_norm_bias`, 28 `vllm.npu_rotary_embedding`.
RoPE schema is `positions, query, key, cos_sin_cache, head_size, rotary_dim,
is_neox_style, offsets`; outputs alias query/key. No ordinary row lookup was
observed, so the current 22 index-select/advanced-index adapter cannot claim
this fused path. It needs a separately reviewed actual reader/implementation.

VAE result:
`/home/17-fault-c2c3-20261117/natural-census-a2/natural-census-vae/probe/result.json`.
Two exact healthy forwards. Two-run counts: Conv3d 288, GroupNorm 240,
format30 select views 1536 and slice views 24. CANN headers identify format30
as base `ACL_FORMAT_NCDHW`, not private NZ. Therefore this is not Fault15's
qualified private-format path. An exploratory NCDHW detachment implementation
was removed without publication or qualification. No model writes were added.
VAE Conv3d uses k3/stride1 after causal padding; the existing p=(0,7,7) patch16
mechanism would change spatial shape. No output cropping or new geometry was
introduced to force Fault17 to pass.

Read-only inventory found Qwen3-0.6B, Qwen3-32B W8A8, DeepSeek-V2-Lite W8A8,
GLM-4-9B, SmolLM2, HunyuanVideo VAE and a DeepSeek-R1-Distill-Qwen directory.
No installed Qwen-VL or other genuine visual patch16 weights were found in the
inspected model roots. DeepSeek MoE remains the next genuine Fault8 census.

## Worker18 implementation ready for a root-queued runtime check

Files:

- `serving_tensor.py`: genuine enqueue/scheduler identity, actual model root and
  logits boundaries, per-request owned adapters, independent observation,
  coverage groups, full hidden/logit OFF and removal of host hooks before
  zero-resource health.
- `natural_rms_worker_probe.py`: 4 healthy actual requests, C2 fault/OFF,
  C3 two fault requests/OFF, and same-Worker healthy generation after cleanup.
- `RMSNormReturnSession`: opt-in post-root **read-only** consumer observation,
  including actual last-token advanced indexing to logits GEMM. No RMSNorm
  target outside the model root can be injected by that path.
- `NaturalTensorGroupProvider`: factored healthy observation join/group open
  and close for a serving host; added INITIALIZED_WORKER scope/request mapping.

`tensor_callable.py` now binds both the original bound `forward` and actual
bound `_forward_method`, including function/code identity, owning instance and
module globals, exact source-file hash and defaults. A changed dispatch method
cannot retain qualification through unchanged `CustomOp.forward` text. Only
the separately reviewed, exact msprobe wrapper can be traversed; generic
`__wrapped__` or unreviewed closure callables are refused.

Executed CPU controller contracts cover four real healthy forwards, C2 fault
and OFF, two C3 fault forwards and OFF, post-root actual last-token indexing and
logits GEMM, full recovery, callback ownership, target implementation drift and
Worker identity drift. These fixtures explicitly declare CPU simulation and
never report real Worker evidence. The actual `_C_ascend` three-return schema
is also exercised with CPU dispatch: only the normalized output is halved;
the actual residual tensor object is preserved.

The focused CPU regression passed 51 tests: serving controller, callable guards,
natural RMS/row-reader groups, observer/statistics windows, old C1 tensor
providers, route8/index22 contracts and shared coverage-session contracts.

Root may queue:

```
python experiments/c2c3_closure_20261117/natural_rms_worker_probe.py \
  --model /model --out /evidence/probe
```

Use the existing genuine vLLM-Ascend 0.18.0/CANN8.5.1 Worker image and Qwen3-0.6B
model. This probe still needs actual RPC-thread/mode ownership, live dispatch
source and final-norm consumer validation. A conservative gate refusal must be
kept, not reclassified as a pass. A separate standard Worker statistics span is
still required; HF E5 does not certify Worker capture. The original WIP commit
`a33a874` plus the subsequent tested hardening form this handoff, and do not
replace any immutable source snapshot already executed by root.

## Fault8 genuine DeepSeek no-injection census handoff

The existing `/home/haoran/models/DeepSeek-V2-Lite-W8A8` has four actual dynamic
W8A8 safetensor shards totaling 17,214,938,536 bytes (about 16.03 GiB). Config
SHA256 is `229913ea2a346ccdb571bb3cb23414ca3a0ee1a9455fe88a15a5788bc837cb75`:
27 layers, first dense layer 1, MoE frequency 1, top-k 6, 64 routed experts and
two shared experts. Config implies 26 routed MoE modules; this is an expected
structural count, not an observed eligible denominator. The probe hashes all
four shards and the quantization description before starting the Worker.

Root may queue one no-injection case:

```
python experiments/c2c3_closure_20261117/natural_tensor_census.py \
  --kind worker --model /model --out /evidence/probe --memory .55 --moe-trace
```

Use the already validated vLLM-Ascend 0.18.0/CANN8.5.1 image, actual model mount
read-only, TP1, max sequences 1, max prompt/model length 128, 64 KV blocks and
the existing host-driven Worker environment. Memory .55 matches the earlier
DeepSeek QMM queue preparation default. It is a resource bound; no routing,
expert count, top-k, layer count, geometry or implementation is changed. Root
controls the shared lock/card selection and may budget 1800 seconds and 64 GiB
host memory for loading/hashing the actual 16 GiB weights. This workflow never
starts a container or takes an NPU lock.

Read-only inspection of the existing stopped `provider-build-851-20261116`
container identified the real source path:

1. `experts_selector.select_experts` returns actual top-k weights and IDs.
2. `MoECommMethod.fused_experts` passes them through routing preparation,
   calls `TokenDispatcherWithAllGather.token_dispatch`, then the real quantized
   MLP, then `token_combine` with its unchanged combine metadata.
3. `token_dispatch` calls `DeviceOperator.npu_moe_init_routing` to produce
   `expanded_row_idx` and expert token groups.
4. `token_combine` computes `abs(expanded_row_idx)` and calls the real
   `npu_moe_token_unpermute` with MLP output, those indices and the same
   `topk_weights`. Its output is the routed result, before shared-expert merge.

Exact reviewed source-file hashes:

| Source beneath `vllm_ascend/ops/fused_moe` | SHA256 |
| --- | --- |
| `experts_selector.py` | `92bebd721ecc86a7033065e37e73fe57facabc01cbd754939c014be8ebb810a8` |
| `moe_comm_method.py` | `a23d681549fa24d69c597f6c1c7d22df84da62efea808f2415a260c4267a8c4f` |
| `token_dispatcher.py` | `21fd94e3d5eacc578cc3feb794bc5c135c148c2d495c611f224f22c8b96ac0cc` |

`NaturalMoETrace` validates these exact installed sources and loaded functions,
then observes their actual Python frames using an owned `sys.setprofile`
callback. The outer `vllm.moe_forward` can be opaque to ordinary model-level
dispatch observation, so a read-only nested dispatch mode is entered only
during the genuine `token_combine` frame. It records each actual native API
call once, real H/I/P metadata, the `abs` producer edge, module path and real
enqueued request ID. No model function or op registration is replaced, no
model input/output is reconstructed, and no operator is replayed. The census
reports actual per-request native counts and source/producer/consumer records.
It does not claim that the index values form a complete permutation; a later
actual-value/lossless acceptance is still required for Fault8 injection.

Two CPU contracts verify actual opaque-call nesting, exact one-call native
execution, preserved probability object and index-producer identity, healthy
model output, profiler cleanup and refusal to overwrite a foreign profiler.
They do not grant NPU or Worker qualification.

## Fault22 actual Worker input boundary, pending adaptation

The observed Qwen Worker uses `vllm.npu_rotary_embedding`, not the existing
ordinary lookup adapter. Inspected installed `ops/rotary_embedding.py` SHA256
`74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7` defines
`rope_forward_oot` at line 153. Its real inputs are positions, query, key,
cos_sin_cache, head_size, rotary_dim and style. It forwards positions/cache to
the Triton reader or `torch_npu._npu_rotary_embedding`; offsets are rejected.
`AscendRotaryEmbedding.forward_oot` passes the actual module cache into this op.

Thus a possible separately reviewed legal-row adapter is a private positions
argument for this exact reader, with only its final position changed to the
first legal position. It must retain the original scheduler positions/cache,
leave all other rows untouched, call the actual rotary op once, preserve its
Q/K aliasing contract, and prove actual returned Q/K propagation plus OFF.
Changing or copying an arbitrary Q/K output row is not an equivalent mechanism.
The existing ordinary-row output proxy does not cover this boundary, and no
new Fault22 qualification or uint32-overflow claim is made by this handoff.

## Actual Worker18 result and fresh-Worker statistics preparation

Root completed and independently audited
`/home/17-fault-c2c3-20261117/natural18-worker-a1/natural-worker-rms-group/probe/result.json`:
113/113 C2 targets, 226/226 C3 calls, nine actual managed requests, exact full
hidden/logits OFF, and zero owned resources after group recovery. Root's audit
matched all 229 C2 and 455 C3 artifact references to canonical bytes. This is
actual standard Worker runtime evidence. Its E5 statistics remain separate
from the already observed HF model-local E5 result.

`natural_rms_worker_statistics.py` now prepares an actual full-request
statistics/mix window in a fresh Worker. Each invocation executes precisely:

1. One ordinary warmup generation, then four actual healthy managed requests
   (native baseline, baseline, independent baseline repeat, lossless).
2. Source/actual-dispatch/independent-observation qualification of all targets.
3. One fifth managed request captured at a common code location. Its arm is
   baseline or fault. This is the sixth explicit driver request including the
   warmup, identically in all fresh Workers.
4. A sixth managed OFF request proving full hidden/logit recovery, owned
   adapter cleanup, then one ordinary healthy generation after removal.

Three separate root-owned containers provide B, B-repeat and F, each with the
same `/source`, `/model` and `/evidence/probe` paths and the same actual request
ordinal. The raw collector captures every API/module from after `advance` to
before `collect`, including actual runner preparation, model, logits and
sampler. Only owned scheduler verification, observer metadata, complete CPU
snapshots and proxy/readback arithmetic use the reviewed public diagnostics
pause. Native RMS return-lineage stride queries also use that pause. The actual
native RMS operator executes outside it. Collector stop/start pauses do not
advance the step or reset raw API ordinals; each fresh Worker writes step0.

No statistic/construct/stack field is filtered or rewritten. Actual token IDs
and whether they changed are reported independently; sampler differences stay
in the raw capture. The same fixed rtol=1e-5/atol=1e-8, null-unobservable policy
applies. This new probe grants no additional C2/C3 coverage by itself. Its
observed target count and every half-return/real-consumer check must pass in
that actual Worker, and E5 still requires a separate raw audit.

Prepare manifests without executing them:

```
python experiments/c2c3_closure_20261117/prepare_tensor_natural_cases.py \
  --source /home/17-fault-c2c3-20261117/<frozen-new-source> \
  --task worker18-statistics --card 2 --out <new-statistics-cases.json>

python experiments/c2c3_closure_20261117/prepare_tensor_natural_cases.py \
  --source /home/17-fault-c2c3-20261117/source-a21 \
  --task moe8-census --card 2 --out <new-moe-cases.json>
```

The first manifest needs the new statistics implementation snapshot; the
already prepared source-a21 contains the MoE census implementation. Both
manifests passed the main worktree's actual `queue_contract.validate_cases`.
Worker18 creates three cases with 1800-second/48-GiB-host budgets. MoE creates
one zero-injection case with its unchanged 1800-second/64-GiB-host budget.

For the independent audit, root can use its existing `audit_statistics.py`
twice: B against B-repeat, then B against F. Pass each case's unmodified
`probe/statistics/observation/step0/rank0`, `config.json`, and
`collector_identity.json`. The audit receives no provider hits, tensor payloads
or target selectors; source raw triplet SHA256 values remain in each result.
Any raw alignment, nonfinite or pre-effect difference must remain a failure or
reported limitation, not be normalized away.

CPU validation passed 58 focused regressions, followed by nine current
window/controller tests including the additional native-RMS lineage pause
test. Those tests explicitly use fake collector controls/CPU fixtures and do
not substitute for the root-queued raw capture.
