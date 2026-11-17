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
