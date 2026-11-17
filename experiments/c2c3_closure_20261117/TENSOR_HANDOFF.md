# Tensor workflow handoff, 2026-11-17

Root requested that this workflow save progress and release its agent slot while
the postwrite workflow fixes a separate collector ownership failure. No new NPU
or container was started by this workflow. All new raw statistics stay on 124.

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
No qualification count was changed by this workflow.

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

## Saved Worker18 work in progress — not ready for NPU acceptance

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

Latest CPU checks: 13 tests passed (RMSNorm, model-local groups, statistics
windows, old tensor Worker contracts), including an actual CPU last-token
selection/GEMM consumer and no outside-root injection. Both new serving files
parse. **The serving controller itself has not been executed in vLLM.**

Before declaring this WIP ready:

1. Review exact live callable identity for vLLM `CustomOp.forward` plus its
   `_forward_method`. The current source guard authenticates the reviewed
   msprobe wrapper and original `forward`, but a generic forwarding method
   alone does not bind a changed `_forward_method` implementation.
2. Exercise genuine Worker RPC-thread/mode ownership, final-norm logits
   materialization, fused RMS tuple output lineage, and every failure path.
3. Review the trusted `qualify_observations` host entry and the explicit OFF
   before group-close sequence. It must not become an arbitrary RPC evidence
   submission interface.
4. Verify lifecycle ownership of the serving host's pre/post/logits hooks;
   they are removed before group health. Failure abort retains unresolved
   group evidence and does not claim recovered fault state.
5. Add the standard-Worker statistics span separately; HF E5 cannot qualify it.

Root should continue using `50a822b` for the queued HF18 capture. The WIP commit
must not silently replace that immutable executed source snapshot.
