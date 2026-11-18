# RoPE22 shared-site and fused-input checkpoint

## Actual source42 refusal and ancestor repair

The root's next zero-injection receipt remains failed and unchanged:
`/home/17-fault-c2c3-20261117/rope22-runtime-census-card2-a2/natural-rope22-census/probe/result.json`,
SHA256 `5ef82f834791077a5a3105dfc8ad4122da9500a318407be52d9271314f482397`.
It discovered all28 shared RoPE registrations, but failed before the first
reader. The actual traceback runs Qwen3 root→decorated Qwen2Model `__call__`→
direct `forward`→`embed_tokens`; the model ancestor's Module hooks are bypassed.
The actual pinned Qwen3 source also registers tied `lm_head` as the same embedding
object. A root-only hook stack therefore cannot select the embedding alias.
Read-only image source hashes are Qwen3
`153ac85de337ebfec8fd505be600bf93b0f4c55aa3e0ee3f4a91ca6637f75c76`
and compilation decorators
`09d05a7ffd3f3e7692b331bd01dbbede404eab2b0e787b1d45fd51c74fbe63c0`.

`ModuleSites` now additionally uses real executing frames of frozen, bound
ancestor forward methods. A matching `self` variable or class/path string is
insufficient. Every ancestor registration edge and live method/code identity
is rechecked; absent/ambiguous ancestry still refuses. The four new CPU cases
cover the actual direct-forward/tied-alias pattern, a missing live ancestor,
a fake `self` local, and changed intermediate graph/forward. The full RoPE CPU
set is now19 passing contracts. This source repair does not convert the failed
source42 receipt into a pass and does not grant C2/C3.

The original checkpoint was based on integration commit
`9a406c9d57790ab814178ed1e74eb88160726159`; the ancestor repair and group driver
are based on `1e15888ccd1df3e26a6e59bff84e4865b8e00e7a`.
These changes cover source and CPU contracts only. **No new NPU execution,
standard-Worker group acceptance, C2, C3, or E5 is claimed.** The existing ordinary lookup-output
proxy and all earlier receipts are unchanged. The root controller still owns
all device calls and container scheduling.

## Actual evidence that motivated the repair

Read-only SHA verification of the original receipt succeeded:

`/home/17-fault-c2c3-20261117/natural-vision17-moe8-rope22-card2-a1/natural-rope22-census/probe/result.json`

SHA256: `0660280aa90a1a120a28876d2824f148fa104e4bf18c97176f9d2020e4ccc0ec`.
That zero-injection run has 56 successful readers across two requests and exact
healthy/repeated output. It does not pass the earlier observer: `named_modules`
deduplicates one RoPE object shared by 28 parent Attention objects, and actual
`HAS_TRITON=True` execution has no nested Torch rotary operator to join.
The original failing result is preserved; these source changes do not repair it
retroactively.

The existing pinned image was inspected read-only through its overlay metadata,
without starting or entering a container. Image identity:
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.

| Installed source | SHA256 |
| --- | --- |
| `vllm_ascend/ops/rotary_embedding.py` | `74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7` |
| `vllm/model_executor/layers/rotary_embedding/base.py` | `32e2145231012f98c8c613cc37dfe621996a3508af77ee7e0265baefde5c1139` |
| `vllm_ascend/ops/triton/rope.py` | `4c579d5d22b08d7657f1406fe072c2d17f6fa0a87b0421a520b61e6e1177f7f9` |
| `triton/runtime/jit.py` | `55c8b09db2780f1d5106fb1652bda7a5807c7b243b1e43a763265d4755b58dd3` |

## Implemented observer changes

`ModuleSites` preserves every registration edge, then resolves a shared object's
current site using actual active parent objects. It verifies the live graph and
object lifetimes at entry. Multiple aliases beneath the same active parent are
rejected as ambiguous; no first-name, layer-number, or shape fallback exists.
Static graph positions alone do not pass coverage: every position must have one
actual reader in each distinct scheduler-bound request, with a completed root
forward and a returned consumer of both live Q/K output views.

`NaturalRoPETrace` separately observes the actual outer custom operator, exact
Python reader, and either nested Torch native rotary or pinned Triton JIT path.
The Triton trace observes the actual launcher call and return and captures its
JIT callsite's live `bound_args`, `non_constexpr_vals`, grid, and stream. It checks
that positions/cache objects and Q/K storage spans join the actual reader, and
that non-constexpr arguments match the JIT binding. A JIT return without an
observed launcher invocation is not complete; equal-valued cloned positions do
not satisfy object identity.

**These are source-bound runtime callsite arguments, not decoded final device
ABI payloads.** `raw_device_launch_payload_observed` remains false. No kernel
build/hash expected profile or injection hit log is used to manufacture a join.
The actual JIT kernel name/hash are recorded after its returned launch. This
does not independently audit generated device instructions or packed ABI bytes.

The default observer policy still requires exact scheduler positions. Optional
`exact_or_last_from_first` independently recognizes only identity or a separate
positions allocation with the final legal index replaced by the original first
index; it reads actual bytes without using the injector's selected/hit state.
Root positions and the entire original cache must remain unchanged. Producer
cache construction still predates installation and stays unobserved.

## Separate fused-input variant

`FusedWrongRowSession` is the new, bounded
`LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW` variant. It binds the
actual reviewed opaque operator under dynamically resolved model sites; the
Worker driver supplies actual scheduler request identity via `bind_request`.
Users do not specify layer names. Only selected calls allocate a new positions
tensor and replace its last element with its first. Unselected calls retain the
original argument objects. The original fused operator executes once; cache,
model weights and Q/K outputs are not patched after execution.

A healthy per-site reference is mandatory. The adapter verifies unchanged
non-target producer rows and non-target output rows, whole-cache and original
positions readback. Complete input equality and later propagated input changes
are reported separately. Healthy output comparison at later sites is not
misrepresented as an isolated causal trial.

The independent observer must be entered **before** the adapter so its lower
dispatch mode sees actual post-adapter arguments. It then observes wrong-row
positions entering the real reader/JIT/launcher and Q/K reaching a real consumer.
Wrong ordering fails the outer-to-reader identity join rather than trusting the
adapter's event. The adapter never issues a coverage proof itself.

## Initial CPU checkpoint

Fifteen CPU contracts passed: six existing census contracts, four shared-site /
Triton-shaped callsite contracts, and five fused-input contracts. The three-site
shared-object fixture covers six actual calls, wrong final-argument clones,
missing sites, missing launcher calls, real CPU effects, unchanged non-target
rows, repeated faults and exact OFF recovery. CPU simulation rows use no real
kernel-source identity and remain explicitly unqualified.

Run locally with the existing CPU interpreter and
`PYTHONPATH=src;fault39/src;.;tests`:

```text
python -B -m unittest discover -s tests -p "test_rope*.py" -v
```

For root's next **zero-injection** runtime check, the existing
`prepare_tensor_natural_cases.py --task rope22-census` entry and
`natural_tensor_census.py --rope-trace` require no driver change. Deploy a new
immutable source snapshot, run source/CPU preflight, and then let the root-owned
queue apply its lock, device gates and bounded request plan. The new factory
refuses different Triton source hashes or dispatch ownership. Read all actual
outer/reader/launcher and consumer rows; do not reduce acceptance to a counter.

The original checkpoint had no standard-Worker group driver. The following
source update supplies that driver; its actual NPU acceptance remains pending.
It is not interchangeable with the existing `22.wrong_row` ordinary
lookup-output group. Vision17 final-argument and first-consumer work is deferred;
no Vision source changed here.

## Prepared Worker group and bounded root handoff

`fault_injection/rope_group.py` and `rope_serving.py` use the existing actual
Worker request/context/logits machinery and `CoverageSession`. They leave the
shared tensor provider, model methods, reader and launcher implementations
unchanged. The mechanism scope is explicitly `22.wrong_row.fused_positions`.
The root probe is `natural_rope_worker_probe.py`; `prepare_rope22_group.py`
prepares one case using the existing source/model mounts and resource gates.

The queued sequence has nine independently enqueued, fresh-prefill requests:

1. Native baseline, instrumented baseline, repeat and identity/lossless request.
   Full root inputs, hidden output and logits must match exactly. Each actual
   shared parent site must have one independently joined reader, outer operator,
   launcher and consumer. The denominator comes from these calls, crosschecked
   against the live registration graph; no fixed layer count is accepted.
2. One C2 fault request covering all qualified sites, then a full exact OFF.
3. Two every-match C3 requests using the **original live C2 capability inside
   the same Worker process**, then a full exact OFF. The proof is never copied,
   serialized or reconstructed through RPC.

The upper independent pre-observer issues each original-call ticket before
admission; the lower trace independently reads the actual post-adapter arguments
and actual reader/JIT/launcher/QK consumers. Healthy reader evidence is copied
and hashed before fault admission. Hit completion requires actual wrong-row
positions, changed Q/K output, unchanged non-target input/output rows, unchanged
entire original cache and root positions, and a returned consumer. A matching
injector event alone cannot complete a hit. Later propagated input changes stay
explicit and are not isolated causal evidence. Missing sites, launcher calls,
original object lifetimes or original live C2 proof refuse the group.

The bounded input scope is the existing natural prompt, TP1 BF16 eager,
fresh prefill, no chunking/prefix reuse/async scheduling, and at most two C3
requests. No decoder/cross-request cache generality or integer-overflow root
cause is claimed. Actual runtime source hashes are checked on every phase.
The original source42 failure and all old receipts remain unchanged.

Owned modes/profile/module hooks and per-request replacement-position tensors
are released at collection. A bounded CPU healthy-reference pool remains alive
between C2 and C3 for independent comparisons and is cleared when the serving
session closes. A zero persistent fault-buffer budget therefore does not mean
zero diagnostic/model/allocator memory. Post-cleanup healthy generation must
return the original token on the same Worker.

Final CPU verification: **26 RoPE contracts and 9 existing serving Tensor/MoE
contracts passed**. The new CPU fixture executes a hook-bypassed parent, tied
embedding aliases and a shared three-site RoPE; it verifies C2 three hits,
C3 six hits, OFF recovery and rejection paths. These simulation results do not
qualify a real Worker. Actual group execution, raw B/F statistics and E5 remain
pending; `raw_device_launch_payload_observed` and public qualification remain
false. The JIT/launcher evidence remains source-bound callsite observation,
not final device ABI decoding.

Root preparation command after freezing a new reviewed source snapshot:

```text
python experiments/c2c3_closure_20261117/prepare_rope22_group.py --source /home/NEW_FROZEN_SOURCE --card 2 --image sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 --out NEW_CASES.json
```

This command only writes a new manifest. The root-owned queue must perform its
resource lock and per-case device gates and preserve a new attempt directory.
No device calls or containers were started for this implementation checkpoint.
