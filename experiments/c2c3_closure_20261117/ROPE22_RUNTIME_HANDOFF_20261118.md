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

This checkpoint is based on integration commit `9a406c9d57790ab814178ed1e74eb88160726159`.
It changes source and CPU contracts only. **No new NPU execution, standard-Worker
group acceptance, C2, C3, or E5 is claimed.** The existing ordinary lookup-output
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

## CPU verification and root next step

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

**The fused-input variant is not yet wired into the standard-Worker group
driver or `CoverageSession`.** That integration, independent C2 denominator and
per-site effects, live C2-to-C3 admission, cleanup/OFF, and fresh B/F raw
statistics remain future work. It is not interchangeable with the existing
`22.wrong_row` ordinary lookup-output group. Vision17 final-argument and first
consumer work is deferred to a later checkpoint; no Vision source changed here.
