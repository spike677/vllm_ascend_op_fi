# Fault28 FIA: source87 TND prefill adaptation boundary

Read-only review of integration commit `d04ee94b8432a201abbe58dbbc11633101083012`.
Neither existing Fault28 FIA build supports fault admission on the observed
CANN 8.5.1 TND2048 route. The old builds remain BNSD/CANN9.1. The newer fault33
builds include a normal TND prefill fallback, but their selectable arch22 policy
is Q1/page16 and does not implement either Fault28 strategy for TND2048.

The smallest proposed implementation is a **new normal-only TND2048 arch22
package first**, followed by a separate fault patch only if its stock/private
normal comparison passes. This review does not change code, run a census,
execute a kernel, or grant C1/C2/C3.

## Checked originals and current geometry

The local source87 result rehashes to
`3b7143f1b3873211214a79e3d22afc21a86e80b8a86409496bfe6e15dfba3340`.
Its final retained evidence contains two fresh requests, 28 calls each. All 56
public calls have this same descriptor pattern:

| Tensor | Shape | Strides | Element offset | Backing bytes |
| --- | --- | --- | ---: | ---: |
| Q, FP16 | `[2048,16,128]` | `[2048,128,1]` | 0 | 8388608 |
| K, FP16 | `[2048,8,128]` | `[1024,128,1]` | 0 | 4194304 |
| V, FP16 | `[2048,8,128]` | `[4096,128,1]` | 3072 | 16777216 |
| mask, int8 | `[2048,2048]` | `[2048,1]` | 0 | 4194304 |

The actual supplied arguments are TND, Hq16/Hkv8, scale `128**-0.5`,
`sparse_mode=3`, `block_size=128`, `block_table=None`, and Q/KV lengths `[2048]`.
The scheduler/attention metadata binds one fresh 2048-token request, no decode,
and the actual mask object. The public schema declares omitted
`inner_precise=0`, `pre_tokens=next_tokens=2147483647`, and LSE false. These
defaults are **not** a readback of final Native attributes.

The completed independent audit already reconciles 56 FIA launches plus
56 Slice launches, not 112 FIA calls. Its actual stock FIA key is
`5000000000000200103`, blockDim20. Stock POD, exact tensor-pointer ABI, selected
ACLNN resolver ownership, and internal entry reachability remain unproven by
that audit. BlockDim20 alone does not establish the platform's AIC/AIV budget.
There is no reason to repeat its association-only census.

The strided V view must remain the actual input. A normal probe may observe
the existing Slice/copy path; it must not substitute a contiguous test tensor,
repack KV, change the public options, or silently drop the Slice from the
Native denominator.

## Concrete source and build gaps

| Current file/contract | Why it cannot admit source87 |
| --- | --- |
| `attention28_prefill.py:67–117` | Requires BNSD, rank4 contiguous QKV, sparse2, bool mask, next_tokens0 and API block_size0. |
| `attention28_prefill.py:120–157` | Verifies the original CANN9.1 interfaces; a new 8.5.1 artifact cannot inherit this receipt. |
| `fault28_fia_policy.h:117–139` | Requires 16/32 cores, BNSD shapes, nonaccumulated lengths, sparse2 and the old normal M512/S2=512 context. |
| `base_policy.py:80–88` and `fault28_causal_tile.h` | Causal pruning is explicitly BNSD `(G,S1)` order. Its modulo-S1 formula is wrong for TND `(S1,G)` order. |
| `apply_policy.py` | Existing legal and replay dispatch keys instantiate BNSD; replay additionally requires the M128/S2=128 BRC, no-SplitKV template. |
| `attention_native.py:240–284` | Requires the old variant/key set, old policy bytes and old serializer contract. It must stay strict. |
| `fia/build_fresh.py:256–264` | CANN8.5.1 OpDef compatibility is currently restricted to fault33 variants. |
| `paged33_arch22_bridge.h:30–70` | Normal route override requires Q1, page16, a block table and KV1025–2048. Source87 has Q2048 and no page table. |

The locally available source94 completion SHA is
`860f27e5cd472e639e1a93f56f0e908aa2bb5f349bf93b771c9cde9d3c5f982f`.
It records ten compiled entries: four Q1/page arch22 entries and six normal
FAI/FAI-FD entries, including `5000000000000200103`. It does **not** compile the
TND/no-page arch22 key `103000000030000003`, or new TND legal/replay entries.
The current source108 all-normal recipe still selects the same compiled-key
family and retains a Q1/page16 scope. This is a source-level statement; its
new build/runtime result is owned by the root and is not inferred here.

## Exact c77 kernel evidence

The eight cached host-source files all match their manifests. The five newly
supplied git blobs also match every size/hash in archive
`v11-tnd28-kernel-sources-a1.tar.gz`, SHA
`e24f9f691af92cd03fd82c49a901d0061458ae782ec7b4703b5ec8055d4ef7df`.
They are from commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`, not a dirty build
checkout. Key source facts are:

- `fia_tiling_nonquant.cpp:253–324` selects native S2=512 and M=512 for TND.
  Its host split information sets `isS1G=true` for TND at lines 328–340.
- `fused_infer_attention_score_tiling_v3.cpp:1199–1251,1371–1419` declines the
  arch22 path when the TND/no-page GQA, inner0, equal-D128 conditions select
  `CheckSpecConditions`. A new exact normal route override is needed to use
  the existing arch22 implementation for this input.
- In the checked private source, tiling-key header line251 names
  `5000000000000200103` as the no-page causal TND SplitFuse key. The main kernel
  at lines251–255 invokes `SplitFuse::FAInfer` with FP16/TND, page=false,
  causal-mask semantics. This maps the private source; a matching number does
  not decode the stock binary's internal implementation.
- `fused_infer_attention_score_v3.cpp:915–918` already instantiates
  `103000000030000003` with FP16, TND query and KV, page=false, FD=false and
  BRC=false. This is the normal arch22 compile target for the proposed package.
- `fia_public_define.h:251–262` defines TND row coordinates as
  `s1=gS1Idx/gSize`, `g=gS1Idx%gSize`. The kernel sets
  `gS1Idx=gS1Cur*mBaseSize` at line606; the vector mask selects `SG` at
  lines593–596. Merely removing the old BNSD static assertion is incorrect.
- For a bounded flat row tile, its last query is
  `floor((gS1Start + actualMRows - 1)/G)`. Any future TND causal pruning must
  retain every key through that query, preserve original masking of aligned
  extra columns, and be checked against the real split-core traversal. This
  source-derived coordinate is not yet a new kernel implementation.
- `fia_block_vec_nonquant.h:647–678` reads the preceding online denominator
  from `inSumTensor` and selects BRC/non-BRC SoftmaxFlashV2. This is a plausible
  location for the original second-S2-block replay semantics after normal
  closure. Current c77 dispatch has no TND M128 BRC entry or TND 109/119 policy
  entries. They require fresh source, host registration and kernel compilation.

## Minimal implementation sequence

**First, normal only.** Add independent files such as
`fia/tnd28_prefill_normal.h`, `fia/apply_tnd28_prefill_normal.py`, and
`fia/tnd28_prefill_geometry.py`. Their predicate is exactly the fresh B1,
TND2048, FP16, Hq16/Hkv8, D128, no-page/sink/rope/quant scope above. Bind actual
Native attributes, cumulative length interpretation, platform AIC/AIV,
serializer and stream/scope identity; reject unknown values. The route
preference remains the same normal implementation when the observation token
is absent, during OFF and after unregister. Do not make the token select
normal arithmetic.

The patcher should only add this narrow exception to `CheckSpecConditions`
and an owner-scoped typed host capture. Keep original M512/S2=512, split-core
logic, precision, and kernel arithmetic bytes unchanged. Use a new CANN8.5.1
fingerprint and compile the normal no-page key plus the required unchanged
fallback entries. Reuse checked build machinery only after root coordinates
the shared builder change; do not edit the active fault33 recipe.

Add a separate Worker boundary/native binding, for example
`attention28_tnd_prefill.py` and `attention28_tnd_native.py`, preserving old
BNSD admission and `FiaNativeCall`. They must bind the actual public view and
any native temporary copy to the new package ABI, retain both Slice/FIA raw
launches, and join only the exact current FIA host record/launch. The existing
Core owner and lifecycle remain authoritative; no new control state machine.

**Then, only after normal acceptance**, add separate TND legal/replay variants.
Legal retains the original coupled M512/S2=512 versus M128/S2=128 intent;
replay retains the original second in-core S2 block denominator-mass operation.
Validate TND `(S1,G)` pruning, BRC allocation, all tails, workspace, split-core
coverage, and FD=false before adding unique dispatch keys. No key is qualified
merely by adding it to an allowlist or compiling its template.

## New stock/private normal prerequisite matrix

| Case | Required evidence and stopping condition |
| --- | --- |
| CPU source/geometry | Exact c77 anchors/hashes; legal TND descriptors and V strided extent accepted as candidates. Wrong offset/span/dtype, page table, lengths, sparse mode, optional feature, owner or thread rejected. No payload rewrite. |
| Real SDK normal build | Fresh 8.5.1 host, decoder and unchanged normal kernel compile. Measured serializer/ABI and actual compiled no-page entry; no old object reuse. Build success alone permits no fault. |
| Stock repeat | Two identical fresh2048 requests, complete 28-site module/identity order, full per-site input/output equality and complete model-output health. Preserve all launches and actual argument/default distinction. |
| Private normal repeat | Same fixed workload and actual 28-site descriptor/byte sequence. Require new host M512/S2=512, FD=false, actual no-page TND key/entry, workspace/stream and complete raw-POD-launch agreement. Any missing host record, key mismatch or extra unclassified launch refuses. |
| Stock/private join | Compare complete actual Q/K/V/mask and FIA outputs for every paired site, including the original V view and its real copy lineage. Require exact normal equality; do not relax to token equality or reuse source87's hash-only observations as tensor bytes. |
| Same private Worker OFF | Disable admission, drain owned work, remove hooks/scopes, then run ordinary healthy inference. Require full output recovery and zero owned pending/resources. No process rebuild may substitute for OFF. |

Stock/private comparisons can use separately initialized, identity-bound
packages; same-Worker recovery is a separate requirement within the private
Worker. Unmatched input bytes make a pair incomparable rather than evidence
of kernel mismatch. These requests provide new normal equivalence evidence;
they are not another association-only census and do not inherit fault33 Q1
qualification. Fault runs remain disabled until the complete matrix passes.

An independent CPU-only normal predicate/patch-anchor adapter is feasible now,
using the supplied real c77 sources and negative metadata fixtures. It would
not establish numerical equivalence, actual platform values, or a supported
fault geometry. The fault patch is deliberately deferred until the normal
matrix and complete native split/causal contracts succeed.
