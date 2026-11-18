# Source87 integer routing diagnostic

This is a local, stdlib-only review of the completed zero-fault source87 census.
It is not new C1/C2/C3 acceptance, routed-row qualification, an E5 comparison,
or a rerun of the model. No SSH, container, SDK, framework import or NPU operation
was performed by this review. Raw observations and integer arrays are not copied
into Git; the accompanying JSON contains hashes, counts and metadata only.

## Exact originals

| Original | Bytes | SHA256 |
| --- | ---: | --- |
| Final result | 965,884 | `2ac9936b3188ce2ce6b2377a7f4a8920d87df8c4c54251d3c77d6cb0e9fc3130` |
| Uncompressed inventory | 78,609,192 | `e67e9cbb6e2c9027bcf801d28df26be759b64d5f6d43888693bca261be776b21` |
| Supplied gzip | 2,997,091 | `06ba800647866aa465f5283ae45f3687fd352bd2561b62fd1417f516ba2efd72` |
| Diagnostic summary | — | `01f76cca7c350886cc60f5ed01f999921d625c2b9d262927341b8142e3416353` |

All 13 runtime source hashes recorded by setup match the reviewed checkout.
The integer observer itself is
`d302ccfc6c66a336476bd93eddadfd679c27afa93549b1186ef72d4453712bd9`.
The actual DynamicQuant group library remains
`6eb6c19cff6006bc9cd0b1072504dc5d6b7db3a0d7b393af7c88f4db1c8bd2fd`,
build manifest `b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`,
in production image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
Those identify the quantization library, not the separate GMM/unpermute kernels.

## What the actual records establish

Both B8 first-decode forwards retain all 135 projections: 270 total projection
calls, 270 Native parents, 270 actual prepares and 270 executes, with complete
public/Native bindings. The public API count is 540. No fault owner was installed
and no fault was executed. Both healthy output lists compare equal after cleanup.

Each forward contains exactly 26 parent records for `model.layers.1` through
`model.layers.26`. Every parent has seven completed, ordered original calls:
top-k, routing, cumsum, first GMM, second GMM, abs and unpermute. Each original
call count is one. There are **no missing, incomplete, refused or overflowed
integer records** in this attempt.

Each parent has all five integer snapshots: int32 top-k IDs `[8,6]`, int32 raw
indices `[48]`, int64 counts `[64]`, int64 cumulative counts `[64]`, and int32
absolute indices `[48]`. This is 1600 bytes per parent, 41,600 per forward,
83,200 total. The independent reader recalculates every snapshot's little-endian
bytes and SHA, checks unchanged metadata/write epoch and exact parent/forward
binding, and verifies:

- Six distinct expert IDs per source token, all within `[0,64)`.
- The raw indices form exactly the nonnegative permutation `[0,48)`; abs does
  not hide negative sentinels.
- Counts equal the actual top-k histogram; cumulative counts equal its prefix
  sums and end at 48.
- Inverting the original-slot-to-destination mapping yields sorted expert
  groups. No stable order among equal experts is assumed.
- The reported token/route-slot proof is precisely the independently rebuilt
  map, with its original evidence hash intact.

All 130 corresponding snapshots are equal across the two forwards when joined
by actual layer path and snapshot role. This equality is replay evidence, not
permission to reuse tensor addresses or request identities in another call.

The reader also joins current stage outputs to the exact recorded object,
storage, extent, stride, version and device of their next consumers. This covers
the top-k IDs, routing output and scale, cumulative/count GMM inputs, first GMM
outputs and per-token scale, second GMM output, original indices, abs output and
the original top-k probabilities. Each parent event is SHA-linked to its route
record. All actual Python source-frame hashes, operator schemas and current
PrivateUse1 dump observations are retained; none is relabeled as Native owner
authentication.

The actual GMM2 formals are `split_item=2`, `group_type=0`, `group_list_type=1`,
`act_type=0`, BF16 output, one x and one per-token-scale input, with the recorded
optional transformation/bias arguments absent. GMM1 has no bias/offset.
Unpermute is unpadded with `restore_shape=None`. These are actual call formals,
not defaults inferred from dimensions.

## Helper accounting and remaining row boundary

Each forward records 130 integer-read helpers, each with one actual `_to_copy`,
`resolve_conj` and `resolve_neg`: 390 helper dispatches, 780 total. Their explicit
classification is `OBSERVER_INTEGER_READBACK_NOT_MODEL_INVOCATION`; they do not
increase the 270 projection/Native denominator. Copy inputs and outputs are
integer tensors; no float activation payload is exported. Peak owned CPU tensor
copy size is 512 bytes. Helper modes, tensor references, parent references,
hooks, method/API patches, profile callbacks and scheduler/live-metadata owners
are all zero after cleanup. Integer payloads remain intentionally in the evidence.

Readback times are 0.252060 and 0.462435 seconds; route-observer hook times are
1.412734 and 1.629836 seconds; source-profile times are 0.014234 and 0.014274
seconds. These intervals overlap and must not be added as disjoint overhead.
They are not a comparison against an unobserved execution or statistics capture.

Request-row coverage stays **10/135 per forward**, with 125 UNVERIFIED candidates
and the full denominator intact. The existing ten paths are the first layer's
three attention and two MLP projections, the second layer's three attention
projections, and its two shared-expert projections. Only the first MoE parent
has a resumed, current request-row input proof; the remaining 25 explicitly
report `MOE_CURRENT_REQUEST_INPUT_ROWS_UNPROVEN`. All 26 integer maps can be
arithmetically valid without proving that their source-token index names an
already-proven request row.

The first routed chain is events 755 (routing), 758 (GMM1), 761 (GMM2) and 763
(unpermute). They remain unqualified. After its actual return, event 933 aliases
the known shared output while event 934 aliases the unknown routed output.
Events 935 `aten.mul_.Tensor` and 936 `aten.add_.Tensor` explicitly invalidate
storage aliases. Event 939 AddRms remains unproved, followed by the first
unproved projection `model.layers.2.self_attn.kv_a_proj_with_mqa`. Each forward
retains 12,853 operator events and 12,783 conservative unknown edges, with zero
missing events and zero observer errors. Unknown-edge count is not target count.

## Exact remaining work

1. Authenticate the actual loaded implementations of
   `npu::npu_grouped_matmul_swiglu_quant`, `npu::npu_grouped_matmul`, and
   `npu::npu_moe_token_unpermute`. The current PrivateUse1 origins are all
   `build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413` (schema debug at
   `:8068`). These build-path strings do not identify a loaded DSO/function.
   Top-k and custom routing likewise report `csrc/torch_binding.cpp:702` without
   loaded-library ownership proof.
2. Review the actual wrapper/Native semantics: GMM1's cumulative-group boundaries,
   GMM2's count-mode grouping, row-local SwiGLU/quantization, and unpermute's exact
   inverse-map and probability-reduction convention. Dimensions alone do not
   prove any output row ordering. The existing 153-file source bundle lacks
   these implementations. `moe_comm_method.py:160–161` only delegates to
   `moe_mlp.unified_apply_mlp`; the MLP/argument-builder/finalizer closure must also
   be source-bound to the actual route.
3. After those semantics are reviewed, a new live adapter may carry internal
   `(source_token, route_slot, expert)` rows through the exact current tensors.
   These 48 expanded rows must never be labeled 48 requests. Only a complete,
   actual unpermute chain can collapse them back to the eight source tokens,
   and only a still-current input request-row proof can attach request IDs.
4. Separately adapt the actual in-place post-route consumers with their exact
   source-bound arguments, immutable parameters, live objects and alias epochs.
   Both data operands of addition must carry the same actual row mapping.
   Historical graph events do not save all scalar arguments; source defaults
   cannot fill those missing observations. Preserve unknown writes and refuse
   reuse, altered indices, changed owners and incomplete chains.

There is no reason to repeat a census merely to obtain the same integer arrays.
Root has now supplied a new fixed-image MLP source bundle
`e971dcaa5d9cbaab85e96ed471e24c4bacce65d079e0ed90afb39f4f3c268c65`;
reviewing that source is the next offline step, not evidence that its compiled
code is the actual loaded Native implementation. Any new row adapter still
requires a fresh zero-fault natural acceptance run. Complete-set quantization
qualification additionally needs the existing full-target Native leases,
lossless/effect/OFF and same-Worker health gates; none is granted here.

## Reproduce the read-only audit

```text
python -B experiments/worker_core_v11_20261118/audit_quant_source87_routes.py --inventory <original-inventory.json.gz> --result <original-result.json> --out <new-summary.json>
```

The CLI rejects changed original hashes and writes only a new summary file.
It imports no framework or runtime provider. The actual 52 records passed;
eight local corruption checks also rejected negative indices, wrong counts,
wrong cumulative sums, duplicate experts, changed metadata, changed epochs,
altered proof slots and an attempted Native qualification flag. No production
adapter, Native ABI or user input was changed.
