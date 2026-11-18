# Quant routed-chain implementation boundary after source87

This is a source review and implementation proposal, not a new runtime adapter
or qualification. The completed source87 audit is in
`QUANT_SOURCE87_INTEGER_DIAGNOSTIC.md`. Its 52 integer route records passed,
but only the first of 26 MoE inputs in each forward has current request rows;
the complete projection denominator remains 135 per forward, with 10 observed
request-row paths and 125 unresolved paths. No additional index-only run is
needed.

## Newly reviewed production source

The fixed-image, device-free bundle is
`e971dcaa5d9cbaab85e96ed471e24c4bacce65d079e0ed90afb39f4f3c268c65`
(1,048,272 bytes). All 115 embedded source hashes match their text. The relevant
Python files are:

| File under `vllm_ascend/ops/fused_moe` | SHA256 | Reviewed behavior |
| --- | --- | --- |
| `moe_mlp.py` | `1a65440a1357af84641a87fda630cf84efcbc8b34fa66ceedd2c403893fd840d` | `unified_apply_mlp` lines 366–443 extracts the current typed payload and passes its flags and objects to `quant_apply_mlp`. Fusion converts count groups to cumulative groups for GMM1; GMM2 receives the original count groups. |
| `moe_runtime_args.py` | `41bb24c3fe464e5fc99792ad86afdd2d44ad71a5992d106ee4209cf982bde3f5` | `build_mlp_compute_input` lines 204–225 preserves the actual dispatcher output objects and the fused-expert weights/quantization objects. It computes `fusion` from the actual quant type and `use_fusion_ops`. |
| `prepare_finalize.py` | `f46b378b2ba197fafd744c409735aa9a011d93e073507c1f87cf1a60ef2ecae4` | AllGather prepare/finalize has distinct DP, PCP and sequence-parallel branches. A TP1/DP1 label alone does not prove that the same tensor took an identity-return branch. |

The earlier pinned `token_dispatcher.py` source
`21fd94e3d5eacc578cc3feb794bc5c135c148c2d495c611f224f22c8b96ac0cc`
returns the actual routed tensors in `MoETokenDispatchOutput` and stores the
actual top-k probabilities and expanded indices in combine metadata. Its
`token_combine` passes those objects to `npu_moe_token_unpermute`, with `abs` on
the current indices. Source87 already proves the observed indices were
nonnegative, so `abs` did not conceal a negative-index case.

The currently reviewed DeepSeek model source
`bad4921f88f8343a965e333a399edf6b73f9d3031f6297e0dbf56afe37a31b01`
has the post-route operations in `DeepseekV2MoE.forward`: actual tuple unpack at
line 371, routed scaling at line 379, shared addition at line 386, then optional
parallel communication and the final view. The source87 events 935/936 match
the operator kinds, but the old graph did not retain their actual scalar
arguments. The source literal or model name cannot replace those observations.

## Minimal next implementation

1. Extend the existing source/frame observer to the typed builder, exact
   `unified_apply_mlp` and `quant_apply_mlp`, and the selected prepare/finalize
   functions. Bind current function/code/globals, model/runner/dispatcher owners,
   actual branch flags and return objects. Check the actual payload class and
   fields instead of accepting an arbitrary object with similar attributes.
   Hold the current tensors only until their real consumer/return is observed;
   do not join later calls by reused addresses. Retain the current integer
   evidence once per parent with its existing budget and separate helper ledger.
2. Produce a distinct Python call-chain record for current input, dispatch
   payload, GMM inputs/outputs, combine metadata, unpermute result, finalize
   return and the exact routed member of the public tuple. Require original
   calls exactly once, current forward/request identity, unchanged metadata and
   write epochs, and exact input/output object relationships. Expanded rows
   remain `(source_token, route_slot, expert)` records: 48 slots are never 48
   requests. A complete logical chain is not itself Native row qualification.
3. Add a narrow, source-bound rule for the two actual post-route in-place
   consumers. For multiply, require the real current scalar argument and the
   current immutable configuration owner; initially accept only an observed
   ordinary finite Python scalar equal to that owner's scaling field. A tensor
   operand or another representation remains unresolved until independently
   proved. For add, both actual data operands must already have the same current
   request-row mapping, with the actual supported `alpha` value recorded. Equal
   shapes or equal bytes do not establish this condition.
4. Preserve the ordinary write invalidation first. Increment the observed epoch
   for every overlapping storage alias, including inference tensors with no
   `_version`. Only after one successful original operation and unchanged
   geometry may the adapter bind a fresh proof to the exact returned target at
   the new epoch. Old aliases stay stale; an unobserved mutation, exception,
   unexpected output alias, foreign frame or changed owner leaves every affected
   row unresolved. Do not decrement epochs or restore old proof entries.

The implementation can expose
`PYTHON_CALL_CHAIN_VERIFIED_NATIVE_ROW_SEMANTICS_UNVERIFIED` as descriptive
evidence while Native prerequisites remain missing. Such records must not enter
the graph's admission-capable verified rows, provider qualification, or
CoverageSession target sets. The in-place rule can propagate an independently
qualified input; it cannot repair an unknown routed input. This keeps the two
proof levels explicit rather than laundering an unverified kernel through a
source-bound Python return.

Proposed owned files are `quant_token_routes.py`, `quant_moe_rows.py`,
`quant_source_rows.py`, and focused quant tests; add a separate small routed
chain helper only if necessary. No Runtime/Host/Coverage API, Native algorithm,
ABI, input, model weight or fault configuration change is proposed.

## Specific remaining source and Native gaps

The new bundle imports but does not contain `vllm_ascend/device/device_op.py`,
`fused_moe/moe_stage_contracts.py` or `fused_moe/moe_stage_params.py`. The actual
`DeviceOperator.npu_grouped_matmul_swiglu_quant` and
`DeviceOperator.npu_grouped_matmul_gmm2` methods still need their fixed-image
definitions and any selected device-specific delegation. The typed payload
classes and quant-type properties need their definitions as well. This is a
small source-closure request, not another model/index census.

The 112 non-main-Python files in this bundle predominantly implement custom
Weight-NZ/TensorList GMM variants. They do not establish the stock `npu::`
operator semantics used in source87. There is no `npu_moe_token_unpermute` or
`aclnnMoeTokenUnpermute` definition in the new bundle. Do not substitute these
custom kernels, generated headers, or build-directory paths for actual loaded
implementations.

Native acceptance still needs the current registration/loaded implementation
binding and reviewed semantics for the observed routing, GMM1, GMM2 and
unpermute paths. In particular, GMM1 must preserve the cumulative-group row
order, GMM2 must preserve the count-group row order, and unpermute must use the
observed inverse index and per-token/per-slot probability convention. Source87
records exact schemas, active dispatcher origins and object connections, but
no loaded DSO/function ownership for those kernels. A later qualification must
bind these facts to that live Worker, not just to a source archive or matching
symbol name. Unsupported overrides, optional branches and mutations refuse.

## Targeted CPU and eventual natural checks

CPU contracts should include cross-parent and cross-forward substitution;
changed builder class/function/globals; equal-shape swapped group lists;
changed inference write epoch; unexpected prepare/finalize communication;
wrong return tuple member; missing GMM/unpermute stage; different scaling owner
or scalar; same-shape residual carrying another request order; an aliased
mutating second operand; partial-overlap storage; inference tensors without
versions; and exception/cleanup retry with no lost retained reference.

Positive fixtures may establish Python chaining and explicit kernel-contract
assumptions, but must keep runtime/Native qualification false. Existing source87
integer evidence should be reused for offline review. The next natural run,
after source review and root approval, should test the new live wrapper,
consumer and epoch joins with two healthy requests and zero faults, retaining
all 135 candidates and first refusal reasons. It is not a refreshed C1 run or
an automatic C2/C3 promotion.
