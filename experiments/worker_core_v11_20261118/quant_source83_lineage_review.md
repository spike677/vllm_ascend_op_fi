# Source83 row-lineage audit and next boundary

Source83 confirms **10/135 row-proven projections per request**, including the
two new shared-expert projections. It does not qualify 135 Native fault
targets. The two healthy B8 requests retain 270 projection calls, 540 public
API calls, 270 actual Native DynamicQuant prepare/execute pairs and all 135
structural candidates. The next unproven projection is
`model.layers.2.self_attn.kv_a_proj_with_mqa`; its preceding routed-MoE output
still lacks a token-to-request restoration proof. This commit changes no
adapter, Native algorithm, ABI, admission rule or qualification status.

The local result SHA256 is
`77e1d8d449a268dcb27274f2850df054fe18779a255042dc6a5598f3fa487ac9`.
The compressed inventory SHA256 is
`e758f9da6e8617e4adbbf6afde829e4dbece3cc7d538f2fc2d6b6acfcaad344e`;
decompression produces exactly 72,031,898 bytes with SHA256
`f5369a15a10df118eaf208e1f225f72b79d18b4df79ca8af1f17ac1c4d4a15d3`.
Those originals stay outside Git. The adjacent JSON contains only this
offline review's counts, hashes, event references and scope limits.

## Checked against actual records

The stdlib-only audit checked 4,109 conditions across 45 classes. It joins
every projection to its same-forward public calls, actual Q/scale consumed
by the owned public QMM, and all five raw DynamicQuant parent/GWS/execute
events. It checks original-call counts, registration lifetime, executor,
stream, unchanged typed formals, complete denominators and cleanup. These
checks do not turn the public QMM observation into a Native QMM lease proof.

For both forwards the newly proven chain is:

| Record | Actual observation |
| --- | --- |
| Node 95 / event 749 | Previously proven actual hidden-state view |
| Node 96 / parent event 751 | Same input resumes at the reviewed runner, epoch 0→1; parent mutable invalidation remains recorded |
| Node 98 | `layer1.shared_experts.gate_up_proj`: actual DynamicQuant output and scale consumed by its owned public QMM |
| Node 99 / event 848 | Actual SwiGLU consumes node 98 |
| Node 101 | `layer1.shared_experts.down_proj`: current SwiGLU output reaches its actual DynamicQuant/QMM pair |
| Event 933 / node 102 | Actual alias of the shared output |

The row digest is recomputed from each forward's own scheduler mapping;
all 103 nodes agree with that forward, and the two forwards have different
digests. The actual tensor/storage metadata, Native binding and public API
indices match the new nodes. The current raw dispatcher dump has one active
PrivateUse1 registration, no shadowed registration, and definition/kernel
origin `vllm/utils/torch_utils.py:789` on Torch 2.9.0+cpu. The three loaded
observer modules match this audit checkout byte-for-byte. The first MoE
parent has no refusal or revoked proof; the remaining 25 parents explicitly
retain `MOE_CURRENT_REQUEST_INPUT_ROWS_UNPROVEN`.

Each forward has 12,853 retained row events, no overflow/missing calls or
source errors, and exactly one original call per event. Healthy before/after
outputs match. No fault owner was installed and no fault was executed.
Owned hooks, API/method patches, row/early/nested modes, profiles, scheduler
references and live metadata records are zero after cleanup. This describes
owned resources; it does not claim that the resident bounded Native census
storage has been unloaded. C1/C2/C3 and Native-qualified target counts remain
false/zero for this controlled observation run.

## Actual unresolved routed chain

| Event | Actual operation and remaining evidence gap |
| --- | --- |
| 750 | Router `vllm.unquantized_gemm`: real input/weight/output observed, router row bridge not yet enabled |
| 752 | `_C_ascend.moe_gating_top_k`: probabilities and int32 expert IDs have shape `[8,6]`; actual scalar argument values are not retained |
| 755 | `_C_ascend.npu_moe_init_routing_custom`: int8 routed data `[48,2048]`, int32 row indices `[48]`, int64 expert counts `[64]`, float scale `[48]`; no integer values retained |
| 757–758 | Actual count cumsum then grouped matmul/SwiGLU/quant; exact object chain is visible, routed row order is unproven |
| 760 | `aten.set_` releases/rebinds the earlier routed-input Tensor; later object/storage reuse is visible |
| 761–763 | Grouped matmul, actual `abs` of routing indices, then `npu_moe_token_unpermute`; original probability object reaches the consumer |
| 934–939 | Routed alias, scaling and shared add remain unproven; the next AddRms receives that unknown input |

In both forwards the unpermute output is the exact routed return of the MoE
parent. There is no gap in call observation here. The missing part is the
semantic mapping of routed slots back to the original request rows. Neither
`48 == 8 * 6`, equal tensor shapes nor equal values is sufficient evidence.
The current source83 event schema contains no actual scalar-argument ledger
or complete integer readbacks, so these cannot be reconstructed retrospectively.

## Proposed next increment, pending review

1. Bind router GEMM to its exact current registered implementation and actual
   model-owned gate weight, using the existing reviewed GEMM guard. Propagate
   rows only through its actual input/return. Retain the conservative outer
   MoE invalidation and current-source admission checks.
2. Extend the same source/profile owner to the real top-k producer,
   routing producer, expert MLP and combine frames. Reuse the object-join
   checks in `NaturalMoETrace`/`MoERestoreTrace` as design references, without
   installing a competing profile or inheriting Fault8 qualification.
   Record actual bound scalar/optional arguments, not schema defaults.
3. Capture bounded integer arrays while their real producer/consumer objects
   are live: expert IDs, signed expanded row indices, actual `abs` indices,
   expert counts and the actual cumulative group list. For the observed
   geometry these are 48/48/48 int32 and 64/64 int64 elements: 1,600 bytes per
   snapshot. Record readback operations separately, with parent/source call,
   thread/device/stream, object/storage/version/epoch and full array hash.
   Observe changes before consumption and reject them; do not replay a model
   operator or keep stale pointers after event 760. Existing call budgets
   remain unchanged until actual measured costs are reviewed.
4. Keep a separate bounded token-route proof for repeated expert rows, with
   origin `(request, token position, route slot, expert)` and actual group
   boundaries. Do not put 48 duplicated requests into the eight-request row
   graph. Validate the documented index direction, sign, range, uniqueness,
   counts and optional padding/drop semantics only after obtaining the exact
   production implementation contract. Prove actual grouped MLP row order and
   live model weight ownership; then verify that unpermute consumes that same
   route proof, indices and probability producer. Only this complete inverse
   join may restore the original B8 row proof and enable the existing shared
   add/AddRms rules.

Still needed before implementation: the pinned production
`experts_selector.py`, `moe_comm_method.py`, `token_dispatcher.py`, the actual
quantized expert-MLP implementation, and the custom routing kernel contract
for `row_idx_type` and signed indices. The existing three Python source pins
in `moe_census.py` identify relevant files but do not substitute for reading
their current complete source or observing actual argument values.

CPU negatives should cover wrong/duplicate/missing/out-of-range indices,
changed counts or cumsum, wrong route direction, unsupported padding/drop,
cross-request/cross-call joins, alias writes and allocator reuse, replaced
weights and changed current registration. A future zero-fault two-healthy
census should retain all 135 candidates and demonstrate those actual maps
before any proposed group qualification. Propagation-time calls must observe
their own current routing; healthy-run indices cannot authorize later calls.
