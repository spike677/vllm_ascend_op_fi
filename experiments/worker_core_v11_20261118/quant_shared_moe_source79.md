# Source79 shared-expert row boundary

The two source79 healthy B8 requests have 270 actual projection/Native calls,
with eight row-proven projection sites per request. This is not a 135-target
qualified set. The first unproven projection is
`model.layers.1.mlp.shared_experts.gate_up_proj`. Its actual input is the
already-proven view at event 749; event 751 is
`vllm.moe_forward_shared.default`, whose schema declares `hidden_states`
mutable. The previous observer correctly invalidates that storage and its
aliases. Its two outputs must not acquire row labels from their shapes.

The supplied source79 metadata identifies source commit
`94b4cefd67cd5c55b80699ec4a413215b2f9a7a0` and inventory SHA256
`d59e19e9c8c93c8e7cbec1c346f4600641d59e37c9ee7be0271c40cb52bc6bd8`.
The production source closure shows a separate shared branch inside
`AscendSharedFusedMoE`. Routed experts execute first, then
`_forward_shared_experts` receives the original hidden-state object. This
source fact alone does not prove that an actual call preserved its input.

The additive observer binds the loaded default runner and Ascend functions,
their compiled code/globals/source hashes, the actual held model paths, and
the current `moe_forward_shared` registration. The pinned Torch 2.9 dispatcher
must show a single active PrivateUse1 kernel, no shadowed/inactive kernel,
and the exact reviewed library-construction origin. A public Library override
is rejected even when it calls the original reviewed Python function.

The outer mutable invalidation remains unchanged. The observer suspends a
copy of the current input proof before it, installs nested read-only dispatch
observation at the actual opaque implementation entry, and permits that proof
to resume only at the actual reviewed runner frame. At this point there must
have been no child operator. The actual forward-context lookup, model,
runner, shared module, configuration, original input object, complete
metadata, row digest and exactly one parent invalidation epoch must match.
Only serial, untransformed TP/DP/EP/SP/PCP=1, inference-mode, eval-model calls
are in scope. Input/router storage overlap, transformed inputs, overlap
streams, expert gates and changed owners remain unsupported.

Every nested operator is then observed normally. A real child write or an
unknown mutating operator invalidates the input, including inference tensors
without a version counter. Entering shared part1/part2 rechecks the original
object/epoch and the exact owner. Public row queries are visible only inside
those actual shared frames; routed pipeline queries remain unproven. A
shared output needs a complete observed operator chain and an exact return
object, disjoint from the input and routed output. The routed result is never
certified by this adapter. All 135 candidates and all unknown reasons remain
in the census. Nothing changes the quantization algorithm or Native ABI.

`shared_moe_row_diagnostic` on each actual parent records the first refusal,
raw current registration/origin, actual input metadata, source event and
model/runner/shared identities, resumed node and observed shared output.
`shared_moe_adapter` reports whether the optional source closure was enabled.
These are detached snapshots. Suspended input/output references live only
until their actual operator returns or fails. Existing retryable profile/mode
cleanup remains responsible for all owned observers.

Local CPU validation passed 193 tests across row, source, MLA, metadata,
embedding, parameter, inventory, C1 and cleanup contracts. A final 36-test
subset passed after adding configuration-identity, operand-alias and snapshot
isolation tests; it includes 16 new real CPU dispatcher/profile tests. The
tests also cover two inference forwards, unknown rows, actual child mutation,
input replacement, foreign model lookup, training, overlap/parallel modes,
registration override and original-call failure with cleanup. These are CPU
contracts, not production NPU row qualification.

Root should freeze a new Python package, include `quant_moe_rows.py` and its
test in the source-only export, run production Torch 2.9 CPU contracts, then
queue the existing controlled two-healthy, zero-fault probe:

```
python /source/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py --group-abi --build-sha256 b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac
```

Use the same reviewed image/model/cards and root queue gates as source79.
No SDK rebuild is needed. Both quant probes include the new installed module
in their loaded-source hashes. The controlled census is not public RPC C2/C3
acceptance. The expected distinguishing observation is whether the first
shared gate/up and down projections obtain actual input/output row proofs;
the later routed combine remains unknown. Preserve any first new refusal,
observer overflow, complete Native/public denominator, healthy repeat and
zero owned hook/mode/profile resources. No C1/C2/C3 status is granted by this
commit.
