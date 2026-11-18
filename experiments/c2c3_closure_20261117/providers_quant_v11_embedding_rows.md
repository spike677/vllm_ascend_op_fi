# V1.1 quant: actual TP1 embedding producer and residual branch

Source70's two healthy first-decode B8 batches retained all 270 Native
parents, prepares and executes, zero faults, identical healthy replays and
zero owned resources. Each request had three row-proven projections after
the singleton-view repair. The full inventory SHA256 is
`f68a3b2296ca3aae2d579c3252eee89a088fa1d6346c20da0e128a1e1bf5b114`.
This original remains on124; the provider agent used only root's small
non-tensor metadata extraction and exact image source files.

The next lost branch was not the same object as the recorded embedding.
The actual sequence establishes why: event2 `aten.embedding` produced the
embedding, event3 `maybe_pad_and_reduce` returned that same object, event4
`aten.clone` produced the later residual, and event5 RMSNorm produced the
first seeded normalized tensor. Thus the residual's real clone happened
before the old graph seed. Later event291 `maybe_chunk_residual` returned
exactly that residual object; no shape-based seed is needed or permitted.

The new source-bound profile observes the already executing
`AscendVocabParallelEmbedding._forward_origin` and its actual children. It
does not call the model, lookup or communication function again. Its input
must be the exact current `input_ids` tensor supplied by the existing
`snapshot_first_decode` handoff, whose device readback already matches the
actual scheduler IDs/positions. The actual model parameter object, metadata
and observed write epoch must remain current. The accepted path consists of:

1. Optional actual `aten.to.dtype` converting int32 to int64 with the observed
   default nonblocking/copy/memory-format arguments; an already int64 input
   can remain the same object.
2. Actual `aten.embedding` using that exact indices tensor and the current
   model's frozen weight, with observed `padding_idx=-1`, no scale-by-frequency
   and no sparse mode. The source-bound embedding method must receive the
   same owning layer, indices and result objects.
3. Actual `maybe_pad_and_reduce` with `is_ep_comm=False`, then the observed
   `tensor_model_parallel_all_reduce` → `get_tp_group` →
   `GroupCoordinator.all_reduce` call chain. The returned group must be the
   actual embedding module's group, with world size1/rank0. The checked
   `all_reduce` implementation must take its original-object early return,
   with no observed child operators. All return objects and source-call
   parent sequences must agree.

Only the complete source return seeds the embedding. Its later actual clone
then uses the existing row-preserving rule; the existing no-chunk and RMSNorm
rules still require both actual operand row proofs. Unknown/permuted residuals
never borrow rows from the other operand. Unknown embedding operations,
nondefault arguments, TP>1, alternate group objects, parameter replacements,
mutated IDs and incomplete source children refuse the new seed. Actual
operator/schema/scalar arguments and source-child receipts are saved in the
new node; source70's missing scalar arguments are not retroactively inferred
from schema defaults.

The new source closure is pinned to the same image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`:

| Source | SHA256 |
| --- | --- |
| Ascend embedding | `27d9c4f38c11d051b22dadc903ed0009269a303199639287ba937d882442f094` |
| vLLM embedding method | `9dff6cab7c17798991cc48e854f275a6b596cc637e5694f166c20cdc68d31879` |
| Ascend custom registrations | `c5eaccde6568f6f480c00a07cc582c8650e6849f25933c8e82787f4b0477ca9b` |
| vLLM communication wrapper | `3c3f8ac60db38ece891d39dc1e2f6f1947ad438685893d2e5777d072320bd1b8` |
| vLLM group implementation | `1cb040ec56cca83fc82613413b8bda2aa697c71981ba895189713e902ff4e570` |

All functions are checked against loaded source bytes, compiled code,
function/global ownership and actual frames. The current scheduler tensor is
strongly held until the source observer ends, then cleared. The zero-resource
report and remove gate include this reference; foreign profile ownership
retains the reference and permits retry only after the owner is restored.
Failed installation before taking profile ownership retains no input tensor.

CPU tests exercise real dispatcher/profile calls with explicit CPU source
and Native-op substitutes. They do not qualify production communication,
Attention, MoE or a Worker. Root should freeze a new Python package and reuse
the unchanged source64 Native build
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`.
The same two-healthy, zero-fault controlled inventory entry and manifest
command in `providers_quant_v11_singleton_views.md` apply. This is neither a
repeat of C1 nor public Core V1.1 C2/C3 acceptance. Keep all 135 candidates per
request and each remaining refusal, then inspect the next actual unresolved
input. No current full-set/C2/C3 qualification is claimed.
