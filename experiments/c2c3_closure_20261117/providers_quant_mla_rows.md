# Source-bound MLA row observation, CPU preparation

This increment adds a conditional row adapter to the isolated group-ABI B8
inventory. It changes no Native source or ABI and still uses the reviewed
`native-quant-group-v1-a4` bundle. It is not an NPU result, a qualified target
set, or C1/C2/C3 acceptance. The complete structural/public/Native denominator
and every unverified candidate remain in the existing inventory.

The early observer starts in the existing host `begin_decode_capture()` call,
before the next real `llm.step()`. It observes the actual
`AscendMLAMetadataBuilder.build_decode_metadata` and
`get_cos_and_sin_mla` calls, without replacing either callable. The source file,
compiled code, loaded function, module globals, actual runner/thread and
profile ownership are checked. The trace must contain both actual cache-index
operations, their actual positions tensor, the corresponding full buffer
copies and the actual returned decode-metadata object. The model pre-hook joins
the producer's position storage to the already verified scheduler/model
positions. No trigonometric oracle, equal-value substitute or producer replay
creates this proof.

The model observer enters an owned nested dispatch mode inside the actual
`mla_forward` Python implementation. Current reviewed paths include explicit
query-head axis changes, the installed derived `W_UK_T`/`W_UV` tensors,
normalization with actual frozen parameters, rotary operands from the observed
metadata producer, eager nonquantized FIA with a complete pure B8 decode,
NBSD-to-B output transfer and actual full-extent output copies. Both row-carrying
operands of an in-place sum must have the same proved mapping; an unproved
same-shape operand is refused. Scalar and current owned broadcast parameters
have their separate checks. All aliases receive write epochs, including
inference tensors without `_version` and exact-extent views without `_base`.

Source and intermediate proofs are not inferred from dimensions. A missing
producer, changed positions/order, changed cache/index/output identity, later
write, replacement weight/function/globals, unknown FIA layout, prefill,
speculation, FA quantization, graph capture or truncated trace cannot acquire
rows. The unreviewed opaque MoE boundary also hides existing outer input proofs
from its internal public projection calls. It cannot silently borrow a known
input tensor's prior label. The full MoE routing/index/MLP/combine adapter is
still outstanding.

Source extraction used only existing layers of pinned image
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`.
No container, model import, weight read or NPU call was used. Raw source and
manifests stay on 124 under `/home/17-fault-c2c3-20261117/quant-row-source-a1`
through `quant-row-source-a6`. The production pins are in `quant_source_rows.py`
and `quant_mla_metadata_rows.py`; they include the actual extracted MLA,
custom-op registration and rotary files. A different loaded source disables
the adapter with its exact refusal in the inventory report; it does not remove
any target from the denominator. The
[Ascend transpose-batchmatmul converter](https://gitee.com/ascend/torchair/blob/89b42adf4dd499bb1de7c6ab9ba91263d9ac0096/python/torchair/_ge_concrete_graph/ge_converter/custom/transpose_batchmatmul.py)
also documents the schema used for the explicit output permutation; runtime
arguments and actual installed weight identity are still checked.

Validation is CPU-only. The focused suite exercises real PyTorch dispatcher,
profile and serving-hook lifecycle calls with explicit CPU operator/journal
substitutes. It covers the positive metadata/index/copy/consumer join, complete
FIA/V-up row transfer, foreign or missing producers, identical-value clones,
storage reuse and writes, row-order changes, unknown layouts, budget gaps,
opaque-call exceptions, changed code/globals and observer cleanup. These tests
do not validate CANN numerical behavior or claim a naturally supported target.

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension;fault39/tests'
& 'C:/Users/admin/.codex/worktrees/fault28-native-fa-publish/demoncatpfi/.codex-tmp/unified-env/Scripts/python.exe' -m unittest test_quant_source_rows test_quant_mla_decode_rows test_quant_mla_metadata_rows test_quant_group_inventory test_shared_scale_inventory test_quant_inventory test_quant_row_provenance test_worker_quant test_quant_group_binding
```

Root's next action is a new source/install package and the existing
`providers_quant_prepare_inventory_case.py --group-abi` command documented in
`providers_quant_group_natural.md`, using the unchanged a4 bundle. The probe
now hashes these two installed source modules as well. The case is read-only
and automatically records which actual candidates have rows; it installs no
Native fault owner. Profile/mode ownership and retained live metadata records
are explicitly checked on removal. Live producer tensors are released at the
end of the actual model forward; detached reports cannot authorize a later
consumer.

Worker Core V1 can reuse this boundary through three actual callbacks:
before the host's next decode (actual pending request IDs and runner), before
model entry (verified ordered rows and actual positions tensor), and after the
same model forward (stop observations, release producer references). A future
limited C1 controller may use only naturally proved candidates with actual
per-target reference/lossless/current-input effect/OFF and same-Worker recovery.
It must retain the other candidates as unverified and must keep functional
effect acceptance separate from E5 statistics. Group `all_eligible` and a live
C2-to-C3 transition still require the full original group contract.

The cleanup review adds explicit ownership phases for the early profile,
outer graph mode and nested opaque-call modes. A frame return only releases
its tensor references; an independent ledger retains every nested mode until
the actual dispatcher stack confirms removal. Cleanup refuses a foreign top
mode or profile and leaves the Worker owner active. After the foreign owner
removes itself, the original thread can retry without exiting an already
removed mode twice. Reports expose retained nested modes and cleanup pending;
unregister checks both before releasing the Worker owner.

The focused command above now passes 110 CPU tests. The added regressions use
actual dispatcher stacks, including a foreign mode installed from the pinned
CPU kernel's model property access, exits that raise before or after popping,
profile removal followed by a retry, foreign profile/thread refusal, and a
metadata release failure. They verify no false zero-resource report, no
foreign-mode pop, and ordinary same-process CPU operations after cleanup.
This remains CPU observer lifecycle evidence, with no CANN execution or C1,
C2 or C3 acceptance.
