# V1.1 quant: source-bound MLA singleton views

This is a Python row-provenance change, not a new fault algorithm or Native
ABI. It grants no new C1/C2/C3 status. The source64 one-site C1 originals stay
unchanged; the remaining 133 candidates are not relabeled by this change.

The root's non-tensor extraction from source64 identifies the first lost edge
in the first healthy decode:

| Actual event | Input | Output | Existing result |
| --- | --- | --- | --- |
| 187, `aten.view.default` | shape `[8,16,64]`, stride `[3072,192,1]`, offset 128 | shape `[8,16,1,64]`, stride `[3072,192,64,1]`, same storage/offset | no row proof |
| 197, `aten.view.default` | shape `[8,16,512]`, stride `[512,4096,1]` | shape `[8,16,1,512]`, stride `[512,4096,512,1]`, same storage/offset | no row proof |
| 198, `aten.contiguous.default` | the preceding query view | default contiguous copy | no row proof |

The exact loaded MLA source remains pinned to
`17892014ecd431a1804fb81f5380509275e1440b41e1f9e0870ba8440c365516`.
`rope_single` inserts S=1 after the feature split; `_forward_decode` inserts
S=1 after the head-axis transpose and calls `contiguous()`. The previous
rule required contiguous tensors even for those actual singleton views.

The new rule is restricted to a currently observed real `aten.view.default`
inside an already source/code/globals/owner-checked MLA frame. It requires
the current input proof/write epoch, exact input snapshot, identical storage
identity/address/extent, dtype/device/offset and identical ordered pairs of
every non-singleton dimension and stride. The request axis must remain one
separate dimension. Only size-one dimensions may appear or disappear. The
result retains the actual parent node, call sequence, source boundary,
input/output metadata and epoch in its proof.

The following actual `aten.contiguous.default` is admitted only in that MLA
source scope, with its actual input object, default contiguous memory format,
unchanged shape/dtype/device, current epoch and a truly contiguous result.
Its result must be the original object or disjoint storage. Nondefault memory
formats and calls outside the source scope remain unknown. No tensor values
or CPU quantization oracle are used.

CPU contracts cover both source64 strides, a nonzero feature offset,
noncontiguous decode inputs through the existing CPU FIA/output bridge,
request-axis preservation, no proof from unknown input, altered shape/stride/
storage/offset rejection, alias writes under inference mode, two successive
inference forwards and all existing fresh Native-admission/cleanup contracts.
These tests do not establish a natural Attention or MoE result.

Validation: 101 CPU tests passed across `test_quant_mla_view_rows`,
`test_quant_mla_decode_rows`, `test_quant_source_rows`,
`test_quant_row_provenance`, `test_quant_group_c1` and
`test_quant_group_inventory`; 12 are new singleton/default-contiguous cases.

## Root's minimal discriminating run

Freeze/install the new Python source. Reuse the exact source64 Native build
manifest SHA256
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`;
no Native source changes or SDK rebuild are needed. The root supplies the
existing immutable bundle path and rechecks every bundle file as usual.

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --group-abi --installed "$INSTALLED" --quant "$SOURCE64_NATIVE_BUNDLE" \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_inventory_probe.py" \
  --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --model /home/haoran/models/DeepSeek-V2-Lite-W8A8 --card 2 \
  --out "$NEW_ATTEMPT_CASES"
```

This is the existing **two-healthy-decode inventory entry**. Do not pass
`--single-site-c1` and do not use `providers_quant_c1_probe.py`: the latter has
the seven qualification/fault phases. The inventory entry enqueues two fresh
B8 batches, captures each first decode, explicitly collects, and performs
same-Worker health and complete observer cleanup. It never installs a fault
owner or calls Native arm/admit. Expected faults and helpers are zero.
It is a controlled observation entry using deployment RPCs, not an acceptance
run of the public Worker Core V1.1 fault API. Its evidence cannot grant public
RPC C2/C3; those require a later run through the matching public provider entry.

Retain the entire independent Native/public inventory and row graph in the
new attempt, including each of the expected 135 actual parents per decode
(expectation is not a qualification). Compare the two actual candidate sets,
per-site row-proof status, first unresolved public input and its preceding
actual operator/source-boundary events. Missing or extra calls, unknown
MoE/Attention/indexing, any observer error and all original refusal reasons
remain visible. A successful census alone is not C2: all eligible targets
still require independent stable identities, per-target reference/lossless/OFF,
group recovery and then an actual bounded all-target once run before live C3.
