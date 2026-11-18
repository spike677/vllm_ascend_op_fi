# Quant ND exact flat storage: source61 evidence-constrained adapter

Root's source61 healthy run `quant-metadata-inventory-source61-card2-a1`
completed two B8 first decodes, with 270 actual Native model calls and no fault.
The original artifacts remain on 124:

- result SHA256: `42620b2db6724092a1d48dcd051a9e3e00fa3b7dfb2c3284a74c71905d98d7da`
- inventory SHA256: `a98019062d5b474fd07fa9d464dbee0fc794c288bf94357b97dd878e533b0c59`
- diagnostic SDK bundle build.json SHA256: `b032fe3e57ee3258eb61b35baba5c8a20519b755a78a68020f776d50659db7a4`

The first complete parent was 2296. Its input was BF16/dtype27, ND/format2,
offset0, view/stride/storage counts `[2,2,1]`, view `[8,2048]`, strides
`[2048,1]` and storage `[16384]`. All seven getters returned zero, the call mask
was 127, and null/truncated arrays were zero. Q had identical geometry and int8/
dtype2. Scale was fp32/dtype0, rank1, shape/storage `[8]`, stride `[1]`, bytes 32.
The 270 actual input layouts comprise 216 width 2048, two width 10944 and 52
width 2816 calls. Every input used the same exact ND flat-storage representation.
These are quant's own observed descriptors; FIA's NCL descriptors are unrelated.

The new `view()` branch accepts only rank2 view/stride with rank1 storage,
ND format, observed BF16 or int8 dtype, offset0, stride `[columns,1]`, positive
bounded dimensions and exact `storage[0] == rows*columns`. Its byte span must be
nonzero, nonoverflowing and element-aligned. Padding, undersized storage,
strided views, offset slices, unknown formats/dtypes, rank mismatches and missing
getter results remain rejected. Original rank1 and same-rank storage handling
is retained. FP16 flat storage remains unverified; its existing same-rank path
is unchanged. No dimension, dtype, address or byte count comes from public
Tensor metadata.

The actual original descriptors and raw getter arrays are preserved. Canonical
View describes the proven logical byte span, not a recreated descriptor. Native
correct-domain reference and flattened fault helpers still use their original
reviewed preparation and execution rules. The engine, numerical fault, original
C1 library, child registration/lease identity, workspace gates and failure
recovery are unchanged. Every actual helper remains in the raw census.

The C++ memory-double suite now has 27 scenarios. New cases exercise full
reference/lossless/OFF/F/helper/OFF from flat original descriptors; all three
actual B8 widths including 10944 and 2816; padding, short/negative storage,
offset, stride, format, dtype, null/misaligned/overflowing address, dimension
overflow; and prepared executors retained after insufficient execute workspace.
All SDK getter arrays must be released. These tests are prepared for root's
fresh SDK build; local Python results do not constitute a C++ or NPU result.

Root's source63 build `native-quant-group-nd-flat-a1` compiled successfully but
failed the fifteenth runtime scenario, the existing padded-storage passthrough
contract. The original receipt and failed bundle must remain unchanged. This
was a production passthrough defect: the installed observer authenticated its
unselected-call receipt with `same_call`, whose `same_view` rejects refusal
flags even when both records are identical. It also affected same-rank padded
storage and rank1 padded scale; the padding itself correctly remained ineligible.

The repair changes only `dcat_dqg_pass_v1` to compare every actual Call/View field
exactly, including refusal flags, under the existing current callback, owner,
thread, device and once-only decision guards. It leaves `same_call`, `same_view`,
`valid_call`, Native admission and owned helper validation unchanged. The
original negative scenario still requires the real original GWS and execute to
complete once while preserving refused Native metadata. Three additional
contracts cover same-rank/rank1 padding passthrough, twelve altered receipt
fields refused before any original call, and exact refused observations that
cannot authorize admission. No test accepts a model failure as passthrough.
Local validation after this repair: 63 Python tests passed across build,
binding, diagnostics, device-free load-probe fixtures, initialization diagnostics
and inventory. There is no local C++ compiler; the 27 runtime scenarios remain
pending root's fresh SDK execution.

Python CPU contracts verify that the exact source/bundle fingerprint is required
even when all ABI sizes are unchanged. A prior source profile is refused before
CDLL. The public/native join still compares actual native geometry; valid raw
storage diagnostics cannot fill the zeros of an older rejected Native View.
The actual workspace requirement remains enforced independently.

Use a fresh immutable source/package and build directory:

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_build_group.py" \
  --source "$SOURCE" --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --out /home/17-fault-c2c3-20261117/native-quant-group-nd-flat-a2
```

The additive diagnostic ABI sizes remain snapshot 416, record 1304, state 72.
Nevertheless the production source and DSO fingerprint change, so the source61
bundle and failed source63 a1 cannot be reused. Root must compile the 27 runtime
scenarios plus the existing core and seven ELF loader contracts, record actual new SHA values,
and run the existing device-free managed load probe before a device attempt.

Source61's two healthy requests and final cleanup passed, but native/public and
row qualification were false. The metadata builder also requires the separately
reviewed actual second slice chain. This layout patch alone grants no target,
C1, C2, C3 or E5 status, and no unknown candidate is removed from the denominator.
Root alone controls the next device case. This agent has not run SDK, containers,
SSH or NPU for the patch.
