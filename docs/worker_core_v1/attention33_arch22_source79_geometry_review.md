# Attention33 source79 normal parameter review

This is a CPU review of the source79 private normal call's parameter refusal.
It fixes a Python/reviewer interpretation error and does not establish a
stock/private output bridge, Native qualification or permission to run F.
No Native source, library, tiling arithmetic or fault gate changes.

## Observed refusal

The preserved attempt is
`/home/17-fault-c2c3-20261117/v11-33-normal-source79-card5-a2`.
The stock arm completed five phases. The private arm retained only
`baseline_before` and failed with
`ARCH22_FINAL_PUBLIC_ARGUMENT_GEOMETRY_DIFFERS`. Its one host record belongs
to the first actual API return (`ORIGINAL_RETURN_CAPTURED`, original calls
1). This was the second validation check, not evidence of a second API call.
The failed result remains failed and unchanged.

Only `maskParams.nextToken` differed from the old expected geometry: actual
Native `0`, expected `2147483647`. The other 18 expected fields matched.
Both omitted Python token defaults are `2147483647`; these declarations do
not state the effective Native values.

The actual `attention` source explains the difference:

1. `GetPreNextToken` calls `UpdatePreNextTokenBySparseMode` before clamping.
   `SPARSE_MODE_RIGHT_DOWN = 3` sets pre/next to `(2147483647, 0)`.
2. `GenerateInfo` copies the effective values into `FiaTilingInfo`.
3. Nonquant `FillTilingMaskParams` serializes these values into the POD.
4. The later `ANTI_QUANT && s1Size == 1` override is excluded by the actual
   FP16 Q/K/V, absent optional quant inputs and zero quant modes. Legacy IFA
   is also excluded by the decoded `isLegacyIfa == 0`.

The runtime and independent reader pin the same actual built sources under
`build/run/source`; `experimental/attention` cannot substitute for them:

| Source relative to `source/` | SHA256 |
| --- | --- |
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp` | `380248471fef863d9c7ae5f77f00aafb71185330d350388652c6f01557a9f271` |
| `attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.h` | `54753b541981eb176636bc0243c728830f5076a39d4e69721816d0d4eb6d44f0` |
| `attention/common/op_host/fia_tiling_info.h` | `d201e4ed6ca93fb90a7a60c9da5fbfbb5727810768ffd6cd74e8c67115febda4` |

## Independent CPU replay

The locally provided original metadata identities are:

| Original file | SHA256 |
| --- | --- |
| `host.jsonl` | `295d444122b921587a112021a632e1eaa593e44a45aacf9d29940089216f5632` |
| `runtime.jsonl` | `9fd90fb7329896ea8e340c1ed1d3fefb7af4b26534b57ec36386c1df0dec4d3c` |
| `boundary.json` | `8e744ec8a5bbd1511d803dbee3b7263b4d001ebd460551941b1939daf09b5092` |
| `fia_abi_schema.json` | `20dd6a361142aa9acd73f0787673e24660daef46846e5b362eb9306b15fe035c` |

The actual compiled kernel declaration and matching metadata derive tiling
pointer slot 280. The observed host-placeholder descriptor locates its POD
at byte 296. All 1576 bytes match the host snapshot; all 65 decoded fields,
including both serializer alignment fields, reconcile independently. POD
SHA256 is `2b7b4ee666c645af04f154d13c089c3a28ccd0fc87f6899f081d2a8ff9460010`.
The actual entry remains key `103000000010300003`, block 16, schedule 1.

This replay used pure decoding/validation functions and local JSON/source
files. It did not import torch, load a DSO or access an NPU. Registered ELF
identity was supplied separately; its actual bytes were not downloaded or
rehashed here. No tensor bins were available for this local diagnosis, so
there is no new claim about immediate output or full-model equality.

## Repair and remaining gate

Runtime and reviewer now require precisely `(INTMAX, 0)` for the supported
nonquant sparse3 route. Native `nextToken == INTMAX` is rejected even when
both host and raw snapshots agree on it. Other scalar/shape/capacity checks,
complete POD/array/alignment checks, original-call-once, actual cache binding
and owned-resource recovery remain in force. The receipts separately record
supplied Python attributes, declared omitted defaults and actual decoded
Native values; the original call dictionary is never rewritten.

CPU contracts cover all checked geometry fields, native default substitution,
other sparse/legacy modes, quantized Q/K/V with matching audit metadata,
optional quant/padding features, changed parser/header bytes, missing actual
source with an experimental copy, unchanged arguments/return identity, F
refusal and cleanup failure/retry. The complete generated two-arm fixture
still checks five phases, 435 full payload pairs and 112 private normal
calls. These are CPU fixtures, not Native qualification.

Root must freeze the repaired Python and reviewer sources, install their
matching packages, and prepare a new pair of normal-only cases with the
[existing preparer and independent reviewer](attention33_arch22_normal_review.md).
The same unchanged arch22 Native build can be used, with its complete
`build/run/source` directory mounted read-only so the new pins can be checked.
Keep the source79 failure and earlier 67-element output discrepancy. Only
fresh complete raw evidence can resolve the numeric normal bridge. F remains
closed, and C1/C2/C3/E5 are not granted by this repair.
