# Fault33 actual FAI strategy review

Status: source implementation and CPU contracts only. No C2/C3, public Worker,
device-buffer legality, statistical effect, or native-launch qualification is
granted by this review. The root queue must build and execute the new bundle.

## Actual route, and rejected alternatives

The reviewed vLLM 0.18.0 `attention_v1.py` SHA
`bbe69c3015ee15789dea0d6ec9795243458b68494294c4c023ed5a5d665c550b`
calls `npu_fused_infer_attention_score`, TND/page16, for natural Qwen decode.
Prefill also calls FIA; `_get_fia_params` supplies native no-cache TND input.

Pinned ops-transformer `c77c0829f1f0f0b8f882db1d9b051a758b605472`
`op_host/arch22/fused_infer_attention_score_tiling_v3.cpp` has explicit routing:
`CheckSpecCondNoPA` (1165), `CheckSpecCondPA3Dim` (1171), and
`CheckSpecConditions` (1199) accept FP16 GQA, innerPrecise0, D128, sparse3,
and page16. `RouteToFia` (1411) consequently returns false. The actual host is
`TilingProcess4SplitFuse` / `FAInferTiling`, with `FAInferTilingData`.
The previous four-key arch22 build `tnd33-build-a2` genuinely completed, but its
strategy is not reached. Its schema and artifacts remain investigation evidence.

A split-core proposal was rejected before implementation: actual root-run 28
host receipts in `attention28-launch-a1/fia-legal-baseline/run/host-{0,1}.jsonl`
report AIC=16, AIV=32, blockDim=16. With B1/Hkv8/KV1025, natural FAI flash
decode is false: 8 exceeds 0.4*16, while the long-sequence threshold is8192.
This implementation does not force routing or decode flags. Its name is
`tnd_page16_fai_s2_256`, independent from the old BNSD and arch22 strategies.

## Source changes

`apply_tnd33_fai_policy.py` requires a pristine pinned checkout and makes all
anchor checks before writing. It changes four official files:

- `flash_attention_infer_tiling.h`: a default512 member selects the KS tile;
  all four workspace slices use its Q128*tile block size. The selected256 key
  adds a reserved offset. Official task and workspace calculations still run.
- `fused_infer_attention_score_tiling.cpp`: selection occurs inside the actual
  FAI route before `DoTiling`. The final external allocation uses the same block
  size, then serializes and captures all52 public fields and all raw bytes.
- `fia_kernel_common.hpp`: only the two new keys compile MAX_KV_STACK_LEN=256.
  Every original key still compiles512. The kernel workspace block size,
  KS-loop count/start/tail, paged block stack, MMAD initialization and masks
  consume that constant through the original FAI implementation.
- `fused_infer_attention_score.cpp`: the two new keys are registered exactly
  once and select the same non-FD FP16 TND paged templates with the new constant.

Eligibility is restricted to the actual 16-core, FP16, B1/Hq16/Hkv8/D128,
query1, KV1025..2048, TND/page16, sparse3, precision0 route. Neither FD nor
decoding optimization, sink, PSE, NZ cache, tiling sink, or LSE output is
accepted. API arguments and all tensor/cache bytes are retained.

Compiled key set (decimal strings preserve uint64 identity):

| Purpose | No mask | Causal mask |
| --- | --- | --- |
| Original no-cache prefill512 | `5000000000000200100` | `5000000000000200103` |
| Original paged decode512 | `5000000000010200100` | `5000000000010200103` |
| Scoped paged decode256 | `5900000000010200100` | `5900000000010200103` |

Each key must occur exactly once in the fresh compiler metadata and binary
registration support list. No prebuilt kernel is copied. A full8.5.1 host and
kernel rebuild uses the previously reviewed unsupported-SoC registration fix.

## Buffer and dependent geometry checks

The reviewed shape has Q tile128 and D128. KV256 is divisible by page16 and
the128-wide matrix tile, and a Q128*KV256 allocation exceeds Q128*D128 output
storage. This establishes required bounds; it does not prove device execution.

For16 cores and the existing triple workspace allocation, define
`unit=16*128*tile*3`. Actual host CPU execution must produce MM1=4*unit,
softmax=2*unit, MM2=4*unit, update=4*unit, total=14*unit; zero split-LSE and
split-O bytes; eight complete head-group tasks. The external allocation is
16MiB+16*128*tile*4*3*4. `compile_tnd33_fai_schema.py` executes the actual
fresh host class for both masks and both tile sizes, and generates an independent
public serializer layout using scalar and every array-element sentinels.
The builder runs these checks before compiling the matching device binary.

NPU validation must independently bind the actual function/key to the exact
new registered ELF and capture final inline raw arguments. The runtime adapter
requires the final entry, all geometry fields, workspace slices, schedule1 and
blockDim16 to agree. A key constant in source alone cannot satisfy this check.

## Root build and execution handoff

Root only: run `build_tnd33_124.py --source <new-source> --out <fresh-a3>
--interfaces <exact8.5.1-sdk_interfaces.json> --strategy actual-fai`.
It uses the standard8.5.1 Worker image, runc, no devices, no network, no cache
objects. The FAI schema is emitted at
`build/run/evidence/fai_schema/fai_abi_schema.json`, bound into completion.

Use a1024-token legitimate prefill followed by two generated tokens, yielding
one natural first decode with KV1025. Before accepting a fault case, require
baseline repeatability, complete hidden/logit/layer/cache results, an OFF stage
using the original key with exact represented bytes, cleared scope/wrappers,
and healthy requests. Capture B/F statistics using identical configuration and
paths; only the independent raw comparator may choose E5. Preserve null/NaN,
missing fields, and stack/construction mismatches. No historical percentage or
old certificate is inherited.
