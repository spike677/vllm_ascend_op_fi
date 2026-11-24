# Attention28 TND prefill: source review and implementation boundary

CPU/source design only, based on the current public entry and the retained
c77-derived build sources. This does not change a Provider, Native algorithm,
fault definition or admission gate. No NPU, container, SSH or build was used.
The three mechanisms remain distinct; none inherits Fault33 acceptance.

## The first FIA implementation must be a real TND prefill adapter

Reuse `AttentionExecutionAdapter`'s owned actual-call observation and original
continuation. Add a dedicated TND prefill validator/native driver rather than
altering the original BNSD validator or converting QKV to BNSD. The first scope
is the observed single fresh 2048-token request described below.
Quant/prefix/rope/padding/sink absence, physical tensor metadata and omitted
scalar defaults still need actual typed admission. `block_size=128` alone does
not prove paging: the observed `block_table` is null.

The original actual API must remain `npu_fused_infer_attention_score` for the
two FIA mechanisms. It is called exactly once with the original arguments,
mask, return object, current request and Worker owner. FA requires a separately
observed real FA invocation; a FIA API name or matching QKV shape is not FA.

### What the source71 original actually establishes

The local gzip expands to 32,635,935 bytes with SHA256
`6a3c6adf0e1090899057f7e0d37a6a041f1ed9ecc2fe736c5d90685040469b89`.
The checked source is `requests[*].worker.forwards[*].calls`, not the top-level
status or qualification booleans. The artifact is not committed here.

| Actual field | Both requests, all 28 calls per request |
| --- | --- |
| API and layout | `torch_npu.npu_fused_infer_attention_score`, `TND` |
| Q / K / V | FP16 `[2048,16,128]` / `[2048,8,128]` / `[2048,8,128]` |
| Heads and lengths | 16 / 8, Q/KV cumulative lengths `[2048]` / `[2048]` |
| Mask | int8 `[2048,2048]`, recorded complete-content hash `d016ca7be6be829fd102f03317c3eed915d3df3d3fd54279a8858caea2dd57de` |
| Scalars / paging | sparse 3, scale `0.08838834764831845`, API block size 128, `block_table=null` |
| Request | one fresh request, computed tokens 0, scheduled/prompt tokens 2048, CPU positions `0..2047` |
| Actual calls | 28 original calls once each, all returned, captured inputs unchanged; repeated request also 28 |
| Native observation | `native=null`; original Native call count explicitly unobserved; no key, symbol, registered binary or final POD |

The internal request joins agree between the public mapping, Worker forward,
each call and scheduler. Scheduler cache block size is 16; that is not the API
block size and does not make this a paged decode. The census observes no FA API.
All calls fail the old BNSD validator before its mask-value verification;
the rejection text contains `TND_PAGE16_IS_SEPARATE`, but page16 is not an
observation about these calls. The recorded mask hash and causal scheduler
metadata do not replace the new validator's actual mask-value check.

The call input has exactly 13 supplied keys. It has no `pre_tokens`,
`next_tokens`, `inner_precise`, optional-feature inventory, QKV strides,
storage offsets, storage identity or device format. Do not turn omitted Python
arguments into observed Native values. The scheduler separately discloses
`device_input_bytes_equality=NOT_ESTABLISHED_BY_CPU_MIRROR`.
The final cleanup receipt has zero hooks/wrappers/leases/inflight while still
reserving its registered ticket; the later provider state is unregistered and
unreserved. This is a zero-F Python census, not a Native normal certificate.

There are two different FIA Native branches to distinguish before patching:

* The retained c77 `RouteToFia` returns false when `CheckSpecConditions`
  recognizes TND, Dq=Dk=Dv<=256, sparse 0/3/4, no split rope or output quant,
  and the accepted MHA/GQA precision options. For nonpaged D128/GQA sparse3
  **with final `inner_precise=0`**, the subsequent public host route is
  `TilingProcess4SplitFuse`/`FAInferTiling`.
  This is a source prediction for that argument set, not an observed stock key.
* The old Fault28 policy patches `FiaTilingNonQuant::CalcMBaseSize` in
  `attention/common/op_host/arch22/fia_tiling_nonquant.cpp`. It cannot affect
  a call which entered SplitFuse. Its current BNSD guard also rejects TND.

Therefore the minimal next native preparation is a separate **28 TND prefill
normal** scope, with actual stock launch/registered image capture. If selecting
the unchanged arch22 NonQuant route is required, that route preference must be
an explicit normal-only build and pass the complete stock/private bridge first.
The Q1/page16 `paged33_arch22_bridge.h` matcher is not this matcher. No prefill
normal bridge, actual stock key or M128 legality is claimed by source inspection.

## Exact arch22 M and causal-coordinate findings

Source references below are relative to the supplied
`tnd33-arch22-normal-build-a1/build/run/source`, not to an SDK inferred by name.
The seven-file source packet SHA256 is
`0f5b327f8572a447848ab857907451a42e532c1c5341f6599656dbd29916a677`.
All seven text payloads were rehashed and matched their recorded file SHA256.
The two-file addition has SHA256
`ca5f82def7dd249c5f68350e86a647058002e01662b253c6333c3bab79896443`;
its GQA and mask-helper payloads also match their recorded SHA256. These source
files are not committed here. The host-route sources are the separately
retained c77-derived build material; their route is not a stock 8.5.1 receipt.

| Source | Decisive code |
| --- | --- |
| `attention/common/op_host/arch22/fia_tiling_nonquant.cpp` | `CalcInnerSize` 253–297 defaults TND S2=512; `CalcMBaseSize` 300–321 defaults M512 and chooses BRC iff M<=128; `CreateSplitInput` 339 sets `isS1G`; `Split` 411–444 recomputes native arrays/FD/workspace dependencies. |
| `attention/common/op_kernel/fia_public_define.h` | `GetGS1Idx` 252–261 maps TND `s1=gS1Idx/gSize`, `g=gS1Idx%gSize`; BNSD uses division/modulo by `qSeqSize`. |
| `attention/common/op_kernel/arch22/fia_kernel_nonquant.h` | 606 binds `gS1Idx=gS1Cur*mBaseSize`; 612–619 derives actual M/S2 tails; `CalcCurS2StartEndWithSparse` 993–1047 derives causal S2 bounds in the correct S1G coordinates. |
| `attention/common/op_host/split_core.cpp` | 129–147 uses the same S1G coordinate conversion before allocating S2 ranges. |
| `attention/fused_infer_attention_score/op_kernel/fused_infer_attention_score_v3.cpp` | 915–918: source entry `103000000030000003` instantiates `FiaKernelNonQuant`, `FiaBlockCubeNonQuantGqa`, TND Q/KV, no PA, no FD, no BRC. No `103000000030400003` TND BRC entry is declared. |
| `attention/common/op_kernel/arch22/fia_block_vec_nonquant.h` | 383–423 splits M between AIVs and UB pieces; 575–599 supplies TND mask layout `SG`, accepting bool-compatible int8/uint8; 647–678 owns the actual online softmax recurrence. |
| `attention/common/op_kernel/arch22/fia_block_cube_nonquant_gqa.h` | 557–559 selects the real TND Q address calculator; 968/1098 use `Axis(info.actMBaseSize)` in both matmuls; 976/1026 split QK work into M128 blocks; 1094/1114 split PV work into M128 blocks. |
| `attention/common/op_kernel/vector_common.h` | 900–946 copies SG masks for query indices `floor(gs1/G)`, then repeats each query's mask across G; 967–982 uses the same SG coordinates for mask skipping; 1076–1100 applies the byte mask with the actual row/column sizes. |

The existing `fault28_causal_tile.h::InnerWidth` cannot be reused unchanged.
It assumes flattened G,S1 and uses `gS1Start % s1`. For TND G=2, M128 covers
64 query positions. At `gS1Start=2048`, those queries are 1024..1087; the old
helper would cap K at 128 instead of the required 1088. This loses valid
causal contributions, so merely deleting the BNSD static assertion is wrong.

A layout-specific version of the same causal-tail operation must derive
`last_query=(gS1Start+actual_M-1)/G`, cap at `last_query+1` for the equal-length
causal scope, then preserve the existing 32-element alignment and actual S2
tail. It must agree with both native sparse scheduling functions. The original
mask still filters intra-tile future positions. A CPU enumeration of 20,480
query ranges over M={32,64,128,256,512}, S2={M,512}, G=2/S1=S2=2048 proved
contiguous once-only logical coverage for this coordinate formula. That is
address algebra, not device-template legality.

| Proposed M / coupled S2 | Query positions per M, G=2 | Softmax branch | Source fact, not qualification |
| --- | ---: | --- | --- |
| 512 / 512 | 256 | non-BRC | Existing TND source specialization; native TND default. |
| 256 / 256 | 128 | non-BRC | Same source specialization can be evaluated; host scheduling, mask and cube legality still need proof. It does not substitute for the requested M128 fault. |
| 128 / 128 | 64 | BRC | Requires a new TND BRC specialization, absent from the supplied dispatcher. This is the required next candidate for the existing coupled/replay semantics. |
| 64 / 64 and 32 / 32 | 32 and 16 | BRC | Additional candidates, not part of the minimal adaptation or accepted scope. |

The 2-KiB softmax buffer holds 512 FP32 entries per preload slot. M128 gives
64 rows/AIV and BRC width 8, exactly 512 entries. The GQA cube's M128 constants
are subdivision sizes, not a requirement for an outer M512; both matmuls use
the actual M. This supports one M128 block without changing Q/K/V addressing.
The SG mask helper handles head/middle/tail G repetition and uses the same
`floor(gs1/G)` coordinates. For the proposed M128/N128 case its 64-row AIV
piece uses 8192 mask bytes and 8192 FP32 QK elements, within the existing
32-KiB QK buffer. These are concrete source-level necessary checks; they do
not establish generated CANN template legality, alignment of actual input
storage, native core scheduling or numerical equivalence.

The minimal new specialization is the existing TND, FP16, nonpaged,
non-FD `FiaKernelNonQuant` + `FiaBlockCubeNonQuantGqa` + `FiaBlockVecNonQuant`
combination with `SOFTMAX_WITH_BRC=true`. The unextended native key arithmetic
would describe it as `103000000030400003`; that key is **not present** in the
supplied dispatcher and is not claimed as a measured launch. It needs a new
explicitly registered branch plus distinct causal/replay trait keys. Do not
reuse the existing BNSD trait keys or select a generic cube silently. Existing
33 binaries also did not compile these nonpaged prefill keys. Source entry
presence is not a compiled artifact.

## Replay must stay at the actual denominator recurrence

Keep the existing semantic: on the second in-core S2 block, double the old
denominator mass before its normal max-rescaling; preserve the maximum,
QK, numerator/PV, output conversion and writes.

The concrete insertion point is
`FiaBlockVecNonQuant::SoftmaxFlashV2Compute`, after lines 667–670 select
`inSumTensor` from the previous preload slot, before lines 671–678 invoke
`SoftmaxFlashV2`. Retain `!info.isFirstSInnerLoop && info.s2Idx == 1` and the
separate replay template trait. For BRC, the affected count remains
`dealRowCount * brcbNum`; the non-BRC buffer has a different representation
and must not receive that multiplication count.

Baseline and replay must use the same newly proved M128/S2=128 TND causal
template; only the replay trait/key differs. For rows with at most 128 visible
keys there is no second valid S2 block and no replay. For later rows, the
scalar oracle is `numerator / (denominator + first_block_mass)`, not constant
output scaling. It must be checked with changing maxima and every causal row.

`numOfFdHead == 0` and complete per-query S2 ownership are mandatory. The
actual kernel sets `isFirstSInnerLoop` at a core's `curS2Start`, and can set
`tndIsS2SplitCore` for partial head/tail ranges. A split reduction cannot inherit
the old second-block proof. Run the actual `SplitCore` CPU implementation for
each candidate/core budget and decode all resulting arrays; reject FD or
partial ownership instead of clearing a flag or using Fault33's S2=256 path.

Preserve the existing strategy-arm distinction, translated to proved TND
templates rather than their old BNSD keys:

| Mechanism | Baseline/control | Fault | OFF |
| --- | --- | --- | --- |
| `28.fia.legal_tiling` | coupled M512/S2=512, causal-tail trait | coupled M128/S2=128, same causal semantics | ordinary native host selection, no causal/replay trait |
| `28.fia.denominator_replay` | coupled M128/S2=128, no replay | same M128/S2=128 with the single denominator replay | ordinary native host selection, no causal/replay trait |

The current plan expects the ordinary native geometry to be M512/S2=512;
validate that receipt rather than forcing OFF to those values. A legal tiling
result is not a denominator-replay result, and replay's M128 control cannot be
replaced by M512 while retaining its existing causal effect definition.

## Files and build scope for the minimal implementation

Keep the existing BNSD builders/artifacts intact. Add separate TND policy and
driver identities with the same three mechanism IDs and separate concrete
artifact fingerprints:

1. **Normal preparation:** add `attention28_tnd.py` beside
   `attention28_prefill.py` and a TND driver beside `attention_native.py`.
   Keep the original bound call arguments and return value. The new matcher
   first records the missing physical metadata, defaults versus supplied
   scalars, complete optional inventory and actual causal mask. Use a narrow
   owner/PID/TID/request-scoped TND2048
   prefill matcher and host capture, fresh public serializer compilation,
   actual loader/API-image binding and final argument observer. If a route
   preference is needed, record it separately; keep M/S2 and recurrence
   unchanged. Do not reuse a 33 certificate, key allowlist or vendor.
2. **FIA legal:** add a separate TND source policy under
   `fault_injection/native_adapters/attention28/fia/`, rather than expanding
   `fault28_fia_policy.h`'s BNSD accepted set. Derive selected M and coupled S2
   at `FiaTilingNonQuant::CalcInnerSize/CalcMBaseSize`, before the original
   `Split`, `FillTiling`, core/schedule/workspace and key stages. Add the
   layout-correct causal-tail trait/helper and a TND M128/BRC dispatch branch
   only after its concrete cube/mask checks pass. The old BNSD
   `109000000000400000` key cannot label TND. New keys must be derived from
   the actual source traits, registered and independently observed.
   Register matching classes in
   `op_host/arch22/fused_infer_attention_score_tiling_v3.cpp`, compile the
   exact branches in `op_kernel/fused_infer_attention_score_v3.cpp`, and
   regenerate the public serializer from the actual build headers.
3. **FIA replay:** a separately built replay variant of that same TND healthy
   M128 template, with only the denominator operation described above. Full
   host and matching kernel rebuild is required, including all state-layout
   dependents. No historical `.o`, `lib_cache` or acceptance receipt is input.
4. **FA legal:** attach only to an observed real
   `npu_fusion_attention_v3`/FlashAttentionScore boundary. Its current typed
   plan and `FlashAttentionScoreTilingS1s2Bn2gs1` factor selector are BNSD-only
   and CANN9.1-bound. A real TND FA branch needs its own source/schema/template
   review and M/n_ratio dependency recomputation. Routing a FIA call through a
   replacement FA API, repacking QKV or renaming FIA does not complete FA.

The current CANN8.5.1 Worker cannot inherit the old 9.1 build interface proof.
Use the reviewed fresh-build dependency workflow with a new explicit TND28
variant, exact runtime SDK interface manifest and current source compatibility
receipt. Recompile the actual TND keys and all warm/non-target entries required
by that same natural Worker; enumerate them from observation, not old 33 keys.

Normal admission requires two stable actual requests, complete natural call
denominator, read-before Q/K/V/mask/scalars and immediate original API outputs,
actual registered ELF/function/key/POD binding, full model outputs, unchanged
non-target calls and zero owned resources followed by same-Worker health.
The existing full-output bridge comparison is not weakened. A route difference
or unproved M128 template remains an incomplete adaptation, not blanket
technical inapplicability. Public F remains closed until the new normal proof
and mechanism-specific live admission are implemented and independently reviewed.

## Remaining bounded evidence

The actual source71 receipt, GQA cube and SG helper have now been reviewed.
The next missing runtime evidence is one owner-scoped **stock prefill**
registered-image/function/key/launch join plus final Native argument/POD
capture, bound to the same real TND call. It must include the omitted precision
and pre/next scalars, optional nullness, QKV physical metadata and current
runtime SDK identity. The zero-F Python census supplies none of these Native
fields. None is inferred from the model name, API, shape or the 33 decode run.

The concrete source-level tests before a build are: all actual SplitCore
arrays give complete unsplit ownership at the real core budget; TND S1G
indexing and SG mask replication including AIV/UB boundaries; M128 BRC and
M256 non-BRC buffer counts; unchanged max/numerator with replay applied once
at the specified denominator recurrence; and rejection of PA, padding,
unequal sequence lengths, quant, rope, FD, foreign owner and mismatched POD.
Compile the actual new M128 TND/BRC specialization in root's isolated build;
passing a Python geometry model cannot replace that compilation.

If stock actually uses SplitFuse and a normal arch22 candidate cannot pass the
unchanged full-output bridge, the next implementation must operate in that
actual SplitFuse prefill recurrence/tiling branch. The first bounded source
request for that branch is
`attention/fused_infer_attention_score/op_host/flash_attention_infer_tiling.h`
and `op_kernel/flash_attention_interface.cpp` plus its directly named
prefill recurrence header and `fia_kernel_common.hpp`. Its public
`FAInferTilingData` must be freshly compiled; arch22 POD offsets cannot parse
that payload. The page16 decode adapter is not a substitute. The
arch22 source work here remains useful but does not authorize selecting that
different normal algorithm silently.

## CPU review performed

Rehashed both source packets and all nine source texts, and rehashed the
uncompressed source71 JSON. Checked all 56 actual call rows and exact repeated
input captures for all 28 callsites. A stdlib-only enumeration checked 20,480
causal query ranges and 32,552 SG head/middle/tail replication cases, including
odd starts and AIV/UB-sized pieces, plus the M128 buffer-size calculations and
the old-helper counterexample. No Native code was compiled or executed. This
review grants no C1/C2/C3/E5 or numerical normal-bridge qualification.
