# Fresh native FAI flash-decode repair

Status: source preparation and local CPU contracts only. The new SDK build,
actual platform preflight, stock/private bridge, native fault/OFF launches and
independent raw statistics remain required. C2=false; C3=false. No Agent4
container or NPU call was made. This document supersedes the six-key build
instructions in `attention33_worker_handoff.md` for the new attempt only.

The preserved failed attempt is
`/home/17-fault-c2c3-20261117/attention33-worker18-card2-a1/attention33-natural-baseline-a2/run/result.json`,
SHA256 `c2321ef8e5f05729ad82f5f87f2cebb52b5f385d2f46fdfe87e4afe4c8ea8b99`.
The standard Worker initialized after the dual-library repair, but its warm
request failed on missing native key `5100000000010200103`. No strategy session
or successful launch of that key occurred. The actual API was FIA V3; the new
library-ownership check includes both V3 symbols as well as the existing pair.

## Actual source boundary

All four patched sources are pinned to pristine commit
`c77c0829f1f0f0b8f882db1d9b051a758b605472`; their exact before hashes are checked
by `apply_tnd33_fai_fd_policy.py`. The unchanged original function
`ComputeSeqLenAndDecodeFlagsForFAI` has SHA256
`742a14b0a230a8f09d4ceb3407ab6ea58e63fc8ad8b4bcba2e64298222f43eb2`.
It derives FD from the actual lengths, number of KV heads, platform AIC count,
page/mask/precision/sink conditions. The long-sequence arm requires
`KV >= aic * 512`; the short-sequence arm requires `batch * kvHeads <= .4*aic`
and `KV >= 1024`. Thus B1/Hkv8/Q1/KV1025 cannot naturally select FD at 16 cores.
It implies at least 20 cores in this scope, but **does not establish an actual
card2 core count**. That remains unknown until root's actual preflight.

The supported experimental range is aic20..25: the original public arrays have
26 elements and `InitSplitInfoArrays` writes through index `aic` inclusive.
This range is a guard, not a measured platform fact. Native `needCoreNum`, not
the platform count, determines active launch blocks. Schedule mode remains 1.
For example, a CPU-only aic24/Q1/KV1025/native512 fixture needs 17 active blocks;
it is not a Worker observation.

The original ConvertContext/Compute functions leave `FAInferContext.maxQSeqlen`
and `maxKvSeqlen` at their default zero. Actual task geometry reads the Q/KV
length lists. The prior policy's assumed max fields would prevent eligibility.
The new policy reads the actual B1 lists and records the untouched max fields,
actual aic, native FD/decoding/tiling-sink flags, key, blocks and workspace.
No routing flag, API tensor, length, mask, page table or cache writer is changed.

## Exact entry set and dependencies

New variant: `tnd_page16_fai_fd_s2_256`; new vendor:
`dcat_f33_fai_fd_transformer`. It has exactly eight unique entries:

| Role | Keys |
| --- | --- |
| Native nonpaged, non-FD healthy branches | 5000000000000200100, 5000000000000200103 |
| Native paged, non-FD healthy branches | 5000000000010200100, 5000000000010200103 |
| Native paged FD healthy branches | 5100000000010200100, 5100000000010200103 |
| Matched paged FD KV256 branches | 6000000000010200100, 6000000000010200103 |

The new aliases attach only to `_SPLITFUSE_TILING_FD`. Their exact template is
`SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor, half, half, float,
true, true, MASK_CAUSAL-or-NO_MASK, inputLayout::TND>`: paged=true and FD=true.
Only the new keys compile `MAX_KV_STACK_LEN=256`; all six healthy keys retain
512. The prior controlled non-FD `590...` entries are excluded. Source presence
alone grants no admission: new compiler metadata, ELF, registration, actual
runtime function binding and final public arguments must each agree.

The tile changes the actual host `GetKSBlockTile`, all four host workspace
slices, and external allocation; the original splitter and reduction planner
recompute their complete arrays. In `flash_attention_regular.h`, the same
device constant controls L1 V reservation, both MMAD initializers, KV loops and
tails, mask/page stacking, workspace offsets and staging. KV256 is divisible by
page16 and matrix128 and leaves at least a Q128*D128 output/update slice. The
original reduction consumes the recomputed core/split arrays and split LSE/O
workspace. Separate `decodingFlag`/`flash_attention_regular_decode.h` is not
selected by this strategy. No output arithmetic or tensor payload is patched.

The exact SDK CPU build compiles the original decision function and actual
patched `FAInferTiling::DoTiling`. Its 24,576 FD rows cover aic20..25,
KV1025..2048, mask0/1 and KV512/256, plus the four native non-FD healthy keys.
It derives flags through the original function; it never hand-sets FD=true.
An independent serial coverage/reduction validator checks all 18 public arrays
of length26, active blocks, complete nonoverlapping KV coverage, contribution
counts, split offsets, split LSE/O sizes and every workspace slice. The real
public serializer/sentinel check still binds all 52 declared fields. The full
matrix, summary, validator bytes, serializer and compile receipt are hashed into
the fresh completion and required by the live adapter. No old certificate is
reused. An SDK or matrix failure must stop the build, with the failure retained.

## Root-only fresh build handoff

Use a new source snapshot containing this commit and a new output path. The
existing exact8.5.1 SDK interface manifest may be reused only as a header-hash
input; no previous host, kernel, decoder or vendor object is an input.

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$R/source-NEW/experiments/c2c3_closure_20261117/build_tnd33_124.py" \
  --source "$R/source-NEW" --out "$R/tnd33-fai-fd-build-a1" \
  --interfaces "$R/attention-preflight-a4/cann851/attention-preflight/sdk_interfaces.json" \
  --strategy native-fd
```

The controller pins the locally inspected
`quay.io/ascend/vllm-ascend:v0.18.0` image, uses runc with no network/devices,
4 CPUs/16GiB, and a 7200-second CPU-build bound. The prior image receipt was
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`;
root must compare the new actual image identity. The existing exact8.5.1
registration compatibility removes only unsupported mc62/ascend950 registrations,
preserving 910B and tiling/kernel bytes. Build failure is not a reason to reuse
the six-key package or cross-admit CANN9.1 binaries.

Review fresh `build/run/evidence/completion.json`,
`fai_schema/compile_receipt.json`, `fai_schema/fai_fd_geometry_matrix.json`,
`binary_summary_registration.json`, new host/kernel ELF and metadata hashes.
The actual native FD branch and both new specializations must appear exactly
once in compiler and registration metadata. Completion must still report
`CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION`, `npu_executed=false`, and
`actual_worker_platform_observed=false`.

## Device gates after the build

Root first owns a short preflight that records the actual platform AIC count,
actual length lists, native FD/decoding flags and original key. A CPU matrix or
device name cannot supply that fact. Out-of-range platform/branch observations
stop this new strategy rather than forcing a branch. A standalone preflight
does not qualify the natural Worker.

Create new wheels with the FD adapter, validator and V3 ownership check; do not
mix old packages-a9 with the new vendor. Keep imports under `/installed` and
retain the exact8.5.1 observer receipt/header/library mounts. Use the existing
natural Qwen3-0.6B, page16, fixed1024-token prefill plus two-token decode flow.
The native vendor path is now
`/native/stage/vendors/dcat_f33_fai_fd_transformer`; public schema remains
`/native/evidence/fai_schema/fai_abi_schema.json`. Native warm/profile branches
are retained. No old result is edited or reclassified.

Proceed in order: same-input stock run, new private baseline and repeat, exact
stock/private complete-output bridge, then fault and explicit OFF/healthy-after.
The bridge includes hidden/logit/all28-attention outputs and every valid logical
KV element, permits only allocator-address normalization, and requires finite
exact payloads. Recheck actual stock Norm/private FIA-V3 symbol ownership after
standard Worker initialization. A successful library-only CPU preflight does
not establish successful no-device extension import or Worker execution.

Each measured call must independently bind the new registered ELF, actual
function/key, successful launch, full final public POD and host serialization,
then match actual API page capacities, sparse mask, native branch and complete
FD geometry. Native OFF uses `510...`; fault uses `600...`. Preserve full
representation comparisons and separately evaluate raw collector E5 using the
root's fixed tolerance and missing/null/NaN rules. C3 recovery and public
qualification remain separate gates; neither build success nor a selected key
alone is qualification.
