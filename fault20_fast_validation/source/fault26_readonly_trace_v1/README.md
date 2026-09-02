# Fault26 read-only path tracer v1

This component is the first, read-only phase of Fault26. It does not implement a
workspace underrun, an out-of-bounds write, or a buffer stomp.

## Purpose

The tracer records enough metadata to answer, without reading tensor contents:

```text
FIA workspace pointer / requested size
FIA dynamic Run tensor addresses
ReduceSum dynamic Run tensor addresses
residual-consumer candidate addresses
API order, stream, cache and timing
exact address reuse, range overlap and slot-proof status
```

It deliberately does not claim that an observed nearby or reused address was
corrupted. That requires a later lossless gate and a controlled proxy experiment.

## Safety boundary

The runtime is enabled only when both variables are present:

```bash
export VLLM_SERVER_DEV_MODE=1
export FAULT26_TRACE_ENABLED=1
```

The build rejects any runtime that references `aclrtMalloc`, `aclrtMemcpy`, or
`aclrtMemset` families. The tracer never calls `aclOpExecutor::GetOpExecCache()`
and never guesses the executor's private layout. It logs metadata and addresses,
not tensor values.

## Build

Build inside the exact inference image:

```bash
export FAULT26_TRACE_OUTPUT_DIR=/fault20/runtime/fault26_readonly_trace_v1
export FAULT26_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT26_REAL_LIBOPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
export FAULT26_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/

bash /fault20/source/fault26_readonly_trace_v1/build.sh
```

The build first verifies that the real `libopapi.so` exports
`aclnnFusedInferAttentionScoreV3` and `aclnnReduceSum`. If either symbol is absent,
it stops instead of advertising an unsupported API through the proxy.

## Phase 1: runtime-only API inventory

Start a new isolated service process. Do not attach this library to an existing
worker and do not co-load the Fault20/39 numeric injection runtime.

```bash
export VLLM_SERVER_DEV_MODE=1
export FAULT26_TRACE_ENABLED=1
export FAULT26_TRACE_ALL_APIS=1
export FAULT26_TRACE_CAPTURE_TENSORS=0
export FAULT26_TRACE_MAX_EVENTS=200000
export FAULT26_TRACE_LOG_PATH=/fault20/runtime/fault26_readonly_trace_v1/inventory.jsonl
export LD_PRELOAD=/fault20/runtime/fault26_readonly_trace_v1/libfault26_readonly_trace_abi0.so
```

At this stage, keep the image's normal `libopapi.so`. Run one fixed baseline
request and confirm which attention, reduction and residual APIs actually occur.
`FAULT26_TRACE_ALL_APIS=1` is only for the bounded inventory request. Tensor
metadata capture is disabled here to keep the inventory low-overhead.

## Phase 2: targeted workspace-size correlation

Only after Phase 1 confirms both V3 FIA and external ReduceSum, start another new
process with the scoped proxy:

```bash
export VLLM_SERVER_DEV_MODE=1
export FAULT26_TRACE_ENABLED=1
export FAULT26_TRACE_ALL_APIS=0
export FAULT26_TRACE_CAPTURE_TENSORS=1
export FAULT26_TRACE_FILTER=FusedInferAttention,InferAttention,ReduceSum,Add,RmsNorm,Residual
export FAULT26_TRACE_LOG_PATH=/fault20/runtime/fault26_readonly_trace_v1/targeted.jsonl
export LD_LIBRARY_PATH=/fault20/runtime/fault26_readonly_trace_v1:$LD_LIBRARY_PATH
export LD_PRELOAD=/fault20/runtime/fault26_readonly_trace_v1/libfault26_readonly_trace_abi0.so
```

The proxy forwards the real FIA/ReduceSum execute calls and supplies their exact
workspace pointer and size to the Run tracer. It does not change any argument.

## Analysis

```bash
python3 /fault20/source/fault26_readonly_trace_v1/analyze_fault26_trace.py \
  /fault20/runtime/fault26_readonly_trace_v1/targeted.jsonl
```

The analysis reports:

- observed API counts;
- proven versus unproven compact Run mappings;
- calls with a known workspace size;
- workspace/tensor interval overlaps;
- FIA-to-victim and ReduceSum-to-consumer address reuse candidates.

An address relation is only a candidate until producer, storage and consumer
semantics are proven with a typed adapter.

## Stop conditions

Stop this phase if any of the following occurs:

- ABI or expected CANN library path check fails;
- the real FIA/ReduceSum symbol is missing;
- baseline response/token differs from the service without the tracer;
- API name, slot, rank, stream or call index is ambiguous;
- metadata capture overflows or compact Run mapping is unproven;
- worker health, HCCL/CANN status or recovery is abnormal.

No stomp mode should be added until a typed victim mapping and an exact lossless
write-back gate have both passed in a separate experimental process.

## Completed A2 gates (2026-08-17)

The current Qwen3-32B dense path did not expose an independent ReduceSum/Add
consumer. The actual residual consumer was proven to be
`aclnnAddRmsNormBias`. A typed Python-to-L4 correlation passed with 1,024 Python
calls, 1,024 GWS calls and 1,024 execute calls, with exact tensor metadata and
address agreement.

The separate v11 write gate then completed one bounded lossless operation on
device 0, BF16 ND contiguous `x1` with shape `[1,5120]` and 10,240 bytes:

```text
D2H -> identical H2D to the same legal storage -> D2H readback -> real execute
```

All runtime statuses were zero, the before/after FNV64 hashes were equal, the
gate automatically disarmed after one hit, and the disabled/hit/post responses
had exactly equal text and token IDs. See:

```text
fault20_fast_validation/reports/fault26_reproduction_progress_20260817.md
```

The later v12 gate moved the lossless target to `x2`, which is the proven
cross-layer residual-storage slot. v13 added a bounded `xor_bf16_sign` mode with
post-execute synchronization and exact restoration. The staged sweep stopped at
256 bytes after 128 BF16 elements produced a 32-token response divergence; the
following disarmed request exactly matched the baseline token sequence.

The v13 controller syntax is:

```bash
fault26_write_gate_ctl arm-lossless x2 0 1 5120 1 10240
fault26_write_gate_ctl arm-xor-sign x2 0 1 5120 1 10240 0 256
```

Numeric mode requires exactly one hit and a BF16-aligned interval wholly inside
the proven storage. It automatically disarms on claim. The runtime verifies the
injected value by readback, lets the real `aclnnAddRmsNormBias` execute, waits for
that stream, restores the original bytes, and verifies the restored hash before
returning.

This is an in-range residual-buffer corruption proxy. No out-of-bounds write has
been run. In the current allocator layout the FIA workspace was not adjacent to
residual storage, so the result must not be named
`FIA_REAL_OUT_OF_BOUNDS_WRITE_PROVEN` or
`FIA_TRIGGERED_VICTIM_BUFFER_CORRUPTION_PROVEN`.

The exact completed classifications are:

```text
FAULT26_RESIDUAL_BUFFER_STOMP_PROXY_E2E_PASS
RESIDUAL_VICTIM_STORAGE_PROVEN
RESIDUAL_CORRUPTION_PROPAGATION_PROVEN
MODEL_OUTPUT_DIVERGENCE_REPRODUCED
RECOVERY_PASS
FIA_REAL_OOB_NOT_REPRODUCED
```

This is a semantic reproduction of victim corruption and residual propagation,
not a physical reproduction of the historical FIA workspace OOB root cause.

## Unified QMM-triggered residual stomp (v15, 2026-08-17)

The v13 result above writes at the AddRms consumer boundary. It proves the
victim and propagation effect, but it does not prove that a real QMM call
triggered the write. v15 closes that distinction with one root-cause model:

```text
faulty API              aclnnQuantMatmulWeightNz
injection primitive     QMM_TRIGGERED_RESIDUAL_STOMP_PROXY
fault-type branches     none
attention / MLP         observation context only
```

The read-only v14c audit correlated the real L4 execution thread, not Python
wall-clock order. On each TP rank it observed 762 QMM Runs and mapped all
762/762 to a following AddRms consumer. The distribution was 254 consumers
with one preceding QMM, 254 with two, and four with none. The residual chain
matched on 508/511 transitions; the three mismatches were forward boundaries
and are explicit injection rejections. Typed-to-L4 agreement was exact for all
512 `x2`, all 512 `output.x`, and all 512 executor FIFO entries per rank.

v15 maintains the following bounded state per worker/device:

```text
AddRms GWS captures x2 and output.x metadata into an executor FIFO
  -> completed AddRms output.x becomes current residual state
  -> real aclnnQuantMatmulWeightNz OpExecCache::Run completes
  -> peek the next queued AddRms record
  -> require current residual address == next AddRms.x2 address
  -> apply a bounded in-range write to that proven residual storage
  -> require the exact next AddRms sequence/address to consume it
  -> synchronize, restore the original bytes, and verify the restored hash
```

No activation address is persisted across an unverified boundary. A missing
next-consumer record, shape/metadata mismatch, forward-boundary address change,
unexpected consumer sequence, copy/sync failure, or pending prior hit causes a
skip or immediate restoration. The full `api_info` suffix, cache pointer, and
Attention/MLP location are logged as context but are not stable root-cause
selectors. Matching uses the QMM API prefix plus the validated BF16 ND
`[1,5120]` residual relation.

Build and controller examples:

```bash
bash build_fault26_qmm_residual_stomp.sh

fault26_qmm_residual_stomp_ctl status
fault26_qmm_residual_stomp_ctl arm-lossless 0 1 1 5120 1 10240
fault26_qmm_residual_stomp_ctl arm-xor-sign 0 1 1 5120 1 10240 0 256
fault26_qmm_residual_stomp_ctl disarm
```

The arm commands are accepted only when both `VLLM_SERVER_DEV_MODE=1` and
`FAULT26_QMM_STOMP_ALLOWED=1` are present. Numeric mode is fixed to one hit and
automatically disarms when claimed.

The staged v15 E2E produced these runtime results:

```text
lossless generation     1
QMM hit / restore       1 / 1
consumer matched        true
hash before/injected/   0xa8eb8378763a667e (all equal)
restored
token IDs               [198,32313,11,279] (exact baseline)

numeric generation      2
mode                     xor_bf16_sign
range / changed          256 bytes / 128 BF16 elements
QMM hit / restore       1 / 1
consumer matched        true
hash before              0x5b7972f684d912fb
hash injected            0xc62d55a7e6f8effb
hash restored            0x5b7972f684d912fb
all runtime statuses     0
HTTP / worker health     200 / healthy
```

The generation-2 32-token output happened to remain token-exact to baseline.
Therefore the accurate classifications are:

```text
FAULT26_QMM_TO_NEXT_RESIDUAL_RELATION_READONLY_PASS
FAULT26_QMM_TRIGGERED_RESIDUAL_STOMP_RUNTIME_PASS
QMM_TRIGGERED_RESIDUAL_NUMERIC_CHANGE_PROVEN
NEXT_ADDRMS_CONSUMER_MATCH_PROVEN
RESIDUAL_STORAGE_RESTORE_PASS
TOKEN_DIVERGENCE_NOT_OBSERVED_THIS_HIT
FIA_REAL_OOB_NOT_REPRODUCED
```

This proves the unified developer-mode injection primitive and its lifecycle.
It does not prove a physical QMM/FIA out-of-bounds defect, and it does not claim
model-output divergence for this particular v15 hit.

## QMM/residual declared-span audit (v16c, 2026-08-18)

v15's QMM-to-consumer relation is a lifecycle proof, not an allocation-layout
proof. v16c performs the separate read-only interval audit. On CANN 8.5.1 the
QMM typed front end is obtained from split `libopapi_nn.so` by a specific
library handle, so the audit container uses a same-SONAME metadata proxy for
that split library and preserves the real implementation as
`libopapi_nn_real.so`.

Per TP rank, the full L4 FIFO audit observed:

```text
AddRms GWS / execute       1152 / 1152
QMM Runs                   1719
mapped to next AddRms      1719 / 1719
safe residual chain        1707
forward-boundary rejects   12
```

For all 1,707 safe QMM output samples per rank:

```text
relation                   residual_end < qmm_output_begin
tail adjacency             0
reverse adjacency          0
overlap                    0
absolute gap min/median    512 / 16384 bytes
absolute gap max           130154496 bytes
```

The observed declared output spans therefore do not support a forward QMM
output-tail overrun into residual. `aclTensor.storage_bytes` is not an allocator
block span, so allocator padding and kernel-internal temporary buffers remain
unknown. The result is classified as:

```text
FAULT26_QMM_RESIDUAL_DECLARED_INTERVAL_AUDIT_PASS
QMM_OUTPUT_RESIDUAL_NOT_ADJACENT_IN_DECLARED_SPANS
```

Use `analyze_fault26_qmm_residual_adjacency.py` for the conservative live-queue
view and `analyze_fault26_qmm_residual_adjacency_offline.py` for full L4 FIFO
coverage. Neither analyzer reads tensor contents or modifies runtime state.

## Residual writer timeline audit (v18, 2026-08-18)

v18 adds an explicitly armed, developer-mode-only P0-P4 audit. Each checkpoint
synchronizes the target stream and reads the complete contiguous residual
storage with D2H only. It records hashes and bounded difference statistics,
never tensor contents.

Per TP rank, the fixed request produced:

```text
P0 / P1 / P2 / P3 / P4        512 / 766 / 766 / 508 / 508
complete P0-P4 generations     508
complete live QMM windows      760 / 760
read failures                  0
unexpected live changes        0
```

Four terminal/boundary generations per rank had no later AddRms consumer.
Changes after those lifetimes ended are reported as post-lifetime changes
consistent with address reuse, not as a writer fault. The same-Run visible QMM
declared spans had no exact lower/upper
adjacency and no overlap; minimum lower/upper gaps were 154,624 and 512 bytes.
Allocator blocks and kernel-internal temporaries remain unknown.

The baseline, armed read-only request, and recovery response were token exact:
`[198,32313,11,279]`. The classification is:

```text
FAULT26_RESIDUAL_WRITER_TIMELINE_AUDIT_PASS
FAULT26_RESIDUAL_WRITER_TIMELINE_NO_UNEXPECTED_CHANGE
```

The scope of that classification is intentionally limited to the current
hardware/CANN/model configuration and fixed request. QMM Runs are individually
bracketed by P1/P2; other visible L4 work is covered only in grouped intervals
between adjacent P0-P4 hashes. It does not establish that individual FIA or
other fused operators, wrong-device-pointer paths, allocator padding, or
kernel-internal temporaries cannot corrupt ReduceSum/residual storage. It also
does not prove that a historical defect is fixed.

ReduceSum is not a required victim for this model. Request-time external
`aclnnReduceSum` was absent, while the real residual lineage was proven as:

```text
previous AddRmsNormBias.output.x
  -> live residual
  -> next AddRmsNormBias.input.x2
```

The writer-reproduction stage therefore selected a real writer-capable operator
inside that lifetime instead of continuing to search for a naturally corrupting
operator in the healthy request. FIA passed a lossless output-address
substitution gate and then passed a bounded wrong-destination experiment:

```text
FIA attentionOut binding -> live residual address
real FIA kernel          -> changes residual
original FIA output      -> restored from the kernel-written victim span
next AddRms x2           -> consumes the same corrupted residual
consumer completion      -> original victim bytes restored
```

The fixed request produced token-level response divergence and exact recovery.
The validated name is
`FIA_WRONG_DESTINATION_RESIDUAL_STOMP_PROXY_PROVEN`, not
`FIA_TRIGGERED_REDUCESUM_STOMP_PROXY` and not `REAL_OOB_STOMP`. See the Phase C
and Phase D reports under `fault20_fast_validation/reports/`.

### Follow-up visible-API coverage sweep

The later short, 353-token prefill, and two-request concurrent sessions all
stopped at the ReduceSum provenance gate: external `aclnnReduceSum` occurred
zero times on both ranks during every armed request. The actual legal producer
remained the previous `aclnnAddRmsNormBias output.x`.

Re-analysis also showed that FIA occurred inside grouped `P2 -> P1` residual
live intervals (252, 252, and 882 entries per rank across the three sessions),
with zero persistent changes at the next synchronized hash. This is not an
individual before/after proof for each FIA and cannot detect a modification
that is introduced and restored within one grouped interval.

Use `analyze_fault26_residual_lifetime_coverage.py` for this analysis. The
complete result and evidence boundary are in:

```text
fault20_fast_validation/reports/
  fault26_residual_lifetime_visible_api_audit_20260818.md
```

Use `analyze_fault26_residual_writer_audit.py` on the inventory JSONL. The
native audit is disabled by default and requires an explicit
`fault26_arm_residual_writer_audit` collective RPC while
`VLLM_SERVER_DEV_MODE=1` and `FAULT26_WRITER_AUDIT_ALLOWED=1` are both set.

## QMM allocation-boundary underflow proxy (v2, 2026-08-18)

The read-only live allocator audit found one repeatable QMM candidate per rank
whose active output allocation starts exactly where the live residual
allocation ends:

```text
residual declared span   [0x12c1c235e200, 0x12c1c2377200)
residual allocation      [0x12c1c235e200, 0x12c1c2377400)
QMM output allocation    [0x12c1c2377400, 0x12c1c23f4600)
allocator padding        512 bytes
```

After repeating that proof inside the same Phase 27 worker processes, one
developer-mode-only experiment shifted the rank-0 QMM output base backward by
1,024 bytes. The real QMM crossed its proven allocation boundary and overlapped
512 bytes of the live residual. It changed 510 residual bytes, after which the
runtime restored the complete QMM result to its normal output address. The real
next AddRms consumed the corrupted residual and completed successfully, then the
runtime restored and readback-verified the original residual.

All ACL statuses were zero, both output and residual restoration were
byte-exact, rank 1 did not hit, and post-fault health was 200. The fixed request
happened to remain token-exact to baseline. The accepted classification is:

```text
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN
TOKEN_DIVERGENCE_NOT_OBSERVED_THIS_HIT
```

This is stronger than the Phase D FIA wrong-destination proxy because it crosses
a real, live allocator boundary. It is still a controlled address-fault proxy,
not evidence that the healthy QMM kernel naturally performs an OOB write and not
a reproduction of the historical FIA workspace root cause.

Build the write-capable runtime separately from the read-only tracer:

```bash
export FAULT26_TRACE_OUTPUT_DIR=/fault39/runtime/fault26_qmm_underflow_v2
bash build.sh
bash build_fault26_qmm_output_underflow.sh
```

The container must start with `VLLM_SERVER_DEV_MODE=1`,
`FAULT26_QMM_UNDERFLOW_ALLOWED=1`, and
`FAULT26_QMM_UNDERFLOW_ENABLED=0`. Arming is accepted only through the worker
extension with exact device, output/residual addresses, sizes, proven adjacent
allocation boundaries, and a shift in `[1,4096]`. The v2 native runtime permits
at most one claim per worker process. It cannot be co-loaded with another
write-capable Fault26 gate.

See the complete proof and evidence boundary in:

```text
fault20_fast_validation/reports/
  fault26_writer_allocation_topology_and_qmm_underflow_20260818.md
```

## QMM allocation-boundary bounded multi-hit proxy (v3, 2026-08-18)

The v3 runtime extends the exact-address Phase 27 primitive with two safety and
control properties:

```text
complete crossed-prefix backup
  -> allocator padding restored and verified immediately after QMM
  -> normal QMM output restored and verified
  -> exact next AddRms consumes only the declared residual corruption
  -> residual restored and verified after that consumer

bounded max_hits in [1,16]
  -> default remains 1
  -> no new claim while a prior hit is pending recovery
  -> gate auto-closes at the configured limit
```

Phase 28 repeated the live allocator proof in the new worker processes and then
completed hit numbers 1, 2, and 3 on rank 0. Each exact prefill allocation pair
occurred once per fixed request, so this is a three-request cumulative
multi-hit proof, not three hits in one inference. Every hit changed 510 declared
residual bytes; every ACL status was zero; allocator padding, normal QMM output,
and residual storage were byte-exact after their respective recovery points.
Rank 1 did not hit.

Phase 29 then added byte-size filters to the read-only live allocator probe. It
proved that the decode pair below was active, exactly adjacent, and repeated at
filtered occurrences 3, 5, and 7 in one fixed request:

```text
residual [1,5120]       0x12c1c231aa00, 10240 requested / 10752 actual
QMM out [1,25600]       0x12c1c231d400, 51200 requested / 51712 actual
common block boundary   0x12c1c231d400
padding / overlap       512 / 512 bytes for a 1024-byte backward shift
```

A single rank-0 request then completed three real QMM hits at run indices
898, 900, and 902. They changed 510, 434, and 282 residual bytes. All status
fields were zero; each exact next AddRms consumed the corrupted residual; and
padding, normal output, and residual recovery were byte-exact for all hits.
Rank 1 again did not hit.

All baseline, fault, and recovery responses were identical:

```text
" Deterministic testing ensures consistent and reliable results"
[34907,4532,7497,25351,12966,323,14720,3059]
```

The accepted classification is:

```text
QMM_OUTPUT_BASE_UNDERFLOW_BOUNDED_MULTI_HIT_PROVEN
CROSS_REQUEST_MULTI_HIT_PROVEN
SINGLE_REQUEST_MULTI_HIT_PROVEN
TOKEN_DIVERGENCE_NOT_OBSERVED_FOR_SINGLE_REQUEST_THREE_HIT
```

This remains a controlled output-base fault proxy, not proof that the healthy
QMM implementation naturally performs an OOB write or proof of the historical
physical root cause.

See:

```text
fault20_fast_validation/reports/
  fault26_qmm_underflow_bounded_multihit_20260818.md
```

## QMM allocation-boundary all-eligible-hit proxy (v4, 2026-08-18)

Phase 30 extends the v3 single-address control to a fail-closed list of at most
eight exact allocator-proven address pairs. The worker RPC validates every
pair, and the native runtime independently reparses and matches the list for
each candidate. `max_hits` now supports a hard limit up to 4096; its default
remains one. `all_hits=true` is explicit and visible in every runtime record.

The filtered live allocator probe repeated its proof inside the Phase 30
workers. Three decode pairs were active and exactly adjacent; a fourth
same-shape but non-adjacent combination was excluded. One rank-0 request then
hit every occurrence of the three eligible pairs:

```text
eligible pair 1       7 hits
eligible pair 2       7 hits
eligible pair 3     217 hits
total               231 hits
failed/aborted        0
hard limit reached    false
```

All 231 real QMM Runs, exact next AddRms consumers, and padding/output/residual
recoveries passed. The all-hit response diverged at completion token 7:

```text
baseline
" Deterministic testing ensures consistent and reliable results"
[34907,4532,7497,25351,12966,323,14720,3059]

all-eligible-hit
" Deterministic testing ensures consistent and predictable outcomes"
[34907,4532,7497,25351,12966,323,44107,19554]
```

After explicit disarm, the next response returned exactly to baseline and the
service remained healthy. The accepted classification is:

```text
ALL_ELIGIBLE_QMM_HITS_PROVEN
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN x231
MODEL_RESPONSE_DIVERGENCE_PROVEN
POST_FAULT_EXACT_RECOVERY_PROVEN
```

`all eligible` means every call matching one of the live-proven adjacent pairs,
not every unexamined QMM address or shape. This remains a controlled operator-
fault proxy and does not claim that the healthy kernel naturally performs OOB.

See:

```text
fault20_fast_validation/reports/
  fault26_qmm_underflow_all_eligible_20260818.md
```

## Sustained all-hit dump validation, 2026-09-02

This validation records controlled Fault26 sustained all-hit behavior on the local A2 / Ascend 910B4 Qwen3-32B-w8a8-nopdmix environment. Raw dump and runtime evidence are intentionally kept outside Git under `/home/haoran/fault_evidence`.

Developer-mode msprobe control was added through vLLM `/collective_rpc`:

- `fault26_start_msprobe_dump(config_path=None)`
- `fault26_query_msprobe_dump()`
- `fault26_stop_msprobe_dump()`

The debugger is started after the worker model is loaded by calling `PrecisionDebugger.start(model=<worker model>)`. This avoids the invalid early-start path where `level=mix` fails because msprobe has not received a model object. This is a dump-control path only; it does not change the Fault26 Phase D arm ABI and does not add a Python dispatcher fault hook.

Evidence roots:

```text
/home/haoran/fault_evidence/fault26_sustained_20260902
fault20_fast_validation/evidence/fault26_sustained_dump_20260902
/home/haoran/fault_evidence/fault26_sustained_dump_20260902_v4
/home/haoran/fault_evidence/fault26_sustained_dump_20260902_v6_applied
```

Dump matrix:

| Case | Fault mode | Request | Dump status | Writer status | Response status |
| --- | --- | --- | --- | --- | --- |
| `nohit_dumpon` | disarmed | 1 token | `step0/rank0`, `step0/rank1`, 6 files | initialized only | text: `' The'` |
| `allhit1024_dumpon` | all-hit, `max_hits=1024` | 64-token request, manually stopped after dump/applied proof | `step0/rank0`, `step0/rank1`, 6 files | `applied=703`, `restored=703` | response text not used as evidence |
| `allhit4096_dumpon` | all-hit, `max_hits=4096` | 128-token request | `step0/rank0`, `step0/rank1`, 6 files | `applied=191`, `restored=191` | early EOS: text `' The '` |

The 1024 case proves that dump collection and actual Phase D writer application can coexist, but its response text is not used because the run was manually stopped after dump and writer evidence had already landed. The 4096 case reproduces the sustained all-hit early-EOS collapse while dump is enabled.

Boundary: this validates the controlled `aclnnFusedInferAttentionScoreV3` wrong-destination residual stomp proxy under developer mode. It does not claim that the healthy vendor kernel naturally performs an out-of-bounds write.
