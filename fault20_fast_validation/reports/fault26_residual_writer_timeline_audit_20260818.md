# Fault26 residual writer timeline audit

Date: 2026-08-18

## Result

The synchronized P0-P4 audit passed on both TP ranks:

```text
FAULT26_RESIDUAL_WRITER_TIMELINE_AUDIT_PASS
FAULT26_RESIDUAL_WRITER_TIMELINE_NO_UNEXPECTED_CHANGE
```

For every consumer-confirmed residual generation observed in the fixed request:

```text
P0 legal AddRms output producer
  == P1 before each QMM
  == P2 after each real QMM
  == P3 at the next AddRms wrapper entry
  == P4 immediately before the real AddRms call
```

No live residual byte changed between checkpoints. This is a no-injection
baseline result; it does not reproduce the historical physical stomp.

The strongest valid statement is deliberately narrower:

> On this hardware, CANN build, model configuration, and fixed request, no
> unexpected modification of the audited residual storage persisted across
> any adjacent synchronized P0-P4 checkpoint interval.

QMM Runs are individually bracketed by P1/P2. Other visible L4 work is covered
only as grouped inter-checkpoint intervals; it is not individually attributed.

This result must not be generalized to mean that no model operator can corrupt
ReduceSum/residual storage, or that a historical defect has been fixed.

## Environment and controls

```text
model          Qwen3-32B-w8a8-nopdmix
CANN           8.5.1
vLLM           0.18.0
TP             2, logical devices 0 and 1
mode           eager, VLLM_SERVER_DEV_MODE=1
container      fault26-qwen3-32b-readonly-inventory-
               20260818_phase13_residual_writer_v18
port           18028
session        2026081802
```

The native audit is inert by default. It requires both
`FAULT26_WRITER_AUDIT_ALLOWED=1` at process start and an explicit developer-mode
worker RPC arm. Disarm completed immediately after the fixed request.

The only device-data operation in the audit path is synchronous
`ACL_MEMCPY_DEVICE_TO_HOST`, after `aclrtSynchronizeStream`. The build rejects
H2D, D2D, memset, async memcpy, and device allocation paths. No tensor content
is logged: only hashes, changed-byte counts, first/last differing offsets, and
declared addresses/sizes are recorded.

## Legal producer and checkpoint semantics

The legal producer for an audited residual generation is the previous
`aclnnAddRmsNormBias` `output.x`. Its declared storage is required to match the
later AddRms `input.x2` before that generation is classified as consumer
confirmed.

```text
P0  previous AddRms returned; stream synchronized; output.x sampled
P1  immediately before a real QMM OpExecCache::Run
P2  immediately after that real QMM Run; stream synchronized
P3  next AddRms execute-wrapper entry; stream synchronized
P4  immediately before calling the real AddRms implementation
```

P4 is a custom-opapi wrapper checkpoint. The first v17 attempt tried to infer
P4 from an internal `OpExecCache::Run`, but this AddRms path did not traverse
that hook. v18 therefore uses an explicit, typed wrapper callback and proves
P4 directly.

## Coverage

Per rank (`pid 218` and `pid 219`):

```text
P0 / P1 / P2 / P3 / P4        512 / 766 / 766 / 508 / 508
consumer-confirmed generations 508
complete P0-P4 generations     508
consumer-confirmed QMM windows 760
complete P1-P2 QMM windows     760 / 760
read failures                  0
device-write events            0
unexpected live changes        0
```

There were four terminal/boundary generations per rank with no later P3/P4
consumer. Five changes per rank were observed only after those generations had
no proven future consumer. They are reported separately as post-lifetime
changes consistent with address reuse and are not classified as an illegal
writer. Treating those changes as a stomp would incorrectly extend residual
lifetime across forward boundaries.

## Visible neighbors

For each of the 760 consumer-confirmed QMM windows per rank, the analyzer also
searched the same real QMM Run's visible tensor/workspace declared spans on
both sides of the residual:

```text
lower-neighbor observations    760
upper-neighbor observations    760
minimum lower gap              154,624 bytes
minimum upper gap              512 bytes
exact lower adjacency          0
exact upper adjacency          0
overlap                        0
```

Roles varied by Run (`x1`, `out`, and `x2_scale` appear among the closest
visible spans). This is not an allocator inventory. Caching-allocator block
boundaries, padding, and kernel-internal temporary buffers remain unknown.

## End-to-end result

The unarmed baseline, armed read-only audit request, and disarmed recovery
request all returned HTTP 200 with exactly:

```text
text       "\nOkay, the"
token IDs  [198, 32313, 11, 279]
```

The service remained healthy after disarm and was left running on port 18028.

## Accurate conclusion

### Evidence boundary

| Claim | Current evidence |
|---|---|
| This no-injection request contained a residual stomp | Not supported; no live P0-P4 change was observed |
| A declared QMM output tail directly overran into the residual | Not supported by the declared intervals |
| QMM was the historical corrupting writer | Not proven |
| A persistent residual change crossed a grouped FIA-containing interval | Not observed in the later shape sweep; individual FIA calls were not bracketed |
| ReduceSum produced the audited residual | Not proven; external ReduceSum was absent from the later request sessions |
| No operator in this model has a buffer-corruption risk | Cannot be concluded |
| The historical defect is fixed in the current CANN version | Cannot be concluded from this request |

The evidence supports:

```text
legal residual producer identified
consumer-confirmed residual lifetime identified
no spontaneous residual change in P0-P4 windows
no visible same-Run QMM declared-span adjacency
post-lifetime changes separated from live corruption
```

It does not support:

```text
historical physical stomp reproduced
QMM proven as an illegal writer
every FIA or L4 call individually bracketed
ReduceSum-to-residual lineage proven
wrong-device-pointer corruption excluded
all model buffer-corruption risks excluded
historical defect proven fixed
allocator-block neighbors known
kernel-internal temporary-buffer neighbors known
```

The specific hypothesis most strongly constrained by v16c/v18 is:

```text
current request's declared QMM output
  -> small forward overrun from the declared output tail
  -> next AddRms residual
```

The following paths remain open:

```text
FIA output/workspace overrun -> ReduceSum storage
another fused operator's temporary overrun -> ReduceSum storage
an incorrect device pointer writes directly to ReduceSum storage
an upper allocation underflows into ReduceSum storage
an allocator-block/padding overrun crosses an undeclared boundary
ReduceSum storage is corrupted after production but before residual consumption
```

## Proposed next gate and follow-up outcome

The next audit must change its subject from the QMM-to-residual relation to the
provenance and full live interval of a candidate ReduceSum-backed storage:

```text
residual entry
  <- view / copy / cast aliases, if any
  <- ReduceSum storage

last legal ReduceSum write
  -> every intermediate operator execution window
  -> residual-entry consumption
```

After proving this typed address/alias chain, take synchronized before/after
hashes around the intermediate execution sequence. First bisect large operator
intervals, then narrow the first changing interval to one L4 Run. A candidate
writer exists only when the storage changes across that Run and the operator
has no legal write permission to it. If the fully covered lifetime remains
unchanged, the conclusion is only that the current build and test input did not
naturally trigger the historical corruption.

The follow-up short, 353-token prefill, and two-request concurrent sessions
stopped at the provenance gate: neither rank executed external
`aclnnReduceSum` during the armed requests. They did show FIA in grouped
residual-live intervals, with no persistent hash change across those intervals.
There is therefore no request-time ReduceSum output to associate with this
residual, and no first changing interval to bisect. See
`fault26_residual_lifetime_visible_api_audit_20260818.md`.

The existing `QMM_TRIGGERED_RESIDUAL_STOMP_PROXY` remains a controlled FI
primitive for testing propagation and diagnosis. A future controlled
`FIA_TRIGGERED_REDUCESUM_STOMP_PROXY` could likewise reproduce the source
timing, victim, and propagation effect for a diagnostic-tool test. Neither
proxy is evidence that the corresponding real kernel naturally performed a
physical out-of-bounds write. No such FI is part of this v18 read-only result.

## Evidence

Remote:

```text
/home/haoran/fault39_dev_20260812/runtime/fault26_residual_writer_timeline_v18/
  inventory_20260818_phase13_residual_writer_v18.jsonl
  qmm_20260818_phase13_residual_writer_v18.jsonl
  analysis_writer_timeline.json
  baseline_unarmed.json
  response_audit.json
  recovery_unarmed.json
  arm.json
  disarm.json
  query_after.json
```

The same files are archived locally under
`.fault26_evidence/20260818_phase13_v18/` and intentionally excluded from Git.
