# Fault41 State/Cache Tail Corruption Roadmap

## Fault family

Internal family name:

```text
state/cache tail corruption
```

The first supported implementation should use an explicit proxy name:

```text
OP_TRIGGERED_STATE_CACHE_TAIL_CORRUPTION_PROXY
```

If the target is proven to be a concrete KV-cache slot, the implementation can
be narrowed to:

```text
KV_CACHE_POST_WRITE_TAIL_CORRUPTION_PROXY
```

## Boundary

Fault41 must not claim:

```text
healthy cache writer naturally performed physical cache OOB
```

until a natural OOB is directly proven. The intended developer-mode proxy is:

```text
real cache writer runs
→ modify a proven live cache storage tail region at the historical timing point
→ later attention consumes the corrupted cache
→ observe token/model/diagnostic behavior
```

## Reuse of existing L4 infrastructure

Fault41 should reuse the validated lower-level project framework instead of
starting a separate architecture:

| Existing capability | Fault41 reuse |
| --- | --- |
| Fault20 metadata lifecycle | identify operator, tensor metadata, executor/cache identity |
| Fault20 buffer substitution | reuse control plane, hit selector, rank/shape/call limits |
| Fault26 victim/writer topology | build writer → live state → consumer relation |
| Fault39 post-Run observer | record before/after hash, values, non-finite counts, downstream propagation |

Recommended structure:

```text
common L4 runtime
├── metadata registry
├── request/hit control
├── stream synchronization
├── D2H/H2D helpers
└── evidence logger

Fault41 adapter
├── cache writer observer
├── cache victim resolver
├── tail region calculator
└── corruption transform
```

Fault41 should not become a long-term pile of additional `if (mode == ...)`
branches inside the Fault20 adapter.

## Tail definition

The implementation must define which “tail” is targeted before any corruption is
armed:

1. final logical elements of the whole tensor;
2. tail of the current token's `head_dim`;
3. tail of a paged KV-cache block/slot;
4. allocation block tail or allocator padding;
5. stale state that is still live until a later consumer.

First-pass recommendation:

```text
specified request
+ specified decode step
+ specified K or V cache
+ proven cache block/slot
+ last N logical elements of that slot
```

Avoid the naïve form:

```text
cache_address + total_storage_bytes - N
```

because paged cache may include block layout, strides, padding, quantization
scale, or private storage layout. The physical end of a storage allocation may
not correspond to the target token's logical tail.

## Required phases

### Phase 0 — read-only relation proof

No data mutation.

Required evidence:

```text
cache writer typed metadata > 0
cache address/bytes/dtype/format known
writer-after cache hash readable
later attention consumer uses same storage
request/step/block/slot relation is explicit
```

Output labels:

```text
CACHE_STORAGE_IDENTITY_PROVEN
CACHE_WRITER_CONSUMER_RELATION_PROVEN
```

If only an opaque address is available and no logical slot can be calculated,
Fault41 must not proceed to tail mutation.

### Phase 1 — lossless gate

```text
writer completes normally
→ target cache region D2H
→ original bytes H2D writeback
→ readback
→ downstream consumer continues
```

Required result:

```text
changed_bytes = 0
response token IDs exactly match baseline
HTTP 200
worker healthy
```

### Phase 2 — single finite corruption

First mutation should remain finite so the fault does not collapse into Fault39
non-finite semantics.

Initial modes:

```text
xor_sign_bit
zero_tail
low_bits_clear
finite_saturate
```

Initial limits:

```text
rank/device       = 0
cache kind        = K or V, not both
step              = fixed decode step
writer occurrence = 1
cache slot        = proven slot
length            = 16..256 bytes
max_hits          = 1
```

Required evidence:

```text
original/mutated address
logical cache slot
offset and length
dtype/shape/stride
hash before/after
changed elements
consumer address
first divergent token
```

### Phase 3 — sustained fault

Only after Phase 2 passes:

```text
16 hits
1024 hits
all-hit within one request
```

Because cache state is persistent, all-hit must mean:

```text
every legal cache update corrupts its corresponding live slot
```

not repeated writes to a stale address.

### Phase 4 — recovery

Disarm alone is not sufficient if cache storage remains corrupted. Recovery must
be one of:

```text
restore saved original regions
release and reinitialize affected cache blocks
clean restart
```

## Control-plane sketch

Fault41 should use independent commands, for example:

```bash
fault41_ctl arm-cache-tail \
  --device 0 \
  --kind value \
  --step 4 \
  --occurrence 1 \
  --tail-elements 32 \
  --mode xor-sign \
  --max-hits 1
```

Shared state should include:

```text
enabled
session/generation
target_device
target_cache_kind
target_step
target_occurrence
tail_elements
mode
max_hits
hit_count
changed_elements
last_status
```

For sustained online mode, keep:

```text
per-request cap
+ total hard cap
+ explicit disarm
```

Do not clear `hit_count` when `hit_count == max_hits` inside the same request,
because that turns a bounded request into uncontrolled all-hit.

## Evidence policy

Follow the current Fault20/Fault26 pattern:

```text
nohit_dumpon
single_hit_dumpon
allhit1024_dumpon
recovery
```

In addition to msprobe dump, Fault41 needs compact native summaries proving:

```text
real cache storage was hit
mutation happened inside the dump window
changed_elements > 0
later consumer used the same storage
```

Model-level records should include:

```text
no-hit response
single-hit response
all-hit response
token IDs
first divergent token
generation length
HTTP/worker status
```

## Current project status

```text
Fault41 fault model design: feasible
Common L4 infrastructure: available
Old dispatcher LD_PRELOAD route: not interposed
Real cache writer/storage relation: not closed
Model-level cache corruption E2E: not complete
```

Next minimum target:

```text
locate real cache writer
→ capture real cache slot metadata/address
→ prove writer-consumer relation
→ complete lossless in-place writeback gate
```

No all-hit Fault41 experiment should run before those four conditions are met.
