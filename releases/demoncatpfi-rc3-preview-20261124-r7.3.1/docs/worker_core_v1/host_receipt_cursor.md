# Optional Host receipt cursor

This interface changes receipt transport only. It does not grant F, C1, C2,
C3 or E5, resolve the Fault33 Group time/capacity gate, or change Worker
Recovery. No Group, Runtime, registry, Native or exporter code is changed.

## Compatibility and API

`WorkerFaultClient(host, config)` and explicit `receipt_mode='full'` retain the
legacy `demoncat.worker_core.host_receipt.v1` action results, field values,
complete internal history and lifecycle behavior. There is no cursor ledger
in full mode. `snapshot()` remains the explicit complete legacy snapshot in
both modes; it can still be expensive.

Trusted Host code can opt in when constructing a new client:

```python
client = WorkerFaultClient(host, config, receipt_mode='cursor')
start = client.receipt_cursor()       # Capture genesis before implicit attach.
control = client.prepare()           # Can append attach AND prepare.
batch = client.receipt_batch(start)  # Complete, isolated original records.
cursor = batch['next_cursor']
status = client.status_receipt()     # Host read only; no Worker RPC.
```

| API | Result / hard boundary |
| --- | --- |
| Action methods in cursor mode | `cursor_control.v1`, at most 128 KiB; no history or complete RPC response |
| `status_receipt()` | Same bounded control schema, no new record and no RPC |
| `receipt_cursor()` | Current live ledger position; no action authority |
| `receipt_batch(cursor, limit=4)` | At most four records and 8 MiB **including its envelope**, each complete record at most 4 MiB |
| `stream_receipt(cursor, sink)` | One original Host record, writes at most 64 KiB per call, at most 128 MiB total |
| `stream_snapshot(sink, cursor=tail)` | Complete legacy v1 value, at most 128 MiB total; cursor must be current tail |
| `snapshot()` | Existing complete v1 in-memory copy, unchanged and not a bounded API |

Byte bounds use canonical UTF-8 JSON: `allow_nan=False`, `sort_keys=True`,
`ensure_ascii=True`, `separators=(',', ':')`. JSON written with other whitespace
can have a different byte length. Limits are fixed implementation constants,
not RPC settings or user-configurable options. `limit` must be an integer from
1 through 4; booleans are rejected.

Control results keep exact live attachment/lease/census tickets when the
control fits. They have a distinct schema and explicit
`history_included=false` / `rpc_response_included=false`; consumers must not
treat their short `response_header` as the original RPC receipt. If even the
control exceeds 128 KiB, the return is bounded and has
`control_complete=false`, `omitted_fields`, and `first_evidence_error`.
An oversized identity or ticket is omitted as a whole, never truncated into
an apparently usable ticket. The internal live ticket and complete original
record remain available to the existing recovery path.

## Ledger and trust boundary

Each cursor-mode client owns a fresh process-local random ledger ID pinned to
its Host owner and tenant. Each position contains exactly:

```text
schema, ledger_id, owner, tenant_id, next_sequence, prefix_sha256
```

The genesis prefix hashes the canonical cursor schema/ledger/owner/tenant
seed. Each appended prefix hashes the canonical object containing its
zero-based `sequence`, `previous_sha256`, `record_sha256`, and `bytes`.
Record bytes are the canonical serialization of the existing Host history
entry `{action, response}` after the existing JSON copy. They preserve every
field of that entry; they are not a claim to preserve original network JSON
whitespace or an unreceived Worker response.

Only the existing Host action recording path appends receipts. The module
does not read files or import a saved ledger, receipt, ticket or cursor to
create Host/Worker state. A cursor is a read position, not a lease or proof.
Wrong ledger/owner/tenant/schema/sequence/prefix and extra fields are rejected.
A rejected read cursor does not append Host history, alter recovery, or poison the
retained first size/stream error.
A newly constructed client has a different ledger even for the same mechanism.
The original one-attempt `prepare()` policy remains in force.

Historical positions remain valid for reading the same retained history,
including after unregister/detach. Reads are idempotent and do not consume a
cursor. Consequently, a tail cursor alone does not prove a complete export:
an independent consumer must join genesis through every sequence/prefix and
the final count. A missing original receipt is not replaced by its control
header or hash.

Cached records are immutable bytes; returned cursors, controls, records and
metadata are fresh copies. Reading a record compares the retained internal
history entry with its cached bytes. Caller mutations cannot change later
reads, tickets or prefixes. Changes to indexed originals or owner are rejected.

## Accepted RPCs and a failed local index

The complete internal Host history remains the original receipt source. The
cursor index is a retryable derivative: each entry commits its raw bytes,
digest and resulting prefix together in one append. Prefix encoding/allocation
failure cannot leave one of two index lists ahead of the other. A later retry
uses the committed index length, so an exception reported after an entry was
actually appended does not duplicate it or advance the prefix twice.

Cursor-mode receipt processing occurs outside the catch path for a Worker RPC
that was already received and accepted. A local index/control error does not
create a second synthetic `RPC_UNCERTAIN` history entry, set the Host's
`_uncertain` flag, or resurrect a released owner. The ordinary full-mode path
keeps its previous behavior.

An action whose original is retained but not indexed returns its actual RPC
status and live ownership fields with `control_complete=false`,
`receipt_index_complete=false`, `cursor=null`, `last_record=null`,
`indexed_record_count`, `original_record_count`, and `current_evidence_error`.
It does not describe the shorter index as a complete cursor at the new tail.
If only control rendering failed after a complete index, the degraded result
still has `control_complete=false` and records that distinct error.

The next action or explicit cursor/batch/status/export read retries only the
unindexed tail from this client's retained Host history. Reads send no Worker
RPC and never accept external files as replacement history. While indexing
still fails, cursor/export operations refuse rather than report completeness;
existing recovery actions can keep recording their actual results. When the
local failure clears, every retained original can be indexed and streamed,
including the actions that completed during the failure. The first evidence
error remains recorded after successful repair. Retrying evidence reads does
not retry `enable`, `detach`, or any other Worker action.

## Streaming and failure handling

The sink must be an empty, seekable binary file opened by the caller, for
example with `open(path, 'xb')`. The interface does not open, truncate, delete
or close it. It precomputes the complete export size and refuses the 128 MiB
overflow before writing. A nonempty sink is refused without overwriting it.

`stream_snapshot` assembles legacy v1 from the retained per-RPC bytes and
current header fields; it does not clone or serialize the entire history into
one new JSON string. It checks all retained records before export. The client
lock holds the selected state stable during export; reentrant mutating action
calls from a sink are rejected before sending a Worker RPC. Read-only cursor
and status calls cannot change the export.

A successful export receipt includes byte count, SHA-256, end cursor,
`complete=true`, and `qualification_granted=false`. It also says
`sink_readback_verified=false`: its hash covers bytes accepted by the sink's
write interface. The caller must independently check the persisted file's
size and hash. A successful write/flush does not establish filesystem
durability or any Native qualification.

Positive short writes are completed. A zero/invalid write, I/O/flush error,
changed size/hash or overflow raises `ReceiptLedgerError`. Export errors
include `complete=false`, confirmed bytes, the partial-prefix hash and
`internal_originals_retained=true`. A partial sink stays partial; the error
does not report it as a complete receipt. Retry can use a fresh sink and the
same still-valid cursor. Streams never retry a Worker action.

Batch overflow is separate from export overflow: a record over 4 MiB remains
in both the complete history and immutable ledger, is never skipped, and can
be streamed if it fits the 128 MiB export boundary. A batch can return the
preceding complete records; the next read at the oversized row refuses. A
record over the export boundary is retained but cannot be exported through
this bounded interface.

The first receipt/evidence error is retained separately from RPC failures.
An export failure neither sets nor clears the Host's RPC-uncertain state.
All original failures and responses remain in history. Existing
`recover`/`disable`/`unregister`/`detach` retain their exact lifecycle behavior.
In particular, a lost prepare response still has no lease ticket: same-tenant
`detach()` is the recovery route, and a cursor cannot substitute for a ticket.
No failed or truncated export authorizes a subsequent fault. These evidence
errors are transport diagnostics, not a new lifecycle gate: later Group
integration must explicitly refuse new F when evidence/cost checks fail while
leaving the existing recovery actions available.

## Scope of CPU validation

The focused tests exercise the real CPU `WorkerFaultRuntime` and Host path
with a test readback provider. They cover default legacy field compatibility,
implicit attach, cursor mismatch/reprepare, isolated copies, exact record and
batch byte boundaries, control overflow, stream/value equivalence, partial
I/O, reentrant mutation refusal, and actual recovery after oversized evidence
and a lost enable reply. They do not start a Worker, load a Native DSO, enqueue
model inference, or establish a device-time budget.

```text
PYTHONPATH=src;fault39/src;.;tests;tests/worker_extension
python -B -m unittest test_host_receipt_cursor test_core_host test_core_host_census -v
```

Group integration, complete 189-RPC independent review, current three-control
timing, final write cost, and pre-F/per-F resource checks require a separate
reviewed increment. This interface-only change supplies none of those
permissions or measurements.

The follow-up CPU contracts retain the independent review's two counterexamples:
one `enable` index `MemoryError`, and a control `MemoryError` after actual
`detach`. They also cover persistent indexing failure throughout actual
`recover`/`unregister`/`detach`, then complete legacy export, plus failures before
prefix commit and after entry commit. These are local evidence-failure tests;
they do not establish operation under process-wide memory exhaustion.
