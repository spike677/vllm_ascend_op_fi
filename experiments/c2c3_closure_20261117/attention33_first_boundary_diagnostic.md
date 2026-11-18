# No-fault first-decode boundary evidence

This opt-in diagnostic addresses source48's stable stock/private difference at
the first recorded attention output. It does not repair that difference, alter
the exact-byte bridge, or authorize F. Existing artifacts and their conclusions
remain unchanged. Both new cases must use one newly frozen source and matching
installed wheels, with the same root-selected card, model and observer build.

The existing case preparer now accepts `--boundary-diagnostic --stage baseline`.
It emits exactly two independent cases, `attention33-fd-first-boundary-stock`
and `attention33-fd-first-boundary-baseline`. Neither enables statistics or F.
Both preload the same reviewed runtime observer. The original immutable
observer receipt may name `/output/.../observer.so`; the preparer adds only a
read-only alias of the same hash-verified observer directory at that path.
No container or device is started by the preparer or CPU tests.

```sh
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/prepare_attention33_cases.py" \
  --source "$SOURCE" --installed "$INSTALLED" --build-root "$BUILD" \
  --observer "$OBSERVER" --model "$MODEL" --out "$FRESH_CASES" \
  --stage baseline --boundary-diagnostic --card 2 --timeout 1800
```

Root reviews and schedules those cases. The driver additionally refuses
`--first-boundary-diagnostic` with fault/off, statistics, or a bridge argument.
The result schema is `demoncat.paged33.first_boundary_worker_run.v1`; the existing
bridge's original run validator rejects this schema. `fault_authorized`, C2 and
C3 remain false. This prevents diagnostic readbacks from silently becoming a
qualification run.

The natural warm request, five existing B/repeat/measured/OFF/healthy phases and
post-clean healthy request remain. In each phase, only the first decode's first
actual, already audited API call is captured, and it must be the real layer-0
attention module with the matching object identity. Later calls and prefill
pass through without these snapshots. The private arm retains its existing
112-launch validation across its four adapter phases; the stock arm uses one
read-only runtime scope per captured phase. Private `healthy_after` still uses
the loaded private vendor, with its scoped adapter removed; it is not a stock
library recovery claim.

Each `phase/first_boundary/` directory contains the actual query's complete
bytes, each returned Tensor's immediate complete bytes, and `boundary.json`.
The query is 4096 bytes at the observed fp16 `[1,16,128]` geometry, but geometry
does not select a Native call. The capture records all supplied scalar/optional
arguments, tensor metadata and complete hashes for mask and block table,
actual block-table values, and complete K/V hashes in the proven logical page
order immediately before that call. The packet's default-overload schema and
omitted defaults are recorded when available, explicitly without claiming an
internal dispatch path was observed. Unavailable schema stays null.

Copies and synchronization run in identical capture code in both arms. They
never replace, repack, or modify a supplied input. The original callable and all
argument objects are retained and checked; the original is called once and the
same result object is returned. In the actual diagnostic the saved callable
must be the actual registered public Torch NPU packet; unknown outer wrappers
are refused. At request completion, all retained input and
output contents, tensor/storage identities and available version counters are
checked again. Changes remain in the report. Inference tensors have no version
counter, so a write followed by restoration between snapshots is explicitly
unobservable. This is checkpoint equality, not a proof of no intervening write.

Stock runtime evidence joins the real scope's PID/TID/time, actual runtime API,
function handle where exposed, registered binary handle/generation, exact
registered binary bytes/hash, key, blocks, schedule and raw argument hash.
Multiple launches are preserved. No private key, private argument offsets or
private kernel ABI classify stock launches. Unobserved entrypoints, missing
joins, invalid raw snapshots, budget overflow and unavailable schedule remain
explicit gaps. Raw records/binaries stay on the evidence host. The report pins
the observed prefix of the append-only runtime JSONL, because subsequent
natural calls legitimately append records.

The capture also hashes actual matching files in `/proc/self/maps`, checking
mapped inode/device and file stability. These prove mapped file fingerprints,
not which ACLNN resolver symbol the original call selected. The existing
observer does not record that selected API-library owner, nor does it guarantee
coverage of every possible runtime entrypoint. Both limitations remain false
or missing evidence; they are not inferred from names, paths, shapes or the
private bundle. The private adapter's existing source/binary/ABI audit is
unchanged.

Cleanup drains before releasing held input/return references. Wrong Worker,
model, rank, thread, an active original call, or a failed drain cannot release
those references. A stock scope retains its observer on uncertain end failure
and can retry on the original thread; the original exception stays first in its
receipt. The reviewed observer's explicit `end == -2` means its token was
cleared but writing the end event failed: capture still fails while resource
cleanup can be confirmed. No original invocation is retried.

CPU contracts cover both real Tensor capture paths, argument/callable clone
refusal, delayed input/output mutations, inference-version gaps, first-error
retention, cleanup retry, multiple stock launches, mismatched owner/function
joins, and the two-case/no-F preparation boundary. They grant CPU evidence only.
