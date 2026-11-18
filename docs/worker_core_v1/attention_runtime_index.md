# Owned runtime log index

This optional CPU reader avoids parsing the same runtime JSON lines on every
call. It does not modify a log, classify an expected kernel, grant admission,
retry a fault, or replace final independent review. Existing callers continue
to use the full reader unless they supply the exact `RuntimeLogIndex` class.
The source108 frozen package and its incomplete private normal run are unchanged.

The index belongs to one process, Python thread, native thread, and resolved
runtime directory. Every snapshot opens the original file, checks its device
and inode, hashes the complete previously accepted prefix, and reads the new
suffix. Truncation, replacement, old-byte changes, partial lines, duplicate
JSON keys, and growth during that read are refused. Append parsing is
transactional: an invalid final row cannot commit earlier rows in the suffix.
The existing 128 MiB runtime limit remains in force.

A snapshot includes every row with the actual call token, every registration
generation for every handle launched in that call, every binding identity for
every corresponding function handle, and all capture-overflow events. It does
not filter these histories by an expected key, generation, or owner. Auxiliary
launches remain present. Original order and one-based raw line numbers are
preserved. Returned dictionaries, lists, and nested values are copies; changing
a verifier's snapshot cannot modify the cache. Registered ELF bytes are still
rehash-checked by the original verifier on every use.

The full readers perform the same scope, registration, argument, ELF, and
completion checks after receiving these rows. The new contracts compare full
and indexed success results and rejection reasons for owner, ordering, key,
generation, binary, raw-argument, synchronization, and overflow failures.
Additional cache-only contracts cover replacement, tampering, incomplete
append, invalid append transactionality, foreign threads, read-time growth,
and snapshot mutation.

This is not equivalence for arbitrary malformed files: the index additionally
rejects invalid unrelated rows, duplicate keys, nonfinite JSON constants and
missing final newlines. These stricter failures are retained, not treated as
successful observations. Parity covers valid observer logs and the tested
conflicting registration/launch histories.

Complete raw logs and binaries remain final-review inputs. This saves repeated
JSON decoding, not repeated old-prefix hashing or binary verification. The
new group probe must measure its complete normal phase cost, including these
checks and capture/cleanup. Neither CPU tests nor the index report establish
that a 60-request protocol fits the unchanged 1800-second case limit.
