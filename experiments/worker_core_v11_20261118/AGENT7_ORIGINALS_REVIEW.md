# Worker Core V1.1 saved-originals review

This change adds offline reviewers. It adds no new device result and does not
change an existing C1/C2/C3 qualification. All returned public qualification,
C2, C3 and E5 flags are false: a saved JSON is not a live capability.

`review39_group_originals.py` reads the original result, every numbered RPC,
Host receipt and explicitly referenced immutable group original. It verifies
their SHA/byte references and rejects traversal, duplicate JSON keys and
nonfinite JSON numbers. It imports no model/runtime and loads no Native DSO.

The auditor reconstructs the two healthy complete projection inventories,
eligible and unsupported candidates, real Worker identities, every admitted
writer/value-consumer pair, original Native open/apply/restore/release/readback,
each lossless/OFF control, the complete non-target ledger and owned resources.
C2 uses independent target replays; C3 has two complete requests with every
eligible target. It checks the same live-C2 object-binding receipt is consumed
by C3, but does not manufacture live authority from that saved binding.

`contract_satisfied`, probe `status` and probe `c2/c3` are not acceptance oracles.
The initial single-site six requests are checked separately. Changed or absent
effects, incomplete denominators, missing Native returns or recovery are errors.
Failed/partial attempts preserve their error and cannot become completed groups.

Two output digest domains must remain distinct: the group's `output_sha256`
is a complete `(hidden, logits)` bundle, whereas `actual_requests.output_sha256`
is a logits digest. Each is compared with its own reference, OFF and health.
The common actual request, observed inventory and frozen source establish the
join; the auditor does not equate two differently defined hashes. Tensor bytes
are not independently re-read by this reviewer. Ordinary post-unregister health
checks tokens/text; preceding post-release health has the complete digest.

Run on124 **after** root obtains the exact original/result and source pins:

```sh
python experiments/worker_core_v11_20261118/review39_group_originals.py \
  --evidence-root "$CASE" --result-sha256 "$RESULT_SHA" \
  --source-root "$ACTUAL_FROZEN_SOURCE" --source-commit "$COMMIT" \
  --source-manifest-sha256 "$MANIFEST_SHA" --native-library "$ACTUAL_DSO"
```

This rehashes the actual loaded source set, source manifest and DSO. External
model/weight digests remain saved source-bound receipts; a build fingerprint
string does not prove a complete compiler/build receipt. The summaries state
these limitations. No statistics or injection log selects an E5 first difference.

Local validation: `tests.test_worker_core_group_original_review` has 16 passing
CPU contract/negative tests against originals produced by the real public
Host/Runtime CPU fixture. Its Native transport is an explicit double, so this
is not NPU acceptance. No SSH, container or NPU call was made for this change.

`review26_group_originals.py` adds a separate complete-set audit for
`26.residual_stomp` and `26.fia_destination`. Its scope is one cumulative fresh
request containing every stable eligible writer, followed by actual OFF and
health. It independently joins all Native entries to callbacks, original
writer execution, the previous residual producer and next value consumer;
checks Reference/Lossless/qualification-OFF/F/OFF for every target; rehashes
the original execution, recovery, complete-census and health artifacts; and
checks real owned resources and SDK metadata-array readbacks.

Missing or unresolved candidates block the whole set. A stale Native producer
from a previous request can be excluded only with an uninterrupted current
request trace proving no preceding producer and the exact next producer.
The test suite covers trace gaps, changed thread and a false absence witness.

FIA additionally checks all typed descriptors/public returns, destination and
executor joins, exact descriptor release, two stock repetitions and every
owned Reference/Lossless/OFF output file. Files remain on124. It reports the
number and total bytes rehashed, without publishing tensor contents. The last
typed state is the actual health request's after-state; a separate post-cleanup
typed state is not invented.

```sh
python experiments/worker_core_v11_20261118/review26_group_originals.py \
  --evidence-root "$CASE" --result-sha256 "$RESULT_SHA" \
  --mechanism 26.fia_destination --expected-writers "$FULL_NATIVE_COUNT" \
  --source-root "$ACTUAL_WORKER_SOURCE" --source-commit "$WORKER_COMMIT" \
  --source-manifest-sha256 "$WORKER_MANIFEST_SHA" \
  --native-root "$ACTUAL_NATIVE_BUILD" --native-build-sha256 "$BUILD_SHA" \
  --native-source-root "$ACTUAL_COMPILE_SOURCE" --native-source-commit "$COMPILE_COMMIT" \
  --native-source-manifest-sha256 "$COMPILE_MANIFEST_SHA"
```

The worker and compiler snapshots are independently pinned: a newer Python
source tree cannot stand in for the Native build's actual source. Writer26
JSON digests retain their original default separators; coverage digests retain
their compact separators. Neither original byte stream is rewritten.

Local validation adds 21 tests in `test_worker_core26_group_original_review`.
The typed CPU fixture has three stock writers, two eligible targets, 58 typed
records and 16 complete output artifacts; these are test data, not124 results.
The CPU public fixture deliberately has simplified Native ABI doubles. Its
topology mode cannot pass the natural-Native audit. No CLI switch enables that
CPU mode for device acceptance.

Source71's reported `Exact workload ... scope required` refusal occurred before
the shared F request completed. It remains failed/partial even if a summary
claims success. This change has not connected to124 or independently read that
attempt; the root must run this reviewer over the retained originals.
