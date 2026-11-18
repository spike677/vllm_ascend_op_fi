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
