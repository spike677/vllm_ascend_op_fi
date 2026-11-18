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

CPU portability correction: `postwrite_evidence.process_start()` stores a bare
start tick on Linux, whereas the Worker stores `proc:boot:PID:tick`; Windows uses
the explicit `cpu-contract:PID:nonce` fallback. The CPU auditor now checks that
a Linux tick exactly equals the recorded Worker's tick with the same PID, or
that the fallback contains that PID and a numeric nonce. Natural Worker review
still requires complete identity equality and does not accept the CPU tick
projection. Four fixed-example tests exercise both platforms without depending
on the host OS. No provider, original receipt or device qualification changes.

Current producer receipts can now retain the complete `proc:boot:PID:tick`
identity in CPU mode too. The auditor accepts that exact valid marker only
when its embedded PID matches the recorded Worker PID; historical tick and
Windows CPU-contract forms remain supported under their previous checks.
A fifth fixed-example lifetime test covers this source76 representation and
rejects foreign boot, PID, tick and malformed complete markers. No original
receipt is generated or rewritten to make this test pass.

Current-producer regression review is based on integration
`1653be27e1f79a5b0da2a62ada2976b8d7e54813`, including the producer that now
copies the held Worker process identity. The unmodified Windows CPU protocol
reproduced all three `ACTUAL_WORKER_TARGET_LIFETIME` failures before the fix.
Its current marker is `process_local:PID:32-lowercase-hex-nonce`, produced by
`identity._process_start()` when `/proc` is unavailable. CPU review now accepts
this form only on exact marker equality, matching PID and valid nonce syntax.
Natural review requires a complete, matching `proc` identity; even two equal
CPU `process_local` markers cannot qualify as a natural Worker lifetime.

After the fix, all 25 tests in `test_worker_core_group_original_review` pass
against that current producer. These include the actual host-platform full
fixture plus two complete public CPU protocol runs with deterministic Linux
and Windows identity-reader inputs. The producer, collector, RPC flow and
audit remain real CPU implementations in those runs; saved originals are not
edited. This is a portability/receipt regression result, not NPU acceptance.

Native26 two-generation review is based on current integration
`c0045f283de5af99628da9631d19d3ee9769d66d` (source77). The same CLI now accepts
both `c2_once` and `c2_c3` original structures. It checks the stable public RPC
generation separately from the fresh C3 Native generation, the completed C2
`prior_c2` originals at the transition, and the final C3 originals. Each run's
complete Native denominator, every eligible writer/consumer, Reference,
Lossless, F and OFF, complete output receipts and zero owned resources are
reconciled independently. C3 must include both F requests and exact repeated
output evidence. Unsupported Native calls stay in each generation's census
and execution ledger.

The transition review compares the original issuance and consumption fields,
object binding, consumer identity, scope, owner, next Native generation, full
target set and uninterrupted Native/typed trace boundaries. It does not
recreate a live proof object from JSON. Every offline report continues to say
`c2=false`, `c3=false`, `e5=false`, `public_qualification_granted=false`;
`group_completion_reconciled` only describes consistency of saved originals.
Failures and missing request originals remain partial, with the earlier C2
facts and each run's observed denominator retained. A saved success flag
cannot supply a missing writer, outcome, original file or generation.

The source77 held-function environment profile is reconciled without loading
libraries or resolving symbols: all 16 source-defined getter slots, exact
saved executable mappings, image hashes, retained collision groups and scope
material must agree. Both the actual Worker and independent Native compiler
snapshots remain required by provenance review. Current implementation hashes
include `environment.py`; older C2 source records retain their original hash
composition. The reviewer hashes every available run's provenance separately.
Saved mapping/function records are not a fresh runtime mapping attestation.

Validation uses the actual current public CPU driver for QMM and typed FIA,
including all 17 requests and a post-unregister ordinary health request. The
FIA CPU case retains three Native writers, two eligible targets, and C2/C3
hits of two/four. A separate, explicitly enhanced CPU ABI test envelope checks
the C3 typed verifier's 66 records and 18 full output artifacts. Negative cases
cover wrong generations, copied object bindings, scope or target loss, missing
second-F outcomes, missing OFF proof, leaked resources, altered original SHA,
missing held symbols and incomplete/failed groups. These are CPU contracts;
this change performs no SSH, container or NPU action and adds no device
qualification or new statistics data.

The six-module local regression passed 76 tests: the existing C2 originals,
new C2/C3 originals, current-producer Fault39 originals, C2 public driver,
C2/C3 public driver and current every-match provider contracts. The separate
CLI help check also passed. These counts describe one regression invocation,
not additional independent device cases.
