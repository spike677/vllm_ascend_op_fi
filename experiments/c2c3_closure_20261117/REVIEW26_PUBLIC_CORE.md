# Independent public Worker Core Native26 review

`review26_worker_core_result.py` is a new stdlib-only reader for the public
`WorkerFaultClient(FaultConfig(26, variant))` attempt. It does not load runtime
code or Native libraries, execute a model, or reuse the previous direct-entry
C1 decision. Its two pinned independent helper files supply only atomic Native
trace, current allocator/consumer, descriptor, output-file and recovery checks.

The input is the unchanged evidence mount root, with `probe/result.json`, the
original `probe/host_receipt.json`, all `probe/worker_core_rpc/*.json`, and the
complete immutable `worker-core26-runtime/.../core-evidence` files. Allocator
segments and typed FIA stock/output binaries must remain at their original
paths below that same root. The reader checks SHA256 and length before reading
referenced files, rejects duplicate JSON keys and nonfinite JSON numbers, and
rejects path escape. It prints JSON to stdout and does not edit evidence.

The exact 33 public actions are joined to original RPC replies and their
arguments. Eight deferred, naturally enqueued requests must have distinct
actual scheduler identities. Query contains the completed current readback but
does not invent a new saved observation; only each explicit collect appends one
immutable full observation. Final finish appends the ninth cleanup original.
Private Runtime owner, actual Native owner words, generation and Worker/process
lifetime are joined across the whole sequence. Disabled/unregistered resources
and the later ordinary same-Worker health request are checked independently of
the result status or `evidence_complete` flag.

Every actual writer remains in the denominator. The two healthy native/public
inventories are reconstructed, including unsupported/unverified candidates.
Selection must be the original minimum stable resolved key with no fallback.
The selected site's five receipts must be fresh reference, lossless, qualifying
OFF, one F, and recovery OFF. Current public tensor lifetime, actual Native
writer and consumer, changed residual bytes, exact full residual restoration,
descriptor release, metadata getter-array ownership and zero remaining owned
resources are required. Typed FIA additionally verifies its own NCL descriptors,
complete typed call records and rehashes literal stock/output files.

Functional F effect uses the actual residual mutation and completed consumer.
F logits may be nonfinite or unchanged; those facts are reported separately.
All B/L/OFF/health logits must be finite and identical. E5 is not derived from
this reader. No C2/C3 authority or public qualification is created by JSON.

The immutable Core originals do not contain the separate logits-boundary
object. Per-request complete output shapes/digests therefore join the frozen
runtime's final-logits source guard. Full input, residual and logits bytes are
runtime readback/hash receipts; only saved allocator JSON and typed FIA output
files are independently rehashed. The post-unregister ordinary request compares
tokens/text, whereas the preceding health request captures full logits after
Native removal. External model code and weight digests remain original
source-bound receipts; the reader does not read those large/external files.

## Source67 typed FIA command

Root runs this CPU-only command against the retained full originals on 124.
`REVIEW_SOURCE` is the source snapshot containing this new reader and its exact
two pinned helpers; `SOURCE67` is the actual source67 deployment snapshot.

```sh
python3 -I -S -B "$REVIEW_SOURCE/experiments/c2c3_closure_20261117/review26_worker_core_result.py" \
  --evidence-root /home/17-fault-c2c3-20261117/writer26-core-fia-source67-card2-a1/providers26-core-fia-smol-c1_single_effects \
  --result-sha256 452837771bcc28a4899ce614bee4bb8f135853690aee5ab00abdb94d5d76a4f0 \
  --source-root "$SOURCE67" \
  --source-commit d8d14ffff58e98f2eca429b8cc5ccce1bfe30ecf \
  --source-manifest-sha256 81e574bc2b7f5962f635c1635837b8a60ca0d7ba645838d2970f742c810af089 \
  --native-root /home/17-fault-c2c3-20261117/native26-fia-typed-group-a4 \
  --native-build-sha256 2a6e4b861fe0342cb4274614d6edeb187bb36e3c87acab9ecc99be6f0e01bbae \
  --native-source-root /home/17-fault-c2c3-20261117/source59 \
  --native-source-commit 112f2b56f0f29fcb50567b6d9f5b327c94efe673 \
  --native-source-manifest-sha256 e6e142630331bd7951d48efd590d8fcb6224a1a4f3942ef96f7cc7040f70ab2f \
  --mechanism 26.fia_destination --expected-writers 30
```

QMM uses `--mechanism 26.residual_stomp --expected-writers 135` and the exact
new public attempt/source/build SHA pins; the old source54 direct experiment
cannot be substituted. An unavailable or inconsistent original yields exit 2.
Exit 0 reconciles saved evidence within these stated limits; it is not a new
device run or permission to arm a fault.

When an unchanged Native bundle is used with a newer Worker package, its actual
compilation snapshot is independently pinned with all three options:

```sh
--native-source-root "$NATIVE_COMPILE_SOURCE" \
--native-source-commit "$NATIVE_COMPILE_COMMIT" \
--native-source-manifest-sha256 "$NATIVE_COMPILE_MANIFEST_SHA256"
```

Omitting all three keeps the byte-equivalent Worker snapshot default; it does
not assert that the original compilation used that snapshot. Partial identity
is refused. Explicit identity must also match the commit and the unique
read-only `/source` mount in the original build receipt. Use the retained
original source directory, not a later tree with coincidentally identical files.
The Native snapshot's actual manifest bytes, commit and unique path set are
checked independently. Every compile input must still match the build receipt,
the saved Native manifest and the actual file SHA/length. Worker source and
loaded-module checks continue to use the newer Worker snapshot. The output
reports both identities explicitly; a passing build-status string cannot
replace either provenance chain. This distinction does not authorize a new
Native ABI or inherit a previous runtime qualification, and needs no new NPU
execution when all original runtime and build evidence already exists.

The original group builder did not record its source-manifest SHA. For those
receipts the manifest pin is explicitly attributed to this review, while the
receipt still binds the original commit, mount and compile-file hashes. A
recorded manifest SHA, if present, must additionally agree. For the QMM bundle
with manifest `b81b8b9bb1f0b5c0e56109775a987bf503b88a00adcfb873964a258e8bf4e03e`,
the original source is `/home/17-fault-c2c3-20261117/source-a50`, commit
`006916a9f9bd37549f0f21fc986ecace23e22e9b`, manifest
`c87d33a804b46f16ed44536a24863966fd08f0c99b2a21135a367b47d4c714cf`.
Source68 remains the separate Worker source for its new public attempt.

CPU tests use synthetic saved files and a three-writer QMM fixture. They test
the actual reader and isolated `python -I -S -B` CLI, including re-signed
negative originals, owner/consumer/restoration failures, independent F-effect
reporting, source/Native file mismatches and no evidence changes. They do not
claim new FIA/NPU acceptance; the existing pinned typed helper has separate
typed CPU contracts, and root must run this new public reader on the actual
complete FIA attempt.
