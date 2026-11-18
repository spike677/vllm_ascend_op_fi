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
  --mechanism 26.fia_destination --expected-writers 30
```

QMM uses `--mechanism 26.residual_stomp --expected-writers 135` and the exact
new public attempt/source/build SHA pins; the old source54 direct experiment
cannot be substituted. An unavailable or inconsistent original yields exit 2.
Exit 0 reconciles saved evidence within these stated limits; it is not a new
device run or permission to arm a fault.

CPU tests use synthetic saved files and a three-writer QMM fixture. They test
the actual reader and isolated `python -I -S -B` CLI, including re-signed
negative originals, owner/consumer/restoration failures, independent F-effect
reporting, source/Native file mismatches and no evidence changes. They do not
claim new FIA/NPU acceptance; the existing pinned typed helper has separate
typed CPU contracts, and root must run this new public reader on the actual
complete FIA attempt.
