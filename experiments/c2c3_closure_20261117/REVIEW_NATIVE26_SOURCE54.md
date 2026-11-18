# Source54 single-site QMM artifact review

`review_native26_single_source54.py` reads the original result, detailed writer
inventory, serving receipt, allocator snapshots, frozen source manifest and
Native bundle. It uses only the Python standard library. It imports no runtime,
loads no DSO, opens no connection and writes only the compact report to stdout.
Runtime/library/network audit guards are installed by the CLI as an additional
check. Run it on the machine holding the originals; do not copy tensor arrays
into a chat or edit an original artifact.

```sh
R=/home/17-fault-c2c3-20261117
python3 -B /path/to/review_native26_single_source54.py \
  --evidence-root "$R/native26-single-c1-source54-card2-a1/providers26-v2-qmm-deepseek-c1_single_effects" \
  --source-root "$R/source-a54" \
  --native-root "$R/native26-pending-consumer-a2" \
  > /path/to/new-native26-independent-review.json
```

`--evidence-root` must be the directory actually containing `result.json` and
`writer-v2/`. Recorded `/evidence/` allocator paths map only beneath this root.
The CLI pins the three original JSON SHA256 values, source54 manifest and base
commit, Native build receipt, trace/writer images, selected key/index, 135-call
denominator, two resolved candidates and 728 aliases. There is no CLI option to
replace these pins. Pure-data CPU tests use a separate small synthetic fixture;
they confer no Native qualification.

The reader independently reconciles:

- Eight distinct enqueue → scheduler → actual root/logits request mappings,
  prompt/position digests, complete logits geometry and healthy/OFF hashes.
- Every Native event and counter delta, every natural QMM call, public producer,
  callback and next actual AddRms consumer. It rebuilds all candidate rows and
  two-healthy classifications, including unsupported and unverified entries.
  Global non-target calls between root/logits windows are retained separately;
  a QMM or detailed-event gap cannot be relabeled as such a call.
- The minimum resolved target key, actual module/method identity, every alias,
  per-request occurrence, unchanged Native formal signature and complete census
  hashes. Other resolved sites do not receive a lease.
- Saved relevant allocator bytes and active-block adjacency for each recorded
  allocation, including proof of nonadjacency used for unsupported entries.
  Public residual producer/consumer object and storage identities are joined to
  the same request and Native producer sequence; equal addresses alone do not
  establish the public lifetime.
- Reference/lossless/OFF/F/OFF leases and full Native receipts: same owner,
  generation, target, call, registration, stream, actual execution count,
  temporary lossless destination, 32-byte QMM residual span, complete residual
  restoration and acknowledgment. The first observed receipt must agree with
  the validated receipt; release may change only the release field.
- Native callback/passthrough/admission/completion counters, pre-fault
  reference/lossless/OFF issuance digest, final retired Native state and zero
  Python hooks, wrappers, leases and retained public-storage owners.
- Four loaded metadata-counter image identities and symbol-provider mappings,
  stable instances, monotonic counters, positive interval activity, no getter
  errors and no live arrays. Every Native build file and reviewed frozen source
  file is independently rehashed, including the separately referenced build
  receipt and its declared compilation inputs.

Successful per-call completion need not set `drained=1`: the Native receipt sets
that bit only when explicit drain sees an active transaction. Ordinary successful
consumption restores and acknowledges the lease before final stop/drain/remove.
The reader checks this ordered receipt/state chain and final zero resources;
it does not manufacture an extra per-call drain receipt.

Exit 0 means the saved **source-bound single-QMM evidence** reconciled. Exit 2
means a contradiction or required artifact/schema gap. Neither exit grants a
live capability or public qualification. `status`, `c1`, `evidence_complete`,
`healthy_repeated` and `recovered` are not authorities for the result.

Full tensor arrays were not saved, so their runtime hashes cannot be independently
recomputed from device bytes. The original complete allocator snapshot is also
represented only by a hash; only its saved relevant segments are rehashed.
External vLLM/model method source/code hashes bind recorded target identity but
are not a new live code-object observation. Strong-reference duration and exact
Native readbacks are corroborated by the pinned source and its full receipts.
The eighth health request checks full logits after Native removal; the later
ordinary generation checks tokens after Python cleanup, not full logits.

The result stays scoped to one actual QMM target in this fixed fresh-prefill
Worker lifetime. It does not cover all resolved targets, C2/C3, FIA or E5. The
separate E5 comparison reported an earlier KV-cache input difference and
incomparable preceding rows; a changed final logits hash here does not establish
fault-only attribution or unique fault location.

CPU validation:

```sh
python -B -m unittest tests.test_native26_single_source54_review
```
