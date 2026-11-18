# Source64 quant single-site original-evidence review

`review_quant_c1_source64.py` is a read-only, standard-library CLI. It imports only
the pinned artifact/hash helpers from the adjacent
`review_native26_single_source54.py` (SHA256
`8151f93dca35ead3b32b98d66881e5bc23432884bf5af8c1c6694f1b59fb8b5d`). It does not
import the injector, Torch, the Worker, or a Native library. Keep both files
adjacent when copying the reviewer to the CPU audit location.

The five original JSON files, source manifest, and Native build manifest are
pinned in the reviewer. Reading uses explicit paths; there is no glob search,
name-only substitution, raw artifact mutation, or qualification flag shortcut.
The source snapshot remains commit
`46cd8ad153ad07ce1057cfcbac73d7d9fb76e226`. The reviewer has its own later commit.

Run on the host CPU, with the actual source64 directory substituted below:

```sh
python3 -I -B /review/review_quant_c1_source64.py \
  --evidence-root /home/17-fault-c2c3-20261117/quant-c1-nd-flat-source64-card2-a2/providers-quant-group-deepseek-single-site-c1 \
  --source-root /home/17-fault-c2c3-20261117/source64 \
  --native-root /home/17-fault-c2c3-20261117/native-quant-group-nd-flat-a2 \
  > /new-review-directory/review.json
```

The exact source directory may use a different frozen-source name. The manifest
must hash to `c8a63053271e169f8317e76c5196348ee4b33eb48c591d07bc50eec3c3e018e6`.
This command creates only the redirected new report. It neither starts a
container nor accesses a device. Original JSON totals about 338 MB; the reader
loads each once and avoids a second full JSON serialization when hashing.

Exit 0 means the frozen single-site **source-bound receipts** reconcile. Exit 2
means missing, refused, inconsistent, or changed evidence. A source or schema
gap is retained as a compact issue, never repaired in the original artifact.
`c1`, `status`, `functional_effect_proven`, and Native qualification totals do
not determine the conclusion. C2 and C3 remain false and E5 remains
`NOT_ASSESSED`, including after exit 0.

The reader checks:

- Every raw event in each of seven actual decode windows, the 135 model parents
  per window, actual public DynamicQuant/QMM owners, and Native counter deltas.
  The extra F reference is a Native helper, never a model-parent denominator.
  Unresolved row-lineage candidates remain visible.
- Two complete healthy censuses and the first qualifying natural candidate in
  their original order, including the exact method, weight storage, geometry,
  scheduler rows and closed producer-to-consumer proof. An unsupported later
  path does not erase a completed selected prefix.
- Fresh request/forward/node/storage/write-epoch proof at Native admission,
  actual dispatcher input, and its recomputed digest in the Native lease.
  Host owner, private Runtime provider owner, and Native owner words are
  distinct identities; every original RPC binds the former two, and the
  Native owner words are recomputed from the private owner and Worker lifetime.
- R/L/qualification-OFF/F packets, all actual GWS and execute events, complete
  typed arguments, original-call counts, child roles, registration/executor,
  stream/workspace, owned-buffer separation, current-input reference, flat
  effect domain, changed-byte bounds, input preservation and publication
  readback. F output is consumed before a later fresh OFF request; no late
  public-output rollback is accepted.
- All seven B8 request mappings, complete hidden/logits hashes, selected input
  and Q/scale hashes, exact healthy/OFF restoration, and a real full-model F
  effect even when generated tokens remain identical.
- Native stop/drain/removal under the same owner before fresh OFF; no live
  hooks, API/method patches, row/source/early/nested modes or metadata owners;
  36 original Host protocol replies, unregister/detach, and the same Worker's
  ordinary post-unregister health token output.

No complete tensor arrays were saved. The reader therefore cannot independently
rehash full hidden/logits/Q/scale bytes or remeasure the changed-byte counts.
Those are source-bound runtime/readback receipts. It does rehash the original
JSON, required frozen source files and every Native build artifact, and it
recomputes serialized proof and receipt digests. External model weights and SDK
binaries are not rehashed. The final ordinary health call records token output,
not post-unregister full logits. No statistics comparison, E5 fault-only
localization, all-eligible qualification, or generic model/ABI qualification is
granted.

CPU contracts use the real public Host/Runtime and actual CPU producer/consumer
execution with the existing explicitly non-CANN Native ABI fixture. Negative
cases cover changed artifacts, missing denominator/helper, wrong owner/role,
old input/executor, changed proof request/epoch, same-shape consumer clone,
early prefix errors, incorrect flat domain, missing readback/drain, attempted
late restore, changed OFF output, uncertain RPC and retained resources.
