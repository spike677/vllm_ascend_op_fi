# Fault17 complete-original audit

`review17_worker_originals.py` reviews the complete public Worker result from
the current implementation, starting from integration
`94b4cefd67cd5c55b80699ec4a413215b2f9a7a0`. It does not import the injection
provider, load libraries, create a model, contact a server or call an NPU.

The CLI requires the complete original's SHA256 and the actual frozen source
manifest SHA256/commit. An `extraction_only` sample is explicitly refused,
including a sample carrying the real original's hash in its metadata. The
probe's recorded source hash and all 13 mechanism source identities are
rehashed against the frozen source files; the public driver, cache helper and
Core dependencies are checked against that same source manifest.

The review independently checks:

- All ten request/encoder phases, actual image/prompt identity, upper original
  Conv3d arguments, lower final arguments and unchanged input/weight/bias hashes.
- Padding as the sole changed argument, the returned Conv3d object's lineage
  through actual aliases/copies and the exact-object first numerical consumer.
- Four healthy controls, three F effects, repeated F, both OFFs and final
  cleanup encoder/hidden/logits identities; returned first-token/text changes
  are reported separately.
- The full one-projection candidate, each original C2/C3 call and canonical
  effect/recovery reference, live-proof issuance/consumption identity and next
  generation, zero held resources, and all 28 host-recorded RPC actions through
  unregister, detach and binding removal.

Saved status, hit and `logits_changed` summaries are not acceptance oracles.
The output reports recalculated consistency and facts only. `c2`, `c3`, `e5`,
`public_qualification_granted`, `native_qualification`,
`aclnn_launch_arguments`, `device_payload_observed` and
`multi_target_verified` remain false. The actual scope is a single Torch
parameter boundary, with one live projection. No ACLNN/device payload,
multi-target, continuous-generation or E5 claim follows from this audit.

The result's RPC history is embedded host evidence, not a separately captured
transport trace. Tensor identities are source-bound runtime hashes: no private
tensor is downloaded/read, and argmax is not recomputed. Framework code and
model-weight identities are retained from the originals; this auditor does
not reopen the installed image's framework files or model weights. Source
disk hashes do not by themselves prove loaded-module identity. These limits
are also included in every complete review result.

Run on124 over its retained originals (replace the placeholders with the
actual source77 pins; a later source snapshot is not a substitute):

```sh
python experiments/worker_core_v11_20261118/review17_worker_originals.py \
  --result "$ATTEMPT/probe/result.json" \
  --result-sha256 36f17fa31ca8b4b3a04e114baaacdc1b54baefcb162129e1b41b408d586a361a \
  --source-root "$ACTUAL_SOURCE77" \
  --source-commit "$ACTUAL_SOURCE77_COMMIT" \
  --source-manifest-sha256 "$ACTUAL_SOURCE77_MANIFEST_SHA"
```

Local validation: 36 CPU tests passed across the new original-audit tests,
current public Vision17 group provider contracts and cache-control contracts.
The new fixture actually executes the current public CPU provider's ten
phases; it is explicitly a CPU runner fixture, not a standard NPU Worker.
Negative cases cover changed final/upper arguments, input/weight/bias changes,
fake object lineage, missing encoder execution, F/OFF/cleanup mismatch, request
or proof identity drift, missing ledger entries, retained resources, extracts,
wrong original/source hashes and incomplete cleanup. Summary output omits the
large repeated RPC bodies. No actual124 original was opened by this change;
the root controller must run the CLI against the full retained original.
