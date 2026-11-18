# Source51 MODEL_LOCAL 39.output_proxy independent review

`review39_output_source51.py` reads the preserved B/F JSON and mapped artifact
bytes with the Python standard library. It imports no model/runtime framework,
loads no shared library, runs no command or inference, and writes only a compact
JSON report to stdout. The caller can redirect that report to a new file. The
original results are never edited. Each input must match its pinned SHA256;
duplicate JSON keys, nonfinite JSON numbers and inputs above 768 MiB are refused.

Frozen source: `3ec12a7bc3bd5daebc404c70342083238e075a00`.
Manifest: `86878d43bc63dc1572b9d571c02def8a4061da66fd66153198748ac3e390866f`.
Native: `8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622`.
The reviewer verifies all 13 recorded implementation hashes plus the collector
binding helper, its serving-context dependency and the model capture helper.
It rehashes the Native **file**, without loading it. Explicit mappings let it
rehash weight/config/index and model source files. Unmapped container sources
remain individually unavailable. They are never resolved against ambient
installed packages. `build_fingerprint` is a joined reference; no separate build
receipt was supplied.

The following command is for the root coordinator to run on the remote CPU. It
does not create a container or touch a device. Use the new reviewer file from
the reviewed commit, outside the frozen source51 snapshot.

```sh
python3 -B /path/to/review39_output_source51.py \
  --baseline /home/17-fault-c2c3-20261117/postwrite39-repaired-source51-card2-a1/postwrite39-c2-baseline/result.json \
  --fault /home/17-fault-c2c3-20261117/postwrite39-repaired-source51-card2-a1/postwrite39-c2-fault/result.json \
  --source-root /home/17-fault-c2c3-20261117/source-a51 \
  --native-library /home/17-fault-c2c3-20261117/native-repair-builds-a1/postwrite/cache/c62b223b0359b9be007f4d2b6f1eb3436f31157960b8e853a5cca72e51ae67fa/attempt-7f5ea87745f042ab9e0aea361f4eb008/native/libdcat_postwrite.so \
  --path-map /model=/home/gyl/models/Qwen3-0.6B \
  > /path/to/new-review39-source51-a1.json
```

Default B SHA:
`e88e3f3f248a7f07e62e76a108c28a87a05342ba278de2031ed1a8e9f779cf01`.
Default F SHA:
`e9cc8a8c0875bc8136b76d86531c5e4b01aa9011da5b200e7865956d064465d0`.
`/source` is mapped automatically. Add `--path-map RECORDED_PREFIX=READ_ONLY_EXPORT`
only for an existing matching source export. The original files are about
316/320 MB; they are processed sequentially and only bounded issue examples,
counts, hashes and file references are emitted. Memory is several times JSON
size. No arrays or per-target records are copied to the report.

The reviewer independently reconstructs:

- Both admission-independent healthy inventories, every adapted projection
  (including unsupported calls), actual module calls and the explicit ModuleList
  structural noncall inventory. Exactly 140 eligible targets must remain.
- C1's actual first eligible producer, operation-specific contraction geometry,
  returned-storage identity, selected-element consumer views and native effect.
  Both B and F processes contain this **one C1 fault invocation**. Only B's C2
  window has 140 zero-effect controls and zero C2 fault hits.
- Every forward and MODEL_LOCAL invocation ID, complete call order, worker/model
  lifetime, owner/generation, candidate identities, bounded admission schedule,
  decisions/receipts and original calls. A model name or same tensor shape never
  joins distinct objects. These invocation IDs are not scheduler request IDs.
- Every target's lossless before/after hash, actual Native effect/readback,
  selected-element consumer storage/view relation, complete target output hash,
  and reference to the full model output hash. It checks execution, candidate,
  discovery, observation and per-target/worker recovery artifact hashes.
- OFF's full output equality, complete target set and no interventions, drained
  and released leases, zero owned-resource receipts, and final same-model OFF.
  B/F logical universes, fixed inputs, source/weights/build references and healthy
  complete model output hashes must agree across their distinct process objects.

`contract_satisfied` and hit totals are not verdict inputs. The script recomputes
coverage from the complete ledger. Exit 0 means C2 records and available pinned
bytes reconcile; exit 2 means inconsistent or incomplete C2 evidence. A refused
C3 does not invalidate a separately reconciled C2. C3 recovery is checked even
when its fault forward aborted and no complete fault output was recorded.

C1's cumulative `rejections` include unrelated ineligible candidates (on source51,
the oversized lm_head output). They are retained and reconciled with the actual
transport refusal reasons; they are not assumed to be failures of the selected
target. The selected producer/effect/consumer checks remain mandatory. The first
review's zero-candidate-refusal assumption was incorrect; its result is retained.

Limits remain explicit:

- Complete model/target tensors were hashed by frozen runtime code, but no tensor
  bins were preserved. Hash references and exact OFF relations can be reconciled;
  complete tensor bytes cannot be independently rehashed from these JSON files.
- C1 contains actual producer operand geometry; group producer binding relies on
  the frozen dispatcher observer and actual call metadata. Per-forward input
  stability is enforced by that source's fixed-input guard, with the stored C1
  input digest and workload scope. The raw input bytes are absent.
- Recovery is a reconciled source-bound receipt. It contains no separate final
  hook-table, dispatch-stack or Native owner-table readback. The review does not
  invent those observations or authorize resource release.
- Source51 snapshots C2 before issuing its live proof; `c2.live_c2_proof.issued`
  can therefore be empty. A C3 consumption receipt is checked against the actual
  C2 report/recovery/discovery hashes and same owner/scope. Missing issuance stays
  missing. Frozen live-object enforcement is described separately, without
  claiming that a missing receipt proves C3 did not execute.
- No public qualification is granted. This is **MODEL_LOCAL 39.output_proxy**,
  not standard Worker evidence, not `39.matmul_range`, and not E5. Capture dump
  statistics are reviewed separately by the root coordinator.

CPU contracts use real CPU dispatcher executions for controls, isolated faults,
successful C3 and a concurrency-refused C3. Negative artifacts exercise dropped
targets, wrong request/owner/generation, same-shape foreign storage, disjoint
consumer slices, bad producer geometry, changed OFF output, effect/hash mismatch,
nonzero resources, missing/mismatched live proof and changed source/Native/weight
files. They grant no NPU qualification.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:fault39/src:.:tests \
  python3 -B -m unittest tests.test_review39_output_source51 -v
```
