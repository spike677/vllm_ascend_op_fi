# Public Core22 source-a50 independent review

`review_worker_core_rope_result.py` is a standard-library-only, offline reader.
It reads the complete original result, final Host history and all 29 original
RPC files on the evidence host. It imports no Torch, provider, vLLM or device
runtime and reads no tensor array files. JSON receipts cannot create a live
`C2Proof` or grant new execution authority.

The reviewed snapshot is fixed in the script:

| Artifact | SHA256 / commit |
| --- | --- |
| Source commit | `006916a9f9bd37549f0f21fc986ecace23e22e9b` |
| Source manifest | `c87d33a804b46f16ed44536a24863966fd08f0c99b2a21135a367b47d4c714cf` |
| Probe | `0e087f1d5996164d5d9c5f843d226354273666230853781302bd7e1f70873dfb` |
| Implementation | `999051f61294f6f3d196083c4591a31c3442ff2cca8e05de06d2b5852bde5537` |
| Original result | `6515fa39db4fc554649aed8265be3611db157bb0487d24661e89df3db20f8900` |

The manifest itself and every listed source byte are verified. Extra source
files, symlinks, path escapes, duplicate paths/JSON keys and nonfinite JSON are
refused. The implementation digest is independently recomputed from the
existing common/provider/group/binding source-hash chain; it is not adopted
from the result's success flag.

The complete reader, shared-alias, input/output byte digest, legal positions,
cache, outer/native/JIT call, target denominator, per-call owner/generation and
OFF recovery checks are reused from the unchanged source46 reviewer. That
file is itself SHA-pinned. The new entry loads a separate private instance
whose two deployment source constants are the exact source-a50 pins above.
The old file, its normal imported module, its CLI pins and its conclusion that
source46 lacks proof issuance/use evidence remain unchanged.

New checks join the complete original evidence:

- C2 has exactly one issuance and no consumption; C3 has exactly one consumption
  and no issuance. Both receipts bind the same actual process and Python object
  identity, full proof fields and proof digest. Scope, owner, source/consumer
  generations, session IDs and the entire candidate target set must match.
- The original C2 report hash is recomputed using its frozen CoverageSession
  schema, removing only `live_c2_proof` and the six known group decoration
  fields. Candidate/discovery/observation/recovery hashes are recomputed from
  the full raw rows and reports. Unknown fields are refused.
- All 29 original RPC files match their indexed file hashes and the final Host
  history. The exact attach/prepare/register/check/enable, ten advance/query
  pairs, finish/disable/unregister/detach order is required. Tenant, attachment,
  Worker, public owner, provider owner and generation remain connected. Every
  failure, uncertain reply, hidden recovery or missing phase is rejected.
- Each query's full returned row equals the original phase observation. C2
  issuance appears only after its OFF; C3 appears only after its two faults and
  OFF. Opening receipts join the same actual coverage sessions and denominator.
- Advance 10 proves the nine-request group cleanup before scheduling health.
  Query 10 proves the actual same-Worker input, complete hidden/logits bytes and
  sampled token equal baseline, and zero fault hooks. Its cleanup request list
  now contains all ten completed requests. Final owner release requires the
  exact successful unregister and detach receipts.

Deploy the new reviewer **outside** the frozen source directory, with the
unchanged `review_natural_rope_result.py` beside it. Run only against the
server-local original evidence:

```sh
python3 -I -B /review/review_worker_core_rope_result.py \
  /home/17-fault-c2c3-20261117/rope22-core-source50-card2-a1/natural-rope22-worker-core/probe/result.json \
  --source /home/17-fault-c2c3-20261117/source-a50 --expected-targets 28
```

`--expected-targets` is an optional count check. The denominator always comes
from every independently observed reader, including real shared aliases. The
script emits one compact JSON review and returns zero only when both C2 and
C3 evidence pass. The report separately states that no live capability was
reconstructed and no public qualification was granted. E5, device ABI payload,
integer-overflow semantics, yellow-zone framework adoption and CANN 9.x/A3
acceptance are outside this review.

CPU tests generate real Host/Runtime/torch hook receipts through ten fixture
enqueues and mutate original-object bindings, all proof hashes/scopes, RPC
order/owner/generation, full reader/call/byte records and cleanup/health rows.
They also check raw artifact hashing, source bytes and unmanifested code.
CPU evidence can pass content checks but can never pass the real-Worker
qualification fields:

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
python -B -m unittest -v test_worker_core_rope_result_review test_rope_result_review
```
