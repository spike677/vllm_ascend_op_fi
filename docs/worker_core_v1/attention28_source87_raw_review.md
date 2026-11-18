# Source87 TND stock census: original gap and offline resolution

This increment adds a stdlib-only reader and CPU contracts. It changes no
provider, observer, Native algorithm, registry or qualification gate. It does
not run SSH, a container, a build or a device. F, C1/C2/C3 and E5 remain ungranted.

The source87 result has SHA256
`3b7143f1b3873211214a79e3d22afc21a86e80b8a86409496bfe6e15dfba3340`.
Its two actual requests each retain 28 API calls. All 56 native-census records
retain `ValueError('CURRENT_SCOPE_FUNCTION_BIND_NOT_UNIQUE')`; the original
observed count is zero. This is a reader coverage failure, not evidence that
the stock Native operator did not execute. None of the old fields is rewritten.

## What the preserved bytes establish

The actual runtime SHA256 is
`9bcba3ff4be32243ad6fbfa90a731ae1e0e6dd8aed67ba49e3ff31e3ef39a635`.
Each of the 56 scopes contains one begin, two launches, one function bind,
two successful device syncs and one end, all in PID/native-TID 1. Python ident
is recorded separately. The two launch profiles are:

| Actual entry | Key | Registration | Block / schedule | Argument bytes | Binding |
| --- | --- | --- | --- | --- | --- |
| `Slice_cf62034947577763b2b546efd5444507_high_performance_524000000` | `524000000` | generation 28 | 40 / 0 | 208 | Cached, last effective bind before this scope |
| `FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347_5000000000000200103_mix_aic/aiv` | `5000000000000200103` | generation 327 | 20 / 0 | 600 | Fresh bind inside this scope |

The registered files are original bytes, not loaded by the reader:

- `registered-1-28.bin`: 2,363,808 bytes,
  SHA256 `ceadb83454273d7a9dd67c343c2dd7ca0f756ff0f44340d42126ec584a36b63d`.
- `registered-1-327.bin`: 20,097,744 bytes,
  SHA256 `c182034ed30e94e277feeb4ae3ac859027758efc49f5418dcbaabecea236f11a`.

The Slice function is a 1,940-byte global STT_FUNC in `.text`; its complete
body SHA256 is `cb0cd7e627aaf03cad49e1364c23ccb4de32397b3a0c3e412086a1cd3dda04b8`.
The FIA aic/aiv function bodies are 10,204 / 20,852 bytes with SHA256
`2ed3c7fa9ef012e8f9442214eb895199f0c0314e00f38795edd9b4d99e88509e` /
`46947fdab9c951fdf4872d3ac59d5fb53f06417414a76d115e0cbf3e04996a62`.
Matching four-byte weak STT_OBJECT companions are retained separately. Their
presence cannot invalidate a defined callable entry or make them executable.
The former Attention-name filter also missed the actual Slice symbols entirely.

The local CPU check reconciled all 112 launches: 56 cached and 56 fresh binds,
without dropping the Slice launch. That local check used result/runtime and
the two registered ELFs only. Full Host/RPC and per-call file verification is
a separate reader run over the original directory; no complete remote review
is claimed here.

All 56 API input before/after records independently satisfy the declared TND
geometry and causal mask hash. In the actual first call, Q and K are contiguous;
V has stride `[4096,128,1]`, offset 3072 and 16,777,216-byte backing storage.
The reader preserves that view. Slice input/output pointer ownership is not
decoded from an unknown ABI, so the V view is not asserted to be its operand.
Actual omitted declarations are `inner_precise=0`, `pre_tokens=next_tokens=INTMAX`,
and `softmax_lse_flag=false`; these remain declarations, not final Native POD.

## Independent read rules

The new offline resolver requires the actual launch's latest registered
handle/generation and complete successful function-bind history to agree.
The unique latest effective bind must precede that launch in raw order and
time, with the same PID/native-TID. A fresh bind must belong to this scope.
A cached bind must precede its begin; if it came from an earlier named scope,
that scope must have ended before the current begin. Failed or conflicting
history, stale registration, tied timestamps, foreign thread, incomplete
capture or failed completion rejects the join. Cached binding is explicitly
labeled; it is never relabeled as a new bind or a fault-admission receipt.

The reader inspects each actual key in the full ELF symbol table, including
non-Attention kernels. Callable entries must be global/weak STT_FUNC with
nonempty bodies entirely inside an executable PROGBITS section. It records
complete body hashes and non-function companions. Internal names elsewhere
in the same ELF are marked as co-resident evidence, not proof that an entry
reaches that internal algorithm. There is no expected-key allowlist, guessed
stock ABI, private POD decode or selected ACLNN resolver-owner assertion.

The complete CLI also checks all 15 original public messages and Host history,
actual request mappings, all 56 call artifacts, owner/generation/Worker/model
identity, raw-prefix hashes where captured, original call once, physical QKV
storage extents, exact before/after inputs and repeated full model hashes.
It recomputes the full int8 causal mask hash. Cleanup must show zero hooks,
wrappers, leases, scopes, pending drains and references, then unregister,
detach, binding removal and a new same-Worker token-only health request.
QKV/model arrays were not saved; their equality remains source-bound full-hash
evidence. The reader grants no fault or numerical normal-bridge qualification.

## Root-only CPU command

Use the new reviewer files from the reviewed checkout. `SOURCE87` is the
unchanged frozen source directory, and `P87/cann851` is the existing observer
root containing `attention-preflight/result.json`. No new observer build or
model execution is needed.

```sh
python3 -B "$REVIEWER_SOURCE/experiments/c2c3_closure_20261117/review_attention28_stock_census.py" \
  --root "$R/v11-observations-source87-card5-a1/attention28-tnd-stock-native-census/probe" \
  --source "$R/source87" \
  --source-manifest-sha256 4624e735fc244e5133e712db2695bfc6d34ad1196e1905738f24be36f0ad092b \
  --observer "$P87/cann851" \
  --result-sha256 3b7143f1b3873211214a79e3d22afc21a86e80b8a86409496bfe6e15dfba3340 \
  --runtime-sha256 9bcba3ff4be32243ad6fbfa90a731ae1e0e6dd8aed67ba49e3ff31e3ef39a635 \
  --registered-elf-sha256 registered-1-28.bin=ceadb83454273d7a9dd67c343c2dd7ca0f756ff0f44340d42126ec584a36b63d \
  --registered-elf-sha256 registered-1-327.bin=c182034ed30e94e277feeb4ae3ac859027758efc49f5418dcbaabecea236f11a \
  --out "$FRESH_REVIEW/review.json"
```

The new report preserves the 56 original errors and labels its new offline
resolution separately. Existing raw data and source87 remain unchanged.

The first full-artifact audit, run with the source90 reviewer, retained report
SHA `11e87456c49638746de48bed0209388d200bb94bf4c5723f5232b2bdaf43b3cd`.
Its only failure was a reader protocol error: `collect_replay()` is a Host
convenience method that sends `advance` with `settings={"collect": true}`.
The original history contains 15 actions, including two
`advance / query / advance` triplets. The corrected reader checks that wire
sequence, original action replies and request-state transitions. It does not
invent settings absent from the retained history. A CPU regression runs the
real `HostAdapter` and `WorkerFaultClient` against a serialized Worker endpoint,
verifies both actual collect payloads and reads the resulting original Host
history. Convenience method names, pending collect results, premature evidence,
foreign tickets and changed query evidence are rejected. Re-run into a fresh
report; neither a1 nor any original artifact is rewritten.

## Bounded next adaptation evidence

The actual key is now measured rather than predicted. The previously supplied
build source's `fused_infer_attention_score_tiling.cpp:41` registers that key
with `FAInferTilingData`, which is not the arch22 public serializer type. This
source association does not establish that a stock ELF was compiled from that
exact tree. Do not apply the arch22 decoder or the source88 Q1/page16 normal
certificate to TND2048. The registered entry's matching production kernel
metadata and `fused_infer_attention_score_tilingkey.h` are the next narrow source
join, followed by `flash_attention_infer_tiling.h`, `flash_attention_interface.cpp`
and its directly named recurrence header. This identifies the actual SplitFuse
candidate path without substituting a differently compiled normal algorithm.

Source88's separately reviewed 33 normal pair passed its 281 checks; this
28-specific cached auxiliary launch does not overturn that result or authorize
33 F. Public F/OFF wiring still requires its own live, scoped bridge and Native
variant proof.
