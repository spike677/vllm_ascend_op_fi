# Fresh S2 package: first-layer normal/OFF comparison

This entry runs the new `tnd_page16_arch22_s2_256` package only in
`arch22_s2_off` mode (S2=512). It does not select S2=256, accept an old bridge
certificate, or authorize F, C1, C2, C3, or E5. The source88 normal result is
not inherited. The Native recipe, headers, original SplitCore matrix driver,
and build controller from the preceding candidate commit are unchanged.

The root reported the source94 device-free build's original-source matrix
complete: 22,528 rows, receipt SHA
`bf1a6ad663feb40ce9a48bd3d2e5d2a3ba42cf39a02bc77529401dbff5b30682`,
matrix SHA
`04524c76acfa6b5bc53b3c424222fd309c75e36d65cbebd36484e4accecfcdde`.
The local receipt was checked against the exact frozen matrix program and
validator hashes. Root subsequently reported the host/kernel build exited
zero. Its downloaded completion SHA is
`860f27e5cd472e639e1a93f56f0e908aa2bb5f349bf93b771c9cde9d3c5f982f`;
the actual variant, both recipe hashes, both headers, matrix and validator
fields match this source. This agent did not compile the C++ program, read
the remote matrix/artifact bytes, or execute a device. The preparer must still
rehash those complete artifacts on the host, and a fresh real normal pair
remains required.

## Actual call and ownership

`paged33_arch22_s2_normal.py` reuses the existing normal-call verifier,
final-argument decoder, launch scope and complete-output probe. The two arms
use the same driver, installed code, observer and first-boundary capture.
This is the existing standard Worker `collective_rpc` deployment probe; it
does not complete the public Core fault entry or change its registry.

Only `model.layers.0.self_attn.attn`, nonquantized FP16 TND Q `[1,16,128]`,
actual KV length 1025, page16, Hq=16/Hkv=8/D=128, the reviewed int8 mask,
and the actual request/writer/page mapping enter the owned Native scope.
The same live-input helper in both arms retains the actual Worker/model,
module/context, Q/K/V views, caches, table and slot objects. It checks full
query/mask/table hashes and complete valid logical K/V before and after the
original invocation, preserves its exact return objects and records the
immediate outputs. Arguments and the model's forward call are not replaced.

The private adapter creates a fresh opaque owner, a Native owner token and
generation 1 for each probe session. The receipt joins those identities to
the actual request, first-layer object, Python thread identifier, OS native
TID, PID and independently observed begin/launch/end scope. The Native host
must report `arch22_s2_off`, the matching phase/target and live observer TLS
match. Final serialized arguments must agree with the actual host record,
all normal scalar checks, every SplitCore/FD array and workspace arithmetic.
No stock POD is decoded using the private ABI.

Each arm retains the full 28-layer natural denominator and five phases with
87 complete tensor payloads each. Only four first-layer private Native calls
are scoped: baseline-before, baseline-repeat, measured normal and OFF. Each
of these also records the other 27 original calls exactly once. The fifth
`healthy_after` phase uses the independent observer without a policy scope.
The private package remains loaded; this does not claim it was unloaded.
Both arms have five first-boundary launch scopes. Including the initial warm
request and the final ordinary request, each case performs seven requests.
The final ordinary request proves tokens/text recovery only; its full
logits are not captured.

If stop/drain fails, the adapter keeps the live input references and first
error. A retry must confirm its own Native scope and environment are clear
before releasing those references. Foreign threads are rejected. The
existing probe retains its owner when hook/API cleanup is uncertain.

## Root preparation and review

Freeze a new source and installed package containing this increment. Reuse
the completed source94 Native package only if the preparer confirms all its
recipe/header/matrix/schema/artifact hashes still match. Build inputs are
reconciled independently: the complete 22,528-row matrix is streamed and
rechecked, and original split-source, compiler-driver, logging-shim and
validator bytes are pinned. No compiler, Torch, injector, DSO or device is
loaded by the preparer.

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$NEW_SOURCE/experiments/c2c3_closure_20261117/prepare_arch22_s2_normal_bridge_cases.py" \
  --source "$NEW_SOURCE" --source-manifest-sha256 "$NEW_SOURCE_MANIFEST_SHA" \
  --installed "$NEW_INSTALLED" \
  --build-root "$R/tnd33-arch22-s2-build-a1/build/run" \
  --observer "$NEW_OBSERVER_ROOT" --model /home/gyl/models/Qwen3-0.6B \
  --card 5 --timeout 1800 --out "$R/NEW_ATTEMPT/cases.json"
```

`NEW_OBSERVER_ROOT` is the folder whose child `attention-preflight/result.json`
records the same freshly packaged observer. The generated cases preserve its
actual readonly build-path alias. Root must apply the already reviewed
same-byte production-library readonly mounts used by the accepted source88
pair, including the production aarch64 tiling/master libraries. The preparer
does not invent host staging paths. Mapped path/inode/device/hash failures
remain failures; they are never ignored or rewritten as successful capture.

The two case IDs are `attention33-arch22-s2-normal-stock` and
`attention33-arch22-s2-normal-baseline`. Execute them serially through the
existing root queue. Then use the same frozen reader with the explicit new
profile:

```sh
python3 -B "$NEW_SOURCE/experiments/c2c3_closure_20261117/review_arch22_normal_bridge.py" \
  --arch22-s2-normal \
  --source "$NEW_SOURCE" --source-manifest-sha256 "$NEW_SOURCE_MANIFEST_SHA" \
  --build-root "$R/tnd33-arch22-s2-build-a1/build/run" \
  --completion-sha256 "$S2_COMPLETION_SHA" \
  --observer "$NEW_OBSERVER_ROOT" --observer-receipt-sha256 "$NEW_OBSERVER_RECEIPT_SHA" \
  --stock-run "$R/NEW_ATTEMPT/attention33-arch22-s2-normal-stock/run" --stock-sha256 "$STOCK_RESULT_SHA" \
  --baseline-run "$R/NEW_ATTEMPT/attention33-arch22-s2-normal-baseline/run" --baseline-sha256 "$PRIVATE_RESULT_SHA" \
  --out "$R/NEW_REVIEW/review.json"
```

Cross-arm same-phase comparisons require complete raw first-input signatures
and all 435 payload pairs to be exact. Within-arm repetition reports raw
input equality separately and permits only independently proven physical
page relocation with complete logical K/V equality. Unknown differences,
missing calls, failed originals, unclosed scopes or retained owned resources
are refused. Query and immediate-output bins are rehashed; mask and pre-call
logical K/V remain hash-only, with later matching complete logical K/V bins
independently rehashed. Selected stock ACLNN owner remains unobserved.

## Source closure and CPU contracts

The existing required-source closure is retained and extended by `S2_REQUIRED`
in `arch22_normal_bridge_evidence.py`: the new preparer/runtime/build-evidence
helper, `paged33_logical_binding.py`, the S2 and normal recipe/headers,
geometry validator, matrix compiler, build controller, and existing build/
registration/base-policy modules. Required installed files must be the same
bytes as the frozen source. No raw remote result, SDK source, tensor bin,
matrix output, ELF or library is added to Git.

CPU contracts cover identical stock/private capture, original-call and return
identity, owner/thread/phase rejection, full-array and workspace mismatches,
partial-stop retry, all 27 non-target returns, missing matrix rows despite
success flags, source pins, distinct no-F case manifests and cold stdlib-only
preparer initialization. The actual probe `finish` path is also exercised
with a real CPU Torch hook: failed Native stop retains the Worker owner and
hook, retry removes both after zero scope, and the first error remains in
the receipt. The shared regression ran 145 tests with no skips; the final
13-test new-entry pass includes two added cleanup contracts. These are
fixtures and do not grant real Native or
Worker qualification. No shared Runtime, Host, registry or exporter changes
are part of this increment.
