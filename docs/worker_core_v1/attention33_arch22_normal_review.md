# Attention33 fresh arch22 normal comparison

This is a new stock/private **normal-only** experiment. The preparation tool
has no fault-stage, prior bridge-certificate or statistics option. It does not
run a process, import torch, load a native library or access a device. The root
controller alone schedules the two cases, serially, with the same newly frozen
source, installed packages, observer, model and image.

The source73 build is at
`/home/17-fault-c2c3-20261117/tnd33-arch22-normal-build-a1/build/run`.
Its vendor is `stage/vendors/dcat_f33_arch22_normal_transformer`; its new public
schema is `evidence/arch22_schema/fia_abi_schema.json`. The tools rehash the
completion, controller, complete vendor artifacts, source-derived kernel entry
ABI, schema and schema compilation receipt. The recipe/header/contract must
still equal the newly frozen source. The old FAI-FD vendor or certificate is
rejected.

Freeze the new tools with source76 or later before preparing the cases. Set
`SOURCE`, `INSTALLED`, `OBSERVER` to the root-reviewed exact inputs below;
`OBSERVER` is the directory **containing** `attention-preflight`, not that
subdirectory. `SOURCE_MANIFEST_SHA` must be the frozen manifest digest.

```sh
R=/home/17-fault-c2c3-20261117
SOURCE="$R/source76"                  # or the actual newer frozen directory
INSTALLED=/absolute/path/to/the/matching/installed/root
OBSERVER=/absolute/path/to/the/fresh/observer/build
BUILD="$R/tnd33-arch22-normal-build-a1/build/run"
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/prepare_arch22_normal_bridge_cases.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --installed "$INSTALLED" --build-root "$BUILD" --observer "$OBSERVER" \
  --model /home/gyl/models/Qwen3-0.6B --card 2 --timeout 1800 \
  --out "$R/attention33-arch22-normal-cases-a1.json"
```

The output is two ordinary root-queue case records:
`attention33-arch22-normal-stock` and `attention33-arch22-normal-baseline`.
Each mounts the original observer receipt's absolute library directory as a
read-only alias to the same checked bytes. Only the baseline arm mounts the
new private vendor. Both retain the original function invocation, request
arguments and return objects, and capture the same first decode boundary.
The root controller supplies `/evidence` and runs the prepared records; this
tool does not start the queue.

After both cases finish, retain every original result and failure. Run the
reviewer on the host CPU, using the SHA256 identities recorded at run end:

```sh
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/review_arch22_normal_bridge.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --build-root "$BUILD" --completion-sha256 "$COMPLETION_SHA" \
  --observer "$OBSERVER" --observer-receipt-sha256 "$OBSERVER_RECEIPT_SHA" \
  --stock-run "$ATTEMPT/attention33-arch22-normal-stock/run" --stock-sha256 "$STOCK_RESULT_SHA" \
  --baseline-run "$ATTEMPT/attention33-arch22-normal-baseline/run" --baseline-sha256 "$PRIVATE_RESULT_SHA" \
  --out "$R/review-attention33-arch22-normal-a1.json"
```

The reviewer uses stdlib plus NumPy for bounded CPU interpretation of raw
floating-point bins. It reuses the existing complete-payload, logical-cache,
first-boundary, ELF and current-call runtime readers, without importing the
injector. It selects only pure decoder functions from the frozen verifier
source; no top-level code from that native module executes.

It independently checks five ordered phases per arm; all 87 complete payload
files per phase; each arm's four repeats; all 435 cross-arm payload pairs;
first Q bytes, immediate API outputs and later retained output equality;
actual mask/table/scalar/optional arguments; real writer/cache/consumer and
request joins; the 112 private Native normal calls; and zero owned scope,
hooks, API wrappers, environment and retained boundary references. Every
private final argument POD field and array element is decoded from the actual
runtime snapshot and compared with the host serializer. Actual registered
ELF bytes, generation, function binding, key, AIC/AIV symbols, block and
schedule are checked, including the private `healthy_after` request with no
Native policy owner. The last ordinary request after cleanup is tokens/text
only; its full logits are not captured.

Identical historical binds are resolved only by a unique current-call latest
effective bind; conflicts in handle, generation, key, owner or order refuse.
The original runtime gap stays in the report beside the new offline
resolution. All original failures remain verbatim. A missing first-boundary
proof does not prevent the separate complete-payload comparison from running.

Limitations stay explicit: stock selected ACLNN callsite owner is unobserved;
the stock POD is not decoded using the private ABI; the mask has a recorded
hash but no saved raw bin; logical KV before the API has a recorded hash,
compared with the complete saved matching **post-consumer** bin. Checkpoint
equality cannot exclude intervening writes. Declared omitted Python defaults
are separate from the actual decoded private Native values.

Exit 0 means `SOURCE_BOUND_EXACT_NORMAL_BRIDGE_RECORDED_NO_F_AUTHORITY` only.
Exit 2 preserves differences, missing evidence or failures. Even an exact
result has `fault_authorized=false`, `bridge_certificate=false`, C1/C2/C3
false and E5 `NOT_ASSESSED`. It cannot satisfy the old FD certificate reader.
The original 67-element normal output discrepancy remains an original fact;
these fresh cases neither alter it nor claim the new candidate fixed it
before actual execution and independent review.

The [source79 parameter review](attention33_arch22_source79_geometry_review.md)
corrects the expected nonquant sparse3 Native token pair to `(INTMAX, 0)`.
The actual attention parser and both headers are now hash-pinned under the
same build root. Python omitted defaults remain separately recorded as
`(INTMAX, INTMAX)`; this change grants no normal bridge or fault authority.

CPU contracts in `tests/test_arch22_normal_bridge_review.py` include a complete
two-arm file fixture, all five phases/435 payload comparisons/112 Native
scopes, and negative cases for old FD reuse, extra artifacts, raw array tails,
alignment bytes, owner/function conflicts, failed sync, pending cleanup,
late-output substitution, signed-zero differences and partial-run failures.
The fixture uses generated CPU bytes and grants no Native qualification.
