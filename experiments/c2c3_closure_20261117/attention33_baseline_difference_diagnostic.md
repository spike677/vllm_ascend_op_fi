# Source48 read-only baseline difference diagnostic

Run the new diagnostic from a separate reviewed file, against the unchanged
source48, stock run, private B run and fresh build directory. It prints one JSON
summary to stdout. It does not create output files or change any bridge gate.
Redirect stdout to a fresh review artifact under the root's normal evidence
ownership policy. Only CPU NumPy and the original source48 validators are used;
framework imports, shared-library loads, subprocesses and network connections
are blocked during the diagnostic.

```sh
R=/home/17-fault-c2c3-20261117
python3 -B "$DIAGNOSTIC" \
  --source-root "$R/source-a48" \
  --stock-run "$R/attention33-fd-stock-baseline-source48-card2-a1/attention33-fd-natural-stock/run" \
  --baseline-run "$R/attention33-fd-baseline-mount-source48-card2-a1/attention33-fd-natural-baseline/run" \
  --build-root "$R/tnd33-fai-fd-build-a1/build/run"
```

Use the actual fresh build directory if its preserved path differs. The source
manifest must have base commit `78f40eab2f91bd8882b2f77ee69dd6f21b7cac09` and the
reviewed helper bytes must match that manifest. Source48's driver, contract and
bridge are unchanged relative to this diagnostic's base `46ebb682`.

Run in a fresh Python process. The diagnostic explicitly binds the three PEP
420 namespace parents `demoncat`, `demoncat.runtime` and
`demoncat.runtime.vllm_ascend` to the frozen `source/src` directories. A host's
installed regular package can otherwise override an earlier namespace path.
The original frozen `worker_extension` package and helpers are imported
unchanged; their bytes, including package/init/config/host dependencies, are
checked against the source manifest. An already imported `demoncat` package is
refused, not replaced. The original a1 diagnostic remains a separate artifact.

The original bridge's `hidden` error is a dictionary-order first mismatch, not
the first model layer. The same error text is used inside each run's repeated
phase checks. This diagnostic separates those cases, validates all raw tensor
hashes/finiteness, and compares all 435 cross-run payload pairs plus each run's
four repeats. It follows the actual audit's layer occurrence order and reports
the earliest recorded attention-output difference, changed layers, exact prior
outputs, shape/dtype/byte/hash identities, changed elements and maximum absolute
difference. The first differing K/V and the cache at the earliest differing
output are additionally split into the 1024 prefill rows and final decode row.
Signed-zero byte differences remain failures even if numerical difference is
zero. No values, arrays, raw launch arguments or statistics are emitted.

The private Native audit is run separately through the unchanged bridge's
`audit_native`: all 112 raw launch/host/binary/ABI/FD geometry joins and post-init
Norm/FIA library ownership must pass. Its outcome does not override a failed
output comparison. Private launch profiles are summarized by exact typed-data
hashes, key, block dimensions and schedule mode; the raw arrays remain on host.

Source48 does not capture stock Native launches or stock symbol ownership, nor
the values/hash of each attention query. Thus matching model/driver/contracts,
actual scheduler tokens, query geometry, logical K/V and previous attention
outputs cannot be upgraded to a complete input or kernel-causality proof.
K/V payloads are retained logical readbacks after consumers. The private process
keeps its globally loaded private vendor through `healthy_after`, which only
removes the scoped observers. These boundaries are explicit in the JSON.

The diagnostic never grants bridge acceptance, a fault run, C2 or C3. Actual
numeric findings require the root's CPU execution against the preserved files.

The root's preserved a1 artifact (SHA256
`4fd74349e14f83029035eaa104b9a4230a8df605c52107c5e4852b8b6d70ccff`)
reported exact equality for all four repeats within each run. Each of the five
cross-run comparisons had 85 differing payloads and two exact payloads. The
first recorded differing attention output was layer 0; that layer's complete
logical K/V payloads were exact, while layers 1–27 differed. Prompt tokens,
returned tokens, all 28 scheduler rows and query metadata agreed. This locates
a stable cross-run numerical divergence no later than the layer-0 attention
output. Later-layer differences are consistent with propagation; they do not
identify 27 separate root causes. Neither equal token IDs nor within-run
stability satisfies the complete-byte bridge. The original a1 Native audit
was not executed because its frozen namespace dependency could not import;
the import-only fix must produce a new artifact, preserving a1.

After the offline Native audit, the smallest new natural-call evidence that
can distinguish an upstream-input change from an attention implementation
change is the first decode's actual layer-0 API boundary in a matched stock/B
pair. Record the complete actual query (4096 bytes for this observed fp16
shape), the actual mask tensor's shape/dtype/full byte hash, all actual scalar
and optional arguments, exact used block-table values, and a complete logical
K/V hash immediately before that same original API call. Bind those captures
to the actual Worker/request/module/sequence and call token. Retain an exact
returned-output snapshot at that API's return, with a subsequent unchanged
readback check, so deferred copying cannot masquerade as the API result.
Any copies/synchronization belong to the new diagnostic pair and must be
identical in both arms; never substitute copied/repacked inputs into the call.
Call the original natural operator exactly once, with no policy or fault.

The same pair needs independent launch/registered-binary/API-library ownership
observations in both arms, including the stock arm currently missing them.
The existing private raw validator remains unchanged. A stock trace may use a
different kernel ABI or multiple launches; record that as an observed route
and do not impose the private key/ABI to manufacture equivalence. If query or
other inputs differ, follow that demonstrated input difference upstream. If
all complete effective inputs match and the immediate API outputs differ,
the divergence is localized to the stock/private attention execution; compare
their actual launch routes, binaries, precision/reduction options and source
build provenance. A successful private launch audit alone cannot establish
stock/private numerical equivalence. This is a new diagnostic scope, not an
authorization for F or a change to the exact-byte prerequisite.
