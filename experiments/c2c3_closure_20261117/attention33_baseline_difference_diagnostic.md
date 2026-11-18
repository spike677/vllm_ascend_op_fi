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
