# Public TND prefill stock launch census

This is the first implementation step of
[the TND source design](attention28_tnd_source_design.md). It observes the
ordinary loaded Worker through the existing public Attention `observe` and
`observe_repeat` protocol. It does not implement either FIA fault, change a
Native algorithm, force an arch22/private route, substitute FA or use the
fault33 Q1/page16 path. Native fault execution, C1/C2/C3 and E5 remain false.

The case preserves the actual source71 supplied API semantics: the original
`torch_npu.npu_fused_infer_attention_score` packet, FP16 TND Q
`[2048,16,128]`, K/V `[2048,8,128]`, int8 causal mask `[2048,2048]`,
lengths `[2048]`, sparse mode 3, API block size 128 and null block table.
The ordinary scheduler's cache block size remains a separate recorded value.
Nothing reshapes or replaces the original arguments. The original callable
is invoked once and its original return object is passed back unchanged.

Source71 did not capture omitted defaults. This census reads the actual
packet's default overload schema and records supplied attributes separately
from every omitted declaration, including `inner_precise`, `pre_tokens`,
`next_tokens` and `softmax_lse_flag`. It does not set `inner_precise=0` to
manufacture the predicted SplitFuse route. A schema declaration is explicitly
not an observation of the final Native typed attributes.

For every matching actual call, the read-only adapter records the actual
Q/K/V and mask shapes, strides, storage identities, object identities, offsets,
storage capacities, complete value hashes and finite checks, plus live
sequence/mask metadata and actual scheduler request binding. These reads occur
outside the observed API scope. Complete hashes are captured again after the
API. Raw QKV tensor bins are not retained by this census; the ordinary public
request receipts still contain full hidden/logits hashes.

The CANN observer is loaded before Torch/vLLM by an explicit parent `exec`
with `LD_PRELOAD`. It is only the existing observer DSO, not a fault provider
DSO or private vendor. Each call gets a new owned token, captured PID, Python
thread ident and OS Native TID. The row also records the real public
owner/generation, Worker/runner/model/module objects, rank, logical device and
request. A matching scope must contain actual begin/launch/end records in
that Native thread domain, successful completion and a current registration
and function binding with no conflicting handle/key/generation history.

The stdlib reader rehashes registered ELF content, reads its bounded ELF64
symbol tables, and joins the observed launch key to actual entry symbols. It
retains the raw runtime JSONL, original registered binary files, argument-byte
hashes and current stable-prefix hash. It never loads those ELF files or
decodes an unknown stock POD using a private ABI. Selected ACLNN library owner,
Native tensor pointer/formal binding and un-intercepted runtime entrypoints
remain unobserved. Missing registrations, symbols, failed syncs, conflicting
history or incomplete scopes stay `UNOBSERVED_OR_INCOMPLETE`.

All calls stay in the natural denominator, including mismatched layouts,
unknown signatures and unobserved Native edges. The probe requires two actual
requests with 28 calls each and reports observed and unobserved scopes
separately. `PUBLIC_ATTENTION_STOCK_CENSUS_PROTOCOL_RECORDED_PENDING_RAW_REVIEW`
means the protocol and cleanup were recorded; it does not assert 56 complete
Native scopes or grant a fault qualification. The existing BNSD fault input
validator remains unchanged, and an observer-only deployment cannot advance
to a Native baseline or F.

Cleanup retains pending drain/reference ownership as well as the actual TLS
scope. Stop failures keep the original first error. After-end exceptions use
the existing actual TLS readback, retain references until cleanup is confirmed
and permit retry by the same owner. Foreign token/thread or uncertain drain
cannot release a binding. Query only reads state and does not synchronize,
advance requests or create a forward. After unregister/detach, the probe uses
the same Worker for ordinary token-only health. It does not claim post-clean
full-logits observation.

## Files and root deployment

New production files are `attention28_stock_census.py` and
`attention_stock_launch.py`. The only shared production edits are
`attention_execution.py`, `serving_attention.py` and
`providers/attention_serving.py`. The existing natural probe's protocol driver
accepts an explicit observer-only cleanup expectation for the new dedicated
probe. Core Runtime, Host, registry, exporter and Native C++ are unchanged.

Root should freeze a new source and installed package after integrating this
commit. The source84 observer can be reused only when its actual source hash,
library bytes, CPU selftest and SDK-header receipts still match. `OBSERVER_ROOT`
below is the existing directory containing `attention-preflight/result.json`;
the preparer keeps that receipt's original read-only `/output/...` library
alias so it does not rewrite the receipt or guess a DSO path.

```sh
python3 -B "$SOURCE/experiments/c2c3_closure_20261117/prepare_attention28_stock_census.py" \
  --source "$SOURCE" --source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --installed "$INSTALLED" --observer "$OBSERVER_ROOT" \
  --model /home/gyl/models/Qwen3-0.6B --card 5 --timeout 1800 \
  --out "$FRESH_ATTEMPT/attention28-stock-census-cases.json"
```

The preparer reads and verifies source/package/observer bytes only. It emits
one case for root's existing serialized queue, with one physical card (2 or
5), TP1 eager, at most 8 CPUs, 48 GiB and 1800 seconds. It sets no private
`ASCEND_CUSTOM_OPP_PATH`. There is no F or statistics stage. It also supplies
the required actual deployment SoC label. The case retains:

- `probe/result.json`, `probe/host_receipt.json` and every public RPC JSON;
- `probe/native_calls/<actual-token>.json` for completed matching captures;
- `probe/runtime/runtime.jsonl` and original `registered-<pid>-<generation>.bin` files;
- two complete request/call rows, every unobserved reason, final cleanup and
  binding removal, then ordinary same-Worker token health.

The Native files and metadata originals stay in root's private evidence area,
not in the repository. No SSH, container operation, Native build or device
execution was performed to implement or test this increment.

CPU contracts cover real public Host/Runtime actions and 56 ordinary CPU API
calls using explicit fake Native/geometry seams; complete original calls,
query purity, baseline refusal and same-Worker post-clean use are verified.
Separate contracts read real synthetic raw JSONL/ELF files, reject stale or
conflicting identities and incomplete scopes, and verify raw input storage
metadata and declared defaults with actual CPU tensors. These tests establish
the observation/ownership contract only, never NPU qualification.
