# Worker Core and remaining providers: new execution evidence

This is an incremental report for `codex/17-fault-c2c3-closure-20261117`.
[Source51 onward](WORKER_INCREMENT_20261118.md) records the completed
writer26 single-site run, output39 original-file/statistics reviews and quant
loader/observer fixes. Read that continuation for the latest state; older failed
attempts below remain evidence, not current queue status.
The existing six strict C2 and five strict C3 scopes are preserved. A new
runtime success requires its own independent review; this report does not
claim17/17, yellow-zone integration, CANN9.x Worker or A3 acceptance.

The public API is `WorkerFaultClient(HostAdapter(...), FaultConfig(id, variant))`.
Serving owns Worker construction and scheduling. The trusted deployment adapter
selects actual objects/calls and supplies the matching native package. End users
do not select layers, pointers, ranks or library paths. There are17 catalog routes,
11 RPC factories and6 separately implemented routes awaiting Worker RPC adapters.
See the [route and acceptance matrix](C2_C3_MATRIX.md), not catalog presence,
for what actually works on each model.

All runs below use one serial root queue on124 card2, the existing shared lock,
fresh live allocation gates, exact CANN8.5.1/A2 image and independent temporary
standard inference Workers. No Docker entry isolation was restored. Original
statistics/mix, arrays, logs and native binaries stay on124. Published content
is source, audits, summaries and SHA256 references only.

## New shared-path RoPE execution

Source50 `006916a9f9bd37549f0f21fc986ecace23e22e9b` ran the public Core22 probe
with the original Qwen3-0.6B workload and no model invocation inside fault RPCs.
All10 enqueued requests completed: four healthy qualifications, C2 once and OFF,
two C3 every-match requests and OFF, then post-cleanup same-Worker health.
The runtime records28/28 and56/56 hits,29 original RPC receipts, live C2 object
issuance/use and final `DETACHED`. The separate source-pinned reviewer checked
all774 frozen source files and9567 evidence conditions on124, with no issues.
Both C2 and C3 pass in this scope, including complete outputs, target/alias/call
joins, the exact original live capability's issuance/use, OFF and the tenth
same-Worker healthy request. It does not reconstruct live authority from JSON.
The original source46 missing-receipt result remains unchanged.

Independent review `core22-review-source50-a1/review.json` SHA256:
`8b23010fee90d3342977262f02f32002652d8531f175897a710658cd230acff2`.
See [the reproducible reviewer contract](worker_core_rope_review.md). Its32 CPU
tests also passed locally; they do not replace the actual-file review.

Original result:
`rope22-core-source50-card2-a1/natural-rope22-worker-core/probe/result.json`
SHA256 `6515fa39db4fc554649aed8265be3611db157bb0487d24661e89df3db20f8900`.
The source manifest SHA is
`c87d33a804b46f16ed44536a24863966fd08f0c99b2a21135a367b47d4c714cf`.
This route changes legal RoPE position-row selection, not integer overflow.
It records source-bound Triton calls, not a raw device-ABI proof or E5 statistics.
This particular probe imports the frozen source tree; it does not establish
installed-wheel inference by itself. Separate source50 package construction
and installed dependency smoke passed.

## Writer26: repaired consumer lifetime and remaining allocation conditions

The source48 attempt stopped in its first reference request after an intervening
ordinary QMM. Native state was poisoned while the selected writer was awaiting
its real consumer. Its failed recovery had lost the old residual's public
lifetime witness; it is preserved as failure, never relabelled as OFF.

The new shim only permits observing an unselected **QMM** while a completed
writer awaits its consumer. It still refuses a second selected lease, actual
overlap and FIA interleaving. The Python owner separately retains the exact
producer/consumer/writer tensors and allocations through failure/recovery,
checks current object/storage identity before writing and preserves the first
error plus both failed-callback and unwind receipts. No old failed pointer was
reused. Four actual-shim CPU contracts and the SDK ABI control passed in the
new device-free build; an independent Python review covered12 failure cases.

New bundle `native26-pending-consumer-a2/build.json`:
`b81b8b9bb1f0b5c0e56109775a987bf503b88a00adcfb873964a258e8bf4e03e`.
Its build receipt is
`614519e6bd9678cb61ebaa25e99e8ba33e95211aa658c6ccda0c21738bdf1b20`.
The source50 run now completes target000 reference, lossless and OFF with
identical complete output/residual hashes and an actual consumer. Target001
then fails the fresh allocation-adjacency check. The current effects plan
requires all resolved candidates to qualify first, so **no fault ran**.
Native pending/executing/device/host/descriptor counts and owned hooks/API/
tensor references are zero after cleanup. This is successful cleanup of a
failed qualification, not a completed B/F/OFF fault experiment.

`native26-effects-source50-card2-a1/providers26-v2-qmm-deepseek-effects/result.json`
SHA256 `c8ffee1ccea65a9e7b4b6fe39dcbd19d4115a741452eb15b9497962af21afab1`.
The next single-site exploration must retain the complete135-call inventory
and explicitly limit its scope; it must not weaken all-eligible admission.

## FIA26: descriptor format is still a real unresolved boundary

Source49's read-only SmolLM2 inventory observes60 actual API returns.
Every return has public ND format2 and native descriptor NCL format47.
The matched view is BF16 `[13,9,64]`, stride `[576,64,1]`, zero public offset,
contiguous; raw address/span match. Actual SDK `acl_base_rt.h` defines47 as
`ACL_FORMAT_NCL`; it is not a TND enum. Header SHA256:
`83e189e4bc63ad3e836b6c607f30152704b0c274e475e3b9d9be1689e91de07e`.

The inventory ABI does not independently carry complete native storage-shape
and offset facts. Shape/pointer matching cannot authorize treating NCL as ND.
Both Python and Native format gates remain strict; no fault or C1/C2/C3 was
granted. A future typed descriptor adapter must preserve the actual tag and
verify complete original/replacement descriptors before independent B/L/F/OFF.

`native26-fia-inventory-source49-card2-a1/providers26-v2-fia-smol-inventory/result.json`
SHA256 `e6a725725ff7664287645953e28b73f4792b8134f97b41e8962342ef1515d583`.

## Quant shared-scale: construction failure precedes registration

The new public single-site Core entry has fresh per-call row provenance and
retryable owned-observer cleanup. Source49/packages22 nevertheless failed while
vLLM constructed its healthy DeepSeek-V2-Lite-W8A8 Worker: DynamicQuantV2
returned161001 before binding, registration or fault requests. Zero experiment
phases completed; this is not an injected numerical effect.

`quant-c1-core-source49-card2-a1/providers-quant-group-deepseek-single-site-c1/result.json`
SHA256 `1c16800bd9d0f929006693ad8f5c3711a68e8004be41b13cb38b369f6a531169`.
The new optional failure diagnostic uses only an already mapped, exact-hash
DSO's `RTLD_NOLOAD` ABI/state/journal read exports. It performs no SDK resolve,
Native install, synchronization or recovery; original exceptions are retained.
161001 alone cannot distinguish a shim rejection from a vendor error.

Source51 retained the same native a4 library and workload and captured the
failure in the same process. The first public GWS entry was rejected while
the resolver was not ready: native prepares/executes were both zero, with no
installed owner or allocated group resources. Exact vendor `libopapi.so`
SHA256 `b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c`
defines both required DynamicQuantV2 functions but has no local definition of
`CanUsePTACache`. This static fact did not establish a null `dlsym` result;
the earlier missing-export root-cause attribution was too strong.
Diagnostic SHA256:
`00bcfd9d967823d94bcee297f16f474b713a83f62634fd9e55b6ccd388a301e9`.

The reviewed source52 repair separates mandatory execution APIs from optional
cache advice and checks symbols against their actual loaded ELF owner. Missing
cache advice returns false; self or dependency-owned execution symbols remain
rejected. Fresh `native-quant-group-v1-a5` passed the actual SDK build and seven
host-only real ELF loader cases, including global vendor preload, absent cache
advice, missing functions and wrong symbol owners. Build manifest SHA256:
`c9137e994eb29f76070c519f245aeeeb4c044a2b398aabd4b76ddde30ae78b71`.
These are CPU contracts, not new NPU or C1 acceptance. The new source52/a5
Worker still rejected its first GWS before vendor execution or registration;
result SHA `8ca8ae0f4a44b075662a54ec51fb80c8388cd798550feaad54c612697da867cd`.
The exact real-library CPU probe subsequently found the cache function in the
`libnnopbase.so` dependency. All required APIs resolved, but a5 rejected that
optional function's foreign ownership. The next repair ignores that optional
callback, returns false for unsupported cache consultation, and preserves exact
owner checks for the mandatory V2 functions. It requires a fresh SDK/NPU run.
See
[the loader repair and limits](providers_quant_loader_repair.md).

## Fault33: stable stock/private bridge mismatch, no fault admission

Both source48 stock and private-baseline runs independently repeat byte-exactly
across all five phases. Across stock/private, each phase has85 differing payloads
out of87; only layer0 complete logical K/V match. The earliest **recorded**
attention output difference is layer0:67/2048 FP16 elements differ,
maximum absolute difference `0.00006103515625`. Full final logits have146941
changed elements, maximum difference `0.021484375`; returned tokens agree.
These are auxiliary full-array comparisons, not a statistics/mix first-difference
report and not a fault effect. Later differences include propagation.

The independent native audit validates112 private-baseline launches, AIC20,
active blocks17, key5100000000010200103. Stock launch/profile ownership was not
recorded. Query values were not captured; existing K/V are post-consumer reads
and output references are read later. Thus current evidence does not uniquely
attribute the stock/private change to FIA arithmetic, nor prove equal complete
operator inputs. Removing scoped observers leaves the private library loaded;
that health phase is not a return to the stock library.

`attention33-baseline-diagnostic-a2.json` SHA256
`d4b863434444639548e957c471db704c9166a0a7c28bdfc4886b27849a673566`.
The original a1 numerical analysis is retained; its separate native audit
failed due to host namespace-package shadowing, corrected in a2 by verifying
the frozen source namespace. Neither revision changes raw evidence or the
exact-byte bridge gate. F remains blocked pending matched immediate input/output
and actual stock-execution evidence.

## Reproducible review and delivery

Use `tools/export_integration_bundle.py` to review, export and verify a source-only
bundle with one canonical `sources/` tree, eight section indexes and the three
existing distributions. No parallel runtime/lifecycle is introduced.
The yellow-zone CLI/catalog/controller implementation is still unavailable;
the package is a documented host integration contract, not a verified drop-in
replacement for that code. Fixed statistics tolerance remains baseline-reference
`rtol=1e-5`, `atol=1e-8`; nulls and alignment failures remain observable limits.
