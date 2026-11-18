# Source93 routed-call metadata review

All 52 actual MoE calls have a complete independently reconciled Python
frame/object/scalar/epoch chain. This does **not** prove the Native operations'
row semantics or qualify another projection. Each healthy forward still has
10 row-proven projections and 125 unresolved projections out of 135. Only the
first of its 26 MoE entries already has request-row provenance; the other 25
retain `NO_COMPLETE_ACTUAL_ROW_PRESERVING_OPERATOR_CHAIN`.

The frozen Worker source is `62c8cc43aa02edb55f85c0761fa8a7b5f47280de`, manifest
`8859466a881b322895a44919335470cd1aaeec2325d34f0d325e84103e811c75`.
The reader checked all 14 installed source-file hashes against those frozen
Git bytes. It did not import Torch, execute a model, contact a server, or
alter the observer. Native qualifications and C1/C2/C3 remain false.

| Original | SHA256 |
| --- | --- |
| Uncompressed inventory, 86,724,790 bytes | `3f639448f70fa200d5038dc50b7933904ff11093d4e5796c7051a574c97b1489` |
| Compressed inventory | `cff49dc89725c22845fe6d8631f49302c239c1e5da6e6efdbd5cfee4ef7f9235` |
| Final result | `d7331f8e734fc32647df195c6187b14f30315cfb0e0187b45471185346e7959b` |

The committed JSON summary contains hashes, counts and refusal diagnostics;
it contains no activation values, integer-route arrays or raw tensor
descriptors. Full originals remain outside Git.
Its SHA256 is `9fd8a3ea4a89a2e55bf50067a453964b1717997cf4d984b84d7e5b9aed1cde8b`.
The reader, observer regression and prior source87 audit suites passed 59 CPU
tests, including 29 reader contracts. These tests establish audit/refusal
behavior only; source93 supplies the actual observation evidence.

## Independent reconciliation

`review_quant_routed_frames.py` requires explicit input hashes and the frozen
installed source root. It does not authorize from `python_chain_complete`,
`completed`, or qualification flags. Its positive tests deliberately turn
completion flags off while retaining the same complete raw evidence.

For every call it checks the actual model owner and source-frame ancestry,
13 exact frame roles and entry order, source hashes, all seven actual operator
stages, current parent/forward/request references, and the integer record's
content hash. It independently recomputes integer counts/permutation relations,
then joins each snapshot to its producer, epoch and separately classified
readback helper. All five helpers and their original dispatches are retained.

The typed prepare, dispatch, builder, MLP and combine payloads must refer to
the same recorded objects and fields. GMM operands, weights, scales, count
versus cumulative group lists, optional arguments, returns and the unpermute
inputs are joined to the actual operator records. Finalize must return the
same recorded routed object. Matching shape or storage address alone cannot
replace those joins.

The reader replays the complete raw write-epoch ledger, including alias
invalidations. It follows actual `aten.alias` returns to the observed `mul_`
and `add_`, checks the current configuration owner, actual scalar and alpha,
the same shared operand, one original call, unchanged extent and exactly one
write epoch. The final actual view must produce the recorded model return.
An old storage address is never used to invent a current producer. Sparse
epoch histories avoid copying the full allocation map for every event.

The quant denominator is also recomputed from originals: each of the 270
model parents has its own prepare-enter/return and execute-enter/return,
one current registration and public owner, matching Native/public descriptors,
executor, workspace and stream. All 540 public quant/QMM calls remain joined
to the 135 projections in each forward. Summary counters cannot fill a missing
event. This checks the existing healthy pass-through evidence, not a new
Native lease or row qualification.

| Observation | Forward 1 | Forward 2 |
| --- | ---: | ---: |
| Model parents / independent complete Python chains | 26 / 26 | 26 / 26 |
| Actual dispatch events / unresolved graph edges | 12,853 / 12,783 | 12,853 / 12,783 |
| Integer snapshot bytes / separate helper dispatches | 41,600 / 390 | 41,600 / 390 |
| Frame callbacks | 27,440 | 27,440 |
| Charged frame metadata bytes, total | 1,310,662 | 1,308,845 |
| Charged bytes per parent, min–max | 47,599–53,254 | 47,599–53,140 |
| Frame callback body time | 1.837420360 s | 1.837350964 s |

The total metadata is for 26 parents; every parent stays below its 512 KiB
bound. The callback timing excludes shared profile/graph overhead and overlaps
other timing categories. It is not an observer-free serving latency. Both
requests finish with all owned hooks, modes, profile callbacks, tensor/payload
holds and metadata references zero; healthy-before and healthy-after match.
Actual scaling is the Python scalar `1.0` in all 52 calls, with actual add
alpha `1`. These scalar observations do not certify GMM/unpermute semantics.

## Source87 review and remaining Native evidence

Before source93, the pinned production source already established the entry
order prepare → dispatch → builder → MLP → combine → finalize; the shared
result was then returned with the routed result, followed by actual aliases,
scalar multiply, add and view. Source87 recorded those latter object links,
but did not record the typed payloads or actual post-route scalar arguments.
Those were the main untested boundaries of observer `0aadb777`; source93 now
supplies the missing Python observations for all 52 calls. No source87 pins or
originals changed, and its reader must still use its own frozen source root.

The remaining gates are the actual loaded implementations of routing, GMM1,
GMM2 and unpermute, and their grouping/row-local/reduction semantics. Python
delegates, schemas and `compiler_depend.ts` registration strings do not identify
the executed DSO. The separately supplied disk-only material bundle
`9d7495ed47e1906efdbe8d332cb10899d1877e797165573f78cec002d90d75cd`
contains three candidate ELFs and 50 hash-checked header/source texts. It was
not loaded or executed. The stock `libopapi.so` exports base/V2 GMM1,
base/V2–V5 GMM and base unpermute pairs; headers cannot tell us which pair the
current torch_npu wrapper actually resolves. Existing custom WeightNZ source
does not stand in for the observed stock GMM operation.

The minimum next step is a narrow read-only observation of those four calls,
after reviewing their actual resolver closure:

1. Obtain the actual torch_npu wrapper/registration and `EXEC_NPU_CMD` /
   `GetOpApiFunc` resolver sources, including cache behavior. Find the current
   vllm custom extension and vendor op_api ELF through the production loading
   code. Keep disk candidates separate from loaded/executed identity.
2. Before Worker initialization, install a bounded observer which records the
   actual resolved function pointer and its mapped ELF owner, build ID and
   file hash. A same-name lookup, environment variable or a successful dlsym
   alone is insufficient. A cached or explicit-handle path that bypasses the
   observer must be reported missing, not inferred as observed.
3. At the actual Python stage, bind a thread-scoped read-only parent ticket to
   the real GWS/execute calls: Worker lifetime, thread/device, exact descriptor
   and integer scalar arguments, fresh executor registration, stream, required
   and supplied workspace, return status and original-call count. Correlate
   completion with that registration, not an executor address reused later.
   Preserve unmatched calls, errors and overflow. Run each original once,
   retaining all descriptors only for their valid call lifetime and releasing
   getter-owned arrays on all paths.
4. Only after loaded ownership is established, review the implementation
   actually used for cumulative versus count grouping, row-local grouped
   MLP/SwiGLU/quantization, and inverse-index/probability reduction. No floating
   activation export or reconstructed numerical oracle is needed for this
   metadata step. Unproven Native semantics continue to block graph binding.

This is an observation proposal, not an implemented Native adapter or an NPU
authorization. It reuses the existing per-stage observer rather than creating
another provider/admission state machine.

## Reproduction

The reader and adjacent `audit_quant_source87_routes.py` require only Python's
standard library. `--source-root` must name source93, not the current mutable
Worker checkout or frozen source87. The paths below are placeholders for the
root-controlled original and frozen snapshot locations:

```text
python -B <review-source>/experiments/worker_core_v11_20261118/review_quant_routed_frames.py --inventory <source93-inventory.json.gz> --inventory-sha256 3f639448f70fa200d5038dc50b7933904ff11093d4e5796c7051a574c97b1489 --result <source93-result.json> --result-sha256 d7331f8e734fc32647df195c6187b14f30315cfb0e0187b45471185346e7959b --source-root <frozen-source93-root> --out <new-summary.json>
```

The output path is exclusive; an existing summary is never overwritten. Missing
raw links remain explicit incomplete records. Artifact/hash/denominator errors
produce an audit-failure receipt and exit 2. A successful raw audit returns no
Native or row authority, even when every Python chain is complete.
