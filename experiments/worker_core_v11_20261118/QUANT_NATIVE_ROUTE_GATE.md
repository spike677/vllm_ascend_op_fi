# Quant routed request rows: Native container gate

The proposed GWS-only proxy is **blocked and not implemented**. The actual
CANN 8.5.1 public `acl_meta.h` makes `aclTensorList` and `aclIntArray` opaque.
It exposes `aclGetTensorListSize` and `aclGetIntArraySize`, but no member/value
getter. A count cannot identify GMM2 input, weight, scale and output members or
the routing active-expert range. Reading an assumed private struct, or filling
those members from Python arguments, would not establish the actual Native
call. No such access has been added.

The previous source93 observation remains **10/135 request-row-proven
projections per forward, 125 unknown**. This source review does not remeasure
that result. It issues no new row, Native, C1, C2 or C3 qualification. It runs
zero operators and zero faults; the existing model/observer code is unchanged.

## Recomputed production facts

The accompanying stdlib reader checks four immutable original bundles, every
embedded source SHA, five exact ELF identities, defined and unversioned API
symbols, the actual dependencies and the public declarations. It reproduces
`BLOCKED_OPAQUE_CONTAINER_MEMBER_IDENTITY`. Exit zero means that this blocker
was reproduced. It does not mean the proposed proxy passed.

| Evidence | Exact identity / finding |
|---|---|
| Production image | `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0` |
| Public metadata header | `be93460abfc7aa14113d7b9a3536fe2dd45d5148a357f55e15847655c9d5ffed`; opaque containers, size getters only |
| Stock opapi | `b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c` |
| Custom opapi | `de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f`; depends on `libopapi.so` |
| Actual custom extension | `e63806486e0311c2df8ce9b120ae98f55e8c8eb91ccaf204ef0cf7d6b469ed67`; also depends on `libopapi.so` |
| Stock torch_npu | `dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041`; exact upstream commit unavailable |
| Custom resolver source | `142023f0f0e682b3b79aeb987044ecbd9b7db0b63b20de967447ba6f9e3d6b16` |

These are disk identities, not loaded/executed-owner evidence. The custom
resolver explicitly opens the custom handle before the stock handle and
caches addresses. Its real conversion code invokes `aclCreateTensorList` at
lines 353–365 and `aclCreateIntArray` at lines 323–329. Their original creator
arguments are observable while that stack is live; the corresponding public
destroy APIs also exist. This establishes a possible **custom** interception
point. It does not prove that the stock torch_npu GMM wrapper resolves or calls
those APIs through the same path. The existing `pta_resolver.py` remains an
exact-wheel resolver-shaped diagnostic, not a proof of a particular GMM call.

Disk exports and headers narrow potential boundaries to custom routing,
GMM1/base, GMM2/V4 or V5, and unpermute/base. They do not select the version used
by the natural request. In particular, GMM2's documented M-axis grouping and
count/cumulative modes cannot be applied until the actual current tensors,
group list, flags, implementation owner and executor are joined.

## One next diagnostic hypothesis

This is a proposed diagnostic, **not code, an accepted build, or a queued case**.
Its sole question is whether the exact stock wrapper's real container creator
and destroy calls can be observed without changing the operator computation.

1. Reuse the reviewed private custom-opapi proxy packaging: a SONAME-renamed
   original dependency, exact file/loaded executable-segment identities, and
   `RTLD_NOLOAD` resolution of the already-loaded real `libnnopbase.so`.
   Export only the four typed functions `aclCreateTensorList`,
   `aclDestroyTensorList`, `aclCreateIntArray`, `aclDestroyIntArray`. Verify
   them against the actual SDK declarations. Do not intercept global `dlsym`,
   guess a struct, disable PTA caching, or alter the existing quant group DSO.
2. Install the private library before Worker startup so cached lookup cannot
   silently retain an earlier function. Preserve the exact stock functions,
   arguments, return pointers/status and one original call. Use the inspected
   stock search order and custom resolver only to propose placement. Require
   **actual typed entry counters and caller PCs** to show which placement
   worked; a successful `dlsym` lookup alone is insufficient.
3. Observe one bounded healthy B8 first-decode request. Existing public/module
   inventory retains all 135 projections. Native raw entry/return records
   precede source-frame classification; unclassified calls and failed calls
   remain in the record. Attach current process, thread, source operator
   sequence and scope timing only when actually available. A different or
   late execution thread is an explicit gap, not a guessed parent join.
4. At a real creator entry copy only bounded host integer values or the actual
   descriptor-pointer array and SDK tensor metadata. Record actual returned
   container pointer, fresh registration, original status and precise real
   function/caller image offsets. The SDK getter arrays must be released on
   every path. No activation readback or extra numerical operator is needed.
   A successful destroy retires that exact registration; failed, duplicate,
   missing and reused-address lifetimes remain distinguishable. No later
   dereference of saved descriptor pointers is permitted.
5. Bound this diagnostic at one decode, 4096 Native function invocations,
   8192 entry/return records, 128 members/integers per creator, and 8 MiB copied
   metadata. These are proposed caps to verify before deployment, not measured
   costs. Overflow is explicit and invalidates the diagnostic; it never drops
   an inconvenient row. Preserve raw counts, timings and cleanup even on
   refusal. The root's existing 1800-second case limit stays unchanged.

**Proceed condition:** actual creator and destroy entries from the current
stock implementation are observed, exact function/caller owners are verified,
all scoped objects have distinct registration lifetimes, no record is missing
or truncated, both healthy output and cleanup remain unchanged. This only
permits designing the subsequent GWS/executor join; it is still not routed-row
authority. Four functions alone do not prove that list members survive to GWS,
that child descriptors are unmodified, or that an executor uses those members.

**Stop condition:** missing typed entries; a cached or direct lookup bypass;
foreign/ambiguous implementation; pre-existing unobserved containers; a
creator outside the current provable source scope; missing/failed destroy;
address reuse without a fresh registration; overflow; or changed healthy
output. Stop before any fault, preserve the entire candidate denominator and
the original refusal. Do not respond by changing the Native algorithm or
adding a synthetic call to manufacture a trace.

Only after this diagnostic succeeds can a narrowly reviewed subsequent join
connect actual creator registration → original GWS arguments → actual executor
registration → execute/stream, with full current-object/epoch evidence. That
join must prove the existing token permutation, GMM row-local grouping and
unpermute inverse/probability reduction apply to the same tensors. Once the
routed output obtains request rows, normal row propagation can reach later
MoE inputs. Actual two-pass full discovery and per-target reference/lossless/OFF
then gate public C2; C3 requires the newly issued original live C2 object in the
same Worker/owner/scope with the next generation. None of those later steps is
claimed here.

## Reproduction

Keep the four original bundles outside Git. `--out` must be a new file:

```text
python -B experiments/worker_core_v11_20261118/review_quant_native_route_materials.py \
  --sdk-operators <originals>/v11-quant-native-disk-materials-a1.json \
  --native-disk <originals>/v11-quant-native-disk-materials-a3.json \
  --loading <originals>/v11-quant-native-loading-sources.json \
  --getters <originals>/v11-quant-sdk-getter-headers.json \
  --out <new-output>/native-route-materials-review.json
```

Actual local recomputation of all four fixed originals produced summary SHA
`fc0bac20e670156e7f5339f7231dc91a62b97f78211ac093b913d3ebcbe48410`.
The 19 parser/refusal CPU contracts passed locally (0 skips). They are in
`tests/worker_extension/test_quant_native_route_materials.py`; they do not
simulate Native success or grant runtime authority. No SDK build is needed
for this reader, and no new proxy build or NPU run was performed.
