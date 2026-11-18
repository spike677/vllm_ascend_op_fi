# Native39 repaired-build requalification

Root accepts **C2 and C3 for this bounded standard-Worker scope** after separate
coverage/effect/source and SDK-array audits. The current strict total is
**C2 3/17, C3 3/17**: Fault18, MoE8 and this new Native39 build. Earlier leaked
builds remain withdrawn; their receipts have not been relabelled.

Actual run: `native39-cache-witness-c2c3-card2-a2`,124 card2, Ascend910B4/A2,
CANN8.5.1, pinned vLLM0.18 image, Qwen3-0.6B BF16, TP1, the original fixed
14-token fresh prefill. Running source was **source-a42/packages-a16**, not
the newer developer checkout or source45 reviewer. Worker creation and shutdown
were performed by the standard runtime in the root-owned temporary container.

| Independent gate | Recorded result |
| --- | --- |
| Complete natural denominator | 113 actual sites, distinguished by module/call plus live cache key/epoch; nine C2 and ten C3 phases |
| C2 | 113/113 eligible hits;339 observed coverage calls including controls |
| C3 | 226/226 eligible hits in two requests;452 observed coverage calls including controls; original live C2 handoff |
| Actual effects | Each target's complete output bytes changed; finite replacement inputs/readback and original storage preservation verified |
| OFF and health | Complete healthy output restored; same-Worker healthy token576, text ` The`, after each group |
| SDK arrays | Five real registration witnesses, each three captures/nine acquired and nine released arrays; actual loaded counter instances, cache lifetimes and cross-group monotonicity verified |
| Final resources | No remaining owner, leases, inflight operations, SDK arrays, owned hooks or serving binding |

The array reviewer checks689 C2 and803 C3 snapshots. Reusing existing caches
means no new getter activity is required inside each group: every actual key
instead joins its recorded typed-GWS capture and the same live loaded counter
instance. Nonzero lifetime totals alone do not pass. This is SDK getter-array
ownership, not a claim that the model allocator releases all cached memory.

The source audit also checks the fourth implementation-hash component:
`fault_injection/metadata_arrays.py`. Actual installed and frozen-source bytes
both hash to `9bf06627e266a968edc8d32fd16d98f908e47010450f128e0d2847326fe5ff4c`.

## Statistics remains a separate incomplete status

Only original statistics/mix, stack, construct, configuration and collector
identity enter the numerical comparator; tolerance remains baseline-reference
`rtol=1e-5, atol=1e-8`. C2 first observable transition is record58,
`Functional.linear.0.forward/output/0/Max`: **5.1875 → +Inf**, with126 preceding
null fields. Later NaN fields make the complete capture incomparable. No finite
relative error is assigned to this transition. C3 has **MISSING_RAW_CAPTURE**;
its statistics are not reconstructed from C2 or injection logs. E5 is not passed.
Later targets receive propagated nonfinite values;113 hits are not113 isolated
causal trials. These E5 limits do not erase independently verified effects/OFF.

## Original references

All paths below are under `/home/17-fault-c2c3-20261117` and remain on124.

| Original artifact | SHA256 |
| --- | --- |
|`native39-cache-witness-c2c3-card2-a2/providers39-qwen-c2c3/result.json`|`8eb8df3b1065ee7f3c4dfb602b37860acf0c59ccb3ec54683c15a0f81f0d56d2`|
|`native39-cache-witness-independent-a1/summary.json`|`aeef6f5a509e415da95ed82aae8eca1bf1d944ecf44da40317451ab43f774b0a`|
|`review-native39-metadata-20261118-a2/c2.json`|`8592421c05d6fa7bc992963638604b1a38db5a17bfb3ded99825313a2899de98`|
|`review-native39-metadata-20261118-a2/pair.json`|`f064f55c2c42a2e0910af42b10d2a8c34e5e4e2bfb42416fbb6cf55b9e9efb5a`|
|Actual Native build manifest|`8ff299b35ba6637a86adfad24203e87eb227e188a05e3fc78a0c0719cafb05e3`|
|Actual Native build profile|`7c73c38ada62cb57030b5ae60482902c2653bc9869b5f800b407e0bfcc9238d2`|

The container exited0 and its owned cleanup completed. A changed Kubernetes
checkpoint made the controller's post-case device gate report busy, so the
next case did not start. The subsequent independent read-only live-allocation
review found no current owner; that is separate from this Worker's already
completed OFF proof. Container removal itself is never recovery evidence.
No other model, CANN version, A3 or multi-Worker scope is qualified here.
