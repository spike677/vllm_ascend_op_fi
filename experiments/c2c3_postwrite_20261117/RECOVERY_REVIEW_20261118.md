# Postwrite recovery and recorded group review, 2026-11-18

Worktree starts at root `91ce0982`; it preserves the integrated `9dfdb0c6`
multi-lease and retryable Native ownership implementation. This agent performed
CPU work and read-only SSH only. No NPU/container action, qualification-matrix
edit, raw statistics edit, or raw artifact commit was made.

## Delivered changes

- `77fcc22`: preserve verified collector ancestry across C2 close / C3 reinstall;
  group cleanup failure can retry without resetting or resealing its observation.
- `1d3eafd`: classify exact unmodified ModuleList containers explicitly while
  retaining the full structural inventory, actual child coverage and all unknown
  module refusals. Save complete independent discovery diagnostics on failure.
- `7585dfd`: read every effective K/V prefix byte selected by the actual page
  table and sequence length, retain physical ranges separately, and verify these
  bytes in healthy repeats, lossless, baseline controls, once-exhausted requests,
  and OFF. No cache clearing, repacking, or raw-field filtering.

75 related CPU tests pass. This includes the integrated multi-lease/owner-thread
tests, 39 structural discovery, all 41 lifecycle/coverage tests, and execution of
the exact pinned collector code objects described below. Native .cpp/.h and the
39 Native/Python lease implementation are unchanged from `91ce0982`.

## Collector ownership failure

The old cleanup accepts only the object saved as `original`, its owned wrapper,
or a verified collector wrapper whose child reaches one of those objects. C2
cleanup preserves the collector wrapper and replaces its owned child with the
original API. Its registry's `ori_api_attr` also retains that original API.

On C3 install, the saved `original` is now the preserved collector wrapper.
Consequently the registry's legitimate leaf API is no longer object-identical to
that outer `original`. The old traversal wrongly attempts to interpret the leaf
as another collector wrapper and rejects it. This was reproduced with the exact
pinned factories and actual Provider install/close methods. The recorded first
cleanup receipt has 5 patches removed and 13 owned links removed, consistent with
this lifecycle. Source41 does not start a new capture for C3; no collector rebuild
is required to reproduce the failure.

The old raw receipt does not serialize the live callable graph. It remains a
failure; the reproduction is not permission to reinterpret an unknown old owner
as safe. The fix captures the actual pre-install collector chain, verifies its
source/code/globals/API name/ApiTemplate object and exact child links, and checks
that same chain again before uninstall. It admits only those exact ancestor
objects as additional anchors. A verified rebuilt collector is kept intact;
unknown current wrappers, changed ancestry, changed ApiTemplate instances, and
foreign registry owners still cause an atomic refusal before any uninstall.

Pinned source hashes, independently read from 124:

| Source | SHA256 |
|---|---|
| core/dump/api_dump/api_registry.py | `1b6d5cc168034e994d3cb425474a318c8058366b28afe6d8014835ce4f98edeb` |
| pytorch/dump/api_dump/api_register.py | `7f1f1b930941b5c49f4674f2174d73d80b7b84c00fb63a1b1ac0529cbe6073ba` |
| core/dump/service.py | `d51804b819a07453733e52eb81ec132f2cfa9934bfa5b243aaf7ceecd8ad97b3` |

`tests/test_postwrite_collector_pinned.py` compiles those unmodified full source
bytes, executes their actual ApiWrapper/ApiRegistry/ApiTemplate code objects and
BaseService start/stop/registration methods, and runs Provider install / repeated
start-stop / C2 close / C3 reinstall / rebuild-or-preserve / foreign-owner refusal /
retry. CPU module-hook, dump-I/O and device services are explicit doubles. This is
not a full PrecisionDebugger dump and not an NPU acceptance result. Set
`DCAT_POSTWRITE_COLLECTOR_SOURCES` to a local directory containing these three
files to reproduce it; no source hash override is used by this test.

## Independent review of the recorded C2 interval

Raw root: `/home/17-fault-c2c3-20261117`.

| Receipt | SHA256 |
|---|---|
| postwrite-groups-card2-a1/postwrite41-c2-baseline/result.json | `1e734a69601fb707b7e33d2ffb0df7944750f8e3060d91325a44170c342ac355` |
| postwrite-groups-card2-a1/postwrite41-c2-fault/result.json | `6a95af53d77c5abb5d3af8ca3bcab39ce2c32cf7684354355b14ec222ae48ae7` |

`review41_group_receipts.py` was run read-only against these originals. It joins
actual observer call IDs, forward IDs and request IDs to outcomes; it does not
use the provider's `contract_satisfied` flag as its acceptance criterion. In each
arm, two healthy inventories and the four C2 phase inventories contain all 28
distinct modules, 28 writers and 28 original consumers called exactly once.
The C2 session joins 56 actual eligible calls plus 28 OFF calls. B controls all
28 targets with zero hits; F has exactly one Native hit for each of the 28 targets.

All 28 Native execution receipts match the common owner/generation, report exact
readback and the expected changed-element count, and report consumer return,
restore and release. Every recorded current-slot/query before/after hash agrees.
All 28 target OFF recovery proof hashes and the independent normal-request health
proof hash were recomputed from their referenced artifacts. The actual health
request IDs match driver enqueue IDs, and health uses the same model object.

At the end of this C2 interval, inflight calls, Native leases, owned hooks, API
patches, held/pending common leases and recovery errors are all zero. The healthy
full root output hash is
`cc0ecd83f69576537e41ffdc35ac4fbe4f08b573ea37bb2645e9bb8af64445ae`.
F's first C2 output differs; its second once-exhausted output, OFF output and
separate normal health output all equal that healthy hash. All 28 second-request
FIA output hashes and current slot/query hashes also equal B at the same step.

These checks support the recorded bounded all-target-once, current-slot proxy
interval independently of the later C3 failure. They do not prove equality of
every effective K/V prefix byte in the old run, equality of unused cache pages,
or a broader public qualification. The new full-prefix probe must produce fresh
evidence; it cannot retroactively strengthen this old sample.

C3 has 56 recorded hits but fails final uninstall. Its failure snapshot retains
28 common reservations, 58 hooks and 5 patches (Native leases are zero). It is
not accepted. The overall F job remains FAILED_WITH_EVIDENCE.

## All 38 step1 differences retained

The separate raw audits under `postwrite41-groups-a1-raw-review` were checked
against the current SHA of every referenced dump/stack/construct/config/collector
file. All hashes match. Same-step B/F stack and structure align. Comparison stays
`abs(F-B) > 1e-8 + 1e-5*abs(B)`; 6072 null fields remain unobservable, with zero
NaN/Inf/unsupported fields. Null does not mean equal.

Step0 has 3706 exceedances. Its first observable change remains downstream
`Functional.linear.1.forward/output/0.Max`, 3.0625 -> 2.96875, with 138 preceding
null fields. This is a visible consumer output, not proof of a KV root cause.

Step1 has 38 exceedances although once has already been exhausted:

1. 36 fields are pre-write destination extrema of `Tensor.__setitem__` calls.
   The actual stack points to attention_v1.py:839,
   `output[:num_tokens] = attn_output[:num_tokens]`. All affected destination
   shapes are [1,16,128], the slice is [:1], and the RHS shape and recorded
   statistics match B/F. This assignment overwrites the full destination; the
   compared extrema belong to its contents before that overwrite. These 36
   differences stay in the raw audit and are not reclassified as numerical equal.
2. Two fields are layer22 whole-V-cache Max (writer input value_cache and FIA
   input value), 25 -> 25.125. In F step0, the real writer's current V input has
   Max25.125 and writes slot174 on page10. Whole-cache Max is 25 before that write
   and 25.125 at its FIA consumer. Step1 uses page11, slot190, length15; its current
   V input Max is 25, while whole-cache Max remains 25.125. B uses the same page
   schedule, with Max25 throughout.

The active step1 range is page11 rows0..14, physical token slots176..190;
page10/slot174 is outside that request's valid prefix. This strongly supports
retained prior-request page contents as the explanation for the two aggregate
cache extrema. The old dump does not locate the maximum element or hash every
active prefix byte, so it does not alone prove page11 byte equality. That exact
limitation is addressed by `7585dfd`, without touching these raw fields.

Baseline step0 versus step1 has 122 differences; the first is slot index174 ->
190. Different requests/pages are not a numeric repeat baseline. Baseline self
has zero observable exceedances. All 38 same-step B/F differences are retained.

## 39 discovery and next root run

The 39 baseline receipt at
`postwrite39-multilease-card2-a1/postwrite39-c2-baseline/result.json` has SHA
`377a0449b230c2d153647f9a2ba60645698c71f799907c673fc0c0fc6e647e9d`.
It fails discovery before a group capture executes; 140 eligible projections,
254 total projection returns and zero live leases are recorded. Its actual model
source inventory includes one ModuleList. The same failure is reproduced on a
real CPU model that iterates a ModuleList's children. The old raw report omitted
the unresolved discovery rows, so it cannot establish that this was the only
unresolved item. The new driver records them even on refusal.

Root should package the three changes above into a fresh source bundle and use
the existing root-verified SDK Native directory/build fingerprint from source41;
its library SHA is
`8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622`.
Do not use the older a2 build identity from the 20261117 handoff for this ABI.
No Native rebuild is needed for these Python-only fixes. Existing
`prepare_coverage_cases.py` and `prepare41_coverage_cases.py` remain the bounded
driver entry points; all actual NPU execution and final review remain with root.
