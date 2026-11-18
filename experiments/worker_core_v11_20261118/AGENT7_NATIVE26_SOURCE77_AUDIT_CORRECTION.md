# Source77 Native26 original-review corrections

This change corrects two audit assumptions. It does not edit any original,
provider, Native library or acceptance flag, and performs no SSH, container
or NPU action. The unchanged CLI must be rerun against the complete originals
on 124 before the revised consistency result can be reported.

The locally supplied source77 review has 20 issues, not 21: 15 request-boundary
comparisons, one C2/C3 boundary comparison and four custom-getter image joins.
Its existing source/compiler provenance and typed FIA checks are consistent.
The diagnostic is an extraction, not a substitute for the original trace.

| Local review input | SHA256 |
| --- | --- |
| `v11-source77-core26-independent-review-a1.json` | `427fc723be44f49d464b27e8083e5c34fe96503c5abeebdd7365790d95614bc3` |
| `v11-source77-core26-audit-diagnostic.json` | `9ab0df5fca200fad30fbfbc2f6fe450f88ed5b434a59bfc5bc2d68a03782ec03` |

## Request boundaries

The recorded diagnostic shows `all_runs` and `non_target_runs` each increase
by 13 at each of the 16 boundaries: 208 counted non-target cache runs. Every
other state field, including all FIA, QMM, AddRms and stored/total event
counters, is unchanged. The Native `Inventory::run(bool qmm)` source defines
these paired increments; FIA and AddRms also have separate event counters.
The earlier C1 original auditor already permits this strictly accounted case.

The group auditor now requires the complete 22-field integer state on both
sides, the conservation relation `all_runs = qmm_runs + non_target_runs`,
equal nonnegative increments of the two permitted counters, and exact equality
of every other field. Missing fields, booleans, negative counters, one-sided
increments, target events and error/overflow changes fail. Typed FIA boundaries
retain their separate exact check. No target, callback, consumer, outcome or
unsupported call is removed from either generation's denominator.

Reports retain each request gap and the generation gap separately. The trace
does not identify the individual non-target operators, so their names or
specific serving-stage cause are not inferred from the count.

## Getter image identities

The SDK metadata-array image list covers trace/opapi/opapi_nn/group ownership
counters. It does not include `libcust_opapi.so`; requiring the custom getter
to appear in that list mixed two evidence domains.

The CLI now rereads the pinned group `build.json`, its `build-receipt.json` and
every listed file before reviewing held addresses. For the cached-address
profile it also joins the copied parent dependencies and parent metadata
receipt. The group builder explicitly rebuilds `libcust_opapi.so`, so its
final file SHA comes from the group manifest, not the parent's older custom
image SHA. The original custom CPU build contract is checked but confers no
device qualification. The complete external parent bundle is not reread;
that limitation is explicit in the binding report.

Every one of the 16 actual saved held-symbol edges must match the pinned group
build SHA and final image bytes, its unique executable mapping and target ELF
identity. The group getter additionally retains the SDK metadata-image and
Native lifetime joins. Missing build proof, a substituted image, a changed
parent dependency or a foreign build remains a refusal. Full Worker-source
and Native-compiler-source provenance is still required separately. This is
saved-evidence reconciliation, not a fresh symbol resolution or live mapping
attestation.

## CPU validation and limits

The focused regression passed 48 tests across the C2 originals, C2/C3
originals and new byte-level build-binding suites. It includes the current
public CPU producer/driver, every-target denominator checks, counted boundary
unit fixtures, all-field negative boundary mutations, actual file rehashing
fixtures, altered parent/custom files and missing build evidence. These tests
are explicitly CPU contracts and synthetic bytes; no test claims to load a
Native library. CLI help and `git diff --check` also passed.

`c2`, `c3`, `e5` and `public_qualification_granted` remain false in every
offline report. Original failure records and the previous inconsistent audit
must remain available. A new successful complete-original review may establish
consistency; it does not itself issue a live proof or establish statistics E5.
