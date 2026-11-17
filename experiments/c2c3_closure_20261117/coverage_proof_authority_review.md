# Live C2 proof authority review

The previous C3 gate checked a private issuer marker on a frozen dataclass.
`dataclasses.replace` and shallow copy preserved that marker. A caller could
therefore copy an issued CPU receipt, change its scope to `natural_worker`, and
satisfy the old marker check with a matching requested scope. This was a real
admission bug, not an NPU result.

Commits `9499d57` and `13955ef` replace marker-only authority with an internally
registered original object, a SHA256 digest of its actual dataclass fields, and
the issuing `os.getpid()`. Copies, reconstruction, forced field changes, nested
scope changes and fork-inherited objects fail before C3 reservation/admission.
Verification does not dispatch through an instance-overridden `report` method.
Weak references remove registrations when receipts are no longer retained.
All existing workload, implementation, environment, candidate, target lifetime,
owner and increasing-generation checks remain. The public constructor and
report format remain available for data; they do not issue authority.

The registry is an internal API guard for trusted provider code. It is not a
Python security sandbox against code that edits module internals. Provider
execution, raw effects, complete discovery and recovery still need independent
review; a live receipt alone does not grant public mechanism qualification.

## Frozen 18/20/39 route review

Root supplied the actual source snapshot commits and driver hashes. The exact
Git bytes for all three drivers match those hashes. The driver, serving wrapper
and coverage backend paths were reviewed at those commits, including the proof
handoff through provider bindings. The exact file hashes and line references
are in [coverage_proof_authority_review.json](coverage_proof_authority_review.json).

| Mechanism | Actual frozen source | Proof path |
| --- | --- | --- |
| `18.api_half`, standard Worker | source-a20, `8967dffc825600ebc12e4b38617f3ac7ca08ca6f` | `natural_rms_worker_probe.open_group` obtains `session.provider.group.c2_proof()` in that Worker and passes the original object directly to `provider.open_group`, then `CoverageSession`. |
| `20.buffer_substitute` | source-a21, `5a8b54486161132b34a0af7d427a851519134a8f` | `QuantGroupBackend.finalize` issues the proof after recovery; `Serving20CoverageSession.close` stores that exact object on the Worker. The next binding passes it to the backend, which preserves its logical owner and uses generation + 1. |
| `39.matmul_range` | source-a21, `5a8b54486161132b34a0af7d427a851519134a8f` | `MatmulCoverageBackend.finalize` issues the proof; `Serving39CoverageSession.close` retains it on the Worker. The coverage binding forwards the same object through `Fault39CoverageReplay` to the new backend. |

None of these reviewed frozen paths copies, replaces, deserializes or relabels
the proof. The source routes therefore do not depend on the discovered weakness.
The NPU processes are historical: this review cannot recover their Python object
IDs or retroactively assert that the new registry executed there. Prior accepted
results remain bound to their original source, workload, environment and evidence;
they do not certify the new HEAD or create a fresh live C2 gate.

## CPU verification and Native26 review

The changed module was loaded into the current integration tree without editing
that tree. All 88 selected shared/provider tests passed, including live C2→C3 for
18, 20 and 39, the saved-evidence auditor, and the 20 QMM/WeightNz C1 regressions.
The final process-binding increment passed all 9 proof-authority tests. Negative
contracts cover copy/deepcopy/replace, JSON/pickle/direct reconstruction, top-level
and nested field mutation, overridden reporting, CPU→Worker relabelling, and a
mocked fork PID change. An unchanged original proof still admits the matching
later generation; unrelated owner, target lifetime and scope remain rejected.

Native26's in-progress complete-set integration was reviewed read-only with its
owner. It retains UNVERIFIED rows and refuses a subset, freezes per-target
reference/lossless/OFF, and records actual Native changed-residual bytes plus the
real consumer and restoration in each effect artifact. Whole-model output hashes
are only healthy-control evidence. Unknown calls before an anomaly or during OFF
now fail; post-anomaly additions remain separate passthrough records. A failed
fresh geometry check cannot remove a previously eligible target from the call
denominator. Stop/drain precedes target release, and the full healthy request and
zero owned hooks complete the same-Worker recovery gate.

Using the new shared proof module, Native26's 5 complete-group and 7 candidate
CPU fixture tests passed, including its original live once→every_match chain.
Its provider owner is adding further unknown-call and cloned-proof rejection
tests. These are CPU/mock contracts only. No container or NPU action, raw-statistic
edit, historical report rewrite or additional C2/C3 acceptance occurred here.
