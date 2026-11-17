# Retained checkpoint: current-owner corroboration

The 2026-11-17 17:27:57 UTC read-only check returned
`NO_LIVE_OWNER_CORROBORATED` for physical cards 2 and 5. The two original
checkpoint blockers for Pod UID `33991897-e3ed-4910-aa2c-835b9fe682a0` remain
unchanged in `stale_checkpoint_allocations`, together with their evidence hash.
This is evidence of no current observable owner, not proof that kubelet's
internal device allocation state is empty, that `aclInit` will succeed, or
that the host has exclusive resource isolation.

The application of this review is limited to the current root-controlled fresh
attempt after the explicitly authorized termination of that exact Pod. The
gate must run again against the new attempt's source. If initialization again
returns Resource_Busy, preserve that failure and stop; do not clear a checkpoint
or broaden the exemption. This change launches no workload or container and
performs no host configuration, reset, deallocation, or process termination.

## Predicate and identity

The existing health/process/FD/Docker/runtime/checkpoint gate runs first. Live
review is considered only if every blocker is an explicit per-Pod checkpoint
allocation. An independent final execution of the complete gate must preserve
the exact blocker, runtime, OCI permission and checkpoint inventories. An
additional blocker, changed hash, unknown owner, or failed read keeps the
original allocations blocking.

The live review uses the actual kubelet argv's root directory and kubeconfig.
It requires the reviewed executable version `Kubernetes v1.28.15`, the same
PID/start ticks/executable/boot lifetime before and after, and the actual Unix
socket's `SO_PEERCRED` PID/UID/GID plus stable filesystem device/inode. The
node-authenticated TLS client must have a `system:node:<name>` certificate CN;
admin, token/exec authentication and insecure TLS are refused. Its Node UID,
node name, machine ID, boot ID, kubelet version and fresh Ready condition must
match the current host. Credential contents are never reported.

The reviewed RPC is `/v1.PodResourcesLister/List`. A bounded, uncompressed
gRPC frame with HTTP/2 200 and gRPC status 0 is parsed against the exact
[v1.28.15 official proto](https://github.com/kubernetes/kubernetes/blob/v1.28.15/staging/src/k8s.io/kubelet/pkg/apis/podresources/v1/api.proto).
Unknown wire fields, unsupported dynamic resources, malformed responses or
unknown device/resource identities fail closed. `GetAllocatableResources`
is not queried because capacity is not current availability.

Two complete node-scoped API Pod lists bracket List. Their normalized full
UID/name/namespace/status/deletion/container/resource scope must agree. The
exact regular-container set must also equal List's set. Regular, init and
ephemeral container requests/limits are checked; a nonterminal accelerator
request, resource claim, unknown extended resource, missing/unknown phase,
pagination, new Pod or scope disagreement blocks the review. The checkpoint
Pod UID must be absent from both API snapshots and all discovered Kubernetes
OCI task annotations. Ordinary persistent infrastructure device permissions
remain inventory; they are not treated as exclusive reservations.

## Observed evidence and limits

The final receipt used host Python 3.12.4, kubelet PID 2125466 with start ticks
55778951, and `/home/mindcluster/kubelet/pod-resources/kubelet.sock`, whose peer
was that same kubelet. The node-authenticated API and List each contained 439
Pods and 439 regular containers, with matching sets and no listed device
assignment. The API had no nonterminal NPU request or resource claim and did
not contain the terminated UID. Both complete device gates found no current
compute/FD blocker. OCI task identities and hashes were stable. Unrelated Pod
names, full raw frames and the wider OCI inventory are local evidence only,
not committed here.

The checkpoint bytes retained SHA-256
`b570d0ea0d0280157a26d48c536a6aacff8cdf4fb55037f3b3882e85c1395414`.
Its observed mtime changed from `2026-11-17T16:44:53.399855+00:00` to
`2026-11-17T17:11:24.586580+00:00` while bytes stayed identical. Therefore it
would be incorrect to claim that the checkpoint was never rewritten.

The official [List implementation](https://github.com/kubernetes/kubernetes/blob/v1.28.15/pkg/kubelet/apis/podresources/server_v1.go)
enumerates current `GetPods()` and their regular containers after updating
allocation bookkeeping. In the reviewed
[device manager implementation](https://github.com/kubernetes/kubernetes/blob/v1.28.15/pkg/kubelet/cm/devicemanager/manager.go),
that update depends on source readiness and includes `pendingAdmissionPod`;
checkpoint persistence is a separate operation. Thus an empty device list
does not establish that every internal `allocatedDevices` or pending-admission
entry was removed. The status and three explicit false proof flags retain
this distinction. The evidence hash field named `release_evidence_sha256`
identifies this restricted no-current-owner evidence; it does not certify
driver or internal allocation clearance.

The earlier FA a3 result remains `aclInit 507899 Resource_Busy` with zero calls.
FA a4 independently encountered an explicit NPU process blocker. The later
checkpoint and current-owner observations do not establish a3's root cause.

## Validation and handoff

`python -B -m unittest test_device_allocation_gate test_kubelet_live_allocation_gate`
passed all 44 CPU contracts. Coverage includes retained/unknown checkpoint
schemas and paths, registered-only device inventory, incomplete/unknown RPCs,
non-node credentials, socket peer replacement, wrong machine/boot/version,
pending regular/init/ephemeral accelerator requests, claims, live assignments,
API/OCI ownership, scope/process/hash changes, and preservation of the original
blocker after query or final-gate failure. These are gate contracts, not C2/C3
qualification or evidence of driver availability.

`coverage_stale_checkpoint_target_summary.json` contains only target-card
records, public identity summaries, exact code and raw-receipt hashes, and the
joint proof. The uncommitted raw final receipt is
`coverage_stale_checkpoint_gate_readonly_a2.json`, SHA-256
`048704bfd59cf5a18728b257e4617f4aabdb5dcacd4d6fd1df60d8d60d26640a`.
The root agent should independently review this commit, run its own CPU tests,
and publish a new source snapshot before its one authorized FA attempt. Frozen
earlier source snapshots and historical evidence remain unchanged.
