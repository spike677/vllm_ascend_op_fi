# Read-only device-allocation gate review

The revised gate refuses cards 2 and 5 while kubelet records their allocation to Pod
`07ddbadb-06a1-4edf-b8f8-b96cd33adc79`, container `ascend`. In the final read-only
host check, both cards had no visible NPU compute process, but the allocation
record still existed. Its release is unverified. No workload, container, driver,
allocation, or host setting was changed by this review. This work adds no C2/C3
qualification and does not attribute the earlier FA a3 failure to Kubernetes.

## Observed sources and the missed path

The original `fault39/tools/run_124.py` inspected only
`/run/containerd/io.containerd.runtime.v2.task`. Actual running daemons use:

| Daemon PID | Actual config | Actual state |
| --- | --- | --- |
| 3088 | `/var/run/docker/containerd/containerd.toml` | `/var/run/docker/containerd/daemon` |
| 311972 | `/home/mindcluster/containerd/config.toml` | `/run/mindcluster-containerd` |

The Kubernetes OCI configurations were under
`/run/mindcluster-containerd/io.containerd.runtime.v1.linux/k8s.io/`.
The revised gate resolves `--state` or the running daemon's actual `--config`,
then inventories both v1 and v2 task directories. Configuration hashes, paths,
and relevant device mappings are recorded; complete configurations and process
environments are not exported. Unreadable, malformed, imported-but-unresolved,
or otherwise unknown active sources block the gate with a diagnostic.

Kubelet PID 2125466 has `--root-dir=/home/mindcluster/kubelet`, while its device
plugin manager uses `/var/lib/kubelet/device-plugins`. The actual socket is
`/var/lib/kubelet/device-plugins/kubelet.sock`: inode `105606902` in its Unix socket
table matches the process's owned FD `17`. The gate uses that owned socket path;
an unowned socket never overrides the root-dir fallback. A retained default
checkpoint is inspected even when the workload or kubelet process has exited.
Missing or unreadable checkpoints for observed active allocation sources refuse
admission rather than implying that a card is free.

The checkpoint is
`/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint`, SHA256
`9735aca82c5cca8adfa3aa2b0af1317876df5e55f4841aa00682dc3f2638b331`.
Its `Data.PodDeviceEntries` explicitly assigns `Ascend910-2` and `Ascend910-5`
under `huawei.com/Ascend910` to the Pod and container above. Each selected card
gets a `KUBELET_DEVICE_ALLOCATION` blocker with
`ALLOCATION_RECORDED_RELEASE_UNVERIFIED`. Disappearance of a compute process does
not clear it. `RegisteredDevices` is a global registration list, not a Pod
assignment, and never creates a reservation by itself. Unsupported accelerator
resource names, device IDs, or allocation schemas yield `UNKNOWN` and refuse.

The reviewed infrastructure OCI tasks had permissions for both cards, including
one `/dev` directory bind. Those eight matching grants remain permission
inventory only. No exclusive reservation is inferred for a persistent idle
container or infrastructure Pod merely because it can access a device. The
existing separate running `ptq_` dedicated-job rule is preserved.

## Timing and attribution limits

All times below are UTC on 2026-11-17.

| Observation | Time | Meaning |
| --- | --- | --- |
| Retained checkpoint mtime | 15:47:16.801035 | File metadata observed later; not a contemporaneous snapshot or a unique driver-error cause. |
| FA a3 card 5 old pre-gate | 15:47:28.392430 | Old gate returned idle with an empty runtime permission inventory. |
| FA a3 container start/end | 15:47:29.890440675 / 15:47:38.096846651 | `aclInit` returned 507899 `Resource_Busy`, before any FA call. |
| FA a3 old post-gate | 15:47:38.819504 | Old gate again returned idle. |
| FA a4 card 2 old pre-gate | 15:49:15.866352 | Correctly blocked visible NPU PID 3513971; no attempt container started. Card 5 also showed PID 3513972. |
| Revised gate | 16:04:59.295710 | No visible NPU compute processes; two explicit retained allocation blockers; idle false. |

The a3 and a4 controllers are respectively
`/home/17-fault-c2c3-20261117/fa28-launch-short-a3/controller.json` (SHA256
`9a2fe497843367535cdaae1aaf5ecc33720bed159d28e260feafa04d90ea7de3`) and
`/home/17-fault-c2c3-20261117/fa28-launch-short-a4/controller.json` (SHA256
`d3f95eb05550f19c31b4ebfb67897b67a18269c15810b9f62b4babd493f7e948`).
The a3 snapshot's lack of a blocker and a4's positive process evidence are
different observations. A later checkpoint and its mtime do not prove that the
specific Kubernetes allocation caused a3's initialization failure. The original
controllers, logs, source-a32 snapshot, and failed attempt outcomes are unchanged.

## Verification and remaining gate

`tests/test_device_allocation_gate.py` passes 19 CPU contracts. They cover explicit
Pod versus global registration, selected versus unselected cards, malformed or
unknown schemas/IDs, missing checkpoints, retained allocations after process
exit, actual containerd config/state and both runtime versions, owned versus
unowned kubelet socket paths, bind-mount permission inventory, the existing
dedicated-job distinction, and unavailable TOML parsing.

The host was verified as Python **3.12.4** with stdlib `tomllib`; no package was
installed. If the reader is absent on an older host, the gate reports
`TOML_READER_UNAVAILABLE` and refuses; it does not silently assume a default root.

The final receipt is
`coverage_device_allocation_gate_readonly_a2.json`. Its executed source SHA256 is
`6579891ef3bf85f0f0384ab7ec2b603cad8e9fd40ddddae04b0d90953b7a81b3`, verified
against the exact local `run_124.py` bytes. It used `python3 -B` over SSH, executed
only `idle_gate([2, 5])` in memory, and imported the existing deployment helper
from source-a32. No runner main function was invoked. The earlier
`coverage_device_allocation_gate_readonly.json` is retained as a preliminary
receipt: it already blocked both allocations but also reported the missing
custom-root checkpoint before the actual owned socket path was resolved.

The cards remain blocked until a fresh read can establish that no selected-card
allocation or other blocker remains. This code does not clear checkpoint entries
or release foreign allocations. It is a read-only admission snapshot, not a
Kubernetes scheduling lock: a later independent allocation is still possible.
The existing cooperating-job lock does not provide host-wide isolation.
