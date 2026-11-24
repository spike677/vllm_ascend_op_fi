# Group resource gates

The new group scope has a fixed **512 MiB runtime JSON limit**. Existing index,
boundary and offline callers default to 128 MiB. There is no CLI or RPC setting
for an arbitrary cap. The larger live index requires the exact nonserializable
`GroupRuntimeLogScope`, created from the trusted reconstructed normal object
and matched source manifest, completion hash, attempt directory, process and
both thread domains. Tensor, registered-binary and individual-line limits do
not grow. One complete append-only log keeps all registration generations.

New group-normal captures record the complete initial and final runtime prefix
hashes, index validation time, and a bounded inventory of actual phase files.
The independent normal reader checks those prefixes against the unchanged full
log and checks the file inventory against the original phase directory. It
also totals the original registered binary files. Each phase remains at most
128 MiB of capture JSON and 512 MiB of other files. This adds observations; it
does not retroactively add them to source108 or authorize its incomplete arm.

`group_resource_budget.plan` uses only a new same-package complete five-phase
normal set. Runtime growth is the maximum observed phase increment, including
inter-phase additions, multiplied by **2** for all 60 requests. It separately
includes initialization bytes, observed final unscoped log growth and a reserve
of three full requests for F/OFF/recovery. More than 512 MiB is
`NO_F_CAPACITY_REFUSED`. Time uses constant work plus the measured index work
scaled by the sum of projected cumulative prefixes, separate initialization
and ordinary health time, recovery time, and at least 120 seconds/15% margin.
More than the unchanged 1800-second case budget is `NO_F_TIME_REFUSED`.

Disk estimates include all 60 full payload/boundary/host/capture files,
registered binaries, the complete runtime log, recovery files and journal
margin. Memory estimates conservatively include parsed runtime bytes and
retained controls/capture metadata. Before each F, `before_fault` recomputes
the sealed plan and requires fresh same-process/thread CPU resource counters,
the actual current prefix size, remaining wall time, free disk, and available
memory bounded by both host and cgroup counters. It reserves the remaining
protocol and fresh OFF/recovery before another F. An unavailable or stale
counter refuses admission; no device query is made.

These are conservative admission estimates, not reservations against other
processes. Unexpected failures still retain the owner until actual OFF/drain
and cleanup. No uncertain F is retried. The public group adapter must call the
live gate before every F; wiring and final raw C2 review are a separate commit.
The saved resource scope and plan cannot authorize a fault or C2/C3 qualification.
