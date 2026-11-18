# Execution Adapter contract v1 — optional mechanism-specific capability

This extends the existing Host, Provider, Adaptation and Evidence contracts; it
does not replace Worker Core, CoverageSession or any Native ABI. An adapter
proves how a real model call reaches the intervention boundary and consumer.
Model labels, equal shapes and saved JSON do not grant a live binding.

## Lifecycle owned by existing components

The serving runtime creates Workers and enqueues model work. WorkerFaultRuntime
owns the attachment/lease. A Provider optionally owns an execution adapter for
that same scope. No global execute_model replacement, background forward,
automatic restart, cross-owner hook removal or implicit fault retry is allowed.

| Operation | Contract |
|---|---|
|inspect(worker, model)|Read actual Worker/model identity, module objects, aliases and supported execution boundaries; no inference or fault.|
|observe(workload)|Arm a bounded owner-scoped observer for a real, already enqueued request; return control to serving. Collect only after its actual completion.|
|enumerate_calls()|Return the complete observed call set, raw ordinals, logical roles, actual operators and supported/unverified/unsupported reasons. Never filter it down to the selected subset.|
|bind(candidate)|Resolve a candidate to current live objects, request/rank, Native descriptors and consumer; return a process-local binding. A serialized candidate is not authority.|
|verify_execution(binding)|Join actual original-call/launch/writer and consumer receipts with that binding. Keep dispatch success, numerical effect and propagation separate.|
|verify_recovery(binding)|Use the existing owner-scoped recovery and fresh resource readbacks. A bool returned by a cleanup command does not prove release. Retain owner on uncertainty.|

These describe responsibilities, not a mandatory base class or a new generic
runtime. Existing mechanism-specific methods may implement the operations.
Any common Python facade must delegate to them without inventing evidence or
granting qualification. RMS/Conv parameter proxies can use a lightweight
adapter; Native/kernel adapters additionally prove the corresponding API/ABI,
source/build, launch arguments and storage semantics.

## Binding and evidence

A live binding includes actual Worker/process/model generation, owner, request,
rank/device and call occurrence. It retains the required strong storage/object
references for their real lifetime. Observer state belongs to that binding,
not an unscoped module-name map. An independent census supplies the denominator.

Maintain three distinct sets: observed calls, resolved candidates, and currently
qualified targets. For QMM, adjacency and a live residual consumer remain
required; FIA destination uses typed destination/lifetime evidence, not the
QMM adjacency rule. For Batch/MoE, request-row lineage must follow actual row
transformations; matching row counts cannot establish a mapping. Attention
adapters preserve Q/K/V, masks, actual layout and page/sequence metadata.

Reference, Lossless, F, OFF and group qualification stay in the Provider and
CoverageSession. A stock/private normal-bridge failure prohibits F attribution.
The four Attention mechanisms retain separate IDs and fault semantics. Failed
or unsupported observations remain visible. C1, C2, C3, Recovery and E5 are
independent fields; diagnostic runtime reads are not raw statistics/mix E5.

## Compatibility and transport

Host routing selects explicit currently discovered Worker identities, never
response positions or an assumed rank0. Each endpoint has its own attachment,
nonce, lease and generation; a group report cannot erase a failed endpoint.
Native preload remains a pre-start serving deployment action. Build provenance
and Worker source provenance are independently pinned when a Native build is
reused. The adapter must inspect actual loaded symbols/libraries; a successful
CPU build or version-family match is not NPU qualification.

Source-only exports preserve this contract without importing any previous live
authority. Yellow-zone Registry/Controller integration remains independently
pending until those real host implementations can be tested.
