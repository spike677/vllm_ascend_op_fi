# Quant single-site C1 through Worker Core V1

This entry uses `FaultConfig('quant', 'batch_shared_scale')`, `HostAdapter` and
`WorkerFaultClient` with the existing QuantProvider lifecycle. Deployment only
provides the reviewed a4 bundle, prompts and private evidence directory. It
does not accept a layer, tensor address or row label from the user. The new
`QuantGroupC1ServingBinding` is distinct from the legacy C1 binding and loads
only the separate group DSO. Its Native sources and a4 SDK fingerprint are
unchanged by this Python integration.

The seven naturally scheduled B8 first-decode phases are discovery 1,
discovery 2, reference, lossless, qualification OFF, one fault, and healthy
OFF. Automatic selection requires exactly one occurrence of a site in each
complete healthy census, the same actual module/method/weight/build/lifetime,
complete public-to-Native formals and a current independent request-row proof.
Equal shape or values cannot supply a missing row edge. Selection is limited
to one proved site, and the complete original structural/public/Native census
is preserved in `discovery.json`. Unadapted Attention/MoE rows and unobserved
structural modules remain explicit. This is not all-eligible qualification.

A later independent unadapted path can leave the full-forward row report
incomplete without invalidating an already closed C1 target/consumer prefix.
Both real query snapshots carry observer/forward identity, monotonic time and
call counters. They must have no earlier observer errors or missing calls,
the same actual ordered requests, and the consumer must be the completed owned
QMM chain. Any source-parent frames referenced by those snapshots must finish
as observed; all observer resources must still cleanly release. An unknown
edge in the target's own row chain, an error before its consumer, a foreign
request or an unclosed source parent refuses selection. The original candidate
`UNVERIFIED` status, full-forward errors and reasons remain unchanged. C2/C3
receive no qualification from this limited prefix.

The Native owner is installed only after the two censuses. Its budget admits
four selected model calls: reference, lossless, qualification OFF and fault.
Each has the exact current owner/generation/target/request proof and actual
parent lifetime. Every other model Native parent is independently observed
and passed unchanged. F records one model parent and eight raw Native events:
two real GWS calls and two real execute calls, including the correctly shaped
reference on the same current input. Both child roles, registrations, streams,
workspace ownership and complete readbacks must match the original lease.
Actual Q or broadcast scale bytes must differ before publication. The adapter
does not use a CPU rounding oracle or old B output as F's numerical reference.

An idle registered owner remains visible as `native_registered=True` with
its full `installed`, `sealed` and retained host-byte state. `leases` counts
active public/parent call leases. A sealed site cannot arm outside the one
current managed public call. The real Worker Runtime and Host API tests check
this distinction and refuse a genuinely busy Native call. Provider code uses
the actual RPC identity plus held Worker/runner/model and fresh process,
thread and device checks; it does not recursively invoke the identity RPC
while Runtime's non-reentrant lock is held.

After F, stop precedes whole-device synchronization and target drain. Native
restore/release and callback removal must complete before the healthy OFF
request. No expired public Q/scale storage is reconstructed for late restore.
Failure retains ownership; successful publication alone cannot release the
Worker owner before healthy OFF. An installed owner with no targets uses the
reviewed `cancel_empty` barrier. All metadata profiles, nested/outer dispatch
modes and model/method/API hooks must be removed before C1 is reported ready
for review. NaN/Inf in the F model output is retained and does not replace the
required current-input Q/scale effect receipt. Healthy OFF remains exact.

CPU tests use real PyTorch hooks and the real QuantProvider, Worker Runtime
and HostAdapter, with explicit fake scheduler/Native event fixtures. They do
not validate CANN arithmetic or natural NPU row mapping. E5 is not assessed by
this entry; C2/C3 remain false. The real NPU result is still pending root's
queue and review.

The focused suite passes 127 CPU tests with no skips when the real Core source
is present (or supplied as the explicit local overlay below). The new cases
cover the complete seven-phase route, retained unknown candidates, missing or
relabelled child receipts, current-device mismatch, busy Native leases, a lost
RPC after mutation, retryable drain failure, refusal to release the Worker
before healthy OFF, immutable report snapshots, and nonfinite F output with
exact finite OFF recovery. `py_compile` and `git diff --check` also pass.
The additional prefix contracts cover an independent later refusal, an error
before the target or its real consumer, changed query order/request identity,
and detached report snapshots that cannot rewrite the internal proof.

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension;fault39/tests'
# Only needed in the isolated branch before the Core commits are integrated:
$env:DCAT_QUANT_C1_CORE_SOURCE='C:/Users/admin/.codex/worktrees/fault28-native-fa-publish/demoncatpfi/.publish/c2c3-closure-20261117/src'
& 'C:/Users/admin/.codex/worktrees/fault28-native-fa-publish/demoncatpfi/.codex-tmp/unified-env/Scripts/python.exe' -m unittest test_quant_group_c1 test_quant_group_inventory test_shared_scale_inventory test_quant_inventory test_quant_source_rows test_quant_mla_metadata_rows test_quant_mla_decode_rows test_quant_row_provenance test_worker_quant test_quant_group_binding
```

Root's preparation command (creates the review manifest, starts nothing):

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_prepare_inventory_case.py" \
  --group-abi --single-site-c1 --installed "$INSTALLED" \
  --quant /home/17-fault-c2c3-20261117/native-quant-group-v1-a4 \
  --probe "$SOURCE/experiments/c2c3_closure_20261117/providers_quant_c1_probe.py" \
  --source-commit "$COMMIT" \
  --image-id sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0 \
  --model /home/haoran/models/DeepSeek-V2-Lite-W8A8 --card 2 \
  --out /home/17-fault-c2c3-20261117/providers-quant-group-c1-a1-cases.json
```

This requires the next source/package containing both the observer cleanup
fix and Core's `advance_batch`/`begin_decode_capture` methods. Use the unchanged
a4 manifest SHA `8d8d02476a911b2b0251eb70b6af3e2cd45ea2498aa65d1beb1a316822f3012d`
and library SHA `b25807b379345c35139138b5eeb1b57d00b421594d88aa2b62dc6ebd8021f557`.
All raw results stay on 124; only reviewed summaries belong in Git.
