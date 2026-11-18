# Native26 pending consumer investigation

The source48 QMM effects attempt failed in `target_000_reference`: one writer
was admitted, no public outcome completed, and the actual AddRms consumer
returned with status zero. Its raw result is retained under
`/home/17-fault-c2c3-20261117/native26-effects-source48-card2-a1/providers26-v2-qmm-deepseek-effects`.
The result SHA256 is
`1e51c3407aca14a0ac4c1067748ae021d1111cbb46bb06b5de7112a2922250ee`.

The preserved Native state has one pending lease, no executing writer, one
completion callback, two rejections and final error `Consumer (-10)`. The last
raw events contain selected QMM 1821, another QMM 1823 with producer 249, then
AddRms consumer 250. The independent trace reports no error or overflow. The
old runtime rejects every new writer callback while a lease is pending,
before the Python observer can classify the call as unselected. That `Busy`
error poisons the selected receipt; even a later successful consumer cannot
pass the required zero-error receipt check. This code path is consistent with
the saved state; the old artifact did not preserve the first failing receipt,
so its exact initial error cannot be reconstructed from the final state alone.

The narrow change permits observation only after the selected writer finished
and while its consumer remains pending, with no Native error or execution in
flight. An unselected call passes unchanged. A second selected lease still
fails `Engine::prepare`, and an actually executing overlapping writer remains
refused. Writer output, effect, consumer identity and restore checks are
unchanged. No candidate is removed from the independent trace.

`test_writer26_pending_runtime.cpp` includes the actual shim and supplies
explicit host-memory SDK doubles. It tests successful pass-through between
writer and consumer, refusal of a second selected lease, and refusal of an
actually executing overlap. The normal builder compiles and executes this
test with the installed CANN SDK and retains `pending-runtime-cpu.json` in its
build receipt. This test is not a CANN arithmetic or NPU acceptance test.

The local five Python builder contracts pass. The new C++ test awaits root's
device-free SDK build; this Windows environment has no C++ compiler. ABI sizes
and exports stay unchanged, but the DSO and source fingerprint change, so a
fresh SDK bundle is required. Do not reuse a previously built Native26 DSO
as evidence for this change. C1/C2/C3 remain unproved for this
attempt. The original failed artifacts are preserved.

The Python follow-up gives each selected pending call independent strong
references to its actual producer Tensor and storage. At the real public
consumer entry it additionally binds the consumer and writer objects. These
references survive the inventory advancing to the next producer after AddRms
returns. OFF still requires stop/drain and the same held objects, storage
identities, full spans, shapes, strides, dtypes and devices. Rebinding a Tensor
to another allocation refuses recovery without issuing any restore. Unknown
public output/consumer ownership still refuses; no pointer-only fallback is
introduced. References are released after confirmed Native release/close.

The first failure is preserved across later always-call root cleanup failures.
The failed consumer callback receipt is saved before validation, and request
unwind additionally reads the current Native receipt. Both are retained in
pending-outcome diagnostics in `writer_inventory.json` and
`failed_request.json`. Public reports return independent snapshots.

The following 59 local CPU contracts pass, including an actual Python public
consumer return that removes the old residual from the inventory before its
lease fails, failed-drain retry, changed-storage refusal and preserved first
failure/receipt diagnostics. Their Native operations are explicit CPU doubles;
the C++ shim contract above still requires root's SDK build.

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension;fault39/tests'
& 'C:/Users/admin/.codex/worktrees/fault28-native-fa-publish/demoncatpfi/.codex-tmp/unified-env/Scripts/python.exe' -m unittest test_writer26_pending_consumer test_writer26_group_experiment test_writer26_coverage test_writer26_public_return test_writer26_inventory test_writer26_loader test_writer26_group_build
```
