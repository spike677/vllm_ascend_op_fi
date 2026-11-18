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
attempt. Python failure diagnostics and public tensor retention are a separate
follow-up; the original failed artifacts are preserved.
