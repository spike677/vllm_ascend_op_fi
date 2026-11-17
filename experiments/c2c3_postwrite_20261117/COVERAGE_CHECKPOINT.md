# Postwrite coverage checkpoint

Natural C1 implementation commits are 69e1a18c and db36372f. Collector ownership
repair is 3fea52d. Root owns actual NPU runs and qualification review.

Actual a1 feedback: HF Qwen39 B/F returned/recovered, with a real aten.matmul
return and mul consumer; C2/C3 were not attempted. Root's raw audit v2 found the
first observed Max 536 -> Inf, 228 preceding nulls and no incomparable preceding
fields, but 7492 later NaNs keep whole E5 false. Nulls are not equality evidence.

41 a1 failed during collector-wrapper cleanup after successful baseline phases.
The original failure remains evidence. Root's a2 retry completed B/F, exact OFF,
separate same-worker health and zero leases. Its independent raw audit aligned:
first observed downstream Functional.linear.1.forward/output/0.Max 3.0625 ->
2.96875, 138 preceding nulls, no preceding incomparable fields, 6072 total nulls
preserved. This locates a downstream visible linear output, not a KV root cause.

The additive Native `dcat_pw_read_span` was built by root without a device in
postwrite-build-a2 from source-a16. It reads a bounded span without diagnostic
Tensor ops inside the collector. No Native bytes changed in the group commits.
Its exact identity and root-only run preparation are in ROOT_GROUP_HANDOFF.md.

The current ProjectionCoverage implementation is a CPU-verified checkpoint:

- Separate admission-independent module/projection inventory, with two healthy
  repeats, all actual modules, rejected projections and explicit non-targets.
- Requires actual same-process C1 effect plus exact OFF before group creation.
- All-target Native lossless bridge; one logical lease per target, source/lifetime
  identities, common CoverageSession, once-per-target bounded fixed-input replays.
- One native lease remains active until the real forward ends. Later eligible
  calls stay in the denominator and report the common concurrency refusal.
- Per-target byte snapshots, actual consumers, full output, group OFF, real
  device synchronization and common group recovery are tracked separately.
- C3 consumes the live common C2 receipt and fails on the next eligible call when
  the one-lease concurrency boundary cannot satisfy every-match. No NaN or skipped
  call is silently reclassified to reduce the healthy denominator.

The root-queued natural39_coverage_probe.py is complete in 06abdf7. It binds actual
safetensors/config/model source SHA and natural_pretrained_model MODEL_LOCAL
scope, repeats C1 in-process, and runs identical bounded B/F C2 workloads.

The 41 common group implementation is 5a3bd60:

- Two new healthy requests record an independent module/writer/consumer universe
  after same-process C1. Unknown Attention implementations refuse completeness.
- All-target lossless checks actual slot and query bytes, actual consumer output,
  original-call count, exact whole-model output and Native restore/release.
- C2 uses two first-decode requests: each target is injected once on the first;
  the second remains in the denominator and passes through. B uses identity
  controls, records zero group hits and cannot issue a C2 proof.
- C3 consumes the live C2 proof, repeats all-target lossless and injects every
  actual eligible call over two new requests. Every slot lease drains before the
  next consumer, so one concurrent Native lease is sufficient for this boundary.
- Nonfinite query/slot input is retained as a failed eligible call; it is never
  silently removed. An external writer keeps the Native lease. A consumer error
  remains primary when recovery also fails.
- Exact per-target and whole-model OFF precede the common whole-group barrier.
  Owned APIs/hooks are removed before a separate normal-request health readout;
  only its exact full-model output permits common reservations to be released.
- Actual weights, loaded model Python source, source bundle, schema, worker,
  process, model/module/storage lifecycle, enqueued request and forward IDs are
  bound separately. All public qualification flags remain false.

60 related CPU tests pass, including real CPU model/operator execution, two-layer
C2/C3, identity controls, nonfinite and budget refusal/recovery, incomplete
universes, changed healthy input, primary-error preservation and health failure.
These are CPU contracts, not natural NPU C2/C3 acceptance. Root still needs the
fresh 39 and 41 group B/F runs, raw statistics comparison and qualification review.

Shared natural-model scope commit 28a3ca7 is locally cherry-picked as 56cd68ab;
root already has it, so do not replay that dependency.
