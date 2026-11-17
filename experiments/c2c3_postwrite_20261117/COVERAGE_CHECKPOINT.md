# Postwrite coverage checkpoint

Natural C1 implementation commits are 69e1a18c and db36372f. Collector ownership
repair is 3fea52d. Root owns actual NPU runs and qualification review.

Actual a1 feedback: HF Qwen39 B/F returned/recovered, with a real aten.matmul
return and mul consumer; C2/C3 were not attempted. The 41 baseline completed two
healthy observations, lossless, baseline and OFF with 28 layers and zero leases,
then failed during API-wrapper cleanup. The original a1 evidence is unchanged.
Root is retrying 41 B/F in a2 using the precise cleanup fix.

The additive Native `dcat_pw_read_span` needs a fresh source-matching CANN build
before coverage. It reads a bounded span without adding diagnostic Tensor ops
inside the collector window. Existing C1 operations do not require that export.

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

Six CPU coverage tests pass, including distinct same-shaped layers, every target
once, unchanged denominators, C3 refusal/recovery, incomplete modules, stale
inputs/weights and mandatory live C1/C2 gates. This is not NPU C2 qualification.

Required next work on this branch:

1. Add the root-queued HF coverage probe with actual safetensors/config/model
   source SHA, native source/build identity and the natural_pretrained_model
   scope. It must perform fresh C1 in the same process before C2.
2. Validate the full group on natural Qwen, including exact per-target OFF,
   bounded forward/time budgets, 140-target discovery if it remains stable,
   raw B/F statistics and retained incomplete/nonfinite classifications.
3. Wire the same common group contracts to Postwrite41WorkerProvider after the
   real a2 result. Use actual Worker/request identities and sequential per-call
   live-slot leases. All-target lossless and repeated first-decode requests need
   genuine evidence before granting C2 or trying C3.
4. Review any group/observer drift against actual runtime evidence. Do not
   convert this checkpoint, CPU receipts or Native build success into acceptance.

Shared natural-model scope commit 28a3ca7 is locally cherry-picked as 56cd68ab;
root already has it, so do not replay that dependency.
