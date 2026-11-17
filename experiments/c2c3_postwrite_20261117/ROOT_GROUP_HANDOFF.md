# Root-only natural postwrite group handoff

No NPU or container calls were made by this agent. All generation scripts only
write case JSON; root runs its shared queue under `/var/tmp/fault22_npu_2_5.lock`.
The Python case generators pass root's queue_contract.validate_cases with 2 cases.

## Source and Native identity

39 group core/probe: bd3ae56 then 06abdf7. 41 group core: 5a3bd60, followed by this
probe/handoff commit. Shared scope dependencies are already present on root and
must not be replayed. Root already integrated Native 29f706d as 54471a4c.

Use a fresh pinned source bundle containing these commits. Its actual source SHA
manifest is emitted by each probe. Do not reuse source-a15/a16 as the new Python
group implementation. The Native .cpp/.h bytes remain identical to source-a16.

Root-provided CANN 8.5.1 a2 Native directory:

```text
/home/17-fault-c2c3-20261117/postwrite-build-a2/cache/e81dfedf9f3356c8e861baace93192fdb49a12d8cf456e7cc8bc41c3b74ee4a8/attempt-d4098782389e467ab57a0b0ae2582229/native
```

`libdcat_postwrite.so` SHA256:
`3727fe2ed60afebb7764609962a4a32a710c70d97561464c788e910d7320dc3e`

Build fingerprint:
`e81dfedf9f3356c8e861baace93192fdb49a12d8cf456e7cc8bc41c3b74ee4a8`

Both generators require `--source` set to that fresh root-pinned source directory,
`--native` set to the directory above, `--sha256` and `--build-fingerprint` exactly
as above, `--model /home/gyl/models/Qwen3-0.6B`, `--cards 5` (or root's reserved
card 2), and a new `--out` JSON path. They never overwrite an existing manifest.

## 39 cases

```text
python3 experiments/c2c3_postwrite_20261117/prepare_coverage_cases.py [identity/path arguments above]
```

Cases are postwrite39-c2-baseline and postwrite39-c2-fault. Both execute fresh C1
and exact OFF in-process, then independent all-target lossless and bounded C2.
The previous natural C1 had 140 eligible calls; the new run must rediscover its
denominator. At most one target is mutated per fixed-input forward, with every
remaining actual projection call still counted. Max forwards equals eligible
target count; default session 1200 seconds, case 1800 seconds. Model scope is
natural_pretrained_model / MODEL_LOCAL, never real_worker.

B applies identity control once per target and has zero group hits. F must hit
every independently eligible target once. Only F then attempts C3 using its live
C2 proof; the forward-long single Native lease is expected to refuse every-match
at the next eligible call. The refusal and fresh OFF/group recovery are recorded.
Nonfinite downstream tensors and statistics remain visible; no NaN filtering.

## 41 cases

```text
python3 experiments/c2c3_postwrite_20261117/prepare41_coverage_cases.py [identity/path arguments above]
```

Cases are postwrite41-c2-baseline and postwrite41-c2-fault. Each uses the standard
serial eager TP1/PP1/DP1 Worker, BF16 first pure decode, max_tokens=2, prefix caching
off, 64 blocks, and at most 28 expected Attention targets. The probe refuses a
different actual universe; this bound is not used as a fabricated denominator.
Default group session 600 seconds, case 1800 seconds, 32 GiB/8 CPU queue budget.

Each arm runs fresh C1 observe/observe/lossless/fault/OFF before group admission.
Thus each arm contains one setup C1 hit; only C2 is measured in the B/F capture.
After two fresh independent healthy group requests and all-target lossless:

| Phase | Requests | Expected group behavior with a verified 28-target universe |
|---|---:|---|
| B C2 | 2 | 28 identity controls, 0 hits; 56 eligible calls; no C2 proof |
| F C2 | 2 | 28 hits exactly once; all 56 actual eligible calls counted |
| F C3 | 2 | 56 hits; new lossless proof; live C2 proof mandatory |
| OFF + health | 2 per group | exact per-target and full-model OFF, then normal request after uninstall |

Each Native lease holds only the actual just-written live V slot through its real
TND FIA consumer and restores before the next target. The mechanism remains a
controlled legal-address tail-column proxy; it makes no out-of-bounds claim.

Capture is raw statistics/mix, identical steps [0,1] for the two C2 requests in B
and F. C1, discovery, lossless, OFF, health and C3 are outside the capture. Native
read_span evidence adds no diagnostic Tensor operations. Root should preserve raw
artifacts on 124 and compare fixed rtol=1e-5, atol=1e-8, retaining null/NaN/mismatch.
Independent first differences may locate downstream visibility; they do not
establish a KV root cause from an injector hit log.

## Remaining gates

CPU success does not prove fresh NPU snapshot shape/format acceptance, actual FIA
schema availability, repeated slot/query determinism, all-target effects, or
collector compatibility across repeated start/stop and reinstall. Any failure
keeps actual evidence and leaves common qualification false. Native recovery
failure retains the lease; health failure retains common target reservations.
Cleanup failure is recorded separately without replacing the original failure.
No group is accepted until root inspects the fresh actual receipts and raw audit.
