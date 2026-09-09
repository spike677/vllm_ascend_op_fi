# Qwen two-card FakeRequant: current measured effect

This is a preserved English companion to the Chinese canonical report. It covers the completed private evaluation only; it does not re-run a model and it does not claim an effect for another model or configuration.

## Mechanism

All 66 of 66 fault transactions had valid mechanism evidence. Rank 0 reached the final 4096-hit request budget with changed elements recorded. Rank 1 had zero hits in every transaction and only the expected `rank_filter` rejection.

## Probability effect

Each of three teacher-forced runs scored 189 reference tokens with the same result:

| Metric | Baseline | FakeRequant | Delta |
| --- | ---: | ---: | ---: |
| NLL | 544.7059 | 556.7009 | +11.9950 |
| PPL | 17.8507 | 19.0203 | 1.0655x |

This is stable probability degradation for the measured configuration, not evidence for other models.

## Output and task boundaries

Short probes diverged in 3/4 responses per cycle; long probes diverged in 5/5. Their recovery responses were exactly equal to baseline. No fault-induced repetition, punctuation spam, early stop, runaway generation, empty output, obvious garbage, or crash was observed.

GSM8K-10 was `10/10 → 9/10 → 9/10` in cycle 1 and `9/10 → 9/10 → 9/10` in cycles 2–3. Task damage was therefore not repeatably demonstrated. This must not be reported as a 10% GSM8K degradation. The full recovery gate remains `PARTIAL` because the clean task baseline varied.
