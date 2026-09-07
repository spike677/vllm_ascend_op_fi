# Qwen 2-card Fault Effect Evaluation V1 — audited summary

Recorded V4 status: `PASS`<br>
Baseline audit label: `QWEN_2CARD_BASELINE_VALIDITY_AUDIT_PASS`

This is a derived, offline audit of the immutable recorded V4 raw phases. The original `effect_evaluation_summary.md` remains unchanged.

## GSM8K baseline validity

Observed **strict** GSM8K-10 score: `4/10`. The explicit-final-answer metric is `4/4` (`answer_accuracy=1.0`), with `6` unresolved responses.

All six strict misses are `6` evidence-backed `C_MAX_TOKENS_TRUNCATED` cases: each reached `finish_reason=length` at `128/128` tokens before an explicit final answer. This audit found `0` demonstrated model-math errors, `0` parser false negatives, and `0` format-only misses.

Accordingly, this report does **not** interpret `4/10` as baseline mathematical accuracy. It is an observed strict protocol score; the six truncated cases are unresolved.

## Historical comparison boundary

The earlier `8/10` observation differs from V4's strict score, but no historical result/config artifact was recoverable in the audited paths. Prompt/template/root-cause comparison is therefore `UNKNOWN`, not causal evidence.

## Reference-token likelihood / PPL

WikiText quality is teacher-forced reference-token likelihood from OpenAI-compatible echo log probabilities. It is neither free-generation-text PPL nor hidden-state-tensor PPL.

The V4 baseline scored `56` reference tokens; baseline PPL was `276.79339315101316` and strong-fault PPL was `14720553.587854471`.

## Strong-fault raw-output phenotype

Short outputs: `n=19`, exact divergence `19/19`, format violations `18/18`, median first divergent token `1`.

Long outputs: `n=1`, periodic loops `1/1`, longest repeated token run `10`, and labels `{"long-generation": ["REPETITIVE_COLLAPSE"]}`.

The thresholds are fixed in `smoke_phenotype_summary.json`: a loop is a 1–8 token period repeated three times; a repetitive collapse is a loop or a repeated-token run of at least six.

## Freeze decision

`READY`: all six strict baseline misses have a primary, evidence-backed attribution. The report must retain the unresolved-truncation and unavailable-historical-artifact caveats above.
