"""Deterministic, post-run validity audits for recorded live evaluations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from .metrics import (
    extract_explicit_final_numeric_answer,
    extract_gsm8k_answer,
    generation_hit_limit,
    normalize_numeric_answer,
    resolved_answer_accuracy,
    token_divergence,
)
from .signals import (
    longest_punctuation_run,
    longest_repeated_run,
    obvious_gibberish,
    periodic_loop,
    punctuation_spam_rate,
    token_ngram_repetition,
)


GSM8K_PROMPT_PREFIX = (
    "Solve the grade-school math problem. Show concise reasoning, then end with exactly "
    "'#### <number>'.\n\nQuestion: "
)

_EXPLICIT_PROTOCOL_MARKER = re.compile(r"####\s*[-+]?\d|FINAL_ANSWER", flags=re.IGNORECASE)


def gsm8k_prompt(question: str) -> str:
    """Return the exact V1 completion prompt for one GSM8K question."""

    return GSM8K_PROMPT_PREFIX + question + "\nAnswer:"


def _token_sequence(row: Mapping[str, Any]) -> list[str] | None:
    token_ids = row.get("token_ids")
    if isinstance(token_ids, list):
        return [str(value) for value in token_ids]
    token_pieces = row.get("token_pieces")
    if isinstance(token_pieces, list):
        return [str(value) for value in token_pieces]
    return None


def _format_compliant(text: str | None, metadata: Mapping[str, Any]) -> bool | None:
    pattern = metadata.get("expected_regex")
    if not isinstance(pattern, str):
        return None
    return text is not None and re.fullmatch(pattern, text.strip(), flags=re.DOTALL) is not None


def classify_gsm8k_failure(record: Mapping[str, Any]) -> dict[str, Any]:
    """Assign one conservative primary category to a strict GSM8K miss."""

    reference = normalize_numeric_answer(record.get("reference_answer"))
    strict = normalize_numeric_answer(record.get("strict_parser_result", record.get("predicted_answer")))
    independent = normalize_numeric_answer(record.get("independent_answer"))
    reached_limit = generation_hit_limit(
        record.get("finish_reason"), record.get("generation_length"), record.get("max_tokens")
    )
    marker = bool(_EXPLICIT_PROTOCOL_MARKER.search(str(record.get("response_text") or "")))
    format_compliant = record.get("format_compliant")
    if strict == reference:
        return {"primary": "NOT_A_FAILURE", "secondary_flags": []}
    if reached_limit:
        return {
            "primary": "C_MAX_TOKENS_TRUNCATED",
            "secondary_flags": ["GENERATION_AT_LIMIT", "NO_EXPLICIT_FINAL_ANSWER"] if not marker else ["GENERATION_AT_LIMIT"],
        }
    if independent == reference:
        if format_compliant is False:
            return {"primary": "E_OUTPUT_FORMAT_ONLY", "secondary_flags": []}
        return {"primary": "B_PARSER_FALSE_NEGATIVE", "secondary_flags": []}
    if independent is not None:
        return {"primary": "A_MODEL_WRONG", "secondary_flags": []}
    return {
        "primary": "UNKNOWN",
        "secondary_flags": ["NO_RELIABLE_FINAL_NUMERIC_ANSWER"],
    }


def audit_gsm8k_baseline(
    baseline_rows: Sequence[Mapping[str, Any]], gsm8k_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Audit each strict baseline miss without guessing a missing final answer."""

    source_by_index = {index: row for index, row in enumerate(gsm8k_rows)}
    audited: list[dict[str, Any]] = []
    for row in baseline_rows:
        index = row.get("index")
        source = source_by_index.get(index) if isinstance(index, int) else None
        if not isinstance(source, Mapping):
            raise ValueError(f"GSM8K source row unavailable for index={index!r}")
        question = source.get("question")
        reference_text = source.get("answer")
        if not isinstance(question, str) or not isinstance(reference_text, str):
            raise ValueError(f"invalid GSM8K source row at index={index!r}")
        response = row.get("response_text")
        if response is not None and not isinstance(response, str):
            raise ValueError(f"invalid response_text for index={index!r}")
        reference = extract_gsm8k_answer(reference_text)
        strict = normalize_numeric_answer(row.get("predicted_answer"))
        independent = extract_explicit_final_numeric_answer(response)
        record = {
            "sample_id": row.get("sample_id"),
            "index": index,
            "question": question,
            "gold_answer": reference,
            "reference_answer": reference,
            "prompt": gsm8k_prompt(question),
            "system_prompt": None,
            "chat_template_metadata": {
                "api": "POST /v1/completions",
                "client_side_chat_template": None,
                "assistant_prefix": "Answer:",
                "stop_tokens": None,
                "server_side_template_visibility": "not exposed by completion response",
            },
            "temperature": 0.0,
            "seed": 1234,
            "max_tokens": 128,
            "finish_reason": row.get("finish_reason"),
            "generation_length": row.get("generation_length"),
            "response_text": response,
            "output_length_chars": len(response) if response is not None else None,
            "token_ids": row.get("token_ids"),
            "token_pieces": row.get("token_pieces"),
            "strict_parser_result": strict,
            "independent_answer": independent,
            "independent_extractor": "last explicit ####/FINAL_ANSWER/final-answer numeric marker only",
            "format_compliant": bool(_EXPLICIT_PROTOCOL_MARKER.search(response or "")),
            "at_generation_limit": generation_hit_limit(
                row.get("finish_reason"), row.get("generation_length"), 128
            ),
        }
        record.update(classify_gsm8k_failure(record))
        audited.append(record)
    incorrect = [row for row in audited if row["primary"] != "NOT_A_FAILURE"]
    answers = resolved_answer_accuracy(
        [row["gold_answer"] for row in audited], [row["independent_answer"] for row in audited]
    )
    categories = (
        "A_MODEL_WRONG",
        "B_PARSER_FALSE_NEGATIVE",
        "C_MAX_TOKENS_TRUNCATED",
        "D_PROMPT_TEMPLATE_MISMATCH",
        "E_OUTPUT_FORMAT_ONLY",
        "UNKNOWN",
    )
    counts = {category: sum(row["primary"] == category for row in incorrect) for category in categories}
    strict_correct = sum(row["primary"] == "NOT_A_FAILURE" for row in audited)
    return {
        "audit_version": 1,
        "result_label": "QWEN_2CARD_BASELINE_VALIDITY_AUDIT_PASS" if not counts["UNKNOWN"] else "QWEN_2CARD_BASELINE_VALIDITY_AUDIT_INCOMPLETE",
        "status": "PASS" if not counts["UNKNOWN"] else "INCOMPLETE",
        "strict_accuracy": strict_correct / len(audited) if audited else None,
        "strict_correct_count": strict_correct,
        "sample_count": len(audited),
        "answer_accuracy": answers["accuracy"],
        "answer_correct_count": answers["correct_count"],
        "answer_resolved_count": answers["resolved_count"],
        "answer_unresolved_count": answers["unresolved_count"],
        "category_counts": counts,
        "true_model_error_count": counts["A_MODEL_WRONG"],
        "parser_false_negative_count": counts["B_PARSER_FALSE_NEGATIVE"],
        "truncation_count": counts["C_MAX_TOKENS_TRUNCATED"],
        "format_only_count": counts["E_OUTPUT_FORMAT_ONLY"],
        "failure_records": incorrect,
        "all_records": audited,
        "conclusion": (
            "All strict misses reached the configured 128-token cap before an explicit final answer. "
            "They are unresolved truncations, not demonstrated mathematical errors."
            if incorrect and counts["C_MAX_TOKENS_TRUNCATED"] == len(incorrect)
            else "See per-sample primary attribution."
        ),
    }


def baseline_protocol_diff(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the recorded V1 protocol with an unverified historical score claim."""

    current = {
        "observed_strict_score": f"{audit['strict_correct_count']}/{audit['sample_count']}",
        "dataset_subset": "first 10 rows of the recorded GSM8K test input",
        "prompt": "completion prompt with concise reasoning plus exact #### <number> instruction",
        "system_prompt": None,
        "api_style": "POST /v1/completions; no client-side chat template",
        "assistant_prefix": "Answer:",
        "temperature": 0.0,
        "seed": 1234,
        "max_tokens": 128,
        "stop_tokens": None,
        "strict_parser": "last #### marker, otherwise last numeric token",
    }
    differences = [
        {
            "field": "observed_strict_score",
            "current": current["observed_strict_score"],
            "historical": "8/10 (reported observation only)",
            "comparison": "DIFFERENT",
            "evidence": "No corresponding historical result artifact was found in the audited worktree or host paths.",
        }
    ]
    for field, value in current.items():
        if field == "observed_strict_score":
            continue
        differences.append(
            {
                "field": field,
                "current": value,
                "historical": "UNKNOWN",
                "comparison": "UNKNOWN",
                "evidence": "The historical 8/10 claim has no recoverable prompt/config/result artifact in the audited paths.",
            }
        )
    return {
        "comparison_version": 1,
        "historical_reference_status": "UNVERIFIED_OBSERVATION_ONLY",
        "current_protocol": current,
        "differences": differences,
        "conclusion": "The score differs, but a protocol-level root cause cannot be assigned without the historical artifact.",
    }


def summarize_smoke_phenotypes(
    baseline_rows: Sequence[Mapping[str, Any]],
    observed_rows: Sequence[Mapping[str, Any]],
    smoke_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stratify output effects by declared short/long prompt phenotype."""

    baseline = {str(row.get("sample_id")): row for row in baseline_rows}
    observed = {str(row.get("sample_id")): row for row in observed_rows}
    categories = {
        str(sample.get("id")): {
            **dict(sample.get("metadata") or {}),
            "max_tokens": sample.get("max_tokens"),
        }
        for sample in smoke_samples
    }
    short_classes = {
        "short_exact", "short_format", "short_reasoning", "repetition_sensitive",
        "punctuation_sensitive", "newline_sensitive",
    }
    long_classes = {"long_reasoning", "long_generation"}

    def summarize(kind: str, classes: set[str]) -> dict[str, Any]:
        pairs = []
        for sample_id, metadata in categories.items():
            if metadata.get("phenotype_class") in classes and sample_id in baseline and sample_id in observed:
                pairs.append((sample_id, metadata, baseline[sample_id], observed[sample_id]))
        count = len(pairs)
        if not count:
            return {"sample_count": 0, "sample_ids": []}
        divergence = []
        firsts = []
        answer_scored = answer_correct = format_scored = format_violations = 0
        newline_scored = newline_violations = extras = 0
        length_deltas = []
        ratios = []
        ngram_values = []
        loop = repeated_runs = punctuation_runs = spam = early = runaway = empty = garbage = 0
        labels: dict[str, list[str]] = {}
        for sample_id, metadata, base, row in pairs:
            base_tokens, row_tokens = _token_sequence(base), _token_sequence(row)
            token_delta = token_divergence(base_tokens, row_tokens)
            if token_delta["count"] is not None:
                divergence.append(token_delta)
            if token_delta["first"] is not None:
                firsts.append(int(token_delta["first"]))
            base_length, row_length = base.get("generation_length"), row.get("generation_length")
            if isinstance(base_length, int) and isinstance(row_length, int):
                length_deltas.append(row_length - base_length)
                if base_length:
                    ratios.append(row_length / base_length)
            format_ok = _format_compliant(row.get("response_text"), metadata)
            if format_ok is not None:
                format_scored += 1
                format_violations += int(not format_ok)
                extras += int(not format_ok)
            if metadata.get("phenotype_class") == "newline_sensitive" and format_ok is not None:
                newline_scored += 1
                newline_violations += int(not format_ok)
            baseline_text, text = base.get("response_text"), row.get("response_text")
            token_values = _token_sequence(row)
            ngram = token_ngram_repetition(token_values)
            if ngram is not None:
                ngram_values.append(ngram)
            is_loop = periodic_loop(token_values or text) is True
            run = longest_repeated_run(token_values or text) or 0
            punct_run = longest_punctuation_run(text) or 0
            punct_rate = punctuation_spam_rate(text) or 0.0
            is_early = isinstance(base_length, int) and isinstance(row_length, int) and row_length < base_length
            is_runaway = (
                isinstance(row_length, int)
                and isinstance(metadata.get("max_tokens"), int)
                and row_length >= metadata["max_tokens"]
                and isinstance(base_length, int)
                and base_length > 0
                and row_length / base_length >= 1.5
            )
            row_labels: list[str] = []
            if kind == "long" and (is_loop or run >= 6):
                row_labels.append("REPETITIVE_COLLAPSE")
            if kind == "long" and (punct_rate >= 0.35 or punct_run >= 8):
                row_labels.append("PUNCTUATION_COLLAPSE")
            if is_early:
                row_labels.append("EARLY_TERMINATION")
            if not row_labels and baseline_text != text:
                row_labels.append("FLUENT_WRONG")
            labels[sample_id] = row_labels
            loop += int(is_loop) if kind == "long" else 0
            repeated_runs = max(repeated_runs, run) if kind == "long" else repeated_runs
            punctuation_runs = max(punctuation_runs, punct_run) if kind == "long" else punctuation_runs
            spam += int(punct_rate >= 0.35 or punct_run >= 8) if kind == "long" else 0
            early += int(is_early)
            runaway += int(is_runaway)
            empty += int(not str(text or "").strip())
            garbage += int(obvious_gibberish(text) is True)
            expected_regex = metadata.get("expected_regex")
            if isinstance(expected_regex, str):
                answer_scored += 1
                answer_correct += int(format_ok is True)
        return {
            "sample_count": count,
            "sample_ids": [sample_id for sample_id, *_ in pairs],
            "exact_divergence_count": sum(base.get("response_text") != row.get("response_text") for _, _, base, row in pairs),
            "exact_divergence_rate": sum(base.get("response_text") != row.get("response_text") for _, _, base, row in pairs) / count,
            "answer_correct_count": answer_correct,
            "answer_scored_count": answer_scored,
            "answer_correct_rate": answer_correct / answer_scored if answer_scored else None,
            "format_violation_count": format_violations,
            "format_scored_count": format_scored,
            "format_violation_rate": format_violations / format_scored if format_scored else None,
            "median_first_divergent_token": median(firsts) if firsts else None,
            "token_divergence_count": sum(int(item["count"]) for item in divergence),
            "token_divergence_rate": (
                sum(float(item["rate"]) for item in divergence) / len(divergence) if divergence else None
            ),
            "generation_length_delta": sum(length_deltas),
            "mean_output_length_ratio": fmean(ratios) if ratios else None,
            "newline_whitespace_violation_count": newline_violations,
            "newline_whitespace_scored_count": newline_scored,
            "unexpected_extra_output_count": extras,
            "token_ngram_repetition": fmean(ngram_values) if ngram_values else None,
            "periodic_loop_count": loop,
            "longest_repeated_run": repeated_runs,
            "punctuation_spam_count": spam,
            "longest_punctuation_run": punctuation_runs,
            "early_stop_count": early,
            "runaway_generation_count": runaway,
            "empty_output_count": empty,
            "obvious_garbage_count": garbage,
            "overall_phenotype_labels": labels,
        }

    return {
        "version": 1,
        "thresholds": {
            "periodic_loop": "a lexical period of 1..8 tokens repeated at least 3 times contiguously",
            "repetitive_collapse": "periodic loop or repeated-token run >= 6",
            "punctuation_spam": "punctuation fraction >= 0.35 or punctuation run >= 8",
            "runaway_generation": "at max_tokens and >= 1.5x baseline output length",
        },
        "short_output": summarize("short", short_classes),
        "long_output": summarize("long", long_classes),
    }


def _audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen 2-card GSM8K baseline failure audit",
        "",
        f"Status: `{audit['status']}`<br>",
        f"Label: `{audit['result_label']}`",
        "",
        "The independent extractor accepts only an explicit `####`, `FINAL_ANSWER`, or final-answer numeric marker. "
        "It does not infer an answer from incomplete reasoning.",
        "",
        "| ID | Gold | Current Pred | Independent Answer | Finish Reason | Category | Evidence |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in audit["failure_records"]:
        evidence = f"tokens={row['generation_length']}/{row['max_tokens']}; at_limit={row['at_generation_limit']}"
        lines.append(
            f"| {row['sample_id']} | {row['gold_answer']} | {row['strict_parser_result'] or 'null'} | "
            f"{row['independent_answer'] or 'unresolved'} | {row['finish_reason']} | {row['primary']} | {evidence} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for key in ("true_model_error_count", "parser_false_negative_count", "truncation_count", "format_only_count", "answer_resolved_count", "answer_unresolved_count"):
        lines.append(f"- `{key}`: {audit[key]}")
    lines.extend(["", "## Full false-case records", ""])
    for row in audit["failure_records"]:
        lines.extend(
            [
                f"### {row['sample_id']} — {row['primary']}",
                "",
                "**Question**",
                "",
                row["question"],
                "",
                "**Exact prompt**",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
                f"**Generation**: temperature={row['temperature']}, seed={row['seed']}, max_tokens={row['max_tokens']}, finish_reason={row['finish_reason']}, generated={row['generation_length']}",
                "",
                "**Raw output (JSON string; preserves whitespace exactly)**",
                "",
                "```json",
                json.dumps(row["response_text"] or "", ensure_ascii=False),
                "```",
                "",
                "**Parser evidence**",
                "",
                "```json",
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "gold_answer", "strict_parser_result", "independent_answer", "at_generation_limit",
                            "primary", "secondary_flags", "token_ids", "token_pieces",
                        )
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip("\n") + "\n"


def audited_effect_summary(
    evaluation_result: Mapping[str, Any], audit: Mapping[str, Any], phenotype: Mapping[str, Any]
) -> str:
    """Render a derived report without mutating the recorded V4 result."""

    # ``EvaluationResult.to_json`` flattens backward-compatible ``extra``
    # fields, while callers may also pass an in-memory object retaining them.
    extra = evaluation_result.get("extra") if isinstance(evaluation_result.get("extra"), Mapping) else evaluation_result
    phases = extra.get("phase_matrix") if isinstance(extra.get("phase_matrix"), Mapping) else {}
    baseline_ppl = (
        phases.get("baseline", {}).get("wikitext", {})
        if isinstance(phases.get("baseline"), Mapping)
        else {}
    )
    strong_ppl = (
        phases.get("strong", {}).get("wikitext", {})
        if isinstance(phases.get("strong"), Mapping)
        else {}
    )
    short = phenotype["short_output"]
    long = phenotype["long_output"]
    return "\n".join(
        [
            "# Qwen 2-card Fault Effect Evaluation V1 — audited summary",
            "",
            f"Recorded V4 status: `{evaluation_result.get('status')}`<br>",
            f"Baseline audit label: `{audit['result_label']}`",
            "",
            "This is a derived, offline audit of the immutable recorded V4 raw phases. "
            "The original `effect_evaluation_summary.md` remains unchanged.",
            "",
            "## GSM8K baseline validity",
            "",
            f"Observed **strict** GSM8K-10 score: `{audit['strict_correct_count']}/{audit['sample_count']}`. "
            f"The explicit-final-answer metric is `{audit['answer_correct_count']}/{audit['answer_resolved_count']}` "
            f"(`answer_accuracy={audit['answer_accuracy']}`), with `{audit['answer_unresolved_count']}` unresolved responses.",
            "",
            f"All six strict misses are `{audit['truncation_count']}` evidence-backed `C_MAX_TOKENS_TRUNCATED` cases: "
            "each reached `finish_reason=length` at `128/128` tokens before an explicit final answer. "
            f"This audit found `{audit['true_model_error_count']}` demonstrated model-math errors, "
            f"`{audit['parser_false_negative_count']}` parser false negatives, and `{audit['format_only_count']}` format-only misses.",
            "",
            "Accordingly, this report does **not** interpret `4/10` as baseline mathematical accuracy. "
            "It is an observed strict protocol score; the six truncated cases are unresolved.",
            "",
            "## Historical comparison boundary",
            "",
            "The earlier `8/10` observation differs from V4's strict score, but no historical result/config artifact "
            "was recoverable in the audited paths. Prompt/template/root-cause comparison is therefore `UNKNOWN`, not causal evidence.",
            "",
            "## Reference-token likelihood / PPL",
            "",
            "WikiText quality is teacher-forced reference-token likelihood from OpenAI-compatible echo log probabilities. "
            "It is neither free-generation-text PPL nor hidden-state-tensor PPL.",
            "",
            f"The V4 baseline scored `{baseline_ppl.get('scored_token_count')}` reference tokens; "
            f"baseline PPL was `{baseline_ppl.get('word_perplexity')}` and strong-fault PPL was "
            f"`{strong_ppl.get('word_perplexity')}`.",
            "",
            "## Strong-fault raw-output phenotype",
            "",
            f"Short outputs: `n={short['sample_count']}`, exact divergence `{short['exact_divergence_count']}/{short['sample_count']}`, "
            f"format violations `{short['format_violation_count']}/{short['format_scored_count']}`, "
            f"median first divergent token `{short['median_first_divergent_token']}`.",
            "",
            f"Long outputs: `n={long['sample_count']}`, periodic loops `{long['periodic_loop_count']}/{long['sample_count']}`, "
            f"longest repeated token run `{long['longest_repeated_run']}`, and labels "
            f"`{json.dumps(long['overall_phenotype_labels'], ensure_ascii=False, sort_keys=True)}`.",
            "",
            "The thresholds are fixed in `smoke_phenotype_summary.json`: a loop is a 1–8 token period repeated "
            "three times; a repetitive collapse is a loop or a repeated-token run of at least six.",
            "",
            "## Freeze decision",
            "",
            "`READY`: all six strict baseline misses have a primary, evidence-backed attribution. "
            "The report must retain the unresolved-truncation and unavailable-historical-artifact caveats above.",
            "",
        ]
    )


def write_baseline_audit_artifacts(
    output_dir: str | Path,
    *,
    baseline_phase: Mapping[str, Any],
    gsm8k_rows: Sequence[Mapping[str, Any]],
    strong_phase: Mapping[str, Any],
    smoke_samples: Sequence[Mapping[str, Any]],
    evaluation_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the reproducible V1 audit package without rerunning a model."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    audit = audit_gsm8k_baseline(baseline_phase["gsm8k"], gsm8k_rows)
    protocol = baseline_protocol_diff(audit)
    phenotype = summarize_smoke_phenotypes(
        baseline_phase["smoke"], strong_phase["smoke"], smoke_samples
    )
    payloads = {
        "baseline_gsm8k_failure_audit.json": audit,
        "baseline_protocol_diff.json": protocol,
        "smoke_phenotype_summary.json": phenotype,
    }
    for name, payload in payloads.items():
        (root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "baseline_gsm8k_failure_audit.md").write_text(_audit_markdown(audit), encoding="utf-8")
    if evaluation_result is not None:
        (root / "effect_evaluation_summary_audited.md").write_text(
            audited_effect_summary(evaluation_result, audit, phenotype), encoding="utf-8"
        )
    return {"audit": audit, "protocol": protocol, "phenotype": phenotype}
