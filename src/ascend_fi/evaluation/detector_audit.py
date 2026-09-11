"""Offline verification of frozen detectors, never a second scoring system.

Synthetic regression expectations and independently reviewed real labels have
separate denominators.  A declaration of human review is provenance supplied by
the caller, not something this tool can authenticate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .distortion.contracts import DistortionProbe, content_hash, load_suite
from .distortion.evaluator import SCORER_VERSION, compare
from .distortion.detectors import DETECTORS

SCHEMA = "DETECTOR_AUDIT_V1"


def interval_union(spans, length):
    """Half-open intervals in one explicitly supplied coordinate system."""
    if type(length) is not int or length < 0:
        raise ValueError("invalid coordinate length")
    checked = []
    for span in spans:
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ValueError("interval must contain start/end")
        begin, end = span
        if not (type(begin) is type(end) is int and 0 <= begin < end <= length):
            raise ValueError("interval is outside the declared text/token sequence")
        checked.append([begin, end])
    merged = []
    for begin, end in sorted(checked):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([begin, end])
    covered = sum(end - begin for begin, end in merged)
    return {"intervals": merged, "covered": covered,
            "fraction": covered / length if length else None,
            "first_observed_offset": merged[0][0] if merged else None}


def _locations(comparison, response):
    value = response["response_text"]
    char_spans, token_spans = [], []
    token_basis = None
    for result in comparison["detectors"]:
        if result["triggered"] is not True:
            continue
        evidence = result["evidence"]
        fragment = evidence.get("fault_fragment") or {}
        for position in fragment.get("positions", []):
            if position.get("character_span") is not None:
                char_spans.append(position["character_span"])
            if fragment.get("comparison_basis") == "token_pieces":
                token_spans.append(position["token_span"])
                token_basis = "token_pieces"
        punctuation = evidence.get("fault_punctuation") or {}
        if punctuation.get("longest_run_character_span"):
            char_spans.append(punctuation["longest_run_character_span"])
        if result["detector"] == "garbage":
            for index in (result.get("fault_value") or {}).get("bad_character_positions", []):
                char_spans.append([index, index + 1])
    char_result = interval_union(char_spans, len(value))
    char_result["unit"] = "unicode_code_points"
    pieces = response.get("token_pieces")
    token_result = interval_union(token_spans, len(pieces)) if token_basis and isinstance(pieces, list) else None
    if token_result is not None:
        token_result["unit"] = "token_pieces"
    return {"characters": char_result, "tokens": token_result, "token_ids": None,
            "scope": "DETECTOR_RETAINED_EVIDENCE_LOWER_BOUND",
            "not_complete_pathology_coverage": True,
            "first_offset_is_earliest_retained_evidence_not_proven_first_anomaly": True}


def _full_response(row, *, recorded):
    if not isinstance(row, dict) or not isinstance(row.get("response_text"), str):
        raise ValueError("complete response_text required, not a dashboard excerpt")
    if row.get("text_is_complete") is not True:
        raise ValueError("text_is_complete=true required; generation may still end at length")
    value = row["response_text"]
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    if row.get("text_sha256") not in (None, digest):
        raise ValueError("raw text SHA256 mismatch")
    if recorded and not row.get("source_ref"):
        raise ValueError("real model output requires original source_ref")
    return {**row, "text_sha256": digest}


def _trust(label, recorded):
    # Do not promote agent/fixture/candidate annotations to human gold labels.
    if not recorded:
        return False, "SYNTHETIC_REGRESSION_NOT_REAL_QUALITY_EVIDENCE"
    if label.get("source") != "independent_human_review":
        return False, "LABEL_NOT_INDEPENDENT_HUMAN_REVIEW"
    if not (label.get("review_status") == "confirmed" and label.get("independent") is True
            and label.get("reviewer") and label.get("review_method") and label.get("evidence_note")):
        return False, "LABEL_PROVENANCE_INCOMPLETE"
    if label.get("baseline_eligible") is not True:
        return False, "BASELINE_NOT_ELIGIBLE_FOR_THIS_DETECTOR"
    if type(label.get("expected_trigger")) is not bool:
        return False, "LABEL_NOT_BINARY"
    return True, "DECLARED_INDEPENDENT_REVIEW"


def _counts():
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "case_ids": [], "groups": set()}


def _summarize(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {**{k: counts[k] for k in ("tp", "fp", "fn", "tn", "case_ids")},
            "denominator": tp + fp + fn + tn, "group_count": len(counts["groups"]),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None}


def audit_cases(cases, frozen_suite):
    """Score complete pairs, preserving protocol strata and prompt-group splits.

    ``frozen_suite`` must be the actual frozen suite file, not caller thresholds.
    Raw records are retained in the result. Existing live evidence is never
    edited. Labels can be multi-label and baseline eligibility is per detector.
    """
    suite = load_suite(frozen_suite)
    cases = list(cases)
    seen_ids, split_by_prompt, split_by_family = set(), {}, {}
    groups, outcomes, quality, holdout = {}, [], {}, {}
    quality_seen, deduplicated_quality_records = {}, 0
    regression_total = regression_pass = 0
    pending = []
    for case in cases:
        identity = case.get("case_id")
        if not isinstance(identity, str) or not identity or identity in seen_ids:
            raise ValueError("unique nonempty case_id required")
        seen_ids.add(identity)
        if case.get("source_kind") not in ("synthetic_regression", "recorded_model_output"):
            raise ValueError("source_kind must distinguish synthetic and recorded output")
        recorded = case["source_kind"] == "recorded_model_output"
        protocol = case.get("protocol")
        if not isinstance(protocol, str) or not protocol:
            raise ValueError("explicit protocol required; completion/chat must not be pooled silently")
        split = case.get("split", "regression")
        if split not in ("development", "check", "regression", "pending_review"):
            raise ValueError("unknown split")
        probe = DistortionProbe.from_dict(case["probe"])
        if set(probe.applicable_detectors) - set(DETECTORS):
            raise ValueError("unknown detector")
        original_key = case.get("original_case_key")
        if not probe.prompt and not original_key:
            raise ValueError("original prompt or explicit original_case_key required")
        identity_text = probe.prompt if probe.prompt else "recorded-case-key:" + original_key
        prompt_hash = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()
        family = case.get("case_family", prompt_hash)
        if not isinstance(family, str) or not family:
            raise ValueError("case_family must be nonempty")
        # All repeats of a prompt AND all members of a declared family remain
        # in one split, even when caller changes group names or transport.
        for mapping, key in ((split_by_prompt, prompt_hash), (split_by_family, family)):
            if key in mapping and mapping[key] != split:
                raise ValueError("prompt/family split leakage: " + identity)
            mapping[key] = split
        group = groups.setdefault(prompt_hash, {"prompt_sha256": prompt_hash if probe.prompt else None,
                                               "group_basis": "original_prompt" if probe.prompt else "original_case_key_prompt_missing",
                                               "original_case_key": original_key,
                                               "case_ids": [], "split": split, "protocols": set()})
        group["case_ids"].append(identity)
        group["protocols"].add(protocol)
        baseline = _full_response(case["baseline"], recorded=recorded)
        response = _full_response(case["response"], recorded=recorded)
        comparison = compare(probe, baseline, response, suite["thresholds"])
        observed = {row["detector"]: row for row in comparison["detectors"]}
        label_status = {}
        for detector, label in case.get("labels", {}).items():
            if detector not in DETECTORS:
                raise ValueError("unknown label detector")
            if not isinstance(label, dict):
                raise ValueError("label must contain provenance")
            if label.get("evidence_spans") is not None:
                interval_union(label["evidence_spans"], len(response["response_text"]))
            trusted, reason = _trust(label, recorded)
            predicted = observed.get(detector, {}).get("triggered")
            if observed.get(detector, {}).get("baseline_failed") is True:
                trusted, reason = False, "DETECTOR_REPORTS_BASELINE_FAILED"
            label_status[detector] = {"trusted_quality_subset": trusted, "reason": reason,
                                      "predicted_trigger": predicted, "provenance": label}
            regression_label = (not recorded and label.get("source") == "fixture_spec" or
                                recorded and label.get("source") == "historical_audit_expectation")
            if regression_label and type(label.get("expected_trigger")) is bool:
                regression_total += 1
                regression_pass += predicted is label["expected_trigger"]
            if trusted and type(predicted) is bool:
                truth = label["expected_trigger"]
                duplicate_key = (protocol, detector, prompt_hash,
                                 baseline["text_sha256"], response["text_sha256"])
                if duplicate_key in quality_seen:
                    if quality_seen[duplicate_key] != truth:
                        raise ValueError("conflicting trusted labels for the same complete pair")
                    deduplicated_quality_records += 1
                    label_status[detector]["quality_counted"] = False
                    label_status[detector]["quality_exclusion"] = "EXACT_PAIR_DUPLICATE_ACROSS_CYCLES"
                    continue
                quality_seen[duplicate_key] = truth
                label_status[detector]["quality_counted"] = True
                entry = "tp" if predicted and truth else "fp" if predicted else "fn" if truth else "tn"
                key = protocol + ":" + detector
                for collection in ([quality, holdout] if split == "check" and label.get("not_used_for_tuning") is True else [quality]):
                    count = collection.setdefault(key, _counts())
                    count[entry] += 1
                    count["case_ids"].append(identity)
                    count["groups"].add(family)
            else:
                pending.append({"case_id": identity, "detector": detector,
                                "reason": reason if not trusted else "DETECTOR_UNKNOWN_OR_NOT_APPLICABLE"})
        outcomes.append({"case_id": identity, "source_kind": case["source_kind"],
                         "protocol": protocol, "group": prompt_hash, "case_family": family,
                         "cycle_id": case.get("cycle_id"), "split": split,
                         "probe": probe.to_dict(), "baseline": baseline, "response": response,
                         "comparison": comparison, "labels": label_status,
                         "locations": _locations(comparison, response)})
    root = Path(__file__).with_name("distortion")
    return {"schema": SCHEMA, "scorer_version": SCORER_VERSION,
            "scorer_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("detectors.py", "evaluator.py", "contracts.py")},
            "frozen_suite_hash": suite["hash"], "threshold_hash": content_hash(suite["thresholds"]),
            "thresholds_changed": False, "case_count": len(outcomes),
            "groups": [{**group, "protocols": sorted(group["protocols"])} for group in groups.values()],
            "regression": {"expected_assertions": regression_total, "passed_assertions": regression_pass,
                           "status": "NOT_RUN" if not regression_total else "PASS" if regression_pass == regression_total else "FAIL"},
            "trusted_label_quality": {key: _summarize(count) for key, count in quality.items()},
            "independent_check_subset": {key: _summarize(count) for key, count in holdout.items()},
            "quality_deduplicated_exact_pairs": deduplicated_quality_records,
            "quality_denominator": "unique complete baseline/fault pairs per prompt/protocol/detector; family count reported separately",
            "quality_status": "DECLARED_INDEPENDENT_REVIEW_SUBSET_ONLY" if quality else "INDEPENDENT_LABELS_PENDING",
            "generalization_certified": False,
            "label_authenticity": "caller-supplied provenance; not independently authenticated by this tool",
            "pending_or_excluded_labels": pending, "cases": outcomes,
            "no_npu_used": True, "no_llm_judge": True}


def historical_explanation_cases(path, raw_source=None):
    """Import only hash-verified full text, never a 600-character excerpt.

    This adapter is for the existing offline case_explanation_audit contract.
    Original prompts and token pieces are absent there; report that limitation,
    group repeats by its original sample key and use lexical token comparison.
    Existing scorer classifications are regression expectations, not new labels.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = []
    for source in data["selected_case_audit"]:
        if source["case_id"] not in ("gsm8k-test-0001", "gsm8k-test-0002", "gsm8k-test-0004"):
            continue
        rows = {}
        for name in ("baseline", "fault"):
            recorded = source["responses"][name]
            value = recorded.get("full_text")
            if not isinstance(value, str) or len(value) != recorded.get("character_count"):
                raise ValueError("historical audit lacks complete full_text")
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            if digest != recorded.get("response_sha256"):
                raise ValueError("historical complete-text hash mismatch")
            rows[name] = {"response_text": value, "text_is_complete": True,
                          "text_sha256": digest, "source_ref": recorded["source_ref"],
                          "generation_length": recorded.get("generation_length"),
                          "finish_reason": recorded.get("finish_reason"),
                          "http_status": recorded.get("http_status"), "token_ids": None,
                          "token_pieces": None}
        sample = source["case_id"]
        cases.append({"case_id": source["cycle_id"] + ":" + sample,
                      "cycle_id": source["cycle_id"], "case_family": sample,
                      "original_case_key": sample, "source_kind": "recorded_model_output",
                      "protocol": "legacy_completion", "split": "regression",
                      "probe": {"probe_id": sample, "category": "historical_pathology_audit",
                                "prompt": "", "generation_config": {}, "expected": {},
                                "applicable_detectors": ["repetition", "garbage", "crash"]},
                      "baseline": rows["baseline"], "response": rows["fault"],
                      "labels": {"repetition": {"source": "historical_audit_expectation",
                                 "expected_trigger": sample in ("gsm8k-test-0001", "gsm8k-test-0004"),
                                 "evidence_note": "Existing documented loop/undelivered-budget cases; not an independent label set",
                                 "review_status": "pending_independent_review",
                                 "audit_source": str(path)}}})
    if not cases:
        raise ValueError("no historical complete cases found")
    if raw_source is not None:
        from .distortion.replay import RawSource, _legacy_probe
        original = RawSource(raw_source)
        try:
            for cycle in sorted({case["cycle_id"] for case in cases}):
                phase_rows = {phase: original.rows("cycles/" + cycle + "/" + phase + "/gsm8k_responses.jsonl")
                              for phase in ("baseline", "fault")}
                for case in (c for c in cases if c["cycle_id"] == cycle):
                    sample = case["original_case_key"]
                    for phase, key in (("baseline", "baseline"), ("fault", "response")):
                        source = phase_rows[phase][sample]
                        if source.get("response_text") != case[key]["response_text"]:
                            raise ValueError("raw ZIP / complete-text audit mismatch")
                        # Keep complete generated token pieces and source request;
                        # avoid serializing huge unrelated native hit histories.
                        for name in ("token_ids", "token_pieces", "request"):
                            case[key][name] = source.get(name)
                    case["probe"] = _legacy_probe(sample, "gsm8k", phase_rows["baseline"][sample])
                    case["labels"]["repetition"]["raw_source"] = str(raw_source)
        finally:
            original.close()
    return cases


def write_audit(cases_path, frozen_suite, output):
    """Write a new report only. JSON serialization rejects NaN/Infinity."""
    payload = json.loads(Path(cases_path).read_text(encoding="utf-8-sig"))
    report = audit_cases(payload["cases"], frozen_suite)
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report
