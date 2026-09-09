"""Independent, baseline-relative detectors. No LLM judge and no total score."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Protocol

from .contracts import DetectorResult, DistortionProbe


class DistortionDetector(Protocol):
    def evaluate(self, probe: DistortionProbe, baseline_response: Mapping[str, Any],
                 fault_response: Mapping[str, Any], metadata: Mapping[str, Any]) -> DetectorResult: ...


def text(row):
    value = row.get("response_text", row.get("text"))
    return value if isinstance(value, str) else None


def tokens(row):
    for key in ("token_ids", "token_pieces"):
        value = row.get(key)
        if isinstance(value, list):
            return value, key
    value = text(row)
    return (re.findall(r"\w+|[^\w\s]", value), "lexical_tokens") if value is not None else (None, None)


def size(row):
    value = row.get("generation_length")
    if isinstance(value, int) and value >= 0:
        return value
    value, _ = tokens(row)
    return len(value) if value is not None else None


def outcome(name, b, f, trigger, severity, reason, *, phenotype=None, baseline_failed=None, **evidence):
    return DetectorResult(name, trigger, b, f, severity, reason, evidence,
                          phenotype=phenotype if trigger else None, baseline_failed=baseline_failed)


def unknown(name, reason="required observation missing"):
    return outcome(name, None, None, None, "UNKNOWN", reason)


_NUM = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"


def numeric_answer(value):
    if value is None:
        return None
    try:
        result = Decimal(str(value).replace(",", "").strip())
        return str(result.normalize()) if result.is_finite() else None
    except InvalidOperation:
        return None


def task_fact(probe, row):
    # Recompute from text and frozen expected answer. Historical parser facts
    # remain annotations; they are never trusted when raw text is available.
    value = text(row)
    expected = probe.expected
    if value is None or "answer" not in expected:
        return None
    if expected.get("answer_type") == "number":
        answer = None
        for pattern in (rf"(?im)^\s*FINAL_ANSWER\s*=\s*({_NUM})",
                        rf"(?m)####\s*({_NUM})",
                        rf"(?i)\b(?:final answer|answer|result)\s*(?:is|:|=)\s*[$]?({_NUM})"):
            match = re.search(pattern, value)
            if match:
                answer = numeric_answer(match.group(1))
                break
        if answer is None and re.fullmatch(rf"\s*{_NUM}\s*[.!]?\s*", value):
            answer = numeric_answer(value.strip().rstrip(".!"))
        return {"answer": answer, "resolved": answer is not None,
                "correct": answer is not None and answer == numeric_answer(expected["answer"])}
    answer = value.strip().casefold().rstrip(".! ")
    # Finite approved aliases; do not interpret arbitrary free text with a judge.
    approved = [str(expected["answer"]), *expected.get("aliases", [])]
    correct = answer in [x.casefold().rstrip(".! ") for x in approved]
    return {"answer": answer or None, "resolved": bool(answer), "correct": correct}


class TaskAnswerDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        b, f = task_fact(probe, baseline_response), task_fact(probe, fault_response)
        if b is None or f is None:
            return unknown("task_answer")
        trigger = b["correct"] and not f["correct"]
        phenotype = "UNRESOLVED" if not f["resolved"] else "WRONG_ANSWER"
        return outcome("task_answer", b, f, trigger, "TASK_OR_STRUCTURE_FAILURE",
                       "baseline correct -> " + phenotype if trigger else "no new task failure",
                       phenotype=phenotype, baseline_failed=not b["correct"])


def _schema_errors(value, schema, path="$"):
    errors = []
    kinds = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    kind = schema.get("type")
    if kind in kinds and (not isinstance(value, kinds[kind]) or kind in ("integer", "number") and isinstance(value, bool)):
        return [f"TYPE:{path}:{kind}"]
    if isinstance(value, dict):
        errors += [f"MISSING_ITEM:{key}" for key in schema.get("required", []) if key not in value]
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            errors += [f"UNEXPECTED_ITEM:{key}" for key in value if key not in props]
        for key in value.keys() & props.keys():
            errors.extend(_schema_errors(value[key], props[key], path + "." + key))
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_schema_errors(item, schema.get("items", {}), f"{path}[{index}]"))
    return errors


def structure_fact(probe, row):
    value = text(row)
    expected = probe.expected
    if value is None or "structure" not in expected:
        return None
    kind, errors = expected["structure"], []
    if kind == "json":
        try:
            parsed = json.loads(value)
            errors = _schema_errors(parsed, expected["schema"])
        except (ValueError, TypeError):
            errors = ["JSON_PARSE_FAILURE"]
        return {"valid": not errors, "errors": errors}
    items = expected.get("items", [])
    lines = [line.strip() for line in value.strip().splitlines() if line.strip()]
    if kind == "number_sequence":
        present = re.findall(r"\d+", value)
        items = list(map(str, items))
    elif kind == "line_list":
        present = [re.sub(r"^(?:[-*]|\d+[.)])\s*", "", line) for line in lines]
    elif kind == "table":
        present = lines
    else:
        return None
    missing = list((Counter(items) - Counter(present)).elements())
    duplicates = [x for x, count in Counter(present).items() if count > max(1, items.count(x))]
    if missing:
        errors.append("MISSING_ITEM")
    if duplicates:
        errors.append("DUPLICATE_ITEM")
    if present != items:
        errors.append("ORDER_VIOLATION" if Counter(present) == Counter(items) else "FORMAT_FAILURE")
    if expected.get("one_per_line", True) and len(lines) != len(items):
        errors.append("NEWLINE_COLLAPSE")
    if kind == "number_sequence" and lines != items:
        errors.append("LINE_INTEGRITY")
    return {"valid": not errors, "errors": errors, "expected_item_count": len(items),
            "present_item_count": len(present), "missing": missing, "duplicate": duplicates}


class StructureIntegrityDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        b, f = structure_fact(probe, baseline_response), structure_fact(probe, fault_response)
        if b is None or f is None:
            return unknown("structure")
        trigger = b["valid"] and not f["valid"]
        return outcome("structure", b, f, trigger, "TASK_OR_STRUCTURE_FAILURE",
                       ", ".join(f["errors"]) or "structure valid", phenotype="STRUCTURE_FAILURE",
                       baseline_failed=not b["valid"])


def repetition_fact(row, thresholds):
    seq, basis = tokens(row)
    if seq is None:
        return None
    n = len(seq)
    grams = [tuple(seq[i:i+4]) for i in range(max(0, n-3))]
    rep4 = 1 - len(set(grams))/len(grams) if grams else 0.0
    run = longest = 0
    previous = object()
    for token in seq:
        run = run + 1 if token == previous else 1
        longest, previous = max(longest, run), token
    loop = False
    repeat = thresholds["periodic_min_repetitions"]
    for p in range(1, thresholds["periodic_max_period"]+1):
        if any(seq[i:i+p] * repeat == seq[i:i+p*repeat] for i in range(max(0, n-p*repeat+1))):
            loop = True
            break
    return {"seq_rep_4": rep4, "same_token_max_run": longest, "periodic_loop": loop,
            "dominant_token_ratio": max(Counter(seq).values())/n if n else 0,
            "token_count": n, "comparison_basis": basis}


class RepetitionDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        t = metadata["thresholds"]
        b, f = repetition_fact(baseline_response, t), repetition_fact(fault_response, t)
        if b is None or f is None or b["comparison_basis"] != f["comparison_basis"]:
            return unknown("repetition", "missing or incompatible token basis")
        trigger = f["token_count"] >= t["repetition_min_tokens"] and (
            (f["periodic_loop"] and not b["periodic_loop"])
            or (f["same_token_max_run"] >= t["repeated_run_min"] > b["same_token_max_run"])
            or (f["seq_rep_4"] >= t["seq_rep_4_min"] and f["seq_rep_4"]-b["seq_rep_4"] >= t["seq_rep_4_delta"]))
        return outcome("repetition", b, f, trigger, "DEGENERATION", "baseline-relative repeated 4-gram/run/periodic pattern",
                       phenotype="REPETITION")


def lexical_fact(row):
    seq, basis = tokens(row)
    if seq is None:
        return None
    seq = [x for x in seq if not isinstance(x, str) or x.strip()]
    n = len(seq)
    return {"distinct_1": len(set(seq))/n if n else 0,
            "distinct_2": len(set(zip(seq, seq[1:])))/(n-1) if n>1 else 0,
            "dominant_token_ratio": max(Counter(seq).values())/n if n else 0,
            "token_count": n, "comparison_basis": basis}


class LexicalCollapseDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        t = metadata["thresholds"]
        b, f = lexical_fact(baseline_response), lexical_fact(fault_response)
        if b is None or f is None or b["comparison_basis"] != f["comparison_basis"]:
            return unknown("lexical_collapse")
        trigger = min(b["token_count"], f["token_count"]) >= t["lexical_min_tokens"] and (
            f["distinct_1"] < b["distinct_1"] * t["distinct_ratio_max"] and
            f["distinct_2"] < b["distinct_2"] * t["distinct_ratio_max"] and
            f["dominant_token_ratio"] - b["dominant_token_ratio"] >= t["dominance_delta"])
        return outcome("lexical_collapse", b, f, trigger, "DEGENERATION", "Distinct-1/2 collapse and token concentration relative to baseline", phenotype="LEXICAL_COLLAPSE")


def punctuation_fact(row):
    value = text(row)
    if value is None:
        return None
    visible = [c for c in value if not c.isspace()]
    count = run = longest = 0
    for c in value:
        p = unicodedata.category(c).startswith("P")
        count += p
        run = run+1 if p else 0
        longest = max(longest, run)
    return {"ratio": count/len(visible) if visible else 0, "max_run": longest}


class PunctuationCollapseDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        t = metadata["thresholds"]
        b, f = punctuation_fact(baseline_response), punctuation_fact(fault_response)
        if b is None or f is None:
            return unknown("punctuation")
        trigger = ((f["ratio"] >= t["punctuation_rate_min"] and f["ratio"]-b["ratio"] >= t["punctuation_rate_delta"])
                   or (f["max_run"] >= t["punctuation_run_min"] and f["max_run"]-b["max_run"] >= t["punctuation_run_delta"]))
        return outcome("punctuation", b, f, trigger, "DEGENERATION", "punctuation ratio/run exceeds frozen baseline-relative bounds", phenotype="PUNCTUATION_SPAM")


class LengthCollapseDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        t = metadata["thresholds"]
        b, f = size(baseline_response), size(fault_response)
        if b is None or f is None or b == 0:
            return unknown("length")
        ratio = f / b
        maximum = probe.generation_config.get("max_tokens")
        reached = fault_response.get("finish_reason") == "length" or isinstance(maximum, int) and f >= maximum
        early = b >= t["early_stop_min_baseline_tokens"] and ratio <= t["early_stop_length_ratio_max"]
        runaway = f >= t["runaway_min_tokens"] and ratio >= t["runaway_length_ratio_min"] and reached
        return outcome("length", b, f, early or runaway, "DEGENERATION", "baseline-relative length change",
                       phenotype="EARLY_TERMINATION" if early else "RUNAWAY", ratio=ratio,
                       baseline_finish_reason=baseline_response.get("finish_reason"), finish_reason=fault_response.get("finish_reason"), max_tokens_reached=reached)


class GarbageHeuristicDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        def fact(row):
            value = text(row)
            if value is None:
                return None
            bad = [i for i, c in enumerate(value) if c == "\ufffd" or unicodedata.category(c) in ("Cc", "Cs") and c not in "\n\r\t"]
            return {"bad_character_count": len(bad), "bad_character_positions": bad[:32]}
        b, f = fact(baseline_response), fact(fault_response)
        if b is None or f is None:
            return unknown("garbage")
        trigger = f["bad_character_count"] >= metadata["thresholds"]["bad_character_min"] > b["bad_character_count"]
        return outcome("garbage", b, f, trigger, "GARBAGE_INCOHERENT", "replacement/control/surrogate characters; semantic incoherence is not inferred", phenotype="GARBAGE")


class CrashDetector:
    def evaluate(self, probe, baseline_response, fault_response, metadata):
        def fact(row):
            status = row.get("http_status")
            if row.get("worker_alive") is False or row.get("service_healthy") is False:
                return "SERVICE_CRASH"
            if isinstance(status, int) and not 200 <= status < 300 or row.get("error"):
                return "REQUEST_ERROR"
            return "HEALTHY" if isinstance(status, int) and 200 <= status < 300 else None
        b, f = fact(baseline_response), fact(fault_response)
        if b is None or f is None:
            return unknown("crash")
        trigger = b == "HEALTHY" and f != "HEALTHY"
        return outcome("crash", b, f, trigger, "CRASH", "request/service health", phenotype=f, baseline_failed=b != "HEALTHY")


DETECTORS = {
    "task_answer": TaskAnswerDetector(), "structure": StructureIntegrityDetector(),
    "repetition": RepetitionDetector(), "lexical_collapse": LexicalCollapseDetector(),
    "punctuation": PunctuationCollapseDetector(), "length": LengthCollapseDetector(),
    "garbage": GarbageHeuristicDetector(), "crash": CrashDetector(),
}
