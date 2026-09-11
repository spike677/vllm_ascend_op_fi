"""Offline fixed-reference echo parsing; no model call or fault transaction.

Unlike the historical word-PPL projection, this keeps exact token/mask and
context identity.  It cannot certify a TargetPlan teacher-forced transaction.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math

from .distortion.contracts import content_hash
from ..contracts import EvaluationResult

IDENTITY_KEYS = ("model_identity", "tokenizer_identity", "template_identity",
                 "reference_sha256", "window_identity", "context_policy",
                 "logprob_semantics", "cache_policy")


def _exp(value):
    if value == math.inf:
        return None, "POSITIVE_INFINITY"
    if value == -math.inf:
        return 0.0, "ZERO_FROM_NEGATIVE_INFINITY"
    try:
        result = math.exp(value)
    except OverflowError:
        return None, "OVERFLOW"
    return (None, "UNDERFLOW") if result == 0 else (result, "FINITE")


def _reference_error(reference):
    if not isinstance(reference, dict) or any(not reference.get(k) for k in IDENTITY_KEYS):
        return "REFERENCE_IDENTITY_INCOMPLETE"
    ids, mask = reference.get("token_ids"), reference.get("score_mask")
    if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids):
        return "REFERENCE_TOKEN_IDS_INVALID"
    if not isinstance(mask, list) or len(mask) != len(ids) or any(type(v) is not bool for v in mask):
        return "SCORE_MASK_INVALID"
    if mask and mask[0]:
        return "FIRST_TOKEN_HAS_NO_PROVEN_CONTEXT"
    try:
        content_hash(reference)
    except (ValueError, TypeError):
        return "REFERENCE_IDENTITY_NOT_JSON_FINITE"
    return None


def parse_echo_reference(reference, record):
    """Parse one window, including prefix tokens and an explicit scoring mask.

    ``record`` holds original request/raw_response and the captured ``reference``
    identity. Only prompt-position token logprobs count, never generated ones or
    absent top-k entries. Returned integer token IDs are required to prove echo
    alignment; the old live.py projection alone is intentionally insufficient.
    """
    result = {"status": "UNSCORABLE", "reason": None, "request_id": record.get("request_id"),
              "reference_hash": None, "scored_token_count": None,
              "sum_nll": None, "sum_nll_status": "UNAVAILABLE", "mean_nll": None,
              "mean_nll_status": "UNAVAILABLE", "ppl": None, "ppl_status": "UNAVAILABLE",
              "missing_scored_positions": [], "execution_status": "LIVE_UNVERIFIED"}

    def reject(reason):
        result["reason"] = reason
        return result

    error = _reference_error(reference)
    if error:
        return reject(error)
    result["reference_hash"] = content_hash(reference)
    if record.get("reference") != reference:
        return reject("CAPTURED_REFERENCE_IDENTITY_MISMATCH")
    if not record.get("request_id"):
        return reject("REQUEST_ID_REQUIRED")
    ids, mask = reference["token_ids"], reference["score_mask"]
    count = sum(mask)
    result["scored_token_count"] = count
    if not count:
        return reject("NO_SCORED_TOKENS")
    request = record.get("request")
    if (not isinstance(request, dict) or request.get("echo") is not True or request.get("prompt") != ids
            or any(type(i) is not int for i in request.get("prompt", []))):
        return reject("NOT_FIXED_TOKEN_ECHO_REQUEST")
    if record.get("http_status") != 200 or record.get("error"):
        return reject("REQUEST_FAILED")
    raw = record.get("raw_response")
    if not isinstance(raw, dict) or not isinstance(raw.get("usage"), dict):
        return reject("RAW_ECHO_RESPONSE_REQUIRED")
    if raw["usage"].get("prompt_tokens") != len(ids):
        return reject("PROMPT_TOKEN_COUNT_MISMATCH")
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return reject("ONE_ECHO_CHOICE_REQUIRED")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("logprobs"), dict):
        return reject("ECHO_LOGPROBS_REQUIRED")
    logprobs = choice["logprobs"]
    returned_ids = choice.get("token_ids", logprobs.get("token_ids"))
    values = logprobs.get("token_logprobs")
    if (not isinstance(returned_ids, list) or len(returned_ids) < len(ids)
            or any(type(i) is not int for i in returned_ids[:len(ids)]) or returned_ids[:len(ids)] != ids):
        return reject("RETURNED_REFERENCE_TOKEN_IDS_UNPROVEN")
    if not isinstance(values, list) or len(values) < len(ids):
        return reject("REFERENCE_LOGPROBS_MISSING_OR_MISALIGNED")
    losses, infinite = [], False
    for index, enabled in enumerate(mask):
        if not enabled:
            continue
        value = values[index]
        if value is None:
            result["missing_scored_positions"].append(index)
            continue
        if type(value) not in (float, int):
            return reject("INVALID_REFERENCE_LOGPROB_TYPE")
        try:
            value = float(value)
        except OverflowError:
            return reject("REFERENCE_LOGPROB_OUT_OF_RANGE")
        if math.isnan(value) or value > 0:
            return reject("INVALID_REFERENCE_LOGPROB_VALUE")
        if value == -math.inf:
            infinite = True
        else:
            losses.append(-value)
    if result["missing_scored_positions"]:
        return reject("MISSING_SCORED_REFERENCE_LOGPROB")
    if infinite:
        result.update(status="COMPLETE", sum_nll_status="POSITIVE_INFINITY",
                      mean_nll_status="POSITIVE_INFINITY", ppl_status="POSITIVE_INFINITY")
        return result
    try:
        total = math.fsum(losses)
        total_status = "FINITE"
    except OverflowError:
        total, total_status = None, "OVERFLOW"
    # Preserve a representable mean even if the accumulated sum overflows.
    mean = math.fsum(v / count for v in losses)
    ppl, ppl_status = _exp(mean)
    result.update(status="COMPLETE", sum_nll=total, sum_nll_status=total_status,
                  mean_nll=mean, mean_nll_status="FINITE", ppl=ppl, ppl_status=ppl_status)
    return result


def compare_echo_reference(reference, baseline, fault):
    """Compare exact same prefix/targets/mask using mean-NLL difference in log space."""
    b = parse_echo_reference(reference, baseline)
    f = parse_echo_reference(reference, fault)
    result = {"baseline": b, "fault": f, "status": "UNSCORABLE",
              "mean_nll_delta": None, "mean_nll_delta_status": "UNAVAILABLE",
              "ppl_ratio": None, "ppl_ratio_status": "UNAVAILABLE",
              "execution_status": "IMPLEMENTED_OFFLINE_LIVE_UNVERIFIED",
              "probability_transaction_gate": "NOT_IMPLEMENTED",
              "model_effect_attribution": "UNPROVEN_REQUIRES_THIS_ECHO_REQUEST_TARGET_PLAN_AND_HITS",
              "reference_hash": b["reference_hash"], "scored_token_count": b["scored_token_count"],
              "units": "natural-log NLL per reference token; token PPL, not word PPL",
              "top1_flip_rate": None, "top1_status": "ARGMAX_EVIDENCE_NOT_IMPLEMENTED"}
    if b["status"] != "COMPLETE" or f["status"] != "COMPLETE":
        return result
    if baseline.get("request") != fault.get("request"):
        result["reason"] = "BASELINE_FAULT_REQUEST_CONFIGURATION_MISMATCH"
        return result
    if baseline["request_id"] == fault["request_id"]:
        result["reason"] = "BASELINE_FAULT_REQUEST_ID_REUSED"
        return result
    # A shared response cache must never erase the fault phase.
    if baseline.get("response_cache_key") and baseline.get("response_cache_key") == fault.get("response_cache_key"):
        result["reason"] = "BASELINE_FAULT_RESPONSE_CACHE_KEY_REUSED"
        return result
    bn = math.inf if b["mean_nll_status"] == "POSITIVE_INFINITY" else b["mean_nll"]
    fn = math.inf if f["mean_nll_status"] == "POSITIVE_INFINITY" else f["mean_nll"]
    result["status"] = "COMPLETE_OFFLINE_COMPARISON"
    if bn == fn == math.inf:
        result.update(mean_nll_delta_status="UNDEFINED_INFINITY_MINUS_INFINITY",
                      ppl_ratio_status="UNDEFINED_INFINITY_OVER_INFINITY")
        return result
    delta = fn - bn
    result["mean_nll_delta_status"] = ("FINITE" if math.isfinite(delta) else
                                       "POSITIVE_INFINITY" if delta > 0 else "NEGATIVE_INFINITY")
    result["mean_nll_delta"] = delta if math.isfinite(delta) else None
    result["ppl_ratio"], result["ppl_ratio_status"] = _exp(delta)
    return result


def attach_reference_probability(evaluation: EvaluationResult, reference, baseline, fault):
    """Add read-only analysis without promoting formal quality metrics or PASS.

    Offline fixtures and old echo observations must not certify newly selected
    fault targets. Keep existing quality_performance/status unchanged.
    """
    extension = compare_echo_reference(reference, baseline, fault)
    if "reference_probability_offline" in evaluation.extra:
        raise ValueError("reference probability extension already exists; do not overwrite evidence")
    return replace(evaluation, extra={**evaluation.extra, "reference_probability_offline": copy.deepcopy(extension)})
