"""Paired evaluator: fixed probes, same-probe stability, no severity average."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path

from ...contracts.evaluation import EvaluationResult
from .contracts import DistortionProbe, FaultConfigurationRecord, V1_FROZEN_SHA256, V2_FROZEN_SHA256, content_hash
from .detectors import DETECTORS, text, tokens, task_fact, structure_fact, punctuation_fact
from .mechanism import cleanup_gate, validate_mechanism


VISIBLE = {"TASK_OR_STRUCTURE_FAILURE", "DEGENERATION", "GARBAGE_INCOHERENT"}
PRIORITY = ["CRASH", "GARBAGE_INCOHERENT", "DEGENERATION", "TASK_OR_STRUCTURE_FAILURE"]
SCORER_VERSION = 'DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY'


def task_outcome(probe, row):
    fact = task_fact(probe, row)
    structure = structure_fact(probe, row)
    if fact is not None:
        status = ('CORRECT' if fact['correct'] else 'WRONG_ANSWER' if fact['resolved'] else
                  'ANSWER_UNDELIVERED_WITHIN_BUDGET' if row.get('finish_reason') == 'length' else 'UNRESOLVED_ANSWER')
        return {'status': status, 'validator_kind': 'answer', **fact,
                'mathematical_error_localized': False}
    if structure is not None:
        return {'status': 'STRUCTURE_VALID' if structure['valid'] else 'STRUCTURE_INVALID',
                'validator_kind': 'structure', **structure}
    return {'status': 'NOT_EVALUATED', 'validator_kind': None}


def validator_restored(before, after):
    if before['validator_kind'] is None:
        return None
    keys = ('validator_kind', 'answer', 'resolved', 'correct', 'valid', 'errors')
    return all(before.get(k) == after.get(k) for k in keys)


def detector_routes(probe, baseline, response, thresholds):
    # Answer validation is category-specific. Generic output pathology is not.
    names = list(probe.applicable_detectors)
    lengths = [len(tokens(row)[0] or []) for row in (baseline, response)]
    added = []
    if max(lengths) >= thresholds['repetition_min_tokens']:
        added.append('repetition')
    if max(lengths) >= thresholds['lexical_min_tokens']:
        added.append('lexical_collapse')
    if max(lengths) >= thresholds['repetition_min_tokens'] or any(
            (punctuation_fact(row) or {}).get('max_run', 0) >= thresholds['punctuation_run_min'] for row in (baseline, response)):
        added.append('punctuation')
    return names + [d for d in added if d not in names], [d for d in added if d not in names]


def compare(probe, baseline, response, thresholds, calibration=None):
    names, added = detector_routes(probe, baseline, response, thresholds)
    results = [DETECTORS[name].evaluate(probe, baseline, response, {"thresholds": thresholds}) for name in names]
    if calibration is not None:
        results = [replace(r, triggered=None, severity='UNKNOWN', phenotype=None,
                           reason='pre-fault calibration: ' + calibration.get(r.detector, {}).get('status', 'MISSING'),
                           evidence={**r.evidence, 'uncalibrated_result': r.to_dict()})
                   if r.detector not in added and calibration.get(r.detector, {}).get('status') != 'ELIGIBLE' else r for r in results]
    triggered = [r for r in results if r.triggered is True]
    unknown = [r.detector for r in results if r.triggered is None]
    baseline_failures = [r.detector for r in results if r.baseline_failed is True]
    btext, ftext = text(baseline), text(response)
    divergence = btext != ftext if btext is not None and ftext is not None else None
    # Task completion failure is still measured, but is not by itself pathology.
    visible_triggers = [r for r in triggered if r.phenotype != 'UNRESOLVED']
    pathology = [r for r in triggered if r.detector not in ('task_answer',)]
    verdict = next((p for p in PRIORITY if any(r.severity == p for r in visible_triggers)), None)
    observation = "OBSERVED" if not unknown and not baseline_failures else "INCOMPLETE"
    if verdict is None and observation == "OBSERVED" and divergence is not None:
        verdict = "DIVERGED_VALID" if divergence else "HEALTHY"
    bt, bb = tokens(baseline)
    ft, fb = tokens(response)
    first = None
    comparable = bt is not None and ft is not None and bb == fb
    if comparable:
        first = next((i for i, pair in enumerate(zip(bt, ft)) if pair[0] != pair[1]), None)
        if first is None and len(bt) != len(ft):
            first = min(len(bt), len(ft))
    return {"verdict": verdict, "observation_status": observation,
            'scorer_version': SCORER_VERSION,
            'task_outcome': task_outcome(probe, response),
            'generation_termination': {'finish_reason': response.get('finish_reason'),
                'budget_exhausted': response.get('finish_reason') == 'length',
                'generated_tokens': response.get('generation_length'), 'max_tokens': probe.generation_config.get('max_tokens')},
            'output_pathology': {'confirmed': bool(pathology), 'families': sorted({r.phenotype for r in pathology if r.phenotype}),
                'semantic_incoherence': 'NOT_PROVEN', 'detectors_executed': names},
            'detector_routing': {'added_by_actual_output': added,
                'calibration_scope': 'added generic detectors use full same-cycle baseline-relative checks; old calibration NOT_APPLICABLE was category routing, not proof of absence'},
            'display_status': ('OBSERVED_OUTPUT_PATHOLOGY' if pathology else 'TASK_ANSWER_MISMATCH' if any(r.phenotype == 'WRONG_ANSWER' for r in triggered)
                               else 'NO_COVERED_PATHOLOGY_DETECTED' if not unknown else 'INCOMPLETE_OBSERVATION'),
            "semantic_status": 'VALIDATOR_SCOPED' if probe.expected else 'SEMANTIC_REVIEW_REQUIRED',
            "families": sorted({r.phenotype for r in visible_triggers if r.phenotype}),
            "detectors": [r.to_dict() for r in results], "unknown_detectors": unknown,
            "baseline_failed_detectors": baseline_failures, "response_diverged": divergence,
            "first_divergence": first, "comparison_basis": bb if comparable else None,
            "token_ids": None if bb != "token_ids" or fb != "token_ids" else {"baseline": bt, "response": ft}}


def pairing(probe, rows, official, protocol=None):
    requests = [row.get("request") for row in rows]
    if any(not isinstance(r, dict) for r in requests):
        return {"status": "UNKNOWN", "reason": "raw request missing"}
    if any(r != requests[0] for r in requests[1:]):
        return {"status": "FAIL", "reason": "baseline/fault/recovery request mismatch"}
    if any(row.get("sample_id", row.get("probe_id")) != probe.probe_id for row in rows):
        return {"status": "FAIL", "reason": "sample identity mismatch"}
    if official:
        req = requests[0]
        content_matches = req.get('prompt') == probe.prompt or req.get('messages') == [{'role': 'user', 'content': probe.prompt}]
        if not content_matches or any(req.get(k) != v for k, v in probe.generation_config.items()):
            return {"status": "FAIL", "reason": "request differs from frozen suite"}
        if protocol and (req.get('messages') != [{'role': 'user', 'content': probe.prompt}]
                         or any(req.get(k) != protocol[k] for k in ('chat_template_kwargs', 'logprobs', 'top_logprobs'))):
            return {'status': 'FAIL', 'reason': 'request differs from frozen chat protocol'}
    return {"status": "PASS", "request_hash": content_hash(requests[0])}


def excerpt(row):
    value = text(row)
    return {"text": value[:600] if value is not None else None,
            'full_text': value, 'character_count': len(value) if value is not None else None,
            'generation_length': row.get('generation_length'), 'max_tokens': row.get('request', {}).get('max_tokens'),
            "truncated_excerpt": value is not None and len(value) > 600,
            "http_status": row.get("http_status"), "finish_reason": row.get("finish_reason"),
            "source_ref": row.get("source_ref"), "response_sha256": row.get("response_sha256"),
            "format_exact": row.get("format_exact"), "newline_anomaly": row.get("newline_anomaly"),
            "continuation_after_answer": row.get("continuation_after_answer")}


def evaluate_cycles(suite, config: FaultConfigurationRecord, cycles, *, evaluation_id="distortion-v1", analysis=None):
    scorer_files = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('evaluator.py', 'detectors.py', 'contracts.py', 'mechanism.py')}
    official = suite.get("version") in ("DISTORTION_PROBE_SUITE_V1", "NORMAL_DIAGNOSTIC_SUITE_V2")
    hash_valid = suite.get("hash") == content_hash(suite)
    probes = [DistortionProbe.from_dict(p) for p in suite["probes"]]
    ids = [p.probe_id for p in probes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("probe identifiers must be non-empty and unique")
    cycle_ids = [c["cycle_id"] for c in cycles]
    if len(set(cycle_ids)) != len(cycle_ids):
        raise ValueError("duplicate cycle id is not independent repeatability")
    results, gates = [], []
    for cycle in cycles:
        pairs = cycle.get("pairs", {})
        complete = set(pairs) == set(ids)
        clean = cleanup_gate(cycle.get("after_remove"), cycle.get("service_healthy"), config.world_size)
        before = cleanup_gate(cycle.get("before_baseline"), cycle.get("service_healthy_before"), config.world_size)
        matrix, cycle_gates = [], []
        for probe in probes:
            pair = pairs.get(probe.probe_id, {})
            b, f, r = (pair.get(phase, {}) for phase in ("baseline", "fault", "recovery"))
            if not all((b, f, r)):
                complete = False
            gate = validate_mechanism(f, config, cycle.get("service_healthy"))
            gate["probe_id"] = probe.probe_id
            gate["cycle_id"] = cycle["cycle_id"]
            cycle_gates.append(gate)
            pair_gate = pairing(probe, [b, f, r], official, suite.get('request_protocol'))
            calibration = cycle.get('calibration', {}).get('matrix', {}).get(probe.probe_id)
            effect = compare(probe, b, f, suite["thresholds"], calibration)
            recovery = compare(probe, b, r, suite["thresholds"], calibration)
            baseline_task = task_outcome(probe, b)
            task_recovered = validator_restored(baseline_task, recovery['task_outcome'])
            eligible = gate["status"] == "PASS" and pair_gate["status"] == "PASS" and before["status"] == "PASS" and hash_valid
            visible = eligible and effect["verdict"] in VISIBLE
            equivalent = text(b) is not None and text(b) == text(r)
            healthy_recovery = (recovery["verdict"] in ("HEALTHY", "DIVERGED_VALID")
                                and not recovery["unknown_detectors"] and not recovery["baseline_failed_detectors"])
            restored = (clean["status"] == "PASS" and r.get("http_status") == 200
                        and (equivalent or healthy_recovery)) if visible else None
            matrix.append({"probe_id": probe.probe_id, "category": probe.category, "cycle_id": cycle["cycle_id"],
                           "baseline": excerpt(b), "fault": excerpt(f), "recovery": excerpt(r),
                           "effect": effect, "recovery_effect": recovery, "recovery_restored": restored,
                           "recovery_exact": equivalent, "pairing": pair_gate, "eligible": eligible,
                           "calibration": calibration,
                           'task_outcomes': {'baseline': baseline_task, 'fault': effect['task_outcome'], 'recovery': recovery['task_outcome']},
                           'task_validator_recovered': task_recovered,
                           "visible_distortion": visible, "mechanism": gate,
                           "attribution": "VALID_FAULT_RUN" if eligible else "INVALID_FAULT_RUN"})
        # Probability requests are not visible probes, but their mechanism must
        # also be checked when replaying a full historical cycle.
        for auxiliary in cycle.get("auxiliary_fault_requests", []):
            gate = validate_mechanism(auxiliary, config, cycle.get("service_healthy"))
            gate.update(probe_id=auxiliary.get("sample_id"), cycle_id=cycle["cycle_id"], auxiliary=True)
            cycle_gates.append(gate)
        gates.extend(cycle_gates)
        same_process = before.get('pids') == clean.get('pids') and before['status'] == clean['status'] == 'PASS'
        live_recovery = (same_process and all(x['recovery'].get('http_status') == 200 and
                         (x['recovery_exact'] or x['recovery_effect']['verdict'] in ('HEALTHY', 'DIVERGED_VALID')) for x in matrix))
        results.append({"cycle_id": cycle["cycle_id"], "complete": complete,
                        "same_process_recovery": same_process, "all_case_recovery": live_recovery,
                        "clean_before": before, "clean_after": clean, "matrix": matrix,
                        "visible_count": sum(x["visible_distortion"] for x in matrix),
                        "crash_count": sum(x["effect"]["verdict"] == "CRASH" for x in matrix),
                        "invalid_count": sum(not x["eligible"] for x in matrix),
                        "recovery_gate": "PASS" if clean["status"] == "PASS" and all(
                            x["recovery_restored"] is True for x in matrix if x["visible_distortion"])
                        else "FAIL"})
        scoped = {'fault_induced_phenotype': {'recovered': sum(x['recovery_restored'] is True for x in matrix),
                    'denominator': sum(x['visible_distortion'] for x in matrix)},
                  'all_text_exact': {'matched': sum(x['recovery_exact'] for x in matrix), 'denominator': len(probes)},
                  'all_task_validator': {'matched': sum(x['task_validator_recovered'] is True for x in matrix),
                    'denominator': sum(x['task_validator_recovered'] is not None for x in matrix),
                    'mismatch_ids': [x['probe_id'] for x in matrix if x['task_validator_recovered'] is False],
                    'not_applicable_ids': [x['probe_id'] for x in matrix if x['task_validator_recovered'] is None]},
                  'runtime_clean_health': clean, 'same_process': same_process}
        results[-1]['recovery_scopes'] = scoped
        results[-1]['full_recovery_gate'] = 'PASS' if (complete and clean['status'] == 'PASS' and same_process
             and not scoped['all_task_validator']['mismatch_ids'] and all(x['recovery'].get('http_status') == 200 for x in matrix)
             and results[-1]['recovery_gate'] == 'PASS') else 'FAIL'
    n = len(results)
    stable, family_probes, repeatability = [], {}, {}
    enough = n >= suite["robust_gate"]["min_complete_cycles"] and all(c["complete"] for c in results)
    for pid in ids:
        rows = [next(x for x in c["matrix"] if x["probe_id"] == pid) for c in results]
        count = sum(x["visible_distortion"] for x in rows)
        repeatability[pid] = {"distorted_cycles": count, "total_cycles": n, "stable": bool(enough and count == n)}
        if enough and count == n:
            stable.append(pid)
            shared = set.intersection(*(set(x["effect"]["families"]) for x in rows))
            for family in sorted(shared):
                family_probes.setdefault(family, []).append(pid)
    mechanism_pass = bool(gates) and all(g["status"] == "PASS" for g in gates)
    crashed = sum(c["crash_count"] for c in results)
    recovered = sum(x["recovery_restored"] is True for c in results for x in c["matrix"])
    affected = sum(c["visible_count"] for c in results)
    recovery_pass = bool(results) and all(c['full_recovery_gate'] == 'PASS' for c in results)
    rule = suite["robust_gate"]
    conditions = {
        "mechanism": mechanism_pass,
        "frozen_16_probe_suite": official and hash_valid and suite.get("hash") in (V1_FROZEN_SHA256, V2_FROZEN_SHA256) and len(probes) == 16,
        "complete_cycles": enough,
        "pairing_and_clean_baseline": bool(results) and all(c["invalid_count"] == 0 for c in results),
        "per_cycle_breadth": bool(results) and all(c["visible_count"] >= rule["min_distorted_per_cycle"] for c in results),
        "same_probe_stability": len(stable) >= rule["min_stable_probes"],
        "phenotype_breadth": len(family_probes) >= rule["min_stable_families"],
        "no_crash": crashed <= rule["max_crashes"], "recovery": recovery_pass,
    }
    robust = all(conditions.values())
    distortion = {"schema_version": "DISTORTION_EVALUATION_V1", "probe_suite": suite["version"],
                  'scorer_version': SCORER_VERSION,
                  "suite_hash": suite.get("hash"), "suite_hash_valid": hash_valid,
                  "probe_count": len(probes), "complete_cycles": sum(c["complete"] for c in results),
                  "total_cycles": n, "mechanism_gate": "PASS" if mechanism_pass else "INVALID",
                  "mechanism_requests": len(gates), "mechanism_valid_requests": sum(g["status"] == "PASS" for g in gates),
                  "measurement_hits": sum(g["measurement_hit_count"] for g in gates if g["status"] == "PASS"),
                  "changed_elements": sum(g["changed_elements"] or 0 for g in gates if g["status"] == "PASS"),
                  "mechanism_evidence": gates, "stable_distorted_probes": stable,
                  "stable_phenotype_families": family_probes, "same_probe_repeatability": repeatability,
                  "crash_count": crashed, "crash_classification": "TOO_SEVERE_UNSUITABLE_PROFILE" if crashed else None,
                  "recovery": {"gate": "PASS" if recovery_pass else "FAIL", "restored_pairs": recovered,
                               "fault_induced_pairs": affected, "definition": "gate includes ALL task/validator states + runtime health; phenotype fraction remains explicitly scoped",
                               'phenotype_recovery_gate': 'PASS' if all(c['recovery_gate'] == 'PASS' for c in results) and results else 'FAIL',
                               'per_cycle_scopes': {c['cycle_id']: c['recovery_scopes'] for c in results},
                               "stable_probes_restored": [pid for pid in stable if all(next(x for x in c["matrix"] if x["probe_id"] == pid)["recovery_restored"] is True for c in results)]},
                  "robust_gate": "ROBUST_VISIBLE_DISTORTION_PASS" if robust else "ROBUST_VISIBLE_DISTORTION_FAIL",
                  "robust_conditions": conditions, "cycles": results,
                  "interpretation": "Finite engineering smoke suite, no significance claim; DIVERGED_VALID means no configured detector failure, not semantic correctness."}
    # The supplemental task has THREE independent outcomes. Keep the earlier
    # strict breadth gate unchanged; do not force two pathologies to pass live.
    stable_recovered = distortion['recovery']['stable_probes_restored']
    delivery = {
        'tool_live_validation': 'PASS' if enough and mechanism_pass and all(c['all_case_recovery'] for c in results) else 'PARTIAL',
        'frozen_r1_visible_effect': 'OBSERVED' if affected else 'NOT_OBSERVED',
        'reproducible_normal_cases': [pid for pid in stable_recovered if any(pid in pids for pids in family_probes.values())],
        'case_reproduction': 'PASS' if any(pid in stable_recovered for pids in family_probes.values() for pid in pids) else 'NOT_PROVEN',
        'does_not_require_two_pathology_families': True,
    } if official else {'status': 'NOT_APPLICABLE_LEGACY_REPLAY'}
    return EvaluationResult.from_dict({
        "evaluation_id": evaluation_id, "capability_id": config.capability or "UNKNOWN", "candidate_id": None,
        "baseline": None, "fault": None, "recovery": None,
        "mechanism": {"resolved": mechanism_pass, "hit_count": distortion["measurement_hits"] if mechanism_pass else None,
                      "changed_elements": distortion["changed_elements"] if mechanism_pass else None},
        "model_effect": {"crash": bool(crashed)}, "reliability": {"injection_hit_rate": sum(g["status"] == "PASS" for g in gates)/len(gates) if gates else None},
        "quality_performance": {}, "status": "PASS" if robust else "FAIL",
        "reason": distortion["robust_gate"], "evidence_refs": [],
        "fault_configuration": config.to_dict(), "distortion": distortion,
        'scorer': {'version': SCORER_VERSION, 'sha256': content_hash(scorer_files), 'source_files': scorer_files},
        "normal_diagnostic_delivery": delivery,
        "analysis_details": analysis or {"status": "NOT_REQUESTED"},
    })
