"""One baseline/fault pair in EvaluationResult; recovery remains unmeasured."""
from ascend_fi.contracts import EvaluationResult
from ascend_fi.evaluation.distortion.contracts import content_hash, DistortionProbe
from ascend_fi.evaluation.distortion.evaluator import compare, pairing, task_outcome
from ascend_fi.evaluation.distortion.mechanism import rpc_workers, validate_mechanism
from ascend_fi.evaluation.numerical_effect import sampled_numerical_effect


def numerical_from_request(fault, expected_ranks):
    """One final snapshot per worker, per request; never add repeated queries."""
    workers = []
    for row in rpc_workers(fault.get('fault_transaction', {}).get('measurement_query')) or []:
        identity = row.get('input_contract', {}).get('worker_identity', {})
        ranks = {m.get('tp_rank') for m in row.get('matched_modules', [])}
        rank = identity.get('global_rank')
        if rank is None and len(ranks) == 1:
            rank = next(iter(ranks))  # Legacy pure-TP evidence, source explicitly marked.
        workers.append({'rank': rank, 'effect_summary': row.get('effect_summary')})
    return sampled_numerical_effect(workers, expected_ranks=expected_ranks,
                                    request_generation=fault.get('request_generation'))


def evaluate_single(suite, config, plan, run, *, probe_id):
    if suite.get('hash') != content_hash(suite):
        raise ValueError('FROZEN_SUITE_HASH_MISMATCH')
    probe = DistortionProbe.from_dict(next(p for p in suite['probes'] if p['probe_id'] == probe_id))
    b, f = run['pair'].get('baseline') or {}, run['pair'].get('fault') or {}
    paired = pairing(probe, [b, f], True, suite.get('request_protocol'))
    mechanism = validate_mechanism(f, config, f.get('service_healthy'))
    effect = compare(probe, b, f, suite['thresholds'])
    eligible = paired['status'] == mechanism['status'] == 'PASS' and run['window']['mechanism'] == 'PASS'
    return EvaluationResult.from_dict({
        'evaluation_id': 'single-' + plan['plan_sha256'][:16] + '-' + probe_id,
        'capability_id': 'fake_requant', 'candidate_id': plan['plan_sha256'],
        'baseline': b or None, 'fault': f or None, 'recovery': None,
        'mechanism': {'hit_count': mechanism.get('measurement_hit_count'),
                      'changed_elements': mechanism.get('changed_elements'), 'validation': mechanism},
        'model_effect': {'response_equal': None if effect['response_diverged'] is None else not effect['response_diverged'],
                         'first_divergent_token_piece': effect['first_divergence'],
                         'comparison_basis': effect['comparison_basis'], 'attribution_eligible': eligible},
        'reliability': {'complete_cycles': 0, 'repeatability': 'NOT_MEASURED', 'recovery_rate': None},
        'quality_performance': {}, 'fault_configuration': config.to_dict(),
        'status': 'UNKNOWN', 'reason': 'SINGLE_LOSSY_MEASURED_STOP_PENDING' if eligible else 'INVALID_OR_INCOMPLETE_FAULT_RUN',
        'single_request_effect': {'probe_id': probe_id, 'pairing': paired, 'effect': effect,
            'baseline_task': task_outcome(probe, b), 'eligible': eligible,
            'thresholds': suite['thresholds'], 'suite_hash': suite['hash'],
            'scorer_version': effect['scorer_version'], 'window': run['window'],
            'raw': {'baseline': b, 'fault': f, 'recovery': None},
            'confirmed_output_pattern': 'RULE_SCOPED_NOT_INDEPENDENT_SEMANTIC_CONFIRMATION'},
        'numerical_effect': numerical_from_request(f, [w['rank'] for w in plan['workers']]),
        'instance_lifecycle': {'policy': run['execution_policy'], 'cleanup': run['cleanup'],
            'stop_confirmed': False, 'stop_record': 'host writes final_instance_state.json separately',
            'hot_recovery': 'NOT_MEASURED', 'restart_text_recovery': 'NOT_MEASURED'},
        'evidence_refs': [r['source_ref'] for r in (b, f) if r.get('source_ref')]})
