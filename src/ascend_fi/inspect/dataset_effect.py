"""Dataset-scoped delivery over the existing EvaluationResult and V2 scorer.

No detector thresholds or historical 16-probe/3-cycle acceptance are changed.
This adapter explicitly does not claim that an eight-sample observation is that
robustness experiment. It verifies raw request provenance before invoking it.
"""
import hashlib

from ascend_fi.contracts import EvaluationResult
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.evaluator import evaluate_cycles
from .dataset import SCOPE


def evaluate_dataset(suite, config, cycles, *, evaluation_id):
    if suite.get('version') != SCOPE or suite.get('hash') != content_hash(suite):
        raise ValueError('FROZEN_DATASET_SUITE_REQUIRED')
    if len(cycles) != 1:
        raise ValueError('THIS_DELIVERY_IS_ONE_CYCLE_NOT_REPEATABILITY')
    if set(cycles[0]['pairs']) != {p['probe_id'] for p in suite['probes']}:
        raise ValueError('INCOMPLETE_TARGETED_SUBSET')
    for p in suite['probes']:
        sample = p['dataset_sample']
        if (p['prompt'] != sample['prompt'] or hashlib.sha256(p['prompt'].encode()).hexdigest() != sample['prompt_sha256']
                or sample['source_sha256'] != suite['dataset']['sha256']
                or sample['source_revision'] != suite['dataset']['revision']):
            raise ValueError('SOURCE_PROMPT_MISMATCH')
        for row in cycles[0]['pairs'][p['probe_id']].values():
            req = row.get('request', {})
            if (req.get('messages') != [{'role': 'user', 'content': p['prompt']}]
                    or any(req.get(k) != v for k, v in p['generation_config'].items())
                    or any(req.get(k) != suite['request_protocol'][k] for k in ('chat_template_kwargs', 'logprobs', 'top_logprobs'))):
                raise ValueError('FROZEN_REQUEST_MISMATCH')
    result = evaluate_cycles(suite, config, cycles, evaluation_id=evaluation_id)
    data = result.to_dict()
    d = data['distortion']
    # Keep the old conditions for provenance, but they are NOT this experiment's gate.
    d['non_applicable_legacy_robust_conditions'] = d['robust_conditions']
    d['robust_conditions'] = {}
    d['robust_gate'] = 'NOT_APPLICABLE'
    budgets = []
    for p in suite['probes']:
        f = cycles[0]['pairs'][p['probe_id']]['fault']
        workers = f.get('fault_transaction', {}).get('measurement_query', {}).get('results', [])
        for w in workers:
            ranks = {m.get('tp_rank') for m in w.get('matched_modules', [])}
            history = w.get('tensor_effect_history', [])
            summary = w.get('effect_summary', {})
            full_aggregate = summary.get('schema') == 'BOUNDED_EFFECT_SUMMARY_V1'
            population = list(summary.get('per_module', {}).values()) if full_aggregate else history
            cap = w.get('armed_config', {}).get('max_hits')
            budgets.append({'probe_id': p['probe_id'], 'pid': w.get('pid'),
                'rank': next(iter(ranks)) if len(ranks) == 1 else None,
                'hit_count': w.get('hit_count'), 'max_hits': cap,
                'changed_elements': sum(h.get('changed_elements', 0) for h in population),
                'hit_layers': sorted({h['layer'] for h in population if h.get('layer') is not None}),
                'numeric_population': ('ALL_ACCEPTED_CALLS_AGGREGATE' if full_aggregate else 'RETAINED_RAW_EVENTS'),
                'budget_exhausted': w.get('hit_count') >= cap if cap is not None and w.get('hit_count') is not None else None,
                'reject_reasons': w.get('reject_reasons'),
                'no_claim_of_sustained_full_answer_injection': True})
    rows = d['cycles'][0]['matrix']
    for row in rows:
        gate = row['mechanism']
        row['mechanism_state'] = ('CONFIRMED' if gate['status'] == 'PASS' else
            'NOT_HIT' if gate.get('measurement_hit_count') == 0 else
            'NO_CHANGE' if gate.get('changed_elements') == 0 else 'UNKNOWN')
    integration = (d['mechanism_gate'] == 'PASS' and d['recovery']['gate'] == 'PASS'
                   and all(r['eligible'] and r['recovery_exact'] and all(r[phase]['http_status'] == 200
                       for phase in ('baseline', 'fault', 'recovery')) for r in rows))
    observed = [r['probe_id'] for r in rows if r['visible_distortion']]
    delivery = {'DATASET_BACKED_LOSSY_INTEGRATION': 'PASS' if integration else 'PARTIAL',
                'VISIBLE_DISTORTION_OBSERVATION': 'OBSERVED' if observed else 'NOT_OBSERVED_BY_COVERED_RULES',
                'dataset': suite['dataset'], 'question_ids': [p['dataset_sample']['question_id'] for p in suite['probes']],
                'samples': [p['dataset_sample'] for p in suite['probes']],
                'turn_index': 0, 'complete_cycles': 1, 'repeatability': 'NOT_PROVEN',
                'official_mt_bench_score': 'NOT_COMPUTED', 'semantic_status': 'SEMANTIC_REVIEW_REQUIRED',
                'visible_sample_ids': observed, 'budget_evidence': budgets,
                'exact_recovery': sum(r['recovery_exact'] for r in rows),
                'response_divergence': sum(r['effect']['response_diverged'] is True for r in rows),
                'legacy_16_probe_3_cycle_gate': 'NOT_APPLICABLE'}
    data.update(dataset_delivery=delivery,
                normal_diagnostic_delivery={'status': 'NOT_APPLICABLE_DATASET_TARGETED_SUBSET'},
                status='PASS' if integration else 'FAIL', reason='DATASET_BACKED_LOSSY_INTEGRATION:' + delivery['DATASET_BACKED_LOSSY_INTEGRATION'])
    from .single_effect import numerical_from_request
    data['numerical_effect'] = {'aggregation_scope': 'PER_REQUEST_ONLY; no repeated cumulative query summation',
        'per_probe': {p['probe_id']: numerical_from_request(cycles[0]['pairs'][p['probe_id']]['fault'],
                        list(range(config.world_size))) for p in suite['probes']},
        'detector_thresholds_changed': False}
    return EvaluationResult.from_dict(data)
