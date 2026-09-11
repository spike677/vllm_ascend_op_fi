"""Explicit configuration exploration and evidence gates, separate from R1."""
import copy
import math
import re
from .dataset_effect import evaluate_dataset
from ..contracts import EvaluationResult

ROLE_GROUPS = [
    ['MLP_DOWN'], ['MLP_DOWN', 'MLP_GATE_UP_FUSED'],
    ['MLP_DOWN', 'MLP_GATE_UP_FUSED', 'ATTN_OUTPUT'],
    ['MLP_DOWN', 'MLP_GATE_UP_FUSED', 'ATTN_OUTPUT', 'ATTN_QKV_FUSED'],
]


def make_experiment(config_id, r1, module_rows, roles, multiplier, max_tokens, max_prompt_tokens, *, bounded=False):
    chosen = [m for m in module_rows if m['logical_role'] in roles]
    names = sorted(m['module_name'] for m in chosen)
    if len(names) != 64 * len(roles) or len(set(names)) != len(names):
        raise ValueError('LIVE_64_LAYER_MODULE_SET_REQUIRED')
    for role in roles:
        if {m['layer_index'] for m in chosen if m['logical_role'] == role} != set(range(64)):
            raise ValueError('INCOMPLETE_LIVE_ROLE_COVERAGE')
    if any(a != b and b.startswith(a + '.') for a in names for b in names):
        raise ValueError('PARENT_CHILD_DOUBLE_TARGET')
    arm = copy.deepcopy(r1)
    arm['module_regex'] = r'\A(?:' + '|'.join(re.escape(n) for n in names) + r')\Z'
    arm['scale_multiplier'] = multiplier
    arm['max_hits'] = 4096 if bounded else len(names) * (max_tokens + max_prompt_tokens + 16) * 2
    return {'config_id': config_id, 'arm': arm, 'selected_modules': names, 'module_families': roles,
        'window_policy': 'ARM_BOUNDED_R1_CONTROL' if bounded else 'COMPLETE_EXCLUSIVE_REQUEST_FINITE_CAP',
        'request_timeout_seconds': 3600,
        'cap_rationale': {'module_count': len(names), 'max_tokens': max_tokens,
                         'prompt_token_upper_bound': max_prompt_tokens, 'chunking_margin': 16, 'safety_factor': 2,
                         'not_a_scheduler_mapping': True, 'must_verify_no_cap_rejections': True}}


def window_gate(row, config):
    failures = []
    tx = row.get('fault_transaction', {})
    win = row.get('request_window_evidence', {})
    if win.get('single_request_completed') is not True:
        failures.append('request completion/isolation missing')
    rows = []
    for worker in tx.get('measurement_query', {}).get('results', []):
        ranks = {m.get('tp_rank') for m in worker.get('matched_modules', [])}
        if len(ranks) != 1:
            failures.append('rank unresolved'); continue
        rank = next(iter(ranks))
        hits, eligible = worker.get('hit_count', 0), worker.get('candidate_seen', 0)
        reject = worker.get('reject_reasons', {})
        modules = worker.get('per_module', {})
        if rank == 0:
            if (worker.get('enabled') is not True or hits <= 0 or hits != eligible
                    or hits >= config.hit_policy['max_hits'] or any(reject.values())):
                failures.append('target window incomplete')
            if set(modules) != set(config.actual_module_names) or any(
                    m.get('hit_count', 0) <= 0 or m.get('hit_count') != m.get('candidate_seen')
                    or m.get('hook_seen') != m.get('hit_count') for m in modules.values()):
                failures.append('per-module window incomplete')
        elif rank != 1 or hits != 0 or set(reject) != {'rank_filter'} or reject.get('rank_filter') != worker.get('hook_seen'):
            failures.append('rank policy mismatch')
        rows.append({'rank': rank, 'hit': hits, 'eligible': eligible, 'hook_seen': worker.get('hook_seen'),
                     'reject': reject, 'target_call_coverage': hits / eligible if eligible else None,
                     'per_module': modules, 'per_rank': worker.get('per_rank'),
                     'phase': 'UNKNOWN', 'generation_token_coverage': 'UNKNOWN'})
    if {r['rank'] for r in rows} != {0, 1} or len(rows) != 2:
        failures.append('missing worker')
    return {'status': 'PASS' if not failures else 'WINDOW_INCOMPLETE', 'failures': failures, 'ranks': rows,
            'basis': 'private generation key + exclusive lease + single engine completion + all target counters',
            'request_generation_is_isolation': False}


def dose_summary(row):
    groups = {}
    for w in row.get('fault_transaction', {}).get('measurement_query', {}).get('results', []):
        for m in w.get('effect_summary', {}).get('per_module', {}).values():
            key = m['module'].split('.')[-1]
            g = groups.setdefault(key, {'calls': 0, 'changed_calls': 0, 'elements': 0, 'changed_elements': 0,
                 'clip_count': 0, 'sampled_calls': 0, 'sampled_elements': 0, 'sampled_nonzero_elements': 0,
                 'sampled_new_zero_elements': 0, 'sampled_input_sq_sum': 0., 'sampled_error_sq_sum': 0.,
                 'fault_scale_min': None, 'fault_scale_max': None, 'scale_multiplier': m['scale_multiplier']})
            for source, dest in [('hit_count','calls'), ('total_elements','elements'), ('changed_calls','changed_calls'),
                    ('changed_elements','changed_elements'), ('clip_count','clip_count')]:
                g[dest] += m[source]
            for k in ('sampled_calls','sampled_elements','sampled_nonzero_elements','sampled_new_zero_elements',
                      'sampled_input_sq_sum','sampled_error_sq_sum'):
                g[k] += m[k]
            if m['fault_scale_min'] is not None:
                g['fault_scale_min'] = min(g['fault_scale_min'] if g['fault_scale_min'] is not None else math.inf,m['fault_scale_min'])
                g['fault_scale_max'] = max(g['fault_scale_max'] if g['fault_scale_max'] is not None else -math.inf,m['fault_scale_max'])
    for g in groups.values():
        g['clip_rate'] = g['clip_count']/g['elements'] if g['elements'] else None
        g['new_zero_rate_sampled'] = g['sampled_new_zero_elements']/g['sampled_nonzero_elements'] if g['sampled_nonzero_elements'] else None
        g['relative_l2_sampled'] = (g['sampled_error_sq_sum']/g['sampled_input_sq_sum'])**.5 if g['sampled_input_sq_sum'] else None
        g['base_scale_min_derived'] = g['fault_scale_min']/g['scale_multiplier'] if g['fault_scale_min'] is not None else None
        g['base_scale_max_derived'] = g['fault_scale_max']/g['scale_multiplier'] if g['fault_scale_max'] is not None else None
    return {'groups': groups, 'denominator': 'cumulative activation-element events, not unique weights',
            'dose_sample_rule': 'first and every 64th accepted call per module; FP32 sums',
            'effective_code_values': 'NOT_SAMPLED', 'scale_source': 'original tensor_absmax; base derived from actual fault_scale / multiplier'}


def contextual_findings(matrix):
    """Deterministic context annotation; leave original scoring/thresholds intact."""
    result = []
    for row in matrix:
        for d in row['effect']['detectors']:
            if not d.get('triggered') or d.get('baseline_failed'):
                continue
            confirmed = d['detector'] in ('repetition', 'lexical_collapse', 'punctuation', 'garbage')
            context = 'RULE_SUPPORTED_PATTERN_NOT_SEMANTIC_JUDGEMENT'
            value = row['fault']['full_text'] or ''
            span = None
            if d['detector'] == 'punctuation':
                span = d['fault_value'].get('longest_run_character_span')
                if span:
                    a,b = span
                    line = value[value.rfind('\n',0,a)+1:value.find('\n',b) if value.find('\n',b)>=0 else len(value)]
                    if re.fullmatch(r'\s*\|[-:| \t]+\|\s*',line) or re.fullmatch(r'\s*[-*_]{3,}\s*',line):
                        confirmed, context = False, 'MARKDOWN_SEPARATOR_NOT_CONFIRMED_SPAM'
            if d['detector'] == 'length':
                confirmed, context = False, 'LENGTH_CHANGE_ONLY_NOT_PATHOLOGY'
            result.append({'probe_id': row['probe_id'], 'detector': d['detector'], 'raw_rule_triggered': True,
                           'confirmed_pattern': confirmed, 'context': context, 'character_span': span,
                           'phenotype': d.get('phenotype')})
    return result


def evaluate_exploration(suite, config, cycle, *, evaluation_id):
    result = evaluate_dataset(suite, config, [cycle], evaluation_id=evaluation_id)
    data = result.to_dict()
    data['dataset_delivery']['budget_evidence'] = []  # Replace legacy per-event projection with explicit new evidence below.
    windows = {pid: window_gate(pair['fault'], config) for pid,pair in cycle['pairs'].items()}
    doses = {pid: dose_summary(pair['fault']) for pid,pair in cycle['pairs'].items()}
    matrix = data['distortion']['cycles'][0]['matrix']
    findings = contextual_findings(matrix)
    exact_pieces = all(p['baseline'].get('token_pieces') is not None and
                       p['baseline']['token_pieces'] == p['recovery'].get('token_pieces') for p in cycle['pairs'].values())
    gate = (data['status'] == 'PASS' and exact_pieces and all(w['status'] == 'PASS' for w in windows.values()))
    data.update(exploration={'config_id': config.extra.get('config_id'), 'window': windows, 'dose': doses,
         'contextual_findings': findings, 'confirmed_sample_ids': sorted({f['probe_id'] for f in findings if f['confirmed_pattern']}),
         'recovery_token_pieces_exact': exact_pieces,
         'REQUEST_WINDOW_COVERAGE': 'PASS' if all(w['status']=='PASS' for w in windows.values()) else 'WINDOW_INCOMPLETE'},
         status='PASS' if gate else 'PARTIAL', reason='EXPLORATION_INTEGRATION_' + ('PASS' if gate else 'PARTIAL'))
    return EvaluationResult.from_dict(data)
