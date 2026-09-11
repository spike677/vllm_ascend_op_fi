"""Explicit model-aware plans for the existing reference backend, not a controller."""
from __future__ import annotations

import copy
import math
import re

from .model_descriptor import digest
from .model_adapters import expected_role_gaps

SCHEMA = 'FAKE_REQUANT_TARGET_PLAN_V1'


def verify_plan(plan):
    value = copy.deepcopy(plan)
    checksum = value.pop('plan_sha256', None)
    if value.get('schema') != SCHEMA or checksum != digest(value):
        raise ValueError('TARGET_PLAN_INTEGRITY_MISMATCH')
    if plan.get('state') != 'DRAFT' or plan.get('enabled') is not False:
        raise ValueError('ONLY_DISABLED_DRAFT_PLAN_ACCEPTED')
    arm, window = plan['prospective_arm'], plan['request_window']
    names = plan['selected_modules']
    if not names or len(names) != len(set(names)) or len(names) > 512 or any(
            a != b and b.startswith(a + '.') for a in names for b in names):
        raise ValueError('INVALID_PLAN_TARGET_SET')
    expected_regex = r'\A(?:' + '|'.join(re.escape(n) for n in sorted(names)) + r')\Z'
    multiplier = arm.get('scale_multiplier')
    if (arm.get('mode') != 'REFERENCE_REQUANT' or arm.get('scale_provider') != 'tensor_absmax'
            or arm.get('interval') != 1 or arm.get('module_regex') != expected_regex
            or type(multiplier) not in (int, float) or not math.isfinite(multiplier) or not 0 < multiplier <= 20):
        raise ValueError('UNSAFE_PLAN_POLICY')
    if (any(type(window.get(k)) is not int or window[k] <= 0 for k in ('max_tokens', 'max_prompt_tokens'))
            or window.get('max_hits') != len(names) * (window['max_tokens'] + window['max_prompt_tokens'] + 16) * 2
            or window['max_hits'] != arm.get('max_hits') or window['max_hits'] > 2**21
            or window.get('timeout_seconds') != 3600):
        raise ValueError('UNSAFE_PLAN_WINDOW_BUDGET')
    if not plan['workers'] or len({w['rank'] for w in plan['workers']}) != len(plan['workers']):
        raise ValueError('INVALID_WORKER_SET')
    if arm.get('rank_filter') != sorted(w['rank'] for w in plan['workers'] if w['targeted']):
        raise ValueError('INVALID_RANK_POLICY')
    for w in plan['workers']:
        if {t['module_name'] for t in w['targets']} != set(names) or len(w['targets']) != len(names):
            raise ValueError('WORKER_TARGET_SET_MISMATCH')
        for t in w['targets']:
            b = t.get('binding') or {}
            selector = b.get('selector') or {}
            if (t.get('bound') is not True or t.get('blockers') or not selector
                    or selector.get('tensor_selector') == 'first_float'
                    or any(arm.get(k) != v for k, v in selector.items())
                    or b.get('dtype') not in ('torch.float16', 'torch.bfloat16', 'torch.float32')
                    or b.get('contiguous') is not True or b.get('ndim') != 2):
                raise ValueError('UNSAFE_INPUT_BINDING')


def build_target_plan(reports, *, roles, rank_filter, scale_multiplier, reference_arm,
                      max_tokens, max_prompt_tokens, phase='ALL_OBSERVED'):
    if not reports or not roles or len(roles) != len(set(roles)):
        raise ValueError('EXPLICIT_UNIQUE_ROLES_REQUIRED')
    if not rank_filter or any(type(r) is not int or r < 0 for r in rank_filter) or len(set(rank_filter)) != len(rank_filter):
        raise ValueError('EXPLICIT_UNIQUE_RANKS_REQUIRED')
    if type(scale_multiplier) not in (int, float) or not math.isfinite(scale_multiplier) or not 0 < scale_multiplier <= 20:
        raise ValueError('MULTIPLIER_OUTSIDE_REFERENCE_V1_VALIDATED_CONFIG_RANGE')
    if (reference_arm.get('mode') != 'REFERENCE_REQUANT'
            or reference_arm.get('scale_provider') != 'tensor_absmax'):
        raise ValueError('REFERENCE_REQUANT_TENSOR_ABSMAX_REQUIRED; no fallback')
    if any(type(v) is not int or v <= 0 for v in (max_tokens, max_prompt_tokens)):
        raise ValueError('FINITE_GENERATION_BUDGET_REQUIRED')
    if phase not in ('ALL_OBSERVED', 'prefill', 'decode'):
        raise ValueError('UNKNOWN_PHASE_POLICY')
    workers, selectors, model_configs, epochs = [], [], [], set()
    for report in reports:
        check = copy.deepcopy(report)
        if check.pop('report_sha256', None) != digest(check):
            raise ValueError('TARGET_REPORT_INTEGRITY_MISMATCH')
        d = report['descriptor']
        if d.get('origin') != 'LIVE':
            raise ValueError('RECORDED_REPORT_REQUIRES_LIVE_REBIND')
        reference = d.get('backend', {}).get('reference') or {}
        if not re.fullmatch(r'[0-9a-f]{64}', reference.get('source_sha256') or ''):
            raise ValueError('REFERENCE_BACKEND_SOURCE_HASH_UNAVAILABLE')
        if d.get('enumeration', {}).get('complete') is not True or d['enumeration'].get('truncated'):
            raise ValueError('ENUMERATION_PARTIAL')
        rank = d.get('ranks', {}).get('global', {}).get('value')
        if type(rank) is not int or rank in {w['rank'] for w in workers}:
            raise ValueError('GLOBAL_RANK_MISSING_OR_DUPLICATE')
        if d['service_epoch'] in epochs:
            raise ValueError('DUPLICATE_WORKER_EPOCH')
        epochs.add(d['service_epoch'])
        topology = d.get('topology', {})
        if (topology.get('pipeline_parallel_size') != 1 or topology.get('data_parallel_size') != 1
                or topology.get('enable_expert_parallel') is not False
                or topology.get('tensor_parallel_size') != len(reports)):
            raise ValueError('TOPOLOGY_REQUIRES_EXPLICIT_TP_ONLY_CONTRACT')
        if d.get('execution', {}).get('enforce_eager') is not True:
            raise ValueError('EXECUTION_MODE_UNPROVEN; will not silently enable eager')
        if report.get('adapter', {}).get('qualification') != 'CURRENT_EPOCH_IDENTITY_REQUIRED':
            raise ValueError('VARIANT_READONLY_UNTIL_LIVE_ADAPTED')
        gaps = expected_role_gaps(d, [], [dict(t, layer_index=t['layer']) for t in report['targets']], roles)
        if gaps:
            raise ValueError('ARCHITECTURE_EXPECTED_COVERAGE_GAP:' + str(gaps[:5]))
        selected = [t for t in report['targets'] if set(t['roles']) & set(roles)]
        if not selected or any(not t['bound'] or t['blockers'] for t in selected):
            raise ValueError('SELECTED_TARGET_NOT_BOUND')
        names = [t['module_name'] for t in selected]
        if len(names) != len(set(names)) or any(a != b and b.startswith(a + '.') for a in names for b in names):
            raise ValueError('DUPLICATE_OR_PARENT_CHILD_TARGET_CONFLICT')
        if phase != 'ALL_OBSERVED' and any(phase not in t['observed_phases'] for t in selected):
            raise ValueError('REQUESTED_PHASE_NOT_PROVEN')
        selectors.extend(t['binding']['selector'] for t in selected)
        # The reference owner currently installs one exact module set on TP workers.
        if workers and set(names) != {t['module_name'] for t in workers[0]['targets']}:
            raise ValueError('HETEROGENEOUS_WORKER_TARGETS_REQUIRE_OWNER_ADAPTER')
        model_configs.append(d['config'])
        workers.append({'rank': rank, 'service_epoch': d['service_epoch'],
                        'runtime_fingerprint': d['runtime_fingerprint'],
                        'model_fingerprint': d['model_fingerprint'],
                        'backend': copy.deepcopy(d['backend']),
                        'adapter': report['adapter'], 'targeted': rank in rank_filter,
                        'targets': copy.deepcopy(selected)})
    if set(rank_filter) - {w['rank'] for w in workers} or any(c != model_configs[0] for c in model_configs):
        raise ValueError('WORKER_MODEL_OR_RANK_POLICY_MISMATCH')
    if any(s != selectors[0] for s in selectors):
        raise ValueError('MIXED_SELECTORS_NOT_SUPPORTED_BY_SINGLE_REFERENCE_OWNER; no first_float fallback')
    names = sorted(t['module_name'] for t in workers[0]['targets'])
    if len(names) > 512:
        raise ValueError('CURRENT_OBSERVER_TARGET_BUDGET_EXCEEDED')
    cap = len(names) * (max_tokens + max_prompt_tokens + 16) * 2
    if cap > 2**21:
        raise ValueError('CONSERVATIVE_CAP_EXCEEDS_REFERENCE_SAFETY_LIMIT; do not silently clamp')
    arm = copy.deepcopy(reference_arm)
    arm.update(selectors[0])
    arm.update(mode='REFERENCE_REQUANT', scale_provider='tensor_absmax', scale_multiplier=scale_multiplier,
               module_regex=r'\A(?:' + '|'.join(re.escape(n) for n in names) + r')\Z',
               layer_filter=sorted({t['layer'] for t in workers[0]['targets']}),
               rank_filter=sorted(rank_filter), interval=1, max_hits=cap)
    value = {'schema': SCHEMA, 'state': 'DRAFT', 'enabled': False, 'arm_allowed': False,
             'workers': workers, 'selected_modules': names, 'roles': list(roles),
             'strategy': 'REFERENCE_REQUANT', 'scale_provider': 'tensor_absmax',
             'multiplier_semantics': 'multiply_quantization_step; multiplier=1 is NOT IDENTITY',
             'prospective_arm': arm,
             'request_window': {'policy': 'EXCLUSIVE_SINGLE_REQUEST_TRANSACTION', 'phase': phase,
                 'max_tokens': max_tokens, 'max_prompt_tokens': max_prompt_tokens,
                 'max_hits': cap, 'timeout_seconds': 3600,
                 'cap_formula': 'module_count * (max_tokens + max_prompt_tokens + 16) * 2',
                 'scheduler_step_coverage': 'UNPROVEN; counters alone never certify full window',
                 'log_limit': 32, 'dose_sample_every_n_module_calls': 64},
             'conflict_check': 'PASS', 'required_validation': ['same-epoch identity', 'exact installed set',
                 'exclusive service lease', 'exact token-piece recovery', 'independent execution coverage'],
             'coverage_scope': 'selected module entry boundaries; not fused internal kernels',
             'report_refs': [r['report_sha256'] for r in reports]}
    value['plan_sha256'] = digest(value)
    return value
