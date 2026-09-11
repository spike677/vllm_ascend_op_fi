"""Explicit target selection and readable previews; never install or arm hooks."""
from __future__ import annotations

import copy
import json
from .model_descriptor import digest

SELECTION_VERSION = 'EXPLICIT_TARGET_SELECTION_V1'


def parse_ranks(values):
    if values == ['all']:
        return 'all'
    try:
        ranks = [int(v) for v in values]
    except (ValueError, TypeError):
        raise ValueError('RANKS_MUST_BE_EXPLICIT_GLOBAL_INTEGERS_OR_ALL') from None
    return _unique(ranks, 'RANKS', int)


def participating_ranks(reports):
    if not reports:
        raise ValueError('NO_WORKER_REPORTS')
    ranks, epochs = [], set()
    for r in reports:
        d = r['descriptor']
        rank = d.get('ranks', {}).get('global', {}).get('value')
        topology = d.get('topology', {})
        if (type(rank) is not int or rank < 0 or rank in ranks
                or not d.get('service_epoch') or d['service_epoch'] in epochs):
            raise ValueError('MISSING_OR_DUPLICATE_WORKER_IDENTITY')
        if (topology.get('tensor_parallel_size') != len(reports)
                or topology.get('pipeline_parallel_size') != 1
                or topology.get('data_parallel_size') != 1
                or topology.get('enable_expert_parallel') is not False):
            raise ValueError('PARTICIPATING_REQUEST_WORKERS_UNPROVEN_TP_ONLY_REQUIRED')
        tp = d.get('ranks', {}).get('tp', {}).get('value')
        if tp is not None and tp != rank:
            raise ValueError('REFERENCE_GLOBAL_TP_RANK_MAPPING_UNSUPPORTED')
        ranks.append(rank)
        epochs.add(d['service_epoch'])
    # Existing reference and evaluation contracts use a single pure-TP group.
    # Do not reinterpret local/device ranks, DP replicas or incomplete groups.
    if set(ranks) != set(range(len(reports))):
        raise ValueError('REFERENCE_NONCONTIGUOUS_GLOBAL_RANKS_UNSUPPORTED')
    return sorted(ranks)


def _unique(values, name, typ):
    if values is None:
        return None
    if (not values or any(type(v) is not typ for v in values)
            or len(values) != len(set(values))):
        raise ValueError('EXPLICIT_UNIQUE_' + name + '_REQUIRED')
    if typ is int and any(v < 0 for v in values):
        raise ValueError('NEGATIVE_' + name)
    return list(values)


def scoped_targets(report, *, roles=None, layers=None, module_names=None):
    """All user filters intersect. Multiple matching roles do not duplicate a row."""
    return [t for t in report['targets']
            if (roles is None or set(t['roles']) & set(roles))
            and (layers is None or t['layer'] in layers)
            and (module_names is None or t['module_name'] in module_names)]


def resolve_selection(reports, *, mode='manual', roles=None, layers=None,
                      module_names=None, rank_filter=None):
    if mode not in ('manual', 'all_bindable'):
        raise ValueError('UNKNOWN_SELECTION_MODE')
    roles = _unique(roles, 'ROLES', str)
    layers = _unique(layers, 'LAYERS', int)
    module_names = _unique(module_names, 'MODULES', str)
    if mode == 'manual' and not roles and not module_names:
        raise ValueError('MANUAL_SELECTION_REQUIRES_ROLES_OR_EXACT_MODULES')
    participants = participating_ranks(reports)
    ranks = participants if rank_filter == 'all' else _unique(rank_filter, 'RANKS', int)
    if not ranks or set(ranks) - set(participants):
        raise ValueError('REQUESTED_RANK_NOT_IN_PARTICIPATING_WORKERS')
    selected, excluded, duplicate_rows = {}, [], []
    for report in reports:
        check = copy.deepcopy(report)
        if check.pop('report_sha256', None) != digest(check):
            raise ValueError('TARGET_REPORT_INTEGRITY_MISMATCH')
        d = report['descriptor']
        if not d.get('enumeration', {}).get('complete') or d['enumeration'].get('truncated'):
            raise ValueError('ENUMERATION_PARTIAL')
        scoped = scoped_targets(report, roles=roles, layers=layers, module_names=module_names)
        rank, rows = d['ranks']['global']['value'], {}
        for t in scoped:
            name = t['module_name']
            if name in rows:
                # Exact duplicate report rows can only denote the same canonical
                # physical module. Shared aliases remain blocked by the resolver.
                if rows[name] != t:
                    raise ValueError('AMBIGUOUS_DUPLICATE_TARGET:' + name)
                duplicate_rows.append({'rank': rank, 'module': name, 'reason': 'EXACT_DUPLICATE_REPORT_ROW'})
                continue
            rows[name] = t
        if module_names and set(rows) != set(module_names):
            raise ValueError('EXACT_MODULE_FILTER_CONFLICT_OR_MISSING; filters intersect')
        if roles and set(roles) - {role for t in rows.values() for role in t['roles']}:
            raise ValueError('REQUESTED_ROLE_MISSING_IN_SCOPE')
        if layers and set(layers) - {t['layer'] for t in rows.values()}:
            raise ValueError('REQUESTED_LAYER_MISSING_IN_SCOPE')
        if not rows:
            raise ValueError('EMPTY_SELECTED_WORKER_TARGET_SET')
        good = []
        for t in rows.values():
            blocked = not t.get('bound') or bool(t.get('blockers'))
            if blocked and mode == 'manual':
                raise ValueError('MANUALLY_SELECTED_TARGET_NOT_BOUND:' + t['module_name'])
            if blocked:
                excluded.append({'rank': rank, 'module': t['module_name'], 'blockers': t.get('blockers', []),
                                 'reason': 'NOT_BINDABLE'})
            else:
                good.append(t['module_name'])
        if not good:
            raise ValueError('EMPTY_BINDABLE_WORKER_TARGET_SET')
        selected[str(rank)] = sorted(good)
    # The current single-reference owner installs one exact set on each TP
    # worker. Never take the intersection silently if a rank has a binding gap.
    if any(names != next(iter(selected.values())) for names in selected.values()):
        raise ValueError('WORKER_BINDING_GAP_HETEROGENEOUS_TARGETS_UNSUPPORTED')
    return {'version': SELECTION_VERSION, 'mode': mode,
            'intent': {'roles': roles, 'layers': layers, 'module_names': module_names,
                       'ranks': 'all' if rank_filter == 'all' else sorted(ranks)},
            'filter_semantics': 'INTERSECTION; roles match ANY; exact modules never widened',
            'participating_global_ranks': participants, 'targeted_global_ranks': sorted(ranks),
            'per_worker_modules': selected, 'excluded': excluded, 'deduplicated': duplicate_rows,
            'structural_gaps': [{'rank': r['descriptor']['ranks']['global']['value'],
                'gaps': r.get('structural_gaps', r.get('expected_role_gaps', []))} for r in reports],
            'budget_scope': 'PER_WORKER_ARM_LIFETIME; not a shared global budget',
            'coverage': 'supported bound module entries; not arbitrary float/native kernels'}


def next_action(blockers):
    if not blockers:
        return 'PREVIEW_PLAN_THEN_SAME_EPOCH_IDENTITY'
    joined = ' '.join(blockers)
    if 'STALE' in joined:
        return 'REOBSERVE_AND_REBIND_CURRENT_EPOCH'
    if 'NOT_OBSERVED' in joined or 'OBSERVATION_PARTIAL' in joined:
        return 'COMPLETE_CLEAN_OBSERVATION_AND_STOP_OBSERVER'
    return 'ADAPTER_OR_INPUT_CONTRACT_REVIEW_REQUIRED'


def target_view(reports, *, plan=None):
    chosen = {(w['service_epoch'], t['module_name']) for w in (plan or {}).get('workers', [])
              if w['targeted'] for t in w['targets']}
    rows = []
    for r in reports:
        d = r['descriptor']
        for t in r['targets']:
            b = t.get('binding') or {}
            rows.append({'worker_epoch': d['service_epoch'], 'ranks': d.get('ranks', {}),
                'layer': t['layer'], 'roles': t['roles'], 'module': t['module_name'],
                'module_class': t['module_class'], 'parent': t.get('parent'),
                'selector': b.get('selector'), 'semantic_argument': b.get('semantic_argument'),
                'dtype': b.get('dtype'), 'layout': b.get('layout'), 'feature_size': b.get('feature_size'),
                'observations': t.get('input_observations', []),
                'role_proof': t['role_proof'], 'adapter': r['adapter'],
                'report_sha256': r['report_sha256'], 'runtime_fingerprint': d['runtime_fingerprint'],
                'model_fingerprint': d['model_fingerprint'], 'execution': d.get('execution'),
                'states': {'discovered': t.get('discovered'), 'observed': t.get('observed'),
                           'bound': t.get('bound'), 'selected': (d['service_epoch'], t['module_name']) in chosen,
                           'identity': 'NOT_RUN_BY_PREVIEW', 'hit': 'NOT_MEASURED_BY_PREVIEW'},
                'blockers': t['blockers'], 'next_action': next_action(t['blockers'])})
    return {'source': 'MODEL_TARGET_REPORT_V1', 'preview_only': True, 'arm': False, 'rows': rows,
            'selection': (plan or {}).get('selection'), 'plan_sha256': (plan or {}).get('plan_sha256'),
            'budget': (plan or {}).get('request_window'), 'strength': (plan or {}).get('prospective_arm'),
            'selected_worker_targets': len(chosen)}


def render_target_view(view):
    def cell(value):
        return json.dumps(value, ensure_ascii=False).replace('|', '\\|').replace('\n', ' ')
    lines = ['# 注入目标预览（不 arm）', '',
             '不做敏感度排名。所选数量按 worker×module 计，同名模块跨 rank 不合并。', '',
             '来源：' + view.get('observation_source', 'CURRENT_REPORT; execution still requires rebind') + '。', '',
             '| global rank / epoch | layer / role / module | 输入及实际 shape/stride | 依据 | 状态 | blocker / 下一步 |',
             '| --- | --- | --- | --- | --- | --- |']
    for r in view['rows']:
        shapes = []
        for event in r['observations']:
            for parameter in event.get('parameters', []):
                metadata = parameter.get('metadata', {})
                item = {'path': parameter.get('path'), 'shape': metadata.get('shape'),
                        'stride': metadata.get('stride'), 'phase': event.get('phase', 'UNKNOWN')}
                if item not in shapes:
                    shapes.append(item)
        lines.append('| ' + ' | '.join(cell(v) for v in (
            [r['ranks'].get('global'), r['worker_epoch']], [r['layer'], r['roles'], r['module']],
            [r['selector'], r['semantic_argument'], r['dtype'], r['layout'], r['feature_size'], shapes or None],
            [r['module_class'], r['parent'], r['role_proof']],
            r['states'], [r['blockers'], r['next_action']])) + ' |')
    lines += ['', '## 选择、预算与倍率', '', '```json',
              json.dumps({k: view[k] for k in ('selection', 'budget', 'strength', 'selected_worker_targets')},
                         ensure_ascii=False, indent=2, allow_nan=False), '```', '',
              '完整已采集参数、各坐标与来源指纹见同目录 target_preview.json。旧报告缺失 shape/stride 时保持缺失，不由 feature_size 补造。', '']
    return '\n'.join(lines)
