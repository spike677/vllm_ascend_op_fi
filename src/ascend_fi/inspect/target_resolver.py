"""Structural candidates + same-epoch execution metadata -> explicit bindings."""
from __future__ import annotations

import copy
from .model_adapters import structural_targets
from .model_descriptor import digest
from .plans import parameter_selector

FLOAT_DTYPES = {'torch.float16', 'torch.bfloat16', 'torch.float32'}


def resolve_targets(descriptor, modules, events, observation_status):
    adapter, targets, structural_gaps = structural_targets(descriptor, modules)
    rows = []
    observation_errors = []
    if (observation_status.get('terminal_stop_reason') or observation_status.get('errors')
            or observation_status.get('active') or observation_status.get('installed_handles')):
        observation_errors.append('OBSERVATION_PARTIAL_OR_NOT_STOPPED')
    for t in targets:
        seen = [e for e in events if e['module_name'] == t['module_name']]
        blockers = list(observation_errors)
        if not t['class_contract']:
            blockers.append('MODULE_IMPLEMENTATION_CONTRACT_UNKNOWN')
        if len(t.get('aliases', [])) != 1:
            blockers.append('SHARED_MODULE_ALIAS_UNSUPPORTED')
        if not seen:
            blockers.append('MODULE_ENTRY_NOT_OBSERVED; functional/fused bypass not excluded')
        bindings = []
        for event in seen:
            if (event.get('origin') != 'LIVE' or event.get('service_epoch') != descriptor['service_epoch']
                    or event.get('runtime_fingerprint') != descriptor['runtime_fingerprint']
                    or event.get('session') != observation_status.get('session')):
                blockers.append('STALE_OR_RECORDED_CALL')
            floating = [p for p in event['parameters'] if p['metadata'].get('floating') is True]
            # Conservative V1: no ambiguous auxiliaries / storage-view alias contracts.
            if len(floating) != 1:
                blockers.append('MULTIPLE_OR_MISSING_FLOAT_INPUTS_REQUIRE_SEMANTIC_ALIAS_CONTRACT')
                continue
            p = floating[0]
            m = p['metadata']
            if p.get('semantic_argument') not in t['semantic_args']:
                blockers.append('ACTIVATION_ARGUMENT_SEMANTICS_UNPROVEN')
            if p.get('object_alias_paths'):
                blockers.append('INPUT_ALIAS_CONTRACT_UNSUPPORTED')
            if (m.get('dtype') not in FLOAT_DTYPES or m.get('layout') != 'torch.strided'
                    or m.get('is_meta') is not False or m.get('contiguous') is not True
                    or m.get('metadata_status') != 'COMPLETE'
                    or len(m.get('shape') or []) != 2 or not all(type(v) is int and v > 0 for v in m['shape'])):
                blockers.append('INPUT_METADATA_CONTRACT_UNSUPPORTED')
                continue
            feature = t.get('input_feature_size')
            if type(feature) is not int or feature <= 0 or feature != m['shape'][-1]:
                blockers.append('PARTITION_INPUT_FEATURE_NOT_PROVEN')
            try:
                selector = parameter_selector(p['path'])
            except ValueError:
                blockers.append('NESTED_SELECTOR_UNSUPPORTED')
                continue
            bindings.append({'semantic_argument': p.get('semantic_argument'), 'path': p['path'],
                'selector': selector, 'dtype': m['dtype'], 'layout': m['layout'],
                'ndim': 2, 'contiguous': True, 'feature_size': m['shape'][-1],
                'token_dimension': 'dynamic_positive', 'non_inplace_replacement': True})
        if bindings and any(b != bindings[0] for b in bindings):
            blockers.append('INPUT_BINDING_CHANGED_BETWEEN_CALLS')
        row = {'module_name': t['module_name'], 'module_class': t['module_class'],
               'roles': t['roles'], 'layer': t['layer_index'], 'ranks': t['ranks'],
               'scope': t['scope'], 'role_proof': t['role_proof'],
               'discovered': True, 'role_resolved': True, 'observed': bool(seen),
               'bound': bool(bindings) and not blockers, 'identity_pass': False,
               'lossy_mechanism_pass': False, 'visible_effect_pass': False,
               'binding': bindings[0] if bindings else None,
               'observed_phases': sorted({e.get('phase', 'UNKNOWN') for e in seen}),
               'call_count': observation_status.get('module_call_counts', {}).get(t['module_name'], 0),
               'blockers': sorted(set(blockers)), 'observed_samples': len(seen)}
        rows.append(row)
    report = {'schema': 'MODEL_TARGET_REPORT_V1', 'descriptor': copy.deepcopy(descriptor),
              'adapter': adapter, 'targets': rows, 'structural_gaps': structural_gaps,
              'observation_status': copy.deepcopy(observation_status),
              'native_scale_semantics': 'UNPROVEN; tensor_absmax reference does not require them',
              'status': 'OBSERVED' if rows and all(t['bound'] for t in rows) else 'PARTIAL',
              'arm_allowed': False}
    report['report_sha256'] = digest(report)
    return report
