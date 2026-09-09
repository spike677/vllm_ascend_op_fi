"""Candidate projection and DRAFT export. Never broaden selectors or arm here."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from ascend_fi.contracts import Candidate, ValidationResult, ValidationStatus


def parameter_selector(path):
    if len(path) == 2 and path[0] == 'args' and type(path[1]) is int and path[1] >= 0:
        return {'tensor_selector': 'tensor_index', 'tensor_index': path[1]}
    if len(path) == 2 and path[0] == 'kwargs' and type(path[1]) is str and path[1]:
        return {'tensor_selector': 'kwarg:' + path[1], 'tensor_index': None}
    raise ValueError('EXISTING_BACKEND_CANNOT_SELECT_NESTED_PATH')


def module_candidates(session, backend_available=False):
    candidates = {}
    with session.lock:
        events = copy.deepcopy(session.events)
    for event in events:
        module = session.surface.by_name[event['module_name']]
        for parameter in event['parameters']:
            meta, path = parameter['metadata'], parameter['path']
            if meta.get('floating') is not True:
                continue
            missing = []
            try:
                selector = parameter_selector(path)
            except ValueError as e:
                selector = None
                missing.append(str(e))
            if len(module['aliases']) != 1:
                missing.append('SHARED_MODULE_CALLSITE_UNRESOLVED')
            if meta.get('is_meta') or meta.get('metadata_status') != 'COMPLETE':
                missing.append('METADATA_NOT_EXECUTABLE')
            if not backend_available:
                missing.append('RESEARCH_REFERENCE_BACKEND_UNAVAILABLE')
            rank = event['ranks'].get('global', {}).get('value')
            if rank is None:
                missing.append('GLOBAL_RANK_UNRESOLVED')
            key = json.dumps([session.epoch, event['session'], event['module_name'], path], sort_keys=True)
            cid = 'inspect-' + hashlib.sha256(key.encode()).hexdigest()[:20]
            stages = {'FOUND': True, 'OBSERVED': True, 'ROLE_RESOLVED': True,
                      'BACKEND_BOUND': False, 'LOSSLESS_VALIDATED': False}
            unique_float = sum(p['metadata'].get('floating') is True and len(p['path']) == 2
                               for p in event['parameters']) == 1 and len(path) == 2
            candidate = Candidate.from_dict({'candidate_id': cid,
                'capability_id': 'python.fake_requant.activation',
                'observed': {'execution_plane': 'torch', 'symbol': module['module_class'],
                    'rank': rank, 'phase': 'UNKNOWN', 'shape': meta.get('shape'), 'dtype': meta.get('dtype'),
                    'module_name': event['module_name'], 'layer_index': module['layer_index'],
                    'parameter_path': path, 'surface': 'module_args'},
                'binding': {'tensor_role': 'floating_module_input', 'runtime_slot': None,
                            'parameter_selector': selector},
                'proof': {'level': 'OBSERVED', 'evidence_refs': [event['session']], 'stages': stages},
                'validation': {'status': 'UNKNOWN', 'reason': 'explicit IDENTITY/lossless validation required'},
                'recommended': {'strategy': 'REFERENCE_REQUANT', 'params': {}},
                'service_epoch': session.epoch, 'runtime_fingerprint': session.fingerprint,
                'origin': 'LIVE', 'module_name': event['module_name'], 'logical_role': module['logical_role'],
                'quantization_binding': 'REFERENCE_SCALE_POLICY; not MODEL_QUANTIZATION_SCALE',
                'backend': 'FakeRequantReferenceExtension', 'selector_representable': selector is not None,
                'unique_float_input_proven': unique_float,
                'missing_requirements': missing + ['IDENTITY binding and clean recovery'],
                'export_blockers': missing, 'arm_allowed': False, 'status': 'RESEARCH_REFERENCE_ONLY'})
            entry = candidate.to_dict()
            if cid in candidates:
                entry['unique_float_input_proven'] &= candidates[cid]['unique_float_input_proven']
                entry['export_blockers'] = sorted(set(entry['export_blockers'] + candidates[cid]['export_blockers']))
            candidates[cid] = entry
    return list(candidates.values())


def export_plan(selected, universe, *, service_epoch, runtime_fingerprint, frozen_arm,
                scale_source='reference', layer_specific=True):
    if not selected:
        raise ValueError('NO_CANDIDATES_SELECTED')
    if scale_source != 'reference':
        raise ValueError('MODEL_SCALE_PROVIDER_NOT_BOUND; no silent reference fallback')
    if any(c.get('origin') != 'LIVE' for c in selected):
        raise ValueError('RECORDED_EVIDENCE_REQUIRES_LIVE_REBIND')
    if any(c.get('service_epoch') != service_epoch or c.get('runtime_fingerprint') != runtime_fingerprint for c in selected):
        raise ValueError('STALE_SERVICE_EPOCH_OR_RUNTIME')
    if any(c['observed']['execution_plane'] != 'torch' for c in selected):
        raise ValueError('NATIVE_LAYER_SELECTION_UNPROVEN' if layer_specific else 'NATIVE_EXPORT_REQUIRES_EXISTING_CONTROLLER_ADAPTER')
    if any(c.get('export_blockers') for c in selected):
        raise ValueError('BACKEND_SELECTION_UNSUPPORTED:' + str([c.get('export_blockers') for c in selected]))
    selectors = [c['binding']['parameter_selector'] for c in selected]
    if any(s != selectors[0] for s in selectors):
        raise ValueError('MIXED_SELECTORS_REQUIRE_SEPARATE_PLANS')
    modules = sorted({c['module_name'] for c in selected})
    if len(modules) != len(selected):
        raise ValueError('ONE_PARAMETER_PATH_PER_MODULE_PER_PLAN')
    regex = r'\A(?:' + '|'.join(re.escape(n) for n in modules) + r')\Z'
    if {n for n in universe if re.search(regex, n)} != set(modules):
        raise ValueError('REGEX_SELECTION_NOT_EQUIVALENT')
    ranks = sorted({c['observed']['rank'] for c in selected})
    required = {'mode', 'scale_provider', 'scale_multiplier', 'max_hits', 'interval', 'rank_filter'}
    if not required.issubset(frozen_arm) or frozen_arm['mode'] != 'REFERENCE_REQUANT':
        raise ValueError('EXACT_FROZEN_RESEARCH_ARM_CONFIG_REQUIRED')
    if frozen_arm['scale_provider'] != 'tensor_absmax' or frozen_arm['rank_filter'] != ranks:
        raise ValueError('FROZEN_SCALE_OR_RANK_POLICY_MISMATCH')
    arm = copy.deepcopy(frozen_arm)
    arm.update(selectors[0])
    arm['module_regex'] = regex
    arm['layer_filter'] = sorted({c['observed']['layer_index'] for c in selected if c['observed']['layer_index'] is not None}) or None
    return {'schema': 'INTERACTIVE_INSPECT_PLAN_V1', 'state': 'DRAFT', 'enabled': False,
            'arm_allowed': False, 'research_status': 'RESEARCH_REFERENCE_ONLY',
            'service_epoch': service_epoch, 'runtime_fingerprint': runtime_fingerprint,
            'selected_modules': modules, 'candidate_ids': [c['candidate_id'] for c in selected],
            'parameter_path': selected[0]['observed']['parameter_path'], 'ranks': ranks,
            'selector_equivalence_verified': True, 'backend': 'FakeRequantReferenceExtension',
            'unique_float_input_proven': all(c.get('unique_float_input_proven') for c in selected),
            'frozen_arm_input': copy.deepcopy(frozen_arm), 'prospective_arm': arm,
            'requires': ['same-epoch resolve', 'explicit IDENTITY validation', 'exclusive service lease',
                         'existing lifecycle cleanup', 'per-request re-arm'],
            'validation_mode': 'IDENTITY', 'native_addresses': None}


def validate_plan(plan, *, epoch, fingerprint, executor):
    """Delegate to an existing lifecycle adapter; no install/arm RPC in Inspect."""
    cid = ','.join(plan.get('candidate_ids', []))
    if plan.get('service_epoch') != epoch or plan.get('runtime_fingerprint') != fingerprint:
        return ValidationResult(cid, False, False, False, status=ValidationStatus.REJECTED,
                                reason='STALE_SERVICE_EPOCH_OR_RUNTIME')
    if plan.get('state') != 'DRAFT' or plan.get('enabled') is not False or plan.get('validation_mode') != 'IDENTITY':
        return ValidationResult(cid, False, False, False, status=ValidationStatus.REJECTED,
                                reason='ONLY_EXPLICIT_IDENTITY_DELEGATION_ALLOWED')
    # Adapter owns service lease, disabled/IDENTITY request, finally cleanup + recovery.
    return executor.validate_identity(copy.deepcopy(plan))
