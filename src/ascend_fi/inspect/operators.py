"""Operator Evidence Adapter. Recorded typed probe and native proof remain distinct."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

from ascend_fi.contracts import Candidate
from ascend_fi.discovery.adapters import TraceContext, adapt_qmm_run_trace
from ascend_fi.discovery.resolver import resolve_candidate


META_KEYS = ('dtype', 'shape', 'format', 'strides', 'stride', 'storage_shape', 'contiguous',
             'view_offset', 'numel', 'element_size', 'view_bytes', 'storage_bytes', 'device')
FORMAL_ROLES = {'x1': 'input_activation', 'x2': 'weight', 'x1_scale': 'activation_scale',
                'x2_scale': 'weight_scale', 'y_scale': 'output_scale', 'x1_offset': 'activation_offset',
                'x2_offset': 'weight_offset', 'y_offset': 'output_offset', 'bias': 'bias', 'out': 'output'}


def safe_meta(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('TENSOR_METADATA_MUST_BE_OBJECT_OR_NULL')
    return {k: copy.deepcopy(value[k]) for k in META_KEYS if k in value}


def evidence_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class OperatorEvidenceAdapter:
    """No new live hooks. Never join module/native calls by time, shape or addresses."""
    def __init__(self, capabilities=()):
        self.capabilities = tuple(capabilities)

    def native_parameter(self, record, *, source_ref, runtime=None):
        """Existing formal collection / weight / scale logs, without cross-event guessing."""
        event = record.get('event')
        if event == 'qmm_param_collection':
            params = [{'surface': 'native_formal_args', 'name': name, 'role': role,
                       'formal_ordinal': record.get('formal_ordinals', {}).get(name),
                       'metadata': None, 'runtime_slot_proven': None,
                       'presence': 'UNKNOWN; ordinal -1 is not by itself a tensor absence proof'}
                      for name, role in FORMAL_ROLES.items()]
        elif event in ('qmm_weight_observation', 'qmm_x2_scale_observation'):
            weight = event == 'qmm_weight_observation'
            role = 'weight' if weight else 'weight_scale'
            name = 'x2' if weight else 'x2_scale'
            slot = record.get('x2_slot' if weight else 'scale_slot')
            proven = record.get('weight_slot_proven' if weight else 'scale_slot_proven') is True
            if not weight:
                proven = proven and record.get('scale_role') == 'x2_scale' and record.get('scale_formal_ordinal') == 3
            meta = safe_meta(record)
            meta['shape'] = record.get('view_shape')
            params = [{'surface':'native_formal_args', 'name':name, 'role':role, 'metadata':meta,
                       'runtime_slot_proven':slot if proven and type(slot) is int and slot >= 0 else None,
                       'proof_source':event, 'granularity':'unknown', 'encoding':'unknown',
                       'activation_scale_compatible':False, 'numeric_value':'NOT_CAPTURED'}]
        else:
            raise ValueError('UNSUPPORTED_NATIVE_PARAMETER_EVENT')
        return {'origin':'RECORDED', 'event':event, 'source_ref':source_ref,
                'runtime_fingerprint':runtime or {}, 'parameters':params, 'module_name':None,
                'cross_event_join':'NOT_ATTEMPTED; explicit cache/lifetime evidence required', 'live_arm_allowed':False}

    def native_run(self, record, *, context, origin='RECORDED'):
        if origin != 'RECORDED':
            raise ValueError('LIVE_NATIVE_ADAPTER_NOT_WIRED')
        fact = adapt_qmm_run_trace(record, context)
        if fact is None:
            return None
        # Legacy adapter falls back from rank to device. Do NOT propagate that inference.
        fact = replace(fact, rank=context.rank, phase=context.phase, stream=None, address_metadata={})
        parameters = []
        slots = record['storage_order_slots']
        for name, role in FORMAL_ROLES.items():
            slot = slots.get(name)
            proof = fact.proof.get('semantic_slot_proven') if name == 'x1' else (
                fact.proof.get('output_slot_proven') if name == 'out' else False)
            parameters.append({'surface': 'native_formal_args', 'name': name, 'role': role,
                'presence': 'PRESENT' if name in record and record[name] is not None else
                    'NULL_OR_EMPTY_STORAGE' if slot == -1 else 'METADATA_NOT_IN_THIS_EVENT',
                'metadata': safe_meta(record.get(name)), 'runtime_slot_observed': slot,
                'runtime_slot_proven': slot if proof and isinstance(slot, int) and slot >= 0 else None,
                'slot_proof': 'existing semantic storage-order proof' if proof else 'NOT_PROVEN',
                'granularity': 'unknown', 'numeric_value': 'NOT_CAPTURED'})
        candidates = []
        for cap in self.capabilities:
            candidate = resolve_candidate(fact, cap)
            if candidate:
                d = candidate.to_dict()
                d['binding'].pop('address_metadata', None)
                d.update({'origin': origin, 'module_name': None, 'service_epoch': None,
                          'runtime_fingerprint': context.runtime, 'current_backend': 'NATIVE_BACKEND_UNAVAILABLE',
                          'missing_requirements': ['current worker native runtime', 'current lossless validation'],
                          'layer_selection_supported': False, 'arm_allowed': False})
                candidates.append(Candidate.from_dict(d).to_dict())
        return {'evidence_id': evidence_id(record), 'origin': origin, 'symbol': fact.symbol,
                'overload': record.get('overload'), 'module_name': None, 'module_mapping': 'UNKNOWN',
                'rank': context.rank, 'rank_source': 'explicit evidence context; never RPC array index/device id',
                'phase': context.phase or 'UNKNOWN', 'runtime_fingerprint': context.runtime,
                'parameters': parameters, 'candidates': candidates,
                'runtime_slots': [{'surface':'native_runtime_slots', 'formal_role':p['name'],
                    'observed_slot':p['runtime_slot_observed'], 'proven_slot':p['runtime_slot_proven'],
                    'address':'NOT_EXPORTED; rebind each Run'} for p in parameters],
                'proof': fact.proof, 'source_ref': context.evidence_ref,
                'direct_gws_address_equality_required': False, 'live_arm_allowed': False}

    def typed_probe(self, record, *, source_ref, runtime=None, origin='RECORDED'):
        if origin != 'RECORDED' or record.get('event') != 'typed_qmm_call_exit':
            raise ValueError('ONLY_RECORDED_EXISTING_TYPED_QMM_PROBE_SUPPORTED')
        fields = record.get('tensors', {})
        roles = {'x1': 'operator_input', 'x2': 'operator_weight', 'scale': 'unknown', 'bias': 'bias'}
        params = [{'surface': 'torch_operator_args', 'name': name, 'role': roles.get(name, 'unknown'),
                   'presence': 'PRESENT' if value is not None else 'NONE', 'metadata': safe_meta(value),
                   'native_slot': None, 'semantic_evidence': 'existing typed probe capture; scale target not inferred'}
                  for name, value in fields.items()]
        params.append({'surface': 'torch_operator_args', 'name': 'output', 'role': 'operator_output',
                       'presence': 'PRESENT' if record.get('output') is not None else 'UNKNOWN',
                       'metadata': safe_meta(record.get('output')), 'native_slot': None})
        return {'origin': origin, 'symbol': 'torch_npu.npu_quant_matmul', 'overload': 'NOT_RECORDED',
                'module_name': None, 'module_mapping': 'UNKNOWN', 'parameters': params,
                'schema_limit': 'old probe records args x1/x2/scale and kwarg bias; omitted fields UNKNOWN, not absent',
                'runtime_fingerprint': runtime or {}, 'source_ref': source_ref, 'live_arm_allowed': False}

    def recorded_candidate(self, payload, *, source_ref):
        c = Candidate.from_dict(payload)
        result = c.to_dict()
        result['binding'].pop('address_metadata', None)
        result['observed']['stream'] = None
        result.update({'origin': 'RECORDED', 'source_ref': source_ref, 'module_name': None,
                       'layer_selection_supported': False, 'current_backend': 'NATIVE_BACKEND_UNAVAILABLE',
                       'arm_allowed': False, 'service_epoch': None,
                       'missing_requirements': ['current binding', 'runtime fingerprint', 'new lossless gate']})
        return result
