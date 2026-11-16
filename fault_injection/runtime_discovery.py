"""Read-only operator discovery and selection for the existing NativeSession."""
from dataclasses import asdict
import hashlib
import json
import os
import re
import uuid

from .core import UnsupportedCall
from .coverage import Candidate, CandidateSet, Eligibility, EligibilityResult, TargetIdentity
from .environment import environment_fingerprint


_ATEN = {
    'aten::mm', 'aten::addmm', 'aten::bmm', 'aten::baddbmm', 'aten::matmul', 'aten::linear',
    'aten::scaled_dot_product_attention', 'aten::_scaled_dot_product_flash_attention',
    'aten::_scaled_dot_product_flash_attention_for_cpu', 'aten::_scaled_dot_product_efficient_attention',
}
_NATIVE_BOUNDARIES = {'aclnn_execute', 'op_exec_cache_run', 'host_kernel_launch'}


def install_dispatch_observer(callback, current_owner):
    """Observe executed Torch dispatcher operations, without changing the route."""
    from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode
    from .native import NativeCall

    session_id = uuid.uuid4().hex
    serial = 0

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal serial
            kwargs = {} if kwargs is None else kwargs
            owner = current_owner()
            if owner is None:
                return func(*args, **kwargs)
            schema = getattr(func, '_schema', None)
            if schema is None:
                return func(*args, **kwargs)
            if schema.name not in _ATEN and not schema.name.startswith('npu::'):
                return func(*args, **kwargs)
            serial += 1
            operator = schema.name + '.' + (schema.overload_name or 'default')
            call = NativeCall(owner, f'{session_id}:{serial}', operator, args, kwargs)
            return callback(call, lambda: func(*args, **kwargs))

    mode = Observe()
    mode.__enter__()

    def remove():
        if _get_current_dispatch_mode() is not mode:
            raise RuntimeError('Dispatcher observation mode ownership changed')
        mode.__exit__(None, None, None)

    return remove


def dispatch_tensors(call):
    """Read Tensor descriptors only. No data pointer, device copy or value access."""
    import torch

    records = []
    for name, value in [*enumerate(call.arguments), *call.kwargs.items()]:
        if isinstance(value, torch.Tensor):
            row = dict(argument=str(name), shape=[str(dim) if type(dim) is not int else dim for dim in value.shape],
                       dtype=str(value.dtype), device_type=value.device.type, layout=str(value.layout))
            if value.layout == torch.strided:
                row['stride'] = [str(dim) if type(dim) is not int else dim for dim in value.stride()]
                row['offset'] = value.storage_offset()
            records.append(row)
    return records


def native_facts(details, proof_kinds, fault):
    """Check the schema of a call-bound observation from a trusted adapter."""
    from .native import REQUIREMENTS

    if not isinstance(details, dict):
        raise UnsupportedCall('Read-only provider did not supply runtime_call evidence')
    worker = details.get('worker')
    layout = details.get('layout')
    slots = details.get('slots')
    if not isinstance(worker, dict) or not isinstance(layout, dict) or not isinstance(slots, list):
        raise UnsupportedCall('Worker, layout or formal-slot observation missing')
    if details.get('boundary') not in _NATIVE_BOUNDARIES:
        raise UnsupportedCall('Unsupported native observation boundary')
    proof = details.get('build_profile_sha256')
    if not isinstance(proof, str) or not re.fullmatch('[0-9a-f]{64}', proof):
        raise UnsupportedCall('Native build profile identity missing')
    if details.get('runtime_identity') != 'MATCHED_BUILD':
        raise UnsupportedCall('Loaded-code attestation did not match the build')
    required = set(REQUIREMENTS[fault]) - {'input_transform', 'finite_input_range'}
    missing = required - proof_kinds
    if missing:
        raise UnsupportedCall('Missing read-only native proofs: ' + ', '.join(sorted(missing)))
    for key in ('native_signature', 'cache_identity'):
        value = details.get(key)
        if not isinstance(value, str) or not value or len(value) > 2048:
            raise UnsupportedCall('Missing stable native observation field: ' + key)
    size = details.get('candidate_count')
    if type(size) is not int or size < 1:
        raise UnsupportedCall('Missing native cache/candidate multiplicity')
    if any(type(worker.get(key)) is not int or worker[key] < 0 for key in ('pid', 'tp', 'rank', 'logical_device')):
        raise UnsupportedCall('Incomplete worker identity')
    if worker['pid'] != os.getpid() or not 0 <= worker['rank'] < worker['tp']:
        raise UnsupportedCall('Runtime evidence is from another worker')
    geometry = (layout.get('shape'), layout.get('stride'))
    if (not all(isinstance(field, list) and field for field in geometry)
            or len(geometry[0]) != len(geometry[1])
            or any(type(n) is not int or n <= 0 for n in geometry[0])
            or any(type(n) is not int or n < 0 for n in geometry[1])
            or type(layout.get('offset')) is not int or layout['offset'] < 0):
        raise UnsupportedCall('Incomplete tensor layout')
    for data, keys in ((worker, ('execution_mode', 'soc')), (layout, ('dtype', 'format'))):
        if any(not isinstance(data.get(key), str) or not data[key].strip() for key in keys):
            raise UnsupportedCall('Missing worker or live tensor layout identity')
    slot_names = set()
    for row in slots:
        if (not isinstance(row, dict) or not isinstance(row.get('formal'), str) or not row['formal']
                or type(row.get('slot')) is not int or row['slot'] < 0
                or type(row.get('storage_bytes')) is not int or row['storage_bytes'] <= 0
                or row['formal'] in slot_names):
            raise UnsupportedCall('Invalid formal storage-slot observation')
        slot_names.add(row['formal'])
    if not slots:
        raise UnsupportedCall('Missing formal storage-slot observation')
    result = dict(
        boundary=details['boundary'], native_signature=details['native_signature'], cache_identity=details['cache_identity'],
        build_profile_sha256=proof, candidate_count=size,
        worker={key: worker[key] for key in ('pid', 'tp', 'rank', 'logical_device', 'soc', 'execution_mode')},
        layout={key: layout[key] for key in ('dtype', 'format', 'shape', 'stride', 'offset')},
        slots=[{key: row[key] for key in ('formal', 'slot', 'storage_bytes')}
               for row in sorted(slots, key=lambda row: (row['slot'], row['formal']))],
    )
    material = {key: result[key] for key in ('layout', 'slots')}
    result['layout_sha256'] = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    for source, target, fields in (
        (worker, result['worker'], ('worker_id', 'model_instance_id')),
        (details, result, ('cache_lifecycle',)),
    ):
        for key in fields:
            if key in source:
                value = source[key]
                if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                    raise UnsupportedCall('Invalid live identity: ' + key)
                target[key] = value
    return result


def _same_scope(profile, event, model):
    facts = event['native_facts']
    layout, worker = facts['layout'], facts['worker']
    actual = dict(operator=event['operator'], boundary=facts['boundary'], native_signature=facts['native_signature'],
                  build_profile_sha256=facts['build_profile_sha256'], layout_sha256=facts['layout_sha256'],
                  dtype=layout['dtype'], format=layout['format'], ndim=len(layout['shape']),
                  soc=worker['soc'], execution_mode=worker['execution_mode'],
                  tp=worker['tp'], rank=worker['rank'], logical_device=worker['logical_device'])
    if profile.model_class is not None:
        actual['model_class'] = model['model_class']
    if profile.model_type is not None:
        actual['model_type'] = model['model_type']
    return all(getattr(profile, name) == value for name, value in actual.items())


def _single_target_plan(report, model, profiles, before, after, discovery_finished, first_live_validation):
    """Find a unique repeated match. Selection still needs the provider's lossless gate."""
    events = report['events']
    observed = [event for event in events if event['status'] == 'observed_readonly']
    native = [event for event in observed if event.get('native_facts')]
    result = dict(schema='demoncat.adaptation.v1', mechanism=report['mechanism'],
                  status='RUNTIME_OBSERVATION_REQUIRED', target=None, native_execution_authorized=False,
                  profiles=[asdict(profile) for profile in profiles], discovered_calls=len(observed),
                  native_candidates=len(native), candidates=[], gaps=[],
                  unobserved_candidate_invocations=sum(
                      not row['observed_calls'] for row in report['module_invocations']
                  ))
    if (not (report['closed'] or discovery_finished)
            or report['cleanup_errors'] or report['cleanup_pending']):
        result['status'] = ('OBSERVING' if report['active'] and not report['cleanup_pending']
                            else 'OBSERVATION_NOT_CLEANLY_FINISHED')
        return result
    forwards = report['forwards']
    if (len(forwards) < 2 or any(row['status'] != 'returned' or row['errors'] for row in forwards)
            or report['rejected_root_calls'] or any(event['status'] != 'observed_readonly' for event in events)):
        result['status'] = 'REPEATED_FORWARD_EVIDENCE_REQUIRED'
        return result
    if not native:
        result['status'] = 'NATIVE_OPERATOR_EVIDENCE_REQUIRED'
        result['gaps'] = sorted({event['reason'] for event in observed if event.get('reason')})
        return result
    if not profiles:
        result['status'] = 'PROFILE_REQUIRED'
        return result
    if before is None or after is None or before.get('pid') != os.getpid() or after.get('pid') != os.getpid():
        result['status'] = 'WORKER_ENVIRONMENT_EVIDENCE_REQUIRED'
        return result
    ambiguous = False
    for profile in profiles:
        if (profile.validation_level in ('DECLARED', 'IMPLEMENTED')
                or (profile.validation_level == 'CPU_VERIFIED' and not first_live_validation)):
            result['gaps'].append('NATIVE_VALIDATION_REQUIRED')
            continue
        expected = profile.environment_fingerprint
        names = profile.required_libraries
        if (environment_fingerprint(before, names) != expected
                or environment_fingerprint(after, names) != expected):
            result['gaps'].append('LOADED_ENVIRONMENT_FINGERPRINT_MISMATCH')
            continue
        matches = [event for event in native if _same_scope(profile, event, model)]
        if not matches:
            result['gaps'].append('OPERATOR_OR_WORKER_SCOPE_MISMATCH')
            continue
        if any(event['operator'] == profile.operator and not event.get('native_facts') for event in observed):
            ambiguous = True
            result['gaps'].append('OPERATOR_CALLS_WITH_UNPROVEN_IDENTITY')
            continue
        for path in dict.fromkeys(event['path'] for event in matches):
            calls = [event for event in matches if event['path'] == path]
            facts = calls[0]['native_facts']
            if ([event['forward'] for event in calls] != [row['index'] for row in forwards]
                    or facts['candidate_count'] != 1
                    or any(event['native_facts'] != facts
                           or event['scope_ordinal'] != calls[0]['scope_ordinal'] for event in calls)):
                ambiguous = True
                result['gaps'].append('CANDIDATE_NOT_UNIQUE_AND_STABLE')
                continue
            result['candidates'].append(dict(path=path, object_identity=calls[0]['object_identity'],
                roles=calls[0]['canonical_roles'], role_status=calls[0]['role_status'],
                operator=profile.operator, native_signature=profile.native_signature,
                build_profile_sha256=profile.build_profile_sha256,
                environment_fingerprint=expected, implementation_version=profile.implementation_version,
                validation_level=profile.validation_level, evidence_sha256=profile.evidence_sha256,
                worker=facts['worker'], scope_ordinal=calls[0]['scope_ordinal'],
                call_ids=[event['call_id'] for event in calls]))
    result['gaps'] = sorted(set(result['gaps']))
    if len(result['candidates']) == 1 and not ambiguous:
        result.update(status='LOSSLESS_GATE_REQUIRED', target=result['candidates'][0])
    elif result['candidates']:
        result['status'] = 'AMBIGUOUS_TARGET'
    else:
        result['status'] = 'NO_COMPATIBLE_STABLE_TARGET'
    return result


def _candidate_set(report, model, plan, finished):
    groups = {}
    for event in report['events']:
        if event['status'] != 'observed_readonly':
            continue
        facts = event.get('native_facts', {})
        worker = facts.get('worker', {})
        key = (event['object_identity'], event['operator'], facts.get('build_profile_sha256'),
               facts.get('boundary'), facts.get('cache_identity'), facts.get('cache_lifecycle'),
               facts.get('native_signature'), facts.get('layout_sha256'),
               worker.get('worker_id'), worker.get('model_instance_id'), worker.get('rank'),
               worker.get('logical_device'))
        groups.setdefault(key, []).append(event)
    profiles_by_call = {}
    for candidate in plan['candidates']:
        for call_id in candidate['call_ids']:
            profiles_by_call.setdefault(call_id, []).append(candidate)
    rows = []
    for calls in groups.values():
        first = calls[0]
        facts = first.get('native_facts')
        worker = facts['worker'] if facts else {}
        matches = profiles_by_call.get(first['call_id'], ())
        identity = None
        if (facts and facts.get('cache_lifecycle') and worker.get('worker_id')
                and worker.get('model_instance_id')):
            identity = TargetIdentity(
                worker['pid'], worker['worker_id'], worker['rank'], worker['logical_device'],
                worker['model_instance_id'], model['object_identity'], first['object_identity'],
                tuple(sorted(first['canonical_roles'])), first['operator'], facts['boundary'],
                facts['cache_identity'], facts['cache_lifecycle'],
                facts['native_signature'] + ':' + facts['layout_sha256'],
                facts['build_profile_sha256'],
                provider_identity=(f'{report["mechanism"]}:{matches[0]["implementation_version"]}'
                                   if len(matches) == 1 else None),
                provider_proof=(('layout_sha256', facts['layout_sha256']),) if len(matches) == 1 else (),
            )
        if len(matches) > 1 or (facts and facts['candidate_count'] > 1):
            status, reason = Eligibility.AMBIGUOUS, 'NATIVE_CALL_OR_PROFILE_IDENTITY_AMBIGUOUS'
        elif not facts:
            status, reason = Eligibility.UNVERIFIED, 'NATIVE_OPERATOR_EVIDENCE_REQUIRED'
        elif not identity:
            status, reason = Eligibility.UNVERIFIED, 'WORKER_MODEL_LOAD_OR_CACHE_LIFECYCLE_UNPROVEN'
        elif not matches:
            status, reason = Eligibility.UNVERIFIED, 'STABLE_PROFILE_MATCH_NOT_ESTABLISHED'
        else:
            status, reason = Eligibility.SUPPORTED, None
        rows.append(Candidate(f'call:{first["call_id"]}',
                              EligibilityResult(status, (reason,) if reason else ()), identity))
    unseen = report['unobserved_candidates']
    for path in unseen:
        rows.append(Candidate(f'module:{path}', EligibilityResult(
            Eligibility.NOT_OBSERVED, ('NO_OPERATOR_OBSERVED_FOR_MODULE',))))
    return CandidateSet(tuple(rows), discovery_complete=bool(
        groups and finished and not unseen and not plan['unobserved_candidate_invocations']
        and plan['status'] in ('LOSSLESS_GATE_REQUIRED', 'AMBIGUOUS_TARGET', 'NO_COMPATIBLE_STABLE_TARGET')
    )).report()


def plan_adaptation(report, model, profiles, before, after, *, discovery_finished=False,
                    first_live_validation=False):
    from .providers import coverage_capability

    plan = _single_target_plan(report, model, profiles, before, after, discovery_finished, first_live_validation)
    plan['candidate_set'] = _candidate_set(report, model, plan, report['closed'] or discovery_finished)
    plan['coverage_capability'] = asdict(coverage_capability(report['mechanism']))
    return plan
