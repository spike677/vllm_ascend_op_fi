"""Read-only operator discovery and selection for the existing NativeSession."""
from dataclasses import asdict
import hashlib
import json
import os
import re
import uuid
import weakref
import threading

from .core import UnsupportedCall
from .coverage import Candidate, CandidateSet, Eligibility, EligibilityResult, TargetIdentity
from .environment import environment_fingerprint


_LOCAL_MODELS = weakref.WeakKeyDictionary()
_LOCAL_MODEL_LOCK = threading.Lock()
_PROCESS_NONCE = uuid.uuid4().hex


def _process_start_identity():
    from pathlib import Path
    try:
        raw = Path('/proc/self/stat').read_text()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        start = raw.rsplit(') ', 1)[1].split()[19]
        if int(raw.split(' ', 1)[0]) == os.getpid() and boot and start.isdecimal():
            return f'proc:{boot}:{os.getpid()}:{start}'
    except (OSError, ValueError, IndexError):
        pass
    return f'process_local:{os.getpid()}:{_PROCESS_NONCE}'


def _python_execution_context(model, worker_identity):
    """Observe a model-local replay or consume an already inspected Worker ID.

    No Worker is created. A model-local nonce never becomes a serving identity.
    Existing initialized distributed rank is read, never initialized here.
    """
    import torch
    dist = torch.distributed
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    actual_start = _process_start_identity()
    if worker_identity:
        required = ('process_start_identity', 'worker_start_identity', 'model_instance_id')
        if (not isinstance(worker_identity, dict) or worker_identity.get('pid') != os.getpid()
                or type(worker_identity.get('rank')) is not int or worker_identity['rank'] != rank
                or any(not isinstance(worker_identity.get(k), str) or not worker_identity[k] for k in required)
                or (actual_start.startswith('proc:') and worker_identity['process_start_identity'] != actual_start)):
            raise UnsupportedCall('ACTUAL_WORKER_IDENTITY_MISMATCH')
        return dict(process_id=os.getpid(), process_start_identity=worker_identity['process_start_identity'],
                    worker_id=worker_identity['worker_start_identity'], rank=rank,
                    model_instance_id=worker_identity['model_instance_id'], scope_kind='INITIALIZED_WORKER')
    with _LOCAL_MODEL_LOCK:
        context = _LOCAL_MODELS.get(model)
        if context is None:
            context = (uuid.uuid4().hex, uuid.uuid4().hex)
            _LOCAL_MODELS[model] = context
    return dict(process_id=os.getpid(), process_start_identity=actual_start,
                worker_id='model_local:' + context[0], rank=rank,
                model_instance_id=context[1], scope_kind='MODEL_LOCAL')


def _python_event_signature(event, mechanism):
    fields = ('path', 'ordinal', 'module_object_id', 'module_forward_identity', 'module_type', 'output_shape', 'dtype',
              'input_shape', 'weight_shape', 'boundary')
    row = {key: event[key] for key in fields if key in event}
    if mechanism == '15.alias_detach':
        row.update({key: event.get(key) for key in ('size', 'stride', 'offset', 'actual_format')})
        row['base'] = {key: event.get('base', {}).get(key)
                       for key in ('shape', 'stride', 'offset', 'dtype', 'device', 'actual_format')}
    return row


def _healthy_python_event(event, mechanism):
    if event.get('original_execution_verified') is not True:
        return False
    if mechanism == '15.alias_detach':
        return (event.get('status') == 'routed' and event.get('applied') is False
                and event.get('initial_values_correct') is True
                and event.get('aliases_original_base') is True
                and bool(event.get('writes')) and bool(event.get('consumers'))
                and not event.get('observation_errors')
                and all(w.get('view_write_correct') is True and w.get('base_matches_expected_update') is True
                        and w.get('non_target_unchanged') is True for w in event['writes']))
    return event.get('status') == 'baseline'


def observed_python_candidate_set(model, mechanism, *, reports, worker_identity=None, provider_version='1'):
    """Convert actual 15/17/18 healthy observations to the shared target contract.

    The mechanism adapter owns observation and its data checks. This helper joins
    those observations with live module objects, callable source and model-load
    identity. It does not authenticate arbitrary supplied reports, certify Native
    execution, or authorize a fault. Both reports must be healthy complete runs;
    callers still enforce complete input/output stability and lossless recovery.
    """
    import inspect
    import marshal
    import torch
    from .coverage import BoundaryTargetIdentity, PythonHookBoundary
    from .model_discovery import inspect_structure

    operators = {'15.alias_detach': 'torch.as_strided', '17.spatial_padding': 'torch.nn.functional.conv3d',
                 '18.api_half': 'torch.nn.RMSNorm.forward'}
    if mechanism not in operators or not isinstance(provider_version, str) or not provider_version:
        raise UnsupportedCall('PYTHON_BOUNDARY_ADAPTER_NOT_AVAILABLE')
    reports = tuple(reports)
    if len(reports) != 2 or any(not isinstance(r, dict) for r in reports):
        raise UnsupportedCall('TWO_ACTUAL_HEALTHY_OBSERVATIONS_REQUIRED')
    structure = inspect_structure(model)
    modules = {path or '<root>': module for path, module in model.named_modules()}
    roles = {row['object_identity']: tuple(sorted(row['logical_roles'])) for row in structure['modules']}
    context = _python_execution_context(model, worker_identity)
    proof = hashlib.sha256(json.dumps(reports, sort_keys=True, allow_nan=False).encode()).hexdigest()
    completed = all(r.get('arm') == 'baseline' and not r.get('cleanup_errors')
                    and not r.get('mode_cleanup_pending') and not r.get('rejected_root_calls')
                    and len(r.get('forwards', ())) == 1 and r['forwards'][0].get('status') == 'returned'
                    for r in reports)
    event_maps = []
    for report in reports:
        events = {}
        for event in report.get('events', ()):
            path, ordinal = event.get('path'), event.get('ordinal')
            if not isinstance(path, str) or type(ordinal) is not int or ordinal < 0 or (path, ordinal) in events:
                raise UnsupportedCall('ACTUAL_PYTHON_CALL_IDENTITY_MISSING_OR_DUPLICATE')
            events[(path, ordinal)] = event
        event_maps.append(events)
    keys = list(dict.fromkeys([*event_maps[0], *event_maps[1]]))
    candidates = []
    for path, ordinal in keys:
        pair = [events.get((path, ordinal)) for events in event_maps]
        module = modules.get(path)
        status, reason, target = Eligibility.UNVERIFIED, 'REPEATED_ACTUAL_CALL_REQUIRED', None
        if all(pair):
            if not all(_healthy_python_event(e, mechanism) for e in pair):
                status, reason = Eligibility.UNSUPPORTED, 'MECHANISM_HEALTHY_BOUNDARY_UNPROVEN'
            elif module is None or any(e.get('module_object_id') != id(module) for e in pair):
                reason = 'ACTUAL_MODULE_OBJECT_IDENTITY_CHANGED'
            elif any(e.get('module_forward_identity') != id(getattr(module.forward, '__func__', module.forward))
                     for e in pair):
                reason = 'ACTUAL_MODULE_CALLABLE_IDENTITY_CHANGED'
            elif _python_event_signature(pair[0], mechanism) != _python_event_signature(pair[1], mechanism):
                reason = 'ACTUAL_CALL_GEOMETRY_OR_POSITION_CHANGED'
            else:
                try:
                    # Identity follows the live callable, including a reviewed
                    # collector wrapper. Mechanism-specific admissibility of
                    # that wrapper remains with the observing Provider.
                    function = getattr(module.forward, '__func__', module.forward)
                    source = inspect.getsource(function).encode()
                    code_sha256 = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
                    callable_id = function.__module__ + '.' + function.__qualname__
                    if mechanism == '15.alias_detach':
                        device = torch.device(pair[0]['base']['device'])
                    else:
                        devices = {tensor.device for tensor in (*module.parameters(recurse=False),
                                                               *module.buffers(recurse=False))}
                        if len(devices) != 1:
                            raise ValueError('UNIQUE_LIVE_MODULE_DEVICE_REQUIRED')
                        device, = devices
                    if worker_identity and (device.type != 'npu' or device.index != worker_identity.get('logical_device')):
                        raise ValueError('ACTUAL_WORKER_DEVICE_MISMATCH')
                    signature = hashlib.sha256(json.dumps(_python_event_signature(pair[0], mechanism),
                                                          sort_keys=True, allow_nan=False).encode()).hexdigest()
                    target = BoundaryTargetIdentity(**context, device_type=device.type,
                        logical_device=device.index, model_object_identity=id(model), module_object_identity=id(module),
                        module_roles=roles.get(id(module), ()), operator_api=operators[mechanism], tensor_signature=signature,
                        boundary=PythonHookBoundary(callable_id, hashlib.sha256(source).hexdigest(), ordinal),
                        provider_identity=mechanism + ':' + provider_version,
                        provider_proof=(('callable_code_sha256', code_sha256), ('healthy_observations_sha256', proof)))
                    status, reason = Eligibility.SUPPORTED, None
                except (OSError, TypeError, AttributeError, KeyError, ValueError) as error:
                    reason = 'CALLABLE_OR_DEVICE_IDENTITY_UNPROVEN:' + type(error).__name__
        candidates.append(Candidate(f'python:{path}:{ordinal}',
            EligibilityResult(status, (reason,) if reason else ()), target))
    # For module adapters the mechanism reports all discovered modules, including
    # a branch not executed. Do not silently drop it from the candidate inventory.
    if mechanism != '15.alias_detach':
        known = {path for path, _ in keys}
        unseen = set().union(*(set(r.get('coverage', {})) for r in reports)) - known
        for path in sorted(unseen):
            candidates.append(Candidate('module:' + path, EligibilityResult(
                Eligibility.NOT_OBSERVED, ('NO_CALL_IN_HEALTHY_REPLAYS',))))
        completed = completed and not unseen
    return CandidateSet(tuple(candidates), bool(keys) and completed and set(event_maps[0]) == set(event_maps[1]))


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
