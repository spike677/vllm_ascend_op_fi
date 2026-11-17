"""Execution-path graphs from the existing structure and eager workload observers."""
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import inspect
import threading
import uuid

from fault_injection.coverage import KernelLaunchBoundary
from fault_injection.model_discovery import inspect_structure
from fault_injection.runtime_discovery import _python_execution_context
from fault_injection.tensor_observer import NaturalTensorObserver, callable_identity

from .native_observer import NativeScope


def worker_request_probe(runner, *, phase, expected, expected_max_seqs=1):
    """Use the reviewed scheduler readers inside the actual model pre-hook."""
    if phase not in ('prefill', 'decode'):
        raise ValueError('REVIEWED_PREFILL_OR_FIRST_DECODE_REQUIRED')

    def probe(model, args, kwargs):
        from demoncat.runtime.vllm_ascend.worker_extension.batch_context import snapshot_first_decode_rows
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import snapshot_fresh_prefill
        from vllm.forward_context import get_forward_context

        bound = inspect.signature(model.forward).bind(*args, **kwargs)
        bound.apply_defaults()
        actual = bound.arguments
        if phase == 'prefill':
            row = snapshot_fresh_prefill(runner, model, dict(actual), get_forward_context(),
                expected_prompt_token_ids=expected['prompt_token_ids'],
                expected_internal_request_id=expected['internal_request_id'])
            return dict(request_ids=[row['internal_request_id']], phase=phase,
                evidence=dict(source=row['status'], request_object_id=row['request_object_id'],
                              scheduled_tokens=row['num_scheduled_tokens'], attention_layers=row['attention_metadata_keys']))
        row = snapshot_first_decode_rows(runner, model, actual, get_forward_context(), expected,
                                         expected_max_seqs=expected_max_seqs)
        return dict(request_ids=row['request_ids'], phase=phase, evidence=dict(
            source='ACTUAL_FIRST_PURE_DECODE_SCHEDULER_ROWS', row_request_indices=row['row_request_indices'],
            row_token_positions=row['row_token_positions']))

    return probe


def _roles(modules):
    roles = defaultdict(set)
    for row in modules:
        path = row['path']
        parent = path.rpartition('.')[0] or '<root>'
        for role in row.get('logical_roles', ()):
            roles[path].add(role)
            if role.endswith('_NORM'):
                roles[path].add('NORM')
            elif role.startswith(('ATTENTION_', 'MLP_', 'MOE_')):
                roles[parent].add(role.split('_')[0])
    return roles


def normalize_execution_path(census, *, structure=None, context=None, module_entries=(),
                             native_windows=(), source_routes=None, kernel_launches=None):
    """Normalize live or saved observer evidence. No workload or native loading.

    `source_routes` maps a dispatcher name to a source-pinned native hypothesis.
    `kernel_launches` maps (dispatcher token, native source, first sequence) to
    a separate KernelLaunchBoundary supplied by an independently typed launch
    adapter. Association by this caller-supplied key remains inferred.
    """
    if census.get('schema') != 'demoncat.natural_tensor_census.v1':
        raise ValueError('NATURAL_TENSOR_CENSUS_SCHEMA_REQUIRED')
    structure = structure or {}
    context = context or {}
    module_entries = tuple(module_entries)
    native_windows = tuple(native_windows)
    source_routes = source_routes or {}
    kernel_launches = kernel_launches or {}
    events = census.get('events', ())
    if len({row['sequence'] for row in events}) != len(events):
        raise ValueError('DUPLICATE_OBSERVED_EVENT_SEQUENCE')
    modules = structure.get('modules', census.get('module_inventory', ()))
    module_calls = Counter(entry['path'] for entry in module_entries)
    observed_paths = set(module_calls) | {row['path'] for row in events if 'path' in row}
    module_by_path = {row['path']: row for row in modules}
    roles = _roles(modules)
    nodes = {'model': dict(stage='model', model_class=structure.get('model_class'), context=context)}
    edges = []

    def add_edge(left, right, status, reason):
        edges.append(dict(source=left, target=right, status=status, evidence=reason))

    for path in sorted(observed_paths | set(module_by_path)):
        layer_id = 'layer:' + path
        module = module_by_path.get(path, {})
        nodes[layer_id] = dict(stage='model_layer', path=path, logical_roles=sorted(roles[path]),
            role_status='INFERRED' if roles[path] else 'NOT_VISIBLE',
            object_identity=module.get('object_identity', module.get('module_object_id')),
            module_type=module.get('module_type'), aliases=module.get('aliases', [path]),
            call_count=module_calls[path] if module_entries else None)
        if path == '<outside>':
            add_edge('model', layer_id, 'NOT_VISIBLE', 'DISPATCH_WITHOUT_CURRENT_MODULE_BINDING')
        elif path in observed_paths:
            add_edge('model', layer_id, 'OBSERVED', 'ACTUAL_MODULE_HOOK_OR_DISPATCH_STACK')
        elif module_entries and census.get('complete'):
            add_edge('model', layer_id, 'NOT_EXECUTED', 'NO_ENTRY_IN_THE_CAPTURED_FORWARDS')
        else:
            add_edge('model', layer_id, 'NOT_VISIBLE', 'NO_EXHAUSTIVE_MODULE_ENTRY_CENSUS')

    native_by_token = {}
    scope_keys = ('process_id', 'process_start_identity', 'worker_id', 'rank', 'logical_device', 'model_instance_id')
    for window in native_windows:
        scope = window.get('scope', {})
        token = scope.get('dispatcher_token')
        if (window.get('schema') == 'demoncat.engineering.native_observation.v1' and
                token and context and all(scope.get(key) == context.get(key) for key in scope_keys)):
            if token in native_by_token:
                raise ValueError('AMBIGUOUS_NATIVE_WINDOW_FOR_DISPATCH_TOKEN')
            native_by_token[token] = window

    for row in events:
        event_id = str(row['sequence'])
        layer_id = 'layer:' + row['path']
        if row['kind'] != 'operator':
            # Retain existing view/write and module-return consumer evidence.
            if row.get('consumers'):
                producer_id = 'producer:' + event_id
                nodes[producer_id] = dict(stage='output', observation=row)
                add_edge(layer_id, producer_id, 'OBSERVED', 'OBSERVED_LIVE_OUTPUT_OR_VIEW')
                for index, consumer in enumerate(row['consumers']):
                    consumer_id = f'consumer:{event_id}:{index}'
                    nodes[consumer_id] = dict(stage='consumer', observation=consumer)
                    add_edge(producer_id, consumer_id, 'OBSERVED', 'EXISTING_TENSOR_OBSERVER_CONSUMER_MATCH')
            continue

        api = row['operator']
        operator_id, dispatcher_id = 'operator:' + event_id, 'dispatcher:' + event_id
        nodes[operator_id] = dict(stage='framework_operator', name=api, forward=row['forward'],
                                  parameters=row.get('parameters'), tensors=row.get('input'))
        namespace = row.get('namespace') or api.partition('.')[0]
        nodes[dispatcher_id] = dict(stage='torch_dispatcher', name=api, schema=row['schema'],
            namespace=namespace, dispatch_kind='ATEN' if namespace == 'aten' else 'CUSTOM_DISPATCH_NAMESPACE',
            token=row.get('dispatcher_token'), returned=row.get('original_returned') is True)
        add_edge(layer_id, operator_id, 'NOT_VISIBLE' if row['path'] == '<outside>' else 'OBSERVED',
                 'DISPATCH_WITHOUT_CURRENT_MODULE_BINDING' if row['path'] == '<outside>'
                 else 'ACTUAL_CURRENT_THREAD_DISPATCH_UNDER_MODULE')
        add_edge(operator_id, dispatcher_id, 'OBSERVED', 'TORCH_DISPATCH_MODE_ENTRY')
        output_id = 'output:' + event_id
        nodes[output_id] = dict(stage='output', tensors=row.get('output'))
        add_edge(dispatcher_id, output_id, 'OBSERVED' if row.get('original_returned') else 'NOT_VISIBLE',
                 'ACTUAL_DISPATCH_RETURN' if row.get('original_returned') else 'NO_SUCCESSFUL_RETURN_RECORDED')
        for sequence in row.get('input_producer_sequences', ()):
            producer = 'output:' + str(sequence)
            if producer not in nodes:
                raise ValueError('OBSERVER_TENSOR_PRODUCER_SEQUENCE_MISSING')
            add_edge(producer, dispatcher_id, 'OBSERVED', 'SAME_LIVE_TENSOR_OBJECT_READ_BY_DISPATCH')

        token = row.get('dispatcher_token')
        window = native_by_token.get(token, {})
        joined = [call for call in window.get('calls', ()) if call.get('dispatcher_token') == token
                  and call.get('binding_status') == 'OBSERVED']
        route = source_routes.get(api)
        digest = route.get('source_sha256', '') if route is not None else ''
        if route is not None and (not route.get('native_api') or not route.get('file') or
                not isinstance(digest, str) or len(digest) != 64 or
                any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('SOURCE_ROUTE_REQUIRES_SYMBOL_AND_PINNED_FILE')
        for index, call in enumerate(joined or [None]):
            native_id = f'native:{event_id}:{index}'
            tiling_id = f'host_tiling:{event_id}:{index}'
            kernel_id = f'device_kernel:{event_id}:{index}'
            launch = None
            if call:
                nodes[native_id] = dict(stage='native_api', source=window['source'],
                                        build_sha256=window['build_sha256'], call=call)
                add_edge(dispatcher_id, native_id, 'OBSERVED', 'SYNCHRONOUS_SAME_THREAD_TYPED_NATIVE_CALL')
                launch = kernel_launches.get((row['dispatcher_token'], window['source'], call['first_sequence']))
            else:
                nodes[native_id] = dict(stage='native_api', source=window.get('source'), source_route=route)
                if route:
                    add_edge(dispatcher_id, native_id, 'INFERRED', 'PINNED_SOURCE_ONLY')
                elif window.get('complete') and window.get('synchronous_policy') and not window.get('events'):
                    add_edge(dispatcher_id, native_id, 'NOT_EXECUTED', 'NO_EVENTS_IN_CONFIGURED_NATIVE_ABI_WINDOW')
                else:
                    add_edge(dispatcher_id, native_id, 'NOT_VISIBLE', 'NO_CALL_BOUND_NATIVE_OBSERVER')
            if launch is not None and not isinstance(launch, KernelLaunchBoundary):
                raise ValueError('TYPED_KERNEL_LAUNCH_EVIDENCE_REQUIRED')
            nodes[tiling_id] = dict(stage='host_tiling')
            nodes[kernel_id] = dict(stage='device_kernel', launch=asdict(launch) if launch else None)
            add_edge(native_id, tiling_id, 'NOT_VISIBLE', 'HOST_TILING_CALCULATION_NOT_OBSERVED')
            add_edge(tiling_id, kernel_id, 'NOT_VISIBLE', 'HOST_TO_DEVICE_CAUSAL_EDGE_NOT_OBSERVED')
            if launch:
                add_edge(native_id, kernel_id, 'INFERRED', 'SEPARATE_TYPED_LAUNCH_WITHOUT_OBSERVED_CALL_EDGE')

    return dict(schema='demoncat.engineering.execution_path.v1', context=context,
        nodes=[dict(id=node_id, **node) for node_id, node in nodes.items()], edges=edges,
        module_entries=list(module_entries), forwards=census.get('forwards', []),
        operator_counts=census.get('operator_counts', {}), native_windows=list(native_windows),
        observation_complete=census.get('complete') is True,
        native_windows_complete=all(window.get('complete') is True for window in native_windows)
        if native_windows else None,
        all_dispatch_details=census.get('all_dispatch_details') is True,
        dropped_calls=census.get('dropped_calls'), qualification_granted=False)


@contextmanager
def observe_execution_path(model, *, request_at_forward=None, worker_identity=None,
                           runner=None, known_native_observers=None, max_events=10000):
    """Open a read-only observation scope in the existing workload owner thread.

    `known_native_observers` maps precise Torch dispatcher names to an open
    NativeObserver for the already configured read-only ABI. After exit the
    yielded mapping contains the serializable path report.
    """
    if request_at_forward is not None and (not callable(request_at_forward) or not worker_identity):
        raise ValueError('LIVE_SCHEDULER_PROBE_REQUIRES_ACTUAL_WORKER_IDENTITY')
    context = _python_execution_context(model, worker_identity)
    if worker_identity:
        context.update(logical_device=worker_identity['logical_device'],
                       model_object_identity=id(model))
        if runner is None or runner.model is not model:
            raise ValueError('ACTUAL_WORKER_RUNNER_MODEL_REQUIRED')
    structure = inspect_structure(model, runner=runner)
    module_entries, requests, windows = [], [], []
    capture = {}
    known_native_observers = known_native_observers or {}
    source_cache = {}
    prefix = uuid.uuid4().hex

    def module_enter(path, module, args, kwargs):
        if module is model and request_at_forward is None:
            requests.append({})
        if len(module_entries) >= max_events:
            raise RuntimeError('MODULE_ENTRY_OBSERVER_BUDGET_EXHAUSTED')
        source = source_cache.get(id(module))
        if source is None:
            source = callable_identity(module)
            source.pop('source', None)
            source_cache[id(module)] = source
        module_entries.append(dict(path=path, forward=len(requests) - 1,
                                   ordinal=len(module_entries), callable=source))

    @contextmanager
    def operation(row):
        row['dispatcher_token'] = f"{prefix}:{row['sequence']}"
        native = known_native_observers.get(row['operator'])
        if native is None:
            yield
            return
        request = requests[row['forward']]
        scope = NativeScope(context['process_id'], context['process_start_identity'], context['worker_id'],
            context['rank'], context.get('logical_device'), tuple(request.get('request_ids', ())),
            request.get('phase'), row['dispatcher_token'], bool(request.get('evidence')),
            model_instance_id=context['model_instance_id'])
        with native.capture(scope) as window:
            windows.append(window)
            yield

    observer = NaturalTensorObserver(model, max_events=max_events, on_module_enter=module_enter,
                                     record_all_operators=True, operator_context=operation)
    handle = None
    if request_at_forward is not None:
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused

        def read_request(module, args, kwargs):
            try:
                request = request_at_forward(module, args, kwargs)
            except ContextRefused as error:
                request = {'reason': error.code}
            requests.append(request)

        # Scheduler diagnostics must happen before the model's dispatch mode.
        handle = model.register_forward_pre_hook(read_request, with_kwargs=True)
    def finish(observer):
        census = observer.report()
        census['all_dispatch_details'] = not census['dropped_calls']
        for row, request in zip(census['forwards'], requests, strict=True):
            row.update(request=request or None, request_status='OBSERVED' if request.get('evidence') else 'NOT_VISIBLE')
        capture['report'] = normalize_execution_path(census, structure=structure, context=context,
                                                    module_entries=module_entries, native_windows=windows)

    try:
        with observer:
            capture['inflight'] = lambda: bool(observer.stack or observer.mode is not None)
            yield capture
    except BaseException as error:
        capture['workload_error'] = type(error).__name__
        try:
            finish(observer)
        except (AttributeError, KeyError, TypeError, ValueError) as report_error:
            capture['report_error'] = type(report_error).__name__
        raise
    else:
        finish(observer)
    finally:
        capture.pop('inflight', None)
        if handle is not None:
            handle.remove()


def discover_execution_path(model, workload, **options):
    """Run a normal eager workload once, returning its unchanged result and graph."""
    with observe_execution_path(model, **options) as capture:
        result = workload()
    return result, capture['report']


_WORKER_CAPTURE = '_demoncat_engineering_path_capture'


def _actual_worker(worker):
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    return worker.worker if isinstance(worker, WorkerWrapperBase) else worker


def worker_path_start(worker, *, phase, expected, expected_max_seqs=1, native_observer_factory=None,
                      max_events=10000):
    """RPC entry for an existing, dedicated Worker and a normal driver request."""
    worker = _actual_worker(worker)
    if hasattr(worker, _WORKER_CAPTURE):
        raise RuntimeError('PATH_OBSERVER_ALREADY_INSTALLED')
    identity = worker.demoncat_worker_identity()
    if identity['status'] != 'WORKER_IDENTIFIED' or identity.get('quarantined'):
        raise RuntimeError('PATH_OBSERVER_REQUIRES_IDENTIFIED_WORKER')
    runner = worker.model_runner
    probe = worker_request_probe(runner, phase=phase, expected=expected, expected_max_seqs=expected_max_seqs)
    native = native_observer_factory(worker) if native_observer_factory else None
    cm = observe_execution_path(runner.model, request_at_forward=probe, worker_identity=identity['worker'],
                                runner=runner, known_native_observers=native, max_events=max_events)
    capture = cm.__enter__()
    setattr(worker, _WORKER_CAPTURE, (cm, capture, threading.get_ident()))
    return {'status': 'READ_ONLY_PATH_OBSERVER_INSTALLED'}


def worker_path_finish(worker):
    worker = _actual_worker(worker)
    cm, capture, owner = getattr(worker, _WORKER_CAPTURE)
    if threading.get_ident() != owner:
        raise RuntimeError('PATH_OBSERVER_CLOSE_REQUIRES_OWNER_THREAD')
    if capture['inflight']():
        raise RuntimeError('PATH_OBSERVER_FORWARD_STILL_INFLIGHT')
    try:
        cm.__exit__(None, None, None)
    finally:
        delattr(worker, _WORKER_CAPTURE)
    return capture['report']
