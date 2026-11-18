"""Experimental statistics observer for the unchanged ActualTensorRequest.

This is not a provider, a Runtime extension or a qualification authority.
The exact collector's recursive registration omits the root module. Root hook
inventory must remain identical; no root-return shim or hook reorder is used.
"""
from contextlib import contextmanager
import hashlib
import json
import marshal
from pathlib import Path
import sys
import threading
import types

from experiments.c2c3_closure_20261117.tensor_statistics_capture import TensorStatisticsCapture
from experiments.worker_core_v11_20261118.collector_packet_compat import CustomPacketMetadata
from demoncat.runtime.vllm_ascend.worker_extension import serving_tensor_request as request_module
from demoncat.runtime.vllm_ascend.worker_extension.call_binding import original_before_collector, same_callable


PHASES = {'baseline': 'baseline', 'baseline_repeat': 'baseline_repeat', 'c2_fault': 'fault'}
HOOK_MAPS = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
FLAGS = ('_forward_pre_hooks_with_kwargs', '_forward_hooks_with_kwargs', '_forward_hooks_always_called')
ATTRS = ('forward', 'msprobe_hook', 'msprobe_forward_hook')
RUNTIME_FILES = ('src/demoncat/runtime/vllm_ascend/worker_extension/serving_tensor_request.py',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/serving_tensor.py',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_serving.py')
EXTRA_COLLECTOR_PINS = {
    'core/common/utils.py': '775f7aa605b98c54d338a6cfda18ea5f9d423dcca10e8de42e751e116f6db234',
    'pytorch/common/utils.py': '0ee19c3d063e32e90aae5730f33379ebc743d66237d1353534291a30fc5e2492',
    'pytorch/dump/pytorch_service.py': '12799d280f17c82cefeb8463e272376828e2239e6e057dfd4ca8e4cf62ce9036'}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from codes(value)


def callable_identity(callback):
    func = getattr(callback, '__func__', callback)
    code = getattr(func, '__code__', None)
    return dict(object_id=id(callback), function_id=id(func), owner_id=id(getattr(callback, '__self__', None)),
        name=getattr(func, '__qualname__', type(func).__qualname__),
        source=str(code.co_filename) if code else None,
        code_sha256=hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None)


class PinnedModuleHooks:
    """Authenticate the real ModuleProcessor hook and its synchronized wrapper."""
    def __init__(self, package, pins):
        self.package, self.pins = Path(package).resolve(), pins
        self.processor = None
        self.checked = {}

    def _source(self, function):
        function = getattr(function, '__func__', function)
        require(type(function) is types.FunctionType, 'COLLECTOR_PYTHON_HOOK_REQUIRED')
        path = Path(function.__code__.co_filename).resolve()
        require(path.is_relative_to(self.package), 'FOREIGN_COLLECTOR_HOOK_SOURCE')
        relative = path.relative_to(self.package).as_posix()
        expected = self.pins.get(relative)
        require(expected and sha(path) == expected, 'COLLECTOR_HOOK_SOURCE_BYTES_CHANGED')
        if path not in self.checked:
            self.checked[path] = tuple(codes(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)))
        require(function.__code__ in self.checked[path], 'COLLECTOR_HOOK_EXECUTED_CODE_CHANGED')
        module = sys.modules.get(function.__globals__.get('__name__'))
        require(module is not None and function.__globals__ is vars(module)
                and Path(module.__file__).resolve() == path, 'COLLECTOR_HOOK_GLOBALS_CHANGED')
        return relative

    def verify(self, callback, kind):
        function = callback
        chain = []
        for _ in range(4):
            relative = self._source(function)
            chain.append(callable_identity(function))
            if relative == 'pytorch/dump/module_dump/module_processor.py':
                break
            require(relative == 'core/common/utils.py', 'UNREVIEWED_COLLECTOR_DECORATOR')
            module = sys.modules[function.__globals__['__name__']]
            factory = module.ThreadSafe.synchronized
            self._source(factory)
            require(function.__code__ in tuple(codes(factory.__code__))[1:],
                    'UNREVIEWED_SYNCHRONIZED_WRAPPER_CODE')
            cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
            require('cls' in cells and cells['cls'].cell_contents is module.ThreadSafe,
                    'COLLECTOR_SYNCHRONIZED_CLASS_OWNER_CHANGED')
            wrapped = getattr(function, '__wrapped__', None)
            if wrapped is None:
                wrapped = cells.get('func').cell_contents if 'func' in cells else None
            require(type(wrapped) is types.FunctionType
                    and any(cell.cell_contents is wrapped for cell in cells.values()),
                    'COLLECTOR_DECORATOR_ORIGINAL_NOT_BOUND')
            function = wrapped
        else:
            raise RuntimeError('COLLECTOR_DECORATOR_DEPTH_EXCEEDED')
        module = sys.modules[function.__globals__['__name__']]
        factory = module.ModuleProcessor.build_module_hook
        self._source(factory)
        expected = [code for code in codes(factory.__code__) if code.co_name == kind]
        require(len(expected) == 1 and function.__code__ == expected[0], 'UNKNOWN_COLLECTOR_HOOK_CODE')
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        processor = cells.get('self').cell_contents if 'self' in cells else None
        require(type(processor) is module.ModuleProcessor, 'COLLECTOR_PROCESSOR_OWNER_CHANGED')
        if self.processor is None:
            self.processor = processor
        require(processor is self.processor, 'FOREIGN_COLLECTOR_PROCESSOR')
        require('module_name' in cells and type(cells['module_name'].cell_contents) is str,
                'COLLECTOR_MODULE_PREFIX_MISSING')
        return dict(kind=kind, source_chain=chain, module_prefix=cells['module_name'].cell_contents,
                    processor_id=id(processor))


def inventory(model):
    return {name or '<root>': dict(module=module,
        hooks={key: dict(getattr(module, key, {})) for key in HOOK_MAPS},
        flags={key: dict(getattr(module, key, {})) for key in FLAGS},
        attrs={key: (key in vars(module), vars(module).get(key)) for key in ATTRS},
        original_forward=original_before_collector(module)) for name, module in model.named_modules()}


def public_inventory(snapshot):
    return {name: dict(module_id=id(row['module']),
        hooks={kind: [dict(handle=key, **callable_identity(callback)) for key, callback in entries.items()]
               for kind, entries in row['hooks'].items()}, flags=row['flags'],
        forward=callable_identity(row['module'].forward),
        original_forward=callable_identity(row['original_forward']),
        present_attributes=[key for key, value in row['attrs'].items() if value[0]])
        for name, row in snapshot.items()}


class PublicTensorStatisticsSidecar:
    def __init__(self, session, folder, *, source_pins, _capture=None, _hook_verifier=None):
        require(type(session) is request_module.ActualTensorRequest and session.provider.mechanism == '18.api_half',
                'EXACT_PUBLIC_RMS_REQUEST_REQUIRED')
        require(not session.seen and session.expected is None and session.deferred_request is None
                and session.statistics_capture is None and not session.root_active,
                'FRESH_IDLE_PUBLIC_SESSION_REQUIRED')
        require(sys.getprofile() is None, 'FOREIGN_PROFILE_OBSERVER_PRESENT')
        require((_capture is None and _hook_verifier is None) or session.provider.simulation,
                'FAKE_COLLECTOR_IS_CPU_CONTRACT_ONLY')
        self.session, self.folder, self.thread = session, Path(folder), threading.get_ident()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.source_pins = dict(source_pins)
        self._verify_runtime()
        self.capture = _capture or TensorStatisticsCapture(self.folder/'statistics', session.provider.synchronize)
        self.production_collector = _capture is None
        self.packet_metadata = None
        if _hook_verifier is None:
            identity_path = self.capture.folder/'collector_identity.json'
            identity = json.loads(identity_path.read_text())
            require(all(identity['files'].get(name) == expected for name, expected in EXTRA_COLLECTOR_PINS.items()),
                    'UNREVIEWED_COLLECTOR_HOOK_CONTROL_SOURCE')
            self.verifier = PinnedModuleHooks(identity['package'], identity['files'])
            self.packet_metadata = CustomPacketMetadata(self.capture.collector, identity['package'], identity['files'])
            identity['custom_packet_metadata_compatibility'] = self.packet_metadata.report()
            identity['experimental_public_core_adapter'] = dict(source_sha256=sha(__file__),
                runtime_source_pins=self.source_pins, root_hook_adapted=False,
                native_root_module_record_absent=True,
                hook_order_changed=False, model_operator_filters=False, sampler_captured=False,
                owned_hook_teardown_and_same_model_cache_rearm=True, qualification_granted=False)
            identity_path.write_text(json.dumps(identity, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        else:
            self.verifier = _hook_verifier
        self.service = self.capture.collector.service
        if self.production_collector:
            require(self.verifier._source(self.service.start) == 'core/dump/service.py'
                    and self.verifier._source(self.service._register_module_hook) == 'pytorch/dump/pytorch_service.py',
                    'UNREVIEWED_COLLECTOR_REGISTRATION_BOUNDARY')
        require(type(self.service.hooked_modules) is list and self.service.hooked_modules == [],
                'FRESH_UNOWNED_COLLECTOR_MODULE_CACHE_REQUIRED')
        self.before = None
        self.error = None
        self.events = []
        self.rows = []
        self.installed = False
        self.profile_callback = self._profile
        self.begin_code = request_module._ActualRequest._begin_request.__code__
        self.begin_function = request_module._ActualRequest._begin_request
        self.save()  # Validate receipt output before installing process-local state.
        self.session.statistics_capture = self
        try:
            sys.setprofile(self.profile_callback)
        except BaseException:
            self.session.statistics_capture = None
            raise
        self.installed = True

    @property
    def active(self):
        return self.capture.active

    def _verify_runtime(self):
        require(threading.get_ident() == self.thread, 'STATISTICS_SIDECAR_THREAD_CHANGED')
        for name, expected in self.source_pins.items():
            require(sha(name) == expected, 'STATISTICS_RUNTIME_SOURCE_CHANGED')
        require(len(self.source_pins) == len(RUNTIME_FILES) and all(
            sum(Path(name).as_posix().endswith(relative) for name in self.source_pins) == 1
            for relative in RUNTIME_FILES), 'EXACT_PUBLIC_RUNTIME_SOURCE_SET_REQUIRED')
        path = Path(request_module.__file__).resolve()
        require(str(path) in self.source_pins, 'ACTUAL_REQUEST_SOURCE_NOT_PINNED')
        compiled = tuple(codes(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True)))
        for name in ('_begin_request', '_pre', '_post', '_collect_returned_request', 'expect'):
            expected = [code for code in compiled if code.co_name == name]
            function = getattr(request_module._ActualRequest, name)
            require(len(expected) == 1 and function.__code__ == expected[0]
                    and function.__globals__ is vars(request_module), 'ACTUAL_BEGIN_REQUEST_EXECUTED_CODE_CHANGED')

    def _profile(self, frame, event, returned):
        if event != 'return' or frame.f_code is not self.begin_code or frame.f_locals.get('self') is not self.session:
            return
        phase = frame.f_locals.get('phase')
        if phase not in PHASES:
            return
        try:
            self._verify_runtime()
            require(sys.getprofile() is self.profile_callback and self.session.statistics_capture is self,
                    'STATISTICS_SIDECAR_OWNER_CHANGED')
            require(request_module._ActualRequest._begin_request is self.begin_function and returned is None
                    and self.session.root_active and self.session.root_post_handle is not None
                    and self.session.expected == frame.f_locals['request_id'] and self.session.phase == phase,
                    'ACTUAL_BEGIN_REQUEST_RETURN_NOT_CONFIRMED')
            require([r['phase'] for r in self.rows] == list(PHASES)[:len(self.rows)]
                    and phase == list(PHASES)[len(self.rows)], 'STATISTICS_PHASE_SEQUENCE_CHANGED')
            self.before = inventory(self.session.model)
            self.rows.append(dict(phase=phase, arm=PHASES[phase], actual_step=len(self.rows),
                request_id=self.session.expected, before=public_inventory(self.before), completed=False))
            self.capture.begin(self.session.model, 'statistics_'+PHASES[phase])
            if self.packet_metadata is not None:
                self.packet_metadata.verify()
            require(self.capture.collector.service is self.service
                    and len(self.service.hooked_modules) == 1 and self.service.hooked_modules[0] is self.session.model,
                    'COLLECTOR_MODULE_CACHE_OWNER_CHANGED')
            self._validate_hooks()
            if self.production_collector:
                require(self.verifier.processor is self.service.module_processor,
                        'COLLECTOR_LIVE_SERVICE_PROCESSOR_CHANGED')
            self.rows[-1]['root_hooks_unchanged'] = True
            self.rows[-1]['after_start'] = public_inventory(inventory(self.session.model))
        except BaseException as error:
            self.error = type(error).__name__ + ': ' + str(error)
            self._failed_start()
            raise

    def _validate_hooks(self):
        require(self.before is not None, 'STATISTICS_HOOK_SNAPSHOT_MISSING')
        now = inventory(self.session.model)
        require(set(now) == set(self.before), 'STATISTICS_MODEL_MODULE_SET_CHANGED')
        for name, old in self.before.items():
            current = now[name]
            require(current['module'] is old['module'] and same_callable(current['original_forward'], old['original_forward']),
                    'STATISTICS_MODEL_OR_FORWARD_CHANGED')
            if name == '<root>':
                require(current['hooks'] == old['hooks'] and current['flags'] == old['flags'],
                        'COLLECTOR_ROOT_HOOK_INVENTORY_CHANGED')
            for kind, entries in old['hooks'].items():
                actual = current['hooks'][kind]
                require([key for key in actual if key in entries] == list(entries)
                        and all(actual.get(key) is callback for key, callback in entries.items()),
                        'PREEXISTING_HOOK_REMOVED_CHANGED_OR_REORDERED')
                for key, callback in actual.items():
                    if key in entries:
                        continue
                    require(kind in ('_forward_pre_hooks', '_forward_hooks'), 'UNKNOWN_COLLECTOR_HOOK_KIND')
                    self.verifier.verify(callback, 'forward_pre_hook' if kind == '_forward_pre_hooks' else 'forward_hook')
            for kind, entries in old['flags'].items():
                require(all(current['flags'][kind].get(key) == value for key, value in entries.items()),
                        'PREEXISTING_HOOK_FLAGS_CHANGED')
                hook_kind = '_forward_pre_hooks' if kind == '_forward_pre_hooks_with_kwargs' else '_forward_hooks'
                require(all(key in current['hooks'][hook_kind] and key not in old['hooks'][hook_kind]
                            and value is True for key, value in current['flags'][kind].items() if key not in entries),
                        'FOREIGN_OR_ORPHAN_COLLECTOR_HOOK_FLAG')
            for attr in ('msprobe_hook', 'msprobe_forward_hook'):
                before_present, before_value = old['attrs'][attr]
                present, value = current['attrs'][attr]
                require((present == before_present and value is before_value)
                        or (not before_present and present and value is True), 'COLLECTOR_ATTRIBUTE_OWNER_CHANGED')
        return now

    def _restore_hooks(self):
        now = self._validate_hooks()
        self.rows[-1]['before_restore'] = public_inventory(now)
        for name, old in self.before.items():
            module = old['module']
            for kind, entries in old['hooks'].items():
                mapping = getattr(module, kind)
                for key in list(mapping):
                    if key not in entries:
                        del mapping[key]
            for kind, entries in old['flags'].items():
                mapping = getattr(module, kind)
                for key in list(mapping):
                    if key not in entries:
                        del mapping[key]
            for attr, (present, value) in old['attrs'].items():
                if present:
                    setattr(module, attr, value)
                elif attr in vars(module):
                    delattr(module, attr)
        restored = inventory(self.session.model)
        require(all(restored[name]['hooks'] == old['hooks'] and restored[name]['flags'] == old['flags']
                    and restored[name]['attrs'] == old['attrs'] for name, old in self.before.items()),
                'COLLECTOR_OWNED_STATE_RESTORE_NOT_EXACT')
        self.rows[-1]['after_restore'] = public_inventory(restored)
        self.rows[-1]['owned_hook_restore_exact'] = True
        require(self.capture.collector.service is self.service and type(self.service.hooked_modules) is list
                and len(self.service.hooked_modules) == 1 and self.service.hooked_modules[0] is self.session.model,
                'COLLECTOR_MODULE_CACHE_OWNER_CHANGED')
        self.service.hooked_modules.pop()  # Only the exact stopped, fully detached owned model.
        self.rows[-1]['same_model_cache_rearmed_after_stop'] = True
        self.before = None

    @contextmanager
    def diagnostics(self):
        with self.capture.diagnostics():
            yield

    def end(self, *, completed):
        try:
            self.capture.end(completed=completed)
            self._restore_hooks()
            self.rows[-1]['completed'] = bool(completed)
        except BaseException as error:
            self.error = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            if not completed or self.error or len(self.rows) == len(PHASES):
                self._release()
            self.save()

    def _failed_start(self):
        try:
            if self.active:
                self.capture.end(completed=False)
            if self.before is not None:
                self._restore_hooks()
        except BaseException as cleanup:
            self.events.append(dict(cleanup_error=type(cleanup).__name__+': '+str(cleanup)))
        finally:
            self._release()
            self.save()

    def _release(self):
        require(sys.getprofile() in (None, self.profile_callback), 'FOREIGN_PROFILE_REPLACED_OWNED_OBSERVER')
        if sys.getprofile() is self.profile_callback:
            sys.setprofile(None)
        if self.session.statistics_capture is self and self.before is None and not self.active:
            self.session.statistics_capture = None
        self.installed = False

    def close(self):
        require(not self.session.root_active and not self.session.callbacks_active, 'ACTUAL_MODEL_REQUEST_ACTIVE')
        require(self.session.completion_handle is None, 'CORE_OWNED_COMPLETION_REQUIRES_CORE_RECOVERY')
        if self.active:
            self.end(completed=False)
        elif self.before is not None:
            self._restore_hooks()
        self._release()
        self.save()
        return self.report()

    def report(self):
        return dict(schema='demoncat.public_tensor_statistics_sidecar.v1', experiment_only=True,
            mechanism='18.api_half', source_pins=self.source_pins, sidecar_sha256=sha(__file__),
            actual_begin_function=callable_identity(self.begin_function),
            profile_observer=callable_identity(self.profile_callback),
            runtime_code_modified=False, hook_order_changed=False, root_hook_adapted=False,
            native_root_module_record_absent=True,
            capture_boundary='actual root entry through actual logits processor return; sampler excluded',
            sampler_captured=False, model_operator_filters=False, raw_statistics_rewritten=False,
            global_api_wrappers_restored=False, disposable_worker_required=True,
            phase_steps=[{key: row[key] for key in ('phase', 'arm', 'actual_step', 'request_id', 'completed')} for row in self.rows],
            windows=self.rows, events=self.events, error=self.error, profile_owned=sys.getprofile() is self.profile_callback,
            capture_active=self.active, hook_snapshot_retained=self.before is not None,
            session_capture_attached=self.session.statistics_capture is self,
            public_qualification_granted=False, c2=False, c3=False, e5=False)

    def save(self):
        (self.folder/'collector_sidecar.json').write_text(json.dumps(self.report(), indent=2, allow_nan=False)+'\n', encoding='utf-8')
