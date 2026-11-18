"""Trusted natural writer deployment; existing candidate/group code owns effects."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading

from fault_injection.deployments.writer26_experiment import Writer26Experiment, SINGLE_SITE_STAGE
from fault_injection.deployments.writer26_coverage import Writer26Coverage
from .serving26_group import Serving26GroupExperiment
from .serving_context import require, require_serial_eager


ATTRIBUTE = '_demoncat_writer26_core_bindings'
MECHANISMS = ('26.residual_stomp', '26.fia_destination')
COMPLETE_SET_STAGE = 'c2_once'
EVERY_MATCH_STAGE = 'c2_c3'
ISOLATED_STAGE = 'c2_isolated_once'


@dataclass(frozen=True)
class Writer26CoreBinding:
    worker: object
    runner: object
    model: object
    mechanism: str
    prompt: tuple
    folder: object
    manifest_sha256: str
    runtime_factory: object
    reader: object
    inspection: object
    pid: int
    thread: int
    simulation: bool = False
    stage: str = SINGLE_SITE_STAGE
    max_targets: int = 1
    leases: set = field(default_factory=set, compare=False, repr=False)


def install_writer26_core_binding(worker, *, mechanism, prompt_token_ids, evidence_dir,
        manifest_sha256, bundle='/native', stage=SINGLE_SITE_STAGE, max_targets=1,
        _simulation=False, _runtime_factory=None, _reader=None):
    """Read an already loaded deployment, without hooks, Native owner or forward."""
    from fault_injection.deployments.writer26_group import NativeWriter26Group
    from fault_injection.deployments.writer26_inventory import NativeWriterInventory
    from fault_injection.providers import register_runtime
    require(mechanism in MECHANISMS, 'EXACT_SINGLE_SITE_WRITER_MECHANISM_REQUIRED')
    require(stage in (SINGLE_SITE_STAGE, COMPLETE_SET_STAGE, EVERY_MATCH_STAGE, ISOLATED_STAGE) and type(max_targets) is int
        and 1 <= max_targets <= 128, 'BOUNDED_WRITER26_CORE_STAGE_REQUIRED')
    require(type(_simulation) is bool and (_simulation or _runtime_factory is _reader is None),
            'WRITER26_CORE_OVERRIDES_CPU_ONLY')
    require(not _simulation or callable(_runtime_factory) and _reader is not None, 'EXPLICIT_CPU_WRITER26_CORE_REQUIRED')
    bindings = getattr(worker, ATTRIBUTE, {})
    require(not bindings and not any(hasattr(worker, name) for name in (
        '_demoncat_writer26_group_experiment', '_demoncat_writer26_inventory', '_dcat_fia26_binding',
        '_dcat_qmm26_binding')), 'EXCLUSIVE_WRITER26_CORE_DEPLOYMENT_REQUIRED')
    require(isinstance(prompt_token_ids, (tuple, list)) and 2 <= len(prompt_token_ids) <= 4096
        and all(type(token) is int and token >= 0 for token in prompt_token_ids), 'BOUNDED_FROZEN_PROMPT_REQUIRED')
    require(isinstance(manifest_sha256, str) and len(manifest_sha256) == 64
        and all(c in '0123456789abcdef' for c in manifest_sha256), 'REVIEWED_WRITER26_MANIFEST_REQUIRED')
    runner, model = worker.model_runner, worker.model_runner.model
    require_serial_eager(runner, expected_max_seqs=1)
    require(not model.training, 'LOADED_EVAL_MODEL_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_WRITER26_CORE_EVIDENCE_REQUIRED')
    def runtime_factory(owner, generation):
        return NativeWriter26Group(mechanism=mechanism, bundle=bundle, manifest_sha256=manifest_sha256,
                                   owner=owner, generation=generation)
    factory = _runtime_factory if _simulation else runtime_factory
    inspection = factory('writer26-deployment-inspection', 1)
    state = inspection.state()
    require(not any(state['core'][name] for name in ('installed', 'pending', 'executing', 'device_bytes', 'host_bytes',
        'descriptor_pending')) and not any(state[name] for name in ('callback_active', 'completion_active', 'descriptor',
        'prepared_fia')), 'WRITER26_DEPLOYMENT_NOT_IDLE')
    require(mechanism != '26.fia_destination' or inspection.fia_typed is not None,
            'SINGLE_SITE_FIA_CURRENT_TYPED_STOCK_BRIDGE_REQUIRED')
    reader = _reader if _simulation else NativeWriterInventory(str(Path(bundle)/'libfault26_readonly_trace_abi0.so'),
        inspection.build['files']['libfault26_readonly_trace_abi0.so'])
    reader.state()
    binding = Writer26CoreBinding(worker, runner, model, mechanism, tuple(prompt_token_ids), folder,
        manifest_sha256, factory, reader, inspection, os.getpid(), threading.get_ident(), _simulation, stage, max_targets)
    def resolve(actual_model):
        require(actual_model is model and runner.model is model, 'WRITER26_CORE_MODEL_CHANGED')
        return binding
    register_runtime(mechanism, resolve)
    worker._demoncat_writer26_core_factory = resolve
    setattr(worker, ATTRIBUTE, {mechanism: binding})
    return dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED', mechanism=mechanism, stage=stage,
        hooks_installed=0, native_owner_installed=False, workers_created=0, qualification_granted=False, c2=False, c3=False)


def remove_writer26_core_binding(worker, mechanism):
    from fault_injection.providers import unregister_runtime
    bindings = getattr(worker, ATTRIBUTE, {})
    binding = bindings.get(mechanism)
    require(type(binding) is Writer26CoreBinding and binding.worker is worker, 'WRITER26_CORE_BINDING_REQUIRED')
    require(not binding.leases, 'WRITER26_CORE_OWNER_CLEAN_REQUIRED')
    lease = getattr(getattr(worker, '_demoncat_worker_fault_runtime', None), '_leases', {}).get(mechanism)
    require(lease is None, 'WORKER_RPC_OWNER_STILL_RETAINED')
    unregister_runtime(mechanism, worker._demoncat_writer26_core_factory)
    del bindings[mechanism]
    delattr(worker, ATTRIBUTE)
    delattr(worker, '_demoncat_writer26_core_factory')
    return dict(status='WRITER26_CORE_DEPLOYMENT_REMOVED')


class CoreWriter26Experiment(Writer26Experiment):
    """Owner checks around the unchanged candidate/Native/consumer implementation."""
    def __init__(self, *args, guard, **kwargs):
        self.guard = guard
        self.hook_callbacks = {}
        super().__init__(*args, **kwargs)

    def _thread(self):
        super()._thread()
        self.guard()

    def _callback(self, pointer):
        try:
            self._thread()
            require(self.installed, 'WRITER26_LATE_NATIVE_CALLBACK')
        except BaseException as error:
            self._record_failure(error, 'core_callback_scope')
            return 1
        return super()._callback(pointer)

    def _completed(self, pointer):
        try:
            self._thread()
            require(self.installed, 'WRITER26_LATE_NATIVE_COMPLETION')
        except BaseException as error:
            self._record_failure(error, 'core_completion_scope')
            return 1
        return super()._completed(pointer)

    def _pre(self, *args, **kwargs):
        self._thread()
        require(self.installed, 'WRITER26_LATE_MODULE_CALLBACK')
        return super()._pre(*args, **kwargs)

    def _post(self, *args, **kwargs):
        self._thread()
        require(self.installed, 'WRITER26_LATE_MODULE_CALLBACK')
        return super()._post(*args, **kwargs)

    def install(self):
        try:
            super().install()
        finally:
            for handle in self.handles:
                hooks = handle.hooks_dict_ref()
                require(hooks is not None and handle.id in hooks, 'WRITER26_HOOK_INSTALL_UNCONFIRMED')
                self.hook_callbacks[handle.id] = hooks[handle.id]

    def read_hooks(self):
        for handle in self.handles:
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self.hook_callbacks[handle.id],
                    'WRITER26_MODULE_HOOK_OWNER_CHANGED')

    def close(self):
        self._thread()
        self.read_hooks()
        config = self.runtime.config
        expected_owner = (config.owner_hi, config.owner_lo, config.generation, config.thread)
        try:
            super().close()
        except BaseException:
            # A wrapper can raise after the existing remove actually completed.
            # Only the exact lease's zero-resource readback permits skipping a
            # second Native close on explicit retry; retain the original error.
            try:
                raw = self.runtime.state()
                native = raw['core']
                owned = (native['owner_hi'], native['owner_lo'], native['generation'], native['thread']) == expected_owner
                empty = not any(native[key] for key in ('installed', 'pending', 'executing', 'device_bytes',
                    'host_bytes', 'descriptor_pending')) and not any(raw[key] for key in (
                    'callback_active', 'completion_active', 'descriptor', 'prepared_fia'))
                if self.initialized and owned and empty:
                    self.native_removed = True
            except BaseException:
                pass
            raise
        finally:
            for handle in self.handles[:]:
                hooks = handle.hooks_dict_ref()
                if hooks is not None and handle.id not in hooks:
                    self.handles.remove(handle)
                    self.hook_callbacks.pop(handle.id, None)


class CoreWriter26Coverage(CoreWriter26Experiment, Writer26Coverage):
    """The existing complete-set engine with the public Core's owner guards.

    This is the same-request cumulative multi-target experiment. Unstable or
    unresolved candidates block the entire group; max_targets is only a budget.
    """
    def __init__(self, *args, **kwargs):
        from fault_injection.deployments.writer26_inventory import digest
        super().__init__(*args, **kwargs)
        self.complete_aliases = self._actual_aliases()
        self.capacity_plan = None
        self.finalized = False
        self.implementation_sha = digest([self.implementation_sha,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            hashlib.sha256(Path(__file__).with_name('providers').joinpath('fault26_core.py').read_bytes()).hexdigest()])

    def begin(self, request_id, input_sha256):
        require(not self.failure, 'FAILED_COMPLETE_GROUP_CANNOT_RETRY_OR_SELECT_ANOTHER_SET')
        require(self._actual_aliases() == self.complete_aliases, 'COMPLETE_GROUP_ACTUAL_MODULE_ALIASES_CHANGED')
        return super().begin(request_id, input_sha256)

    def _classify_candidate(self, candidate):
        from fault_injection.deployments.writer26_inventory import digest
        status, reasons = super()._classify_candidate(candidate)
        if status != 'UNVERIFIED':
            return status, reasons
        # Native thread-local producer metadata can predate this fresh root.
        # Missing lineage alone is never an exclusion. Prove absence from the
        # complete current request and the next actual producer's sequence.
        allowed = {'ACTUAL_PUBLIC_PRODUCER_LIFETIME_UNPROVEN',
            'CURRENT_PUBLIC_RESIDUAL_LIFETIME_UNPROVEN', 'NEXT_ACTUAL_NATIVE_CONSUMER_UNPROVEN',
            'PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN'}
        if not reasons or not set(reasons).issubset(allowed):
            return status, reasons
        requests = [row for row in self.rows if any(item is candidate for item in row['candidates'])]
        if len(requests) != 1:
            return status, reasons
        row, native, callback = requests[0], candidate['native'], candidate.get('callback')
        events = row['native_events']
        before, after = row['native_before'], row['native_after']
        complete = (not before['overflow_events'] and not after['overflow_events']
            and not before['errors'] and not after['errors']
            and before['stored_events'] == before['total_events']
            and after['stored_events'] == after['total_events']
            and [item['sequence'] for item in events] == list(range(before['stored_events']+1, after['stored_events']+1))
            and all(item['thread'] == self.native_thread for item in events))
        earlier = [item for item in events if item['kind_name'].startswith('add_rms_')
                   and item['sequence'] < native['sequence']]
        later = [item for item in events if item['kind_name'] == 'add_rms_exit'
                 and item['sequence'] > native['sequence']]
        if (not complete or not callback or callback.get('actual_request_id') != row['request_id']
                or callback.get('original_continuations') != 1 or callback.get('claimed') != 0
                or earlier or not later or later[0]['status'] != 0
                or later[0]['producer'] != native['producer']+1
                or len(candidate['execution']) != 1 or candidate['execution'][0]['status'] != 0):
            return status, reasons
        candidate['current_forward_residual_absence'] = dict(
            schema='demoncat.writer26.current_forward_residual_absence.v1',
            request_id=row['request_id'], complete_native_events_sha256=digest(events),
            native_before=before, native_after=after, writer_sequence=native['sequence'],
            inherited_producer=native['producer'], next_actual_producer=later[0],
            original_unresolved_reasons=reasons, earlier_current_forward_producer_count=0,
            qualification_granted=False)
        return 'UNSUPPORTED', ['CURRENT_FRESH_FORWARD_HAS_NO_RESIDUAL_PRODUCER']

    def _freeze(self):
        super()._freeze()
        if self.blockers:
            return
        self._prepare_complete_plan()
        if self.blockers:
            return
        row = self.rows[1]
        remaining = len(self.plan)-2
        events = row['native_after']['total_events']-row['native_before']['total_events']
        self.capacity_plan = dict(remaining_complete_requests=remaining, planned_target_count=len(self.targets),
            full_native_writer_count=len(self.candidate_inventory), full_request_native_events=events,
            estimated_native_events=events*remaining,
            available_native_events=row['native_after']['capacity']-row['native_after']['stored_events'],
            native_capacity=row['native_after']['capacity'], max_targets_is_selector=False)
        if events*remaining > row['native_after']['capacity']-row['native_after']['stored_events']:
            self.blockers.append('COMPLETE_GROUP_PLAN_EXCEEDS_NATIVE_EVENT_CAPACITY')
            self.plan = self.plan[:2]+[dict(phase='health', mode=None, target=None)]
            return
        if self.fia_typed is not None:
            before, after = row['fia_typed_before'], row['fia_typed_after']
            typed = after['stored']-before['stored']
            owned = self._planned_owned_records(remaining)
            self.capacity_plan.update(full_request_typed_events=typed, additional_owned_execute_records=owned,
                estimated_typed_events=typed*remaining+owned,
                available_typed_events=after['capacity']-after['stored'], typed_capacity=after['capacity'])
            if typed*remaining+owned > after['capacity']-after['stored']:
                self.blockers.append('COMPLETE_GROUP_PLAN_EXCEEDS_TYPED_FIA_EVENT_CAPACITY')
                self.plan = self.plan[:2]+[dict(phase='health', mode=None, target=None)]

    def _prepare_complete_plan(self):
        """A concrete execution profile may expand requests before budgeting."""

    def _planned_owned_records(self, remaining):
        return len(self.targets)*(remaining-1)

    def _requested_mode(self, row):
        from fault_injection.deployments.writer26_experiment import _STABLE_NATIVE
        key = row['target_key']
        if key in self.targets and self.entry['mode'] is not None:
            frozen = self.targets[key]['second_native']
            require(self._actual_aliases() == self.complete_aliases
                and all(row['independent_event'][name] == frozen[name] for name in _STABLE_NATIVE),
                'COMPLETE_GROUP_CURRENT_OBJECT_OR_NATIVE_FORMAL_CHANGED')
            row['complete_group_current_binding_verified'] = True
        return super()._requested_mode(row)

    def _request_complete(self, row):
        from fault_injection.deployments.writer26_inventory import digest
        require(self._actual_aliases() == self.complete_aliases, 'COMPLETE_GROUP_ACTUAL_MODULE_ALIASES_CHANGED')
        for candidate in row['candidates']:
            self._classify_candidate(candidate)
        super()._request_complete(row)
        row['complete_inventory'] = dict(native_writer_count=len(row['candidates']),
            native_events_sha256=digest(row['native_events']), candidates_sha256=digest(row['candidates']),
            actual_aliases_sha256=digest(self.complete_aliases))
        if row['phase'] == 'health':
            require(len(row['candidates']) == len(self.candidate_inventory),
                    'HEALTH_MUST_KEEP_COMPLETE_NATIVE_WRITER_DENOMINATOR')

    def finalize(self):
        if not self.finalized:
            super().finalize()
            self.finalized = True

    def report(self, *, compact_rpc=False, observation_start=None):
        value = super().report(compact_rpc=compact_rpc, observation_start=observation_start)
        aliases = self.complete_aliases
        if compact_rpc:
            from fault_injection.deployments.writer26_inventory import digest
            aliases = dict(count=len(aliases), original_sha256=digest(aliases), compact_rpc_projection=True)
        value.update(alias_inventory=aliases, event_capacity_plan=self.capacity_plan,
            execution_scope='SAME_REQUEST_CUMULATIVE_ALL_STABLE_ELIGIBLE_TARGETS',
            independent_per_target_replay=False, finalized=self.finalized)
        value['c2'] = bool(value['c2'] and self.finalized and self.proof is not None)
        value['c3'] = bool(value['c3'] and self.finalized)
        return value


class Serving26CoreSession(Serving26GroupExperiment):
    def __init__(self, provider):
        self.provider = provider
        binding = provider.binding
        super().__init__(provider.worker, provider.model, provider.backend, provider.native, provider._ticket[0],
            prompt_token_ids=list(binding.prompt), evidence_dir=binding.folder/provider._ticket[0]/(
                'runtime-c3' if provider.current_run == 'c3' else 'runtime'))
        self.hook_callbacks = {}
        self.root_running = self.logits_running = False
        self.evidence_files = []
        self.evidence_attempts = 0
        self.last_evidence_state = None

    def _same_worker(self):
        super()._same_worker()
        self.provider._context()

    def _last(self, module):
        for hooks in (module._forward_pre_hooks, module._forward_hooks):
            owned = [handle.id for handle in self.handles if handle.hooks_dict_ref() is hooks]
            require(len(owned) == 1 and list(hooks)[-1:] == owned, 'WRITER26_FINAL_BOUNDARY_HOOK_REQUIRED')

    def install(self):
        self._same_worker()
        require(not self.installed and not self.handles, 'SERVING_HOOK_ALREADY_INSTALLED')
        self.backend.install()
        for module, callback, pre in ((self.model, self._root_pre, True), (self.model, self._root_post, False),
                (self.processor, self._logits_pre, True), (self.processor, self._logits_post, False)):
            register = module.register_forward_pre_hook if pre else module.register_forward_hook
            handle = register(callback, with_kwargs=True, **({} if pre else dict(always_call=True)))
            self.handles.append(handle)
            self.hook_callbacks[handle.id] = callback
        self.installed = True
        self.backend.root_hooks = len(self.handles)

    def expect(self, request_id):
        self._same_worker()
        require(not self.needs_collect, 'WRITER26_EXPLICIT_COLLECT_REQUIRED')
        return super().expect(request_id)

    def _root_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            self._last(module)
            self._last(self.processor)
            require(not self.root_running and not self.logits_running, 'WRITER26_CONCURRENT_FORWARD')
            self.root_running = True
            return super()._root_pre(module, args, kwargs)
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        try:
            self._same_worker()
            self._last(module)
            return super()._root_post(module, args, kwargs, output)
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            if threading.get_ident() == self.provider.thread:
                self.root_running = False

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            self._last(module)
            self.logits_running = True
            return super()._logits_pre(module, args, kwargs)
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        try:
            self._same_worker()
            self._last(module)
            return super()._logits_post(module, args, kwargs, output)
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            if threading.get_ident() == self.provider.thread:
                self.logits_running = False

    def collect(self, *, force=False):
        self._same_worker()
        require(not self.root_running and not self.logits_running, 'WRITER26_FORWARD_STILL_ACTIVE')
        pending = self.needs_collect
        result = super().collect(force=force)
        if pending:
            self.capture_evidence('collect')
        return result

    def _evidence_state(self):
        return (len(self.backend.rows), self.backend.phase, self.backend.failure, self.backend.installed,
                len(self.handles), len(self.backend.handles), self.backend.recovered)

    def capture_evidence(self, action):
        """Explicit action only: keep independent complete original observations."""
        self._same_worker()
        folder = self.folder/'core-evidence'
        folder.mkdir(exist_ok=True)
        path = folder/(str(self.evidence_attempts).zfill(3)+'-'+action+'.json')
        self.evidence_attempts += 1  # Keep partial failed files; never reuse their name.
        value = dict(schema='demoncat.writer26.core_observations.v1', action=action,
            worker=self.provider.identity, native_owner=self.provider._ticket[0], generation=self.provider._ticket[1],
            rpc_generation=self.provider._ticket[1], native_generation=self.backend.runtime.config.generation,
            run=self.provider.current_run,
            simulation=self.provider.binding.simulation, qualification_granted=False,
            inventory=self.backend.report(), serving_requests=self.rows, serving=self.readback())
        data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
        with path.open('xb') as stream:
            stream.write(data)
        self.evidence_files.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            action=action, complete_requests=len(self.backend.rows)))
        self.last_evidence_state = self._evidence_state()

    def evidence_readback(self):
        """No collect/save; verify the already captured original bytes."""
        if not self.evidence_files:
            return dict(status='NOT_COLLECTED', originals=[])
        for record in self.evidence_files:
            data = Path(record['path']).read_bytes()
            require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                    'WRITER26_CORE_ORIGINAL_EVIDENCE_CHANGED')
        current = (not self.needs_collect and self.current is None and self.expected is None
                   and self.last_evidence_state == self._evidence_state())
        return dict(status='CURRENT_COLLECTED' if current else 'PREVIOUS_COLLECTION_CURRENT_STATE_NOT_CAPTURED',
                    originals=list(self.evidence_files))

    def readback(self):
        self._same_worker()
        self.backend.read_hooks()
        for handle in self.handles:
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self.hook_callbacks[handle.id],
                    'WRITER26_ROOT_LOGITS_HOOK_OWNER_CHANGED')
        return dict(installed=self.installed, owned_hooks=len(self.handles), needs_collect=self.needs_collect,
            expected=self.expected, current=self.current, root_running=self.root_running,
            logits_running=self.logits_running, complete_requests=len(self.rows))

    def close(self):
        self._same_worker()
        require(not self.root_running and not self.logits_running, 'WRITER26_FORWARD_STILL_ACTIVE')
        self.collect(force=True)
        self.backend.close()
        for handle in reversed(self.handles[:]):
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self.hook_callbacks[handle.id],
                    'WRITER26_ROOT_LOGITS_HOOK_OWNER_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:
                    self.handles.remove(handle)
                    self.hook_callbacks.pop(handle.id, None)
                    self.backend.root_hooks = len(self.handles)
            require(handle.id not in hooks, 'WRITER26_ROOT_LOGITS_REMOVE_UNCONFIRMED')
        self.installed = False
        self.expected = self.current = None
        if isinstance(self.backend, CoreWriter26Coverage):
            if self.backend.phase == 'complete' and not self.backend.failure:
                self.backend.finalize()
            if self.backend.coverage is not None:
                recovery = self.backend.coverage.recovery_report()
                require(recovery['complete'], 'WRITER26_CORE_GROUP_RECOVERY_REQUIRED')
        self.backend.save()
        self._save()
        if self.last_evidence_state != self._evidence_state():
            self.capture_evidence('cleanup')

    def report(self):
        # Unlike the legacy experiment report, this cannot collect or stop.
        return self.readback()
