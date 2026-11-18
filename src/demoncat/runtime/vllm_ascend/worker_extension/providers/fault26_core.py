"""Worker RPC ownership bridge for the existing natural writer26 single site."""
from copy import deepcopy
import os
import threading

from ._scope import Scope, reject_unknown
from .. import identity as identities
from ..serving_context import require
from ..serving26_core import (ATTRIBUTE, COMPLETE_SET_STAGE, EVERY_MATCH_STAGE, ISOLATED_STAGE, ISOLATED_C3_STAGE, Writer26CoreBinding, CoreWriter26Experiment,
                            CoreWriter26Coverage, Serving26CoreSession)


class Fault26CoreProvider(Scope):
    execution_boundary = 'native'

    def __init__(self, worker, model, identity, binding):
        super().__init__(worker, model, identity)
        require(type(binding) is Writer26CoreBinding, 'REVIEWED_WRITER26_CORE_BINDING_REQUIRED')
        self.binding, self.runner, self.thread = binding, worker.model_runner, threading.get_ident()
        self.native = self.backend = self.session = None
        self.reserved = self.checked = self.attempted = False
        self.stopped = True
        self.cleanup_requested = False
        self.cleanup_failures = []
        self.native_scope = None
        self.current_run = 'c2' if binding.stage in (COMPLETE_SET_STAGE, EVERY_MATCH_STAGE, ISOLATED_STAGE, ISOLATED_C3_STAGE) else 'c1'
        self.c2_proof = self.c2_evidence = self.c2_session = None
        self.c3_started = False
        self.transition_failures = []
        self._context()
        self._native_state()

    def _context(self):
        value, binding = self.identity, self.binding
        device = getattr(self.worker, 'device', None)
        visible, source = identities._visible_devices()
        require(os.getpid() == value['pid'] and threading.get_ident() == self.thread
            and self.worker.model_runner is self.runner and self.runner.model is self.model
            and self.worker.rank == value['rank'] and getattr(device, 'type', None) == 'npu'
            and getattr(device, 'index', None) == value['configured_logical_device']
            and identities._current_device() == value['logical_device']
            and value['observed_logical_device'] == value['logical_device']
            and visible == value['visible_devices'] and source == value['visible_devices_source'],
            'WRITER26_HELD_WORKER_CONTEXT_CHANGED')
        require(getattr(self.worker, ATTRIBUTE, {}).get(binding.mechanism) is binding
            and binding.worker is self.worker and binding.runner is self.runner and binding.model is self.model
            and binding.pid == os.getpid() and binding.thread == self.thread, 'WRITER26_CORE_BINDING_CHANGED')
        if self.reserved or binding.leases:
            require(self.reserved and binding.leases == {self._ticket}, 'WRITER26_CORE_DEPLOYMENT_OWNER_CHANGED')
        if self.native is not None:
            config = self.native.config
            require((self.native.owner, config.generation, config.owner_hi, config.owner_lo, config.thread) == self.native_scope
                and self.native.owner == self._ticket[0]
                and config.generation == self._ticket[1]+int(self.current_run == 'c3')
                and self.native.mechanism == binding.mechanism, 'WRITER26_CORE_NATIVE_OWNER_CHANGED')

    def _native_state(self):
        native = self.native if self.native is not None else self.binding.inspection
        raw = native.state()
        core = raw['core']
        if core['installed'] or self.backend is not None and self.backend.initialized:
            require(self.native is not None and
                (core['owner_hi'], core['owner_lo'], core['generation'], core['thread']) ==
                    (self.native_scope[2], self.native_scope[3], self.native_scope[1], self.native_scope[4]),
                'WRITER26_CORE_NATIVE_READBACK_OWNER_CHANGED')
        return raw

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        reject_unknown(settings, ('request_id', 'collect') if action == 'advance' else ())
        self._context()
        self._native_state()
        if action == 'prepare':
            require(not self.binding.leases, 'WRITER26_CORE_DEPLOYMENT_ALREADY_OWNED')
        elif action == 'register':
            require(not self.reserved and not self.binding.leases, 'WRITER26_CORE_DEPLOYMENT_ALREADY_OWNED')
            self.binding.leases.add(self._ticket)
            self.reserved = self._registered = True
            native = self.binding.runtime_factory(owner, generation)
            config = native.config
            self.native_scope = (owner, generation, config.owner_hi, config.owner_lo, config.thread)
            self.native = native
            self._context()
            complete = self.binding.stage in (COMPLETE_SET_STAGE, EVERY_MATCH_STAGE, ISOLATED_STAGE, ISOLATED_C3_STAGE)
            backend = CoreWriter26Coverage if complete else CoreWriter26Experiment
            if self.binding.stage == ISOLATED_STAGE:
                from ..serving26_isolated import CoreWriter26IsolatedCoverage
                backend = CoreWriter26IsolatedCoverage
            elif self.binding.stage == ISOLATED_C3_STAGE:
                from ..serving26_isolated import CoreWriter26IsolatedTransition
                backend = CoreWriter26IsolatedTransition
            settings = dict(trigger='once', evidence_kind='cpu_contract' if self.binding.simulation else 'natural_worker') \
                if complete else dict(stage='c1_single_effects')
            self.backend = backend(self.model, native, self.binding.reader,
                guard=self._context, worker_identity=self.identity, folder=self.binding.folder/owner/'candidate-experiment',
                max_targets=self.binding.max_targets, **settings)
            if self.binding.stage in (ISOLATED_STAGE, ISOLATED_C3_STAGE):
                from ..serving26_isolated import Serving26IsolatedSession
                self.session = Serving26IsolatedSession(self)
            else:
                self.session = Serving26CoreSession(self)
        elif action == 'check':
            require(self._registered and self.session is not None, 'WRITER26_CORE_REGISTER_REQUIRED')
            self._native_state()
            self.checked = True
        elif action in ('enable', 'execute'):
            require(self.checked and not self.attempted, 'WRITER26_CORE_ONE_ATTEMPT_REQUIRED')
            self.attempted, self.stopped = True, False
            self.session.install()
        elif action == 'advance':
            require(self.attempted and not self.stopped, 'WRITER26_CORE_ACTIVE_ATTEMPT_REQUIRED')
            if settings == {'collect': True}:
                self.session.collect()
                if (self.binding.stage in (EVERY_MATCH_STAGE, ISOLATED_C3_STAGE) and self.current_run == 'c2'
                        and self.backend.phase == 'complete' and not self.backend.blockers and not self.backend.failure):
                    self._start_c3(owner, generation)
            else:
                require(set(settings) == {'request_id'}, 'WRITER26_CORE_EXPLICIT_ARM_OR_COLLECT_REQUIRED')
                self.session.expect(settings['request_id'])
        elif action in ('finish', 'disable', 'abort', 'recover', 'clean', 'unregister'):
            if action == 'finish':
                require(self.binding.stage not in (EVERY_MATCH_STAGE, ISOLATED_C3_STAGE) or self.current_run == 'c3',
                        'WRITER26_C3_ACTUAL_REQUESTS_REQUIRED')
                require(self.backend is not None and self.backend.recovered and not self.backend.failure
                    and self.backend.phase == 'complete', 'WRITER26_CORE_REAL_OFF_AND_HEALTH_REQUIRED')
                require(not isinstance(self.backend, CoreWriter26Coverage) or self.backend.coverage is not None
                    and not self.backend.blockers, 'WRITER26_COMPLETE_CENSUS_AND_ALL_TARGET_QUALIFICATION_REQUIRED')
            self.cleanup_requested = True
            try:
                if self.session is not None:
                    self.session.close()
                elif self.backend is not None:
                    self.backend.close()
            except BaseException as error:
                self.cleanup_failures.append(dict(action=action, owner=owner, generation=generation,
                    error_type=type(error).__name__, reason=str(error),
                    group_recovery=deepcopy(self.backend.coverage.recovery_report())
                        if isinstance(self.backend, CoreWriter26Coverage) and self.backend.coverage is not None else None))
                raise
            self.stopped = True
            if action in ('clean', 'unregister'):
                self._registered = False
            if action == 'unregister' and self.reserved:
                self.binding.leases.remove(self._ticket)
                self.reserved = False
        else:
            raise RuntimeError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def _start_c3(self, owner, generation):
        require(not self.c3_started and self.current_run == 'c2', 'WRITER26_ONE_LIVE_C3_TRANSITION_REQUIRED')
        try:
            self.session.close()
        except BaseException as error:
            self.transition_failures.append(dict(phase='c2_cleanup', owner=owner, generation=generation,
                error_type=type(error).__name__, reason=str(error),
                group_recovery=deepcopy(self.backend.coverage.recovery_report())))
            raise
        require(self.backend.finalized and self.backend.proof is not None
            and self.backend.coverage.recovery_report()['complete'], 'WRITER26_ORIGINAL_LIVE_C2_REQUIRED')
        self.c2_proof = self.backend.proof
        self.c2_evidence = self.query(owner=owner, generation=generation)['evidence']
        self.c2_session = self.session
        # Old Native and all hooks are confirmed gone. The same Runtime owner
        # starts the existing every-match engine with a fresh Native generation.
        self.backend = self.session = None
        try:
            native = self.binding.runtime_factory(owner, generation+1)
            self.current_run, self.c3_started = 'c3', True
            config = native.config
            self.native_scope = (owner, generation+1, config.owner_hi, config.owner_lo, config.thread)
            self.native = native
            self._context()
            backend_type, session_type = CoreWriter26Coverage, Serving26CoreSession
            if self.binding.stage == ISOLATED_C3_STAGE:
                from ..serving26_isolated import CoreWriter26IsolatedTransition, Serving26IsolatedSession
                backend_type, session_type = CoreWriter26IsolatedTransition, Serving26IsolatedSession
            self.backend = backend_type(self.model, native, self.binding.reader,
                guard=self._context, worker_identity=self.identity, folder=self.binding.folder/owner/'candidate-experiment-c3',
                max_targets=self.binding.max_targets, trigger='every_match', c2_proof=self.c2_proof,
                evidence_kind='cpu_contract' if self.binding.simulation else 'natural_worker')
            self.session = session_type(self)
            self.session.install()
        except BaseException as error:
            self.transition_failures.append(dict(phase='c3_prepare', owner=owner, generation=generation,
                error_type=type(error).__name__, reason=str(error)))
            raise

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        raw = self._native_state()
        core = raw['core']
        serving = self.session.readback() if self.session is not None else None
        isolated = self.binding.stage in (ISOLATED_STAGE, ISOLATED_C3_STAGE)
        evidence = (self.backend.report(compact_rpc=True) if isolated else self.backend.report()) \
            if self.backend is not None else None
        recovery = self.backend.coverage.recovery_report() if isinstance(self.backend, CoreWriter26Coverage) \
            and self.backend.coverage is not None else None
        if evidence is not None:
            if evidence.get('coverage') is not None and not evidence['coverage'].get('compact_rpc_projection'):
                from fault_injection.deployments.writer26_inventory import digest
                coverage = evidence['coverage']
                full_sha = digest(coverage)
                for name in ('calls', 'discovery', 'observation'):
                    value = coverage.pop(name)
                    coverage[name+'_original_sha256'] = digest(value)
                coverage.update(compact_rpc_projection=True, original_report_sha256=full_sha)
            rows = evidence.pop('rows', None)
            candidates = evidence.pop('candidates', None)
            # The complete report (including unselected calls, aliases, typed
            # snapshots and failures) stays in immutable action-owned originals.
            # RPC history retains current Native state and the complete indices.
            if rows is not None:
                evidence['request_index'] = [dict(request_id=row['request_id'], phase=row['phase'],
                native_count=len(row['candidates']), output_sha256=row['output_sha256'],
                admissions=row['admissions']) for row in rows]
            if candidates is not None:
                evidence['candidate_index'] = [dict(index=row['index'], target_key=row['target_key'], status=row['status'],
                reasons=row['reasons'], module_object_identity=(row['callback'] or {}).get('module_object_identity'),
                module_path=(row['callback'] or {}).get('module_path')) for row in candidates]
            evidence['original_observations'] = self.session.evidence_readback() if self.session else None
            if self.c2_evidence is not None:
                self.c2_session.evidence_readback()  # Verify originals; never collect or advance.
                evidence['prior_c2'] = deepcopy(self.c2_evidence)
            evidence['current_run'] = self.current_run
            evidence.pop('outcomes', None)
            evidence.pop('public_output_observer', None)
            evidence.pop('metadata_arrays_initial', None)
            evidence.pop('metadata_arrays_latest', None)
            evidence['public_core'] = dict(worker_identity=deepcopy(self.identity), owner=owner, generation=generation,
                native_owner=[self.native.config.owner_hi, self.native.config.owner_lo], native_generation=self.native.config.generation,
                simulation=self.binding.simulation, qualification_granted=False,
                c2=evidence['c2'], c3=evidence['c3'], stage=self.binding.stage,
                owner_kind='runtime_lease_native_namespace', query_read_only=True, legacy_qualification_inherited=False)
        elif self.c2_evidence is not None:
            self.c2_session.evidence_readback()
            evidence = dict(prior_c2=deepcopy(self.c2_evidence), current_run=self.current_run,
                            c2=False, c3=False, qualification_granted=False)
        failed = bool(self.backend is not None and self.backend.failure or core['error']
            or raw['foreign_calls'] or raw['rejected_calls'])
        resources = bool(core['installed'] or core['pending'] or core['executing'] or core['device_bytes']
            or core['host_bytes'] or core['descriptor_pending'] or raw['descriptor'] or raw['prepared_fia'])
        return dict(execution_boundary='native', native_loaded=True, native_registered=bool(core['installed']),
            registered=self._registered, enabled=False, owner=owner, generation=generation,
            inflight=int(bool(self.session and (self.session.root_running or self.session.logits_running or self.backend.active)))
                + core['executing'] + raw['callback_active'] + raw['completion_active'],
            leases=core['pending'] + core['descriptor_pending'] + raw['descriptor'] + raw['prepared_fia']
                + (len(recovery['held_leases']) if recovery and self.cleanup_requested else 0),
            group_recovery=recovery, cleanup_failures=list(self.cleanup_failures),
            transition_failures=deepcopy(self.transition_failures),
            native=raw, binding_reserved=self.reserved, serving=serving, evidence=evidence,
            replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(failed and (not self.stopped or resources)),
            recovery_validation_pending=bool(self.attempted and not self.stopped or recovery and not recovery['complete']),
            query_read_only=True)
