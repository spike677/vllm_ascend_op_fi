"""Worker RPC ownership bridge for the existing natural writer26 single site."""
from copy import deepcopy
import os
import threading

from ._scope import Scope, reject_unknown
from .. import identity as identities
from ..serving_context import require
from ..serving26_core import (ATTRIBUTE, Writer26CoreBinding, CoreWriter26Experiment, Serving26CoreSession)


class Fault26CoreProvider(Scope):
    execution_boundary = 'native'

    def __init__(self, worker, model, identity, binding):
        super().__init__(worker, model, identity)
        require(type(binding) is Writer26CoreBinding, 'REVIEWED_WRITER26_CORE_BINDING_REQUIRED')
        self.binding, self.runner, self.thread = binding, worker.model_runner, threading.get_ident()
        self.native = self.backend = self.session = None
        self.reserved = self.checked = self.attempted = False
        self.stopped = True
        self.native_scope = None
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
                and self.native.owner == self._ticket[0] and config.generation == self._ticket[1]
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
            self.backend = CoreWriter26Experiment(self.model, native, self.binding.reader,
                guard=self._context, worker_identity=self.identity, folder=self.binding.folder/owner/'candidate-experiment',
                stage='c1_single_effects', max_targets=1)
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
            else:
                require(set(settings) == {'request_id'}, 'WRITER26_CORE_EXPLICIT_ARM_OR_COLLECT_REQUIRED')
                self.session.expect(settings['request_id'])
        elif action in ('finish', 'disable', 'abort', 'recover', 'clean', 'unregister'):
            if action == 'finish':
                require(self.backend is not None and self.backend.recovered and not self.backend.failure
                    and self.backend.phase == 'complete', 'WRITER26_CORE_REAL_OFF_AND_HEALTH_REQUIRED')
            if self.session is not None:
                self.session.close()
            elif self.backend is not None:
                self.backend.close()
            self.stopped = True
            if action in ('clean', 'unregister'):
                self._registered = False
            if action == 'unregister' and self.reserved:
                self.binding.leases.remove(self._ticket)
                self.reserved = False
        else:
            raise RuntimeError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner, generation=generation)

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self._context()
        raw = self._native_state()
        core = raw['core']
        serving = self.session.readback() if self.session is not None else None
        evidence = self.backend.report() if self.backend is not None else None
        if evidence is not None:
            rows = evidence.pop('rows')
            candidates = evidence.pop('candidates')
            # The complete report (including unselected calls, aliases, typed
            # snapshots and failures) stays in immutable action-owned originals.
            # RPC history retains current Native state and the complete indices.
            evidence['request_index'] = [dict(request_id=row['request_id'], phase=row['phase'],
                native_count=len(row['candidates']), output_sha256=row['output_sha256'],
                admissions=row['admissions']) for row in rows]
            evidence['candidate_index'] = [dict(index=row['index'], target_key=row['target_key'], status=row['status'],
                reasons=row['reasons'], module_object_identity=(row['callback'] or {}).get('module_object_identity'),
                module_path=(row['callback'] or {}).get('module_path')) for row in candidates]
            evidence['original_observations'] = self.session.evidence_readback() if self.session else None
            evidence.pop('outcomes')
            evidence.pop('public_output_observer')
            evidence.pop('metadata_arrays_initial')
            evidence.pop('metadata_arrays_latest')
            evidence['public_core'] = dict(worker_identity=deepcopy(self.identity), owner=owner, generation=generation,
                native_owner=[self.native.config.owner_hi, self.native.config.owner_lo], native_generation=generation,
                simulation=self.binding.simulation, qualification_granted=False, c2=False, c3=False,
                owner_kind='runtime_lease_native_namespace', query_read_only=True, legacy_qualification_inherited=False)
        failed = bool(self.backend is not None and self.backend.failure or core['error']
            or raw['foreign_calls'] or raw['rejected_calls'])
        resources = bool(core['installed'] or core['pending'] or core['executing'] or core['device_bytes']
            or core['host_bytes'] or core['descriptor_pending'] or raw['descriptor'] or raw['prepared_fia'])
        return dict(execution_boundary='native', native_loaded=True, native_registered=bool(core['installed']),
            registered=self._registered, enabled=False, owner=owner, generation=generation,
            inflight=int(bool(self.session and (self.session.root_running or self.session.logits_running or self.backend.active)))
                + core['executing'] + raw['callback_active'] + raw['completion_active'],
            leases=core['pending'] + core['descriptor_pending'] + raw['descriptor'] + raw['prepared_fia'],
            native=raw, binding_reserved=self.reserved, serving=serving, evidence=evidence,
            replay_pending=bool(self.attempted and not self.stopped), replay_failed=bool(failed and (not self.stopped or resources)),
            recovery_validation_pending=bool(self.attempted and not self.stopped), query_read_only=True)
