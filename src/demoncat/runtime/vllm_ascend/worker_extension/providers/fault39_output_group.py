"""One existing Runtime owner, actual enqueues, original C2 to C3 proof."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .fault39_output import Fault39OutputServingProvider, _PHASES, require
from ._scope import reject_unknown
from ..serving39_output_group import WorkerProjectionCoverage, Serving39OutputGroupSession
from fault_injection.deployments.postwrite import _equal
from fault_injection.deployments.postwrite_evidence import digest
from fault_injection.deployments.postwrite_coverage import _cpu_digest


class Fault39OutputGroupProvider(Fault39OutputServingProvider):
    session_type = Serving39OutputGroupSession

    def __init__(self, worker, model, identity):
        self.group = None
        self.plan = list(_PHASES)
        self.group_reports = {}
        self.collected_requests = 0
        self.originals = []
        self.capture_attempts = 0
        self.group_failure = None
        self.cleanup_requested = False
        self.cleanup_failures = []
        super().__init__(worker, model, identity)

    @property
    def phase(self):
        if self.recovering and self.group is None:
            return super().phase
        if self.recovering and self.group is not None:
            return 'group_recovery_'+self.recovering
        return self.plan[self.index] if self.index < len(self.plan) else 'complete'

    @property
    def native_phase(self):
        phase = self.phase
        if phase in _PHASES:
            return super().native_phase
        if phase.endswith('_health'):
            return 'health'
        if phase.endswith('_fault'):
            return 'fault'
        return 'group_'+phase.rsplit('_', 1)[-1]

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'finish':
            require(self.collected_requests == len(self.rows) and self.group is not None
                and 'c2' in self.group_reports and (self.binding.stage != 'c2_c3' or 'c3' in self.group_reports),
                'OUTPUT39_ALL_GROUP_COLLECTIONS_REQUIRED')
        if action == 'advance':
            self.verify(owner, generation)
            self._context()
            reject_unknown(settings, ('collect', 'request_id'))
            if settings == {'collect': True}:
                try:
                    self._collect()
                except BaseException as error:
                    self.cleanup_failures.append(dict(action='collect', reason=str(error),
                        group_recovery=self.group.session.recovery_report() if self.group and self.group.session else None))
                    raise
            else:
                require(set(settings) == {'request_id'} and self.collected_requests == len(self.rows),
                        'OUTPUT39_GROUP_EXPLICIT_COLLECT_REQUIRED')
                require(not self.failure and (not self.stopped or self.recovering),
                        'OUTPUT39_FAILED_GROUP_REQUIRES_EXPLICIT_RECOVERY')
                if not self.session.installed:
                    self.session.install()
                self.session.expect(settings['request_id'])
            return self.query(owner=owner, generation=generation)
        if action == 'recover' and self.group is not None:
            self.verify(owner, generation)
            self._context()
            if (len(self.rows) == self.collected_requests+1 and self.rows[-1]['phase'].endswith('_health')
                    and self.group.healthy_after is not None):
                self._collect()
                return self.query(owner=owner, generation=generation)
            self.session.close_window()
            self.p.recover()
            self.group._capture_native_receipts()
            if self.group.mode is not None:
                self.group.fail_observed_forward(RuntimeError(self.group_failure or self.failure or 'EXPLICIT_RECOVERY'))
                self.group.end_observed_forward()
            recovery = self.group.session.recovery_report() if self.group.session is not None else None
            if recovery and not recovery['complete'] or not self.recovered:
                self.recovering, self.stopped = ('off' if self.group.off_hash != self.group.baseline_hash else 'health'), False
                self.cleanup_requested = False
                self.failure = None
                self.collected_requests = len(self.rows)
                if not self.session.installed:
                    self.session.install()
            else:
                self._cleanup()
            return self.query(owner=owner, generation=generation)
        return super().dispatch(action, settings, owner=owner, generation=generation)

    def begin_actual(self, args, kwargs):
        if self.group is None:
            return super().begin_actual(args, kwargs)
        self._context()
        require(not self.failure and self.group is not None and not self.p.live_lease_count,
                'OUTPUT39_GROUP_CURRENT_OWNER_REQUIRED')
        request = self.session.current['actual_internal_request_id']
        require(request not in self.requests, 'OUTPUT39_REUSED_ACTUAL_REQUEST')
        self.requests.add(request)
        self.group._check_request(args, kwargs)
        self.recovered = False
        if self.native_phase == 'health':
            require(self.group.off_hash == self.group.baseline_hash, 'OUTPUT39_GROUP_OFF_BEFORE_HEALTH_REQUIRED')
            return
        mode = ('recovery_retry' if self.recovering else self.phase.rsplit('_', 1)[-1])
        self.group.begin_observed_forward(mode, args, kwargs, request_id=request)
        self.native_generation = self.p.generation

    def end_actual(self, output, result, request):
        if self.group is None:
            super().end_actual(output, result, request)
            from ..serving39_output_group import LOGITS_OUTPUT_DIGEST
            self.rows[-1]['output_digest'] = deepcopy(LOGITS_OUTPUT_DIGEST)
            if self.index == len(_PHASES):
                self.complete, self.stopped = False, False
            return
        phase = self.phase
        self._context()
        if self.native_phase == 'health':
            require(_equal(output, self.p._baseline_output) and not self.p.live_lease_count,
                    'OUTPUT39_GROUP_SAME_WORKER_HEALTH_CHANGED')
            self.group.healthy_after = dict(actual_request_id=request['actual_internal_request_id'],
                output_sha256=_cpu_digest(output), snapshot=deepcopy(request['snapshot']))
            from ..serving39_output_group import STRUCTURED_OUTPUT_DIGEST
            self.group.healthy_after['output_digest'] = deepcopy(STRUCTURED_OUTPUT_DIGEST)
        else:
            try:
                self.group.complete_observed_forward(result[0])
            except BaseException as error:
                self.group.fail_observed_forward(error)
                raise
            finally:
                self.group.end_observed_forward()
            if phase.endswith('_lossless'):
                self.group.open_coverage_session()
        self.rows.append(dict(phase=phase, actual_internal_request_id=request['actual_internal_request_id'],
            snapshot=deepcopy(request['snapshot']), output_sha256=request['model_output_sha256'],
            output_finite=request['model_output_finite'], report=result[2] if result else None,
            complete_inventory=self.p._last_observer.inventory.report() if result else None,
            native_owner=self.native_owner, native_generation=self.p.generation, native_leases=self.p.live_lease_count))
        from ..serving39_output_group import LOGITS_OUTPUT_DIGEST
        self.rows[-1]['output_digest'] = deepcopy(LOGITS_OUTPUT_DIGEST)
        if self.recovering:
            self.recovering = 'health' if phase.endswith('_off') else False
            self.stopped = phase.endswith('_health')
            self.recovered = self.stopped
        else:
            self.index += 1
            self.complete = self.index == len(self.plan)
            self.recovered = phase.endswith('_health')
            self.stopped = self.complete

    def _collect(self):
        require(self.session is not None and not self.session.root_running and not self.session.logits_running
            and self.session.current is None and self.session.expected is None, 'OUTPUT39_ACTUAL_RETURN_BEFORE_COLLECTION_REQUIRED')
        require(len(self.rows) == self.collected_requests+1, 'OUTPUT39_ONE_NEW_COMPLETED_REQUEST_REQUIRED')
        last = self.rows[-1]['phase']
        if self.group is None and self.index == len(_PHASES):
            self.group = WorkerProjectionCoverage(self)
            self.plan.extend(['c2_lossless']+['c2_fault']*len(self.group.eligible)+['c2_off', 'c2_health'])
            if self.binding.stage == 'c2_c3':
                self.plan.extend(['c3_lossless', 'c3_fault', 'c3_fault', 'c3_off', 'c3_health'])
            self.group.prepare_coverage('once')
        elif last in ('c2_health', 'c3_health', 'group_recovery_health'):
            # The actual health return is complete before removing root hooks.
            self.session.close()
            current_session = self.group.session is not None and self.group.session.generation == self.group.generation
            if current_session:
                label = 'c2' if self.group.group.trigger == 'once' else 'c3'
                report = self.group.finish_stage(failed=bool(self.group_failure))
                self.group_reports[label] = deepcopy(report)
                self.group_reports[label]['failed_attempt'] = self.group_failure
            else:
                require(last == 'group_recovery_health' and self.group_failure,
                        'OUTPUT39_GROUP_CURRENT_GENERATION_REQUIRED')
        self._capture('collect')
        self.collected_requests = len(self.rows)
        if last == 'c2_health' and self.binding.stage == 'c2_c3' and not self.group_failure:
            self.group.prepare_coverage('every_match')

    def fail(self, error):
        super().fail(error)
        if self.group is not None and self.group_failure is None:
            self.group_failure = str(error)

    def _cleanup(self):
        self.cleanup_requested = True
        try:
            if self.group is not None:
                self.session.close_window()
                self.p.recover()
                recovery = self.group.session.recovery_report() if self.group.session is not None else None
                require(not recovery or recovery['complete'], 'OUTPUT39_GROUP_LOGICAL_OWNER_RECOVERY_REQUIRED')
            super()._cleanup()
        except BaseException as error:
            self.cleanup_failures.append(dict(reason=str(error), native_owner=self.native_owner,
                native_generation=self.native_generation,
                group_recovery=self.group.session.recovery_report() if self.group and self.group.session else None))
            raise

    def _capture(self, action):
        from ..serving39_output_group import STRUCTURED_OUTPUT_DIGEST, LOGITS_OUTPUT_DIGEST
        folder = self.binding.folder/self._ticket[0]/'group-originals'
        folder.mkdir(exist_ok=True)
        path = folder/(str(self.capture_attempts).zfill(3)+'-'+action+'.json')
        self.capture_attempts += 1
        value = dict(schema='demoncat.output39.worker_group_original.v1', action=action,
            rpc_owner=self._ticket[0], rpc_generation=self._ticket[1], worker=deepcopy(self.identity),
            actual_requests=deepcopy(self.rows), module_aliases=super().evidence()['module_aliases'],
            group=self.group.report() if self.group else None, finished_groups=deepcopy(self.group_reports),
            failure=self.group_failure, failures=deepcopy(self.failures), qualification_granted=False)
        value['digest_scopes'] = dict(actual_requests=deepcopy(LOGITS_OUTPUT_DIGEST),
            forward_artifacts=deepcopy(STRUCTURED_OUTPUT_DIGEST), actual_health=deepcopy(STRUCTURED_OUTPUT_DIGEST))
        raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
        with path.open('xb') as stream:
            stream.write(raw)
        self.originals.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
            actual_requests=len(self.rows), action=action))

    def evidence(self):
        evidence = super().evidence()
        evidence['actual_requests'] = [dict(phase=row['phase'], actual_internal_request_id=row['actual_internal_request_id'],
            output_sha256=row['output_sha256'], output_finite=row['output_finite'], native_owner=row['native_owner'],
            native_generation=row['native_generation'], native_leases=row['native_leases'],
            output_digest=row.get('output_digest')) for row in self.rows]
        evidence.update(scope='INITIALIZED_WORKER_COMPLETE_PROJECTION_SET_FIXED_FRESH_PREFILL',
            stage=self.binding.stage, next_plan=list(self.plan[self.index:]), collection_pending=len(self.rows)!=self.collected_requests,
            group_originals=deepcopy(self.originals), group_failure=self.group_failure,
            group_summary={key: dict(coverage=report['coverage'], eligible_target_count=report['eligible_target_count'],
                all_projection_count=report['all_projection_count'], execution_scope=report['execution_scope'],
                original_report_sha256=digest(report)) for key, report in self.group_reports.items()})
        for summary in evidence['group_summary'].values():
            coverage = deepcopy(summary['coverage'])
            for key in ('calls', 'discovery', 'observation'):
                coverage[key+'_sha256'] = digest(coverage.pop(key))
            summary['coverage'] = coverage
        for key in ('c2', 'c3'):
            report = self.group_reports.get(key)
            evidence[key] = bool(report and report['coverage']['contract_satisfied'] and not report['failed_attempt']
                and not self.binding.simulation)
        return evidence

    def query(self, *, owner, generation):
        for record in self.originals:
            data = Path(record['path']).read_bytes()
            require(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'],
                    'OUTPUT39_GROUP_ORIGINAL_CHANGED')
        state = super().query(owner=owner, generation=generation)
        recovery = self.group.session.recovery_report() if self.group and self.group.session else None
        state.update(group_recovery=recovery, cleanup_failures=deepcopy(self.cleanup_failures))
        if recovery and self.cleanup_requested and not recovery['complete']:
            state['leases'] += len(recovery['held_leases'])
        return state
