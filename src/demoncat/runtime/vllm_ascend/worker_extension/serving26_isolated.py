"""Isolated fresh-request C2 over the existing Native writer and group owner.

Native restoration still completes at the real consumer. Only the already
admitted CoverageSession decision stays active until the real root/logits end,
using the existing concurrency-one gate to prevent a second F in that request.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading

from fault_injection.coverage import Outcome
from fault_injection.coverage_session import target_id
from fault_injection.deployments.writer26_experiment import Writer26Experiment
from fault_injection.deployments.writer26_inventory import digest, correlate_lifetimes
from .serving26_core import CoreWriter26Coverage, Serving26CoreSession
from .serving_context import require


SCOPE = 'ALL_STABLE_ELIGIBLE_TARGETS_ISOLATED_F_OFF_FRESH_REQUEST_PAIRS'


class CoreWriter26IsolatedCoverage(CoreWriter26Coverage):
    def __init__(self, *args, **kwargs):
        require(kwargs.get('trigger', 'once') == 'once', 'ISOLATED_WRITER26_IS_C2_ONLY')
        self.isolated_order = ()
        self.isolated_active = None
        self.isolated_pairs = []
        super().__init__(*args, **kwargs)
        self.implementation_sha = digest([self.implementation_sha,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), SCOPE])

    def _prepare_complete_plan(self):
        self.isolated_order = tuple(self.targets)  # Complete actual census order, never a user selector.
        require(self.isolated_order and len(set(self.isolated_order)) == len(self.targets),
                'ISOLATED_COMPLETE_TARGET_ORDER_REQUIRED')
        self.plan = self.plan[:5]
        for index in range(len(self.isolated_order)):
            self.plan.extend(dict(phase=phase, mode=mode, target='*', isolated_pair=index)
                for phase, mode in (('group_fault', 'fault'), ('group_off', 'off')))
        self.plan.append(dict(phase='health', mode=None, target=None))
        if len(self.candidate_inventory)*2*len(self.isolated_order) > 8192:
            self.blockers.append('ISOLATED_COMPLETE_NATIVE_CALL_PLAN_EXCEEDS_SESSION_BUDGET')
            self.plan = self.plan[:2]+[dict(phase='health', mode=None, target=None)]

    def _planned_owned_records(self, remaining):
        return len(self.targets)*5  # Every target: fresh R/L/OFF, one F, its own OFF.

    def begin(self, request_id, input_sha256):
        require(self.isolated_active is None, 'PREVIOUS_ISOLATED_LOGICAL_DECISION_UNFINISHED')
        if self.phase == 'group_fault':
            require(len(self.isolated_pairs) == self.entry['isolated_pair']
                and all(pair.get('off_request_id') for pair in self.isolated_pairs),
                'PREVIOUS_TARGET_REAL_OFF_BEFORE_NEXT_ISOLATED_F_REQUIRED')
            self.first_anomaly = False  # Only this fresh request's propagation is in scope.
        if self.phase == 'group_fault' and self.coverage is not None:
            # The first F creates the existing group. Later isolated requests
            # retain that exact group; they do not construct another session.
            require(not self.failure, 'FAILED_COMPLETE_GROUP_CANNOT_RETRY_OR_SELECT_ANOTHER_SET')
            require(self._actual_aliases() == self.complete_aliases, 'COMPLETE_GROUP_ACTUAL_MODULE_ALIASES_CHANGED')
            return Writer26Experiment.begin(self, request_id, input_sha256)
        return super().begin(request_id, input_sha256)

    def _observe_callback(self, row, call):
        super()._observe_callback(row, call)
        if self.phase != 'group_fault' or self.coverage is None:
            return
        call_id = str(call.independent_call)
        decision = self.decisions[call_id]
        if decision.admitted:
            require(self.isolated_active is None, 'ISOLATED_SECOND_LOGICAL_ADMISSION')
            self.isolated_active = dict(decision=decision, call_id=call_id,
                target_key=row['target_key'], request_id=self.current['request_id'], execution_sha256=None)
        if self.failure:
            self._fail_isolated_decision()

    def _requested_mode(self, row):
        mode, key = super()._requested_mode(row)
        if not self.phase.startswith('group_'):
            return mode, key
        expected = self.isolated_order[self.entry['isolated_pair']]
        if self.phase == 'group_off':
            return (mode, key) if key == expected else (None, None)
        if mode == 'fault':
            require(key == expected and self.isolated_active is not None
                and self.isolated_active['call_id'] == str(row['independent_call']),
                'ISOLATED_ACTUAL_TARGET_ORDER_OR_ONE_F_CHANGED')
        return mode, key

    def _expected_admissions_for(self, row):
        return 1 if row['phase'].startswith('group_') else super()._expected_admissions_for(row)

    def _before_admission(self, row, mode, target, residual_sha256):
        if mode == 'fault':
            reference = self.target_refs[target]['residual_sha256']
            require(self.current['admissions'] == 0 and residual_sha256 == reference,
                    'ISOLATED_FIRST_F_MUST_START_FROM_ACTUAL_HEALTHY_RESIDUAL')
            require(self.isolated_active is not None and self.isolated_active['target_key'] == target,
                    'ISOLATED_CURRENT_LOGICAL_OWNER_REQUIRED')
            proof = dict(actual_residual_before_sha256=residual_sha256,
                qualified_reference_residual_sha256=reference, earlier_faults_this_request=0,
                uses_existing_digest_no_extra_tensor_copy=True)
            row['isolated_pre_fault_baseline'] = proof
            self.isolated_active['pre_fault_baseline'] = proof

    def _outcome_complete(self, outcome, row):
        if self.phase != 'group_fault':
            super()._outcome_complete(outcome, row)
            if self.phase == 'group_off':
                pair = self.isolated_pairs[self.entry['isolated_pair']]
                require(pair['target_key'] == outcome['target'] and not pair.get('off_request_id'),
                        'ISOLATED_OFF_MUST_MATCH_PRECEDING_FAULT')
                pair.update(off_request_id=self.current['request_id'],
                    off_independent_call=outcome['native_call']['independent_call'],
                    off_outcome_sha256=digest(outcome))
            return
        active = self.isolated_active
        require(active is not None and active['call_id'] == str(outcome['native_call']['independent_call'])
            and active['target_key'] == outcome['target'] and active['execution_sha256'] is None,
            'ISOLATED_ORIGINAL_NATIVE_COMPLETION_REQUIRED')
        # The original Native receipt already proves consumer, restoration and
        # release. Keep its immutable original; retain no extra tensor/lease.
        active['execution_sha256'] = self.artifact('execution-'+active['call_id'], outcome)
        self.first_anomaly = True

    def _fail_isolated_decision(self):
        active = self.isolated_active
        if active is None:
            return
        execution = active['execution_sha256']
        self.coverage.finish(active['decision'], Outcome.FAILED, execution_sha256=execution,
            effect_sha256=execution, native_applied=bool(execution), original_calls=1 if execution else 0)
        self.finished.add(active['call_id'])
        self.isolated_active = None

    def _record_failure(self, reason, source, *, token=None):
        super()._record_failure(reason, source, token=token)
        self._fail_isolated_decision()

    def fail(self, reason):
        row = self.current
        super().fail(reason)
        if self.coverage is None:
            return
        # A real root/logits unwind also terminates unadmitted observation
        # decisions. Recover their original once-only execution from the real
        # Native trace; missing evidence is FAILED, never invented passthrough.
        candidates, observed, read_error = {}, [], None
        if row is not None:
            try:
                _, observed = self.reader.delta(row['native_before'])
                values = correlate_lifetimes(observed, self.callbacks[row['callback_start']:],
                    self.producers[row['producer_start']:], self.runtime.mechanism)
                candidates = {str(item['native']['call']): item for item in values}
            except BaseException as error:
                read_error = str(error)
        for call_id, decision in self.decisions.items():
            if call_id in self.finished:
                continue
            candidate = candidates.get(call_id)
            executions = candidate.get('execution', []) if candidate else []
            passed = not decision.admitted and len(executions) == 1 and executions[0]['status'] == 0
            evidence = self.artifact('unwound-'+call_id, dict(first_failure=self.failure,
                native_events=observed, candidate=candidate, read_error=read_error))
            self.coverage.finish(decision, Outcome.PASSTHROUGH if passed else Outcome.FAILED,
                execution_sha256=evidence, native_applied=False, original_calls=len(executions))
            self.finished.add(call_id)
        self.save()

    def _request_complete(self, row):
        if row['phase'] == 'group_fault':
            active = self.isolated_active
            require(active is not None and active['request_id'] == row['request_id']
                and active['execution_sha256'] and active.get('pre_fault_baseline')
                and row['admissions'] == row['completed'] == 1,
                'ONE_COMPLETE_ISOLATED_WRITER_CONSUMER_BEFORE_REAL_RETURN_REQUIRED')
            self.coverage.finish(active['decision'], Outcome.HIT,
                execution_sha256=active['execution_sha256'], effect_sha256=active['execution_sha256'],
                native_applied=True, original_calls=1)
            self.finished.add(active['call_id'])
            self.isolated_pairs.append(dict(index=row['plan']['isolated_pair'],
                target_key=active['target_key'], target_id=target_id(self.identities[active['target_key']]),
                fault_request_id=row['request_id'], fault_independent_call=int(active['call_id']),
                fault_execution_sha256=active['execution_sha256'], actual_request_returned=True,
                pre_fault_baseline=deepcopy(active['pre_fault_baseline']),
                native_recovered_before_logical_completion=True))
            self.isolated_active = None
        super()._request_complete(row)
        if len(self.rows) > 2 and self.candidate_inventory:
            require(len(row['candidates']) == len(self.candidate_inventory),
                    'ISOLATED_KEEP_COMPLETE_NATIVE_WRITER_DENOMINATOR')

    def save(self):
        # This mutable progress index is not an observation original. The
        # explicit collect persists each complete request, and final cleanup
        # persists the complete merged report, below. Query never calls save.
        value = dict(schema='demoncat.writer26.isolated_progress.v1',
            evidence_location='runtime/core-evidence', evidence_authority=False,
            progress=self.report(compact_rpc=True))
        (self.folder/'writer_inventory.json').write_text(json.dumps(value, allow_nan=False)+'\n')

    def report(self, *, compact_rpc=False, observation_start=None):
        value = super().report(compact_rpc=compact_rpc, observation_start=observation_start)
        value.update(execution_scope=SCOPE, independent_per_target_replay=True,
            scope=SCOPE,
            isolated_pairs=deepcopy(self.isolated_pairs), isolated_logical_inflight=int(self.isolated_active is not None),
            isolated_plan=dict(target_order=list(self.isolated_order),
                group_requests=len(self.plan), ordinary_post_unregister_requests=1,
                full_native_writer_count=len(self.candidate_inventory),
                native_owned_execute_records=5*len(self.isolated_order),
                max_targets_is_selector=False, c3_requested=False))
        return value


class Serving26IsolatedSession(Serving26CoreSession):
    """Full per-request originals, plus a final full merged report, only here."""
    def __init__(self, provider):
        super().__init__(provider)
        self.captured_requests = 0

    def _root_post(self, module, args, kwargs, output):
        error = sys.exc_info()[1]
        if output is None and error is not None:
            try:
                self._same_worker()
                self._last(module)
                self._failure(error)
            finally:
                if threading.get_ident() == self.provider.thread:
                    self.root_running = False
            return
        return super()._root_post(module, args, kwargs, output)

    def _save(self):
        value = dict(schema='demoncat.writer26.isolated_serving_progress.v1',
            boundary=self.boundary_report, requests=self.rows,
            evidence_location='core-evidence', evidence_authority=False)
        (self.folder/'serving_receipt.json').write_text(json.dumps(value, allow_nan=False)+'\n')

    def capture_evidence(self, action):
        self._same_worker()
        folder = self.folder/'core-evidence'
        folder.mkdir(exist_ok=True)
        path = folder/(str(self.evidence_attempts).zfill(3)+'-'+action+'.json')
        self.evidence_attempts += 1
        start = self.captured_requests if action == 'collect' else None
        value = dict(schema='demoncat.writer26.isolated_core_observations.v1', action=action,
            observation_kind='COMPLETE_REQUEST_DELTA' if start is not None else 'FINAL_COMPLETE_REPORT',
            previous_original_sha256=self.evidence_files[-1]['sha256'] if self.evidence_files else None,
            worker=self.provider.identity, native_owner=self.provider._ticket[0],
            generation=self.provider._ticket[1], rpc_generation=self.provider._ticket[1],
            native_generation=self.backend.runtime.config.generation, run=self.provider.current_run,
            simulation=self.provider.binding.simulation, qualification_granted=False,
            inventory=self.backend.report(observation_start=start),
            serving_requests=self.rows[start:] if start is not None else self.rows, serving=self.readback())
        failed = self.backend.folder/'failed_request.json'
        if failed.is_file():
            raw = failed.read_bytes()
            value['failed_request_original'] = dict(path=str(failed), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
        with path.open('xb') as stream:
            stream.write(data)
        self.evidence_files.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
            action=action, complete_requests=len(self.backend.rows), observation_kind=value['observation_kind']))
        self.captured_requests = len(self.backend.rows)
        self.last_evidence_state = self._evidence_state()
