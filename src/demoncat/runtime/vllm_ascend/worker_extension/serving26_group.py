"""Standard Worker entry for bounded native-v2 writer candidate experiments."""
from pathlib import Path

from .serving import ServingReplaySession, complete_tensor
from .serving26_inventory import Serving26InventorySession
from .serving_context import require


class Serving26GroupExperiment(Serving26InventorySession):
    def _fault_phase(self):
        return self.backend.phase != 'complete' and self.backend.entry['mode'] == 'fault'

    def _root_post(self, module, args, kwargs, output):
        if self.current is None:
            return
        try:
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_FRESH_PREFILL_HIDDEN_STATES_REQUIRED')
            _, self.hidden_digest, finite = complete_tensor(output[-1:].contiguous())
            require(finite or self._fault_phase(), 'HEALTHY_HIDDEN_STATES_NOT_FINITE')
            self.current['root_returned'] = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active,
                    'COMPLETE_REAL_ROOT_BEFORE_LOGITS_REQUIRED')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None, 'ACTUAL_LOGITS_HEAD_CHANGED')
            _, value, finite = complete_tensor(actual['hidden_states'])
            require(value == self.hidden_digest and (finite or self._fault_phase()), 'ACTUAL_LOGITS_ROWS_CHANGED')
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise

    def report(self):
        self.collect()
        value = self.backend.report()
        return dict(phase=self.backend.phase, stage=value['stage'], completed_requests=len(self.rows),
            installed=self.installed, owned_root_logits_hooks=len(self.handles), owned_module_hooks=value['owned_hooks'],
            resolved_candidate_count=value['resolved_candidate_count'], qualified_candidate_count=value['qualified_candidate_count'],
            candidate_status_counts={status: sum(row['status'] == status for row in value['candidates'])
                for status in ('UNSUPPORTED', 'UNVERIFIED', 'OBSERVED_NOT_QUALIFIED')},
            full_native_writer_counts=[len(row['candidates']) for row in self.backend.rows],
            failure=value['failure'], blockers=value['blockers'], native_state=value['native_state'],
            recovered=value['recovered'], c2=False, c3=False, scope=value['scope'],
            inventory=str(self.backend.folder / 'writer_inventory.json'), receipt=str(self.folder / 'serving_receipt.json'))


def install_writer26_group_experiment(worker, *, mechanism, prompt_token_ids, evidence_dir,
                                      manifest_sha256, stage='inventory', max_targets=8, statistics=False):
    from fault_injection.deployments.writer26_group import NativeWriter26Group
    from fault_injection.deployments.writer26_inventory import NativeWriterInventory
    from fault_injection.deployments.writer26_experiment import Writer26Experiment
    require(not hasattr(worker, '_demoncat_writer26_group_experiment'), 'EXCLUSIVE_WRITER_EXPERIMENT_REQUIRED')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'), 'ACTUAL_HEALTHY_WORKER_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_WRITER_GROUP_EVIDENCE_REQUIRED')
    runtime = NativeWriter26Group(mechanism=mechanism, bundle='/native', manifest_sha256=manifest_sha256)
    reader = NativeWriterInventory('/native/libfault26_readonly_trace_abi0.so',
        runtime.build['files']['libfault26_readonly_trace_abi0.so'])
    model = worker.model_runner.model
    collector = None
    if statistics:
        from .serving20_coverage import statistics_collector
        collector = statistics_collector(folder / 'statistics')
    backend = Writer26Experiment(model, runtime, reader, worker_identity=identity['worker'],
        folder=folder / 'candidate-experiment', stage=stage, max_targets=max_targets)
    session = Serving26GroupExperiment(worker, model, backend, runtime, runtime.owner,
        prompt_token_ids=prompt_token_ids, evidence_dir=folder / 'runtime', collector=collector)
    session.install()
    worker._demoncat_writer26_group_experiment = session
    return dict(status='NATIVE_V2_CANDIDATE_EXPERIMENT_BOUND_NOT_C2', mechanism=mechanism,
        stage=stage, build=manifest_sha256, worker=identity['worker'], core_methods_replaced=False)


def remove_writer26_group_experiment(worker):
    session = worker._demoncat_writer26_group_experiment
    session.close()
    result = session.report()
    require(not result['owned_root_logits_hooks'] and not result['owned_module_hooks']
            and not session.backend.output_refs and not session.backend.retained, 'WRITER_EXPERIMENT_HOST_RESOURCES_NOT_RELEASED')
    del worker._demoncat_writer26_group_experiment
    return dict(status='WRITER_EXPERIMENT_REMOVED', report=result)
