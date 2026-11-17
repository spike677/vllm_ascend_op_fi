"""Two actual fresh-prefill Worker requests for writer/consumer lifetimes."""
import hashlib
import inspect
from pathlib import Path
import sys

from .serving import ServingReplaySession, complete_tensor
from .serving_context import require


class Serving26InventorySession(ServingReplaySession):
    def install(self):
        self.backend.install()
        try:
            super().install()
        except BaseException:
            self.backend.close()
            raise

    def _root_pre(self, module, args, kwargs):
        super()._root_pre(module, args, kwargs)
        try:
            self.backend.begin(self.current['actual_internal_request_id'], self.current['input_sha256'])
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        if output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return
        super()._root_post(module, args, kwargs, output)

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active,
                    'ACTUAL_COMPLETE_ROOT_BEFORE_LOGITS_REQUIRED')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None,
                    'ACTUAL_LOGITS_HEAD_CHANGED')
            _, value, finite = complete_tensor(actual['hidden_states'])
            require(finite and value == self.hidden_digest, 'ACTUAL_LOGITS_QUERY_ROWS_CHANGED')
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        if output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return
        super()._logits_post(module, args, kwargs, output)

    def close(self):
        self._same_worker()
        self.collect(force=True)
        require(not self.backend.active, 'ACTIVE_WRITER_INVENTORY_CANNOT_DETACH')
        self.backend.close()
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.installed = False
        self.current = self.expected = None
        self._save()

    def report(self):
        self.collect()
        report = self.backend.report()
        return dict(installed=self.installed, completed_requests=len(self.rows),
            owned_root_logits_hooks=len(self.handles), owned_module_hooks=report['owned_hooks'],
            actual_native_writers=[len(row['candidates']) for row in self.backend.rows],
            complete_writer_lifetimes=[sum(not candidate['reasons'] for candidate in row['candidates'])
                                       for row in self.backend.rows],
            healthy_repeated=report['healthy_repeated'], native_prepare_claims=report['native_prepare_claims'],
            native_injections=report['native_injections'], c1=False, c2=False, c3=False,
            scope=report['scope'], inventory=str(self.backend.folder / 'writer_inventory.json'))


def install_writer26_inventory(worker, *, mechanism, prompt_token_ids, evidence_dir, libraries):
    from fault_injection.deployments.residual26 import Residual26Runtime
    from fault_injection.deployments.writer26_inventory import NativeWriterInventory, WriterLifetimeInventory
    require(mechanism in ('26.residual_stomp', '26.fia_destination')
            and not hasattr(worker, '_demoncat_writer26_inventory'), 'EXCLUSIVE_NATIVE_WRITER_INVENTORY_REQUIRED')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'), 'READY_ACTUAL_WORKER_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_WRITER_INVENTORY_REQUIRED')
    model = worker.model_runner.model
    trace = libraries['trace']
    writer = libraries['fia' if mechanism == '26.fia_destination' else 'qmm']
    reader = NativeWriterInventory(trace['path'], trace['sha256'])
    runtime = Residual26Runtime(model, mechanism, library=writer['path'], sha256=writer['sha256'],
        evidence_dir=folder / 'native-control', log_path=folder / 'native-writer.jsonl')
    backend = WriterLifetimeInventory(model, runtime, reader, worker_identity=identity['worker'], folder=folder / 'inventory')
    session = Serving26InventorySession(worker, model, backend, runtime, 'read-only-writer26-inventory',
        prompt_token_ids=prompt_token_ids, evidence_dir=folder / 'runtime')
    session.install()
    worker._demoncat_writer26_inventory = session
    return dict(status='READONLY_ALL_WRITER_INVENTORY_BOUND_NOT_QUALIFIED',
        mechanism=mechanism, native_libraries=libraries, native_arm_enabled=False,
        source_sha256=hashlib.sha256(Path(inspect.getfile(WriterLifetimeInventory)).read_bytes()).hexdigest(),
        worker=identity['worker'], core_methods_replaced=False)


def remove_writer26_inventory(worker):
    session = worker._demoncat_writer26_inventory
    session.close()
    report = session.report()
    require(not report['owned_root_logits_hooks'] and not report['owned_module_hooks']
            and not session.backend.retained, 'WRITER_INVENTORY_RESOURCES_STILL_OWNED')
    del worker._demoncat_writer26_inventory
    return dict(status='READONLY_WRITER_INVENTORY_REMOVED', report=report)
