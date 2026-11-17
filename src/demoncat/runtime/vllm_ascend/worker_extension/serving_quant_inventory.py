"""Two real healthy Worker requests for the complete public quant inventory."""
import json
from pathlib import Path

from .serving import ServingReplaySession, complete_tensor
from .serving_context import require
from .quant_inventory import PublicQuantInventory


class ServingQuantInventorySession(ServingReplaySession):
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
            request = self.current['actual_internal_request_id']
            self.backend.begin(request, self.current['input_sha256'], request_ids=(request,),
                row_mapping=dict(kind='ACTUAL_FRESH_PREFILL_TOKEN_ROWS_NOT_DECODE_BATCH',
                                 scheduler_snapshot=self.current['snapshot']))
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard is not None:
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

    def close(self):
        super().close()
        self.backend.close()
        self._save()

    def _save(self):
        super()._save()
        (self.folder / 'quant_inventory.json').write_text(json.dumps(self.backend.report(), indent=2, allow_nan=False) + '\n')

    def report(self):
        self.collect()
        inventory = self.backend.report()
        return dict(installed=self.installed, completed_requests=len(self.rows),
            structural_candidate_count=inventory['structural_candidate_count'],
            actual_projection_calls=inventory['actual_projection_calls'],
            actual_public_api_calls=inventory['actual_public_api_calls'],
            public_observer_complete=inventory['public_observer_complete'],
            missing_public_call_details=inventory['missing_public_call_details'],
            unowned_public_calls=inventory['unowned_public_calls'],
            complete_healthy_replays=inventory['complete_healthy_replays'],
            repeated_public_inventory_stable=inventory['repeated_public_inventory_stable'],
            qualified_native_targets=0, c1=False, c2=False, c3=False,
            remaining_gates=inventory['remaining_gates'],
            report=str(self.folder / 'quant_inventory.json'))


def install_quant_inventory(worker, *, prompt_token_ids, evidence_dir):
    from fault_injection.quantization import _discover
    require(not hasattr(worker, '_demoncat_quant_inventory')
            and not hasattr(worker, '_demoncat_serving20_factory')
            and not hasattr(worker, '_demoncat_quant_factory'), 'EXCLUSIVE_QUANT_INVENTORY_REQUIRED')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
            'READY_UNQUARANTINED_WORKER_REQUIRED')
    model = worker.model_runner.model
    candidates, rejected = _discover(model, (1,))
    require(candidates, 'NO_ACTUAL_DYNAMIC_QUANT_CANDIDATES')
    # Public Worker identity response wraps the exact process/model identity.
    facts = identity['worker']
    inventory = PublicQuantInventory(model, candidates, worker_identity=facts)
    session = ServingQuantInventorySession(worker, model, inventory, None, 'quant-inventory-readonly',
        prompt_token_ids=prompt_token_ids, evidence_dir=evidence_dir)
    session.install()
    worker._demoncat_quant_inventory = session
    return dict(status='READONLY_QUANT_INVENTORY_INSTALLED_NOT_QUALIFIED',
                structural_candidates=len(candidates), structurally_rejected=rejected,
                core_methods_replaced=False, native_callback_installed=False, native_fault_enabled=False)


def remove_quant_inventory(worker):
    session = getattr(worker, '_demoncat_quant_inventory', None)
    require(session is not None, 'QUANT_INVENTORY_NOT_INSTALLED')
    session.close()
    report = session.report()
    require(not session.handles and not session.backend.patches and not session.backend.api_patches,
            'QUANT_OBSERVER_HOOKS_NOT_RELEASED')
    del worker._demoncat_quant_inventory
    return dict(status='READONLY_QUANT_INVENTORY_REMOVED', report=report)
