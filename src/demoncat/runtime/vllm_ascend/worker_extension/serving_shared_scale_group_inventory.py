"""Automatic natural B8 census over the independent group DSO; never C1 load."""
import os
from pathlib import Path

from .serving_quant import QuantServingSession
from .serving_shared_scale_inventory import SharedScaleInventorySession
from .serving_context import require


class GroupInventoryRuntime:
    """Only the idle state interface reused by the existing B8 runner hooks."""
    def __init__(self, binding):
        self.binding = binding
        self.build_identity = dict(binding.identity)

    def native_state(self):
        state = self.binding.query()
        require(not state['quarantined'] and not state['abi_exceptions'] and not state['orphan_descriptors'],
                'READONLY_GROUP_RUNTIME_OWNERSHIP_UNCERTAIN')
        return dict(armed=0, owned=state['core']['installed'],
                    pending=state['core']['pending_parent'] + state['census']['preparing'] + state['census']['pending'],
                    executing=state['census']['executing'], raw=state)


class SharedScaleGroupInventorySession(SharedScaleInventorySession):
    def _load_runtime(self, root, build_sha256):
        from fault_injection.quant_group_native import QuantGroupNative
        return GroupInventoryRuntime(QuantGroupNative.load(root, approved_build_sha256=build_sha256))

    def __init__(self, *args, **kwargs):
        # Reuse actual scheduler/embedding/logits hooks but bypass the legacy
        # inventory constructor and its C1 Native loader entirely.
        QuantServingSession.__init__(self, *args, **kwargs, baseline_control=True,
                                     collector=None, capture_scope='first_decode')
        from fault_injection.deployments.quant_group_inventory import GroupQuantJournal
        from .shared_scale_group_inventory import SharedScaleGroupInventory
        from .quant_row_provenance import RequestRowProvenance
        import torch_npu
        self.native_reader = GroupQuantJournal(self.runtime.binding, device=int(torch_npu.npu.current_device()))
        self.row_observer = RequestRowProvenance()
        identity = self.worker.demoncat_worker_identity()
        require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined')
                and identity['worker']['rank'] == 0,
                'ONE_ACTUAL_READY_TP1_WORKER_REQUIRED')
        self.inventory = SharedScaleGroupInventory(self.model, self.candidates, worker_identity=identity['worker'],
            native=self.native_reader, folder=self.folder / 'inventory', row_lineage=self.row_lineage,
            row_transfer=self.projection_rows)
        self.norm_evidence = None

    def report(self):
        value = super().report()
        inventory = self.inventory.report()
        value.update(schema='demoncat.shared_scale.group_serving_inventory.v1',
            actual_native_prepare_calls=inventory['actual_native_prepare_calls'],
            actual_native_execute_calls=inventory['actual_native_execute_calls'],
            group_owner_installed=False, legacy_library_loaded=False,
            scope='ALL_GROUP_MODEL_PARENTS_AND_NATIVE_CHILDREN_IN_TWO_ACTUAL_B8_FIRST_DECODES')
        return value


def install_shared_scale_group_inventory(worker, *, prompts, root, build_sha256, evidence_dir):
    require(not any(hasattr(worker, name) for name in ('_demoncat_shared_scale_inventory',
        '_demoncat_quant_factory', '_demoncat_quant_inventory', '_demoncat_serving20_factory')),
        'EXCLUSIVE_NEW_GROUP_QUANT_INVENTORY_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
            'HOST_DRIVEN_SYNCHRONOUS_NATIVE_QUANT_REQUIRED')
    for obj in (worker, worker.model_runner):
        require(not any(name in vars(obj) for name in ('execute_model', '_prepare_inputs')), 'CORE_OVERRIDE_REFUSED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to('/evidence'), 'PRIVATE_GROUP_QUANT_INVENTORY_REQUIRED')
    require(isinstance(prompts, list) and len(prompts) == 8 and all(isinstance(row, list) and len(row) > 1
            and all(type(value) is int and value >= 0 for value in row) for row in prompts), 'EIGHT_FROZEN_PROMPTS_REQUIRED')
    session = SharedScaleGroupInventorySession(worker, worker.model_runner.model, root=root, build_sha256=build_sha256,
                                             folder=folder, prompts=prompts)
    session.install()
    worker._demoncat_shared_scale_inventory = session
    return session.report()
