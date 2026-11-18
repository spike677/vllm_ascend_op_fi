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

    def make_inventory(self, identity):
        from .shared_scale_group_inventory import SharedScaleGroupInventory
        return SharedScaleGroupInventory(self.model, self.candidates, worker_identity=identity,
            native=self.native_reader, folder=self.folder / 'inventory', row_lineage=self.row_lineage,
            row_transfer=self.projection_rows)

    def inventory_worker_identity(self):
        return self.worker.demoncat_worker_identity()

    def __init__(self, *args, **kwargs):
        # Reuse actual scheduler/embedding/logits hooks but bypass the legacy
        # inventory constructor and its C1 Native loader entirely.
        QuantServingSession.__init__(self, *args, **kwargs, baseline_control=True,
                                     collector=None, capture_scope='first_decode')
        from fault_injection.deployments.quant_group_inventory import GroupQuantJournal
        from .quant_row_provenance import RequestRowProvenance
        import torch_npu
        self.native_reader = GroupQuantJournal(self.runtime.binding, device=int(torch_npu.npu.current_device()))
        self.row_observer = RequestRowProvenance()
        identity = self.inventory_worker_identity()
        require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined')
                and identity['worker']['rank'] == 0,
                'ONE_ACTUAL_READY_TP1_WORKER_REQUIRED')
        self.inventory = self.make_inventory(identity['worker'])
        self.norm_evidence = None
        self.source_rows = self.metadata_rows = None
        self.source_row_refusal = None
        try:
            from .quant_source_rows import SourceBoundRows
            from .quant_mla_metadata_rows import MlaMetadataRows
            source_rows = SourceBoundRows.from_runtime(self.row_observer, self.model)
            metadata_rows = MlaMetadataRows.from_runtime(self.runner)
            source_rows.metadata_producers = metadata_rows
            self.source_rows, self.metadata_rows = source_rows, metadata_rows
            self.row_observer.boundaries = source_rows
        except Exception as error:
            # An unavailable adapter keeps the existing Native census and its
            # unverified rows. It cannot remove a candidate or qualify a set.
            self.source_row_refusal = type(error).__name__ + ':' + str(error)

    def begin_observation(self):
        super().begin_observation()
        if self.metadata_rows is not None:
            self.metadata_rows.begin(tuple(self.expected))

    def prepare_source_rows(self):
        if self.metadata_rows is not None:
            self.metadata_rows.close(scope=self.active['decode'], actual_positions=self.actual['positions'])

    def start_source_rows(self):
        if self.source_rows is not None:
            self.source_rows.begin(self.active['decode'])

    def close(self, *, abort=False):
        if self.metadata_rows is not None and self.metadata_rows.active:
            self.metadata_rows.close()
        super().close(abort=abort)
        if self.metadata_rows is not None:
            self.metadata_rows.release()

    def report(self):
        value = super().report()
        inventory = self.inventory.report()
        value.update(schema='demoncat.shared_scale.group_serving_inventory.v1',
            actual_native_prepare_calls=inventory['actual_native_prepare_calls'],
            actual_native_execute_calls=inventory['actual_native_execute_calls'],
            group_owner_installed=False, legacy_library_loaded=False,
            source_row_adapter=dict(enabled=self.source_rows is not None, refusal=self.source_row_refusal),
            metadata_producers=self.metadata_rows.report() if self.metadata_rows is not None else None,
            owned_source_profile_callbacks=sum(int(value is not None and not value.profile_released)
                for value in (self.source_rows, self.metadata_rows)),
            owned_early_dispatch_modes=int(self.metadata_rows is not None and self.metadata_rows.mode is not None),
            owned_nested_dispatch_modes=0 if self.source_rows is None else len(self.source_rows.owned_modes),
            observer_cleanup_pending=any(value is not None and value.active for value in
                (self.source_rows, self.metadata_rows)) or self.row_observer.active,
            owned_live_metadata_records=0 if self.metadata_rows is None else self.metadata_rows.report()['live_producer_records'],
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
