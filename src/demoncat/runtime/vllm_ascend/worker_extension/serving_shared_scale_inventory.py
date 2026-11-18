"""Two actual B8 first decodes; full native census without row-count inference."""
import hashlib
import inspect
import json
import os
from pathlib import Path

from .serving_quant import QuantServingSession
from .serving_context import require


class SharedScaleInventorySession(QuantServingSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, baseline_control=True, collector=None, capture_scope='first_decode')
        from fault_injection.deployments.dynamic_quant_inventory import NativeDynamicQuantInventory
        from .shared_scale_inventory import SharedScaleInventory
        self.native_reader = NativeDynamicQuantInventory(self.runtime)
        from .quant_row_provenance import RequestRowProvenance
        self.row_observer = RequestRowProvenance()
        identity = self.worker.demoncat_worker_identity()
        require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'), 'READY_ACTUAL_WORKER_REQUIRED')
        self.inventory = SharedScaleInventory(self.model, self.candidates, worker_identity=identity['worker'],
            native=self.native_reader, folder=self.folder / 'inventory', row_lineage=self.row_lineage,
            row_transfer=self.projection_rows)
        self.norm_evidence = None

    def install(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
        from vllm.model_executor.layers.layernorm import RMSNorm
        require(not self.installed and not any(self.state()[k] for k in ('armed', 'pending', 'executing', 'owned')),
                'NATIVE_QUANT_INVENTORY_REQUIRES_IDLE')
        self.native_reader.state()
        self.inventory.install()
        try:
            self.handles.append(self.model.register_forward_pre_hook(self.pre, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self.post, always_call=True, with_kwargs=True))
            self.handles.append(self.processor.register_forward_hook(self.logits, always_call=True, with_kwargs=True))
            for _, module in self.model.named_modules():
                if isinstance(module, VocabParallelEmbedding) and not isinstance(module, ParallelLMHead):
                    self.handles.append(module.register_forward_hook(self.embed, with_kwargs=True))
                if isinstance(module, RMSNorm):
                    self.handles.append(module.register_forward_hook(self.norm, with_kwargs=True))
            self.installed = True
        except BaseException:
            self.close(abort=True)
            raise

    def expect(self, request_ids):
        self.collect()
        require(self.installed and not self.error and self.expected is None and len(self.rows) < 2,
                'TWO_BOUNDED_IDLE_QUANT_INVENTORY_GENERATIONS_REQUIRED')
        require(isinstance(request_ids, list) and len(request_ids) == len(set(request_ids)) == 8
                and all(isinstance(key, str) and key and key not in self.seen for key in request_ids),
                'EIGHT_FRESH_ACTUAL_REQUEST_IDS_REQUIRED')
        self.expected = {key: dict(logical_id=index, prompt=self.prompts[index]) for index, key in enumerate(request_ids)}
        self.decode_result = self.prefill_result = None
        self.forward_count = 0
        self.capture_started = False

    def begin_observation(self):
        require(not self.capture_started and not any(self.state()[k] for k in ('armed', 'pending', 'executing', 'owned')),
                'IDLE_NATIVE_QUANT_OBSERVER_REQUIRED')
        self.capture_started = True

    def pre(self, module, args, kwargs):
        super().pre(module, args, kwargs)
        self.norm_evidence = None
        self.active['phase'] = 'inventory_' + str(len(self.rows) + 1)
        if self.active.get('decode'):
            from .serving import complete_tensor
            _, input_sha, _ = complete_tensor(self.actual['input_ids'])
            _, positions_sha, _ = complete_tensor(self.actual['positions'])
            digest = hashlib.sha256(json.dumps([input_sha, positions_sha]).encode()).hexdigest()
            self.inventory.begin(self.active['decode']['request_ids'][0], digest,
                request_ids=tuple(self.active['decode']['request_ids']), row_mapping=self.active['decode'])
            self.prepare_source_rows()
            self.row_observer.begin(self.active['decode']['request_ids'][0])
            self.start_source_rows()
            for _, owner in self.model.named_modules():
                for name, tensor in owner._parameters.items():
                    if tensor is not None and not tensor.requires_grad:
                        self.row_observer.register_broadcast_parameter(owner, name, tensor, kind='_parameters')

    def prepare_source_rows(self):
        """Optional early-producer observer handoff; legacy observation is unchanged."""

    def start_source_rows(self):
        """Optional exact-source graph adapter after the owned mode is entered."""

    def norm(self, module, args, kwargs, output):
        prior = self.normalized
        super().norm(module, args, kwargs, output)
        if prior is None and self.normalized is not None:
            from .serving20 import live_tensor
            source = Path(inspect.getsourcefile(type(module)))
            self.norm_evidence = dict(module_object_identity=id(module), module_class=type(module).__module__ + '.' + type(module).__name__,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                embedding=live_tensor(self.embedding), normalized=live_tensor(self.normalized))
            self.row_observer.seed(self.normalized, dict(verified=True,
                source='ACTUAL_SCHEDULER_EMBEDDING_RMSNORM_SAME_TENSOR_PROJECTION',
                actual_rows=self.active['decode'], norm=self.norm_evidence))

    def row_lineage(self, x):
        return self.row_observer.query(x)

    def projection_rows(self, frame, result):
        rows = [self.inventory.public_calls[index] for index in frame['row']['api_call_indices']]
        dynamic = [row for row in rows if row['operator'] == 'npu_dynamic_quant']
        if len(dynamic) != 1 or not dynamic[0].get('request_row_lineage', {}).get('verified'):
            return self.row_observer.query(result)
        binding = dynamic[0].get('independent_native', {}).get('binding', {})
        if not binding.get('complete_formals'):
            return self.row_observer.query(result)
        return self.row_observer.projection(frame['input'], result, evidence=dict(
            complete_native_dynamic_formals=True, owned_public_qmm_chain=frame['row']['public_chain_verified'],
            actual_native_dynamic_call=binding, native_build=self.runtime.build_identity,
            public_api_indices=frame['row']['api_call_indices'], module_object_identity=frame['row']['module_object_identity'],
            method_source_sha256=frame['row']['method_source_sha256'], weight_object_identity=frame['row']['weight_object_identity']))

    def logits(self, module, args, kwargs, output):
        decode = self.active is not None and bool(self.active.get('decode'))
        if self.row_observer.active:
            self.inventory.row_trace = self.row_observer.end()
        super().logits(module, args, kwargs, output)
        if decode:
            try:
                self.inventory.end(self.decode_result['logits_sha256'], model_output_finite=True)
            except BaseException as error:
                self.error = repr(error)
                self.inventory.fail(self.error)
                self.save()
                raise

    def collect(self):
        if self.expected is None or self.decode_result is None:
            return
        require(not self.error and self.active is None and self.prefill_result is not None and self.forward_count == 2,
                'COMPLETE_ACTUAL_PREFILL_AND_FIRST_DECODE_REQUIRED')
        require(not any(self.state()[k] for k in ('armed', 'pending', 'executing', 'owned')),
                'READONLY_QUANT_NATIVE_STATE_NOT_IDLE')
        if self.rows:
            for phase in ('prefill', 'decode'):
                require(self.rows[0][phase]['logits_sha256'] == getattr(self, phase + '_result')['logits_sha256'],
                        'COMPLETE_HEALTHY_BATCH_LOGITS_CHANGED:' + phase)
        self.rows.append(dict(phase='inventory_' + str(len(self.rows) + 1), prefill=self.prefill_result, decode=self.decode_result))
        self.seen.extend(self.expected)
        self.expected = None
        self.recovered = len(self.rows) == 2
        self.save()

    def close(self, *, abort=False):
        if self.row_observer.active:
            self.inventory.row_trace = self.row_observer.end()
        if not abort:
            self.collect()
        if self.inventory.active:
            self.inventory.fail('HOST_ABORT_READONLY_QUANT_INVENTORY')
        self.inventory.close()
        super().close(abort=abort)

    def report(self):
        report = self.inventory.report()
        return dict(schema='demoncat.shared_scale.serving_inventory.v1', status='READONLY_NATIVE_INVENTORY_NOT_QUALIFICATION',
            error=self.error, recovered=self.recovered, installed=self.installed, actual_generations=len(self.rows), rows=self.rows,
            structural_candidates=report['structural_candidate_count'], actual_projection_calls=report['actual_projection_calls'],
            actual_public_calls=report['actual_public_api_calls'], actual_native_calls=report['actual_native_calls'],
            native_public_binding_complete=report['native_public_binding_complete'],
            request_row_proven_calls=report['request_row_proven_calls'],
            row_operator_observation_complete=bool(report['native_forwards']) and all(
                row.get('row_provenance') is not None and row['row_provenance']['complete'] for row in report['native_forwards']),
            row_operator_unknown_edges=[len((row.get('row_provenance') or {}).get('unknown_edges', []))
                for row in report['native_forwards']],
            native_qualified_targets=0, c1=False, c2=False, c3=False, native_faults=0 if not self.error else None,
            owned_hooks=len(self.handles), owned_method_patches=report['owned_method_patches'],
            owned_api_patches=report['owned_api_patches'], owned_row_dispatch_modes=int(self.row_observer.active),
            native_build=self.runtime.build_identity,
            complete_healthy_replays=report['complete_healthy_replays'],
            scope='ALL_NATIVE_DYNAMIC_QUANT_V2_IN_TWO_ACTUAL_B8_FIRST_DECODES',
            inventory=str(self.inventory.folder / 'shared_scale_inventory.json'))

    def save(self):
        (self.folder / 'serving.json').write_text(json.dumps(self.report(), indent=2, allow_nan=False) + '\n')


def install_shared_scale_inventory(worker, *, prompts, root, build_sha256, evidence_dir):
    require(not hasattr(worker, '_demoncat_shared_scale_inventory'), 'EXCLUSIVE_SHARED_SCALE_INVENTORY_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
            'HOST_DRIVEN_SYNCHRONOUS_NATIVE_QUANT_REQUIRED')
    for obj in (worker, worker.model_runner):
        require(not any(name in vars(obj) for name in ('execute_model', '_prepare_inputs')), 'CORE_OVERRIDE_REFUSED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to('/evidence'), 'PRIVATE_QUANT_INVENTORY_REQUIRED')
    require(isinstance(prompts, list) and len(prompts) == 8 and all(isinstance(row, list) and len(row) > 1
            and all(type(value) is int and value >= 0 for value in row) for row in prompts), 'EIGHT_FROZEN_PROMPTS_REQUIRED')
    session = SharedScaleInventorySession(worker, worker.model_runner.model, root=root, build_sha256=build_sha256,
        folder=folder, prompts=prompts)
    session.install()
    worker._demoncat_shared_scale_inventory = session
    return session.report()


def remove_shared_scale_inventory(worker):
    session = worker._demoncat_shared_scale_inventory
    session.close(abort=not session.recovered)
    report = session.report()
    require(not any(report[key] for key in ('owned_hooks', 'owned_method_patches', 'owned_api_patches', 'owned_row_dispatch_modes')),
            'SHARED_SCALE_INVENTORY_STILL_OWNS_HOOKS')
    require(not report.get('owned_source_profile_callbacks', 0) and not report.get('owned_early_dispatch_modes', 0)
            and not report.get('owned_nested_dispatch_modes', 0) and not report.get('observer_cleanup_pending', False)
            and not report.get('owned_scheduler_input_refs', 0)
            and not report.get('owned_live_metadata_records', 0),
            'SHARED_SCALE_INVENTORY_STILL_OWNS_SOURCE_OBSERVERS')
    del worker._demoncat_shared_scale_inventory
    return report
