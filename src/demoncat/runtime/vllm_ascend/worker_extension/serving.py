"""Owned hooks for a host-driven, standard vLLM serving transaction.

No Worker/runner methods are replaced. This adapter supports one fresh TP1
prefill at a time, on the inspected v1 eager path. The driver must use the
in-process engine and enqueue/advance/wait ordering described by serving_context.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

from .serving_context import (discover_logits_boundary, require, require_serial_eager,
                              snapshot_fresh_prefill, tensor_metadata)
from .call_binding import ForwardBinding


def complete_tensor(value):
    import torch
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and value.numel() > 0 and value.numel()*value.element_size() <= 512*1024*1024,
            'COMPLETE_BOUNDED_TENSOR_REQUIRED')
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256(json.dumps([str(cpu.dtype), list(cpu.shape)]).encode())
    digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
    return cpu, digest.hexdigest(), bool(torch.isfinite(cpu).all())


class ServingReplaySession:
    """The concrete reviewed factory accepted by Fault39Provider."""
    def __init__(self, worker, model, backend, native, owner, *, prompt_token_ids,
                 evidence_dir, collector=None, persist_evidence=True):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.worker, self.model, self.backend, self.native, self.owner = worker, model, backend, native, owner
        self.runner = worker.model_runner
        require(model is self.runner.model and model.training is False, 'LOADED_EVAL_MODEL_REQUIRED')
        require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0', 'HOST_DRIVEN_ENGINE_REQUIRED')
        self._require_runner()
        config = self.runner.vllm_config.model_config.hf_config
        boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor,
                                            actual_vocab_size=config.vocab_size)
        self.processor, self.head = boundary['processor'], boundary['lm_head']
        self.root_binding = ForwardBinding(self.model)
        self.logits_binding = ForwardBinding(self.processor)
        self.boundary_guard = boundary.get('branch_guard')
        self.boundary_report = boundary['report']
        require(isinstance(prompt_token_ids, list) and len(prompt_token_ids) > 1
                and all(type(x) is int and x >= 0 for x in prompt_token_ids), 'FROZEN_PROMPT_REQUIRED')
        self.prompt = list(prompt_token_ids)
        require(type(persist_evidence) is bool and (persist_evidence or collector is None),
                'NO_IMPLICIT_COLLECTOR_IN_MEMORY_SESSION')
        self.folder = None
        if persist_evidence:
            self.folder = Path(evidence_dir)
            require(self.folder.is_absolute() and self.folder.resolve().is_relative_to(Path('/evidence')),
                    'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
            self.folder.mkdir(parents=True, exist_ok=False)
        self.collector = collector
        self.handles = []
        self.installed = False
        self.expected = self.current = self.context = None
        self.logits_active = self.needs_collect = False
        self.collector_active = self.capture_step_executed = False
        self.hidden_digest = None
        self.rows = []
        self.owner_thread = threading.get_ident()

    def _require_runner(self):
        return require_serial_eager(self.runner)

    def _snapshot_prefill(self, actual, context):
        return snapshot_fresh_prefill(self.runner, self.model, actual, context,
            expected_prompt_token_ids=self.prompt, expected_internal_request_id=self.expected,
            seen_internal_request_ids=self.backend.requests)

    def _same_worker(self):
        require(self.worker.model_runner is self.runner and self.runner.model is self.model,
                'WORKER_MODEL_IDENTITY_CHANGED')
        require(threading.get_ident() == self.owner_thread, 'FOREIGN_MODEL_THREAD')

    def install(self):
        self._same_worker()
        require(not self.installed and not self.handles, 'SERVING_HOOK_ALREADY_INSTALLED')
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._root_pre, prepend=True, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._root_post, always_call=True, with_kwargs=True))
            self.handles.append(self.processor.register_forward_pre_hook(self._logits_pre, prepend=True, with_kwargs=True))
            self.handles.append(self.processor.register_forward_hook(self._logits_post, always_call=True, with_kwargs=True))
            self.installed = True
        except BaseException:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            raise

    def expect(self, request_id):
        self._same_worker()
        self.collect()
        require(self.installed and self.expected is None and self.current is None
                and not self.logits_active and self.backend.phase != 'complete', 'PREVIOUS_REQUEST_NOT_CLOSED')
        require(isinstance(request_id, str) and 0 < len(request_id) <= 256
                and request_id not in self.backend.requests, 'FRESH_ENQUEUED_ID_REQUIRED')
        require(not self.backend.failure or self.backend.recovering, 'FAILED_ATTEMPT_REQUIRES_RECOVERY')
        if self.collector is not None:
            self.collector_active = True  # start may partially install before raising
            self.capture_step_executed = False
            try:
                self.collector.start(model=self.model, rank_id=self.worker.rank)
            except BaseException as error:
                self._failure(error)
                raise
        self.expected = request_id

    def _failure(self, error):
        try:
            self.backend.fail(type(error).__name__ + ': ' + str(error))
        finally:
            if self.context is not None:
                self.context.__exit__(None, None, None)
                self.context = None
            self.logits_active = False
            self.current = self.expected = None
            self.needs_collect = True
            self._save()

    def _root_pre(self, module, args, kwargs):
        from vllm.forward_context import get_forward_context
        try:
            self._same_worker()
            require(self.installed and self.current is None and self.expected is not None,
                    'NO_EXPLICIT_ADVANCE_OR_CONCURRENT_REQUEST')
            actual = self.root_binding.bind(module, args, kwargs)
            snapshot = self._snapshot_prefill(actual, get_forward_context())
            token_cpu, token_hash, _ = complete_tensor(actual['input_ids'])
            position_cpu, position_hash, _ = complete_tensor(actual['positions'])
            require(token_cpu.tolist() == self.prompt and position_cpu.tolist() == list(range(len(self.prompt))),
                    'ACTUAL_DEVICE_INPUT_BYTES_NOT_FROZEN_PROMPT')
            snapshot['device_input_bytes_equality'] = 'COMPLETE_DEVICE_READBACK_VERIFIED'
            digest = hashlib.sha256(json.dumps([token_hash, position_hash]).encode()).hexdigest()
            self.current = dict(snapshot=snapshot, input_sha256=digest, phase=self.backend.phase,
                                actual_internal_request_id=self.expected, root_returned=False)
            self.capture_step_executed = True
            self.expected = None
            self.hidden_digest = None
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        if self.current is None:
            return
        try:
            self._same_worker()
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED')
            _, digest, finite = complete_tensor(output[-1:].contiguous())
            require(finite, 'HEALTHY_HIDDEN_STATES_NOT_FINITE')
            self.hidden_digest = digest
            self.current['root_returned'] = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_pre(self, module, args, kwargs):
        import torch
        import torch_npu
        try:
            self._same_worker()
            if self.boundary_guard is not None:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active,
                    'LOGITS_NOT_BOUND_TO_ONE_COMPLETE_FORWARD')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None,
                    'LOGITS_HEAD_OR_BIAS_CHANGED')
            hidden, weight = actual['hidden_states'], self.head.weight
            _, digest, finite = complete_tensor(hidden)
            require(finite and digest == self.hidden_digest, 'LOGITS_ROWS_NOT_LAST_FRESH_QUERY')
            require(hidden.ndim == weight.ndim == 2 and hidden.shape[0] == 1
                    and hidden.shape[1] == weight.shape[1]
                    and hidden.dtype == weight.dtype == torch.bfloat16
                    and torch_npu.get_npu_format(hidden) == torch_npu.get_npu_format(weight) == 2,
                    'FAULT39_BF16_ND_LOGITS_INPUT_REQUIRED')
            # Actual F.linear-style formal mat2 is the transposed logical weight.
            # Guard every byte range/geometry against the real native metadata;
            # any other lowering refuses before a native replacement is made.
            values = [hidden.untyped_storage().data_ptr(), hidden.untyped_storage().nbytes(),
                      weight.untyped_storage().data_ptr(), weight.untyped_storage().nbytes(),
                      int(hidden.shape[0]), int(hidden.shape[1]), int(weight.shape[0]),
                      hidden.storage_offset(), weight.storage_offset(),
                      *hidden.stride(), weight.stride(1), weight.stride(0)]
            self.current['live_inputs'] = dict(hidden=tensor_metadata(hidden), weight=tensor_metadata(weight),
                                               native_guard_values=values)
            self.context = self.native.context(self.owner, [self.current['actual_internal_request_id']])
            self.context.__enter__()
            self.native.bind_serving_inputs(self.owner, values)
            self.backend.begin(self.current['actual_internal_request_id'], self.current['input_sha256'])
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        if not self.logits_active:
            return
        try:
            self._same_worker()
            require(output is not None and list(output.shape) == self.boundary_report['output_contract'],
                    'COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED')
            cpu, digest, finite = complete_tensor(output)
            self.current.update(model_output_sha256=digest, model_output_finite=finite,
                                output_shape=list(cpu.shape), top1=int(cpu.float().argmax(dim=-1).item()))
            self.backend.end(digest, model_output_finite=finite)
            self.rows.append(self.current)
            self.current = None
            self.needs_collect = True
            self._save()
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            if self.context is not None:
                self.context.__exit__(None, None, None)
                self.context = None
            self.logits_active = False

    def collect(self, *, force=False):
        self._same_worker()
        if self.needs_collect or force:
            if self.collector is not None and self.collector_active:
                self.collector.stop()
                if self.capture_step_executed:
                    self.collector.step()
                self.collector_active = False
            self.needs_collect = False

    def reset_for_recovery(self):
        self._same_worker()
        self.collect()
        require(self.installed and not self.logits_active and self.current is None,
                'ACTIVE_FORWARD_CANNOT_RECOVER')
        self.expected = None

    def close(self):
        self._same_worker()
        self.collect(force=True)
        if (self.current is not None or self.logits_active or self.expected is not None) and not self.backend.fault_started:
            # RPC runs on the same model thread only after normal execution has
            # returned/raised. A pre-fault half-transaction has no injected output
            # to qualify; cancel it after an explicit native drain, never as PASS.
            self.backend.fail('HOST_ABORT_BEFORE_FAULT')
            require(self.backend.drain_status == 'DISABLED_CONFIRMED', 'CANCEL_DRAIN_NOT_CONFIRMED')
            if self.context is not None:
                self.context.__exit__(None, None, None)
                self.context = None
            self.current = self.expected = None
            self.logits_active = False
        require(not self.logits_active and self.current is None, 'SERVING_REQUEST_STILL_ACTIVE')
        require(not self.backend.fault_started or self.backend.recovered, 'OFF_RECOVERY_REQUIRED_BEFORE_DETACH')
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.installed = False
        self.expected = None
        self._save()

    def _save(self):
        if self.folder is None:
            return
        value = dict(boundary=self.boundary_report, requests=self.rows,
                     qualification=self.backend.query()['qualification'], native_phases=self.backend.rows)
        (self.folder / 'serving_receipt.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    def report(self):
        self.collect()
        return dict(installed=self.installed, boundary=self.boundary_report,
                    completed_requests=len(self.rows), expected_request_id=self.expected,
                    actual_request_ids=[row['actual_internal_request_id'] for row in self.rows],
                    complete_output_hashes=[row['model_output_sha256'] for row in self.rows],
                    receipt=str(self.folder / 'serving_receipt.json') if self.folder is not None else None)


def install_fault39_serving_binding(worker, *, prompt_token_ids, evidence_dir, collect_statistics=False,
                                   baseline_control=False):
    """Called by the trusted host through standard collective_rpc, once per load.

    This supplies a reviewed deployment to the existing runtime registry. It
    neither registers a fault owner nor chooses a layer supplied by a client.
    """
    from demoncat_fault39.provider import CtypesProvider
    from fault_injection.providers import register_runtime
    from .providers.fault39 import Fault39ServingReplay
    require(not hasattr(worker, '_demoncat_serving_factory'), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
            'READY_UNQUARANTINED_WORKER_REQUIRED')
    native = CtypesProvider.attach_preloaded()
    snapshot = native.snapshot()
    require(snapshot.get('runtime_identity') == 'MATCHED_BUILD' and snapshot.get('runtime_ready') is True,
            'MATCHED_PRELOADED_BUILD_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),
            'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    collector = None
    if collect_statistics:
        import msprobe
        from msprobe.pytorch import PrecisionDebugger
        capture = folder / 'statistics'
        capture.mkdir(parents=True, exist_ok=False)
        config = dict(task='statistics', level='mix', precision='high', rank=[], step=[2, 6],
            dump_path=str(capture/'observation'), async_dump=False,
            statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
        config_path = capture/'config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        package = Path(msprobe.__file__).resolve().parent
        files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}
        (capture/'collector_identity.json').write_text(json.dumps(dict(package=str(package), files=files,
            edited_by_experiment=False, steps={'2':'reference', '6':'comparison_phase'},
            native_replacement_below_python_input_observation=True), indent=2) + '\n')
        collector = PrecisionDebugger(config_path=str(config_path))
    model = worker.model_runner.model
    def make_session(w, m, b, n, o):
        try:
            return ServingReplaySession(w, m, b, n, o, prompt_token_ids=prompt_token_ids,
                                        evidence_dir=folder/'runtime', collector=collector)
        except BaseException as error:
            import traceback
            folder.mkdir(parents=True, exist_ok=True)
            (folder/'binding_failure.json').write_text(json.dumps(
                dict(error=repr(error), traceback=traceback.format_exc()), indent=2) + '\n')
            raise
    def factory(actual_model):
        require(actual_model is model and worker.model_runner.model is model, 'MODEL_LOAD_CHANGED')
        return Fault39ServingReplay(snapshot['build_profile_sha256'], make_session, baseline_control)
    register_runtime('39.matmul_range', factory)
    worker._demoncat_serving_factory = factory
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED', build_profile_sha256=snapshot['build_profile_sha256'],
                statistics=bool(collector), baseline_control=baseline_control,
                core_methods_replaced=False, workers_created=0)


def remove_fault39_serving_binding(worker):
    from fault_injection.providers import unregister_runtime
    factory = getattr(worker, '_demoncat_serving_factory', None)
    require(factory is not None, 'SERVING_DEPLOYMENT_NOT_BOUND')
    # Host must unregister/detach the common Worker lease first. Registry removal
    # alone cannot release hooks or native ownership.
    from demoncat_fault39.provider import CtypesProvider
    state = CtypesProvider.attach_preloaded().snapshot()
    require(state.get('owner') == '' and state.get('enabled') is False
            and state.get('inflight') == state.get('leases') == 0, 'OWNER_CLEAN_REQUIRED')
    unregister_runtime('39.matmul_range', factory)
    del worker._demoncat_serving_factory
    return dict(status='SERVING_DEPLOYMENT_REMOVED')
