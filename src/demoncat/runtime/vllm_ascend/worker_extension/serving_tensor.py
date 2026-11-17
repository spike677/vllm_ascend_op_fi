"""Trusted standard-Worker RMSNorm group deployment using real scheduled requests.

Only owned hooks/modes are installed. Model/runner methods and scheduler inputs
are never replaced. The final norm's actual logits reader stays in the owned
serving span, including materialized last-token selection after model.forward.
"""
import hashlib
import inspect
import threading
import uuid
from pathlib import Path

from .call_binding import ForwardBinding
from .serving_context import require, require_serial_eager, snapshot_fresh_prefill, discover_logits_boundary
from .providers.tensor_modules import _capture, _identity, _same
from .providers.tensor_natural import NaturalTensorGroupProvider, _sha


class ServingTensorGroup:
    def __init__(self, worker, *, prompt, weight_manifest, workload_sha256, environment_sha256,
                 _simulation=False, _synchronize=None):
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        self.worker, self.runner, self.model = worker, worker.model_runner, worker.model_runner.model
        require_serial_eager(self.runner)
        identity = worker.demoncat_worker_identity()
        require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
                'READY_ACTUAL_WORKER_IDENTITY_REQUIRED')
        self.worker_identity = dict(identity['worker'])
        boundary = discover_logits_boundary(self.model, actual_logits_processor_type=LogitsProcessor,
            actual_vocab_size=self.runner.vllm_config.model_config.hf_config.vocab_size)
        self.processor = boundary['processor']; self.boundary_guard = boundary.get('branch_guard')
        self.boundary = boundary['report']
        sources = {}
        for module in self.model.modules():
            try:
                path = Path(inspect.getsourcefile(type(module))).resolve()
                if path.is_file():
                    sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, TypeError):
                pass
        def no_direct_replay(*args):
            raise RuntimeError('STANDARD_WORKER_REQUIRES_REAL_ENQUEUED_FORWARD')
        import torch
        require(type(_simulation) is bool and (_synchronize is None or _simulation),
                'SYNCHRONIZER_OVERRIDE_IS_CPU_CONTRACT_ONLY')
        synchronize = _synchronize if _simulation else torch.npu.synchronize
        require(callable(synchronize), 'ACTUAL_SYNCHRONIZER_REQUIRED')
        self.provider = NaturalTensorGroupProvider(self.model, '18.api_half', replay=no_direct_replay,
            synchronize=synchronize, workload_sha256=workload_sha256, simulation=_simulation,
            environment_sha256=environment_sha256, model_weights_sha256=_sha(weight_manifest),
            model_source_sha256=_sha(sources), worker_identity=identity['worker'])
        self.provenance = dict(worker_identity=identity, weights=weight_manifest, model_sources=sources,
                               logits_boundary=self.boundary, standard_worker=not _simulation,
                               simulation=_simulation)
        self.binding = ForwardBinding(self.model)
        self.prompt = list(prompt)
        self.thread = threading.get_ident()
        self.seen, self.phases, self.handles = [], [], []
        self._callbacks = {}
        self.expected = self.current = self.adapter = self.observer = self.root_post_handle = None
        self.provider.owned_host_hooks = lambda: len(self.handles) + int(self.root_post_handle is not None)
        self._install_context()

    def _install_context(self):
        require(not self.handles, 'SERVING_CONTEXT_ALREADY_INSTALLED')
        try:
            callback = self._pre
            handle = self.model.register_forward_pre_hook(callback, with_kwargs=True, prepend=True)
            self.handles.append(handle); self._callbacks[handle.id] = callback
            callback = self._logits
            handle = self.processor.register_forward_hook(callback, with_kwargs=True, always_call=True)
            self.handles.append(handle); self._callbacks[handle.id] = callback
        except BaseException:
            for handle in reversed(self.handles):
                self._remove_hook(handle)
            self.handles.clear()
            raise

    def _remove_hook(self, handle):
        hooks = handle.hooks_dict_ref()
        require(hooks is not None and (handle.id not in hooks or
            hooks[handle.id] is self._callbacks.get(handle.id)), 'SERVING_TENSOR_HOOK_OWNERSHIP_CHANGED')
        handle.remove(); self._callbacks.pop(handle.id, None)

    def _context(self, *, verify_targets=True):
        require(threading.get_ident() == self.thread, 'SERVING_TENSOR_THREAD_CHANGED')
        require(self.worker.model_runner is self.runner and self.runner.model is self.model,
                'SERVING_TENSOR_MODEL_LIFETIME_CHANGED')
        for handle in self.handles + ([self.root_post_handle] if self.root_post_handle is not None else []):
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self._callbacks.get(handle.id),
                    'SERVING_TENSOR_HOOK_OWNERSHIP_CHANGED')
        if not verify_targets:
            # Cleanup still verifies owned resources but must not depend on a
            # changed target implementation being acceptable for inference.
            return
        identity = self.worker.demoncat_worker_identity()
        require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined')
                and identity.get('worker') == self.worker_identity, 'SERVING_TENSOR_WORKER_IDENTITY_CHANGED')
        require(not self.model.training, 'INFERENCE_MODEL_REQUIRED')
        modules = {path or '<root>': module for path, module in self.model.named_modules()}
        for path, (module, source) in self.provider.target_sources.items():
            require(modules.get(path) is module, 'SERVING_TENSOR_TARGET_MODEL_CHANGED:' + path)
            self.provider.target_guards[path].verify()

    def expect(self, request_id, phase, arm=None):
        from fault_injection.tensor_observer import NaturalTensorObserver
        from fault_injection.rmsnorm_return import RMSNormReturnSession
        self._context()
        require(self.expected is None and self.current is None and self.adapter is None,
                'PREVIOUS_TENSOR_REQUEST_NOT_COLLECTED')
        require(isinstance(request_id, str) and request_id and request_id not in self.seen,
                'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
        if not self.handles:
            self._install_context()
        self.expected = request_id
        self.phase = phase
        self.provider.calls = []
        self.provider.current_invocation = 'worker-forward:' + uuid.uuid4().hex
        self.provider.current_request_ids = (request_id,)
        self.provider._running = True
        if phase.endswith('off'):
            self.provider.enabled = False
        if arm is not None:
            self.adapter = RMSNormReturnSession(self.model, arm=arm,
                admission=self.provider._admission, observe_post_root_consumers=True,
                simulation=self.provider.simulation)
            self.adapter.__enter__()
            self.provider.active_adapter = self.adapter
        self.observer = NaturalTensorObserver(self.model,
            on_module_enter=self.provider._module_enter,
            on_operator_enter=self.provider._operator_enter).__enter__()
        callback = self._post
        self.root_post_handle = self.model.register_forward_hook(callback, with_kwargs=True, always_call=True)
        self._callbacks[self.root_post_handle.id] = callback
        return dict(expected=request_id, phase=phase, arm=arm)

    def _pre(self, module, args, kwargs):
        from vllm.forward_context import get_forward_context
        self._context()
        require(self.expected is not None and self.current is None, 'UNSCHEDULED_OR_CONCURRENT_TENSOR_FORWARD')
        actual = self.binding.bind(module, args, kwargs)
        scheduler = snapshot_fresh_prefill(self.runner, self.model, actual, get_forward_context(),
            expected_prompt_token_ids=self.prompt, expected_internal_request_id=self.expected,
            seen_internal_request_ids=self.seen)
        if self.boundary_guard:
            self.boundary_guard()
        self.current = dict(phase=self.phase, request_id=self.expected, scheduler=scheduler,
                            input=_capture((args, kwargs)), root_returned=False, logits_returned=False)

    def _post(self, module, args, kwargs, output):
        require(self.current is not None and output is not None, 'ACTUAL_TENSOR_ROOT_RETURN_REQUIRED')
        self.current.update(hidden=_capture(output), root_returned=True)

    def _logits(self, module, args, kwargs, output):
        if self.current is None:
            return
        require(self.current['root_returned'] and not self.current['logits_returned'] and output is not None,
                'ACTUAL_LOGITS_AFTER_ROOT_REQUIRED')
        self.current.update(logits=_capture(output), logits_returned=True)

    def collect(self):
        self._context()
        require(self.current is not None and self.current['logits_returned'], 'TENSOR_REQUEST_NOT_COMPLETED')
        self.provider.synchronize()
        self._remove_hook(self.root_post_handle); self.root_post_handle = None
        self.observer.close()
        if self.adapter:
            self.adapter.close()
        input_ = self.current['input']
        output = dict(hidden=self.current['hidden'], logits=self.current['logits'])
        row = dict(phase=self.phase, request_id=self.expected, scheduler=self.current['scheduler'],
            invocation=self.provider.current_invocation, input=_identity(input_), output=_identity(output),
            independent=self.observer.report(), mechanism=self.adapter.report() if self.adapter else None,
            calls=[{key: value for key, value in call.items() if key != 'decision'} for call in self.provider.calls],
            cleanup=self.adapter.runtime_state() if self.adapter else dict(cleanup_confirmed=True,
                hook_owned=False, inflight=0, leases=0), logits_boundary_returned=True)
        require(row['independent']['complete'], 'WORKER_INDEPENDENT_CENSUS_INCOMPLETE')
        self.provider.rows.append(row)
        self.phases.append((input_, output, row))
        if self.provider.group is not None:
            self.provider._finish_calls(row)
        if self.phase.endswith('off'):
            require(_same(input_, self.provider.reference_input) and _same(output, self.provider.reference_output),
                    'WORKER_FULL_HIDDEN_LOGITS_OFF_NOT_EXACT')
            self.provider.off_row = row
        self.seen.append(self.expected)
        self.expected = self.current = self.adapter = self.observer = None
        self.provider.active_adapter = None; self.provider._running = False
        return row

    def qualify(self):
        require(len(self.phases) == 4, 'FOUR_ACTUAL_HEALTHY_REQUESTS_REQUIRED')
        return self.provider.qualify_observations(self.phases)

    def finish_group(self):
        require(self.expected is None and self.current is None and self.adapter is None,
                'SERVING_TENSOR_REQUEST_PENDING')
        for handle in reversed(self.handles):
            self._remove_hook(handle)
        self.handles.clear()
        return self.provider.close_group()

    def abort(self):
        from fault_injection.coverage import Outcome
        self._context(verify_targets=False)
        if self.root_post_handle is not None:
            self._remove_hook(self.root_post_handle); self.root_post_handle = None
        if self.observer is not None:
            self.observer.close(); self.observer = None
        if self.adapter is not None:
            self.adapter.close(); self.adapter = None
        self.provider.active_adapter = None; self.provider._running = False
        self.provider.enabled = False
        if self.provider.group is not None:
            for call in self.provider.calls:
                if 'decision' in call:
                    try:
                        self.provider.group.finish(call['decision'], Outcome.FAILED, original_calls=0)
                    except BaseException:
                        pass
        self.expected = self.current = None
        cleanup = self.close()
        cleanup['group_recovery'] = (self.provider.group.report() if self.provider.group is not None else None)
        cleanup['failed_request_recovery_claimed'] = False
        return cleanup

    def close(self):
        self._context(verify_targets=False)
        require(self.expected is None and self.current is None and self.adapter is None,
                'SERVING_TENSOR_PHASE_PENDING')
        for handle in reversed(self.handles):
            self._remove_hook(handle)
        self.handles.clear()
        return dict(owned_serving_hooks=0, active_adapter=False, independent_observer=False,
                    standard_worker=not self.provider.simulation, requests=self.seen)
