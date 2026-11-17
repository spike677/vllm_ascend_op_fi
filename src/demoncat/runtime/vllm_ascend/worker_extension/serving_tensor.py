"""Trusted standard-Worker RMSNorm group deployment using real scheduled requests.

Only owned hooks/modes are installed. Model/runner methods and scheduler inputs
are never replaced. The final norm's actual logits reader stays in the owned
serving span, including materialized last-token selection after model.forward.
"""
import hashlib
from contextlib import nullcontext
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
        self.statistics_capture = None
        self.statistics_arm = None
        self.statistics_row = None
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
        collecting = phase == 'statistics_request'
        if collecting:
            require(self.statistics_capture is not None and self.statistics_row is None
                    and len(self.seen) == 4 and arm == self.statistics_arm,
                    'SAME_QUALIFIED_FIFTH_REQUEST_STATISTICS_REQUIRED')
        self.provider._statistics_arm = arm if collecting else None
        if phase.endswith('off'):
            self.provider.enabled = False
        if arm is not None:
            self.adapter = RMSNormReturnSession(self.model, arm=arm,
                admission=self.provider._admission, observe_post_root_consumers=True,
                simulation=self.provider.simulation,
                diagnostics_context=self._diagnostics)
            self.adapter.__enter__()
            self.provider.active_adapter = self.adapter
        self.observer = NaturalTensorObserver(self.model,
            on_module_enter=self.provider._module_enter,
            on_operator_enter=self.provider._operator_enter,
            diagnostics_context=self._diagnostics).__enter__()
        callback = self._post
        self.root_post_handle = self.model.register_forward_hook(callback, with_kwargs=True, always_call=True)
        self._callbacks[self.root_post_handle.id] = callback
        if collecting:
            self.statistics_capture.begin(self.model, 'statistics_' + arm)
        return dict(expected=request_id, phase=phase, arm=arm)

    def _diagnostics(self):
        return self.statistics_capture.diagnostics() if self.statistics_capture is not None else nullcontext()

    def configure_statistics(self, capture, arm):
        self._context()
        require(arm in ('baseline', 'fault') and self.statistics_capture is None
                and self.provider.candidates is not None and self.provider.group is None
                and len(self.seen) == 4 and self.expected is None,
                'QUALIFIED_FRESH_WORKER_STATISTICS_REQUIRED')
        self.statistics_capture, self.statistics_arm = capture, arm
        return dict(arm=arm, managed_request_ordinal=5, full_request_api_and_module_capture=True,
                    scheduler_and_sampler_not_filtered=True, grants_coverage=False)

    def _pre(self, module, args, kwargs):
        from vllm.forward_context import get_forward_context
        with self._diagnostics():
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
        with self._diagnostics():
            require(self.current is not None and output is not None, 'ACTUAL_TENSOR_ROOT_RETURN_REQUIRED')
            self.current.update(hidden=_capture(output), root_returned=True)

    def _logits(self, module, args, kwargs, output):
        if self.current is None:
            return
        with self._diagnostics():
            require(self.current['root_returned'] and not self.current['logits_returned'] and output is not None,
                    'ACTUAL_LOGITS_AFTER_ROOT_REQUIRED')
            self.current.update(logits=_capture(output), logits_returned=True)

    def collect(self):
        require(self.current is not None and self.current['logits_returned'], 'TENSOR_REQUEST_NOT_COMPLETED')
        if self.statistics_capture is not None and self.statistics_capture.active:
            self.statistics_capture.end(completed=True)
        self._context()
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
        if self.phase == 'statistics_request':
            selected = set(self.provider.selected)
            events = row['mechanism']['events']
            require({tuple(call['key']) for call in row['calls']} == selected
                    and len(row['calls']) == len(selected) and len(events) == len(selected),
                    'STATISTICS_COMPLETE_INDEPENDENT_DENOMINATOR_REQUIRED')
            require(_same(input_, self.provider.reference_input), 'STATISTICS_ACTUAL_WORKER_INPUT_CHANGED')
            if self.statistics_arm == 'baseline':
                require(_same(output, self.provider.reference_output), 'STATISTICS_BASELINE_NOT_EXACT')
            require(all(event.get('original_execution_verified') and event.get('consumers')
                        and event.get('return_readback_exact')
                        and event.get('applied') == (self.statistics_arm == 'fault')
                        and (self.statistics_arm == 'baseline' or event.get('output_changed')) for event in events),
                    'STATISTICS_ACTUAL_RETURN_EFFECT_AND_CONSUMER_REQUIRED')
            self.statistics_row = dict(arm=self.statistics_arm, observed_targets=len(events),
                local_hits=sum(bool(event['applied']) for event in events),
                input_exact=True, hidden_logits_changed=not _same(output, self.provider.reference_output),
                capture_complete=True, managed_request_ordinal=5, grants_coverage=False)
            row['statistics'] = self.statistics_row
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
        self.provider._statistics_arm = None
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
        if self.statistics_capture is not None and self.statistics_capture.active:
            self.statistics_capture.end(completed=False)
        self._context(verify_targets=False)
        if self.root_post_handle is not None:
            self._remove_hook(self.root_post_handle); self.root_post_handle = None
        if self.observer is not None:
            self.observer.close(); self.observer = None
        if self.adapter is not None:
            self.adapter.close(); self.adapter = None
        self.provider.active_adapter = None; self.provider._running = False
        self.provider.enabled = False
        self.provider._statistics_arm = None
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
        require(self.statistics_capture is None or not self.statistics_capture.active,
                'SERVING_TENSOR_CAPTURE_STILL_ACTIVE')
        for handle in reversed(self.handles):
            self._remove_hook(handle)
        self.handles.clear()
        return dict(owned_serving_hooks=0, active_adapter=False, independent_observer=False,
                    standard_worker=not self.provider.simulation, requests=self.seen)
