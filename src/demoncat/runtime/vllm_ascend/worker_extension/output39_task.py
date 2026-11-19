"""Actual request boundary for the ordinary single-target output39 task.

Reuse the reviewed serial eager scheduler/root/logits identity checks. No
controller forward, census, collector, file dump or new Worker lifecycle.
"""
import hashlib

from .serving import ServingReplaySession, complete_tensor
from .serving39_output import Serving39OutputSession
from .serving_context import require


class Output39TaskSession(Serving39OutputSession):
    def __init__(self, provider):
        binding = provider.binding
        ServingReplaySession.__init__(self, provider.worker, provider.model, provider,
            binding.transport, provider._ticket[0], prompt_token_ids=list(binding.prompt),
            evidence_dir=None, persist_evidence=False)
        self.provider = provider
        self.window = None
        self.root_running = self.logits_running = False
        self.callbacks = {}
        self.hidden = None

    def expect(self, request_id):
        # Preserve the inherited unfinished-request checks. Only a closed,
        # completed four-request task gets the exhaustion diagnostic.
        if (self.installed and self.expected is None and self.current is None
                and not self.logits_active and self.backend.phase == 'complete'):
            require(False, 'OUTPUT39_TASK_PHASES_EXHAUSTED:'
                    'normal,fault,off,health (4 requests) already completed')
        return super().expect(request_id)

    def _save(self):
        # Runtime/Host query is the result outlet. No default evidence folder.
        pass

    def _root_pre(self, module, args, kwargs):
        if not self.provider.attempted:
            return
        try:
            self._last_hook(module)
            self._last_hook(self.processor)
            ServingReplaySession._root_pre(self, module, args, kwargs)
            self.root_running = True
            self.window = self.provider.begin_actual(args, kwargs)
            if self.window is not None:
                self.window.start()
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        if self.current is None:
            return
        try:
            self._same_worker()
            self.root_running = False
            if self.window is not None: self.window.in_compute = False
            self._last_hook(module)
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED')
            # Only a digest is retained. These existing value copies are not
            # dispatched through the target observer or a statistics collector.
            with self._diagnostics():
                _, self.hidden_full_digest, finite = complete_tensor(output)
                _, self.hidden_digest, _ = complete_tensor(output[-1:].contiguous())
            require(finite or self.provider.native_phase == 'fault', 'HEALTHY_HIDDEN_STATES_NOT_FINITE')
            self.current['root_returned'] = True
        except BaseException as error:
            self._failure(error)
            raise

    def _diagnostics(self):
        from torch.utils._python_dispatch import _disable_current_modes, _get_current_dispatch_mode_stack
        stack = _get_current_dispatch_mode_stack()
        require(not stack or len(stack) == 1 and stack[0] is self.window,
                'OUTPUT39_FOREIGN_DISPATCH_MODE_ON_TOP')
        return _disable_current_modes()

    def _logits_pre(self, module, args, kwargs):
        # Reuse semantic selected-last-row binding. Pause observer only for its
        # diagnostic readback, never for the actual logits computation.
        try:
            with self._diagnostics():
                window, self.window = self.window, None
                try:
                    super()._logits_pre(module, args, kwargs)
                finally:
                    self.window = window
            if self.window is not None:
                self.window.in_compute = True
                self.window.compute_boundary = 'logits'
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        if not self.logits_running:
            return
        try:
            self._same_worker()
            self.logits_running = False
            if self.window is not None: self.window.in_compute = False
            self._last_hook(module)
            require(output is not None and list(output.shape) == self.boundary_report['output_contract'],
                    'COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED')
            with self._diagnostics():
                _, digest, finite = complete_tensor(output)
            require(finite or self.provider.native_phase == 'fault', 'HEALTHY_LOGITS_NOT_FINITE')
            combined = hashlib.sha256((self.hidden_full_digest + ':' + digest).encode()).hexdigest()
            self.current.update(model_output_sha256=digest, model_output_finite=finite,
                                hidden_logits_sha256=combined)
            result = self.window.finish(combined) if self.window is not None else None
            self.provider.end_actual(combined, result, self.current)
            self.rows.append(self.current)
            self.current = None
            self.logits_active = False
            self.window = self.hidden = None
        except BaseException as error:
            self._failure(error)
            raise

    def close_window(self):
        self._same_worker()
        require(not self.root_running and not self.logits_running, 'OUTPUT39_ACTUAL_FORWARD_STILL_ACTIVE')
        require(self.current is None or self.provider.failure, 'OUTPUT39_ACTUAL_REQUEST_UNWIND_UNCONFIRMED')
        if self.window is not None:
            self.window.close()
            self.window = None
        self.current = self.expected = self.hidden = None
        self.logits_active = False

    def readback(self):
        # The parent checks exact owned hook entries. Its observer interface is
        # experimental; this window reports only state, never a full inventory.
        window, self.window = self.window, None
        try:
            result = super().readback()
        finally:
            self.window = window
        result['observer'] = None if window is None else dict(active=window.active, owned_hooks=0,
            selected_returns=int(window.candidate is not None), consumer_observed=bool(window.consumers))
        return result
