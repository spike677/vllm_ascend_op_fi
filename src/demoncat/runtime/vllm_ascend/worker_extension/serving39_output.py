"""Trusted deployment and real request hooks for postwrite C1 only."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading

from .serving import ServingReplaySession, complete_tensor
from .serving_context import require, require_serial_eager
from .serving41 import read_native_binding


_ATTRIBUTE = '_demoncat_output39_serving_binding'


@dataclass(frozen=True)
class Output39ServingBinding:
    worker: object
    runner: object
    model: object
    transport: object
    prompt: tuple
    folder: object
    pid: int
    thread: int
    native_sha256: str
    native_library_identity: int
    simulation: bool = False
    native_readback: object = None
    stage: str = 'c1'
    max_live_leases: int = 1
    leases: set = field(default_factory=set, compare=False, repr=False)
    runtime_only: bool = False
    group_context: object = None


def install_fault39_output_serving_binding(worker, *, transport, prompt_token_ids,
        evidence_dir=None, stage='c1', max_live_leases=1, _simulation=False, _native_readback=None,
        runtime_only=False, group_contract=None):
    """Bind an already initialized Worker. No model, hooks or Native leases run."""
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    require(not hasattr(worker, _ATTRIBUTE), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    require(stage in ('c1', 'c2_once', 'c2_c3') and type(max_live_leases) is int and 1 <= max_live_leases <= 256,
            'OUTPUT39_EXPLICIT_BOUNDED_GROUP_STAGE_REQUIRED')
    require(type(_simulation) is bool and (_simulation or _native_readback is None),
            'OUTPUT39_DEPLOYMENT_OVERRIDES_CPU_ONLY')
    require(not _simulation or callable(_native_readback), 'EXPLICIT_CPU_OUTPUT39_FIXTURE_REQUIRED')
    require(_simulation or type(transport) is NativePostwriteLibrary, 'OUTPUT39_VERIFIED_NATIVE_TRANSPORT_REQUIRED')
    require(isinstance(prompt_token_ids, (list, tuple)) and 2 <= len(prompt_token_ids) <= 4096
        and all(type(token) is int and token >= 0 for token in prompt_token_ids), 'OUTPUT39_FIXED_PROMPT_REQUIRED')
    require(type(runtime_only) is bool and (not runtime_only or
        stage == 'c1' and max_live_leases == 1 and evidence_dir is None),
        'OUTPUT39_RUNTIME_SINGLE_TARGET_NO_DUMP_REQUIRED')
    folder = None
    if not runtime_only:
        folder = Path(evidence_dir)
        require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),
                'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    runner, model = worker.model_runner, worker.model_runner.model
    group_context=None
    if group_contract is None:
        require_serial_eager(runner, expected_max_seqs=1)
    else:
        require(runtime_only,'OUTPUT39_TP2_ORDINARY_ONLY')
        from .rms18_tp2 import validate_tp2_group
        group_context=validate_tp2_group(worker,'39.output_proxy',group_contract)
    require(not model.training, 'INFERENCE_MODEL_REQUIRED')
    sha = getattr(transport, 'sha256', None)
    require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
            'OUTPUT39_REVIEWED_NATIVE_HASH_REQUIRED')
    binding = Output39ServingBinding(worker, runner, model, transport, tuple(prompt_token_ids), folder,
        os.getpid(), threading.get_ident(), sha, id(getattr(transport, 'lib', None)), _simulation, _native_readback,
        stage, max_live_leases, runtime_only=runtime_only, group_context=group_context)
    native = read_native_binding(binding)
    setattr(worker, _ATTRIBUTE, binding)
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED', mechanism='39.output_proxy',
        workers_created=0, hooks_installed=0, native_leases_opened=0, native=native,
        qualification_granted=False, c2=False, c3=False)


def remove_fault39_output_serving_binding(worker):
    binding = getattr(worker, _ATTRIBUTE, None)
    require(type(binding) is Output39ServingBinding and binding.worker is worker, 'SERVING_DEPLOYMENT_NOT_BOUND')
    require(not binding.leases, 'OWNER_CLEAN_REQUIRED')
    runtime = getattr(worker, '_demoncat_worker_fault_runtime', None)
    lease = getattr(runtime, '_leases', {}).get('39.output_proxy')
    require(lease is None or getattr(lease.provider, 'binding', None) is not binding,
            'WORKER_RPC_OWNER_STILL_RETAINED')
    delattr(worker, _ATTRIBUTE)
    return dict(status='SERVING_DEPLOYMENT_REMOVED')


class Serving39OutputSession(ServingReplaySession):
    """Reuse scheduler/root/logits binding, with the existing projection observer."""
    def __init__(self, provider):
        binding = provider.binding
        super().__init__(provider.worker, provider.model, provider, binding.transport, provider._ticket[0],
            prompt_token_ids=list(binding.prompt), evidence_dir=binding.folder / provider._ticket[0])
        self.provider = provider
        self.window = None
        self.root_running = self.logits_running = False
        self.callbacks = {}
        self.hidden = None

    def install(self):
        self._same_worker()
        require(not self.installed and not self.handles, 'SERVING_HOOK_ALREADY_INSTALLED')
        for module, callback, pre in ((self.model, self._root_pre, True),
                (self.model, self._root_post, False), (self.processor, self._logits_pre, True),
                (self.processor, self._logits_post, False)):
            register = module.register_forward_pre_hook if pre else module.register_forward_hook
            handle = register(callback, with_kwargs=True, **({} if pre else dict(always_call=True)))
            self.handles.append(handle)
            self.callbacks[handle.id] = callback
        self.installed = True

    def observed_parameter_owners(self):
        return ()

    def _same_worker(self):
        super()._same_worker()
        # Runtime already obtained the real identity; avoid recursive RPC locks.
        if hasattr(self, 'provider'):
            self.provider._context()

    def _last_hook(self, module):
        for hooks in (module._forward_pre_hooks, module._forward_hooks):
            owned = [h.id for h in self.handles if h.hooks_dict_ref() is hooks]
            require(len(owned) == 1 and list(hooks)[-1:] == owned,
                    'OUTPUT39_FINAL_RETURN_OBSERVER_REQUIRED')

    def collect(self, *, force=False):
        self._same_worker()  # No collector, Native operation or request completion.

    def _failure(self, error):
        self.provider.fail(error)
        self.expected = None
        self._save()

    def _root_pre(self, module, args, kwargs):
        if not self.provider.attempted:
            return
        try:
            self._last_hook(module)
            self._last_hook(self.processor)
            super()._root_pre(module, args, kwargs)
            self.root_running = True
            self.provider.begin_actual(args, kwargs)
            if self.provider.native_phase != 'health':
                from .serving39_output_discovery import Output39ObservedForward
                paths = {obj: name for name, obj in self.model.named_modules()}
                require(self.processor in paths, 'OUTPUT39_LOGITS_NOT_IN_HELD_MODEL')
                self.window = Output39ObservedForward(self.provider.p, self.provider.native_phase,
                    manual_modules={self.model: '', self.processor: paths[self.processor]}, guard=self.provider._context,
                    parameter_owners=self.observed_parameter_owners())
                self.window.start()
                # This hook is already executing. No newly registered root pre
                # hook is assumed to run in PyTorch's current hook iteration.
                self.window.enter_manual(module)
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        if self.current is None:
            return
        try:
            self._same_worker()
            self.root_running = False
            self._last_hook(module)
            if self.window is not None and self.window.active:
                self.window.leave_manual(module)
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED')
            self.hidden, _, finite = complete_tensor(output)
            _, self.hidden_digest, _ = complete_tensor(output[-1:].contiguous())
            require(finite or self.provider.native_phase == 'fault', 'HEALTHY_HIDDEN_STATES_NOT_FINITE')
            self.current['root_returned'] = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            self._last_hook(module)
            if self.boundary_guard:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_running,
                    'COMPLETE_REAL_ROOT_BEFORE_LOGITS_REQUIRED')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None,
                    'ACTUAL_LOGITS_HEAD_CHANGED')
            _, digest, finite = complete_tensor(actual['hidden_states'])
            require(digest == self.hidden_digest and (finite or self.provider.native_phase == 'fault'),
                    'ACTUAL_LOGITS_ROWS_CHANGED')
            self.logits_running = self.logits_active = True
            if self.window is not None:
                self.window.enter_manual(module)
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        if not self.logits_running:
            return
        try:
            self._same_worker()
            self.logits_running = False
            self._last_hook(module)
            if self.window is not None:
                self.window.leave_manual(module)
            require(output is not None and list(output.shape) == self.boundary_report['output_contract'],
                    'COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED')
            logits, digest, finite = complete_tensor(output)
            require(finite or self.provider.native_phase == 'fault', 'HEALTHY_LOGITS_NOT_FINITE')
            self.current.update(model_output_sha256=digest, model_output_finite=finite)
            result = self.window.finish((self.hidden, logits)) if self.window is not None else None
            self.provider.end_actual((self.hidden, logits), result, self.current)
            self.rows.append(self.current)
            self.current = None
            self.logits_active = False
            self.window = self.hidden = None
            self._save()
        except BaseException as error:
            self._failure(error)
            raise

    def close_window(self):
        self._same_worker()
        require(not self.root_running and not self.logits_running, 'OUTPUT39_ACTUAL_FORWARD_STILL_ACTIVE')
        require(self.current is None or self.provider.failure,
                'OUTPUT39_ACTUAL_REQUEST_UNWIND_UNCONFIRMED')
        if self.window is not None:
            self.window.close(unwound=True)
            self.window = None
        self.current = self.expected = self.hidden = None
        self.logits_active = False

    def close(self):
        self.close_window()
        for handle in reversed(self.handles[:]):
            hooks = handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks.get(handle.id) is self.callbacks[handle.id]),
                    'OUTPUT39_OWNED_HOOK_CHANGED')
            try:
                handle.remove()
            finally:
                if handle.id not in hooks:
                    self.callbacks.pop(handle.id, None)
                    self.handles.remove(handle)
            require(handle.id not in hooks, 'OUTPUT39_HOOK_REMOVAL_UNCONFIRMED')
        self.installed = False

    def readback(self):
        self._same_worker()
        for handle in self.handles:
            require(handle.hooks_dict_ref() is not None
                and handle.hooks_dict_ref().get(handle.id) is self.callbacks[handle.id], 'OUTPUT39_OWNED_HOOK_CHANGED')
        return dict(installed=self.installed, owned_hooks=len(self.handles), expected=self.expected,
            inflight=int(self.root_running or self.logits_running), pending=self.current is not None,
            observer=self.window.readback() if self.window else None)

    def _save(self):
        if hasattr(self, 'provider'):
            (self.folder/'serving_receipt.json').write_text(json.dumps(dict(
                boundary=self.boundary_report, requests=self.rows,
                provider=self.provider.evidence()), indent=2, allow_nan=False)+'\n')
