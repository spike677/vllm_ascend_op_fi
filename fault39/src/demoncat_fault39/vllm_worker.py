"""Blue-area vLLM 0.18 validation extension, not the unseen internal controller.

The original worker is not subclassed or replaced. Its extension RPC installs
an ownership-scoped wrapper on the *actual* model runner after initialization.
The native libraries must already have been preloaded before worker creation.
"""
from functools import wraps
from collections import deque
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading

from .provider import CtypesProvider
from .schema import Fault39Config


def scheduler_context(scheduler_output):
    counts = getattr(scheduler_output, 'num_scheduled_tokens', None)
    if not isinstance(counts, dict) or not counts or any(
        not isinstance(k, str) or not k or type(v) is not int or v <= 0 for k,v in counts.items()):
        raise RuntimeError('UNPROVEN_REQUEST_MAPPING: actual scheduler token mapping required')
    if sum(counts.values()) != getattr(scheduler_output, 'total_num_scheduled_tokens', None):
        raise RuntimeError('UNPROVEN_REQUEST_MAPPING: inconsistent scheduler token count')
    return dict(request_ids=list(counts), token_counts=dict(counts),
        mapping='SINGLE_SCHEDULER_REQUEST' if len(counts)==1 else 'SCHEDULER_REQUEST_SET_NOT_ROW_ATTRIBUTION')


def _record_error(record, operation, error):
    # Diagnostic failures invalidate capture evidence without changing model
    # results or serializing an exception that might contain tensor contents.
    record['instrumentation_errors'].append(dict(operation=operation, error_type=type(error).__name__))


def _record_snapshot(provider, record, key):
    try:
        snapshot = provider.snapshot()
        # Candidate inventories remain available through the explicit RPC.
        # Repeating them for every decoding step creates quadratic evidence.
        record[key] = {k: v for k, v in snapshot.items() if k != 'candidates'}
    except Exception as error:
        record[key] = None
        _record_error(record, key, error)


def install_runner_scope(worker, provider, *, max_contexts=256):
    if type(max_contexts) is not int or max_contexts <= 0:
        raise ValueError('Positive max_contexts required')
    runner = getattr(worker, 'model_runner', None)
    if runner is None or not callable(getattr(runner, 'execute_model', None)):
        raise RuntimeError('ACTUAL_MODEL_RUNNER_UNAVAILABLE')
    binding = getattr(worker, '_fault39_runner_binding', None)
    if binding is not None:
        if binding[0] is runner and runner.execute_model is binding[2]:
            return
        raise RuntimeError('MODEL_RUNNER_WRAPPER_OWNERSHIP_CHANGED')
    original = runner.execute_model
    signature = inspect.signature(original)
    if 'scheduler_output' not in signature.parameters:
        raise RuntimeError('UNSUPPORTED_MODEL_RUNNER_SIGNATURE')
    if not hasattr(worker, '_fault39_contexts'):
        worker._fault39_contexts = deque(maxlen=max_contexts)
        worker._fault39_context_index = 0
        worker._fault39_contexts_dropped = 0
        worker._fault39_capture = None
        worker._fault39_capture_configured = False
        worker._fault39_capture_finished = False
        worker._fault39_capture_owner = None
        worker._fault39_owner = None
    local = threading.local()

    @wraps(original)
    def scoped(*args, **kwargs):
        owner = worker._fault39_owner
        if not owner:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        scheduled = bound.arguments['scheduler_output']
        if getattr(scheduled, 'total_num_scheduled_tokens', None) == 0:
            return original(*args, **kwargs)
        context = scheduler_context(scheduled)
        if getattr(local, 'inside', False):
            raise RuntimeError('NESTED_MODEL_RUNNER_UNSUPPORTED')
        record = dict(context, index=worker._fault39_context_index, rank=worker.rank,
                      pid=os.getpid(), completed=False, instrumentation_errors=[])
        worker._fault39_context_index += 1
        if len(worker._fault39_contexts) == worker._fault39_contexts.maxlen:
            worker._fault39_contexts_dropped += 1
        worker._fault39_contexts.append(record)
        capture = worker._fault39_capture
        local.inside = True
        try:
            _record_snapshot(provider, record, 'native_before')
            with provider.context(owner, context['request_ids']):
                attempted_capture = False
                try:
                    if capture:
                        attempted_capture = True
                        try:
                            capture.start(model=runner.model, rank_id=worker.rank)
                            record['capture_started'] = True
                        except Exception as error:
                            record['capture_started'] = False
                            _record_error(record, 'capture_start', error)
                    result = original(*args, **kwargs)
                    record['completed'] = True
                    return result
                finally:
                    # A failed start can still have installed hooks. Attempt
                    # cleanup independently; neither error may mask the model.
                    if attempted_capture:
                        for operation in ('stop', 'step'):
                            try:
                                getattr(capture, operation)()
                            except Exception as error:
                                _record_error(record, 'capture_' + operation, error)
        finally:
            try:
                _record_snapshot(provider, record, 'native_after')
            finally:
                local.inside = False
                record['instrumentation_complete'] = not record['instrumentation_errors']

    runner.execute_model = scoped
    worker._fault39_runner_binding = (runner, original, scoped)


def prepare_worker(worker, *, include_snapshot=True):
    provider = getattr(worker, '_fault39_provider', None)
    if provider is None:
        import torch
        rank = getattr(worker, 'rank', None)
        if type(rank) is not int or rank < 0:
            raise RuntimeError('ACTUAL_WORKER_RANK_UNAVAILABLE')
        device = torch.npu.current_device()
        configured = getattr(worker, 'device', None)
        if getattr(configured, 'type', None) != 'npu' or configured.index != device:
            raise RuntimeError('ACTUAL_WORKER_DEVICE_MISMATCH')
        provider = CtypesProvider.attach_preloaded()
        provider.bind_worker_rank(rank)
        install_runner_scope(worker, provider)
        worker._fault39_provider = provider
        worker._fault39_device = device
    elif getattr(worker, '_fault39_runner_binding', None) is None:
        # Detach restores the method only. A later explicit prepare/configure
        # may reinstall our scope; read-only snapshots must not do so.
        install_runner_scope(worker, provider)
    return dict(rank=worker.rank, logical_device=worker._fault39_device, pid=os.getpid(),
        visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        worker_class=type(worker).__module__+'.'+type(worker).__qualname__,
        provider=provider.prepare() if include_snapshot else None,
        integration='BLUE_VLLM_EXTENSION_NOT_INTERNAL_DEMONCAT')


def capture_worker(worker, path, steps):
    from msprobe.pytorch import PrecisionDebugger
    import msprobe
    output = Path(path)
    if not output.is_absolute() or not output.resolve().is_relative_to(Path('/evidence')):
        raise ValueError('Capture must be under this container /evidence mount')
    if worker._fault39_capture is not None or getattr(worker, '_fault39_capture_configured', False):
        raise RuntimeError('CAPTURE_ALREADY_CONFIGURED: use a fresh worker for another B/F group')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=steps,
        dump_path=str(output/'observation'), async_dump=False,
        statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    config_file = output/'config.json'
    config_file.write_text(json.dumps(config,indent=2)+'\n')
    package = Path(msprobe.__file__).resolve().parent
    files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json')}
    (output/'collector_identity.json').write_text(json.dumps(dict(package=str(package),files=files,
        edited_by_fault39=False, native_input_replacement_may_be_below_python_observation=True),indent=2)+'\n')
    worker._fault39_capture = PrecisionDebugger(config_path=str(config_file))
    worker._fault39_capture_configured = True
    worker._fault39_capture_owner = worker._fault39_owner
    return dict(config_path=str(config_file), rank=worker.rank, steps=steps,
                scope='Actual model-runner calls; steps are not fabricated token numbers')


def finish_capture_worker(worker, owner):
    """Close capture between synchronous workloads without changing native owner.

    A selected request may end before the configured step limit. Clearing the
    runner's reference prevents the subsequent OFF request from consuming the
    remaining capture steps. A stop failure is reported and must abort further
    model workloads; clearing the reference alone is not a successful stop.
    """
    if (not isinstance(owner, str) or not owner or owner != worker._fault39_owner
            or getattr(worker, '_fault39_capture_owner', None) not in (None, owner)):
        raise RuntimeError('CAPTURE_OWNER_MISMATCH')
    capture = worker._fault39_capture
    if capture is None:
        finished = getattr(worker, '_fault39_capture_finished', False)
        return dict(status='CAPTURE_FINISHED' if finished else 'CAPTURE_NOT_CONFIGURED',
                    rank=worker.rank, owner=owner, already_finished=finished, capture_active=False)
    result = dict(status='CAPTURE_FINISHED', rank=worker.rank, owner=owner,
                  already_finished=False, capture_active=False,
                  last_context_index=worker._fault39_context_index - 1)
    try:
        capture.stop()
    except Exception as error:
        result.update(status='CAPTURE_FINISH_FAILED', error_type=type(error).__name__)
    finally:
        # scoped() takes a fresh reference at each execute_model entry. Native
        # owner, admission state and the model-runner wrapper stay untouched.
        worker._fault39_capture = None
        worker._fault39_capture_configured = True
        worker._fault39_capture_finished = result['status'] == 'CAPTURE_FINISHED'
    return result


class Fault39WorkerExtension:
    # A single unique method avoids conflicts with vLLM's injected-base checks.
    def fault39_rpc(self, action, payload=None):
        payload = {} if payload is None else payload
        if not isinstance(payload, dict):
            raise ValueError('RPC payload must be a dictionary')
        dispatch_rank = payload.get('dispatch_rank')
        if dispatch_rank is not None:
            if type(dispatch_rank) is not int or dispatch_rank < 0:
                raise ValueError('Invalid dispatch_rank')
            if dispatch_rank != self.rank:
                provider = getattr(self, '_fault39_provider', None)
                if provider is None:
                    provider = CtypesProvider.attach_preloaded()
                return dict(status='RPC_NOT_TARGETED', rank=self.rank, native=provider.snapshot())
        if action == 'prepare':
            return prepare_worker(self)
        if action in ('configure', 'capture'):
            prepare_worker(self, include_snapshot=False)
        provider = getattr(self, '_fault39_provider', None)
        if provider is None:
            raise RuntimeError('FAULT39_PREPARE_REQUIRED')
        if action == 'configure':
            if payload['rank'] != self.rank or payload['device'] != self._fault39_device:
                raise RuntimeError('RPC_WORKER_IDENTITY_MISMATCH')
            owner = payload['owner']
            generation = provider.configure(owner, Fault39Config.from_dict(payload['config']),
                self.rank, self._fault39_device, mode=payload.get('mode','inject'),
                target_signature=payload.get('target_signature',''), input_mask=payload.get('input_mask',3))
            self._fault39_owner = owner
            return dict(generation=generation,rank=self.rank,snapshot=provider.snapshot())
        if action in ('disable','clean'):
            result = getattr(provider,action)(payload['owner'],payload['generation'],payload.get('timeout_ms',10000))
            if action=='clean' and result['status']=='DISABLED_CONFIRMED':
                self._fault39_owner = None
            return result
        if action == 'snapshot':
            return dict(identity=dict(rank=self.rank, logical_device=self._fault39_device, pid=os.getpid(),
                integration='BLUE_VLLM_EXTENSION_NOT_INTERNAL_DEMONCAT'),
                native=provider.snapshot(), contexts=list(self._fault39_contexts),
                contexts_dropped=self._fault39_contexts_dropped,
                runner_scope_installed=getattr(self, '_fault39_runner_binding', None) is not None)
        if action == 'capture':
            return capture_worker(self,payload['path'],payload.get('steps',[0,1,2]))
        if action == 'finish_capture':
            return finish_capture_worker(self, payload.get('owner'))
        if action == 'detach':
            snapshot=provider.snapshot()
            if (snapshot.get('owner') != '' or snapshot.get('enabled') is not False
                    or snapshot.get('inflight') != 0 or snapshot.get('leases') != 0):
                raise RuntimeError('CLEAN_REQUIRED_BEFORE_DETACH')
            binding = getattr(self, '_fault39_runner_binding', None)
            if binding is None:
                return dict(status='DETACHED',rank=self.rank)
            runner,original,scoped=binding
            if runner.execute_model is not scoped:
                raise RuntimeError('MODEL_RUNNER_WRAPPER_OWNERSHIP_CHANGED')
            runner.execute_model=original
            self._fault39_runner_binding=None
            self._fault39_owner=None
            return dict(status='DETACHED',rank=self.rank)
        raise ValueError('Unknown Fault39 RPC action')
