"""Expose new RPC methods on an already initialized serving worker."""

import os
import threading

from .runtime import WorkerFaultRuntime


_RUNTIME_ATTRIBUTE = '_demoncat_worker_fault_runtime'
_runtime_lock = threading.Lock()


def _after_fork():
    global _runtime_lock
    _runtime_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _call(worker, action, payload=None):
    pid = os.getpid()
    with _runtime_lock:
        state = vars(worker)
        runtime = state.get(_RUNTIME_ATTRIBUTE)
        if runtime is not None and not isinstance(runtime, WorkerFaultRuntime):
            return {'status': 'REJECTED', 'reason': 'WORKER_RUNTIME_ATTRIBUTE_CONFLICT'}
        if runtime is None or runtime.pid != pid:
            runtime = WorkerFaultRuntime(pid)
            state[_RUNTIME_ATTRIBUTE] = runtime
    try:
        return runtime.invoke(worker, action, payload)
    except Exception as error:
        from .product_errors import product_error_reply
        body=product_error_reply(error)
        return dict(body,status='REJECTED' if body['exit_code']==4 else 'CONTROL_ERROR',
                    recovery_required=body['exit_code']==5)


class DemoncatWorkerExtension:
    def demoncat_product_binding(self, action, payload):
        from .product_binding import product_binding_rpc
        from .serving_context import ContextRefused
        # Control-plane refusal is data, never a fatal executor exception.
        # Forward/kernel exceptions are deliberately outside this boundary.
        try:
            rank=self.demoncat_worker_identity()['worker']['rank']
            result = product_binding_rpc(self, action, payload)
        except (ContextRefused, ValueError, PermissionError) as error:
            result = dict(status='PRODUCT_BINDING_REFUSED', reason=str(error),
                          error_type=type(error).__name__, exit_code=4)
        except Exception as error:
            result = dict(status='PRODUCT_BINDING_ERROR', reason=str(error),
                          error_type=type(error).__name__, exit_code=5,
                          recovery_required=True)
        return dict(result,rank=locals().get('rank',getattr(self,'rank',None)))

    def demoncat_inspect_readonly(self, offset=0, limit=128):
        from .readonly_inspect import inspect_worker
        try:return inspect_worker(self,offset=offset,limit=limit)
        except Exception as error:
            return dict(status='INSPECTION_UNAVAILABLE',rank=getattr(self,'rank',None),
                error_type=type(error).__name__,qualification_granted=False)

    def demoncat_worker_identity(self):
        return _call(self, 'worker_identity')

    def demoncat_capabilities(self):
        return _call(self, 'capabilities')

    def demoncat_discover(self, payload):
        return _call(self, 'discover', payload)

    def demoncat_census(self, action, payload):
        if action not in ('start', 'query', 'stop', 'cleanup'):
            return {'status': 'REJECTED', 'reason': 'UNSUPPORTED_CENSUS_ACTION'}
        return _call(self, 'census_' + action, payload)

    def demoncat_attach(self, payload):
        return _call(self, 'attach', payload)

    def demoncat_fault_rpc(self, action, payload):
        # Control verbs cannot call the attachment API through this method.
        if not isinstance(action, str) or action.startswith('census_') or action in (
            'worker_identity', 'capabilities', 'attach', 'detach', 'query', 'discover',
        ):
            return {'status': 'REJECTED', 'reason': 'UNSUPPORTED_ACTION'}
        return _call(self, action, payload)

    def demoncat_fault_query(self, payload):
        return _call(self, 'query', payload)

    def demoncat_detach(self, payload):
        return _call(self, 'detach', payload)
