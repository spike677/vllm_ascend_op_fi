"""Live public-input half of the single-target TND denominator plan.

This component is deliberately not a Native admission implementation. It is
constructed by trusted Attention deployment code and reads the existing
execution adapter's current call. It cannot load an executor, reconstruct a
plan from a receipt, or authorize F. The generated Execute side must supply
the separate, synchronous prepared-input/abort implementation before wiring.
"""
from copy import deepcopy
import os
import threading
import uuid

from .attention28_prepared_input import identity
from .attention28_stock_census import capture_inputs, validate_tnd_prefill
from .attention_execution import AttentionExecutionAdapter, bind_actual_api
from .serving_attention import AttentionDeployment, ATTRIBUTE
from .serving_context import require


MECHANISM = '28.fia.denominator_replay'
TENSORS = ('query', 'key', 'value', 'atten_mask')


class _Plan:
    """A same-process object, never an RPC/saved-evidence capability."""
    def __init__(self, owner, current, row, module, context, metadata, original, args, kwargs, actual):
        self.owner = owner
        self.token = uuid.uuid4().hex
        self.current, self.row = current, row
        self.module, self.context, self.metadata = module, context, metadata
        self.original, self.args, self.kwargs, self.actual = original, args, kwargs, actual
        self.request = current['request_id']
        self.phase = current['phase']
        self.path = row['module_path']
        self.ordinal = row['ordinal']
        self.live_call = owner.session.live_api_calls[-1]
        self.before = None
        self.error = None
        self.closed = False

    def __reduce_ex__(self, protocol):
        raise TypeError('TND_REPLAY_LIVE_PLAN_NOT_SERIALIZABLE')

    def __copy__(self):
        raise TypeError('TND_REPLAY_LIVE_PLAN_NOT_COPYABLE')

    def __deepcopy__(self, memo):
        raise TypeError('TND_REPLAY_LIVE_PLAN_NOT_COPYABLE')


class ReplayInputBinding:
    """Hold one actual module and each current call's original input objects.

``begin`` belongs immediately inside the existing natural API wrapper, before
its once-only original call. ``check_current`` also works while that original
call is active (original_calls == 1). It never accepts typed Native metadata
as proof of a prepared workspace. ``close_public`` releases only these public
references: this class has no Native resource and cannot report Native abort.
"""
    def __init__(self, session):
        require(type(session) is AttentionExecutionAdapter,
                'TND_REPLAY_EXISTING_EXECUTION_ADAPTER_REQUIRED')
        self.session = session
        self.deployment = session.deployment
        require(type(self.deployment) is AttentionDeployment and self.deployment.mechanism == MECHANISM,
                'TND_REPLAY_EXACT_ATTENTION_DEPLOYMENT_REQUIRED')
        self.identity = identity()
        require(len(self.deployment.leases) == 1, 'TND_REPLAY_CURRENT_OWNER_REQUIRED')
        self.ticket = next(iter(self.deployment.leases))
        require(type(self.ticket) is tuple and len(self.ticket) == 2
                and type(self.ticket[0]) is str and self.ticket[0]
                and type(self.ticket[1]) is int and self.ticket[1] > 0,
                'TND_REPLAY_CURRENT_OWNER_REQUIRED')
        self.active = None
        self.target = None
        self.used = set()
        self.first_error = None
        self.closed_count = 0
        self._owner()

    def __reduce_ex__(self, protocol):
        raise TypeError('TND_REPLAY_LIVE_BINDING_NOT_SERIALIZABLE')

    def _owner(self):
        session, deployment = self.session, self.deployment
        require(identity() == self.identity and deployment.pid == os.getpid()
                and deployment.thread == threading.get_ident(), 'TND_REPLAY_OWNER_THREAD_CHANGED')
        table = getattr(deployment.worker, ATTRIBUTE, None)
        require(type(table) is dict and table.get(MECHANISM) is deployment
                and session.deployment is deployment and deployment.leases == {self.ticket}
                and deployment.worker.model_runner is deployment.runner
                and deployment.runner.model is deployment.model
                and session.worker is deployment.worker and session.runner is deployment.runner
                and session.model is deployment.model, 'TND_REPLAY_WORKER_OR_RESERVATION_CHANGED')
        session.guard()

    def _current(self, row):
        self._owner()
        session = self.session
        require(session.installed and session.root_active == 1 and session.callbacks == 1
                and session.current is not None and session.current['calls']
                and session.current['calls'][-1] is row and session.stack
                and len(session.live_api_calls) == 1 and session.live_api_calls[-1][0] is row,
                'TND_REPLAY_CURRENT_NATURAL_CALL_REQUIRED')
        session._owned()
        path, module = session.stack[-1]
        context = session._context()
        require(context.model_instance is session.model
                and context.no_compile_layers.get(module.layer_name) is module
                and row['module_object_id'] == id(module) and row['module_path'] == path
                and any(value is module and name == path for name, value in session.candidates)
                and row['request_id'] == session.current['request_id']
                and row['scheduler'] == session.current['scheduler']
                and session.current.get('decode') is False
                and type(row['original_calls']) is int and row['original_calls'] in (0, 1)
                and row.get('original_returned') is False,
                'TND_REPLAY_ACTUAL_MODULE_OR_REQUEST_CHANGED')
        metadata = context.attn_metadata[module.layer_name]
        return session.current, module, context, metadata

    def begin(self, row):
        require(self.active is None, 'TND_REPLAY_PLAN_ALREADY_HELD')
        current, module, context, metadata = self._current(row)
        _, original, args, kwargs = self.session.live_api_calls[-1]
        require(row['original_calls'] == 0 and row['actual_callable']['object_id'] == id(original)
                and type(args) is tuple and type(kwargs) is dict, 'TND_REPLAY_ORIGINAL_API_ENTRY_REQUIRED')
        require(self.target is None or self.target is module, 'TND_REPLAY_DIFFERENT_ACTUAL_TARGET')
        key = (current['request_id'], current['phase'], row['ordinal'])
        require(key not in self.used, 'TND_REPLAY_CALL_ALREADY_PLANNED')
        actual = bind_actual_api(args, kwargs)
        plan = _Plan(self, current, row, module, context, metadata, original, args, dict(kwargs), actual)
        # Own the strong references before a fallible readback. Even a failed
        # capture has an explicit local cleanup path; no Native call is made.
        self.active = plan
        self.used.add(key)
        try:
            before = capture_inputs(original, args, actual, metadata)
            validate_tnd_prefill(before, row, simulation=self.deployment.simulation)
            import torch
            mask = actual['atten_mask'].detach().cpu()
            require(torch.equal(mask, torch.ones_like(mask).triu(1)), 'TND_REPLAY_CAUSAL_MASK_VALUES_CHANGED')
            require(all(a is b for a, b in zip(self._current(row), (current, module, context, metadata))),
                    'TND_REPLAY_CONTEXT_CHANGED_DURING_CAPTURE')
            plan.before = before
            self.target = module
            return plan
        except BaseException as error:
            self._error(plan, error)
            raise

    def _error(self, plan, error):
        message = repr(error)
        self.first_error = self.first_error or message
        plan.error = plan.error or message

    def check_current(self, plan):
        """Re-read actual values, including inference tensors without _version."""
        self._owner()
        require(type(plan) is _Plan and self.active is plan and plan.owner is self
                and not plan.closed and plan.before is not None and plan.error is None,
                'TND_REPLAY_EXACT_LIVE_PLAN_REQUIRED')
        try:
            current, module, context, metadata = self._current(plan.row)
            require(current is plan.current and module is plan.module and context is plan.context
                    and metadata is plan.metadata and current['request_id'] == plan.request
                    and current['phase'] == plan.phase and plan.row['ordinal'] == plan.ordinal
                    and plan.row['module_path'] == plan.path
                    and self.session.live_api_calls[-1] is plan.live_call,
                    'TND_REPLAY_CURRENT_CALL_IDENTITY_CHANGED')
            _, original, args, kwargs = plan.live_call
            require(original is plan.original and args is plan.args, 'TND_REPLAY_ORIGINAL_CALL_CHANGED')
            actual = bind_actual_api(args, kwargs)
            require(all(actual[name] is plan.actual[name] for name in TENSORS),
                    'TND_REPLAY_ACTUAL_INPUT_OBJECT_CHANGED')
            after = capture_inputs(plan.original, plan.args, actual, metadata)
            require(after == plan.before, 'TND_REPLAY_CURRENT_INPUT_CHANGED')
            return self.snapshot()
        except BaseException as error:
            self._error(plan, error)
            raise

    def authorize_fault(self, plan):
        self.check_current(plan)
        raise RuntimeError('TND_REPLAY_NATIVE_TWO_STAGE_ADMISSION_UNBOUND')

    def close_public(self, plan):
        """No executor operation: only release this object's public references."""
        self._owner()
        require(type(plan) is _Plan and self.active is plan and plan.owner is self and not plan.closed,
                'TND_REPLAY_EXACT_LIVE_PLAN_REQUIRED')
        plan.closed = True
        plan.current = plan.row = plan.module = plan.context = plan.metadata = None
        plan.original = plan.args = plan.kwargs = plan.actual = plan.live_call = None
        self.active = None
        self.closed_count += 1

    def snapshot(self):
        self._owner()
        plan = self.active
        return deepcopy(dict(schema='demoncat.tnd28.public_input_plan.v1', identity=self.identity,
            owner=self.ticket[0], generation=self.ticket[1], target_object_id=id(self.target) if self.target else None,
            active_token=plan.token if plan else None, request_id=plan.request if plan else None,
            phase=plan.phase if plan else None, input=plan.before if plan else None,
            public_objects_held=len(TENSORS) if plan else 0, closed_count=self.closed_count,
            first_error=self.first_error, native_prepared_bound=False, native_abort_performed=False,
            workspace_lifetime='REQUIRES_ACTUAL_SYNCHRONOUS_EXECUTE_WINDOW',
            fault_authorized=False, C1=False, C2=False, C3=False))
