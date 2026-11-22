"""Product scope on the existing Runtime owner/lease/action protocol."""
from copy import deepcopy
from ._scope import Scope, reject_unknown
from ..continuous_binding import current_binding
from ..serving_context import require
from ..product_request import ProductRequest
from fault_injection.product_contract import RequestScope


class ProductProvider(Scope):
    execution_boundary = 'python_hook'
    ordinary_only = True
    continuous_protocol = True

    def __init__(self, worker, model, identity, mechanism):
        super().__init__(worker, model, identity)
        self.binding = current_binding(worker, model, mechanism)
        require(self.binding is not None and self.binding.product is not None, 'PRODUCT_BINDING_REQUIRED')
        self.policy = RequestScope(**self.binding.product['scope'])
        self.session = None
        self.last = None
        self.seen = set()
        self.started = self.checked = self.closed = self.reserved = False
        self.error = None

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        self.binding.check()
        reject_unknown(settings, ('request_id', 'collect') if action == 'advance' else ())
        try:
            if action == 'prepare':
                require(self.binding.warmup_valid and not self.binding.leases, 'PRODUCT_STABLE_CANARY_REQUIRED')
            elif action == 'register':
                require(not self.binding.leases, 'PRODUCT_ALREADY_OWNED')
                self.binding.leases.add(self._ticket)
                self.reserved = self._registered = True
            elif action == 'check':
                require(self._registered, 'PRODUCT_REGISTER_REQUIRED')
                self.checked = True
            elif action == 'enable':
                require(self.checked and not self.started, 'PRODUCT_SINGLE_SCOPE_START')
                self.started = True
            elif action == 'advance':
                require(self.started and not self.closed and self.error is None, 'PRODUCT_NOT_ENABLED')
                if settings == {'collect': True}:
                    require(self.session is not None, 'PRODUCT_NO_INFLIGHT_REQUEST')
                    self.session.finished = True  # trusted host completion hook, after engine completion
                    self.last = self.session.collect()
                    self.session = None
                else:
                    require(set(settings) == {'request_id'} and self.session is None, 'PRODUCT_SERIAL_REQUEST_REQUIRED')
                    rid = settings['request_id']
                    require(type(rid) is str and rid and rid not in self.seen, 'PRODUCT_FRESH_INTERNAL_ID_REQUIRED')
                    skip = None if self.policy.includes(len(self.seen)) else 'SCOPE_REQUEST_LIMIT_REACHED'
                    self.session = ProductRequest(self.binding, [rid], forced_skip=skip)
                    self.seen.add(rid)
                    self.session.install()
            elif action in ('disable', 'unregister', 'clean', 'abort', 'recover', 'finish'):
                require(self.session is None, 'PRODUCT_DRAIN_REQUEST_BEFORE_STOP')
                self.closed = True
                if action in ('unregister', 'clean'):
                    if self.reserved:
                        self.binding.leases.remove(self._ticket)
                    self.reserved = self._registered = False
            else:
                raise RuntimeError('UNSUPPORTED_PRODUCT_ACTION')
        except BaseException as error:
            self.error = repr(error)
            raise
        return self.query(owner=owner, generation=generation)

    def query(self, *, owner, generation):
        self.verify(owner, generation)
        self.binding.check()
        session = self.session
        site = getattr(session, 'site', None)
        pending = getattr(site, 'pending', 0)
        if isinstance(pending, dict):
            pending = len(pending)
        return dict(execution_boundary='python_hook', state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, hook_owned=session is not None, owner=owner, generation=generation,
            native_loaded=False, native_registered=False, registered=self._registered, enabled=False,
            inflight=session.inside if session else 0, leases=pending,
            owned_hooks=len(session.hooks) + len(session.patches) if session else 0,
            replay_pending=session is not None, replay_failed=bool(self.error),
            product_mode=True, next_phase='complete' if self.closed else 'fault',
            scope=deepcopy(self.binding.product['scope']), request_count=len(self.seen),
            last_request=deepcopy(self.last), failure=self.error,
            recovery=dict(bytes_restored=self.last is None or self.last['bytes_restored'],
                resources_zero=not self.reserved and session is None,
                canary_exact=None, restored=False),
            qualification_granted=False)
