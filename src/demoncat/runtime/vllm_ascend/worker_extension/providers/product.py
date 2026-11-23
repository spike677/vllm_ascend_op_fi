"""Product scope on the existing Runtime owner/lease/action protocol."""
from copy import deepcopy
from ._scope import Scope, reject_unknown
from ..continuous_binding import current_binding
from ..serving_context import require
from ..product_observer import make_automatic_observer
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
        self.observer = None
        self.last = None
        self.seen = set()
        self.started = self.checked = self.closed = self.reserved = False
        self.error = None

    def dispatch(self, action, settings, *, owner, generation):
        if action == 'prepare':
            self.bind(owner, generation)
        self.verify(owner, generation)
        self.binding.check()
        reject_unknown(settings)
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
                self.observer = make_automatic_observer(self.binding)
                self.observer.install()
                self.started = True
            elif action in ('disable', 'unregister', 'clean', 'abort', 'recover', 'finish'):
                if self.observer is not None:
                    self.observer.close()
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
        snapshot = self.observer.snapshot() if self.observer is not None else dict(
            requests=[], request_count=0, active_request_id=None, leases=0, owned_hooks=0, inflight=0, failure=None)
        active = self.started and not self.closed
        return dict(execution_boundary='python_hook', state_source='OWNED_PYTHON_HOOKS',
            hook_state_available=True, hook_owned=snapshot['owned_hooks']>0, owner=owner, generation=generation,
            native_loaded=False, native_registered=False, registered=self._registered, enabled=active,
            inflight=snapshot['inflight'], leases=snapshot['leases'], owned_hooks=snapshot['owned_hooks'],
            replay_pending=False, replay_failed=bool(self.error or snapshot['failure']),
            product_mode=True, automatic_scheduler_observer=True, next_phase='complete' if self.closed else 'fault',
            scope=deepcopy(self.binding.product['scope']), request_count=snapshot['request_count'],
            # Detailed records use the dedicated paginated status RPC, not every
            # lifecycle reply (the ordinary 512KiB limit remains unchanged).
            last_request=None, failure=self.error or snapshot['failure'],
            recovery=dict(bytes_restored=all(r['bytes_restored'] for r in snapshot['requests']),
                resources_zero=not self.reserved and snapshot['owned_hooks']==snapshot['leases']==snapshot['inflight']==0,
                canary_exact=None, restored=False), qualification_granted=False)
