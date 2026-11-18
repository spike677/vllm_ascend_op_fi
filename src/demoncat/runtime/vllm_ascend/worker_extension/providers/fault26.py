"""Common lifecycle for the explicitly bound, serial FIA inference deployment."""
from ._scope import Scope,ProviderError,reject_unknown


class Fault26FiaProvider(Scope):
    mechanism='26.fia_destination'
    binding_class='FiaServingBinding'
    session_class='FiaServingSession'
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.session=None;self.checked=False;self.attempted=False
        from fault_injection.providers import resolve_runtime
        from .. import serving26
        from ..serving26_core import Writer26CoreBinding
        binding=resolve_runtime(self.mechanism,self.model)
        if type(binding) is Writer26CoreBinding:
            if binding.mechanism!=self.mechanism:raise ProviderError('WRITER26_CORE_MECHANISM_MISMATCH')
            from .fault26_core import Fault26CoreProvider
            self.core=Fault26CoreProvider(worker,model,identity,binding)
        elif type(binding) is getattr(serving26,self.binding_class):
            self.core=None
        else:raise ProviderError('REVIEWED_WRITER26_DEPLOYMENT_REQUIRED')

    def dispatch(self,action,settings,*,owner,generation):
        if self.core is not None:
            return self.core.dispatch(action,settings,owner=owner,generation=generation)
        if action=='prepare':
            self.bind(owner,generation);reject_unknown(settings)
            from fault_injection.providers import resolve_runtime
            from .. import serving26
            binding=resolve_runtime(self.mechanism,self.model)
            if type(binding) is not getattr(serving26,self.binding_class):raise ProviderError('REVIEWED_WRITER26_DEPLOYMENT_REQUIRED')
            self.session=binding.make_session(self.worker,self.model)
            if type(self.session) is not getattr(serving26,self.session_class):raise ProviderError('REVIEWED_WRITER26_SESSION_REQUIRED')
            return self.query(owner=owner,generation=generation)
        self.verify(owner,generation)
        reject_unknown(settings,('request_id','collect') if action=='advance' else ())
        if self.session is None:raise ProviderError('FIA26_PREPARE_REQUIRED')
        if action=='register':self._registered=True
        elif action=='check':
            if not self._registered:raise ProviderError('FIA26_REGISTER_REQUIRED')
            self.checked=True
        elif action in ('enable','execute'):
            if not self.checked or self.attempted:raise ProviderError('SINGLE_REVIEWED_ATTEMPT_REQUIRED')
            self.attempted=True;self.session.install()
        elif action=='advance':
            if settings=={'collect':True}:self.session.collect()
            elif set(settings)=={'request_id'}:self.session.expect(settings['request_id'])
            else:raise ProviderError('EXPLICIT_WRITER26_ARM_OR_COLLECTION_REQUIRED')
        elif action=='finish':
            self.session.collect()
            if not self.session.recovered:raise ProviderError('FIA26_OFF_RECOVERY_NOT_VERIFIED')
            self.session.close()
        elif action in ('disable','abort','recover','clean','unregister'):
            self.session.close(abort=not self.session.recovered)
            if action in ('clean','unregister'):self._registered=False
        else:raise ProviderError('UNSUPPORTED_PROVIDER_ACTION')
        return self.query(owner=owner,generation=generation)

    def query(self,*,owner,generation):
        if self.core is not None:
            return self.core.query(owner=owner,generation=generation)
        self.verify(owner,generation)
        if self.session is None:raise ProviderError('NATIVE_STATE_UNAVAILABLE')
        s=self.session
        raw=s.state()
        return dict(native_loaded=True,registered=self._registered,native_registered=bool(raw['registered']),
            enabled=bool(raw['environment_enabled']),inflight=raw['callback_active']+raw['active_consumer'],
            leases=raw['prepared']+raw['queued_executors']+raw['pending_consumers'],native=raw,
            replay_pending=s.installed,replay_failed=bool(s.error and not s.aborted),
            recovery_validation_pending=s.installed and not s.recovered,serving=s.report())


class Fault26ResidualProvider(Fault26FiaProvider):
    mechanism='26.residual_stomp'
    binding_class='ResidualServingBinding'
    session_class='ResidualServingSession'
