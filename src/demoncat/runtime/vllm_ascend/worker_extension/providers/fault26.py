"""Common lifecycle for the explicitly bound, serial FIA inference deployment."""
from ._scope import Scope,ProviderError,reject_unknown


class Fault26FiaProvider(Scope):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.session=None;self.checked=False;self.attempted=False

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':
            self.bind(owner,generation);reject_unknown(settings)
            from fault_injection.providers import resolve_runtime
            from ..serving26 import FiaServingBinding,FiaServingSession,MECHANISM
            binding=resolve_runtime(MECHANISM,self.model)
            if type(binding) is not FiaServingBinding:raise ProviderError('REVIEWED_FIA26_DEPLOYMENT_REQUIRED')
            self.session=binding.make_session(self.worker,self.model)
            if type(self.session) is not FiaServingSession:raise ProviderError('REVIEWED_FIA26_SESSION_REQUIRED')
            return self.query(owner=owner,generation=generation)
        self.verify(owner,generation)
        reject_unknown(settings,('request_id',) if action=='advance' else ())
        if self.session is None:raise ProviderError('FIA26_PREPARE_REQUIRED')
        if action=='register':self._registered=True
        elif action=='check':
            if not self._registered:raise ProviderError('FIA26_REGISTER_REQUIRED')
            self.checked=True
        elif action in ('enable','execute'):
            if not self.checked or self.attempted:raise ProviderError('SINGLE_REVIEWED_ATTEMPT_REQUIRED')
            self.attempted=True;self.session.install()
        elif action=='advance':self.session.expect(settings.get('request_id'))
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
        self.verify(owner,generation)
        if self.session is None:raise ProviderError('NATIVE_STATE_UNAVAILABLE')
        s=self.session
        if s.current is not None and s.current.get('logits_returned') and not s.error:s.collect()
        raw=s.state()
        return dict(native_loaded=True,registered=self._registered,native_registered=bool(raw['registered']),
            enabled=bool(raw['environment_enabled']),inflight=raw['callback_active']+raw['active_consumer'],
            leases=raw['prepared']+raw['queued_executors']+raw['pending_consumers'],native=raw,
            replay_pending=s.installed,replay_failed=bool(s.error and not s.aborted),
            recovery_validation_pending=s.installed and not s.recovered,serving=s.report())
