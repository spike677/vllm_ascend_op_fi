"""Common ownership and actual native state for a bounded B8 decode replay."""
from ._scope import Scope,ProviderError,reject_unknown


class QuantProvider(Scope):
    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        self.session=None;self.checked=False;self.attempted=False

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':
            self.bind(owner,generation);reject_unknown(settings)
            from fault_injection.providers import resolve_runtime
            from ..serving_quant import QuantServingBinding,QuantServingSession,MECHANISM
            binding=resolve_runtime(MECHANISM,self.model)
            if type(binding) is not QuantServingBinding:raise ProviderError('REVIEWED_QUANT_DEPLOYMENT_REQUIRED')
            self.session=binding.make_session(self.worker,self.model)
            if type(self.session) is not QuantServingSession:raise ProviderError('REVIEWED_QUANT_SESSION_REQUIRED')
            return self.query(owner=owner,generation=generation)
        self.verify(owner,generation)
        reject_unknown(settings,('request_ids','capture_decode') if action=='advance' else ())
        if self.session is None:raise ProviderError('QUANT_PREPARE_REQUIRED')
        if action=='register':self._registered=True
        elif action=='check':
            if not self._registered:raise ProviderError('QUANT_REGISTER_REQUIRED')
            self.checked=True
        elif action in ('enable','execute'):
            if not self.checked or self.attempted:raise ProviderError('SINGLE_REVIEWED_ATTEMPT_REQUIRED')
            self.attempted=True;self.session.install()
        elif action=='advance':
            if 'capture_decode' in settings:
                if settings!={'capture_decode':True}:raise ProviderError('EXPLICIT_DECODE_CAPTURE_ADVANCE_REQUIRED')
                self.session.begin_decode_capture()
            else:self.session.expect(settings.get('request_ids'))
        elif action=='finish':
            self.session.collect()
            if not self.session.recovered:raise ProviderError('QUANT_OFF_NOT_VERIFIED')
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
        if s.expected is not None and getattr(s,'decode_result',None) is not None and not s.error:s.collect()
        raw=s.state()
        return dict(native_loaded=True,registered=self._registered,native_registered=bool(raw['owned']),enabled=bool(raw['armed']),
            inflight=raw['executing'],leases=raw['owned']+raw['pending'],native=raw,
            replay_pending=s.installed,replay_failed=bool(s.error and not s.aborted),
            recovery_validation_pending=s.installed and not s.recovered,serving=s.report())
