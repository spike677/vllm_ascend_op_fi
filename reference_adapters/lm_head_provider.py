"""Closed reference v2 provider; one fixed lm_head row group, rank-local writes."""
import torch
from fault_injection.engines.region_lease import RegionLeases
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import Scope,reject_unknown
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
from reference_adapters.lm_head_rows import LmHeadRows,RowAdmission


class LmHeadCollapseProvider(Scope):
    variant='row12_lm_head_collapse'
    persistent_protocol_version=2
    region_kind='persistent_weight'
    continuous_protocol=True
    execution_boundary='python_hook'
    def __init__(self,worker,model,worker_identity,*,mechanism,region_kind):
        super().__init__(worker,model,worker_identity)
        require(region_kind=='persistent_weight','ROW12_PERSISTENT_WEIGHT_REQUIRED')
        self.context=worker._reference_lm_context
        require(self.context['reader_calls']>0 and not self.context.get('reader_failure'),
                'ROW12_ACTUAL_READER_NOT_OBSERVED:'+str(self.context.get('reader_failure')))
        require(self.context.get('provider') is None,'ROW12_ALREADY_OWNED')
        self.site=None;self.enabled=False;self.failure=None;self.readers=[]
    def boundary(self):
        torch.npu.current_stream().synchronize()
        return dict(self.context['boundary'](),device_readers_complete=True)
    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation);reject_unknown(settings)
        if action=='prepare':self.boundary()
        elif action=='register':
            leases=RegionLeases(owner,generation,stream=torch.npu.current_stream,complete=lambda s:s.synchronize())
            self.site=LmHeadRows(self.context['module'].weight,RowAdmission(**self.context['admission']),
                variant=self.variant,leases=leases,boundary=self.boundary,
                observe_weight=lambda:self.context['module'].weight)
            self._registered=True
        elif action=='check':
            self.boundary();require(self._registered,'ROW12_REGISTER_REQUIRED');self.site.current()
        elif action=='enable':
            self.boundary();require(self.context.get('provider') is None,'ROW12_ALREADY_OWNED')
            self.site.enable();self.context['provider']=self;self.enabled=True
        elif action in ('disable','unregister','clean','abort','recover'):
            self.boundary()
            if self.site is not None:self.site.close()
            if self.context.get('provider') is self:self.context['provider']=None
            self.enabled=False
            if action in ('unregister','clean'):self._registered=False
        else:raise ValueError('ROW12_UNKNOWN_ACTION')
        return self.query(owner=owner,generation=generation)
    def consumed(self,request_id):
        require(self.enabled,'ROW12_READER_OUTSIDE_SCOPE')
        self.site.consumed(self.context['module'].weight,
            consumer=self.context['admission']['actual_weight_consumer'],request_id=request_id)
        self.readers.append(dict(request_id=request_id,weight_read=True))
    def query(self,*,owner,generation):
        self.verify(owner,generation);self.boundary()
        snap=self.site.snapshot() if self.site else dict(events=[],reader_observations=0,leases=0)
        count=snap['leases'];hook=self.context.get('provider') is self
        return dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',hook_state_available=True,
            hook_owned=hook,owner=owner,generation=generation,native_loaded=False,native_registered=False,
            registered=self._registered,enabled=self.enabled,inflight=0,leases=count,persistent_leases=count,
            persistent_protocol_version=2,owned_hooks=int(hook),device_readers_complete=True,
            exclusive_ingress_paused=self.context['boundary']()['accepting'] is False,
            replay_pending=False,replay_failed=self.failure is not None,failure=self.failure,
            selected_site='lm_head.weight[token_ids=10,11,12]',events=snap['events'],
            reader_observations=snap['reader_observations'],readers=list(self.readers),row_adapter=snap,
            recovery=dict(bytes_restored=count==0,resources_zero=count==0 and not hook and not self._registered),
            qualification_granted=False)


class LmHeadSwapProvider(LmHeadCollapseProvider):
    variant='row12_lm_head_swap'
