"""Source-pinned v2 example: first actually observed dense QMM scale, all ranks.

This is an acceptance example, not an executable deployment registration.
The trusted bootstrap supplies the real census and paused-ingress boundary.
The audited unused FP32 scale copy is invalidated while enabled, then restored
as the same object. Chunked/other schemes are refused, not guessed equivalent.
"""
import hashlib
import inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from fault_injection.engines.region_lease import RegionLeases,identity,exact
from demoncat.runtime.vllm_ascend.worker_extension.external_persistent_regions import PersistentRegions,PersistentRegion
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import Scope,reject_unknown
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require

SCHEME_SHA='ac9bd6ec05356b639665ec05ae21b19e1d0f0dc6723f7568b2f0ab70821285de'
WRAPPER_SHA='6cff93c740685821e773333f65f193044962c560e9dee136759acb1994efd6b1'


class WeightScaleProvider(Scope):
    persistent_protocol_version=2
    continuous_protocol=True
    execution_boundary='python_hook'
    def __init__(self,worker,model,worker_identity,*,mechanism,region_kind):
        super().__init__(worker,model,worker_identity)
        require(region_kind=='persistent_weight','V2_EXAMPLE_WEIGHT_ONLY')
        self.region_kind=region_kind;self.context=worker._v2_weight_example_context
        self.runner=worker.model_runner;self.enabled=False;self.hook=None;self.regions=None
        self.failure=None;self.invalidated=False;self.requests=[];self.inside=0
        modules=dict(model.named_modules());chosen=None
        for label,reader in sorted(self.context['census']['scale_readers'].items()):
            if not label.endswith('.weight_scale') or reader['operator']!='npu::npu_quant_matmul' or reader['argument']!='scale':continue
            name=label.removesuffix('.weight_scale');module=modules[name]
            wrapper=getattr(module,'quant_method',None);scheme=getattr(wrapper,'quant_method',None)
            if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':continue
            if getattr(module,'_chunk_size',0)!=0:continue
            for obj,expected in ((wrapper,WRAPPER_SHA),(scheme,SCHEME_SHA)):
                source=inspect.getsourcefile(type(obj))
                require(source is not None and hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,
                        'V2_EXAMPLE_QUANT_SOURCE_CHANGED')
            chosen=(label,module);break
        require(chosen is not None,'V2_NO_ACTUALLY_OBSERVED_DENSE_SCALE')
        self.label,self.module=chosen;self.scale=self.module.weight_scale
        require(self.scale.dtype==torch.bfloat16 and self.scale.is_contiguous(),'V2_EXAMPLE_BF16_ND_SCALE_REQUIRED')
        self.derived=getattr(self.module,'weight_scale_fp32',None)
        require(isinstance(self.derived,torch.Tensor) and self.derived.dtype==torch.float32
                and exact(self.derived,self.scale.float()),'V2_DERIVED_SCALE_IDENTITY_UNPROVEN')
        self.derived_backup=self.derived.detach().clone()
        self.allocation=identity(self.scale)

    def boundary(self):
        b=self.context['boundary']()
        require(self.inside==0,'V2_EXAMPLE_FORWARD_STILL_ACTIVE')
        torch.npu.current_stream().synchronize()
        return dict(b,device_readers_complete=True)

    def install(self):
        require('execute_model' not in vars(self.runner),'V2_RUNNER_ALREADY_OWNED')
        original=self.runner.execute_model;provider=self
        class Consumers(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};schema=func._schema
                values={k.name:v for k,v in zip(schema.arguments,args)};values.update(kwargs)
                scale=values.get('scale')
                match=(schema.name=='npu::npu_quant_matmul' and isinstance(scale,torch.Tensor)
                       and identity(scale)==provider.allocation)
                if match:
                    require(exact(scale,provider.regions.leases._get(provider.regions.entries['scale']['token'])['expected']),
                            'V2_ACTUAL_CONSUMER_SCALE_BYTES_CHANGED')
                result=func(*args,**kwargs)
                if match:
                    provider.regions.consumed('scale','npu::npu_quant_matmul:scale',request_generation=provider.current_request)
                    provider.requests.append(dict(request_id=provider.current_request,consumer='npu::npu_quant_matmul',
                        consumed_changed_scale=True,device_read_complete=True))
                return result
        def execute(scheduler,*args,**kwargs):
            ids=list(scheduler.num_scheduled_tokens)
            require(not ids or len(ids)==1,'V2_EXAMPLE_SERIAL_REQUESTS_ONLY')
            if not ids:return original(scheduler,*args,**kwargs)
            provider.current_request=ids[0];provider.inside+=1
            try:
                with Consumers():return original(scheduler,*args,**kwargs)
            except BaseException as error:
                provider.failure=str(error);raise
            finally:provider.inside-=1
        self.runner.execute_model=execute;self.hook=execute

    def close(self):
        self.boundary()
        if self.hook is not None:
            require(vars(self.runner).get('execute_model') is self.hook,'V2_EXAMPLE_HOOK_OWNER_CHANGED')
            del self.runner.execute_model;self.hook=None
        if self.regions is not None:
            for name in tuple(self.regions.entries):
                e=self.regions.entries[name]
                if not e['events']:self.regions.release_unwritten(name)
                elif e['readers']:self.regions.restore(name)
                else:self.regions.rollback_unconsumed_weight(name)
        if self.invalidated:
            require(not hasattr(self.module,'weight_scale_fp32') and exact(self.derived,self.derived_backup),
                    'V2_DERIVED_SCALE_CHANGED_NO_OVERWRITE')
            self.module.weight_scale_fp32=self.derived;self.invalidated=False
        self.enabled=False

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation);reject_unknown(settings)
        if action=='prepare':self.boundary()
        elif action=='register':
            leases=RegionLeases(owner,generation,stream=torch.npu.current_stream,complete=lambda s:s.synchronize())
            self.regions=PersistentRegions(leases,boundary=self.boundary,maximum_bytes=1024*1024)
            self.regions.prepare([PersistentRegion('scale',self.scale,'persistent_weight',self.allocation,
                lambda:identity(self.module.weight_scale),('npu::npu_quant_matmul:scale',),())])
            self._registered=True
        elif action=='check':
            self.boundary();require(self._registered,'V2_EXAMPLE_REGISTER_REQUIRED')
            require(self.module.weight_scale_fp32 is self.derived,'V2_DERIVED_SCALE_REPLACED')
        elif action=='enable':
            self.boundary();self.install()
            # Audited apply uses only weight_scale for this nonchunked scheme.
            # Invalidate the unused derived copy instead of silently keeping a
            # stale visible copy; unknown access paths fail, never get invented.
            del self.module.weight_scale_fp32;self.invalidated=True
            self.regions.write('scale',self.scale*0.5);self.enabled=True
        elif action in ('disable','unregister','clean','abort','recover'):
            self.close()
            if action in ('unregister','clean'):self._registered=False
        else:raise ValueError('V2_EXAMPLE_UNSUPPORTED_ACTION')
        return self.query(owner=owner,generation=generation)

    def query(self,*,owner,generation):
        self.verify(owner,generation)
        count=len(self.regions.entries) if self.regions else 0
        events=self.regions.snapshot() if self.regions else dict(events=[],reader_observations=0)
        paused=self.context['boundary']()['accepting'] is False
        torch.npu.current_stream().synchronize()
        return dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',hook_state_available=True,
            hook_owned=self.hook is not None,owner=owner,generation=generation,native_loaded=False,native_registered=False,
            registered=self._registered,enabled=self.enabled,inflight=self.inside,leases=count,persistent_leases=count,
            persistent_protocol_version=2,owned_hooks=int(self.hook is not None),device_readers_complete=self.inside==0,
            exclusive_ingress_paused=paused,replay_pending=False,replay_failed=self.failure is not None and self.enabled,
            failure=self.failure,selected_site=self.label,derived_copy='invalidated' if self.invalidated else 'original_object',
            events=events['events'],reader_observations=events['reader_observations'],readers=list(self.requests),
            recovery=dict(bytes_restored=count==0 and not self.invalidated,
                resources_zero=count==0 and self.hook is None and not self._registered),qualification_granted=False)
