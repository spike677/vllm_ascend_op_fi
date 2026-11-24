"""Dedicated B1 persistent V-cache example, 16 observed full-attention layers.

Writes one real value cell per layer/request after its native producer. Later
decode reads keep the fault until the actual scheduler release barrier. This is
an acceptance example, not a public deployment registration or MLA adapter.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from fault_injection.engines.region_lease import RegionLeases,identity,exact
from demoncat.runtime.vllm_ascend.worker_extension.persistent_kv_regions import RequestKVRegions
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import Scope,reject_unknown
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require

ATTENTION_SHA='4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076'


class KVProvider(Scope):
    persistent_protocol_version=2
    continuous_protocol=True
    execution_boundary='python_hook'
    def __init__(self,worker,model,worker_identity,*,mechanism,region_kind):
        super().__init__(worker,model,worker_identity)
        require(region_kind=='persistent_kv','V2_EXAMPLE_KV_ONLY')
        self.region_kind=region_kind;self.worker=worker;self.runner=worker.model_runner
        self.context=worker._v2_weight_example_context;self.enabled=False;self.patches=[];self.inside=0
        self.regions=None;self.failure=None;self.readers=[];self.slot_by_name={};self.current_request=None
        modules=dict(model.named_modules());self.sites=[]
        for name in self.context['census']['full_attention_modules']:
            impl=modules[name].impl;file=inspect.getsourcefile(type(impl))
            require(type(impl).__name__=='AscendAttentionBackendImpl' and file
                    and hashlib.sha256(Path(file).read_bytes()).hexdigest()==ATTENTION_SHA,'KV_ACTUAL_IMPL_SOURCE_CHANGED')
            require(getattr(impl,'kv_sharing_target_layer_name',None) is None,'KV_SHARED_LAYER_NOT_SUPPORTED')
            self.sites.append((name,impl))
        require(len(self.sites)==16,'KV_16_OBSERVED_FULL_LAYERS_REQUIRED')

    def owner_observation(self):
        p=self.worker._v2_kv_notice
        require(p['request_id']==self.current_request,'KV_RUNNER_ENGINE_REQUEST_MISMATCH')
        require(sorted(p['full_attention_layers'])==sorted(n for n,_ in self.sites),'KV_ENGINE_LAYER_MAPPING_CHANGED')
        return p

    def boundary(self):
        require(self.inside==0,'KV_EXAMPLE_FORWARD_ACTIVE')
        b=self.context['boundary']();require(b['accepting'] is False and b['inflight']==0,'KV_HOST_DRAIN_REQUIRED')
        torch.npu.current_stream().synchronize()

    def after_writer(self,name,impl,arguments):
        notice=self.owner_observation();cache=impl.value_cache
        require(isinstance(cache,torch.Tensor) and cache.ndim==4 and cache.is_contiguous()
                and cache.dtype==torch.bfloat16,'KV_EXAMPLE_CONTIGUOUS_BF16_4D_REQUIRED')
        metadata=arguments.get('attn_metadata')
        slots=arguments.get('slot_mapping') if metadata is None else metadata.slot_mapping[:metadata.num_actual_tokens]
        require(isinstance(slots,torch.Tensor) and slots.numel()>0,'KV_ACTUAL_SLOTS_REQUIRED')
        slots=slots.detach().reshape(-1).cpu().tolist();require(all(s>=0 for s in slots),'KV_PADDING_SLOT_UNSUPPORTED')
        if name in self.regions.entries:
            require(self.slot_by_name[name] not in slots,'KV_HELD_SLOT_LEGAL_REWRITE_REFUSED')
            self.regions.acknowledge_disjoint_writer(name);return
        slot=int(slots[0]);block_size=cache.shape[1];page=slot//block_size
        table=self.runner.input_batch.block_table.block_tables[0]
        allocation=notice['allocation_block_sizes']['0']
        require(table.physical_block_size==allocation and table.block_size==block_size
                and allocation%block_size==0 and table.blocks_per_phys_block==allocation//block_size
                and table.dcp_world_size==1,'KV_ACTUAL_HYBRID_BLOCK_MAPPING_REQUIRED')
        physical_page=page//table.blocks_per_phys_block
        require(slot%block_size==0,'KV_EXAMPLE_FIRST_REQUEST_TOKEN_REQUIRED')
        require(0<=page<cache.shape[0] and [0,physical_page] in notice['pages'],'KV_WRITER_SLOT_NOT_OWNED')
        require(all(g==0 for g,p in notice['pages']),'KV_EXAMPLE_SINGLE_FULL_GROUP_ONLY')
        region=cache[page,slot%block_size,0,:1]
        self.slot_by_name[name]=slot
        self.regions.write_after_producer(name,region,region.detach()+1,page=[0,physical_page])
        self.regions.entries[name]['kernel_page']=page
        self.regions.entries[name]['event'].update(request_id=self.current_request,physical_page=physical_page,
            kernel_page=page,kernel_block_size=block_size,allocation_block_size=allocation)

    def install(self):
        provider=self
        for name,impl in self.sites:
            for method in ('reshape_and_cache','do_kv_cache_update','forward_impl'):
                require(method not in vars(impl),'KV_IMPL_ALREADY_WRAPPED')
                original=getattr(impl,method);signature=inspect.signature(original)
                if method!='forward_impl':
                    def writer(*args,_name=name,_impl=impl,_original=original,_sig=signature,**kwargs):
                        bound=_sig.bind(*args,**kwargs);result=_original(*args,**kwargs)
                        provider.after_writer(_name,_impl,bound.arguments);return result
                    wrapper=writer
                else:
                    class Consumer(TorchDispatchMode):
                        def __init__(self,site):super().__init__();self.site=site
                        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                            kwargs=kwargs or {};a={p.name:v for p,v in zip(func._schema.arguments,args)};a.update(kwargs)
                            entry=provider.regions.entries.get(self.site);matched=False
                            if entry is not None and 'fused_infer_attention_score' in func._schema.name:
                                vals=a.get('value');vals=vals if isinstance(vals,(list,tuple)) else [vals]
                                table=a.get('block_table')
                                matched=any(isinstance(v,torch.Tensor) and v.untyped_storage()._cdata==entry['tensor'].untyped_storage()._cdata for v in vals)
                                if matched:
                                    require(isinstance(table,torch.Tensor) and table.ndim==2 and table.shape[0]==1,'KV_READER_BLOCK_TABLE_REQUIRED')
                                    require(int(table[0,0].item())==entry['kernel_page'],'KV_READER_PAGE_NOT_FIRST_REQUEST_BLOCK')
                                    seq=a.get('actual_seq_lengths_kv');seq=seq.detach().cpu().tolist() if isinstance(seq,torch.Tensor) else seq
                                    require(isinstance(seq,(list,tuple)) and len(seq)==1 and seq[0]>0,'KV_READER_ACTUAL_LENGTH_REQUIRED')
                                    lease=provider.regions.leases._get(entry['token'])
                                    require(exact(entry['tensor'],lease['expected']),'KV_READER_BYTES_MISMATCH')
                            result=func(*args,**kwargs)
                            if matched:
                                provider.regions.consumed(self.site)
                                provider.readers.append(dict(site=self.site,request_id=provider.current_request,
                                    consumer=func._schema.name,changed_cache_value_read=True,device_read_complete=True))
                            return result
                    def consumer(*args,_name=name,_original=original,**kwargs):
                        with Consumer(_name):return _original(*args,**kwargs)
                    wrapper=consumer
                setattr(impl,method,wrapper);self.patches.append((impl,method,wrapper))
        require('execute_model' not in vars(self.runner),'KV_RUNNER_ALREADY_OWNED')
        original=self.runner.execute_model
        def execute(scheduler,*args,**kwargs):
            ids=list(scheduler.num_scheduled_tokens)
            require(len(ids)<=1,'KV_EXAMPLE_SERIAL_ONLY')
            if not ids:return original(scheduler,*args,**kwargs)
            self.current_request=ids[0];self.owner_observation();self.inside+=1
            try:return original(scheduler,*args,**kwargs)
            except BaseException as error:self.failure=str(error);raise
            finally:self.inside-=1
        self.runner.execute_model=execute;self.patches.append((self.runner,'execute_model',execute))
        require(not hasattr(self.worker,'_v2_active_kv_provider'),'KV_PROVIDER_ALREADY_ACTIVE')
        self.worker._v2_active_kv_provider=self

    def before_free(self):
        require(self.inside==0,'KV_READER_STILL_ACTIVE')
        self.regions.restore_before_free();self.slot_by_name={}
        return dict(bytes_restored=not self.regions.entries,request_leases=self.regions.leases.count())

    def close(self):
        self.boundary();require(not self.regions or not self.regions.entries,'KV_REQUEST_END_RESTORE_REQUIRED')
        for obj,name,wrapper in reversed(self.patches):
            require(vars(obj).get(name) is wrapper,'KV_HOOK_OWNER_CHANGED');delattr(obj,name)
        self.patches=[]
        if getattr(self.worker,'_v2_active_kv_provider',None) is self:del self.worker._v2_active_kv_provider
        self.enabled=False

    def dispatch(self,action,settings,*,owner,generation):
        if action=='prepare':self.bind(owner,generation)
        self.verify(owner,generation);reject_unknown(settings)
        if action=='prepare':self.boundary()
        elif action=='register':
            leases=RegionLeases(owner,generation,stream=torch.npu.current_stream,complete=lambda s:s.synchronize())
            self.regions=RequestKVRegions(leases,owner_observation=self.owner_observation,maximum_bytes=4096);self._registered=True
        elif action=='check':self.boundary();require(self._registered,'KV_REGISTER_REQUIRED')
        elif action=='enable':self.boundary();self.install();self.enabled=True
        elif action in ('disable','unregister','clean','abort','recover'):
            self.close()
            if action in ('unregister','clean'):self._registered=False
        else:raise ValueError('KV_ACTION_UNSUPPORTED')
        return self.query(owner=owner,generation=generation)

    def query(self,*,owner,generation):
        self.verify(owner,generation);count=len(self.regions.entries) if self.regions else 0
        torch.npu.current_stream().synchronize()
        return dict(execution_boundary='python_hook',state_source='OWNED_PYTHON_HOOKS',hook_state_available=True,
            hook_owned=bool(self.patches),owner=owner,generation=generation,native_loaded=False,native_registered=False,
            registered=self._registered,enabled=self.enabled,inflight=self.inside,leases=count,persistent_leases=count,
            persistent_protocol_version=2,owned_hooks=len(self.patches),device_readers_complete=self.inside==0,
            exclusive_ingress_paused=self.context['boundary']()['accepting'] is False,replay_pending=False,
            replay_failed=self.failure is not None and self.enabled,failure=self.failure,
            selected_sites=[n for n,_ in self.sites],events=list(self.regions.events) if self.regions else [],
            reader_observations=self.regions.reads if self.regions else 0,readers=list(self.readers),
            recovery=dict(bytes_restored=count==0,resources_zero=count==0 and not self.patches and not self._registered),
            qualification_granted=False)
