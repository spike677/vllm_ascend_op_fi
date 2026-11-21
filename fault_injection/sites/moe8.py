"""0.26 actual AllGather MoE combine index/probability group remap.

Dynamic routing is observed per call. No baseline route list is used as the
fault denominator. EP/MC2/All2All are not qualified by this TP-only adapter.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .mla28 import require,fingerprint

DISPATCHER_SHA='80c7061b59d4593f55ce5c18d601bdc7b1831bfe9c268c5bc8129d733d09c215'
PIPELINE_SHA='043f3c28518254347589f7af03793810262d0b29a9a82ca17dfd5fe5f9559f2e'
RUNNER_SHA='b24da9cc65bbf7e2dbf6bfaee403435d88bbe125a9bd589e319988fc7be143a6'


def remap_groups(indices,probs):
    require(indices.ndim==1 and probs.ndim==2 and indices.numel()==probs.numel()
        and indices.dtype in (torch.int32,torch.int64) and indices.is_contiguous() and probs.is_contiguous(),
        'MOE8_REAL_GROUP_LAYOUT_REQUIRED')
    tokens,topk=probs.shape
    require(tokens>0 and topk>0,'MOE8_EMPTY_GROUP')
    require(torch.equal(indices.long().sort().values,torch.arange(indices.numel(),device=indices.device)),
        'MOE8_ACTUAL_COMPLETE_PERMUTATION_REQUIRED')
    groups=(torch.arange(tokens,device=indices.device)+1).clamp_max(tokens-1)
    return indices.reshape(tokens,topk).index_select(0,groups).reshape(-1),probs.index_select(0,groups),groups


class CombineMode(TorchDispatchMode):
    def __init__(self,h,i,p):super().__init__();self.inputs=(h,i,p);self.calls=0;self.output=None;self.index_consumer=i;self.abs_calls=0
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func._schema.name=='aten::abs' and args and args[0] is self.inputs[1]:
            require(self.abs_calls==0 and self.calls==0,'MOE8_INDEX_TRANSFORM_ORDER')
            self.index_consumer=func(*args,**kwargs);self.abs_calls+=1
            return self.index_consumer
        if func._schema.name!='npu::npu_moe_token_unpermute':return func(*args,**kwargs)
        values={a.name:(args[n] if n<len(args) else kwargs.get(a.name,a.default_value))
            for n,a in enumerate(func._schema.arguments)}
        require(self.calls==0 and all(values.get(k) is t for k,t in zip(('permuted_tokens','sorted_indices','probs'),(self.inputs[0],self.index_consumer,self.inputs[2])))
            and not values.get('padded_mode',False) and values.get('restore_shape') is None,'MOE8_REAL_NATIVE_CONSUMER_CHANGED')
        self.calls+=1;self.output=func(*args,**kwargs);return self.output


class MoE8Site:
    def __init__(self,model,runner,*,fault,region_leases):
        self.model=model;self.runner=runner;self.fault=fault;self.region_leases=region_leases
        self.events=[];self.sites=[];self.hooks=[];self.patches=[];self.uncovered=[];self.excluded=[]
        self.parents=[];self.forward_calls=[];self.active=False;self.pipelines={}
    @property
    def pending(self):return self.region_leases.count()
    def install(self):
        from vllm_ascend.ops.fused_moe import token_dispatcher as module
        from vllm_ascend.ops.fused_moe import moe_comm_method as pipeline
        from vllm_ascend.ops.fused_moe import fused_moe as runner_module
        require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==DISPATCHER_SHA,'UNREVIEWED_MOE_DISPATCHER_SOURCE')
        require(hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest()==PIPELINE_SHA,'UNREVIEWED_MOE_PIPELINE_SOURCE')
        require(hashlib.sha256(Path(runner_module.__file__).read_bytes()).hexdigest()==RUNNER_SHA,'UNREVIEWED_MOE_RUNNER_SOURCE')
        require(self.runner.vllm_config.parallel_config.enable_expert_parallel is False,'MOE8_EP_ADAPTER_PENDING')
        def enter(*unused):
            require(not self.active and not self.parents and not self.pending,'MOE8_ROOT_OVERLAP')
            self.active=True;self.forward_calls.append({})
        def leave(*unused):
            require(not self.parents and not self.pending and not self.pipelines,'MOE8_ROOT_INCOMPLETE');self.active=False
        self.hooks.extend([self.model.register_forward_pre_hook(enter),self.model.register_forward_hook(leave)])
        for name,m in self.model.named_modules():
            if not isinstance(m,runner_module.AscendMoERunner):continue
            require(hasattr(m,'top_k') and callable(getattr(m,'no_shared_forward_impl',None)),
                'MOE8_REVIEWED_RUNNER_INTERFACE_CHANGED')
            self.sites.append(name)
            def before(mod,args,kwargs,_name=name):self.parents.append((_name,mod))
            def after(mod,args,out,_name=name):
                require(len(self.parents)==1 and self.parents[0][0]==_name,'MOE8_MODULE_PARENT_ORDER');self.parents.pop()
            self.hooks.extend([m.register_forward_pre_hook(before,with_kwargs=True),m.register_forward_hook(after)])
        require(self.sites,'MOE8_NO_ACTUAL_MODULE_CANDIDATES')
        for cls in (module.TokenDispatcherWithAllGather,module.TokenDispatcherWithMC2,module.TokenDispatcherWithAll2AllV):
            original=cls.token_combine
            require(inspect.getsourcefile(original)==module.__file__ and not getattr(original,'_dcat_owned',False),'MOE8_COMBINE_OWNER_CHANGED')
            def wrapped(dispatcher,hidden_states,combine_metadata,bias=None,_original=original,_cls=cls):
                if not self.active:return _original(dispatcher,hidden_states,combine_metadata,bias)
                require(_cls is module.TokenDispatcherWithAllGather,'MOE8_ACTUAL_NON_ALLGATHER_UNSUPPORTED')
                return self.combine(_original,dispatcher,hidden_states,combine_metadata,bias)
            wrapped._dcat_owned=True;setattr(cls,'token_combine',wrapped);self.patches.append((cls,'token_combine',original,wrapped))
        cls=module.TokenDispatcherWithAllGather;dispatch=cls.token_dispatch
        require(inspect.getsourcefile(dispatch)==module.__file__ and not getattr(dispatch,'_dcat_owned',False),'MOE8_PRODUCER_OWNER_CHANGED')
        def routing(dispatcher,token_dispatch_input):
            if not self.active:return dispatch(dispatcher,token_dispatch_input)
            require(len(self.parents)==1 and id(dispatcher) not in self.pipelines,'MOE8_ROUTING_OVERLAP')
            result=dispatch(dispatcher,token_dispatch_input)
            self.pipelines[id(dispatcher)]=dict(dispatcher=dispatcher,metadata=result.combine_metadata,
                site=self.parents[0][0],request_ids=list(self.runner.input_batch.req_ids),
                topk_ids_sha256=fingerprint(token_dispatch_input.topk_ids),mlp_output=None)
            return result
        routing._dcat_owned=True;cls.token_dispatch=routing;self.patches.append((cls,'token_dispatch',dispatch,routing))
        cls=pipeline.MoECommMethod;mlp=cls._apply_mlp
        require(inspect.getsourcefile(mlp)==pipeline.__file__ and not getattr(mlp,'_dcat_owned',False),'MOE8_MLP_OWNER_CHANGED')
        def expert(comm,mlp_compute_input):
            result=mlp(comm,mlp_compute_input)
            if self.active:
                row=self.pipelines.get(id(comm.token_dispatcher))
                require(row is not None and row['mlp_output'] is None,'MOE8_MLP_WITHOUT_ROUTING')
                row['mlp_output']=result[0]
            return result
        expert._dcat_owned=True;cls._apply_mlp=expert;self.patches.append((cls,'_apply_mlp',mlp,expert))
        return self.sites

    def combine(self,original,dispatcher,h,metadata,bias):
        require(len(self.parents)==1 and bias is None,'MOE8_REAL_MODEL_PARENT_REQUIRED')
        name,mod=self.parents[0];ids=list(self.runner.input_batch.req_ids)
        require(len(ids)==1,'MOE8_SINGLE_REAL_REQUEST_MAPPING_REQUIRED')
        produced=self.pipelines.get(id(dispatcher))
        require(produced is not None and produced['dispatcher'] is dispatcher and produced['metadata'] is metadata
            and produced['mlp_output'] is h and produced['site']==name and produced['request_ids']==ids,
            'MOE8_CURRENT_ROUTING_MLP_COMBINE_JOIN_UNPROVEN')
        i,p=metadata.expanded_row_idx,metadata.topk_weights
        require(h.ndim==2 and h.shape[0]==i.numel() and p.shape[1]==mod.top_k,
            'MOE8_ACTUAL_EXPERT_OUTPUT_LAYOUT')
        require(tuple(metadata.restore_shape)==(p.shape[0],h.shape[1]),'MOE8_RESTORE_GEOMETRY')
        # Same request mapping for each actual routed token; recompute from
        # scheduler CPU mirror on every prefill/decode, never baseline routing.
        from vllm.forward_context import get_forward_context
        meta=get_forward_context().attn_metadata
        rows=list(meta.values()) if isinstance(meta,dict) else [meta]
        actual={int(m.num_actual_tokens) for m in rows if hasattr(m,'num_actual_tokens')}
        require(actual=={p.shape[0]},'MOE8_ROUTED_TOKEN_REQUEST_MAPPING_UNPROVEN')
        with _disable_current_modes():
            ni,np,groups=remap_groups(i,p)
            require(torch.isfinite(h).all().item() and torch.isfinite(p).all().item(),'MOE8_NONFINITE_ACTUAL_INPUT')
            e=dict(site=name,request_ids=ids,eligible=True,applied=False,changed=0,consumed=False,restored=False,failed=False,
                token_rows=p.shape[0],topk=p.shape[1],row_request_ids=[ids[0]]*p.shape[0],
                original_index_sha256=fingerprint(i),original_probs_sha256=fingerprint(p),
                remapped_groups=groups.cpu().tolist(),operator='npu::npu_moe_token_unpermute')
            e.update(actual_producer_join=True,topk_ids_sha256=produced['topk_ids_sha256'])
            self.events.append(e);calls=self.forward_calls[-1];calls[name]=calls.get(name,0)+1
            if self.fault:
                e['before_sha256']=[fingerprint(i),fingerprint(p)]
                tokens=[self.region_leases.acquire(t) for t in (i,p)]
                e['changed']=int((i!=ni).sum().item())+int((p!=np).sum().item())
                for token,value in zip(tokens,(ni,np)):self.region_leases.write(token,value)
                e['applied']=True
        mode=CombineMode(h,i,p)
        with mode:out=original(dispatcher,h,metadata,bias)
        require(mode.calls==1 and out is mode.output,'MOE8_NATIVE_COMBINE_NOT_OBSERVED')
        if self.fault:
            for token in tokens:self.region_leases.consumed(token);self.region_leases.restore(token)
            e.update(consumed=True,restored=True,restored_sha256=[fingerprint(i),fingerprint(p)])
            require(e['restored_sha256']==e['before_sha256'],'MOE8_INPUT_BYTES_NOT_RESTORED')
        e.update(native_calls=mode.calls,index_abs_calls=mode.abs_calls,output_sha256=fingerprint(out))
        del self.pipelines[id(dispatcher)]
        return out

    def close(self):
        require(not self.pending and not self.parents and not self.pipelines,'MOE8_RESOURCES_REMAIN')
        for cls,name,original,wrapped in reversed(self.patches):
            require(getattr(cls,name) is wrapped,'MOE8_WRAPPER_OWNER_CHANGED');setattr(cls,name,original)
        self.patches.clear()
        for h in self.hooks:h.remove()
        self.hooks.clear()
