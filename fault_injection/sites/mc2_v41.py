"""Read-only 0.32 MC2 dispatch/combine join for the existing EP control boundary.

CPU-prepared adapter, not a fault8 qualification. Does not decode opaque MC2
assist buffers or add communication at conditional operator calls.
"""
import hashlib,inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode,_disable_current_modes
from .mla28 import require

SOURCE_SHA='43e57927ed54a2d2a1193ccf8bbcd56d1751417e460954bc0b8b967d2a23b228'


def integer_rows(t):
    require(isinstance(t,torch.Tensor) and t.dtype in (torch.int32,torch.int64),'MC2_INTEGER_ROUTING_REQUIRED')
    return t.detach().cpu().tolist()


class Join:
    def __init__(self,inputs,kwargs,output,context):
        require(kwargs['expert_ids'] is inputs.topk_ids and kwargs['x'] is inputs.hidden_states,'MC2_ACTUAL_DISPATCH_INPUT_JOIN')
        require(inputs.routing.global_redundant_expert_num==0,'MC2_EPLB_PHYSICAL_PLACEMENT_UNADAPTED')
        require(kwargs['expert_shard_type']==0 and kwargs['shared_expert_rank_num']==0,'MC2_SHARD_MODE_UNADAPTED')
        require(len(output)>=7,'MC2_DISPATCH_RETURN_SCHEMA')
        self.ids=integer_rows(inputs.topk_ids);self.map=integer_rows(inputs.routing.expert_map)
        self.weights=inputs.topk_weights.detach().float().cpu().clone()
        require(inputs.topk_ids.ndim==2 and inputs.routing.expert_map.ndim==1 and len(self.ids)==len(context['rows']),
            'MC2_ACTUAL_ROW_MAPPING_REQUIRED')
        mask=kwargs.get('x_active_mask')
        active=[True]*len(self.ids) if mask is None else mask.detach().cpu().tolist()
        require(len(active)==len(self.ids) and all(type(x) is bool for x in active),'MC2_ACTIVE_MASK_LAYOUT')
        require(context.get('owner') and type(context.get('generation')) is int and context['generation']>0
            and type(context.get('invocation')) is int and context['invocation']>=0,'MC2_SCOPE_IDENTITY')
        self.routes=[]
        for i,(row,enabled) in enumerate(zip(context['rows'],active)):
            if not enabled:
                require(row is None,'MC2_PADDING_HAS_REQUEST');continue
            require(type(row) is dict and row.get('request_id') and type(row.get('position')) is int and row['position']>=0,
                'MC2_ACTIVE_ROW_WITHOUT_REQUEST')
            for ordinal,expert in enumerate(self.ids[i]):
                require(type(expert) is int and 0<=expert<len(self.map),'MC2_EXPERT_ID_OUT_OF_RANGE')
                self.routes.append(dict(request_id=row['request_id'],position=row['position'],route=ordinal,expert=expert,source_row=i))
        self.input=inputs;self.kwargs=kwargs;self.output=output;self.context=dict(context);self.metadata=None

    def bind_metadata(self,meta):
        require(meta.topk_ids is self.input.topk_ids and meta.topk_weights is self.input.topk_weights
            and meta.expert_map is self.input.routing.expert_map,'MC2_METADATA_ROUTING_JOIN')
        for name,index in (('assist_info_for_combine',2),('ep_recv_counts',4),('tp_recv_counts',5),('expand_scales',6)):
            require(getattr(meta,name) is self.output[index],'MC2_METADATA_NATIVE_OUTPUT_JOIN:'+name)
        self.metadata=meta

    def combine(self,hidden,meta,kwargs,result):
        require(meta is self.metadata and kwargs['expand_x'] is hidden,'MC2_COMBINE_SCOPE_OR_EXPERT_OUTPUT')
        require(kwargs['expert_ids'] is self.input.topk_ids and integer_rows(kwargs['expert_ids'])==self.ids,
            'MC2_ROUTING_CHANGED_AFTER_DISPATCH')
        require(integer_rows(meta.expert_map)==self.map,'MC2_PLACEMENT_CHANGED_DURING_CALL')
        for k in ('group_ep','ep_world_size','ep_rank_id','expert_shard_type','shared_expert_rank_num','moe_expert_num','global_bs'):
            require(kwargs[k]==self.kwargs[k],'MC2_COMMUNICATION_IDENTITY_CHANGED:'+k)
        require(kwargs['assist_info_for_combine'] is self.output[2] and kwargs['ep_send_counts'] is self.output[4]
            and kwargs['expand_scales'] is self.output[6],'MC2_OPAQUE_ASSIST_NOT_ORIGINAL')
        require(torch.equal(kwargs['expert_scales'].detach().float().cpu(),self.weights),'MC2_COMBINE_WEIGHTS_CHANGED')
        require(isinstance(result,torch.Tensor) and result.shape==self.input.hidden_states.shape,'MC2_COMBINE_ROW_GEOMETRY')
        return dict(owner=self.context['owner'],generation=self.context['generation'],invocation=self.context['invocation'],
            source_ep_rank=self.kwargs['ep_rank_id'],ep_world_size=self.kwargs['ep_world_size'],routes=self.routes,
            local_expert_map=self.map,dispatch_calls=1,combine_calls=1,opaque_assist_preserved=True,
            destination_expert_row_decode_proven=False,qualification_granted=False)


class NativeCalls(TorchDispatchMode):
    def __init__(self,operator):super().__init__();self.operator=operator;self.calls=[]
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        if func._schema.name=='npu::'+self.operator:
            with _disable_current_modes():
                actual={a.name:args[i] if i<len(args) else (kwargs or {}).get(a.name,a.default_value)
                    for i,a in enumerate(func._schema.arguments)}
                self.calls.append((actual,result))
        return result


class MC2Observer:
    def __init__(self,context_provider):
        self.context_provider=context_provider;self.pending={};self.records=[];self.methods=[]
    def install(self,cls):
        require(cls.__name__=='TokenDispatcherWithMC2' and hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()==SOURCE_SHA,
            'MC2_DISPATCHER_SOURCE_UNREVIEWED')
        dispatch=cls.token_dispatch;combine=cls.token_combine
        require(not getattr(dispatch,'_dcat_mc2',False) and not getattr(combine,'_dcat_mc2',False),'MC2_ALREADY_OWNED')
        def produced(dispatcher,token_dispatch_input):
            inputs=token_dispatch_input
            context=self.context_provider(dispatcher,inputs)
            if context is None:return dispatch(dispatcher,inputs)
            require(id(dispatcher) not in self.pending,'MC2_CALL_OVERLAP')
            mode=NativeCalls('npu_moe_distribute_dispatch_v2')
            with mode:result=dispatch(dispatcher,inputs)
            require(len(mode.calls)==1,'MC2_ACTUAL_NATIVE_DISPATCH_MISSING')
            args,output=mode.calls[0];join=Join(inputs,args,output,context);join.bind_metadata(result.combine_metadata)
            self.pending[id(dispatcher)]=join
            return result
        def consumed(dispatcher,hidden_states,combine_metadata,bias=None):
            hidden=hidden_states;metadata=combine_metadata
            join=self.pending.get(id(dispatcher))
            if join is None:return combine(dispatcher,hidden,metadata,bias)
            mode=NativeCalls('npu_moe_distribute_combine_v2')
            with mode:result=combine(dispatcher,hidden,metadata,bias)
            require(len(mode.calls)==1,'MC2_ACTUAL_NATIVE_COMBINE_MISSING')
            args,out=mode.calls[0]
            self.records.append(join.combine(hidden,metadata,args,out));del self.pending[id(dispatcher)]
            return result
        for name,original,fn in (('token_dispatch',dispatch,produced),('token_combine',combine,consumed)):
            fn._dcat_mc2=True;setattr(cls,name,fn);self.methods.append((cls,name,original,fn))
    def close(self):
        require(not self.pending,'MC2_UNCOMBINED_REQUEST_REMAINS')
        require(all(getattr(c,n) is fn for c,n,_,fn in self.methods),'MC2_WRAPPER_OWNER_CHANGED')
        for c,n,original,_ in reversed(self.methods):setattr(c,n,original)
        self.methods.clear()
