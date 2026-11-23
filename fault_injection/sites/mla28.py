"""Pinned eager MLA real-call adapter for the denominator replay torch engine.

No public registration. One actual request, one prefill chunk, no speculation,
no FA quantization. All discovered MLA implementations and all their real calls
participate; each call changes the last query row. Decode uses absorbed latent
KV, then the implementation's real V-up projection and real o_proj consumer.
"""
import hashlib
import inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from fault_injection.engines.attention import recompute_attention_row

SOURCE_SHA = '67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545'


def require(ok, reason):
    if not ok: raise RuntimeError(reason)


def fingerprint(t):
    t=t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_consumer_row(actual, replacement, row):
    """The real projection must read the leased row, not an equal copy."""
    require(actual.shape == replacement.shape and actual.dtype == replacement.dtype
            and actual.device == replacement.device, 'REAL_OPROJ_INPUT_LAYOUT_MISMATCH')
    a, b = actual[row], replacement[row]
    require(a.untyped_storage()._cdata == b.untyped_storage()._cdata
            and a.storage_offset() == b.storage_offset() and a.stride() == b.stride(),
            'REAL_OPROJ_STORAGE_NOT_LEASED')
    require(torch.equal(a, b), 'REAL_OPROJ_INPUT_MISMATCH')


class MLACopyMode(TorchDispatchMode):
    """Observe the source's real copy into o_proj_input, never create a copy."""
    def __init__(self,site):super().__init__();self.site=site
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        matched=[]
        if func._schema.name=='aten::copy_' and len(args)>=2:
            dst,src=args[:2]
            for lease in self.site.pending.values():
                value=lease['replacement']
                if (isinstance(src,torch.Tensor) and src.untyped_storage()._cdata==value.untyped_storage()._cdata
                    and src.storage_offset()==value.storage_offset() and src.shape==value.shape and src.stride()==value.stride()):
                    require('consumer_input' not in lease,'MLA_REPLACEMENT_MULTIPLE_COPIES')
                    matched.append(lease)
        result=func(*args,**(kwargs or {}))
        with _disable_current_modes():
            for lease in matched:
                require(dst.shape==src.shape and dst.dtype==src.dtype and torch.equal(dst,src),'MLA_REAL_COPY_NOT_EXACT')
                lease['consumer_input']=dst
                lease['event'].update(copy_operator=func._schema.name,copy_exact=True)
        return result


class MLA28Site:
    def __init__(self, model, runner, *, fault, block_size=128,region_leases=None):
        self.model=model; self.runner=runner; self.fault=fault; self.block_size=block_size
        self.region_leases=region_leases
        self.events=[]; self.methods=[]; self.hooks=[]; self.pending={}; self.sites=[]

    def install(self):
        seen=set()
        for name,module in self.model.named_modules():
            impl=getattr(module,'impl',None)
            if impl is None or type(impl).__name__!='AscendMLAImpl' or id(impl) in seen: continue
            seen.add(id(impl)); self.sites.append(name)
            require(hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()==SOURCE_SHA,
                    'UNREVIEWED_MLA_SOURCE')
            require(not impl.fa_quant_layer and not impl.use_output_gate and impl.speculative_config is None,
                    'MLA_VARIANT_NOT_SUPPORTED')
            require('forward' not in vars(impl),'MLA_ALREADY_WRAPPED')
            original_forward=impl.forward
            def forward(*args,_original=original_forward,**kwargs):
                with MLACopyMode(self):return _original(*args,**kwargs)
            impl.forward=forward;self.methods.append((impl,'forward',forward))
            for method in ('_forward_prefill','_forward_decode'):
                require(method not in vars(impl),'MLA_ALREADY_WRAPPED')
                original=getattr(impl,method)
                def wrapped(*args,_name=name,_impl=impl,_method=method,_original=original,**kwargs):
                    require(not kwargs,'MLA_POSITIONAL_BOUNDARY_CHANGED')
                    return self.call(_name,_impl,_method,_original,args)
                setattr(impl,method,wrapped);self.methods.append((impl,method,wrapped))
            def pre(mod,args,_name=name):
                lease=self.pending.get(_name)
                if lease is None:return
                x=args[0]; idx=lease['row']
                require('consumer_input' in lease,'MLA_NATIVE_COPY_NOT_OBSERVED')
                verify_consumer_row(x,lease['consumer_input'],idx)
                require(torch.equal(x[idx],lease['replacement'][idx]),'REAL_OPROJ_INPUT_MISMATCH')
                require(torch.npu.current_stream()==lease['stream'],'CONSUMER_STREAM_CHANGED')
                lease['event']['consumer_entered']=True
                lease['event']['consumer_module_type']=type(mod).__name__
            def post(mod,args,out,_name=name):
                lease=self.pending.get(_name)
                if lease is None:return
                require(lease['event'].get('consumer_entered'),'OPROJ_CONSUMPTION_NOT_OBSERVED')
                completion=torch.npu.Event();completion.record(lease['stream']);completion.synchronize()
                lease['event'].update(consumed=True,consumer_operator='module::'+type(mod).__name__+'.forward',
                    consumer_site=_name+'.impl.o_proj',consumer_storage_verified=True,
                    consumer_device_completion='SAME_STREAM_EVENT_SYNCHRONIZED')
                if self.region_leases is None:lease['replacement'][lease['row']].copy_(lease['backup'])
                else:
                    self.region_leases.consumed(lease['token']);self.region_leases.restore(lease['token'])
                restored=torch.npu.Event();restored.record(lease['stream']);restored.synchronize()
                require(torch.equal(lease['replacement'][lease['row']],lease['backup']),'SCRATCH_BYTES_NOT_RESTORED')
                lease['event']['restored_sha256']=fingerprint(lease['replacement'][lease['row']])
                require(lease['event']['restored_sha256']==lease['event']['before_sha256'],'SCRATCH_BYTE_HASH_MISMATCH')
                lease['event']['restored']=True
                del self.pending[_name]
            self.hooks.extend([impl.o_proj.register_forward_pre_hook(pre),impl.o_proj.register_forward_hook(post)])
        require(self.sites,'NO_REAL_MLA_IMPLEMENTATION')
        return self.sites

    def call(self,name,impl,method,original,args):
        require(name not in self.pending,'OVERLAPPING_MLA_LEASE')
        ids=list(self.runner.input_batch.req_ids)
        require(len(ids)==1 and ids[0],'ONE_ACTUAL_REQUEST_REQUIRED')
        result=original(*args)
        qn,qp,kn,kp=args[:4]
        meta=args[6] if method=='_forward_prefill' else args[5]
        require(meta.causal and meta.num_actual_tokens>0,'CAUSAL_REAL_TOKENS_REQUIRED')
        event=dict(site=name,method=method,request_id=ids[0],observed=True,eligible=True,
                   applied=False,changed=0,consumed=False,restored=False,
                   input_shapes=[list(t.shape) for t in (qn,qp,kn,kp)])
        self.events.append(event)
        if not self.fault:return result
        if method=='_forward_prefill':
            p=meta.prefill
            require(meta.num_prefills==1 and meta.num_decodes==0 and p.chunked_context is None,
                    'SINGLE_FRESH_PREFILL_REQUIRED')
            require(p.actual_seq_lengths_q==[qn.shape[0]],'PREFILL_BOUNDARIES_MISMATCH')
            q=torch.cat((qn[-1],qp[-1]),-1);k=torch.cat((kn,kp),-1);v=args[4]
            row=result.shape[0]-1
            value=recompute_attention_row(q,k,v,scale=impl.scale,block_size=self.block_size,replay_blocks=(0,)).reshape(-1)
            event['representation']='expanded_prefill_qkv'
        else:
            require(meta.num_decodes==1 and meta.num_prefills==0 and qn.shape[0]==1,
                    'SINGLE_DECODE_REQUIRED')
            require(args[6] is None if len(args)>6 else True,'QUANTIZED_MLA_NOT_SUPPORTED')
            block=args[4]; length=int(meta.decode.seq_lens_list[0])
            table=meta.decode.block_table[0,: (length+block-1)//block].long()
            def unpack(x,dim):
                if impl.enable_kv_nz:
                    x=x.view(-1,impl.num_kv_heads,dim//16,block,16).permute(0,3,1,2,4).reshape(-1,block,impl.num_kv_heads,dim)
                else:x=x.view(-1,impl.num_kv_heads,block,dim).transpose(1,2)
                return x.index_select(0,table).reshape(-1,impl.num_kv_heads,dim)[:length]
            latent=unpack(kn,impl.kv_lora_rank);rope=unpack(kp,impl.qk_rope_head_dim)
            q=torch.cat((qn[0],qp[0]),-1);k=torch.cat((latent,rope),-1)
            latent_output=recompute_attention_row(q,k,latent,scale=impl.scale,block_size=self.block_size,replay_blocks=(0,))
            value=impl._v_up_proj(latent_output[:,None,:]).reshape(-1);row=0
            event.update(representation='absorbed_decode_latent',key_count=length,kv_nz=impl.enable_kv_nz)
        require(value.shape==result[row].shape and value.dtype==result.dtype,'OUTPUT_LAYOUT_MISMATCH')
        require(torch.isfinite(value).all().item(),'RECOMPUTED_OUTPUT_NOT_FINITE')
        replacement=result.clone();backup=result[row].clone();token=None
        if self.region_leases is None:replacement[row].copy_(value)
        else:
            token=self.region_leases.acquire(replacement[row]);self.region_leases.write(token,value)
        require(torch.equal(replacement[:row],result[:row]) and torch.equal(replacement[row+1:],result[row+1:]),
                'NON_TARGET_ROWS_CHANGED')
        changed_elements=int((replacement[row]!=backup).sum().item())
        event.update(applied=True,changed=int(changed_elements>0),changed_elements=changed_elements,row=row,
                     non_target_exact=True,stream=str(torch.npu.current_stream()),
                     before_sha256=fingerprint(backup),after_sha256=fingerprint(value))
        self.pending[name]=dict(replacement=replacement,backup=backup,row=row,event=event,stream=torch.npu.current_stream(),token=token)
        return replacement

    def close(self):
        require(not self.pending,'DEVICE_CONSUMER_LEASE_STILL_PENDING')
        require(self.region_leases is None or self.region_leases.count()==0,'ATTENTION_REGION_LEASES_STILL_PENDING')
        for impl,name,wrapper in self.methods:
            require(vars(impl).get(name) is wrapper,'WRAPPER_OWNERSHIP_CHANGED')
            delattr(impl,name)
        for h in self.hooks:h.remove()
        self.methods.clear();self.hooks.clear()
        return dict(leases=0,hooks=0,inflight=0)
