"""Pinned Qwen full-attention counterpart; linear/GDN layers are not targets."""
import hashlib,inspect
from pathlib import Path
import torch
from .mla28 import MLA28Site,require,fingerprint
from fault_injection.engines.attention import recompute_attention_row

SOURCE_SHA='4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076'


class GQA28Site(MLA28Site):
    def install(self):
        self.gates={}
        for name,module in self.model.named_modules():
            impl=getattr(module,'impl',None)
            if impl is None or type(impl).__name__!='AscendAttentionBackendImpl':continue
            require(hashlib.sha256(Path(inspect.getsourcefile(type(impl))).read_bytes()).hexdigest()==SOURCE_SHA,'UNREVIEWED_GQA_SOURCE')
            parent=self.model.get_submodule(name.rsplit('.',1)[0])
            require(hasattr(parent,'o_proj') and hasattr(parent,'qkv_proj'),'REAL_ATTENTION_PARENT_REQUIRED')
            require(impl.sliding_window is None and impl.sinks is None,'FULL_ATTENTION_ONLY')
            self.sites.append(name)
            def projected(mod,args,out,_name=name,_parent=parent):
                x=out[0]
                if _parent.attn_output_gate:
                    qgate=x[...,:_parent.q_size*2].view(x.shape[0],_parent.num_heads,-1)
                    self.gates[_name]=qgate.chunk(2,-1)[1].reshape(x.shape[0],-1)
                else:self.gates[_name]=None
            self.hooks.append(parent.qkv_proj.register_forward_hook(projected))
            original=impl.forward_impl
            require('forward_impl' not in vars(impl),'GQA_ALREADY_WRAPPED')
            def wrapped(*args,_name=name,_impl=impl,_original=original,**kwargs):
                require(not kwargs,'GQA_BOUNDARY_CHANGED')
                return self.call_gqa(_name,_impl,_original,args)
            impl.forward_impl=wrapped;self.methods.append((impl,'forward_impl',wrapped))
            def pre(mod,args,_name=name):
                lease=self.pending.get(_name)
                if lease is None:return
                expected=lease['replacement'][lease['row']].reshape(-1)
                gate=self.gates[_name]
                if gate is not None:expected=expected*torch.sigmoid(gate[lease['row']])
                require(torch.equal(args[0][lease['row']],expected),'REAL_GATED_OPROJ_INPUT_MISMATCH')
                require(torch.npu.current_stream()==lease['stream'],'CONSUMER_STREAM_CHANGED')
                if self.region_leases is not None:
                    actual=lease['actual_output'][lease['row']]
                    if torch.equal(actual,lease['backup']):copied=lease['backup']
                    else:
                        require(torch.equal(actual,lease['replacement'][lease['row']]),'GQA_ACTUAL_BUFFER_UNEXPECTED_WRITE')
                        copied=lease['replacement'][lease['row']]
                    self.region_leases.observe_output_copy(lease['actual_token'],copied)
                lease['event']['consumed']=True
            def post(mod,args,out,_name=name):
                lease=self.pending.get(_name)
                self.gates.pop(_name,None)
                if lease is None:return
                require(lease['event']['consumed'],'GQA_CONSUMER_NOT_OBSERVED')
                completion=torch.npu.Event();completion.record(lease['stream']);completion.synchronize()
                if self.region_leases is None:
                    for t in (lease['replacement'],lease['actual_output']):t[lease['row']].copy_(lease['backup'])
                else:
                    for token in (lease['replacement_token'],lease['actual_token']):
                        self.region_leases.consumed(token);self.region_leases.restore(token)
                completion=torch.npu.Event();completion.record(lease['stream']);completion.synchronize()
                require(all(torch.equal(t[lease['row']],lease['backup']) for t in (lease['replacement'],lease['actual_output'])),'GQA_BYTES_NOT_RESTORED')
                lease['event']['restored_sha256']=fingerprint(lease['replacement'][lease['row']])
                lease['event']['actual_output_restored_sha256']=fingerprint(lease['actual_output'][lease['row']])
                require(lease['event']['restored_sha256']==lease['event']['actual_output_restored_sha256']==lease['event']['before_sha256'],'GQA_BYTE_HASH_MISMATCH')
                lease['event']['restored']=True
                del self.pending[_name]
            self.hooks.extend([parent.o_proj.register_forward_pre_hook(pre),parent.o_proj.register_forward_hook(post)])
        require(self.sites,'NO_REAL_FULL_ATTENTION_SITE')
        return self.sites

    def call_gqa(self,name,impl,original,args):
        require(name not in self.pending,'GQA_LEASE_OVERLAP')
        query,key,value,cache,meta,out=args
        ids=list(self.runner.input_batch.req_ids)
        require(len(ids)==1 and meta.causal and len(meta.seq_lens_list)==1,'SINGLE_CAUSAL_REQUEST_REQUIRED')
        result=original(*args)
        count=meta.actual_seq_lengths_q[-1]; length=meta.seq_lens_list[0]
        require(count in (1,length),'SINGLE_CHUNK_OR_DECODE_REQUIRED')
        event=dict(site=name,method='_forward_prefill' if count>1 else '_forward_decode',request_id=ids[0],
                   observed=True,eligible=self.fault,applied=False,changed=0,consumed=False,restored=False)
        self.events.append(event)
        if not self.fault:return result
        q=query.view(-1,impl.num_heads,impl.head_size)[count-1]
        if count==length:
            k=key.view(-1,impl.num_kv_heads,impl.head_size)[:length]
            v=value.view(-1,impl.num_kv_heads,impl.head_size)[:length]
        else:
            kcache,vcache=impl.key_cache,impl.value_cache
            require(kcache.ndim==4 and kcache.dtype==query.dtype,'UNSUPPORTED_KV_LAYOUT')
            block=kcache.shape[1];table=meta.block_tables[0,:(length+block-1)//block].long()
            k=kcache.index_select(0,table).reshape(-1,impl.num_kv_heads,impl.head_size)[:length]
            v=vcache.index_select(0,table).reshape(-1,impl.num_kv_heads,impl.head_size)[:length]
        replacement=result.clone();row=count-1;backup=result[row].clone()
        calculated=recompute_attention_row(q,k,v,scale=impl.scale,block_size=self.block_size,replay_blocks=(0,)).reshape_as(backup)
        tokens={}
        if self.region_leases is None:replacement[row].copy_(calculated)
        else:
            tokens['replacement_token']=self.region_leases.acquire(replacement[row])
            tokens['actual_token']=self.region_leases.acquire(result[row])
            self.region_leases.write(tokens['replacement_token'],calculated)
        require(torch.isfinite(replacement).all().item(),'GQA_NONFINITE_RECOMPUTE')
        require(torch.equal(result[:row],replacement[:row]) and torch.equal(result[row+1:],replacement[row+1:]),'GQA_NON_TARGET_CHANGED')
        event.update(applied=True,changed=int((replacement[row]!=backup).sum().item()),non_target_exact=True,
                     key_count=length,before_sha256=fingerprint(backup),after_sha256=fingerprint(replacement[row]))
        self.pending[name]=dict(replacement=replacement,actual_output=result,backup=backup,row=row,event=event,stream=torch.npu.current_stream(),**tokens)
        return replacement
