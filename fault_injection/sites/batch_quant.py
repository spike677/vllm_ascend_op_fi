"""Trusted eager W8A8 real DynamicQuant -> original QMM continuous site.

Only pinned separate linear quantization; not fused MoE/GroupedMatmul quant.
Original quantized buffers are restored after all real QMM consumers complete.
No public Runtime qualification is granted by installing this experiment site.
"""
import hashlib
import inspect
import threading
from pathlib import Path
import numpy as np
import torch
from fault_injection.engines.dynamic_quant import shared_dynamic_quant
from fault_injection.sites.mla28 import require, fingerprint

SOURCE_SHA='ac9bd6ec05356b639665ec05ae21b19e1d0f0dc6723f7568b2f0ab70821285de'


def cpu_reference(rows):
    x=rows.detach().float().cpu().numpy()
    scale=np.max(np.abs(x))/np.float32(127)
    q=np.clip(np.rint(x/(scale if scale else np.float32(1))),-127,127).astype(np.int8)
    return q,np.full(x.shape[0],scale,dtype=np.float32)


class BatchQuantSite:
    def __init__(self,model,runner,*,fault,arithmetic_device='cpu',region_leases=None,reference_mode='immediate',call_policy=None):
        require(arithmetic_device in ('cpu','input'),'UNKNOWN_QUANT_ARITHMETIC_BACKEND')
        self.arithmetic_device=arithmetic_device
        self.call_policy=call_policy
        require(reference_mode in ('immediate','forward') and (reference_mode=='immediate' or arithmetic_device=='input'),
                'UNREVIEWED_REFERENCE_MODE')
        self.reference_mode=reference_mode;self.reference_queue=None
        self.region_leases=region_leases
        self.model=model;self.runner=runner;self.fault=fault
        self.events=[];self.methods=[];self.sites=[];self.uncovered=[]
        self.local=threading.local();self.forward_index=-1;self.hooks=[];self.pending=0
        self.forward_calls=[]

    def install(self):
        import torch_npu
        self.api=torch_npu
        if self.reference_mode=='forward':
            from fault_injection.engines.deferred_quant_reference import DeferredQuantReference
            def complete(stream):
                event=torch.npu.Event();event.record(stream);event.synchronize()
            self.reference_queue=DeferredQuantReference(stream=torch.npu.current_stream,complete=complete)
        self.original_quant=torch_npu.npu_dynamic_quant;self.original_qmm=torch_npu.npu_quant_matmul
        require(not getattr(self.original_quant,'_dcat_site',False),'QUANT_ALREADY_WRAPPED')
        seen=set()
        for name,module in self.model.named_modules():
            scheme=getattr(getattr(module,'quant_method',None),'quant_method',None)
            if type(scheme).__name__!='AscendW8A8DynamicLinearMethod':continue
            require(id(scheme) not in seen,'SHARED_SCHEME_OBJECT_REQUIRES_ADAPTER');seen.add(id(scheme))
            require(hashlib.sha256(Path(inspect.getsourcefile(type(scheme))).read_bytes()).hexdigest()==SOURCE_SHA,'UNREVIEWED_QUANT_SOURCE')
            require('apply' not in vars(scheme),'SCHEME_ALREADY_WRAPPED')
            original=scheme.apply
            def wrapped(layer,x,*args,_name=name,_original=original,**kwargs):
                require(self.forward_calls,'QUANT_APPLY_OUTSIDE_REAL_FORWARD')
                self.forward_calls[-1].append(_name)
                return self.call(_name,_original,layer,x,args,kwargs)
            scheme.apply=wrapped;self.methods.append((scheme,wrapped));self.sites.append(name)
        require(self.sites,'NO_SEPARATE_W8A8_LINEAR_SITE')
        def quant(x,*args,**kwargs):
            result=self.original_quant(x,*args,**kwargs)
            ctx=getattr(self.local,'context',None)
            if ctx is None:
                self.uncovered.append(dict(operator='npu_dynamic_quant',reason='OUTSIDE_DISCOVERED_LINEAR_SCHEME'));return result
            require(ctx['x'] is x and ctx.get('buffers') is None,'QUANT_INPUT_OR_MULTIPLICITY_CHANGED')
            q,s=result;event=ctx['event']
            event.update(shape=list(x.shape),q_dtype=str(q.dtype),scale_dtype=str(s.dtype))
            require(x.ndim==2 and q.shape==x.shape and s.shape==(x.shape[0],) and q.dtype==torch.int8 and s.dtype==torch.float32,'UNSUPPORTED_QUANT_LAYOUT')
            require(q.is_contiguous() and s.is_contiguous(),'QUANT_LAYOUT_NOT_CONTIGUOUS')
            ctx['buffers']=(q,s)
            if not self.fault or not event['eligible']:return result
            require(len(event['request_ids'])>=2,'MULTIPLE_ACTUAL_REQUESTS_REQUIRED')
            require(self.actual_rows()==x.shape[0],'PADDED_OR_UNMAPPED_QUANT_ROWS')
            backups=(q.clone(),s.clone());ctx['backups']=backups;self.pending+=1
            if self.region_leases is not None:
                ctx['region_tokens']=[]
                for tensor in (q,s):ctx['region_tokens'].append(self.region_leases.acquire(tensor))
            event['before_sha256']=[fingerprint(t) for t in backups]
            newq,news=shared_dynamic_quant(x,arithmetic_device=self.arithmetic_device)
            if self.arithmetic_device=='cpu':
                refq,refs=cpu_reference(x)
                require(np.array_equal(newq.cpu().numpy(),refq) and np.array_equal(news.cpu().numpy(),refs),
                        'CPU_REFERENCE_QUANT_MISMATCH')
                event.update(reference_passed=True,reference_exact=True,reference_scope='FULL_CPU_REFERENCE',
                             arithmetic_backend='torch_cpu_fp32')
            else:
                if self.reference_queue is None:
                    from fault_injection.engines.quant_reference import sampled_reference
                    event.update(sampled_reference(x,newq,news))
                else:
                    self.reference_queue.enqueue(len(self.events)-1,x,newq,news)
                    event['reference_pending']=True
                event['arithmetic_backend']='torch_input_fp32'
            event.update(changed=int((q!=newq).sum().item())+int((s!=news).sum().item()),
                         shared_scale=float(news[0].item()),all_real_rows=True)
            if self.region_leases is None:q.copy_(newq);s.copy_(news)
            else:
                for token,tensor in zip(ctx['region_tokens'],(newq,news)):self.region_leases.write(token,tensor)
            event.update(applied=True,after_sha256=[fingerprint(t) for t in (q,s)])
            return result
        def qmm(x,*args,**kwargs):
            ctx=getattr(self.local,'context',None)
            if ctx is None:return self.original_qmm(x,*args,**kwargs)
            require(ctx.get('buffers') is not None,'QMM_WITHOUT_QUANT')
            q,s=ctx['buffers'];require(x is q and kwargs.get('pertoken_scale') is s,'QMM_DID_NOT_CONSUME_EXACT_Q_SCALE')
            require(torch.npu.current_stream()==ctx['stream'],'QMM_STREAM_CHANGED')
            out=self.original_qmm(x,*args,**kwargs)
            ctx['event']['qmm_calls']+=1
            ctx['event']['consumer_operator']='torch_npu.npu_quant_matmul'
            return out
        quant._dcat_site=True
        self.quant_wrapper=quant;self.qmm_wrapper=qmm
        torch_npu.npu_dynamic_quant=quant;torch_npu.npu_quant_matmul=qmm
        def pre(*unused):
            self.forward_index+=1
            self.forward_calls.append([])
        self.hooks.append(self.model.register_forward_pre_hook(pre))
        if self.reference_queue is not None:
            def post(*unused):self.flush_reference()
            self.hooks.append(self.model.register_forward_hook(post))
        return self.sites

    def flush_reference(self):
        for report in self.reference_queue.flush():
            event=self.events[report.pop('key')]
            require(event.get('reference_pending') is True,'REFERENCE_CALL_ASSOCIATION_CHANGED')
            event.update(report,reference_pending=False)

    def actual_rows(self):
        from vllm.forward_context import get_forward_context
        metadata=get_forward_context().attn_metadata
        candidates=list(metadata.values()) if isinstance(metadata,dict) else [metadata]
        counts={int(m.num_actual_tokens) for m in candidates if hasattr(m,'num_actual_tokens')}
        require(len(counts)==1,'ACTUAL_TOKEN_METADATA_UNPROVEN')
        return counts.pop()

    def call(self,name,original,layer,x,args,kwargs):
        require(getattr(self.local,'context',None) is None,'OVERLAPPING_QUANT_CALL')
        event=dict(site=name,method='_forward_prefill' if self.forward_index==0 else '_forward_decode',
                   request_ids=list(self.runner.input_batch.req_ids),applied=False,changed=0,consumed=False,
                   restored=False,failed=False,qmm_calls=0,eligible=True)
        if self.call_policy is not None:
            policy=self.call_policy()
            require(set(policy)=={'eligible','reason','method','batch_frame'},'QUANT_PRODUCT_POLICY_SCHEMA')
            event.update(policy)
            event['eligible']=self.fault and policy['eligible']
        self.events.append(event)
        ctx=dict(x=x,event=event,stream=torch.npu.current_stream());self.local.context=ctx
        try:
            output=original(layer,x,*args,**kwargs)
            require(ctx.get('buffers') is not None and event['qmm_calls'] in (1,2),'EXPECTED_REAL_CONSUMERS_MISSING')
            require(torch.npu.current_stream()==ctx['stream'],'RETURN_STREAM_CHANGED')
            completion=torch.npu.Event();completion.record(ctx['stream']);completion.synchronize()
            event['consumed']=event['applied']
            return output
        except BaseException:
            event['failed']=True
            raise
        finally:
            if 'backups' in ctx:
                # A same-stream event orders restoration after any enqueued QMM.
                done=torch.npu.Event();done.record(ctx['stream']);done.synchronize()
                if self.region_leases is None:
                    for t,b in zip(ctx['buffers'],ctx['backups']):t.copy_(b)
                else:
                    for token in ctx.get('region_tokens',[]):
                        self.region_leases.consumed(token);self.region_leases.restore(token)
                done=torch.npu.Event();done.record(ctx['stream']);done.synchronize()
                event['restored_sha256']=[fingerprint(t) for t in ctx['buffers']]
                require(event['restored_sha256']==event['before_sha256'],'QUANT_BYTES_NOT_RESTORED')
                event['restored']=True;self.pending-=1
            self.local.context=None

    def close(self):
        require(self.reference_queue is None or self.reference_queue.pending()==0,'QUANT_REFERENCE_SNAPSHOTS_PENDING')
        require(self.region_leases is None or self.region_leases.count()==0,'QUANT_OBJECT_LEASES_PENDING')
        require(self.pending==0 and getattr(self.local,'context',None) is None,'QUANT_LEASES_PENDING')
        require(self.api.npu_dynamic_quant is self.quant_wrapper and self.api.npu_quant_matmul is self.qmm_wrapper,'QUANT_WRAPPER_OWNERSHIP_CHANGED')
        self.api.npu_dynamic_quant=self.original_quant;self.api.npu_quant_matmul=self.original_qmm
        for scheme,wrapped in self.methods:
            require(vars(scheme).get('apply') is wrapped,'SCHEME_OWNER_CHANGED');delattr(scheme,'apply')
        for hook in self.hooks:hook.remove()
        self.methods.clear();self.hooks.clear()
        return dict(leases=0,hooks=0,inflight=0)
