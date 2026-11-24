"""0.26 source-bound exponential-noise site at the real sampling consumer.

The original sampler generates q and performs its cross-stream wait. Only then
does this site alter q at probs.div_(q), on the consumer stream. It neither
reimplements sampling nor replaces token IDs. Persistent weights/KV are excluded.
"""
import hashlib
import inspect
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from .serving_context import require


def fingerprint(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


class _NoiseConsumer(TorchDispatchMode):
    def __init__(self,site,probs):self.site,self.probs=site,probs

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {}
        if func is not torch.ops.aten.div_.Tensor or args[0] is not self.probs:
            return func(*args,**kwargs)
        site=self.site;q=args[1]
        require(site.active and q.shape==self.probs.shape and q.dtype==torch.float32
                and q.is_contiguous() and q.ndim==2 and q.shape[0]==1,
                'EXTERNAL_SAMPLER_REAL_NOISE_MAPPING_REQUIRED')
        with _disable_current_modes():
            event=dict(site='sampler.random_sample.exponential_noise',request_ids=list(site.runner.input_batch.req_ids),
                eligible=True,applied=False,changed=0,consumed=False,restored=False,failed=False,
                consumer_operator='aten.div_.Tensor',restore_kind='TRANSIENT_NOISE_AFTER_DEVICE_CONSUMER')
            site.events.append(event);calls=site.forward_calls[-1];calls[event['site']]=calls.get(event['site'],0)+1
            token=site.region_leases.acquire(q);site.pending[token]=event
            event['before_sha256']=fingerprint(q)
            try:
                value=q.clone();value[:,1::2]*=2.0
                if site.fault:
                    event['changed']=int(torch.count_nonzero(value!=q).item())
                    site.region_leases.write(token,value);event['applied']=True
                event['consumer_input_sha256']=fingerprint(q)
                output=func(*args,**kwargs)
                site.region_leases.consumed(token);event['consumed']=True
                if site.fault:site.region_leases.restore(token)
                else:site.region_leases.release_unmodified(token)
                event['restored_sha256']=fingerprint(q)
                require(event['restored_sha256']==event['before_sha256'],'EXTERNAL_SAMPLER_BYTES_NOT_RESTORED')
                event['restored']=True;del site.pending[token]
                return output
            except BaseException:
                event['failed']=True
                raise


class ExponentialNoiseSite:
    # Exact installed source reviewed in the A2 0.26 image. Other builds require
    # review and a distinct adapter; never infer compatibility from a version.
    SOURCE_SHA256 = '59c614b5a2264c3762bb32529e0da5a4c40fab045ad2680dfb7e7be114c0e38d'

    def __init__(self,model,runner,*,fault,region_leases):
        import vllm_ascend.sample.sampler as module
        self.model,self.runner,self.fault,self.region_leases=model,runner,fault,region_leases
        require(type(runner.sampler).__module__==module.__name__
                and type(runner.sampler).__name__=='AscendSampler', 'EXTERNAL_ASCEND_SAMPLER_REQUIRED')
        require(hashlib.sha256(Path(inspect.getsourcefile(module.random_sample)).read_bytes()).hexdigest()==self.SOURCE_SHA256,
                'EXTERNAL_SAMPLER_SOURCE_UNREVIEWED')
        self.module=module;self.original=module.random_sample
        self.events=[];self.sites=['sampler.random_sample.exponential_noise'];self.forward_calls=[]
        self.pending={};self.uncovered=[];self.active=False;self.wrapper=None

    def install(self):
        require(self.module.random_sample is self.original and self.wrapper is None,'EXTERNAL_SAMPLER_ALREADY_OWNED')
        def wrapped(probs,generators):
            require(self.active and len(self.runner.input_batch.req_ids)==1,'EXTERNAL_SAMPLER_OUTSIDE_CURRENT_REQUEST')
            before=len(self.events)
            with _NoiseConsumer(self,probs):result=self.original(probs,generators)
            require(len(self.events)==before+1,'EXTERNAL_SAMPLER_CONSUMER_NOT_OBSERVED')
            return result
        self.wrapper=wrapped;self.module.random_sample=wrapped

    def begin_sample(self):
        require(not self.active and not self.pending,'EXTERNAL_SAMPLER_PENDING')
        self.forward_calls.append({});self.active=True

    def finish_sample(self):
        require(self.active and self.forward_calls[-1] and not self.pending
                and self.region_leases.count()==0,'EXTERNAL_SAMPLER_COMPLETION_UNPROVEN')
        self.active=False

    def finish_forward(self):pass  # Sampling occurs later than model logits.

    def close(self):
        require(not self.pending and self.region_leases.count()==0,'EXTERNAL_SAMPLER_RESTORE_PENDING')
        if self.wrapper is not None:
            require(self.module.random_sample is self.wrapper,'EXTERNAL_SAMPLER_OWNER_CHANGED')
            self.module.random_sample=self.original;self.wrapper=None
        self.active=False
