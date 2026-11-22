"""Real patch Conv3d argument mutation; no image, weight or output rewrite.

This site does not grant deployment admission. A source-bound MM request
adapter must establish request ownership before entering the encoder.
"""
import torch
from torch.utils._python_dispatch import _disable_current_modes
from .projection39 import ProjectionConsumerMode,projection_region
from .mla28 import fingerprint,require
from ..conv3d_natural import conv_arguments


class VisionConsumerMode(ProjectionConsumerMode):
    def __init__(self,site):
        super().__init__(None);self.site=site

    def complete_consumers(self,matched,operator):
        if matched:
            stream=self.site.stream()
            require(all(item['stream']==stream for item in matched),'VISION17_CONSUMER_STREAM_CHANGED')
            self.site.complete(stream)
        super().complete_consumers(matched,operator)
        for item in matched:
            event=item['event']
            with _disable_current_modes():
                require(self.site.inputs_identity(item['inputs'])==event['input_identity'],
                        'VISION17_INPUT_CHANGED_BY_CONSUMER')
            event.update(restored=True,restored_sha256=event['before_sha256'],
                restore_kind='CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE',same_stream_verified=True)

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        values=conv_arguments(func,args,kwargs or {})
        if values is None:return super().__torch_dispatch__(func,types,args,kwargs)
        site=self.site
        require(site.parents and len(site.parents)==1,'VISION17_PATCH_OWNER_UNPROVEN')
        name,module=site.parents[0]
        with _disable_current_modes():
            x,w,b=values['input'],values['weight'],values['bias']
            require(w is module.weight and b is module.bias and not module.training,
                    'VISION17_LIVE_PARAMETER_OWNER_CHANGED')
            require(tuple(w.shape[2:])==(2,16,16) and tuple(x.shape[2:])==(2,16,16)
                and tuple(values['stride'])==(2,16,16) and tuple(values['padding'])==(0,0,0)
                and tuple(values['dilation'])==(1,1,1) and values['groups']==1
                and not values.get('transposed',False) and not any(values.get('output_padding',()))
                and x.dtype==w.dtype and x.device==w.device,'VISION17_ORIGINAL_GEOMETRY_CHANGED')
            site.forward_calls[-1][name]=site.forward_calls[-1].get(name,0)+1
            key=(name,site.forward_calls[-1][name])
            inputs=(x,w,b);identity=site.inputs_identity(inputs)
            reference=site.references.get(key)
            require(not site.fault or reference is not None,'VISION17_NORMAL_REFERENCE_REQUIRED')
            if reference is not None:require(reference['inputs']==identity,'VISION17_ACTUAL_IMAGE_OR_WEIGHTS_CHANGED')
            before=fingerprint(torch.tensor(values['padding'],dtype=torch.int64))
            event=dict(site=name,request_ids=list(site.request_ids()),eligible=True,
                applied=False,changed=0,consumed=False,restored=False,failed=False,
                input_identity=identity,before_sha256=before,original_padding=[0,0,0],
                actual_padding=[0,7,7] if site.fault else [0,0,0],
                parameter_only=True,storage_mutated=False)
            site.events.append(event)
            call=dict(values,padding=event['actual_padding'])
        output=func(**call)
        with _disable_current_modes():
            require(tuple(output.shape)==(x.shape[0],w.shape[0],1,1,1),'VISION17_OUTPUT_GEOMETRY_CHANGED')
            require(site.inputs_identity(inputs)==identity,'VISION17_CONV_MUTATED_INPUTS')
            actual=fingerprint(output)
            if not site.fault and reference is None:site.references[key]=dict(inputs=identity,output=actual)
            event.update(output_sha256=actual,applied=site.fault,
                changed=int(site.fault and actual!=reference['output']),native_calls=1)
            self.watch(projection_region(output),event,inject=False)
            self.pending[-1]['inputs']=inputs
            self.pending[-1]['stream']=site.stream()
        return output


class Vision17Site:
    def __init__(self,visual,*,fault,references,request_ids,stream,complete):
        self.visual=visual;self.fault=fault;self.references=references;self.request_ids=request_ids
        self.stream=stream;self.complete=complete;self.hooks=[];self.parents=[];self.active=False
        self.events=[];self.forward_calls=[];self.sites=[];self.uncovered=[];self.excluded=[]
        self.mode=VisionConsumerMode(self)

    @staticmethod
    def inputs_identity(values):
        return [None if x is None else dict(shape=list(x.shape),dtype=str(x.dtype),sha256=fingerprint(x)) for x in values]

    @property
    def pending(self):return len(self.mode.pending)

    def install(self):
        def enter(*unused):
            require(not self.active and not self.pending,'VISION17_ENCODER_SCOPE_OVERLAP')
            self.forward_calls.append({});self.mode.__enter__();self.active=True
        def leave(*unused):
            require(not self.pending and not self.parents,'VISION17_OUTPUT_NOT_CONSUMED')
            self.mode.__exit__(None,None,None);self.active=False
        self.hooks.extend([self.visual.register_forward_pre_hook(enter),self.visual.register_forward_hook(leave)])
        for name,module in self.visual.named_modules():
            w=getattr(module,'weight',None)
            if not isinstance(w,torch.Tensor) or w.ndim!=5:continue
            require(tuple(w.shape[2:])==(2,16,16),'VISION17_UNSUPPORTED_PATCH_KERNEL')
            self.sites.append(name)
            def before(mod,args,_name=name):self.parents.append((_name,mod))
            def after(mod,args,out,_name=name):
                require(self.parents==[(_name,mod)],'VISION17_PATCH_CALL_ORDER')
                self.parents.pop()
            self.hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        require(self.sites,'VISION17_NO_REAL_PATCH_CANDIDATES')
        return self.sites

    def close(self):
        require(not self.active and not self.pending and not self.parents,'VISION17_ACTIVE_CONSUMER_REMAINS')
        for h in self.hooks:h.remove()
        self.hooks.clear()
