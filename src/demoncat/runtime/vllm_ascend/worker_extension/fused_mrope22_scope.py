"""Private one-call adapter for source-reviewed Qwen3.5-family full Attention.
No public registry entry or deployment permission. Sites come from structural
inspection. The caller must bind the actual request and live owner separately.
"""
import inspect,json,os,sys,threading
from .serving_context import require
from .v026_text_observer import PROFILES,_verify_function
from fault_injection.fused_mrope22_boundary import FusedMrope22Boundary


def reviewed_attention_structure(module):
    # The reviewed Ascend patch assigns Qwen3NextAttention.forward; it does
    # not replace existing instances' class. The actual forward is separately
    # SHA-verified during install, so this structural filter grants no authority.
    return (type(module).__name__,type(module).__module__) in (
        ('Qwen3NextAttention','vllm.model_executor.models.qwen3_next'),
        ('AscendQwen3NextAttention','vllm_ascend.patch.worker.patch_qwen3_5'))


def _native():
    import torch
    from vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope import triton_split_qkv_rmsnorm_mrope
    return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope.default,triton_split_qkv_rmsnorm_mrope


class FusedMrope22CallScope:
    def __init__(self,module,positions,expected_positions,*,arm,admit,source_profile='v026',simulation=False):
        require(source_profile in PROFILES,'FUSED22_SOURCE_PROFILE_UNKNOWN')
        self.module=module;self.positions=positions;self.expected=expected_positions
        self.arm=arm;self.admit=admit;self.profile=source_profile;self.simulation=simulation
        self.identity=(os.getpid(),threading.get_ident());self.patches=[];self.hooks=[]
        self.inside=False;self.used=False;self.projection=None;self.result=None;self.sources=[]
    def guard(self):
        require(self.identity==(os.getpid(),threading.get_ident()),'FUSED22_SCOPE_THREAD_CHANGED')
    def install(self):
        import torch
        from torch.utils._python_dispatch import TorchDispatchMode
        self.guard();require(not self.patches and not self.used,'FUSED22_SCOPE_ALREADY_INSTALLED')
        m=self.module
        require(reviewed_attention_structure(m),'FUSED22_REVIEWED_MODULE_REQUIRED')
        op,implementation=_native();self.op=op;self.implementation=implementation
        rows=json.loads(PROFILES[self.profile].read_bytes())['files']
        self.sources=[_verify_function(m.forward,rows),_verify_function(implementation,rows)]
        require(m.qkv_proj is not m.attn and callable(m.qkv_proj.register_forward_hook),'FUSED22_ACTUAL_CHILDREN_REQUIRED')
        self.boundary=FusedMrope22Boundary(m,self.positions,self.expected,arm=self.arm,admit=self.admit,simulation=self.simulation)
        outer=self
        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                kwargs=kwargs or {};outer.guard()
                if func is torch.ops.aten.index.Tensor and args and args[0] is m.rotary_emb.cos_sin_cache:
                    require(outer.inside and len(args)==2 and not kwargs and len(args[1])==1
                        and args[1][0] is outer.positions,'FUSED22_REAL_INDEX_SIGNATURE')
                    return outer.boundary.index(func,args[0],args[1][0])
                if func is op:
                    require(outer.inside and outer.projection is not None,'FUSED22_NATIVE_OUTSIDE_PROJECTION')
                    parameters={v.name:x for v,x in zip(func._schema.arguments,args)}
                    require(not set(parameters).intersection(kwargs),'FUSED22_DUPLICATE_NATIVE_ARGUMENT')
                    parameters.update(kwargs)
                    for name in ('q_bias','k_bias'):
                        require(parameters.pop(name,None) is None,'FUSED22_BIAS_VARIANT_UNREVIEWED')
                    def real_call(**values):
                        # Observe the actual registered Python implementation,
                        # not just the name of the torch dispatcher operator.
                        require(sys.getprofile() is None,'FUSED22_EXISTING_PROFILER')
                        events=[]
                        code=inspect.unwrap(implementation).__code__
                        def profile(frame,event,value):
                            if frame.f_code is code:
                                if event=='call':
                                    require(frame.f_locals.get('qkv') is outer.projection,'FUSED22_NATIVE_PROJECTION_CHANGED')
                                    events.append(('call',None))
                                elif event=='return':events.append(('return',value))
                        try:
                            sys.setprofile(profile);output=func(**values)
                        finally:sys.setprofile(None)
                        require(len(events)==2 and events[0][0]=='call' and events[1][0]=='return'
                            and all(a is b for a,b in zip(output,events[1][1])),'FUSED22_REGISTERED_IMPLEMENTATION_NOT_OBSERVED')
                        return output
                    return outer.boundary.fused(real_call,outer.projection,**parameters)
                return func(*args,**kwargs)
        def patch(obj,name,handler):
            require(name not in vars(obj),'FUSED22_EXISTING_INSTANCE_OVERRIDE')
            original=getattr(obj,name)
            def wrapper(*args,**kwargs):return handler(original,*args,**kwargs)
            setattr(obj,name,wrapper);self.patches.append((obj,name,wrapper))
        def projection(module,args,output):
            self.guard();require(self.inside and module is m.qkv_proj and self.projection is None
                and isinstance(output,(tuple,list)) and len(output)==2,'FUSED22_PROJECTION_CALL_ORDER')
            self.projection=output[0]
        def consumer(original,q,k,v,*args,**kwargs):
            require(self.inside and not args and not kwargs,'FUSED22_ATTENTION_CONSUMER_SIGNATURE')
            return self.boundary.consume(original,q,k,v)
        def forward(original,*args,**kwargs):
            self.guard();require(not self.used and not self.inside,'FUSED22_SCOPE_SINGLE_ACTUAL_CALL')
            call=inspect.signature(original).bind(*args,**kwargs)
            require(call.arguments.get('positions') is self.positions,'FUSED22_REQUEST_POSITIONS_CHANGED')
            self.used=True;self.inside=True
            try:
                with Mode():output=original(*args,**kwargs)
                self.result=self.boundary.finish()
                self.result.update(source_bound_observation_verified=True,registered_implementation_observed=True,
                    actual_projection_observed=True,source_profile=self.profile,sources=self.sources,
                    qualification_granted=False,ordinary_runtime_integrated=False)
                return output
            finally:self.inside=False
        try:
            self.hooks.append(m.qkv_proj.register_forward_hook(projection))
            patch(m.attn,'forward',consumer);patch(m,'forward',forward)
        except BaseException:self.close();raise
        return self
    def close(self):
        self.guard();require(not self.inside,'FUSED22_SCOPE_STILL_EXECUTING')
        require(all(vars(obj).get(name) is f for obj,name,f in self.patches),'FUSED22_FOREIGN_WRAPPER_OWNER')
        for handle in reversed(self.hooks):handle.remove()
        self.hooks.clear()
        for obj,name,_ in reversed(self.patches):delattr(obj,name)
        self.patches.clear();self.projection=None
    def snapshot(self):
        return dict(result=self.result,owned_hooks=len(self.hooks),owned_wrappers=len(self.patches),
            inflight=int(self.inside),qualification_granted=False)
