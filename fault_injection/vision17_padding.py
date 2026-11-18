"""Fault17 patch-padding adapter for actual weighted custom Conv3d modules.

The mechanism and shape-preserving geometry checks are inherited unchanged.
Discovery uses registered weight objects and actual calls, not user layer names.
Runtime source/Worker admission is a separate required deployment guard.
"""
from .conv3d_natural import PatchConv3dSession
from .rope_sites import ModuleSites
from .tensor_callable import TensorCallableGuard
from .vision17_trace import vision17_diagnostics
from contextlib import contextmanager, nullcontext


class Vision17PaddingSession(PatchConv3dSession):
    def __init__(self,model,**kwargs):
        import torch
        external=kwargs.get('diagnostics_context')
        @contextmanager
        def diagnostics():
            with vision17_diagnostics(), external() if external is not None else nullcontext():
                yield
        kwargs['diagnostics_context']=diagnostics
        super().__init__(model,**kwargs)
        self.sites=ModuleSites(model)
        self.candidates={path:module for path,module in self.sites.nodes.items()
            if isinstance(getattr(module,'weight',None),torch.Tensor) and module.weight.ndim==5}
        self.actual_guards={id(module):TensorCallableGuard(module) for module in self.candidates.values()}

    def dispatch(self,func,args,kwargs):
        for path,module in self.stack:
            if id(module) in self.actual_guards:
                self.sites.verify_ancestry(path)
                self.actual_guards[id(module)].verify()
        return super().dispatch(func,args,kwargs)

    def report(self):
        result=super().report()
        result.update(final_arguments_proof='REQUIRES_INDEPENDENT_LOWER_VISION17_TRACE',
                      aclnn_launch_arguments=False,native_qualification=False)
        return result
