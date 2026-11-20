"""Owned read-only health observation after Native/attention hooks are removed.

This observer grants no mechanism qualification and never enqueues or executes
inference. Its scheduler snapshot binds the actual first decode to the request.
"""
import threading
from .serving_context import require
from .postwrite41 import _output_hash
from fault_injection.deployments.postwrite import _cpu


class FirstDecodeHealth:
    def __init__(self, provider, request_id, reference, complete):
        self.p, self.request_id, self.reference = provider, request_id, reference
        self.complete_callback = complete
        self.handles, self.owners = [], {}
        self.current = self.result = self.error = None
        self.active = False
        self.thread = threading.get_ident()
        require(not provider.installed and not provider.handles and not provider.patches
                and provider.lease is None, 'DECODE_HEALTH_MECHANISM_RESOURCES_REMAIN')

    def context(self):
        p=self.p
        require(threading.get_ident()==self.thread and p.worker.model_runner is p.runner
                and p.runner.model is p.model and p._identity()==p.model_identity,
                'DECODE_HEALTH_HELD_MODEL_CHANGED')
        require(not p.installed and not p.handles and not p.patches and p.lease is None,
                'DECODE_HEALTH_MECHANISM_REATTACHED')

    def install(self):
        self.context()
        require(not self.handles, 'DECODE_HEALTH_ALREADY_INSTALLED')
        for before, callback in ((True,self.pre),(False,self.post)):
            method=self.p.model.register_forward_pre_hook if before else self.p.model.register_forward_hook
            handle=method(callback,with_kwargs=True,**({} if before else {'always_call':True}))
            self.handles.append(handle); self.owners[handle.id]=callback

    def pre(self,module,args,kwargs):
        try:
            self.context(); self.readback()
            require(not self.active, 'DECODE_HEALTH_REENTRANT_ROOT')
            self.active=True
            if self.p.runner.with_prefill:return
            from vllm.forward_context import get_forward_context
            from .batch_context import snapshot_first_decode_rows
            require(self.current is None and self.result is None, 'DECODE_HEALTH_ONLY_FIRST_DECODE')
            expected={self.request_id:dict(logical_id=0,prompt=self.reference['prefixes'][0]['prompt'])}
            self.current=snapshot_first_decode_rows(self.p.runner,self.p.model,
                self.p.binding.bind(module,args,kwargs),get_forward_context(),expected,expected_max_seqs=1)
            require(self.current['prefixes']==self.reference['prefixes'], 'DECODE_HEALTH_LOGICAL_INPUT_CHANGED')
        except BaseException as error:
            self.error=repr(error);raise

    def post(self,module,args,kwargs,output):
        try:
            self.context();self.readback()
            require(self.active, 'DECODE_HEALTH_POST_WITHOUT_PRE')
            if self.error:raise RuntimeError(self.error)
            if self.current is None:return
            require(output is not None, 'DECODE_HEALTH_ROOT_DID_NOT_RETURN')
            digest=_output_hash(_cpu(output))
            require(digest==self.reference['model_output_sha256'], 'DECODE_HEALTH_OUTPUT_NOT_EXACT')
            self.result=dict(request_id=self.request_id,phase='health',model_output_sha256=digest,
                             exact_healthy_output=True,mechanism_hooks_present=False,
                             input_prefixes_exact=True,model_instance=id(module))
        except BaseException as error:
            self.error=repr(error);raise
        finally:
            if threading.get_ident()==self.thread:
                self.active=False;self.current=None
        self.close()
        try:self.complete_callback(self.result)
        except BaseException as error:
            self.error=repr(error);raise

    def readback(self):
        self.context()
        for handle in self.handles:
            hooks=handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is self.owners[handle.id],
                    'DECODE_HEALTH_HOOK_OWNER_CHANGED')
            require(next(reversed(hooks))==handle.id, 'DECODE_HEALTH_FINAL_ROOT_OBSERVER_REQUIRED')
        return dict(owned_hooks=len(self.handles),active=self.active,error=self.error)

    def close(self):
        self.context()
        require(not self.active and self.current is None, 'DECODE_HEALTH_ROOT_STILL_ACTIVE')
        for handle in reversed(self.handles[:]):
            hooks=handle.hooks_dict_ref()
            require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is self.owners[handle.id]),
                    'DECODE_HEALTH_HOOK_OWNER_CHANGED')
            handle.remove()
            require(handle.id not in hooks,'DECODE_HEALTH_HOOK_REMOVAL_UNCONFIRMED')
            self.handles.remove(handle);del self.owners[handle.id]
