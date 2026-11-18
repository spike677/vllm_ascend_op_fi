"""Owned two-request multimodal census; no fault or model/runner replacement."""
import copy
import sys
import threading

from .vision17_context import Vision17RequestContext, MethodPin, _cpu
from .vision17_trace import Vision17Trace, require, vision17_diagnostics
from .conv3d_natural import tensor_identity
from .tensor_observer import tensors


class Vision17WorkerCensus:
    def __init__(self,worker,*,prompt,simulation=False,_processor=None):
        require(_processor is None or simulation,'PROCESSOR_OVERRIDE_CPU_ONLY')
        self.context=Vision17RequestContext(worker,simulation=simulation)
        self.worker,self.model,self.visual=worker,self.context.model,self.context.visual
        self.prompt=list(prompt);self.simulation=simulation;self.thread=threading.get_ident()
        if simulation:
            require(_processor is not None,'EXPLICIT_CPU_PROCESSOR_REQUIRED')
            self.processor=_processor;self.logits_guard=None
        else:
            from vllm.model_executor.layers.logits_processor import LogitsProcessor
            from demoncat.runtime.vllm_ascend.worker_extension.serving_context import discover_logits_boundary
            self.logits_root_pin=MethodPin(self.model,'compute_logits')
            boundary=discover_logits_boundary(self.model.language_model,actual_logits_processor_type=LogitsProcessor,
                actual_vocab_size=self.context.runner.vllm_config.model_config.hf_config.text_config.vocab_size)
            self.processor=boundary['processor'];self.logits_guard=boundary['branch_guard']
        self.handles=[];self.trace=None;self.pending=None;self.current=None;self.rows=[]
        self.seen=[];self.error=None;self.closed=False;self.callbacks=0;self.visual_active=False;self.root_active=False
        self.completion_handle=None;self.visual_entry_error=self.root_entry_error=None

    def _install(self,module,method,callback,**kwargs):
        handle=getattr(module,method)(callback,**kwargs)
        self.handles.append((handle,callback));return handle

    def start(self):
        require(not self.handles and not self.closed,'CENSUS_SINGLE_START_REQUIRED')
        try:
            self._install(self.visual,'register_forward_pre_hook',self._visual_pre,with_kwargs=True,prepend=True)
            self._install(self.model,'register_forward_pre_hook',self._root_pre,with_kwargs=True,prepend=True)
            self._install(self.model,'register_forward_hook',self._root_post,with_kwargs=True,always_call=True)
            self._install(self.processor,'register_forward_hook',self._logits_post,with_kwargs=True,always_call=True)
        except BaseException:
            self.close();raise

    def verify(self):
        require(threading.get_ident()==self.thread,'CENSUS_THREAD_CHANGED')
        self.context.verify_runtime()
        for handle,callback in self.handles:
            hooks=handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(handle.id) is callback,'CENSUS_HOOK_OWNER_CHANGED')

    def expect(self,request_id):
        self.verify()
        require(not self.closed and self.pending is None and self.current is None and not self.error,
                'CENSUS_PREVIOUS_REQUEST_PENDING_OR_FAILED')
        require(len(self.rows)<2 and isinstance(request_id,str) and request_id and request_id not in self.seen,
                'CENSUS_TWO_FRESH_REQUEST_BUDGET')
        self.pending=request_id

    def _failed(self,error):
        if self.error is None:self.error=type(error).__name__+': '+str(error)

    def _current(self):
        require(self.pending is not None,'CENSUS_UNSCHEDULED_REQUEST')
        if self.current is None:
            self.current=dict(request_id=self.pending,encoder_observed=False,root_returned=False,logits_returned=False,
                qualification_granted=False,fault_injections=0,aclnn_launch_arguments=False)
        require(self.current['request_id']==self.pending,'CENSUS_REQUEST_CHANGED')
        return self.current

    def _visual_pre(self,module,args,kwargs):
        self.callbacks+=1
        try:
            self.verify();require(not self.visual_active and self.trace is None,'CENSUS_CONCURRENT_ENCODER')
            row=self._current();require(not row['encoder_observed'],'CENSUS_MULTIPLE_ENCODER_CALLS_UNADAPTED')
            self.visual_entry_error=sys.exc_info()[1]
            # Context joins only original active code/objects and actual scheduler;
            # no hook is installed on execute_model or the runner.
            row['request_binding']=self.context.snapshot(module,args,kwargs,
                expected_request_id=self.pending,expected_prompt=self.prompt)
            row['encoder_observed']=True;self.visual_active=True
            before=set(module._forward_pre_hooks)
            self.trace=Vision17Trace(self.visual,simulation=self.simulation)
            self.trace.__enter__()
            # PyTorch has already snapshotted this call's pre-hooks.
            for key,callback in tuple(module._forward_pre_hooks.items()):
                if key not in before:
                    require(callback(module,args,kwargs) is None,'CENSUS_OBSERVER_CHANGED_ARGUMENTS')
            self.completion_handle=self._install(module,'register_forward_hook',self._visual_post,
                                                 with_kwargs=True,always_call=True)
        except BaseException as error:
            self._failed(error);self.visual_active=False;raise
        finally:self.callbacks-=1

    def _remove(self,handle):
        entry=next((entry for entry in self.handles if entry[0] is handle),None)
        require(entry is not None,'CENSUS_HOOK_NOT_OWNED')
        hooks=handle.hooks_dict_ref()
        require(hooks is not None and (handle.id not in hooks or hooks[handle.id] is entry[1]),'CENSUS_HOOK_OWNER_CHANGED')
        try:handle.remove()
        finally:
            if handle.id not in hooks:
                self.handles.remove(entry)
                if self.completion_handle is handle:self.completion_handle=None

    def _visual_post(self,module,args,kwargs,output):
        self.callbacks+=1
        try:
            require(self.visual_active and output is not None and sys.exc_info()[1] is self.visual_entry_error
                    and next(reversed(module._forward_hooks))==self.completion_handle.id,'CENSUS_ENCODER_RETURN_UNCONFIRMED')
            with vision17_diagnostics():self.current['encoder_output']=[tensor_identity(_cpu(t)) for t in tensors(output)]
            self.trace.close();self.current['conv_observation']=self.trace.report();self.trace=None
            self._remove(self.completion_handle)
        except BaseException as error:
            self._failed(error);raise
        finally:self.visual_active=False;self.callbacks-=1

    def _root_pre(self,module,args,kwargs):
        self.callbacks+=1
        try:
            self.verify();row=self._current();require(not self.root_active and not row['root_returned'],'CENSUS_REPEATED_ROOT')
            require(self.context.runner.input_batch.req_ids==[self.pending]
                    and set(self.context.runner.requests)=={self.pending},'CENSUS_ROOT_REQUEST_MISMATCH')
            request=self.context.runner.requests[self.pending]
            require(request.req_id==self.pending and request.prompt_token_ids==self.prompt and request.num_computed_tokens==0
                    and len(request.mm_features)==1 and request.mm_features[0].modality=='image',
                    'CENSUS_ROOT_MM_REQUEST_REQUIRED')
            if row['encoder_observed']:
                require(id(request)==row['request_binding']['request_object_id']
                        and id(request.mm_features[0])==row['request_binding']['feature_object_id'],
                        'CENSUS_ROOT_IMAGE_REQUEST_LIFETIME_CHANGED')
            if not row['encoder_observed']:
                row['unresolved_reason']='NO_ACTUAL_ENCODER_CALL_POSSIBLE_CACHE_REUSE_NOT_PROVEN'
            self.root_active=True;self.root_entry_error=sys.exc_info()[1]
        except BaseException as error:self._failed(error);raise
        finally:self.callbacks-=1

    def _root_post(self,module,args,kwargs,output):
        self.callbacks+=1
        try:
            require(self.root_active and output is not None and sys.exc_info()[1] is self.root_entry_error,
                    'CENSUS_ROOT_RETURN_UNCONFIRMED')
            with vision17_diagnostics():self.current['hidden']=[tensor_identity(_cpu(t)) for t in tensors(output)]
            self.current['root_returned']=True
        except BaseException as error:self._failed(error);raise
        finally:self.root_active=False;self.callbacks-=1

    def _logits_post(self,module,args,kwargs,output):
        if self.pending is None:return
        self.callbacks+=1
        try:
            row=self._current()
            require(row['root_returned'] and not row['logits_returned'] and output is not None
                    and sys.exc_info()[1] is self.root_entry_error,'CENSUS_LOGITS_RETURN_UNCONFIRMED')
            require(self.context.runner.input_batch.req_ids==[self.pending]
                    and self.pending in self.context.runner.requests,'CENSUS_LOGITS_REQUEST_CHANGED')
            if not self.simulation:
                self.logits_root_pin.verify()
                from .vision17_context import _frame
                frame=_frame(self.logits_root_pin);del frame
            if self.logits_guard:self.logits_guard()
            with vision17_diagnostics():row['logits']=[tensor_identity(_cpu(t)) for t in tensors(output)]
            row['logits_returned']=True
        except BaseException as error:self._failed(error);raise
        finally:self.callbacks-=1

    def collect(self):
        self.verify();require(self.current is not None and self.current['logits_returned'] and not self.error
            and not self.callbacks and not self.visual_active and not self.root_active and self.trace is None,
            'CENSUS_REAL_REQUEST_NOT_COMPLETED')
        self.rows.append(copy.deepcopy(self.current));self.seen.append(self.pending)
        self.current=self.pending=None;self.context.close()
        return self.rows[-1]

    def close(self):
        require(threading.get_ident()==self.thread and not self.callbacks and not self.visual_active and not self.root_active,
                'CENSUS_ACTUAL_CALLBACK_STILL_ACTIVE')
        if self.trace is not None:
            self.trace.close();self.trace=None
        for handle,_ in reversed(self.handles[:]):self._remove(handle)
        self.context.close();self.current=self.pending=None;self.closed=True
        return dict(owned_hooks=0,owned_modes=0,held_tensors=0,fault_injections=0,qualification_granted=False)

    def report(self):
        rows=copy.deepcopy(self.rows)
        exact=len(rows)==2 and rows[0].get('hidden')==rows[1].get('hidden') and rows[0].get('logits')==rows[1].get('logits')
        observed=len(rows)==2 and all(r['encoder_observed'] and r.get('conv_observation',{}).get('torch_dispatch_chain_complete') for r in rows)
        return dict(rows=rows,two_healthy_outputs_exact=exact,two_actual_encoder_chains_observed=observed,
                    incomplete_request=copy.deepcopy(self.current),pending_request=self.pending,
                    observed_calls=sum(len(r.get('conv_observation',{}).get('observed_calls',[])) for r in rows),
                    error=self.error,c1=False,c2=False,c3=False,e5=False,qualification_granted=False,
                    aclnn_launch_arguments=False,device_payload_observed=False,fault_injections=0)
