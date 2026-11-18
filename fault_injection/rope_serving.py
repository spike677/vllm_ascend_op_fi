"""Standard-Worker fused RoPE requests, independent coverage and live recovery."""
import sys
import uuid

from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor import ServingTensorGroup
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import _capture,_identity,_same
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import _sha
from .rope_census import NaturalRoPETrace
from .rope_fused22 import FusedWrongRowSession
from .rope_group import RoPEGroupProvider
from .tensor_observer import NaturalTensorObserver


class ServingRoPEGroup(ServingTensorGroup):
    def __init__(self,worker,*,_trace_factory=None,_defer_to_forward=False,**kwargs):
        require(_trace_factory is None or kwargs.get('_simulation') is True,'ROPE_TRACE_OVERRIDE_CPU_ONLY')
        self.trace_factory=_trace_factory or NaturalRoPETrace.from_runtime
        self.template=self.trace_factory(worker.model_runner.model)
        self.trace_source_sha=_sha(self.template.sources);self.trace=None
        self.defer_to_forward=_defer_to_forward
        self.deferred_request=None;self.completion_handle=None;self.request_error=None
        self.request_entry_error=None
        super().__init__(worker,_mechanism='22.wrong_row',**kwargs)
        previous=self.provider
        self.provider=RoPEGroupProvider(self.model,template=self.template,replay=previous.replay,
            synchronize=previous.synchronize,workload_sha256=previous.scope.workload_sha256,
            environment_sha256=previous.scope.environment_sha256,simulation=previous.simulation,
            model_weights_sha256=previous.scope.model_weights_sha256,
            model_source_sha256=previous.scope.model_source_sha256,worker_identity=self.worker_identity)
        self.provider.owned_host_hooks=lambda:len(self.handles)+int(self.root_post_handle is not None)
        self.provenance.update(rope_sources=self.template.sources,
            structural_rope_sites=list(self.template.owners),raw_device_launch_payload_observed=False,
            variant='LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW')

    def _context(self,*,verify_targets=True):
        super()._context(verify_targets=False)
        if self.completion_handle is not None:
            hooks=self.completion_handle.hooks_dict_ref()
            require(hooks is not None and hooks.get(self.completion_handle.id) is
                    self._callbacks.get(self.completion_handle.id),'ROPE_COMPLETION_HOOK_OWNERSHIP_CHANGED')
        if not verify_targets:return
        identity=self._read_worker_identity()
        require(identity['status']=='WORKER_IDENTIFIED' and not identity.get('quarantined')
            and identity.get('worker')==self.worker_identity,'SERVING_TENSOR_WORKER_IDENTITY_CHANGED')
        require(not self.model.training,'INFERENCE_MODEL_REQUIRED')
        for path,(module,_) in self.provider.target_sources.items():
            self.template.sites.verify_ancestry(path);self.provider.target_guards[path].verify()
            require(self.template.owners[path] is module and module.cos_sin_cache is self.provider.cache_refs[path],
                    'ROPE_TARGET_OR_CACHE_LIFETIME_CHANGED')

    def expect(self,request_id,phase,arm=None):
        if self.defer_to_forward:
            self._context()
            require(self.deferred_request is None and self.expected is None and self.current is None
                    and self.trace is None,'PREVIOUS_ROPE_REQUEST_NOT_COLLECTED')
            require(isinstance(request_id,str) and request_id and request_id not in self.seen,
                    'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
            require(arm in (None,'baseline','fault'),'EXPLICIT_ROPE_ARM_REQUIRED')
            if not self.handles:self._install_context()
            self.deferred_request=(request_id,phase,arm)
            return dict(expected=request_id,phase=phase,arm=arm,deferred_until_actual_forward=True)
        return self._begin_request(request_id,phase,arm)

    def _begin_request(self,request_id,phase,arm):
        self._context()
        require(self.expected is None and self.current is None and self.adapter is None and self.trace is None,
                'PREVIOUS_ROPE_REQUEST_NOT_COLLECTED')
        require(isinstance(request_id,str) and request_id and request_id not in self.seen,
                'FRESH_REAL_ENQUEUED_REQUEST_REQUIRED')
        require(arm in (None,'baseline','fault'),'EXPLICIT_ROPE_ARM_REQUIRED')
        if not self.handles:self._install_context()
        self.expected=request_id;self.phase=phase;self.provider.calls=[]
        self.provider.current_invocation='worker-rope-forward:'+uuid.uuid4().hex
        self.provider.current_request_ids=(request_id,);self.provider._running=True
        if phase.endswith('off'):self.provider.enabled=False
        self.trace=self.trace_factory(self.model)
        require(_sha(self.trace.sources)==self.trace_source_sha,'ACTUAL_ROPE_RUNTIME_SOURCE_CHANGED')
        self.trace.position_policy='exact_or_last_from_first'
        self.trace.diagnostics_context=self._diagnostics
        # Lower mode observes final arguments; upper independent denominator
        # sees every original call before the injector asks for admission.
        self.trace.__enter__()
        if arm is not None:
            self.adapter=FusedWrongRowSession(self.model,arm=arm,simulation=self.provider.simulation,
                _runtime=self.template,references=self.provider.rope_references,
                admission=self.provider._admission,diagnostics_context=self._diagnostics)
            self.adapter.__enter__();self.provider.active_adapter=self.adapter
        self.observer=NaturalTensorObserver(self.model,on_operator_enter=self.provider.observe_before,
            diagnostics_context=self._diagnostics).__enter__()
        callback=self._post
        self.root_post_handle=self.model.register_forward_hook(callback,with_kwargs=True,always_call=True)
        self._callbacks[self.root_post_handle.id]=callback
        return dict(expected=request_id,phase=phase,arm=arm)

    def _pre(self,module,args,kwargs):
        if not self.defer_to_forward:
            return self._bind_request(module,args,kwargs)
        try:
            require(self.deferred_request is not None,'UNSCHEDULED_OR_CONCURRENT_ROPE_FORWARD')
            self.request_entry_error=sys.exc_info()[1]
            # torch snapshots pre hooks before invoking this hook. Newly
            # installed owned hooks do not run for this root unless explicitly
            # entered once. Refuse foreign root pre hooks whose ordering or
            # argument rewriting this reviewed seam cannot establish.
            before=dict(module._forward_pre_hooks)
            own=[key for key,value in before.items() if value==self._pre]
            require(module is self.model and len(own)==1 and set(before)==set(own),
                    'ROPE_ROOT_PRE_HOOK_ORDER_UNREVIEWED')
            request=self.deferred_request;self.deferred_request=None
            self._begin_request(*request)
            for key,callback in tuple(module._forward_pre_hooks.items()):
                if key==own[0]:
                    self._bind_request(module,args,kwargs)
                elif key not in before:
                    result=(callback(module,args,kwargs) if key in module._forward_pre_hooks_with_kwargs
                            else callback(module,args))
                    require(result is None,'ROPE_OWNED_PRE_HOOK_CHANGED_ARGUMENTS')
            callback=self._collect_returned_request
            self.completion_handle=self.processor.register_forward_hook(callback,with_kwargs=True,always_call=True)
            self._callbacks[self.completion_handle.id]=callback
        except BaseException as error:
            self.request_error=type(error).__name__+': '+str(error)
            raise

    def _bind_request(self,module,args,kwargs):
        super()._pre(module,args,kwargs)
        actual=self.binding.bind(module,args,kwargs)
        scheduler=self.current['scheduler']
        positions=scheduler.get('positions')
        if self.provider.simulation and positions is None:
            positions=actual['positions'].tolist()
        require(isinstance(positions,list),'ACTUAL_SCHEDULER_POSITIONS_REQUIRED')
        self.trace.set_request(self.expected,actual['positions'],positions,scheduler)
        if self.adapter:self.adapter.bind_request(self.expected,actual['positions'],positions)

    def _post(self,module,args,kwargs,output):
        try:
            if self.defer_to_forward:
                require(sys.exc_info()[1] is self.request_entry_error
                        and next(reversed(module._forward_hooks))==self.root_post_handle.id,
                        'ROPE_ROOT_RETURN_NOT_CONFIRMED')
            return super()._post(module,args,kwargs,output)
        except BaseException as error:
            self.request_error=type(error).__name__+': '+str(error)
            raise

    def _collect_returned_request(self,module,args,kwargs,output):
        try:
            require(output is not None and sys.exc_info()[1] is self.request_entry_error
                    and next(reversed(module._forward_hooks))==self.completion_handle.id,
                    'ACTUAL_ROPE_LOGITS_RETURN_REQUIRED')
            # Earlier processor hooks may legitimately transform the return.
            # The complete final output, not the early observer's snapshot,
            # is the value the serving caller will actually receive.
            with self._diagnostics():self.current['logits']=_capture(output)
            self._remove_hook(self.completion_handle);self.completion_handle=None
            self.collect()
        except BaseException as error:
            self.request_error=type(error).__name__+': '+str(error)
            raise

    def collect(self):
        require(self.current is not None and self.current['logits_returned'],'ROPE_REQUEST_NOT_COMPLETED')
        self._context();self.provider.synchronize()
        self._remove_hook(self.root_post_handle);self.root_post_handle=None
        self.observer.close()
        if self.adapter:self.adapter.close()
        self.trace.end_request();self.trace.close()
        independent=self.observer.report();independent['rope_trace']=self.trace.report()
        independent['complete']=independent['complete'] and independent['rope_trace']['complete']
        require(independent['complete'],'WORKER_INDEPENDENT_ROPE_CENSUS_INCOMPLETE')
        input_=self.current['input'];output=dict(hidden=self.current['hidden'],logits=self.current['logits'])
        if self.provider.candidates is not None:
            require(_same(input_,self.provider.reference_input),'ACTUAL_ROPE_WORKER_INPUT_CHANGED')
        if self.adapter and self.adapter.arm=='baseline' and self.provider.rope_references is None:
            self.provider.rope_references=dict(self.adapter.healthy_calls)
        row=dict(phase=self.phase,request_id=self.expected,scheduler=self.current['scheduler'],
            invocation=self.provider.current_invocation,input=_identity(input_),output=_identity(output),
            independent=independent,mechanism=self.adapter.report() if self.adapter else None,
            calls=[{k:v for k,v in call.items() if k!='decision'} for call in self.provider.calls],
            cleanup=self.adapter.runtime_state() if self.adapter else dict(cleanup_confirmed=True,
                hook_owned=False,inflight=0,leases=0),logits_boundary_returned=True)
        self.provider.rows.append(row);self.phases.append((input_,output,row))
        if self.provider.group is not None:self.provider._finish_calls(row)
        if self.phase.endswith('off'):
            require(_same(input_,self.provider.reference_input) and _same(output,self.provider.reference_output),
                    'WORKER_FULL_HIDDEN_LOGITS_OFF_NOT_EXACT')
            self.provider.off_row=row
        self.seen.append(self.expected)
        self.expected=self.current=self.adapter=self.observer=self.trace=None
        self.provider.active_adapter=None;self.provider._running=False
        return row

    def abort(self,*,retain_recovery=False):
        from .coverage import Outcome
        self._context(verify_targets=False)
        failure=dict(phase=getattr(self,'phase',None),request_id=self.expected,error=self.request_error)
        if self.completion_handle is not None:
            self._remove_hook(self.completion_handle);self.completion_handle=None
        if self.root_post_handle is not None:
            self._remove_hook(self.root_post_handle);self.root_post_handle=None
        if self.observer is not None:
            self.observer.close();failure['independent']=self.observer.report();self.observer=None
        if self.adapter is not None:
            self.adapter.close();failure['mechanism']=self.adapter.report();self.adapter=None
        if self.trace is not None:
            if self.trace.request_id is not None:self.trace.end_request()
            self.trace.close();failure['rope_trace']=self.trace.report();self.trace=None
        self.provider.active_adapter=None;self.provider._running=False;self.provider.enabled=False
        if self.provider.group is not None:
            for call in self.provider.calls:
                if 'decision' in call:
                    try:self.provider.group.finish(call['decision'],Outcome.FAILED,original_calls=0)
                    except BaseException:pass
        self.expected=self.current=self.deferred_request=None
        result=(super().close() if retain_recovery else self.close())
        result.update(failed_phase=failure,failed_request_recovery_claimed=False,
            group_recovery=self.provider.group.report() if self.provider.group else None)
        return result

    def close(self):
        require(self.trace is None and self.completion_handle is None and self.deferred_request is None,
                'ROPE_TRACE_STILL_OWNED')
        result=super().close()
        self.provider.rope_references=None
        getattr(self.provider,'cache_refs',{}).clear()
        return result
