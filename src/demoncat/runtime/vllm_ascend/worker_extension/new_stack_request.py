"""Shared request adapter for reviewed new-stack ordinary task providers.

This module does not register a provider or grant deployment acceptance. It
reuses an existing provider's phase/owner/effect/recovery state machine; the
host must pause scheduling, enqueue, advance, then resume normal inference.
"""
from contextlib import nullcontext
import hashlib,json,uuid
from .serving_context import require
from .serving import complete_tensor
from .providers.tensor_modules import _capture,_identity
from .v026_text_observer import SourceBoundTextObserver,TextPrefillTrace


class _TaskTrace(TextPrefillTrace):
    def __init__(self,session):
        super().__init__(session.runner,{session.expected:list(session.prompt)})
        self.session=session
    def model_enter(self,module,args,kwargs):
        super().model_enter(module,args,kwargs)
        self.session.begin(module,args,kwargs)
    def model_return(self,module,args,kwargs,output):
        super().model_return(module,args,kwargs,output)
        self.session.root_return(output)
    def execute_return(self,result):
        super().execute_return(result)
        self.session.state_return(self.state)
    def after_sample(self,result):
        super().after_sample(result)
        self.session.complete_request()


class NewStackTaskRequest:
    """Private adapter, deliberately not reachable through public config yet."""
    def __init__(self,provider,window_factory):
        self.provider=provider;self.runner=provider.worker.model_runner;self.model=provider.model
        self.prompt=tuple(provider.binding.prompt);self.window_factory=window_factory
        self.expected=self.current=self.window=self.observer=None
        self.installed=False;self.root_running=self.logits_running=False
        self.request_error=None;self.output=None;self.last_trace=None;self.root_arguments=None
    def _same_worker(self):
        self.provider._context()
        require(self.provider.worker.model_runner is self.runner and self.runner.model is self.model,
                'NEWSTACK_TASK_WORKER_CHANGED')
    def install(self):
        self._same_worker();require(not self.installed,'NEWSTACK_TASK_ALREADY_INSTALLED')
        self.installed=True
    def expect(self,request_id):
        self._same_worker();self.readback()
        require(self.installed and self.current is None and self.expected is None and self.window is None
                and self.request_error is None and self.provider.phase!='complete','NEWSTACK_PREVIOUS_REQUEST_NOT_CLOSED')
        require(isinstance(request_id,str) and 0<len(request_id)<=256 and request_id not in self.provider.requests,
                'FRESH_ENQUEUED_ID_REQUIRED')
        self._close_observer();self.expected=request_id
        observer=SourceBoundTextObserver(self.runner,{request_id:list(self.prompt)},
            source_profile=self.provider.binding.execution_profile)
        observer.trace=_TaskTrace(self);self.observer=observer
        try:observer.install()
        except BaseException:
            observer.close();self.observer=None;self.expected=None;raise
    def _diagnostics(self):
        return self.window.diagnostics() if self.window else nullcontext()
    def begin(self,module,args,kwargs):
        self._same_worker()
        require(not self.root_running and self.current is None and self.expected is not None,'NEWSTACK_ACTUAL_REQUEST_ORDER')
        token_cpu,token_hash,finite=complete_tensor(kwargs['input_ids'])
        positions,position_hash,position_finite=complete_tensor(kwargs['positions'])
        require(finite and position_finite and token_cpu.tolist()==list(self.prompt)
                and positions.tolist()==[list(range(len(self.prompt)))]*3,'NEWSTACK_DEVICE_INPUT_BYTES_DIFFER')
        self.current=dict(request_id=self.expected,actual_internal_request_id=self.expected,
            phase=self.provider.phase,invocation='newstack-task:'+uuid.uuid4().hex,
            input=_identity(_capture((args,kwargs))),
            input_sha256=hashlib.sha256(json.dumps([token_hash,position_hash]).encode()).hexdigest(),
            snapshot=self.observer.trace.snapshot(),device_input_bytes_equality='COMPLETE_DEVICE_READBACK_VERIFIED',
            root_returned=False)
        self.root_arguments=kwargs
        self.provider.begin_actual(self.current)
        self.root_running=True
        if self.provider.phase in ('normal','fault'):
            self.window=self.window_factory(self.provider);self.window.__enter__()
            # Enter only the newly installed root callback omitted from torch's
            # current pre-hook snapshot. All subsequent calls remain native.
            for handle,callback in getattr(self.window,'entry_hooks',self.window.hooks):
                if handle.hooks_dict_ref() is module._forward_pre_hooks:
                    value=callback(module,args,kwargs) if handle.id in module._forward_pre_hooks_with_kwargs else callback(module,args)
                    require(value is None,'NEWSTACK_ROOT_ARGUMENTS_CHANGED')
    def root_return(self,output):
        require(self.root_running and output is not None and output.ndim==2
                and output.shape[0]==len(self.prompt),'NEWSTACK_COMPLETE_HIDDEN_REQUIRED')
        with self._diagnostics():
            _,full,finite=complete_tensor(output);_,last,_=complete_tensor(output[-1:].contiguous())
        require(finite,'NEWSTACK_HIDDEN_NOT_FINITE')
        self.output=output;self.current.update(hidden_sha256=full,last_hidden_sha256=last,root_returned=True)
        self.root_running=False;self.logits_running=True;self.root_arguments=None
    def state_return(self,state):
        require(self.current is not None and self.current['root_returned'] and state.hidden_states is self.output,
                'NEWSTACK_STATE_NOT_MODEL_OUTPUT')
        with self._diagnostics():
            _,sample,sfinite=complete_tensor(state.sample_hidden_states)
            _,logits,lfinite=complete_tensor(state.logits)
        require(sfinite and lfinite and sample==self.current['last_hidden_sha256'],'NEWSTACK_COMPLETE_LOGITS_NOT_LAST_QUERY')
        self.current['output']=dict(hidden=self.current['hidden_sha256'],logits=logits)
        self.current['mechanism']=self.window.report() if self.window else None
        if self.window:self.window.close();self.window=None
        self.output=None;self.logits_running=False
    def complete_request(self):
        require(self.current is not None and self.window is None and self.current.get('output') is not None,
                'NEWSTACK_EFFECT_BEFORE_ACTUAL_SAMPLE')
        self.current['request_trace']=self.observer.trace.snapshot()
        self.provider.end_actual(self.current);self.current=self.expected=None
    def _close_observer(self):
        if self.observer:
            self.observer.close();self.last_trace=self.observer.snapshot();self.observer=None
    def close_window(self):
        self._same_worker();self.readback()
        if self.request_error and (self.observer is None or self.observer.inside==0):
            self.root_running=self.logits_running=False
        require(not self.root_running and not self.logits_running,'NEWSTACK_ACTUAL_FORWARD_STILL_ACTIVE')
        require(self.current is None or self.request_error,'NEWSTACK_ACTUAL_REQUEST_NOT_FINISHED')
        if self.window:self.window.close();self.window=None
        self._close_observer();self.expected=self.current=self.output=None
    def close(self):
        self.close_window();self.installed=False
    def readback(self):
        self._same_worker()
        row=self.observer.snapshot() if self.observer else {}
        if row.get('first_error') and self.request_error is None:
            self.request_error=row['first_error'];self.provider.fail(RuntimeError(self.request_error))
        state=self.window.runtime_state() if self.window else dict(ownership_verified=True,leases=0)
        require(state['ownership_verified'],'NEWSTACK_OWNED_DISPATCH_CHANGED')
        return dict(installed=self.installed,owned_hooks=row.get('owned_hooks',0)+(getattr(self.window,'owned_hook_count',len(self.window.hooks)) if self.window else 0),
            owned_wrappers=row.get('owned_wrappers',0),inflight=row.get('inflight',0),
            leases=state['leases'],pending=self.expected is not None or self.current is not None,
            request_trace=row or self.last_trace,qualification_granted=False)
