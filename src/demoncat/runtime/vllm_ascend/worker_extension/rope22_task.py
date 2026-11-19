"""One actual fused RoPE call in an ordinary four-request task."""
import sys
from fault_injection.rope_census import NaturalRoPETrace
from fault_injection.rope_fused22 import FusedWrongRowSession
from .providers.tensor_modules import _capture,_identity
from .providers.rms18_task import Rms18TaskProvider
from .rms18_request import Rms18TaskRequest
from .serving_context import require

class _OneRoPE(FusedWrongRowSession):
    def __init__(self,task):
        self.task=task
        factory=task.binding.trace_factory or NaturalRoPETrace.from_runtime
        trace=factory(task.model)
        trace.single_call=True
        trace.selected_site=task.target['path'] if task.target else None
        trace.position_policy='exact_or_last_from_first'
        self.trace=trace
        self.actual=None
        super().__init__(task.model,simulation=task.binding.simulation,_runtime=trace,
            arm='fault' if task.phase=='fault' else 'baseline',references=task.rope_references,
            admission=self._admit)
        trace.diagnostics_context=self.diagnostics

    @property
    def entry_hooks(self):return self.trace.hooks+self.hooks
    @property
    def owned_hook_count(self):return len(self.trace.hooks)+len(self.hooks)

    def __enter__(self):
        self.trace.__enter__()
        try:
            super().__enter__()
            current=self.task.session.current
            positions=self.task.session.root_arguments['positions']
            expected=list(range(len(self.task.binding.prompt)))
            self.trace.set_request(current['request_id'],positions,expected,current['snapshot'])
            self.bind_request(current['request_id'],positions,expected)
            return self
        except BaseException:
            self.close();raise

    def dispatch(self,func,args,kwargs):
        if getattr(getattr(func,'_schema',None),'name',None)!=self.runtime.outer_operator:
            return func(*args,**kwargs)
        path,module=self.stack[-1]
        if self.events or self.task.target is not None and path!=self.task.target['path']:
            return func(*args,**kwargs)
        values={arg.name:arg.default_value for arg in func._schema.arguments}
        values.update(zip((arg.name for arg in func._schema.arguments),args));values.update(kwargs)
        with self.diagnostics():
            self.actual=dict(module_object_id=id(module),cache_object_id=id(module.cos_sin_cache),
                inputs=_identity(_capture(tuple(values[k] for k in ('positions','query','key','cos_sin_cache')))))
        if self.task.target is not None:
            require(self.actual==self.task.target['actual'],'ROPE22_CURRENT_COMPLETE_INPUT_OR_CACHE_CHANGED')
        self.current_schema=str(func._schema)
        output=super().dispatch(func,args,kwargs)
        event=self.events[-1]
        event.update(actual=self.actual,schema=str(func._schema))
        return output

    def _admit(self,event):
        # Schema was observed at this dispatcher; the original code will verify
        # the precise source-bound reader and independent final launch as well.
        event['schema']=self.current_schema
        return self.task.admit(event,self.actual)

    def report(self):
        super().close()
        self.trace.end_request();self.trace.close()
        trace=self.trace.report()
        require(trace['complete'] and trace['observed_readers']==1
            and trace['observation_scope']=='ONE_ACTUAL_SITE','ROPE22_ONE_INDEPENDENT_COMPLETE_READER_REQUIRED')
        result=super().report()
        require(len(result['events'])==1,'ROPE22_ONE_ACTUAL_EVENT_REQUIRED')
        event=result['events'][0];reader=trace['readers'][0]
        require(reader['model_path']==event['path'] and any(c['returned'] for c in reader['consumers']),
            'ROPE22_ACTUAL_RETURN_CONSUMER_JOIN_REQUIRED')
        if self.task.phase=='normal':
            require(reader['independent_position_transform']=='identity','ROPE22_NORMAL_POSITIONS_CHANGED')
            self.task.rope_references=dict(self.healthy_calls)
            self.task.normal_reader=reader
        else:
            normal=self.task.normal_reader
            require(event['same_complete_inputs_as_healthy'] and event['changed_from_healthy']
                and reader['independent_position_transform']=='last_from_first'
                and reader['non_target_qk_before']==normal['non_target_qk_before']
                and reader['non_target_output']==normal['non_target_output']
                and reader['complete_output']!=normal['complete_output']
                and reader['cache_sha256']==normal['cache_sha256'],
                'ROPE22_INDEPENDENT_WRONG_ROW_EFFECT_REQUIRED')
        event.update(original_execution_verified=True,return_readback_exact=True,
            output_changed=bool(event['applied'] and event['changed_from_healthy']),
            consumers=reader['consumers'],independent_final_reader=reader)
        result['rope_trace']=trace
        return result

    def close(self):
        super().close()
        if self.trace.request_id is not None:self.trace.end_request()
        self.trace.close()

    def runtime_state(self):
        state=super().runtime_state()
        if self.trace.active:
            require(sys.getprofile() is self.trace.callback,'ROPE22_PROFILE_OWNER_CHANGED')
        state['leases']+=int(self.trace.outer_mode is not None)
        return state

class Rope22TaskRequest(Rms18TaskRequest):
    def _root_pre(self,module,args,kwargs):
        self.root_arguments=self.root_binding.bind(module,args,kwargs)
        try:return super()._root_pre(module,args,kwargs)
        finally:self.root_arguments=None
    def make_window(self):return _OneRoPE(self.provider)

class Rope22TaskProvider(Rms18TaskProvider):
    mechanism_name='22.wrong_row'
    code_prefix='ROPE22'
    target_selector='FIRST_ACTUAL_SOURCE_BOUND_FUSED_ROPE_READER'
    task_scope='ONE_FUSED_LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.rope_references=None
        self.normal_reader=None
    def make_request(self):return Rope22TaskRequest(self)
    def query(self,*,owner,generation):
        state=super().query(owner=owner,generation=generation)
        fault=next((r for r in self.rows if r['phase']=='fault'),None)
        effect=None
        if fault:
            event=fault['mechanism']['events'][0]
            effect=dict(request_id=fault['request_id'],position=dict(path=event['path'],ordinal=event['ordinal']),
                comparison='MATCHED_FULL_QK_INPUTS_AND_IMMUTABLE_CACHE',actual_final_reader=event['independent_final_reader'],
                complete_qk_changed=event['changed_from_healthy'],non_target_unchanged=event['non_target_output_unchanged'],
                correct_source_row=event['correct_source_row'],replacement_source_row=event['replacement_source_row'],
                native_calls=event['native_calls'],integer_overflow_reproduced=False)
        state['last_fault_effect']=effect;state['evidence']['last_fault_effect']=effect
        return state
