"""One actual MoE unpermute in four ordinary scheduler requests.

Reuse the reviewed producer/MLP/combine source trace and paired I/P remap.
No CoverageSession, experimental replay or whole-model target-set admission.
"""
from types import SimpleNamespace
from fault_injection.route8 import RestoreOffsetSession
from fault_injection.moe_census import NaturalMoETrace
from fault_injection.moe_restore import MoERestoreTrace
from fault_injection.effect_summary import numeric_change
from .providers.tensor_modules import _capture, _identity
from .rms18_request import Rms18TaskRequest
from .providers.rms18_task import Rms18TaskProvider
from .serving_context import require


class _OneMoERestore(RestoreOffsetSession):
    def __init__(self, task):
        self.task = task
        self.trace = None
        self.actual = None
        super().__init__(task.model, arm='fault' if task.phase=='fault' else 'baseline',
            simulation=task.binding.simulation, strict_group_oracle=True, admission=self._admit)
        factory = task.binding.trace_factory or NaturalMoETrace.from_runtime
        self.template = factory(task.model)
        self.paths = tuple(path or '<root>' for path,module in task.model.named_modules()
                           if hasattr(module,'top_k') and callable(getattr(module,'forward_impl',None)))
        require(self.paths, 'MOE8_ACTUAL_STRUCTURAL_MOE_REQUIRED')
        self.source_identity = self.template.sources
        if task.trace_sources is not None:
            require(self.source_identity==task.trace_sources,'MOE8_CURRENT_SOURCE_CHANGED')

    def __enter__(self):
        super().__enter__()
        try:
            proxy = SimpleNamespace(simulation=self.simulation,
                current_request_ids=(self.task.session.current['request_id'],), _event=self._event)
            self.trace = MoERestoreTrace(self.model,self.template,proxy,self,lambda:self.stack,self.paths)
            self.trace.__enter__()
            def close_trace(module,args,output):
                self.trace.close()
            self.hooks.append((self.model.register_forward_hook(close_trace,always_call=True),close_trace))
            return self
        except BaseException:
            self.close()
            raise

    def _event(self,path,operator,metadata):
        require(path in self.paths and self.task.session.root_running,
                'MOE8_CURRENT_MODEL_REQUEST_REQUIRED')

    def dispatch(self,func,args,kwargs):
        name=getattr(getattr(func,'_schema',None),'name','')
        if name=='npu::npu_moe_token_unpermute':
            # The nested source-bound combine trace is mandatory, even when
            # the outer dispatcher can already see this operator.
            require(self.trace is not None and bool(self.trace.owned_modes),
                    'MOE8_SOURCE_BOUND_COMBINE_REQUIRED')
            if self.events or (self.task.target is not None and self.stack[-1][0]!=self.task.target['path']):
                return func(*args,**kwargs)
        return super().dispatch(func,args,kwargs)

    def observe_actual_inputs(self,event,values,original_values):
        self.actual = dict(input=_identity(_capture(original_values)),
            module_object_id=id(self.stack[-1][1]), operator=event['operator'])
        event['actual']=self.actual
        if self.task.target is not None:
            require(self.actual==self.task.target['actual'],'MOE8_ACTUAL_MATCHED_INPUT_CHANGED')

    def _admit(self,event):
        return self.task.admit(event,self.actual)

    def report(self):
        trace=self.trace.report() if self.trace else None
        result=super().report()
        result['actual_moe_trace']=trace
        result['sources']=self.source_identity
        if result['events']:
            event=result['events'][0]
            private=self.private_observations[(event['path'],event['ordinal'])]
            h,i,p=private['inputs']
            reference=(h.float()[i.long()].reshape(p.shape[0],p.shape[1],-1)*p.float().unsqueeze(-1)).sum(1).to(private['output'].dtype)
            event['effect']=dict(comparison='ACTUAL_OUTPUT_VS_EXACT_UNSHIFTED_GROUP_ORACLE',
                values=numeric_change(reference,private['output']),
                normal_input_matched=self.task.target is not None,
                complete_normal_oracle_exact=True)
        return result

    def close(self):
        if self.trace is not None:
            self.trace.close()
        return super().close()


class Restore8TaskRequest(Rms18TaskRequest):
    def make_window(self):
        return _OneMoERestore(self.provider)


class Restore8TaskProvider(Rms18TaskProvider):
    mechanism_name='8.restore_offset'
    code_prefix='MOE8'
    target_selector='FIRST_ACTUAL_SOURCE_BOUND_MOE_UNPERMUTE'
    task_scope='ONE_SOURCE_BOUND_MOE_UNPERMUTE_MATCHED_FRESH_PREFILL'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.trace_sources=None

    def make_request(self):
        return Restore8TaskRequest(self)

    def end_actual(self,row):
        if self.phase in ('normal','fault'):
            report=row['mechanism']; trace=report.get('actual_moe_trace') or {}
            require(trace.get('complete') is True,'MOE8_COMPLETE_ACTUAL_PRODUCER_CONSUMER_REQUIRED')
            events=report['events']
            require(len(events)==1 and events[0].get('actual_producer_join')
                and events[0].get('observed_original_native_calls')==1
                and events[0].get('complete_native_output_oracle_exact'),
                'MOE8_ONE_EXACT_GROUP_REMAP_AND_NATIVE_CONSUMER_REQUIRED')
            if self.trace_sources is None: self.trace_sources=report['sources']
        return super().end_actual(row)

    def query(self,*,owner,generation):
        result=super().query(owner=owner,generation=generation)
        fault=next((row for row in reversed(self.rows) if row['phase']=='fault'),None)
        effect=None
        if fault is not None:
            event=fault['mechanism']['events'][0]
            effect=dict(schema='demoncat.moe8.actual_group_effect.v1',request_id=fault['request_id'],
                phase='fault',historical=True,position=dict(path=event['path'],ordinal=event['ordinal']),
                paired_groups_remapped=event['source_groups'],
                original_native_calls=event['observed_original_native_calls'],
                numerical=event['effect'],actual_producer_join=event['actual_producer_join'])
        current = self.rows[-1] if self.rows and not result['serving']['pending_request'] else None
        result['current_request_effect'] = dict(
            schema='demoncat.moe8.current_group_effect.v1',
            request_id=current['request_id'] if current else None,
            phase=current['phase'] if current else None,
            status='MATCHED_EXACT_GROUP_ORACLE_OBSERVED' if current and current['phase']=='fault'
                else 'NO_FAULT_EFFECT_IN_CURRENT_REQUEST',
            effect=effect if current and current['phase']=='fault' else None)
        result['evidence']['current_request_effect']=result['current_request_effect']
        result['last_fault_effect']=effect
        result['evidence']['last_fault_effect']=effect
        return result
