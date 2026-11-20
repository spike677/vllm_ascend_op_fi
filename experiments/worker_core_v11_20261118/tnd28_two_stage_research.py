"""Explicit TND28 staged Native research driver; no public factory/config route.

This runs original natural calls. It never treats a zero-F census as a fault
phase, and never uses a saved hit/report to authorize a pending executor.
"""
from copy import deepcopy
import ctypes as C
import json
import os
import uuid
from pathlib import Path

from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
from demoncat.runtime.vllm_ascend.worker_extension.attention28_prepared_input import PreparedInputCapture,digest
from demoncat.runtime.vllm_ascend.worker_extension.attention28_value_plan import LivePreparedValuePlan
from demoncat.runtime.vllm_ascend.worker_extension.attention28_typed_plan import SYMBOLS,decode
from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus,capture_inputs,validate_tnd_prefill
from demoncat.runtime.vllm_ascend.worker_extension.attention_stock_launch import observe_scope,sha
from demoncat.runtime.vllm_ascend.worker_extension.providers.attention_serving import (
    FiaDenominatorReplayServingProvider,request_values)


class ResearchTypedPlan(LivePreparedValuePlan):
    def state(self):
        self.binding._owner();output=C.create_string_buffer(4096)
        require(self.functions[SYMBOLS[2]](self.token.encode(),output,len(output))==0,'RESEARCH_NATIVE_PLAN_QUERY_FAILED')
        value=decode(output.value)
        require(value.get('schema')=='demoncat.tnd28.live_typed_plan.v2'
            and type(value.get('fault_authorized')) is bool
            and value['fault_authorized']==bool(value.get('replay_parent_committed')),'RESEARCH_NATIVE_PLAN_AUTHORITY_STATE')
        if value['fault_authorized']:
            require(value['replay_requested']==value['host_completed']==value['copy_launch_read']==1
                and value['host_gate_checks']>0 and value['value_read_mask']==31 and value['first_error']==0,
                'RESEARCH_NATIVE_FINAL_ADMISSION_INCOMPLETE')
        return value


class ResearchInputCapture(PreparedInputCapture):
    def __init__(self,*args,phase,**kwargs):
        require(phase in ('baseline','baseline_repeat','fault','off'),'RESEARCH_EXPLICIT_PHASE')
        self.phase=phase;self.policy={}
        super().__init__(*args,check_live_values=True,check_live_copy_launch=True,**kwargs)
        require(self.borrow_query is not None,'RESEARCH_NATIVE_BORROW_REQUIRED')
        candidate=self.build.get('tnd28_two_stage_candidate',{})
        require(candidate.get('prepare_symbol')=='dcat_tnd28_replay_prepare_gate_v2'
            and candidate.get('copy_launch_symbol')=='dcat_tnd28_copy_launch_read_v1'
            and candidate.get('library_sha256')==self.borrow_query.sha256,'RESEARCH_TWO_STAGE_BUILD_REQUIRED')
    def _capture_phase(self):return self.phase
    def _make_typed_plan(self,token):
        return ResearchTypedPlan(self.borrow_query,self.replay_binding,self.live_plan,token,check_live_copy_launch=True)
    def _check_policy(self):
        actual={k:v for k,v in os.environ.items() if k.startswith('DCAT28_TND_')}
        require(actual==self.policy,'RESEARCH_FOREIGN_OR_CHANGED_POLICY')
    def enter(self,token,row,actual,api):
        self._check()
        if self.phase=='fault':
            from fault_injection.fia_input_journal import public_tensor_metadata
            scope=row['native_census_scope']
            public={name:public_tensor_metadata(actual[name]) for name in ('query','key','value','atten_mask')}
            values=dict(MODE='denominator_replay',PHASE='fault',TOKEN=token,OWNER=scope['owner'],
                GENERATION=str(scope['generation']),OBJECT=str(row['module_object_id']),TARGET=row['module_path'],
                PID=str(self.owner['pid']),TID=str(self.owner['native_tid']),PROCESS_START=self.owner['process_start_ticks'],
                REQUEST_SHA256=digest(row['request_id']),INPUT_SHA256=digest(public),LOG=str(self.root/(token+'.host.jsonl')))
            for k,v in values.items():
                key='DCAT28_TND_'+k
                try:os.environ[key]=v
                finally:
                    if os.environ.get(key)==v:self.policy[key]=v
        result=super().enter(token,row,actual,api)
        result.update(research_phase=self.phase,requested_replay=self.phase=='fault',public_entry=False)
        return result
    def returned(self,output):
        super().returned(output)
        current=self.receipt['native_typed_plan_return']
        require(current['fault_authorized']==(self.phase=='fault'),'RESEARCH_ACTUAL_ADMISSION_DIFFERS')
        self.receipt['fault_authorized']=current['fault_authorized']
        self.receipt['native_execution_qualification']=False
    def stop(self):
        super().stop();self._check_policy()
        for k,v in list(self.policy.items()):
            require(os.environ.get(k)==v,'RESEARCH_POLICY_OWNER_CHANGED')
            del os.environ[k];del self.policy[k]
        if self.receipt is not None:self.receipt['research_policy_released']=True


class TwoStageNativeResearch(StockTndPrefillCensus):
    observer_only=False
    def __init__(self,worker,settings,**kwargs):
        self.worker=worker;self.settings=dict(settings)
        super().__init__(**kwargs)
        self.fingerprint.update(observer_only=False,research_two_stage_candidate=True,private_native_build=settings['build_sha256'])
    def state(self):
        result=super().state();result.update(observer_only=False,public_entry=False,native_launch_qualified=False)
        capture=getattr(self,'prepared_input_capture',None)
        if capture is not None:result['environment_owned']=len(capture.policy)+len(capture.active or {})
        return result
    def execute(self,actual,row,phase,once):
        from demoncat.runtime.vllm_ascend.worker_extension import attention_execution
        session=attention_execution._ACTIVE
        require(session is not None and session.driver is self and session.worker is self.worker
            and session.current['phase']==phase and session.live_api_calls[-1][0] is row,
            'RESEARCH_CURRENT_ORIGINAL_FRAME_REQUIRED')
        require(phase in ('baseline','baseline_repeat','fault','off'),'RESEARCH_SELECTED_PHASE_REQUIRED')
        require(self.prepared_input_capture is None and not self.refs and not self.pending_drain,'RESEARCH_PREVIOUS_RESOURCES_REMAIN')
        _,original,args,_=session.live_api_calls[-1];context=session._context()
        module=session.stack[-1][1];owner,generation=next(iter(session.deployment.leases))
        require(len(session.deployment.leases)==1,'RESEARCH_ONE_ACTUAL_OWNER')
        row['native_census_scope']=dict(owner=owner,generation=generation,worker_object_id=id(self.worker),
            runner_object_id=id(session.runner),model_object_id=id(session.model),rank=self.worker.rank,
            device_type=self.worker.device.type,logical_device=self.worker.device.index,phase=phase)
        api=capture_inputs(original,args,actual,context.attn_metadata[module.layer_name]);validate_tnd_prefill(api,row)
        token=uuid.uuid4().hex;settings=self.settings
        capture=ResearchInputCapture(self.worker,phase=phase,build_completion=settings['build'],
            build_sha256=settings['build_sha256'],evidence_dir=self.folder/(token+'-inputs'),
            execute_borrow_library=settings['execute_borrow_library'],execute_borrow_sha256=settings['execute_borrow_sha256'])
        self.prepared_input_capture=capture
        self.refs=(actual,None);self.pending_drain=True
        record=dict(token=token,phase=phase,target_key=row['target_key'],research_only=True,public_entry=False,
            original_calls=0,selection_applied=False,fault_authorized=False,native_launch_qualified=False)
        try:
            self._synchronize();capture.prepare_plan(row)
            with self.scope.scope(token):
                capture.enter(token,row,actual,api)
                output=once();record['original_calls']=1;self.refs=(actual,output);capture.returned(output)
                self._synchronize();self.scope.observer.synchronize();self.pending_drain=False
            self.stop()
            launch=observe_scope(self.run,token,self.identity)
            keys=[str(item['tiling_key']) for item in launch['launches']]
            expected='119000000030400003' if phase=='fault' else '5000000000000200103'
            require(keys==['524000000',expected],'RESEARCH_ACTUAL_COPY_FIA_KEYS_DIFFER:'+repr(keys))
            require(capture.receipt['native_typed_plan_return']['fault_authorized']==(phase=='fault'),
                    'RESEARCH_RUNTIME_ADMISSION_DISAGREES_WITH_LAUNCH')
            require(capture_inputs(original,args,actual,context.attn_metadata[module.layer_name])==api,'RESEARCH_INPUT_CHANGED')
            record.update(selection_applied=phase=='fault',fault_authorized=phase=='fault',native=launch,
                input_capture=deepcopy(capture.receipt),status='RESEARCH_NATIVE_CALL_RECORDED_PENDING_INDEPENDENT_REVIEW')
            self.calls.append(record);self.prepared_input_capture=None
            path=self.folder/(token+'.json');path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            return output,record
        except BaseException as error:
            self._error('two_stage_research',error)
            raise


from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import AttentionExecutionAdapter


class HealthOnlyAdapter(AttentionExecutionAdapter):
    """Three read-only root/logits observers; no Attention or API interception."""
    def install(self):
        from demoncat.runtime.vllm_ascend.worker_extension import attention_execution
        self._owned()
        require(attention_execution._ACTIVE is None and not self.installed,'RESEARCH_HEALTH_OWNER_ALREADY_PRESENT')
        attention_execution._ACTIVE=self;self.installed=True
        self.root_pre_handle,_=self._hook(self.model,self._pre,before=True)
        self.root_handle,_=self._hook(self.model,self._post)
        self.logits_handle,_=self._hook(self.processor,self._logits)
        require(len(self.handles)==3 and not self.patches,'RESEARCH_HEALTH_NO_FAULT_INTERCEPTION')


class TwoStageResearchProvider(FiaDenominatorReplayServingProvider):
    """Direct research construction only; never added to public registry."""
    def query(self,*,owner,generation):
        result=super().query(owner=owner,generation=generation)
        result['evidence']['fault_admission']='RESEARCH_TWO_STAGE_CANDIDATE_PENDING_INDEPENDENT_REVIEW'
        result['research_request_plan']=dict(phases=list(self.phases),public_entry=False)
        return result
    def _prepare_phase(self,phase):
        require(type(self.binding.native) is TwoStageNativeResearch,'RESEARCH_EXACT_NATIVE_DRIVER')
        if phase=='health':
            require(self.records[-1]['phase']=='off' and not self.session.needs_off,'RESEARCH_PROVEN_OFF_BEFORE_HEALTH')
            self.session.close()
            require(not self.session.patches and not self.session.handles,'RESEARCH_FAULT_HOOKS_REMAIN')
            self.session=HealthOnlyAdapter(self.binding,self._context);self.session.install()
            return
        if phase!='fault':return super()._prepare_phase(phase)
        require(self.selection is not None and [r['phase'] for r in self.records]==
            ['observe','observe_repeat','baseline','baseline_repeat'],'RESEARCH_REPEAT_BASELINE_BEFORE_FAULT')
    def _collect(self):
        super()._collect();row=self.records[-1]
        if row['phase']=='fault':
            base=self.records[-2]
            def calls(r):return [c for f in r['forwards'] for c in f['calls']]
            before={c['target_key']:c for c in calls(base)}
            hits=[c for c in calls(row) if c['native'] is not None]
            require(len(hits)==1 and hits[0]['native']['selection_applied'] is True
                and hits[0]['native']['fault_authorized'] is True,'RESEARCH_ONE_COMMITTED_FAULT')
            match=hits[0];reference=before[match['target_key']]
            require(match['input']==reference['input'] and match['output']!=reference['output'],
                    'RESEARCH_SAME_FIA_INPUT_AND_CHANGED_OUTPUT_REQUIRED')
        if row['phase'] in ('off','health'):
            reference=next(r for r in self.records if r['phase']=='baseline')
            require(request_values(row)==request_values(reference),'RESEARCH_OFF_HEALTH_FULL_HASH_RECOVERY')
            if row['phase']=='off':
                hits=[c for f in row['forwards'] for c in f['calls'] if c['native'] is not None]
                require(len(hits)==1 and hits[0]['native']['fault_authorized'] is False
                    and hits[0]['native']['selection_applied'] is False,'RESEARCH_ACTUAL_OFF_REQUIRED')
                self.session.needs_off=False # Only after actual OFF + complete hidden/logits equality.
            else:
                require(type(self.session) is HealthOnlyAdapter and not self.session.patches
                    and len(self.session.handles)==3 and all(not f['calls'] for f in row['forwards']),
                    'RESEARCH_HEALTH_FAULT_INTERCEPTION_REMAINED')
