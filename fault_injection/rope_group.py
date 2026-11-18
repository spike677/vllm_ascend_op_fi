"""Actual fused-RoPE coverage provider; admission uses a separate pre-observer."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import uuid

from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider,_sha
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import _same,_identity
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require
from .tensor_observer import metadata


class RoPEGroupProvider(NaturalTensorGroupProvider):
    def __init__(self,model,*,template,**kwargs):
        super().__init__(model,'22.wrong_row',**kwargs)
        from . import rope_sites,rope_census,rope_triton_trace,rope_fused22,rope_serving
        self.mechanism='22.wrong_row.fused_positions';self.template=template
        self.source_sha=_sha(dict(common=self.source_sha,files={Path(module.__file__).name:
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in
            (rope_sites,rope_census,rope_triton_trace,rope_fused22,rope_serving)},
            provider=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        self.scope=replace(self.scope,mechanism=self.mechanism,implementation_sha256=self.source_sha)
        self.rope_references=None;self.healthy_reader_rows={};self.healthy_reader_digests={};self.cache_refs={}

    def observe_before(self,func,args,kwargs,stack):
        schema=getattr(func,'_schema',None)
        if getattr(schema,'name',None)!=self.template.outer_operator:return
        require(bool(stack),'ACTUAL_ROPE_CALL_OUTSIDE_ROOT')
        module=stack[-1][1];path=self.template.sites.resolve(module,stack[:-1])
        require(self.template.owners.get(path) is module,'UNCLASSIFIED_ACTUAL_ROPE_OWNER')
        values={arg.name:arg.default_value for arg in schema.arguments}
        values.update(zip((arg.name for arg in schema.arguments),args));values.update(kwargs)
        require(values.get('cos_sin_cache') is module.cos_sin_cache,'ACTUAL_ROPE_CACHE_OWNER_CHANGED')
        if path in self.cache_refs:
            require(module.cos_sin_cache is self.cache_refs[path], 'QUALIFIED_ROPE_CACHE_LIFETIME_CHANGED')
        guard=self.template._guard_by_owner.get(id(module))
        if guard is not None:guard.verify()
        self._event(path,str(func),[metadata(values[name]) for name in ('positions','query','key','cos_sin_cache')])
        self.calls[-1]['schema']=str(schema)

    def qualify_observations(self,phases):
        from .coverage import BoundaryTargetIdentity,OperatorApiBoundary,Candidate,CandidateSet,Eligibility,EligibilityResult
        from .tensor_callable import TensorCallableGuard
        require(self.candidates is None and len(phases)==4,'FOUR_FRESH_ROPE_HEALTHY_REQUESTS_REQUIRED')
        self.reference_input,self.reference_output,_=phases[0]
        require(all(_same(x,self.reference_input) and _same(y,self.reference_output) for x,y,_ in phases),
                'ROPE_FULL_NATIVE_BASELINE_REPEAT_OR_LOSSLESS_NOT_EXACT')
        expected=set(self.template.owners)
        signatures=[]
        for _,_,phase in phases:
            trace=phase['independent']['rope_trace']
            require(phase['independent']['complete'] and trace['complete'],'ROPE_INDEPENDENT_DISCOVERY_INCOMPLETE')
            calls=phase['calls'];readers=trace['readers'];outer=trace['outer_operators']
            require(len(calls)==len(readers)==len(outer)==len(expected)
                    and {r['model_path'] for r in readers}==expected,'ROPE_ACTUAL_STRUCTURAL_DENOMINATOR_INCOMPLETE')
            require(all(tuple(call['key'])==(reader['model_path'],i) and reader['independent_position_transform']=='identity'
                        and reader['candidate_rows_different'] for i,(call,reader) in enumerate(zip(calls,readers))),
                    'ROPE_DISCOVERY_CALLSITE_OR_ROW_SEMANTICS_CHANGED')
            signatures.append([(reader['model_path'],reader['module_object_id'],reader['cache_lifetime_object_id'],
                reader['cache']['storage_id'],reader['cache_sha256'],reader['arguments'],reader['query_before'],
                reader['key_before'],reader['complete_output']) for reader in readers])
        require(all(signature==signatures[0] for signature in signatures),'ROPE_HEALTHY_CALLSITE_LIFETIME_OR_BYTES_UNSTABLE')
        first=phases[1][2];trace=first['independent']['rope_trace'];rows=[]
        for i,(call,reader,outer) in enumerate(zip(first['calls'],trace['readers'],trace['outer_operators'])):
            path=reader['model_path'];module=self.template.owners[path]
            guard=TensorCallableGuard(module);self.target_guards[path]=guard
            self.target_sources[path]=(module,guard.source_sha256);self.cache_refs[path]=module.cos_sin_cache
            device=reader['query_before'].get('device',reader['positions']['device']).split(':')
            signature=dict(path=path,query=reader['query_before']['shape'],key=reader['key_before']['shape'],
                dtype=reader['query_before']['dtype'],arguments=reader['arguments'])
            target=BoundaryTargetIdentity(process_id=self.context['process_id'],
                process_start_identity=self.context['process_start_identity'],worker_id=self.context['worker_id'],
                scope_kind=self.context['scope_kind'],rank=self.context['rank'],device_type=device[0],
                logical_device=None if device[0]=='cpu' else int(device[1]),
                model_instance_id=self.context['model_instance_id'],model_object_identity=id(self.model),
                module_object_identity=id(module),module_roles=('FUSED_ROPE_POSITIONS_READER',),
                operator_api=outer['operator'],tensor_signature=json.dumps(signature,sort_keys=True),
                boundary=OperatorApiBoundary(_sha(outer['schema']),self.source_sha,path+':'+str(i),'torch_dispatch'),
                provider_identity=self.mechanism,provider_proof=tuple(sorted({
                    'cache_object':str(reader['cache_lifetime_object_id']),
                    'cache_storage':str(reader['cache']['storage_id']),'actual_site':path,
                    'runtime_sources_sha256':_sha(self.template.sources)}.items())))
            self.selected[(path,i)]=target;self.healthy_reader_rows[(path,i)]=copy.deepcopy(reader)
            self.healthy_reader_digests[(path,i)]=_sha(reader)
            rows.append(Candidate(path+':'+str(i),EligibilityResult(Eligibility.SUPPORTED),target))
        require(self.rope_references is not None,'HEALTHY_ROPE_REFERENCE_POOL_REQUIRED')
        self.candidates=CandidateSet(tuple(rows),True)
        self.discovery_sha=_sha([phase for _,_,phase in phases]);self.baseline_sha=_sha(_identity(self.reference_output))
        self.input_sha=_sha(_identity(self.reference_input));self.lossless_sha=_sha(phases[3][2])
        return dict(candidate_set=self.candidates.report(),independent_discovery_sha256=self.discovery_sha,
            native_baseline_exact=True,lossless_exact=True,actual_shared_object_sites=len(rows),
            raw_device_launch_payload_observed=False,public_qualification_granted=False)

    def _finish_calls(self,row):
        from .coverage import Outcome
        trace=row['independent']['rope_trace'];require(trace['complete'],'ROPE_FINAL_RUNTIME_TRACE_INCOMPLETE')
        readers=trace['readers'];events=(row['mechanism'] or {}).get('events',[])
        require(len(readers)==len(self.calls)==len(self.selected),'ROPE_PHASE_DENOMINATOR_CHANGED')
        by_key={(event['path'],event['ordinal']):event for event in events}
        for index,(call,reader) in enumerate(zip(self.calls,readers)):
            key=tuple(call['key']);decision=call['decision'];event=by_key.get(key)
            require(key==(reader['model_path'],index),'ROPE_FINAL_READER_CALLSITE_JOIN_FAILED')
            if decision.admitted:
                healthy=self.healthy_reader_rows[key]
                require(_sha(healthy)==self.healthy_reader_digests[key],'FROZEN_ROPE_HEALTHY_EVIDENCE_CHANGED')
                require(event is not None and event['applied'] and event['native_calls']==1
                    and reader['independent_position_transform']=='last_from_first'
                    and reader['non_target_qk_before']==healthy['non_target_qk_before']
                    and reader['non_target_output']==healthy['non_target_output']
                    and reader['complete_output']!=healthy['complete_output']
                    and reader['cache_sha256']==healthy['cache_sha256']
                    and reader['root_positions_unchanged'] and reader['complete_cache_unchanged']
                    and any(c['returned'] for c in reader['consumers']), 'INDEPENDENT_ROPE_EFFECT_OR_CONSUMER_UNPROVEN')
                self.group.finish(decision,Outcome.HIT,native_applied=True,
                    execution_sha256=_sha(dict(reader=reader,outer=trace['outer_operators'][index],
                        triton=[r for r in trace['triton_launches'] if r['reader_sequence']==reader['sequence']])),
                    effect_sha256=_sha(dict(actual=reader,healthy=healthy,model_output=row['output'])))
            else:
                require(reader['independent_position_transform']=='identity' and not (event or {}).get('applied'),
                        'NON_TARGET_ROPE_POSITION_MUTATION')
                self.group.finish(decision,Outcome.PASSTHROUGH,execution_sha256=_sha(reader))

    def close_group(self):
        require(self.group is not None,'ROPE_GROUP_NOT_OPEN')
        recovery=self.group.disable(timeout_seconds=300)
        evidence_sha=_sha(self.rows[self.group_first_row:])
        if self.group.observer.evidence is None:
            self.group.observer.seal(artifact_sha256=evidence_sha,complete=True)
        else:
            # A failed drain/health cleanup may be retried against the same
            # already observed OFF; never reopen or rewrite its denominator.
            require(self.group.observer.evidence['artifact_sha256']==evidence_sha,
                    'SEALED_ROPE_OBSERVATION_CHANGED')
        report=self.group.report()
        report.update(execution_boundary='operator_api',mechanism_semantics=
            'LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW',recovery=recovery,
            phase_rows=self.rows[self.group_first_row:],scheduler_requests_claimed=True,
            raw_device_launch_payload_observed=False,independent_statistics='SEPARATE_RAW_COMPARISON_REQUIRED')
        return report

    def open_group(self,trigger='once',*,forwards=1,c2_proof=None):
        from .coverage import CoverageCapability,QualifiedTargetSet,TargetLease,TargetLosslessProof
        from .coverage_session import CoverageSession,GroupRecoveryHooks,IndependentCallObserver,SessionBudget
        require(self.candidates is not None and type(forwards) is int and 1<=forwards<=8,
                'QUALIFIED_BOUNDED_ROPE_REQUESTS_REQUIRED')
        require(trigger!='once' or forwards==1,'ROPE_C2_REQUIRES_ONE_COMPLETE_REQUEST')
        count=len(self.selected);self.generation+=1
        leases=tuple(TargetLease(target,self.owner,self.generation,'rope-site-'+str(index),0,0)
            for index,target in enumerate(self.candidates.eligible))
        group=QualifiedTargetSet(self.candidates,leases,tuple(TargetLosslessProof(lease,
            'positions_input_identity',self.lossless_sha,(self.input_sha,),(self.input_sha,),
            self.baseline_sha,self.baseline_sha,1,True,True) for lease in leases),
            CoverageCapability(single_target=True,multi_target=True,every_match=True,max_concurrency=count,
                max_resources=0,max_host_bytes=0,group_recovery=True,recovery_scope='target_set',
                stop_policy='bounded_session',lossless_modes=('positions_input_identity',)),
            concurrency=count,selection='all_eligible',trigger=trigger)
        observer=IndependentCallObserver(source='independent_workload_observer',observer_id=uuid.uuid4().hex,
            implementation_sha256=self.source_sha,workload_sha256=self.scope.workload_sha256,
            max_calls=count*(forwards+2))
        workers=tuple(set(target.worker for target in self.selected.values()))
        observer.seal_discovery(self.candidates,observed_targets=self.candidates.eligible,
            artifact_sha256=self.discovery_sha,expected_workers=workers,observed_workers=workers)
        def stop(lease):self.enabled=False
        def drain(lease,seconds):
            self.synchronize();return not self._running and self.active_adapter is None
        session=CoverageSession(group=group,scope=self.scope,observer=observer,
            budget=SessionBudget(count*forwards,600,max_concurrency=count),
            recovery=GroupRecoveryHooks(stop,drain,self._recover,self._health),c2_proof=c2_proof)
        self.enabled=True;self.off_row=None;self.group=session;self.group_first_row=len(self.rows)
        return session
