"""CoverageSession integration for actual Torch Conv3d padding requests.

Qualification is local to fresh live requests in this owner/generation. A prior
census/serialized report cannot authorize F. No ACLNN or device ABI is claimed.
"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path

from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider,_sha
from .vision17_trace import require


def call_signature(row):
    return dict(path=row['path'],sequence=row['sequence'],operator=row['operator'],schema=row['schema'],
        geometry=row['actual_geometry'],inputs=row['inputs_before'],
        input_metadata=[{k:v for k,v in t.items() if k in ('shape','stride','offset','dtype','device','actual_format')}
                        for t in row['actual_input_metadata']])


def output_identity(row):
    return {key:row[key] for key in ('encoder_output','hidden','logits')}


class Vision17Group(NaturalTensorGroupProvider):
    """Reuse the existing coverage open/drain/stop protocol, never replay()."""
    def __init__(self,worker,identity,binding,owner,generation):
        from .coverage_session import CoverageScope
        from .runtime_discovery import _python_execution_context
        self.model,self.visual=worker.model_runner.model,worker.model_runner.model.visual
        self.context=_python_execution_context(self.model,identity)
        self.worker_identity=deepcopy(identity);self.binding=binding
        self.mechanism='17.spatial_padding';self.owner=owner;self.generation=generation-1
        self.simulation=binding.simulation
        base=Path(__file__).parent
        names=('vision17_group.py','vision17_request.py','vision17_worker.py','vision17_context.py',
               'vision17_trace.py','vision17_padding.py','conv3d_natural.py','_scoped_dispatch.py','tensor_callable.py')
        self.source_manifest={name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names}
        for module_name in ('fault_injection.coverage','fault_injection.coverage_session',
            'demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural',
            'demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_group'):
            module=importlib.import_module(module_name)
            self.source_manifest[module_name]=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        self.source_sha=_sha(self.source_manifest)
        self.scope=CoverageScope(self.mechanism,binding.workload_sha256,binding.environment_sha256,self.source_sha,
            evidence_kind='cpu_contract' if binding.simulation else 'natural_worker',
            model_weights_sha256=binding.weights_sha256,model_source_sha256=binding.model_source_sha256)
        self.references=None;self.rows=[];self.calls=[];self.selected={};self.target_guards={}
        self.candidates=self.group=self.active_adapter=self.off_row=self.current_request=None
        self.enabled=False;self._running=False;self.error=None;self.arm=None;self.decisions=[]
        self.owned_host_hooks=lambda:0
        self.synchronize=binding.synchronize if binding.simulation else self._synchronize_npu

    @staticmethod
    def _synchronize_npu():
        import torch
        torch.npu.synchronize()

    def begin_request(self,request,phase,arm):
        require(not self._running and self.active_adapter is None,'GROUP_PREVIOUS_REQUEST_UNFINISHED')
        require(request.model is self.model and request.visual is self.visual,
                'GROUP_ACTUAL_MODEL_VISUAL_LIFETIME_CHANGED')
        for guard in self.target_guards.values():guard.verify()
        self.current_request=request;self.phase,self.arm=phase,arm
        self._running=True;self.calls=[];self.decisions=[]
        self.covering=self.group is not None and not self.group.recovery_report()['complete']

    def observe(self,row):
        request=self.current_request
        require(self._running and request.current and request.current['encoder_observed']
                and request.current['request_binding']['single_image_request_join_proven'],
                'GROUP_ACTUAL_IMAGE_REQUEST_MAPPING_REQUIRED')
        key=(row['path'],row['sequence'])
        self.calls.append(dict(key=key,initial=row))
        if self.covering:
            from .coverage import ObservedCall,Phase
            require(key in self.selected,'GROUP_INDEPENDENT_DENOMINATOR_CHANGED')
            require(row['module_object_id']==self.selected[key].module_object_identity,
                    'GROUP_ACTUAL_TARGET_MODULE_REPLACED')
            phase=Phase.OFF_RECOVERY if self.arm is None else (
                Phase.PRE_FAULT_PROPAGATION if row['sequence']==0 else Phase.POST_FIRST_ANOMALY)
            invocation=request.pending
            call=ObservedCall(self.selected[key],invocation+':'+str(row['sequence']),invocation,(invocation,),phase)
            decision=self.group.begin(self.group.observer.observe(call,eligible=True))
            self.decisions.append(decision)

    def admit(self,event):
        key=(event['path'],event['ordinal'])
        require(self.group is not None and self.arm=='fault' and len(self.calls)>key[1]
                and self.calls[key[1]]['key']==key and len(self.decisions)>key[1],
                'GROUP_INDEPENDENT_OBSERVATION_REQUIRED_BEFORE_F')
        return self.enabled and self.decisions[key[1]].admitted

    def collect(self,row):
        from .coverage import Outcome
        require(self._running and self.current_request.closed,'GROUP_REQUEST_HOOKS_MUST_BE_CLOSED')
        try:
            initial=row['initial_observation'];final=row['conv_observation']
            require(initial['torch_dispatch_chain_complete'] and final['torch_dispatch_chain_complete']
                and not initial['hooks'] and not final['hooks'] and not initial['owned_mode'] and not final['owned_mode']
                and len(initial['observed_calls'])==len(final['observed_calls'])==len(self.calls)>0,
                'GROUP_COMPLETE_INDEPENDENT_ENCODER_CHAIN_REQUIRED')
            require(set(initial['structural_candidates'])=={c['path'] for c in initial['observed_calls']},
                    'GROUP_STRUCTURAL_CANDIDATE_NOT_OBSERVED')
            events=(row['mechanism'] or {}).get('events',[])
            require(row['mechanism'] is None or len(events)==len(self.calls),'GROUP_ADAPTER_CALL_COUNT_MISMATCH')
            for index,(before,after) in enumerate(zip(initial['observed_calls'],final['observed_calls'])):
                require(before['path']==after['path'] and before['sequence']==after['sequence']==index
                    and before['inputs_before']==after['inputs_before'] and before['output']==after['output']
                    and before['inputs_unchanged'] and after['inputs_unchanged'],
                    'GROUP_UPPER_LOWER_ACTUAL_CALL_JOIN_FAILED')
                expected=[0,7,7] if self.arm=='fault' else [0,0,0]
                require(before['actual_geometry']['padding']==[0,0,0]
                    and after['actual_geometry']==dict(before['actual_geometry'],padding=expected),
                    'GROUP_ACTUAL_FINAL_PADDING_OR_NON_TARGET_ARGUMENT_CHANGED')
                if self.candidates is not None:
                    reference=self.rows[1]['initial_observation']['observed_calls'][index]
                    require(call_signature(before)==call_signature(reference),'GROUP_ACTUAL_INPUT_OR_CALL_CHANGED')
                if self.covering:
                    decision=self.decisions[index]
                    if self.arm=='fault':
                        event=events[index];baseline=self.rows[1]['conv_observation']['observed_calls'][index]
                        require(decision.admitted and event.get('applied') is True and event.get('native_calls')==1
                            and event.get('original_execution_verified') is True and event.get('output_changed') is True
                            and after['output']!=baseline['output']
                            and after['consumer']['consumed_values']!=baseline['consumer']['consumed_values']
                            and after['consumer']['outputs']!=baseline['consumer']['outputs'],
                            'GROUP_ACTUAL_PADDING_EFFECT_AND_FIRST_CONSUMER_UNPROVEN')
                        self.group.finish(decision,Outcome.HIT,native_applied=True,original_calls=1,
                            execution_sha256=_sha(after),effect_sha256=_sha(dict(consumer=after['consumer'],output=output_identity(row))))
                    else:
                        require(not events,'GROUP_OFF_MUST_HAVE_NO_ADAPTER')
                        self.group.finish(decision,Outcome.PASSTHROUGH,execution_sha256=_sha(after))
            if self.candidates is not None and self.arm is None:
                require(output_identity(row)==self.reference_output,'GROUP_FULL_OFF_OUTPUT_NOT_EXACT')
                self.off_row=deepcopy(row)
            if self.arm=='fault':
                prior=[r for r in self.rows if r.get('fault_injections')]
                if prior:require(output_identity(prior[-1])==output_identity(row),'GROUP_FAULT_REPEAT_NOT_EXACT')
            self.rows.append(deepcopy(row))
        except BaseException as error:
            self.fail(error);raise
        finally:
            self._running=False;self.current_request=None

    def fail(self,error):
        from .coverage import Outcome
        self.error=repr(error)
        if self.group is not None:
            for decision in self.decisions:
                try:self.group.finish(decision,Outcome.FAILED,original_calls=0)
                except RuntimeError:pass # A finished receipt is immutable.
        self._running=False

    def qualify(self):
        from .coverage import BoundaryTargetIdentity,PythonHookBoundary,Candidate,CandidateSet,Eligibility,EligibilityResult
        from .tensor_callable import TensorCallableGuard
        require(self.candidates is None and len(self.rows)==4,'GROUP_FOUR_NEW_HEALTHY_PHASES_REQUIRED')
        require([r['phase'] for r in self.rows]==['native_baseline','baseline','baseline_repeat','lossless'],
                'GROUP_HEALTHY_PHASE_ORDER_CHANGED')
        first=self.rows[0];base=self.rows[1]
        signatures=[call_signature(c) for c in first['initial_observation']['observed_calls']]
        require(all(output_identity(r)==output_identity(first) and
                    [call_signature(c) for c in r['initial_observation']['observed_calls']]==signatures
                    and [c['output'] for c in r['conv_observation']['observed_calls']]==
                        [c['output'] for c in first['conv_observation']['observed_calls']]
                    for r in self.rows[1:]),'GROUP_HEALTHY_REPEAT_OR_LOSSLESS_BRIDGE_NOT_EXACT')
        # This checkpoint has live single-projection C1. Do not extrapolate its
        # qualification to several natural projections without separate C1.
        require(len(signatures)==1,'GROUP_MULTIPLE_PROJECTIONS_C1_NOT_ADAPTED')
        modules=dict(self.visual.named_modules());candidates=[]
        for row in base['initial_observation']['observed_calls']:
            key=(row['path'],row['sequence']);module=modules['' if key[0]=='<root>' else key[0]]
            guard=TensorCallableGuard(module);self.target_guards[key[0]]=guard
            metadata=row['actual_input_metadata'][0];device=metadata['device'].split(':')
            target=BoundaryTargetIdentity(process_id=self.context['process_id'],process_start_identity=self.context['process_start_identity'],
                worker_id=self.context['worker_id'],scope_kind=self.context['scope_kind'],rank=self.context['rank'],
                device_type=device[0],logical_device=None if device[0]=='cpu' else int(device[1]) if len(device)>1 else 0,
                model_instance_id=self.context['model_instance_id'],model_object_identity=id(self.model),
                module_object_identity=id(module),module_roles=('CONV3D_SPATIAL_PADDING_ARGUMENT',),operator_api=row['operator'],
                tensor_signature=json.dumps(call_signature(row),sort_keys=True),
                boundary=PythonHookBoundary(guard.callable_identity,guard.source_sha256,key[1]),
                provider_identity='vision17_public_torch_parameter',provider_proof=(('implementation_sha256',self.source_sha),))
            self.selected[key]=target
            candidates.append(Candidate(key[0]+':'+str(key[1]),EligibilityResult(Eligibility.SUPPORTED),target))
        self.candidates=CandidateSet(tuple(candidates),True)
        self.reference_output=output_identity(first);self.baseline_sha=_sha(self.reference_output)
        self.input_sha=_sha(signatures);self.lossless_sha=_sha(self.rows[3]);self.discovery_sha=_sha(self.rows)
        return dict(candidate_set=self.candidates.report(),native_baseline_exact=True,lossless_exact=True,
                    actual_target_count=len(candidates),multi_target_verified=False,aclnn_launch_arguments=False,
                    implementation_sha256=self.source_sha,source_manifest=self.source_manifest)

    def _recover(self,lease):
        from .coverage import RecoveryReport
        require(self.off_row is not None and output_identity(self.off_row)==self.reference_output,
                'GROUP_ACTUAL_COMPLETED_OFF_REQUEST_REQUIRED')
        return RecoveryReport(lease,True,True,True,proof_sha256=_sha(self.off_row),off_output_sha256=self.baseline_sha)

    def _health(self,worker,owner,generation):
        from .coverage_session import WorkerHealthProof
        require(not self._running and not self.owned_host_hooks() and self.off_row is not None,'GROUP_REAL_ZERO_HOOKS_REQUIRED')
        return WorkerHealthProof(worker,owner,generation,_sha(self.off_row),self.baseline_sha,
                                 _sha(output_identity(self.off_row)),0,0,0,0,0)

    def close_group(self):
        result=super().close_group()
        result.update(mechanism_semantics='TORCH_CONV3D_PADDING_ARGUMENT_0_TO_0_7_7',
            actual_target_count=len(self.selected),multi_target_verified=False,
            aclnn_launch_arguments=False,device_payload_observed=False,native_qualification=False,E5=False)
        return result
