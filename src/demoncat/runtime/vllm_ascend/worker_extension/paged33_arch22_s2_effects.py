"""Actual first-layer S2 effect, admitted only by fresh normal + live controls."""
import json
import os
import re
import threading
import uuid

from .paged33_arch22_s2_normal import Arch22S2NormalNativeCall
from .paged33_arch22_bridge import validate_normal_call,validate_scalar_geometry
from .paged33_logical_binding import LogicalPagedInput,compare_logical_inputs
from .paged33_s2_evidence import VerifiedNormalEvidence,FIRST,digest
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import verify_launch
from fault_injection.native_adapters.attention28.fia.arch22_s2_build_evidence import verify_scope_geometry

PHASES=('reference','lossless','qualification_off','fault','recovery_off')


def validate_s2_launch(launch,host,actual,boundary,*,token,owner,generation,phase,rows):
    tile=256 if phase=='fault' else 512
    native_phase='fault' if phase=='fault' else 'baseline' if phase in ('reference','lossless') else 'off'
    require(boundary['module_path']==FIRST and boundary['sequence_lengths']==[1025]
        and host.get('schema')=='demoncat.paged33.arch22_s2_host.v1'
        and host.get('mode')==('arch22_s2_256' if tile==256 else 'arch22_s2_off')
        and host.get('phase')==native_phase and host.get('target_path')==FIRST
        and host.get('owner')==owner and type(host.get('generation')) is int and host['generation']==generation
        and host.get('token')==token and (host.get('pid'),host.get('tid'))==(os.getpid(),threading.get_native_id())
        and host.get('actual_observer_scope_matched') is True and host.get('eligible') is True
        and host.get('route')=='FiaTilingNonQuant' and host.get('normal_route_preference_changed') is True
        and host.get('policy_applied') is (tile==256),'S2_EFFECT_ACTUAL_HOST_SCOPE_DIFFERS')
    require(launch['tiling_key']==103000000010300003 and launch['schedule_mode']==1
        and launch['block_dim']==host['block_dim'],'S2_EFFECT_ACTUAL_ENTRY_DIFFERS')
    tokens=validate_scalar_geometry(launch['typed'],actual,boundary,selected_s2=tile)
    require(host['aic'] in rows[tile],'S2_EFFECT_ACTUAL_CPU_MATRIX_ROW_MISSING')
    split=verify_scope_geometry(launch['typed'],host,rows[tile][host['aic']])
    return dict(selected_s2=tile,normal_only=tile==512,actual_split=split,public_to_native_token_window=tokens,
        fault_definition='ORIGINAL_S2_SELECTION_512_TO_256_BEFORE_ORIGINAL_SPLIT_CORE',
        output_mutated_by_adapter=False,stock_abi_decoded=False)


class Arch22S2EffectsNativeCall(Arch22S2NormalNativeCall):
    public_single_site_protocol=True

    def __init__(self,*,normal_evidence,**kwargs):
        require(type(normal_evidence) is VerifiedNormalEvidence and not normal_evidence.simulation,
                'S2_VERIFIED_FRESH_NORMAL_OBJECT_REQUIRED')
        self.normal_evidence=normal_evidence
        self.owner=None;self.owner_ticket=None;self.live_inputs={};self.failed_off_inputs=[];self.attempts=[]
        self.normal_gate=None;self.effect_attempted=False;self.off_confirmed=False;self.cleanup_error=None
        super().__init__(**kwargs)
        data=normal_evidence.data()
        require(data['completion_sha256']==self.fingerprint['completion_sha256']
            and data['observer_receipt_sha256']==kwargs['observer_receipt_sha256'],
            'S2_NORMAL_NATIVE_BUILD_OR_OBSERVER_DIFFERS')
        self.fingerprint.update(single_site=True,normal_only=False,normal_evidence_sha256=normal_evidence.fingerprint,
            actual_scope='LAYER0_TND_Q1_KV1025_PAGE16_FP16',c2=False,c3=False)

    def bind_owner(self,owner,ticket):
        require(self.owner is None and type(ticket) is tuple and len(ticket)==2
            and re.fullmatch('[0-9a-f]{32}',ticket[0] or '') and type(ticket[1]) is int and ticket[1]>0,
            'S2_EXACT_PUBLIC_OWNER_GENERATION_REQUIRED')
        self._check();require(not self.calls and not self.live_inputs,'S2_NATIVE_ALREADY_USED')
        self.owner,self.owner_ticket=owner,ticket

    def _owned(self,owner,generation):
        self._check()
        require(self.owner is owner and self.owner_ticket is not None and self.owner_ticket[1]==generation
            and owner.deployment.leases=={self.owner_ticket},'S2_LIVE_PUBLIC_OWNER_CHANGED')
        owner.guard()

    def state(self):
        value=super().state()
        # These are Python observation owners held between completed requests,
        # not outstanding Native executions or allocations. The adapter reports
        # them separately and refuses final cleanup until every one is released.
        value.update(retained_input_objects=sum(not item.closed for item in self._retained_inputs()),
            single_site_protocol=True,normal_gate_issued=self.normal_gate is not None,
            effect_attempted=self.effect_attempted,off_confirmed=self.off_confirmed,
            normal_evidence_sha256=self.normal_evidence.fingerprint,
            first_error=value.get('first_error') or self.cleanup_error)
        return value

    def _retained_inputs(self):return [*self.live_inputs.values(),*self.failed_off_inputs]

    def prepare_recovery_off(self,owner):
        self._owned(owner,self.owner_ticket[1])
        require(self.effect_attempted and not self.off_confirmed,'S2_OFF_RECOVERY_NOT_PENDING')
        state=self.stop()
        require(not state['scope_owned'] and not state['scope_active'] and state['environment_owned']==0,
                'S2_NATIVE_ZERO_SCOPE_BEFORE_NEW_OFF_REQUIRED')
        # Never retry F. A failed OFF may need a fresh actual request after its
        # original returned model output was uncertain. Keep every old input
        # owner and original receipt until the successful OFF and final drain.
        if 'recovery_off' in self.live_inputs:
            self.failed_off_inputs.append(self.live_inputs.pop('recovery_off'))

    def authorize_normal(self,owner):
        self._owned(owner,self.owner_ticket[1])
        require(self.normal_gate is None and [r['phase'] for r in owner.rows]==list(PHASES[:3])
            and set(self.live_inputs)==set(PHASES[:3]) and len(self.calls)==3,
            'S2_THREE_ACTUAL_NORMAL_REQUESTS_REQUIRED')
        first=self.live_inputs['reference'];comparisons=[]
        for phase in PHASES[1:3]:
            comparisons.append(compare_logical_inputs(first,self.live_inputs[phase],owner=owner,generation=self.owner_ticket[1]))
        require(all(item.complete and item.first_error is None for item in self.live_inputs.values()),
                'S2_ACTUAL_NORMAL_INPUT_INCOMPLETE')
        self.normal_gate=dict(owner=self.owner_ticket[0],generation=self.owner_ticket[1],
            normal_evidence_sha256=self.normal_evidence.fingerprint,
            actual_native_tokens=[row['call_id'] for row in self.calls],
            actual_request_ids=[row['request_id'] for row in owner.rows],live_comparisons=comparisons,
            model_rows_sha256=digest(owner.rows),serializable_authority=False)
        return dict(self.normal_gate)

    def execute(self,actual,boundary,phase,invoke_once,*,binding,owner,generation):
        self._owned(owner,generation)
        require(phase in PHASES and phase not in self.live_inputs and type(binding) is LogicalPagedInput
            and binding.owner is owner and binding.generation==generation and binding.actual is actual
            and binding.worker is owner.worker and binding.model is owner.model,
            'S2_SINGLE_ACTUAL_PHASE_INPUT_REQUIRED')
        self.live_inputs[phase]=binding  # Retain before validation or any uncertain Native call.
        expected=PHASES[len(self.attempts)] if len(self.attempts)<len(PHASES) else None
        require(phase==expected or phase=='recovery_off' and self.effect_attempted,'S2_NATIVE_PHASE_ORDER_CHANGED')
        scope=validate_normal_call(actual,boundary)
        reference_match=self.normal_evidence.match(binding)
        if phase not in ('reference','lossless','qualification_off'):
            require(self.normal_gate is not None,'S2_LIVE_NORMAL_GATE_REQUIRED')
            first=self.live_inputs['reference']
            require(all(getattr(first,key) is getattr(binding,key) for key in
                ('worker','runner','model','module','impl')) and all(a is b for a,b in zip(first.caches,binding.caches)),
                'S2_LIVE_REFERENCE_WORKER_OR_STORAGE_OWNER_CHANGED')
            # The current binding has not executed yet. Its fully checked
            # before-snapshot must agree before opening the actual TLS scope.
            require(self.normal_evidence.match(self.live_inputs['reference'])['logical_input_equivalent'],
                    'S2_REFERENCE_INPUT_AUTHORITY_CHANGED')
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'],platform_ready=True)
        native_phase='fault' if phase=='fault' else 'baseline' if phase in ('reference','lossless') else 'off'
        token=uuid.uuid4().hex;log=self.folder/'host.jsonl';tile=256 if phase=='fault' else 512
        environment=dict(DCAT33_TND_MODE='arch22_s2_256' if tile==256 else 'arch22_s2_off',
            DCAT33_TND_TOKEN=token,DCAT33_TND_LOG=str(log),DCAT33_TND_PID=str(os.getpid()),
            DCAT33_TND_TID=str(threading.get_native_id()),DCAT33_TND_OWNER=self.owner_ticket[0],
            DCAT33_TND_GENERATION=str(generation),DCAT33_TND_PHASE=native_phase,DCAT33_TND_TARGET=FIRST)
        self.attempts.append(phase)
        if phase=='fault':self.effect_attempted=True
        def verify():
            hosts=[json.loads(line) for line in log.read_text().splitlines()]
            hosts=[row for row in hosts if row.get('token')==token]
            require(len(hosts)==1,'S2_ONE_ACTUAL_HOST_CALLBACK_REQUIRED');host=hosts[0]
            launch=verify_launch(self.runtime_dir,token,schema=self.schema,host=host,
                kernel_sha256=self.fingerprint['kernel_sha256'],kernel_metadata=self.metadata,kernel_abi=self.abi)
            geometry=validate_s2_launch(launch,host,actual,boundary,token=token,owner=self.owner_ticket[0],
                generation=generation,phase=phase,rows=self.build['s2_cpu_matrix']['first_decode_rows'])
            if phase!='reference':
                live=compare_logical_inputs(self.live_inputs['reference'],binding,owner=owner,generation=generation)
            else:live=None
            return dict(call_id=token,token=token,phase=phase,native_phase=native_phase,module_path=FIRST,
                module_object_id=boundary['module_object_id'],scheduler=boundary['scheduler'],
                virtual_engine=boundary.get('virtual_engine'),owner=self.owner_ticket[0],generation=generation,
                native=launch,host=host,scope=scope,geometry=geometry,original_calls=1,
                selection_applied=tile==256,api_arguments_changed=False,cache_repacked=False,output_mutated=False,
                mechanism=self.mechanism,reference_match=reference_match,live_input_comparison=live,
                normal_gate=self.normal_gate,logical_input_binding=binding.report())
        return self._run(token,environment,lambda:binding.execute(invoke_once,owner=owner,generation=generation),verify)

    def confirm_off(self,owner):
        self._owned(owner,self.owner_ticket[1])
        require(self.effect_attempted and 'recovery_off' in self.live_inputs
            and self.live_inputs['recovery_off'].complete and self.calls[-1]['phase']=='recovery_off'
            and self.calls[-1]['selection_applied'] is False,'S2_ACTUAL_OFF_RETURN_REQUIRED')
        self.off_confirmed=True

    def release_inputs(self,owner):
        self._owned(owner,self.owner_ticket[1])
        require(not self.effect_attempted or self.off_confirmed,'S2_ACTUAL_OFF_BEFORE_RELEASE_REQUIRED')
        try:
            state=self.stop()
            require(not state['scope_owned'] and not state['scope_active'] and state['environment_owned']==0,
                    'S2_NATIVE_ZERO_SCOPE_BEFORE_RELEASE_REQUIRED')
            for item in self._retained_inputs():item.release(owner=owner,generation=self.owner_ticket[1])
        except BaseException as error:
            if self.cleanup_error is None:self.cleanup_error=repr(error)
            raise
        return self.state()
