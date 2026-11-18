"""Public R/L/OFF/F/OFF protocol for one proved arch22 first-layer site."""
from copy import deepcopy

from .attention_serving import AttentionServingProvider,PagedDecodeServingProvider,request_values,require
from ..paged33_arch22_s2_effects import Arch22S2EffectsNativeCall,PHASES
from ..paged33_s2_execution import Paged33S2ExecutionAdapter,LAYERS
from ..paged33_s2_evidence import FIRST,digest
from ..serving_attention import ATTRIBUTE


def payloads(row):
    values={}
    for key,value in row['payloads'].items():
        metadata=value.get('metadata',value)
        values[key]=dict(shape=metadata['shape'],dtype=metadata['dtype'],bytes=value['bytes'],sha256=value['sha256'])
    return values


def selected(row):
    forwards=row['forwards']
    require(len(forwards)==2 and [item['decode'] for item in forwards]==[False,True],
            'S2_PREFILL_AND_FIRST_DECODE_REQUIRED')
    require(all(len(item['calls'])==28 and {call['module_path'] for call in item['calls']}==set(LAYERS)
        and all(call['original_calls']==1 and call['original_returned'] is True and call['inputs_unchanged'] is True
                for call in item['calls']) for item in forwards),'S2_COMPLETE_56_ORIGINAL_CALL_DENOMINATOR_REQUIRED')
    hits=[call for item in forwards for call in item['calls'] if call['native'] is not None]
    require(len(hits)==1 and hits[0]['module_path']==FIRST and hits[0]['occurrence']==0
        and hits[0] in forwards[1]['calls'],'S2_EXACT_ONE_FIRST_LAYER_NATIVE_CALL_REQUIRED')
    return hits[0]


class Paged33S2ServingProvider(AttentionServingProvider):
    mechanism='33.paged_decode'
    phases=PHASES
    execution_adapter_type=Paged33S2ExecutionAdapter

    def __init__(self,worker,model,identity):
        super().__init__(worker,model,identity)
        require(type(self.binding.native) is Arch22S2EffectsNativeCall or self.binding.simulation
            and getattr(self.binding.native,'public_single_site_protocol',False),'S2_TRUSTED_EFFECT_DRIVER_REQUIRED')
        self.functional_ready=False;self.effect=None;self.single_site_proof=None

    def _prepare_phase(self,phase):
        require(self.binding.native.state()['native_loaded'],'S2_REAL_NATIVE_DEPLOYMENT_REQUIRED')
        if phase=='fault':
            require(self.index==3 and self.binding.native.normal_gate is not None
                and not self.binding.native.effect_attempted,'S2_CURRENT_LIVE_NORMAL_CONTROLS_REQUIRED')
        if phase=='recovery_off':
            require(self.binding.native.effect_attempted,'S2_ACTUAL_FAULT_ATTEMPT_REQUIRED_FOR_OFF')

    def dispatch(self,action,settings,*,owner,generation):
        # After an uncertain F, only a fresh same-owner OFF request may proceed.
        # Archive the failed request, never erase its error or retry the fault.
        if action=='recover' and settings=={} and self.session is not None \
                and self.session.needs_off and self.binding.native.effect_attempted \
                and (self.error is not None or self.session.last_error is not None or self.pending is not None):
            self.verify(owner,generation);self._context()
            try:self.session.prepare_off_recovery()
            except BaseException as error:
                self._failure('prepare_recovery_off',error)
                raise
            self.functional_ready=False;self.pending=None;self.index=4;self.error=None
            return self.query(owner=owner,generation=generation)
        return super().dispatch(action,settings,owner=owner,generation=generation)

    def _collect(self):
        require(self.pending is not None,'S2_ACTUAL_PENDING_REQUEST_REQUIRED')
        row=self.session.collect()
        require((row['request_id'],row['phase'])==self.pending,'S2_ACTUAL_REQUEST_MAPPING_CHANGED')
        self.records.append(row);self.pending=None;self.index+=1
        phase=row['phase'];call=selected(row);native=call['native'];baseline=self.records[0]
        require(native['phase']==phase and native['module_path']==FIRST and native['original_calls']==1
            and (native['owner'],native['generation'])==self._ticket
            and native['logical_input_binding']==row['live_input'],'S2_CURRENT_NATIVE_PUBLIC_INPUT_JOIN_CHANGED')
        require(native['selection_applied'] is (phase=='fault') and native['geometry']['selected_s2']==(256 if phase=='fault' else 512),
                'S2_ACTUAL_SELECTION_DIFFERS_FROM_PROTOCOL')
        if phase=='reference':
            normal=self.binding.native.normal_evidence.data()
            require(all(row['forwards'][1][key]==normal['model_outputs'][key] for key in ('hidden','logits')),
                    'S2_CURRENT_MODEL_OUTPUT_DIFFERS_FROM_FRESH_NORMAL')
            require(all(value['sha256']==expected['sha256'] for value,expected in
                zip(row['immediate_outputs'],normal['immediate_outputs'])) and len(row['immediate_outputs'])==2,
                'S2_CURRENT_FIRST_OUTPUT_DIFFERS_FROM_FRESH_NORMAL')
            self.selection=dict(target_key=digest(dict(module_path=FIRST,module_object_id=call['module_object_id'],
                worker=self.identity,scope='TND_Q1_KV1025_PAGE16_FP16')),module_path=FIRST,
                module_object_id=call['module_object_id'],actual_denominator=56,decode_denominator=28,
                normal_proof_scoped_targets=1,other_decode_targets=27,
                selector='EXACT_ACTUAL_FIRST_LAYER_NORMAL_SCOPE',other_targets='OBSERVED_NOT_IN_CURRENT_NORMAL_PROOF_SCOPE')
        else:
            require(call['module_object_id']==self.selection['module_object_id'],'S2_ACTUAL_TARGET_MODULE_CHANGED')
            for index,(first,current) in enumerate(zip(baseline['forwards'],row['forwards'])):
                require(first['input']==current['input'],'S2_ACTUAL_ROOT_INPUT_DIFFERS')
                require([(x['module_path'],x['actual_operator'],x['occurrence']) for x in first['calls']]==
                        [(x['module_path'],x['actual_operator'],x['occurrence']) for x in current['calls']],
                        'S2_NATURAL_CALL_ORDER_OR_DENOMINATOR_CHANGED')
                if index==0:require(request_values(baseline)[0]==request_values(row)[0],'S2_NON_TARGET_PREFILL_CHANGED')
            if phase!='fault':
                require(request_values(row)==request_values(baseline) and payloads(row)==payloads(baseline),
                        'S2_COMPLETE_NORMAL_OR_OFF_OUTPUT_NOT_EXACT')
            if phase=='qualification_off':self.binding.native.authorize_normal(self.session)
            elif phase=='fault':
                changed=[name for name,value in payloads(row).items() if value!=payloads(baseline)[name]]
                self.effect=dict(changed_payloads=changed,first_attention_output_changed=FIRST+'/output' in changed,
                    full_hidden_changed='hidden' in changed,full_logits_changed='logits' in changed,
                    actual_native_token=native['call_id'])
                require(self.effect['first_attention_output_changed'],
                        'S2_SELECTED_PLAN_WITHOUT_REQUIRED_OBSERVED_OUTPUT_EFFECT')
            elif phase=='recovery_off':
                self.binding.native.confirm_off(self.session);self.session.needs_off=False
                self.functional_ready=bool(self.effect and self.effect['first_attention_output_changed'] and len(self.records)==5
                    and not self.failures and not self.session.failures)
                if self.functional_ready:
                    self.single_site_proof=dict(owner=self._ticket[0],generation=self._ticket[1],
                        target=deepcopy(self.selection),request_ids=[item['request_id'] for item in self.records],
                        native_tokens=[selected(item)['native']['call_id'] for item in self.records],
                        normal_gate=deepcopy(self.binding.native.normal_gate),effect=deepcopy(self.effect),
                        full_off_payloads_exact=True,serializable_authority=False,scope='C1_SINGLE_SITE_ONLY',C2=False,C3=False)

    def query(self,*,owner,generation):
        result=super().query(owner=owner,generation=generation)
        ready=bool(self.functional_ready and not self.binding.simulation and self.stopped
            and result['leases']==result['owned_hooks']==result['owned_api_wrappers']==result['inflight']==0
            and result['serving']['resource_state'].get('owned_input_references',0)==0 and not self.failures)
        result.update(C1=ready,C2=False,C3=False,public_qualification_granted=ready)
        result['evidence'].update(single_site_c1=deepcopy(self.single_site_proof),effect=deepcopy(self.effect),
            stock_private_bridge='FRESH_ORIGINAL_RAW_REPLAY_WITH_LIVE_INPUT_BINDING',
            fault_admission='CURRENT_LIVE_CONTROLS_REQUIRED' if self.binding.native.normal_gate is None else 'ONE_OWNED_FIRST_LAYER_SCOPE',
            qualified_targets=[deepcopy(self.selection)] if ready else [],
            normal_evidence_sha256=self.binding.native.normal_evidence.fingerprint,
            lossless_control_scope='SECOND_ORIGINAL_S2_512_REPLAY_NO_FAULT_SELECTION_OR_OUTPUT_REWRITE',
            functional_protocol_complete=self.functional_ready,
            C2_blocker='OTHER_27_NATURAL_DECODE_TARGETS_HAVE_NO_LIVE_NORMAL_AND_EFFECT_PROOF',
            C3_blocker='NO_LIVE_COMPLETE_C2_PROOF_ISSUED_OR_USED')
        return result


def resolve_paged_decode_provider(worker,model,identity):
    """Registry mapping keeps stock census available, selects only exact driver."""
    table=getattr(worker,ATTRIBUTE,None);binding=table.get('33.paged_decode') if type(table) is dict else None
    from ..paged33_group_native import GroupEffectsNativeCall
    if binding is not None and type(binding.native) is GroupEffectsNativeCall:
        from .paged33_group import Paged33GroupServingProvider
        return Paged33GroupServingProvider(worker,model,identity)
    if binding is not None and type(binding.native) is Arch22S2EffectsNativeCall:
        return Paged33S2ServingProvider(worker,model,identity)
    return PagedDecodeServingProvider(worker,model,identity)
