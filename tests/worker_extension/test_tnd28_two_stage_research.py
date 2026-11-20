"""Research-only CPU contracts. No CANN, executor or kernel is simulated as qualified."""
from copy import deepcopy
from types import SimpleNamespace as NS
import ctypes as C
import json
import unittest
from unittest.mock import patch
from experiments.worker_core_v11_20261118 import tnd28_two_stage_research as R
from demoncat.runtime.vllm_ascend.worker_extension.attention28_typed_plan import SYMBOLS
from demoncat.runtime.vllm_ascend.worker_extension.providers.attention_serving import FiaDenominatorReplayServingProvider

class ResearchContracts(unittest.TestCase):
    def test_research_binding_reuses_worker_checks_without_public_allowlist_extension(self):
        from demoncat.runtime.vllm_ascend.worker_extension import serving_attention as S
        import os,threading
        worker=NS(model_runner=NS(model=NS(training=False)))
        native=R.TwoStageNativeResearch.__new__(R.TwoStageNativeResearch)
        native.worker=worker;native.mechanism='28.fia.denominator_replay'
        state=dict(native_loaded=True,scope_active=False,scope_owned=False,environment_owned=0)
        with patch.object(R.TwoStageNativeResearch,'state',return_value=state),patch.object(S,'require_serial_eager'):
            with self.assertRaisesRegex(RuntimeError,'ATTENTION_REVIEWED_NATIVE_ADAPTER_REQUIRED'):
                S.install_attention_serving_binding(worker,native.mechanism,prompt_token_ids=[1],provenance={},native=native)
            self.assertFalse(hasattr(worker,S.ATTRIBUTE))
            bound=R.install_research_binding(worker,prompt_token_ids=[1],provenance={},native=native)
            self.assertIs(bound.native,native);self.assertFalse(bound.simulation)
            self.assertEqual(bound.pid,os.getpid());self.assertEqual(bound.thread,threading.get_ident())
            S.remove_attention_serving_binding(worker,native.mechanism)
            state['scope_owned']=True
            with self.assertRaisesRegex(RuntimeError,'RESEARCH_NATIVE_NOT_INERT'):
                R.install_research_binding(worker,prompt_token_ids=[1],provenance={},native=native)
            self.assertFalse(getattr(worker,S.ATTRIBUTE))

    def test_real_tnd_boundary_uses_existing_strict_contract(self):
        import torch
        import test_attention28_stock_census as CENSUS
        from demoncat.runtime.vllm_ascend.worker_extension import attention28_stock_census as C
        from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import AttentionExecutionAdapter
        a=AttentionExecutionAdapter.__new__(AttentionExecutionAdapter)
        a.model=object();module=NS(layer_name='actual');row=CENSUS.call_row();row['module_object_id']=id(module)
        a.deployment=NS(mechanism='28.fia.denominator_replay',simulation=True)
        a.current=dict(decode=False,scheduler=row['scheduler'],calls=[row]);a.live_api_calls=[(row,object(),(),{})]
        a._context=lambda:NS(model_instance=a.model,no_compile_layers={'actual':module},attn_metadata={'actual':object()})
        actual=dict(input_layout='TND',atten_mask=torch.ones(2,2,dtype=torch.int8).triu(1))
        with patch.object(C,'capture_inputs',return_value=CENSUS.captured()):
            result=a._boundary('npu_fused_infer_attention_score',actual,row['module_path'],module)
            self.assertTrue(result['input_contract_matches']);self.assertFalse(result['public_worker_qualified'])
            actual['atten_mask'][0,0]=1
            with self.assertRaisesRegex(RuntimeError,'CAUSAL_MASK'):a._boundary('npu_fused_infer_attention_score',actual,row['module_path'],module)
        bad=CENSUS.captured();bad['inputs']['block_table']['value']=[1]
        with patch.object(C,'capture_inputs',return_value=bad):
            with self.assertRaisesRegex(RuntimeError,'SEMANTICS_DIFFER'):a._boundary('npu_fused_infer_attention_score',actual,row['module_path'],module)

    def native_state(self,value):
        p=R.ResearchTypedPlan.__new__(R.ResearchTypedPlan);p.binding=NS(_owner=lambda:None);p.token='a'*32
        def query(token,out,capacity):
            raw=json.dumps(value).encode()+b'\0';C.memmove(out,raw,len(raw));return 0
        p.functions={SYMBOLS[2]:query};return p.state()
    def test_final_state_requires_each_native_gate(self):
        good=dict(schema='demoncat.tnd28.live_typed_plan.v2',fault_authorized=True,replay_parent_committed=1,
            replay_requested=1,host_completed=1,copy_launch_read=1,host_gate_checks=4,value_read_mask=31,first_error=0)
        self.assertTrue(self.native_state(good)['fault_authorized'])
        for name,value in [('replay_parent_committed',0),('replay_requested',0),('host_completed',0),
                           ('copy_launch_read',0),('host_gate_checks',0),('value_read_mask',15),('first_error',1)]:
            with self.subTest(name=name):
                row=dict(good,**{name:value})
                with self.assertRaises(RuntimeError):self.native_state(row)
    def provider(self,rows):
        p=R.TwoStageResearchProvider.__new__(R.TwoStageResearchProvider);p.records=rows
        p.binding=NS(native=R.TwoStageNativeResearch.__new__(R.TwoStageNativeResearch));p.selection={'target_key':'actual'}
        p.session=NS(needs_off=True);return p
    def row(self,phase,hidden='normal',native=None,output='normal'):
        calls=[] if phase=='health' else [dict(target_key='actual',input='same-input',output=output,native=native)]
        return dict(phase=phase,forwards=[dict(input='same-request',hidden=hidden,logits=hidden,calls=calls)])
    def collect(self,p):
        with patch.object(FiaDenominatorReplayServingProvider,'_collect',return_value=None):p._collect()
    def test_fault_requires_repeat_baseline_and_exact_driver(self):
        p=self.provider([])
        with self.assertRaises(RuntimeError):p._prepare_phase('fault')
        p.records=[dict(phase=x) for x in ('observe','observe_repeat','baseline','baseline_repeat')]
        p._prepare_phase('fault');p.binding.native=NS()
        with self.assertRaises(RuntimeError):p._prepare_phase('fault')
    def test_fault_output_must_change_with_same_inputs(self):
        baseline=self.row('baseline_repeat')
        fault=self.row('fault',hidden='changed',native=dict(selection_applied=True,fault_authorized=True),output='changed')
        p=self.provider([baseline,fault]);self.collect(p)
        for key,value in [('input','different'),('output','normal')]:
            altered=deepcopy(fault);altered['forwards'][0]['calls'][0][key]=value
            with self.assertRaises(RuntimeError):self.collect(self.provider([baseline,altered]))
    def test_off_latch_clears_only_after_complete_values_and_actual_native_off(self):
        baseline=self.row('baseline')
        off=self.row('off',native=dict(selection_applied=False,fault_authorized=False))
        p=self.provider([baseline,off]);self.collect(p);self.assertFalse(p.session.needs_off)
        for key,value in [('hidden','wrong'),('logits','wrong')]:
            altered=deepcopy(off);altered['forwards'][0][key]=value;p=self.provider([baseline,altered])
            with self.assertRaises(RuntimeError):self.collect(p)
            self.assertTrue(p.session.needs_off)
        altered=deepcopy(off);altered['forwards'][0]['calls'][0]['native']['fault_authorized']=True
        p=self.provider([baseline,altered])
        with self.assertRaises(RuntimeError):self.collect(p)
        self.assertTrue(p.session.needs_off)
    def test_health_rejects_attention_interception(self):
        p=self.provider([self.row('baseline'),self.row('health')])
        p.session=R.HealthOnlyAdapter.__new__(R.HealthOnlyAdapter);p.session.patches=[];p.session.handles=[1,2,3]
        self.collect(p)
        p.session.patches=['fault-wrapper']
        with self.assertRaises(RuntimeError):self.collect(p)
    def test_normal_capture_still_observe_only(self):
        p=R.PreparedInputCapture.__new__(R.PreparedInputCapture)
        self.assertEqual(p._capture_phase(),'observe')
        self.assertFalse(R.TwoStageNativeResearch.observer_only)
        self.assertEqual(R.TwoStageResearchProvider.phases,('observe','observe_repeat','baseline','baseline_repeat','fault','off','health'))

if __name__=='__main__':unittest.main()
