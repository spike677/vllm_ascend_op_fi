"""Real CPU module/API calls through public Host/Runtime; Native is a seam.

No simulated call is reported as NPU or C1. The 28-module model is deliberately
small arithmetic and shares CPU cache storage; it is not a real-model proof.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime,attention_execution
from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import complete
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding,remove_attention_serving_binding
from demoncat.runtime.vllm_ascend.worker_extension.paged33_arch22_s2_effects import Arch22S2EffectsNativeCall
from demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding import LogicalPagedInput
from demoncat.runtime.vllm_ascend.worker_extension import paged33_s2_evidence as evidence
from demoncat.runtime.vllm_ascend.worker_extension.providers.paged33_s2 import Paged33S2ServingProvider
from test_paged33_logical_binding import Fixture,PATH
from test_extension import _extended_worker
from test_core_host import ExistingExecutor


class CpuEffects(Arch22S2EffectsNativeCall):
    simulation=True
    def __init__(self,normal,folder):
        self.normal_evidence=normal;self.folder=folder;folder.mkdir()
        self.thread=threading.get_ident();self.owner=self.owner_ticket=None
        self.live_inputs={};self.failed_off_inputs=[];self.attempts=[];self.calls=[];self.failures=[];self.normal_gate=None
        self.effect_attempted=self.off_confirmed=False;self.cleanup_error=None
        self.active_phase=None;self.fail_stop=False;self.fail_after_fault=False;self.fail_after_off=False
    def _check(self):
        if threading.get_ident()!=self.thread:raise RuntimeError('CPU_FOREIGN_THREAD')
    def state(self):
        self._check()
        return dict(native_loaded=True,scope_active=False,scope_owned=False,environment_owned=0,
            retained_input_objects=sum(not value.closed for value in self._retained_inputs()),
            calls=len(self.calls),failures=deepcopy(self.failures),simulation=True,
            first_error=self.cleanup_error,
            effect_attempted=self.effect_attempted,off_confirmed=self.off_confirmed)
    def stop(self):
        self._check()
        if self.fail_stop:raise RuntimeError('CPU_UNCERTAIN_STOP')
        return self.state()
    def execute(self,actual,boundary,phase,once,*,binding,owner,generation):
        self._owned(owner,generation)
        self.normal_evidence.match(binding)
        if phase=='fault' and self.normal_gate is None:raise RuntimeError('CPU_NO_LIVE_NORMAL_GATE')
        if phase in self.live_inputs:raise RuntimeError('CPU_DUPLICATE_ACTUAL_PHASE')
        self.live_inputs[phase]=binding;self.attempts.append(phase)
        if phase=='fault':self.effect_attempted=True
        self.active_phase=phase
        try:result=binding.execute(once,owner=owner,generation=generation)
        finally:self.active_phase=None
        receipt=dict(call_id=f'{len(self.calls)+1:032x}',phase=phase,module_path=PATH,original_calls=1,
            owner=self.owner_ticket[0],generation=generation,selection_applied=phase=='fault',
            geometry={'selected_s2':256 if phase=='fault' else 512},logical_input_binding=binding.report(),simulation=True)
        self.calls.append(receipt)
        if phase=='fault' and self.fail_after_fault:raise RuntimeError('CPU_NATIVE_FAILED_AFTER_ORIGINAL_RETURN')
        if phase=='recovery_off' and self.fail_after_off:raise RuntimeError('CPU_OFF_REPLY_UNCERTAIN_AFTER_RETURN')
        return result,receipt


class DeviceOperator:
    @staticmethod
    def reshape_and_cache(**kwargs):return None


class Attention(torch.nn.Module):
    def __init__(self,fixture,api,index,packed):
        super().__init__();self.fixture=fixture;self.api=api;self.index=index;self.layer_name=f'actual.{index}'
        self.impl=NS(key_cache=packed[0],value_cache=packed[1],num_kv_heads=8,num_heads=16,head_size=128,
                     sliding_window=None,sinks=None)
        self.kv_cache=[packed]
    def forward(self,value):
        meta=self.fixture.context.attn_metadata[self.layer_name]
        DeviceOperator.reshape_and_cache(key_cache=self.impl.key_cache,value_cache=self.impl.value_cache,
                                        slot_mapping=meta.slot_mapping)
        return self.api.npu_fused_infer_attention_score(query=value,key=self.impl.key_cache.view(80,16,1024),
            value=self.impl.value_cache.view(80,16,1024),atten_mask=self.fixture.mask,block_table=meta.block_tables,
            input_layout='TND',block_size=16,sparse_mode=3,actual_seq_lengths=[1],actual_seq_lengths_kv=[1025],
            num_heads=16,num_key_value_heads=8,scale=128**-.5)[0]


class Logits(torch.nn.Module):
    # A real consumer can hide the operator difference. C1 must not invent a
    # requirement that the final logits also differ.
    def forward(self,hidden_states):return hidden_states[:, :8].float().clamp(max=10)*2


class Model(torch.nn.Module):
    def __init__(self,fixture,api,packed):
        super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList()
        self.logits=Logits()
        for index in range(28):
            layer=torch.nn.Module();layer.self_attn=torch.nn.Module()
            layer.self_attn.attn=Attention(fixture,api,index,packed);self.model.layers.append(layer)
    def forward(self,input_ids,positions):
        value=input_ids.sum().to(torch.float16).reshape(1,1,1).expand(1,16,128).contiguous()
        for layer in self.model.layers:value=layer.self_attn.attn(value)
        return value.reshape(1,2048)[:,:1024].contiguous()


class PublicS2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        self.enterContext(patch.object(identity,'_current_device',return_value=0))
        self.temp=self.enterContext(tempfile.TemporaryDirectory());self.worker=_extended_worker(rank=0);self.worker.load_model()
        self.fixture=Fixture();packed=self.fixture.module.kv_cache[0];self.api_calls=0;self.driver=None
        def original(**kwargs):
            self.api_calls+=1
            delta=2 if self.driver is not None and self.driver.active_phase=='fault' and not getattr(self.driver,'suppress_effect',False) else 1
            return (kwargs['query']+delta,torch.empty(0,dtype=torch.float32))
        self.original=original;self.api=NS(npu_fused_infer_attention_score=original)
        self.model=Model(self.fixture,self.api,packed).eval();self.worker.model_runner.model=self.model
        self.runner=self.worker.model_runner;self.fixture.worker=self.worker
        self.fixture.module=self.model.get_submodule(PATH)
        self.refresh('warm',False)
        with torch.inference_mode():hidden=self.model(torch.tensor([1]),torch.tensor([1024]));logits=self.model.logits(hidden)
        args=self.fixture.arguments('normal');args['forward_context'].model_instance=self.model
        probe=LogicalPagedInput(**args)
        normal=evidence.VerifiedNormalEvidence(evidence._MINT,dict(reference_input=probe.before['signature'],
            model_outputs=dict(hidden=complete(hidden),logits=complete(logits)),
            immediate_outputs=[{'sha256':complete(args['actual']['query']+1)['sha256']},
                               {'sha256':complete(torch.empty(0))['sha256']}]),simulation=True)
        probe.release(owner=args['owner'],generation=args['generation'])
        self.driver=CpuEffects(normal,Path(self.temp)/'native')
        self.providers=[]
        def resolve(mechanism,version,worker,model,actual):
            value=Paged33S2ServingProvider(worker,model,actual);self.providers.append(value);return value
        self.enterContext(patch.object(runtime,'_resolve_provider',side_effect=resolve))
        self.enterContext(patch.object(runtime,'_provider_capabilities',return_value=[
            dict(mechanism='33.paged_decode',implementation_version='1')]))
        context=dict(api_module=self.api,device_operator=DeviceOperator,
            candidates=[(f'model.layers.{index}.self_attn.attn',layer.self_attn.attn) for index,layer in enumerate(self.model.model.layers)],
            processor=self.model.logits,context=lambda:self.fixture.context,
            snapshot=lambda request_id,actual,context:dict(request_count=1,counts=[1],token_row_count=1,
                row_token_positions=[1024],request_ids=[request_id]))
        self.binding=install_attention_serving_binding(self.worker,'33.paged_decode',prompt_token_ids=[1],
            provenance={'cpu_only':True},native=self.driver,_simulation=True,_test_context=context)
        self.client=WorkerFaultClient(HostAdapter(ExistingExecutor([self.worker])),FaultConfig(33,'paged_decode'))
        self.addCleanup(self.dispose)
        for action in ('prepare','register','check','enable'):self.ok(getattr(self.client,action)())

    def refresh(self,request,relocate):
        args=self.fixture.arguments(request,list(reversed(range(65))) if relocate else None)
        context=args['forward_context'];context.model_instance=self.model
        for layer in self.model.model.layers:
            module=layer.self_attn.attn;context.attn_metadata[module.layer_name]=args['metadata'];context.no_compile_layers[module.layer_name]=module
        self.fixture.context=context

    def ok(self,reply):
        response=reply.get('response',reply);state=response.get('provider_state',response.get('native',{}))
        self.assertEqual(reply['status'],'ACTION_COMPLETED',repr(dict(error=response.get('error'),
            reason=response.get('reason'),serving=state.get('serving',{}).get('failures'),
            failure=state.get('failure'),provider_error=state.get('serving',{}).get('error'))))
        return reply['response']['provider_state']

    def request(self,index,*,relocate=False,collect=True):
        request=f'actual-{index}';self.refresh(request,relocate)
        self.ok(self.client.advance(request))
        with torch.inference_mode():
            for prefill in (True,False):
                self.runner.with_prefill=prefill
                hidden=self.model(torch.tensor([1]),torch.tensor([1024]));self.model.logits(hidden)
        before=self.api_calls;reply=self.client.query()
        self.assertEqual(reply['status'],'QUERY_OK');self.assertEqual(self.api_calls,before)
        return self.ok(self.client.collect_replay()) if collect else None

    def dispose(self):
        self.driver.fail_stop=False
        for provider in self.providers:
            session=provider.session
            if session and session.handles:
                # Test cleanup only; assertions inspect retained production
                # state before this fixture discards its simulated process.
                session.needs_off=False;self.driver.off_confirmed=True;session.close()
        self.assertIsNone(attention_execution._ACTIVE)

    def test_five_real_requests_preserve_full_denominator_and_public_cleanup(self):
        start=self.api_calls
        for index in range(5):state=self.request(index,relocate=index in (1,3))
        self.assertEqual(self.api_calls-start,5*56)
        self.assertTrue(state['evidence']['functional_protocol_complete']);self.assertFalse(state['C1'] or state['C2'] or state['C3'])
        self.assertTrue(state['evidence']['effect']['first_attention_output_changed'])
        self.assertFalse(state['evidence']['effect']['full_logits_changed'])
        self.assertEqual(state['evidence']['selection']['other_decode_targets'],27)
        self.assertEqual(len(state['evidence']['requests']),5)
        self.assertEqual([len(row['payloads']) for row in state['evidence']['requests']],[87]*5)
        self.assertTrue(self.driver.normal_gate);self.assertTrue(self.driver.off_confirmed)
        self.assertTrue(all(call['input_contract_matches'] for row in state['evidence']['requests'] for call in row['forwards'][1]['calls']))
        final=self.ok(self.client.finish());self.assertEqual(final['leases'],0)
        self.ok(self.client.disable());self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        removed=remove_attention_serving_binding(self.worker,'33.paged_decode')
        self.assertIs(self.api.npu_fused_infer_attention_score,self.original)
        self.check_original_protocol_reader(final,removed)

    def check_original_protocol_reader(self,final,removed):
        # The reader sees the actual Host/Runtime-produced JSON lifecycle,
        # including separate public/private owners and query-time row counts.
        # Its CPU-only option exists only on this helper; CLI never sets it.
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/c2c3_closure_20261117'))
        import review_arch22_s2_single as review
        run=Path(self.temp)/'review';run.mkdir();host=self.client.snapshot()
        requests=[dict(phase=row['phase'],mapping={'internal_request_id':row['request_id']},worker=row)
                  for row in final['evidence']['requests']]
        data=dict(setup={'worker':host['worker']},requests=requests,final_cleanup=final,binding_removal=removed,
            warm_healthy={'outputs':[{'tokens':[1]}]},post_unregister_health=dict(mapping={'internal_request_id':'new-health'},
                response={'outputs':[{'tokens':[1]}]},scope='TOKENS_TEXT_ONLY_NO_POST_UNREGISTER_HIDDEN_OR_LOGITS_OBSERVER'))
        def publish(value):
            path=run/'host.json';path.write_text(json.dumps(value))
            data['host_receipt']=dict(path=path.name,sha256=review.E.sha(path))
            refs=[]
            for index,row in enumerate(value['history']):
                path=run/f'rpc-{index}.json';path.write_text(json.dumps(row))
                refs.append(dict(path=path.name,sha256=review.E.sha(path),action=row['action']))
            data['worker_core']=dict(receipts=refs,ownership_retained=False)
        publish(host)
        context=review.protocol(run,data,allow_cpu_contract=True)
        self.assertTrue(context['simulation']);self.assertNotEqual(context['host_owner'],context['provider_owner'])
        with self.assertRaisesRegex(ValueError,'CPU_CONTRACT_IS_NOT_NATIVE'):review.protocol(run,data)
        for change in ('owner','query_advanced','retained_input','registered_early'):
            bad=deepcopy(host);reply=bad['history'][6]['response'];state=reply['provider_state']
            if change=='owner':state['owner']=bad['owner']
            elif change=='query_advanced':state['serving']['pending_request']=None
            elif change=='retained_input':state['native_state']['retained_input_objects']=0
            else:bad['history'][1]['response']['provider_state']['registered']=True
            # Keep original response aliases consistent so these reach the
            # lifecycle proof, instead of merely rejecting JSON-copy changes.
            for row in bad['history']:
                response=row['response']
                if 'provider_state' in response:
                    response['native']=deepcopy(response['provider_state'])
                    if row['action']!='query':response['provider']=deepcopy(response['provider_state'])
            publish(bad)
            with self.subTest(change=change),self.assertRaises(ValueError):review.protocol(run,data,allow_cpu_contract=True)

    def test_uncertain_fault_cannot_release_until_new_actual_off_and_never_qualifies(self):
        for index in range(3):self.request(index)
        self.driver.fail_after_fault=True
        with self.assertRaises(RuntimeError):self.request(3)
        self.assertTrue(self.driver.effect_attempted);self.assertTrue(self.providers[-1].session.needs_off)
        self.assertEqual(self.client.unregister()['status'],'PROVIDER_ACTION_FAILED')
        self.assertTrue(self.binding.leases);self.assertTrue(self.providers[-1].session.handles)
        self.driver.fail_after_fault=False
        self.ok(self.client.recover())
        state=self.request(4)
        self.assertFalse(state['evidence']['functional_protocol_complete']);self.assertFalse(state['C1'])
        self.assertTrue(state['serving']['failures']);self.assertTrue(self.driver.off_confirmed)
        self.driver.fail_stop=True
        self.assertEqual(self.client.finish()['status'],'PROVIDER_ACTION_FAILED')
        self.assertTrue(self.binding.leases);self.assertTrue(self.providers[-1].session.handles)
        self.driver.fail_stop=False
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        self.assertTrue(self.driver.cleanup_error);self.assertTrue(all(item.closed for item in self.driver.live_inputs.values()))

    def test_selection_without_operator_effect_preserves_plan_and_requires_actual_off(self):
        for index in range(3):self.request(index)
        self.driver.suppress_effect=True
        with self.assertRaises(AssertionError):self.request(3)
        provider=self.providers[-1]
        self.assertTrue(self.driver.calls[-1]['selection_applied']);self.assertTrue(provider.session.needs_off)
        self.assertFalse(provider.effect['first_attention_output_changed'])
        self.assertEqual(self.client.unregister()['status'],'PROVIDER_ACTION_FAILED')
        self.ok(self.client.recover());state=self.request(4)
        self.assertFalse(state['C1']);self.assertFalse(state['evidence']['functional_protocol_complete'])
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')

    def test_foreign_thread_and_wrapper_cannot_release_public_owner(self):
        self.request(0);session=self.providers[-1].session;errors=[]
        def foreign():
            try:self.driver.release_inputs(session)
            except BaseException as error:errors.append(repr(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertTrue(self.binding.leases)
        self.assertFalse(self.driver.live_inputs['reference'].closed)
        owned=self.api.npu_fused_infer_attention_score
        self.api.npu_fused_infer_attention_score=lambda **kw:owned(**kw)
        self.assertEqual(self.client.unregister()['status'],'PROVIDER_ACTION_FAILED')
        self.assertTrue(self.binding.leases);self.assertTrue(session.handles)
        self.api.npu_fused_infer_attention_score=owned
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')

    def test_uncertain_off_retains_old_inputs_and_allows_only_new_off_after_explicit_recover(self):
        for index in range(4):self.request(index)
        self.driver.fail_after_off=True
        with self.assertRaises(RuntimeError):self.request(4)
        old=self.driver.live_inputs['recovery_off'];self.assertFalse(old.closed)
        self.assertEqual(self.client.unregister()['status'],'PROVIDER_ACTION_FAILED')
        self.driver.fail_after_off=False;self.ok(self.client.recover())
        self.assertIs(self.driver.failed_off_inputs[0],old);self.assertFalse(old.closed)
        result=self.request(5);self.assertFalse(result['C1']);self.assertTrue(self.driver.off_confirmed)
        self.assertEqual(self.driver.attempts.count('fault'),1)
        self.ok(self.client.unregister());self.assertTrue(old.closed)
        self.assertEqual(self.driver.state()['retained_input_objects'],0)
        self.assertEqual(self.client.detach()['status'],'DETACHED')


if __name__=='__main__':unittest.main()
