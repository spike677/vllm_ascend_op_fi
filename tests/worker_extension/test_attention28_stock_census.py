"""Actual CPU calls and explicit fake Native/geometry seams; no NPU authority."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import test_core_attention_serving as core
from demoncat.runtime.vllm_ascend.worker_extension import attention28_stock_census as census
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding


def captured():
    result=dict(positional_count=3,original_callable=dict(type='torch._ops.OpOverloadPacket',object_id=17),
        public_default_schema='explicit CPU schema fixture; never an actual NPU schema',
        omitted_argument_defaults=dict(pre_tokens=2147483647,next_tokens=2147483647,inner_precise=0,softmax_lse_flag=False),
        live_metadata=dict(object_id=19,num_prefills=1,num_decodes=0,num_actual_tokens=2048,causal=True,
            seq_lens_list=[2048],actual_seq_lengths_q=[2048],mask_is_actual_metadata_object=True),inputs={})
    for name,shape,dtype in (('query',[2048,16,128],'torch.float16'),('key',[2048,8,128],'torch.float16'),
            ('value',[2048,8,128],'torch.float16'),('atten_mask',[2048,2048],'torch.int8')):
        width=1 if name == 'atten_mask' else 2;stride=[];size=1
        for n in reversed(shape):stride.insert(0,size);size*=n
        result['inputs'][name]=dict(kind='tensor',snapshot=dict(metadata=dict(shape=shape,stride=stride,
            storage_offset=0,storage_identity=111,object_id=222,storage_nbytes=size*width,dtype=dtype,device='cpu'),
            version=0,bytes=size*width,sha256='0'*64,finite=True,path=None))
    for name,value in dict(block_table=None,input_layout='TND',block_size=128,num_heads=16,num_key_value_heads=8,
            scale=128**-.5,sparse_mode=3,actual_seq_lengths=[2048],actual_seq_lengths_kv=[2048]).items():
        result['inputs'][name]=dict(kind='scalar',value=value)
    result['supplied_argument_names']=sorted(result['inputs'])
    return result


def scheduler(request='r'):
    return dict(status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',internal_request_id=request,
        num_requests=1,num_computed_tokens=0,num_prompt_tokens=2048,num_scheduled_tokens=2048,block_size=16)


def call_row(request='r'):
    return dict(actual_operator='torch_npu.npu_fused_infer_attention_score',request_id=request,scheduler=scheduler(request),
        module_path='actual.attention',module_object_id=20,occurrence=0,ordinal=0,target_key='a'*64,original_calls=0,
        native_census_scope=dict(owner='explicit-cpu-ticket',generation=1,phase='observe'))


def driver_fixture(folder):
    driver=census.StockTndPrefillCensus.__new__(census.StockTndPrefillCensus)
    driver.simulation=True;driver.mechanism=census.MECHANISMS[1];driver.identity=census.thread_identity()
    driver.scope,state=core.NativeScopeRecoveryTests().fixture()
    driver.scope.observer.synchronize=lambda:None
    driver.environment={};driver.run=folder;driver.folder=folder/'calls';driver.folder.mkdir()
    driver.calls=[];driver.failures=[];driver.refs=None;driver.pending_drain=False;driver.first_error=None
    driver.fingerprint=dict(simulation=True,observer_only=True)
    state.update(syncs=0,fail_sync=False,begins=0)
    original_begin=driver.scope.observer.lib.dcat_attention_observer_begin_v1.fn
    def begin(token):state['begins']+=1;return original_begin(token)
    driver.scope.observer.lib.dcat_attention_observer_begin_v1.fn=begin
    def sync():
        state['syncs']+=1
        if state['fail_sync']:raise RuntimeError('cpu drain failed')
    driver._synchronize=sync
    return driver,state


class TndInputContracts(unittest.TestCase):
    def test_capture_records_actual_cpu_storage_view_and_actual_schema_defaults_without_mutating(self):
        base=torch.arange(32,dtype=torch.float16).reshape(4,8);view=base[1:3]
        mask=torch.ones(2,2,dtype=torch.int8).triu(1)
        metadata=NS(num_prefills=1,num_decodes=0,num_actual_tokens=2,causal=True,
            seq_lens_list=[2],actual_seq_lengths_q=[2],attn_mask=mask)
        original=NS(default=NS(_schema=torch.ops.aten.add.Tensor._schema))
        before=base.clone()
        record=census.capture_inputs(original,(view,),dict(query=view,atten_mask=mask),metadata)
        snap=record['inputs']['query']['snapshot']
        self.assertEqual(snap['metadata']['storage_offset'],8)
        self.assertEqual(snap['metadata']['stride'],[8,1])
        self.assertEqual(snap['metadata']['object_id'],id(view))
        self.assertEqual(record['omitted_argument_defaults']['alpha'],1)
        self.assertFalse(record['actual_native_attributes_observed'])
        self.assertTrue(torch.equal(base,before))
        view[0,0]+=1
        after=census.capture_inputs(original,(view,),dict(query=view,atten_mask=mask),metadata)
        self.assertNotEqual(snap['sha256'],after['inputs']['query']['snapshot']['sha256'])

    def test_exact_source71_semantics_keep_omitted_defaults_and_native_gap_separate(self):
        value=census.validate_tnd_prefill(captured(),call_row(),simulation=True)
        self.assertEqual(value['api_block_size'],128)
        self.assertEqual(value['scheduler_cache_block_size'],16)
        self.assertFalse(value['paged'])
        self.assertIn('UNOBSERVED',value['native_key'])
        data=captured();data['omitted_argument_defaults']['inner_precise']=1
        census.validate_tnd_prefill(data,call_row(),simulation=True)
        self.assertEqual(data['omitted_argument_defaults']['inner_precise'],1)

    def test_wrong_layout_decode_page16_or_optional_cannot_be_relabelled_tnd_prefill(self):
        for key,value in (('input_layout','BNSD'),('block_size',16),('block_table',[1]),
                ('actual_seq_lengths',[1]),('actual_seq_lengths_kv',[1025])):
            data=captured();data['inputs'][key]=dict(kind='scalar',value=value)
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'SEMANTICS_DIFFER'):
                census.validate_tnd_prefill(data,call_row(),simulation=True)
        data=captured();data['inputs']['quant_scale1']=dict(kind='scalar',value=1)
        with self.assertRaisesRegex(RuntimeError,'ARGUMENT_SET_DIFFERS'):
            census.validate_tnd_prefill(data,call_row(),simulation=True)

    def test_missing_schema_mask_identity_storage_extent_or_current_request_reject(self):
        cases=(lambda d:d.__setitem__('public_default_schema',None),
            lambda d:d['live_metadata'].__setitem__('mask_is_actual_metadata_object',False),
            lambda d:d['inputs']['query']['snapshot']['metadata'].__setitem__('storage_offset',1),
            lambda d:d['omitted_argument_defaults'].pop('inner_precise'))
        for edit in cases:
            data=captured();edit(data)
            with self.subTest(edit=edit),self.assertRaises(RuntimeError):
                census.validate_tnd_prefill(data,call_row(),simulation=True)
        row=call_row();row['scheduler']['internal_request_id']='foreign'
        with self.assertRaisesRegex(RuntimeError,'REQUEST_UNBOUND'):
            census.validate_tnd_prefill(captured(),row,simulation=True)


class StockCensusOwnershipContracts(unittest.TestCase):
    def setUp(self):
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver,self.native=driver_fixture(self.folder)
        self.enterContext(patch.object(census,'capture_inputs',side_effect=lambda *a:deepcopy(captured())))
        self.enterContext(patch.object(census,'observe_scope',return_value=dict(status='explicit CPU raw seam',kernel_argument_abi='NOT_DECODED')))
        self.mask=torch.ones(3,3,dtype=torch.int8).triu(1)
        self.row=call_row();self.output=object()
        self.addCleanup(self.clean)

    def clean(self):
        self.native['mode']='ok';self.native['fail_sync']=False
        if self.driver.scope.owned:self.native['token']=self.driver.scope.token.encode()
        self.driver.stop()

    def once(self):self.row['original_calls']+=1;return self.output

    def invoke(self, once=None):
        return self.driver.observe(self.once,(),{'atten_mask':self.mask},self.row,None,once or self.once)

    def test_original_result_and_arguments_are_unchanged_and_query_never_syncs(self):
        output,record=self.invoke()
        self.assertIs(output,self.output);self.assertEqual(self.row['original_calls'],1)
        self.assertEqual((self.native['begins'],self.native['ends']),(1,1))
        before=(self.native['syncs'],self.native['ends'],len(self.driver.calls))
        for _ in range(3):self.driver.state()
        self.assertEqual(before,(self.native['syncs'],self.native['ends'],len(self.driver.calls)))
        self.assertFalse(record['native_fault_qualified'] or record['fault_authorized'])
        self.assertIsNone(self.driver.refs)

    def test_original_failure_and_stop_failure_preserve_first_error_and_refs(self):
        self.native['mode']='before'
        def fail():self.row['original_calls']+=1;raise ValueError('original CPU failure')
        with self.assertRaisesRegex(ValueError,'original CPU failure'):self.invoke(fail)
        state=self.driver.state()
        self.assertTrue(state['scope_owned'] and state['pending_drains'] and state['pending_references'])
        self.assertIn('original CPU failure',state['first_error'])
        self.native['fail_sync']=True
        with self.assertRaisesRegex(RuntimeError,'drain failed'):self.driver.stop()
        self.assertIsNotNone(self.driver.refs)
        self.native.update(mode='ok',fail_sync=False);self.driver.stop()
        self.assertEqual(self.row['original_calls'],1)
        self.assertIsNone(self.driver.refs)
        self.assertIn('original CPU failure',self.driver.state()['first_error'])

    def test_after_end_failure_retires_real_scope_but_retains_refs_until_cleanup_retry(self):
        self.native['mode']='after'
        with self.assertRaisesRegex(RuntimeError,'after end'):self.invoke()
        self.assertFalse(self.driver.state()['scope_owned'])
        self.assertEqual(self.driver.state()['pending_references'],1)
        self.native['mode']='ok';self.driver.stop()
        self.assertEqual(self.native['ends'],1)
        self.assertEqual(self.driver.state()['pending_references'],0)

    def test_foreign_thread_or_token_cannot_release_owned_resources(self):
        self.native['mode']='before'
        with self.assertRaises(RuntimeError):self.invoke()
        self.native['mode']='ok';errors=[]
        def foreign():
            try:self.driver.stop()
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join()
        self.assertIn('THREAD_DOMAIN_CHANGED',errors[0])
        self.native['token']=b'foreign'
        with self.assertRaisesRegex(RuntimeError,'TOKEN_OWNER_CHANGED'):self.driver.stop()
        self.assertIsNotNone(self.driver.refs)

    def test_changed_native_tid_even_on_same_python_thread_cannot_release(self):
        tid=self.driver.identity['native_tid']
        with patch.object(census.threading,'get_native_id',return_value=tid+1):
            with self.assertRaisesRegex(RuntimeError,'THREAD_DOMAIN_CHANGED'):self.driver.stop()
        self.assertEqual(self.native['ends'],0)

    def test_failed_begin_does_not_call_original_and_keeps_partial_owner_until_cleanup(self):
        begin=self.driver.scope.observer.lib.dcat_attention_observer_begin_v1.fn
        self.driver.scope.observer.lib.dcat_attention_observer_begin_v1.fn=lambda token:(begin(token),1)[1]
        with self.assertRaisesRegex(RuntimeError,'BEGIN_FAILED'):self.invoke()
        self.assertEqual(self.row['original_calls'],0)
        self.assertEqual(self.driver.state()['pending_references'],1)
        self.driver.stop()
        self.assertEqual(self.driver.state()['pending_references'],0)
        self.assertIn('BEGIN_FAILED',self.driver.state()['first_error'])

    def test_unknown_inputs_stay_in_denominator_without_scope_or_second_call(self):
        with patch.object(census,'validate_tnd_prefill',side_effect=RuntimeError('unadapted actual input')):
            output,record=self.invoke()
        self.assertIs(output,self.output);self.assertEqual(self.row['original_calls'],1)
        self.assertEqual(self.native['begins'],0)
        self.assertEqual(record['status'],'UNOBSERVED_OR_INCOMPLETE')
        self.assertEqual(len(self.driver.calls),1)

    def test_real_path_cannot_trust_a_packet_type_string_over_the_actual_callable(self):
        self.driver.simulation=False
        output,record=self.invoke()
        self.assertIs(output,self.output)
        self.assertEqual(self.native['begins'],0)
        self.assertEqual(record['status'],'UNOBSERVED_OR_INCOMPLETE')
        self.assertIn('ACTUAL_PACKET_OBJECT_REQUIRED',record['reasons'][0])


class ManyAttention(torch.nn.Module):
    def __init__(self, api):
        super().__init__();self.layers=torch.nn.ModuleList([core.Attention(api,str(i)) for i in range(28)])
        self.logits=core.Logits();self.calls=0
    def forward(self,input_ids,positions):
        self.calls+=1;value=input_ids.float().sum()
        for layer in self.layers:value=layer(value)
        return value


class PublicStockCensusContracts(unittest.TestCase):
    def setUp(self):
        self.f=core.CoreAttentionTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver,self.native=driver_fixture(self.folder)
        self.f.model=ManyAttention(self.f.api).eval();self.f.runner.model=self.f.model
        candidates=[('layers.'+str(i),module) for i,module in enumerate(self.f.model.layers)]
        context=NS(model_instance=self.f.model,no_compile_layers={m.layer_name:m for _,m in candidates},
            attn_metadata={m.layer_name:NS(attn_mask=m.mask) for _,m in candidates})
        setup=dict(api_module=self.f.api,candidates=candidates,processor=self.f.model.logits,context=lambda:context,
            snapshot=lambda request,actual,ctx:scheduler(request))
        self.f.binding=install_attention_serving_binding(self.f.worker,self.driver.mechanism,prompt_token_ids=[1,2,3],
            provenance={'cpu_only':True},native=self.driver,_simulation=True,_test_context=setup)
        self.client=core.WorkerFaultClient(core.HostAdapter(self.f.executor),core.FaultConfig(28,'fia.denominator_replay'))
        self.enterContext(patch.object(census,'capture_inputs',side_effect=lambda *a:deepcopy(captured())))
        self.enterContext(patch.object(census,'observe_scope',return_value=dict(status='explicit CPU raw seam',kernel_argument_abi='NOT_DECODED')))
        self.addCleanup(self.restore)

    def restore(self):
        self.native['mode']='ok';self.native['fail_sync']=False

    def test_real_public_two_requests_capture_all_56_calls_and_reject_baseline(self):
        provider=self.f.start(self.client)
        self.f.request(self.client,0);state=self.f.request(self.client,1)
        calls=[call for request in state['evidence']['requests'] for f in request['forwards'] for call in f['calls']]
        self.assertEqual(len(calls),56);self.assertEqual(len(self.f.api_calls),56)
        self.assertEqual(self.native['begins'],56)
        self.assertEqual(len({call['native_census']['call_id'] for call in calls}),56)
        self.assertEqual({call['native_census']['execution_scope']['owner'] for call in calls},{state['owner']})
        self.assertEqual({call['native_census']['execution_scope']['model_object_id'] for call in calls},{id(self.f.model)})
        self.assertTrue(all(call['original_calls'] == 1 and call['original_returned'] and call['inputs_unchanged'] for call in calls))
        self.assertEqual(state['evidence']['stock_launch_census']['observed_call_scopes'],56)
        before=self.native['syncs'];self.assertEqual(self.client.query()['status'],'QUERY_OK')
        self.assertEqual(before,self.native['syncs'])
        reply=self.client.advance('not-a-baseline')
        self.assertEqual(reply['status'],'PROVIDER_ACTION_FAILED')
        self.assertEqual(reply['response']['provider_code'],'ATTENTION_STOCK_CENSUS_OBSERVE_ONLY')
        self.f.cleanup(self.client)
        with torch.inference_mode():self.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
        self.assertEqual(len(self.f.api_calls),84)

    def test_public_cleanup_keeps_ticket_for_before_end_failure_then_recovers(self):
        provider=self.f.start(self.client);self.f.ok(self.client.advance('failed-request'))
        self.native['mode']='before'
        with torch.inference_mode(),self.assertRaisesRegex(RuntimeError,'before end'):
            self.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
        reply=self.client.unregister()
        self.assertEqual(reply['status'],'PROVIDER_ACTION_FAILED')
        self.assertTrue(self.f.binding.leases)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.assertIsNotNone(self.driver.refs)
        self.native['mode']='ok';self.f.cleanup(self.client)
        self.assertIsNone(self.driver.refs)
        self.assertFalse(self.client.snapshot()['ownership_retained'])
        with torch.inference_mode():self.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
        self.assertEqual(len(self.f.api_calls),29)
        self.assertEqual(self.native['begins'],1)

    def test_probe_protocol_finishes_observer_then_unregistered_same_worker_health(self):
        import natural_attention_worker_core_probe as probe
        owner=self
        class Llm:
            def __init__(self):self.llm_engine=NS(has_unfinished_requests=lambda:False);self.index=0
            def reset_prefix_cache(self,**kwargs):return True
            def enqueue(self,*args,**kwargs):
                self.index+=1;self.request='actual-cpu-'+str(self.index);return [self.request]
            def wait_for_completion(self,**kwargs):
                with torch.inference_mode():
                    hidden=owner.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]));owner.f.model.logits(hidden)
                return [NS(request_id=self.request,outputs=[NS(token_ids=[7])])]
        with patch.object(probe,'read_enqueued_identity',side_effect=lambda llm,ids,**kw:
                dict(internal_request_id=ids[0],external_request_id=ids[0])):
            report=dict(status='STARTED',healthy_before=[7]);out=self.folder/'probe';out.mkdir()
            probe.drive(Llm(),self.client,prompt={'prompt_token_ids':[1,2,3]},params=object(),out=out,
                        report=report,expected_native_observer=True)
        self.assertEqual(report['status'],probe.SUCCESS,report.get('error'))
        self.assertFalse(report['worker_core']['ownership_retained'])
        self.assertEqual(self.f.model.calls,3)
        self.assertEqual(self.native['begins'],56)
        self.assertTrue((out/'host_receipt.json').is_file())
        final=report['final_cleanup']
        self.assertEqual((final['owned_hooks'],final['owned_api_wrappers'],final['leases']),(0,0,0))
        self.assertTrue(final['native_state']['observer_only'])
        core.remove_attention_serving_binding(self.f.worker,self.driver.mechanism)


if __name__ == '__main__':unittest.main()
