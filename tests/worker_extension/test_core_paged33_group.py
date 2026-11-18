"""Real CPU module/API/Host/Runtime contracts. No device or raw-NPU proof."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime,attention_execution,paged33_all_normal as normal
from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import AttentionExecutionAdapter
from demoncat.runtime.vllm_ascend.worker_extension.call_binding import ForwardBinding
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import (
    AttentionDeployment,install_attention_serving_binding,remove_attention_serving_binding)
from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_native import GroupEffectsNativeCall
from demoncat.runtime.vllm_ascend.worker_extension.paged33_group_evidence import GroupNormalEvidence,_MINT,VARIANT,PHASES,payload_signatures,cost_gate
from demoncat.runtime.vllm_ascend.worker_extension.paged33_normal_targets import NormalTargetScope
from demoncat.runtime.vllm_ascend.worker_extension.providers.paged33_group import Paged33GroupServingProvider
from test_core_paged33_s2 import Model,DeviceOperator
from test_paged33_logical_binding import Fixture,PATH
from test_extension import _extended_worker
from test_core_host import ExistingExecutor


def tensor_metadata(value,path):
    cpu=value.detach().contiguous().cpu();raw=cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(path=str(path),shape=list(cpu.shape),dtype=str(cpu.dtype),bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),finite=bool(torch.isfinite(cpu).all()),cpu_metadata_only=True)


def cache_metadata(cache,pages,length,path):
    value=torch.cat([cache[page] for page in pages[0]])[:length]
    return tensor_metadata(value,path)


class CpuNative(GroupEffectsNativeCall):
    simulation=True
    def __init__(self,folder,identity,normal_evidence=None):
        self.folder=folder;folder.mkdir();self.worker_identity=identity;self.normal_evidence=normal_evidence
        self.thread=threading.get_ident();self.calls=[];self.failures=[];self.controller=None;self.runtime_dir=folder
        self.budget=cost_gate(.01,1800);self.fail_stop=False;self.fail_after_fault=False;self.active_phase=None
        self.case_started=time.monotonic()
        self.fingerprint={'completion_sha256':'a'*64};self.suppress_effect=False
        self.resource_checks=[];self.fail_resource_stage=None;self.first_resource_error=None
    def check_resources(self,phase,index,stage):
        self._check();self.controller.check_owner()
        if self.fail_resource_stage==stage:
            self.first_resource_error='NO_F_CAPACITY_REFUSED:CPU_RESOURCE_SEAM'
            raise RuntimeError(self.first_resource_error)
        from demoncat.runtime.vllm_ascend.worker_extension.paged33_s2_evidence import digest
        row=dict(phase=phase,phase_index=index,stage=stage,simulation=True)
        result=dict(receipt=row,sha256=digest(row));self.resource_checks.append(result);return deepcopy(result)
    def _check(self):
        if threading.get_ident()!=self.thread:raise RuntimeError('CPU_FOREIGN_THREAD')
    def state(self):
        self._check();controller=self.controller
        return dict(native_loaded=True,scope_active=False,scope_owned=False,environment_owned=0,calls=len(self.calls),
            retained_input_objects=sum(not r['binding'].closed for r in getattr(getattr(controller,'adapter',None),'owned_records',[]) if 'binding' in r),
            failures=deepcopy(self.failures),first_error=None,simulation=True,effect_attempted=bool(controller and controller.native_attempted),
            off_confirmed=bool(controller and controller.last_off and not controller.off_pending))
    def stop(self):
        self._check()
        if self.fail_stop:raise RuntimeError('CPU_UNCERTAIN_DRAIN')
        return self.state()
    def execute(self,actual,boundary,phase,once,*,binding,scope,owner,generation):
        self._check();scope.check(binding.worker,binding.module_path,binding.module,owner,generation)
        admission,coverage_generation=self._admission(phase,binding,scope,owner,generation) if self.controller else ('0'*64,0)
        self.active_phase=phase
        try:result=binding.execute(once,owner=owner,generation=generation)
        finally:self.active_phase=None
        receipt=dict(call_id=f'{len(self.calls)+1:032x}',phase=phase,module_path=binding.module_path,original_calls=1,
            owner=self.owner_token,generation=generation,selection_applied=phase=='fault',simulation=True,
            admission_sha256=admission,coverage_generation=coverage_generation,
            geometry={'selected_s2':256 if phase=='fault' else 512},logical_input_binding=binding.report())
        self.calls.append(receipt)
        if phase=='fault' and self.fail_after_fault:raise RuntimeError('CPU_NATIVE_REPLY_UNCERTAIN_AFTER_RETURN')
        return result,receipt


class PublicGroupContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def setUp(self):
        self.enterContext(patch.object(identity,'_current_device',return_value=0))
        self.enterContext(patch.object(normal,'tensor_file',side_effect=tensor_metadata))
        self.enterContext(patch.object(normal,'logical_cache_file',side_effect=cache_metadata))
        self.enterContext(patch.object(normal,'read_scoped_launches',return_value={'simulation':True}))
        self.enterContext(patch.object(normal,'mapped_api_libraries',return_value={'simulation':True}))
        self.temp=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.worker=_extended_worker(rank=0);self.worker.load_model();self.fixture=Fixture();self.api_calls=0;self.driver=None
        def original(**kwargs):
            self.api_calls+=1;delta=2 if self.driver and self.driver.active_phase=='fault' and not self.driver.suppress_effect else 1
            return kwargs['query']+delta,torch.empty(0,dtype=torch.float32)
        self.original=original;self.api=NS(npu_fused_infer_attention_score=original)
        self.model=Model(self.fixture,self.api,self.fixture.module.kv_cache[0]).eval();self.worker.model_runner.model=self.model
        self.fixture.worker=self.worker;self.fixture.module=self.model.get_submodule(PATH)
        self.actual=self.worker.demoncat_worker_identity()['worker'];self.providers=[]
        self.rows=[(f'model.layers.{i}.self_attn.attn',layer.self_attn.attn) for i,layer in enumerate(self.model.model.layers)]
        self.context=dict(api_module=self.api,device_operator=DeviceOperator,candidates=self.rows,processor=self.model.logits,
            context=lambda:self.fixture.context,snapshot=lambda rid,actual,ctx:dict(request_count=1,counts=[1],token_row_count=1,
                row_token_positions=[1024],request_ids=[rid]))
        self.driver=CpuNative(self.temp/'warm-native',self.actual);self.driver.owner_token='a'*32
        warm=normal.AllNormalWorkerSession.__new__(normal.AllNormalWorkerSession)
        warm.folder=self.temp/'warm';warm.folder.mkdir();(warm.folder/'boundaries').mkdir();warm.private=True;warm.native_phase='off'
        warm.owner_token='a'*32
        deployment=AttentionDeployment(self.worker,self.worker.model_runner,self.model,'33.paged_decode',(1,),self.driver,'{}',
            os.getpid(),threading.get_ident(),28,True,self.context,{(warm.owner_token,1)})
        AttentionExecutionAdapter.__init__(warm,deployment,warm._guard)
        warm.targets=NormalTargetScope(self.worker,self.rows,owner=warm,generation=1);warm.logits_binding=ForwardBinding(self.model.logits)
        warm.records={};warm.payload_refs={};warm.original=None;warm.first_cleanup_error=None
        try:
            warm.install();warm.expect({'warm':dict(prompt=[1])});self.forward('warm');capture=warm.collect()
        finally:warm.close()
        targets={p:dict(reference_input=b['logical_input_binding']['evidence']['before']['signature'])
                 for p,b in capture['all_decode_boundaries'].items()}
        observations=GroupNormalEvidence(_MINT,dict(variant=VARIANT,targets=targets,payloads=payload_signatures(capture),
            source_pins={'cpu.py':'b'*64},execution_contract={'cpu_only':True},normal_max_phase_seconds=.01),simulation=True)
        self.driver=CpuNative(self.temp/'native',self.actual,observations)
        def resolve(mechanism,version,worker,model,actual):
            provider=Paged33GroupServingProvider(worker,model,actual);self.providers.append(provider);return provider
        self.enterContext(patch.object(runtime,'_resolve_provider',side_effect=resolve))
        self.enterContext(patch.object(runtime,'_provider_capabilities',return_value=[dict(mechanism='33.paged_decode',implementation_version='1')]))
        self.binding=install_attention_serving_binding(self.worker,'33.paged_decode',prompt_token_ids=[1],
            provenance={'cpu_only':True},native=self.driver,max_calls=28,_simulation=True,_test_context=self.context)
        self.client=WorkerFaultClient(HostAdapter(ExistingExecutor([self.worker])),FaultConfig(33,'paged_decode'))
        self.addCleanup(self.dispose)
        for action in ('prepare','register','check','enable'):self.ok(getattr(self.client,action)())

    def forward(self,rid,*,relocate=False):
        args=self.fixture.arguments(rid,list(reversed(range(65))) if relocate else None)
        context=args['forward_context'];context.model_instance=self.model
        for _,module in self.rows:
            context.attn_metadata[module.layer_name]=args['metadata'];context.no_compile_layers[module.layer_name]=module
        self.fixture.context=context
        with torch.inference_mode():
            for prefill in (True,False):
                self.worker.model_runner.with_prefill=prefill
                output=self.model(torch.tensor([1]),torch.tensor([1024]));self.model.logits(output)

    def ok(self,reply):
        self.assertEqual(reply['status'],'ACTION_COMPLETED',repr(reply.get('response',{})))
        return reply['response']['provider_state']

    def request(self,index,*,relocate=False):
        rid='request-'+str(index);self.ok(self.client.advance(rid));self.forward(rid,relocate=relocate)
        before=self.api_calls;queried=self.client.query();self.assertEqual(queried['status'],'QUERY_OK')
        self.assertEqual(before,self.api_calls)
        return self.ok(self.client.collect_replay())

    def dispose(self):
        self.driver.fail_stop=False
        for provider in self.providers:
            session=provider.session
            if session and session.handles:
                # Simulated process disposal only, after retention assertions.
                session.needs_off=False;session.close()
        self.assertIsNone(attention_execution._ACTIVE)

    def test_full_60_requests_real_public_protocol_and_live_c2_object_cpu_only(self):
        start=self.api_calls
        for index in range(60):state=self.request(index,relocate=index%2==1)
        self.assertEqual(self.api_calls-start,3360)
        self.assertEqual(len(self.driver.calls),1680)
        self.assertEqual(sum(r['selection_applied'] for r in self.driver.calls),28)
        self.assertFalse(state['C1'] or state['C2'] or state['C3'])
        final=self.ok(self.client.finish());c=self.providers[0].session.control
        report=c.session.report();self.assertEqual((report['hits'],report['observed_calls'],report['planned_targets']), (28,3136,28))
        self.assertTrue(report['contract_satisfied'],report['rejection_reasons']);self.assertIsNotNone(c.c2_proof)
        self.assertEqual(report['live_c2_proof']['issued'][0]['object_binding']['python_object_id'],id(c.c2_proof))
        self.assertEqual(len(final['evidence']['requests']),60);self.assertFalse(final['C2'] or final['C3'])
        self.assertEqual(final['serving']['resource_state']['owned_input_references'],0)
        self.assertEqual(len(self.driver.resource_checks),56)
        from test_arch22_group_c2_review import G,actual_cpu_ledger
        args=actual_cpu_ledger(c,Path(final['evidence']['coverage']['path']))
        checked=G.coverage_join(*args,allow_cpu_contract=True)
        self.assertEqual(checked['recomputed_hits'],28);self.assertFalse(checked['C2'])
        wrong=deepcopy(args[0]);wrong['coverage']['calls'][0]['receipt']['original_calls']=2
        with self.assertRaisesRegex(ValueError,'DECISION_RECEIPT_DIFFERS'):
            G.coverage_join(wrong,*args[1:],allow_cpu_contract=True)
        with self.assertRaisesRegex(ValueError,'REAL_NATIVE_EVIDENCE_REQUIRED'):G.coverage_join(*args)
        coverage=deepcopy(final['evidence']['coverage'])
        self.ok(self.client.disable());self.assertTrue(self.providers[0].binding_reserved)
        self.assertEqual(self.providers[0].coverage_artifact,coverage)
        with self.assertRaisesRegex(RuntimeError,'STILL_OWNED'):remove_attention_serving_binding(self.worker,'33.paged_decode')
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        removal=remove_attention_serving_binding(self.worker,'33.paged_decode')
        before=self.api_calls;self.forward('ordinary');self.assertEqual(self.api_calls-before,56)
        self.assertIs(self.api.npu_fused_infer_attention_score,self.original)
        host=self.client.snapshot();refs=[]
        def write(name,value):
            path=self.temp/name;path.write_text(json.dumps(value,allow_nan=False))
            return dict(path=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
        for index,row in enumerate(host['history']):refs.append(dict(write(str(index)+'.json',row),action=row['action']))
        data=dict(host_receipt=write('host.json',host),worker_core=dict(receipts=refs,ownership_retained=False),
            requests=[dict(phase=row['phase'],worker=row,mapping=dict(internal_request_id=row['request_id'])) for row in c.records],
            setup={'worker':self.actual},final_cleanup=final,binding_removal=removal,warm_healthy={'outputs':['CPU ordinary health']},
            post_unregister_health=dict(mapping=dict(internal_request_id='ordinary'),response={'outputs':['CPU ordinary health']},
                scope='TOKENS_TEXT_ONLY_NO_POST_UNREGISTER_HIDDEN_OR_LOGITS_OBSERVER'))
        checked=G.S.protocol(self.temp,data,group_phases=PHASES,allow_cpu_contract=True)
        self.assertTrue(checked['simulation']);self.assertEqual(len(host['history']),189)

    def test_uncertain_fault_never_retries_and_actual_off_health_releases_failed_owner(self):
        for index in range(3):self.request(index)
        self.driver.fail_after_fault=True;self.ok(self.client.advance('uncertain'))
        with self.assertRaisesRegex(RuntimeError,'CPU_NATIVE_REPLY_UNCERTAIN'):self.forward('uncertain')
        provider=self.providers[0];c=provider.session.control
        self.assertTrue(c.off_pending);self.assertIsNotNone(c.permit)
        self.assertNotEqual(self.client.unregister()['status'],'ACTION_COMPLETED');self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.driver.fail_after_fault=False;self.ok(self.client.recover());self.request('recovery');self.request('recovery-health')
        self.assertEqual(sum(r['selection_applied'] for r in self.driver.calls),1)
        self.ok(self.client.disable());self.assertIsNone(c.c2_proof);self.assertIsNotNone(c.first_error)
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')

    def test_failed_drain_keeps_hooks_references_and_ticket_until_retry(self):
        self.request(0);self.driver.fail_stop=True
        self.assertNotEqual(self.client.disable()['status'],'ACTION_COMPLETED')
        provider=self.providers[0];self.assertTrue(provider.session.handles and provider.session.owned_records)
        self.assertTrue(self.client.snapshot()['ownership_retained'])
        self.driver.fail_stop=False;self.ok(self.client.disable());self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'],'DETACHED')
        self.assertIn('CPU_UNCERTAIN_DRAIN',provider.session.first_cleanup_error)

    def test_no_effect_preserves_strategy_receipt_but_no_c2_and_fresh_off_required(self):
        for index in range(3):self.request(index)
        self.driver.suppress_effect=True;self.ok(self.client.advance('no-effect'));self.forward('no-effect')
        reply=self.client.collect_replay();self.assertNotEqual(reply['status'],'ACTION_COMPLETED')
        c=self.providers[0].session.control;self.assertTrue(c.off_pending);self.assertIsNone(c.c2_proof)
        self.assertTrue(any(r['selection_applied'] for r in self.driver.calls))
        self.ok(self.client.recover());self.request('off');self.request('health')
        self.ok(self.client.disable());self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')

    def test_explicit_budget_gate_and_current_input_hash_are_not_deleted(self):
        with self.assertRaisesRegex(RuntimeError,'EXCEEDS_FIXED'):cost_gate(30,1800)
        for index in range(3):self.request(index)
        # Alter the next actual target's Q via the original model input. The
        # gate must fail before a Native fault; denominator rows remain partial.
        self.ok(self.client.advance('changed'))
        self.fixture.logical.add_(1)
        with self.assertRaisesRegex(RuntimeError,'CURRENT_INPUT_DIFFERS'):self.forward('changed')
        self.assertFalse(any(r['selection_applied'] for r in self.driver.calls))
        self.assertTrue(self.client.snapshot()['ownership_retained'])

    def test_live_capacity_refuses_before_native_and_keeps_owner_for_real_off(self):
        for index in range(3):self.request(index)
        self.driver.fail_resource_stage='before_native';self.ok(self.client.advance('capacity'))
        with self.assertRaisesRegex(RuntimeError,'NO_F_CAPACITY_REFUSED'):self.forward('capacity')
        c=self.providers[0].session.control
        self.assertFalse(c.native_attempted or any(r['selection_applied'] for r in self.driver.calls))
        self.assertTrue(c.off_pending and self.client.snapshot()['ownership_retained'])
        self.assertFalse(c.permit['consumed']);self.assertIsNone(c.c2_proof)
        self.driver.fail_resource_stage=None;self.ok(self.client.recover());self.request('off');self.request('health')
        self.ok(self.client.disable());self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        self.assertIn('NO_F_CAPACITY_REFUSED',c.first_error)

    def test_oversized_capture_never_completes_and_retains_actual_public_owner_for_off(self):
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_group_execution as execution
        for index in range(3):self.request(index)
        self.ok(self.client.advance('bounded-capture'));self.forward('bounded-capture')
        provider=self.providers[0];session=provider.session;c=session.control
        path=session.folder/'runtime_result.json';original=execution._plain_stat
        def oversized(candidate):
            value=original(candidate)
            if candidate!=path:return value
            return NS(st_mode=value.st_mode,st_dev=value.st_dev,st_ino=value.st_ino,
                st_size=execution.MAX_CAPTURE_BYTES+1,st_mtime_ns=value.st_mtime_ns,st_ctime_ns=value.st_ctime_ns)
        with patch.object(execution,'_plain_stat',side_effect=oversized):
            reply=self.client.collect_replay()
        self.assertNotEqual(reply['status'],'ACTION_COMPLETED')
        self.assertIn('GROUP_CAPTURE_FILE_BOUND',c.first_error)
        self.assertTrue(path.is_file() and session.installed and session.handles and session.owned_records)
        self.assertTrue(c.off_pending and self.client.snapshot()['ownership_retained'])
        self.assertIsNone(c.c2_proof);self.assertEqual(len(c.records),3)
        before=path.read_bytes();self.assertEqual(sum(row['selection_applied'] for row in self.driver.calls),1)
        self.ok(self.client.recover());self.request('bounded-off');self.request('bounded-health')
        self.ok(self.client.disable());self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        self.assertEqual(path.read_bytes(),before);self.assertIsNone(c.c2_proof)

    def test_early_disable_after_controls_keeps_observers_for_real_off_and_health(self):
        for index in range(3):self.request(index)
        provider=self.providers[0];session=provider.session;c=session.control
        hooks=list(session.handles);records=list(session.owned_records)
        self.assertEqual(sum(row['selection_applied'] for row in self.driver.calls),0)
        reply=self.client.disable();self.assertNotEqual(reply['status'],'ACTION_COMPLETED')
        self.assertIn('GROUP_PROTOCOL_INCOMPLETE',c.first_error);first=c.first_error
        self.assertTrue(session.installed and c.off_pending and session.needs_off)
        self.assertEqual(session.handles,hooks);self.assertEqual(session.owned_records,records)
        self.assertTrue(all(not record['binding'].closed for record in records))
        self.assertTrue(self.client.snapshot()['ownership_retained']);self.assertIsNone(c.c2_proof)
        self.ok(self.client.recover());self.request('early-disable-off');self.request('early-disable-health')
        final=self.ok(self.client.disable());self.assertFalse(session.installed)
        self.assertEqual(final['serving']['resource_state']['owned_hooks'],0)
        self.assertEqual(final['serving']['resource_state']['owned_input_references'],0)
        self.assertIsNone(c.c2_proof);self.assertEqual(c.first_error,first)
        self.assertFalse(final['C2'] or final['C3']);self.assertEqual(len(c.records),5)
        self.assertEqual(sum(row['selection_applied'] for row in self.driver.calls),0)
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        remove_attention_serving_binding(self.worker,'33.paged_decode')
        before=self.api_calls;self.forward('ordinary-after-cancel');self.assertEqual(self.api_calls-before,56)
        self.assertIs(self.api.npu_fused_infer_attention_score,self.original)


if __name__=='__main__':unittest.main()
