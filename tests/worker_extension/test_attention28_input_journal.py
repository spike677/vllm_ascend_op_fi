"""Real CPU public calls plus explicit Native control doubles; no NPU grant."""
from copy import deepcopy
import ctypes as C
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
import test_attention28_stock_census as stock
from fault_injection import fia_input_journal as J
from fault_injection import quant_container_native as Q
from demoncat.runtime.vllm_ascend.worker_extension import attention28_stock_census as census


def attach(driver,state,*,worker=1001,model=1002):
    """Use production Python journal enter/leave with SDK return-code doubles."""
    journal=J.NativeFiaInputJournal.__new__(J.NativeFiaInputJournal)
    identity=dict(pid=os.getpid(),native_tid=threading.get_native_id(),process_start_identity='explicit-cpu-fixture-only')
    config=Q.instance(Q.Config);config.owner_hi=1;config.owner_lo=2;config.generation=1;config.thread=identity['native_tid']
    control=dict(scope_token=0,events=[],begin_status=0,end_status=0)
    journal.containers=NS(config=config,query=lambda:dict(scope_token=control['scope_token']))
    def begin(c,p):
        value=C.cast(p,C.POINTER(J.ScopeIdentity)).contents
        token=value.call_id.decode()
        if not state['token'] or state['token'].decode()!=token:raise AssertionError('journal entered outside actual launch token')
        control['scope_token']=value.scope.token;control['events'].append(('enter',token))
        return control['begin_status']
    def end(c,token):
        control['events'].append(('leave',int(token)))
        if not control['end_status']:control['scope_token']=0
        return control['end_status']
    journal.functions=dict(scope_begin=begin,scope_end=end)
    journal.started=True;journal.active=None;journal.scopes=[];journal.window_limit=64
    journal.startup=dict(identity=identity);journal.bound_worker=dict(pid=os.getpid(),worker_object_id=worker,model_object_id=model)
    journal.profile=dict(originals=dict(observer=dict(file_sha256='1'*64)))
    driver.input_journal=journal;driver._journal_current=None;driver._journal_requests={}
    driver.fingerprint['observer_library_sha256']='1'*64
    return journal,control,identity


class InputJournalWindowContracts(unittest.TestCase):
    def setUp(self):
        self.folder=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver,self.launch=stock.driver_fixture(self.folder)
        self.journal,self.control,self.identity=attach(self.driver,self.launch)
        self.enterContext(patch.object(J,'process_identity',return_value=self.identity))
        self.enterContext(patch.object(census,'capture_inputs',side_effect=lambda *a:deepcopy(stock.captured())))
        self.enterContext(patch.object(census,'observe_scope',return_value=dict(status='CPU raw seam',kernel_argument_abi='NOT_DECODED')))
        self.row=stock.call_row();self.row['native_census_scope'].update(phase='observe',worker_object_id=1001,
            model_object_id=1002,device_type='cpu')
        self.base=torch.arange(16384,dtype=torch.float16)
        self.v=self.base.as_strided((2,8,128),(4096,128,1),3072)
        self.mask=torch.ones(3,3,dtype=torch.int8).triu(1)
        self.actual=dict(query=self.v,key=self.v,value=self.v,atten_mask=self.mask)
        self.output=object();self.addCleanup(self.restore)

    def restore(self):
        self.launch.update(mode='ok',fail_sync=False);self.control['end_status']=0
        if self.driver.scope.owned:self.launch['token']=self.driver.scope.token.encode()
        self.driver.stop()

    def once(self):
        self.assertIs(self.driver.refs[0]['value'],self.v)
        self.assertEqual(self.v.stride(),(4096,128,1));self.assertEqual(self.v.storage_offset(),3072)
        self.assertEqual(self.control['events'][-1][0],'enter')
        self.row['original_calls']+=1;return self.output

    def invoke(self,once=None):
        return self.driver.observe(self.once,(self.v,self.v,self.v),self.actual,self.row,None,once or self.once)

    def test_actual_noncontiguous_input_and_launch_token_survive_once(self):
        before=self.base.clone();output,record=self.invoke()
        self.assertIs(output,self.output);self.assertEqual(self.row['original_calls'],1)
        self.assertTrue(torch.equal(self.base,before))
        value=record['input_journal']['public_inputs']['value']
        self.assertEqual(value['object_id'],id(self.v));self.assertEqual(value['stride'],[4096,128,1])
        self.assertEqual(value['storage_offset'],3072);self.assertEqual(value['storage_address'],self.base.data_ptr())
        self.assertEqual(self.journal.scopes[0]['identity']['call_id'],record['call_id'])
        self.assertTrue(record['input_journal']['window_entered'] and record['input_journal']['window_left'])
        self.assertEqual([e[0] for e in self.control['events']],['enter','leave'])
        self.assertIsNone(self.journal.active);self.assertIsNone(self.driver.refs)

    def test_query_does_not_call_journal_or_add_sync(self):
        self.invoke();before=(self.launch['syncs'],list(self.control['events']))
        self.journal.query=lambda:(_ for _ in ()).throw(AssertionError('query must not call journal'))
        for _ in range(3):self.driver.state()
        self.assertEqual(before,(self.launch['syncs'],self.control['events']))

    def test_original_error_wins_and_pending_leave_is_retained(self):
        def fail():self.row['original_calls']+=1;raise ValueError('original once failed')
        self.control['end_status']=7
        with self.assertRaisesRegex(ValueError,'original once failed'):self.invoke(fail)
        with self.assertRaisesRegex(RuntimeError,'DRAIN_OR_PENDING'):self.driver.stop()
        self.assertIsNotNone(self.journal.active);self.assertIsNotNone(self.driver.refs)
        self.assertEqual(self.row['original_calls'],1)
        self.assertIn('original once failed',self.journal.active['original_error'])
        self.control['end_status']=0;self.driver.stop()
        self.assertEqual(self.journal.scopes[0]['status'],'ORIGINAL_FAILED')
        self.assertIsNone(self.driver.refs);self.assertIn('original once failed',self.driver.first_error)

    def test_uncertain_journal_entry_keeps_owner_and_does_not_call_original(self):
        self.control['begin_status']=9
        with self.assertRaisesRegex(RuntimeError,'SCOPE_BEGIN_FAILED'):self.invoke()
        self.assertEqual(self.row['original_calls'],0);self.assertIsNotNone(self.journal.active)
        self.assertIsNotNone(self.driver.refs)
        self.driver.stop();self.assertIsNone(self.journal.active)
        self.assertEqual(self.row['original_calls'],0)

    def test_unknown_journal_type_and_changed_model_refuse(self):
        self.driver.input_journal=object()
        with self.assertRaisesRegex(RuntimeError,'EXACT_INPUT_JOURNAL'):self.driver._validate_journal()
        self.driver.input_journal=self.journal;self.row['native_census_scope']['model_object_id']=999
        with self.assertRaisesRegex(RuntimeError,'ACTUAL_JOURNAL_OBJECT_SCOPE_CHANGED'):self.invoke()
        self.assertEqual(self.row['original_calls'],0)

    def test_completion_failure_cannot_claim_window_was_left(self):
        original=self.driver._synchronize;counter=[0]
        def sync():
            counter[0]+=1
            if counter[0]==2:raise RuntimeError('completion failed')
            return original()
        self.driver._synchronize=sync
        with self.assertRaisesRegex(RuntimeError,'completion failed'):self.invoke()
        self.assertEqual(self.row['original_calls'],1);self.assertIsNotNone(self.journal.active)
        self.assertFalse(self.row['native_census']['input_journal']['window_left'])
        self.driver._synchronize=original;self.driver.stop()
        self.assertIsNone(self.journal.active);self.assertIn('completion failed',self.driver.first_error)


class PublicTwoRequestJournalContracts(unittest.TestCase):
    def setUp(self):
        self.fixture=stock.PublicStockCensusContracts();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.journal,self.control,self.identity=attach(f.driver,f.native,worker=id(f.f.worker),model=id(f.f.model))
        self.enterContext(patch.object(J,'process_identity',return_value=self.identity))
        self.addCleanup(lambda:self.control.update(end_status=0))

    def test_all_56_actual_cpu_public_calls_enter_and_leave_same_tokens(self):
        f=self.fixture;f.f.start(f.client);f.f.request(f.client,0);result=f.f.request(f.client,1)
        calls=[c for r in result['evidence']['requests'] for forward in r['forwards'] for c in forward['calls']]
        self.assertEqual(len(calls),56);self.assertEqual(len(f.f.api_calls),56);self.assertEqual(len(self.journal.scopes),56)
        self.assertEqual(len(self.control['events']),112)
        self.assertEqual({s['identity']['call_id'] for s in self.journal.scopes},
            {c['native_census']['call_id'] for c in calls})
        self.assertTrue(all(c['original_calls']==1 and c['native_census']['input_journal']['window_left'] for c in calls))
        self.assertEqual({s['identity']['scope']['forward'] for s in self.journal.scopes},{1,2})
        self.assertEqual({s['identity']['module'] for s in self.journal.scopes},{id(m) for m in f.f.model.layers})
        before=list(self.control['events']);f.client.query();self.assertEqual(before,self.control['events'])
        f.f.cleanup(f.client)
        with torch.inference_mode():f.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
        self.assertEqual(len(f.f.api_calls),84);self.assertEqual(len(self.journal.scopes),56)
        self.assertFalse(result['C1'] or result['C2'] or result['C3'])


class ProbeLifecycleContracts(unittest.TestCase):
    """Real public Core+CPU model; explicit final Native snapshot/release doubles.

    The real journal enter/leave control flow is exercised above and here. SDK
    installation/readback cannot be emulated as NPU evidence by these fixtures.
    """
    def setUp(self):
        from experiments.worker_core_v11_20261118 import natural_attention28_input_journal_probe as probe
        from demoncat.runtime.vllm_ascend.worker_extension import serving_context
        self.probe=probe;self.fixture=stock.PublicStockCensusContracts()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.journal,self.control,self.identity=attach(f.driver,f.native,worker=id(f.f.worker),model=id(f.f.model))
        self.enterContext(patch.object(J,'process_identity',return_value=self.identity))
        self.enterContext(patch.object(probe.common,'actual',side_effect=lambda worker:worker))
        self.source=dict(path=str(Path(probe.__file__).resolve()),sha256=probe.common.sha(probe.__file__))
        for key,value in dict(_HELD=self.journal,_CENSUS=f.driver,_SOURCE=self.source,
                _CLEANUP_ATTEMPTS=0,_BINDING_INSTALLED=True,_BINDING_REMOVED=False).items():
            self.enterContext(patch.object(probe,key,value))
        self.order=[];self.released=False;self.fail_request=None;self.fail_release=False
        def stop():
            self.assertIsNone(self.journal.active);self.order.append('journal_stop');self.journal.started=False
        def snapshot():
            self.order.append('journal_snapshot');n=len(self.journal.scopes)
            state=dict(accepted_windows=n,prepare_returns=n,execute_returns=n,total_calls=2*n,
                retained_events=4*n,total_events=4*n,first_error=0,overflow=0,exceptions=0,
                foreign_thread_calls=0,getter_failures=0,pending_executors=0,untracked_executors=0,
                failed_prepare_outputs_unknown=0)
            return dict(explicit_cpu_final_native_snapshot_double=True,state=state,public_scopes=self.journal.scopes)
        def release():
            self.order.append('journal_release')
            if self.fail_release:raise RuntimeError('CPU explicit release refusal')
            self.released=True
            return dict(fia={'installed':0},containers={'installed':0},global_wrappers_unloaded=False)
        self.journal.stop=stop;self.journal.snapshot=snapshot;self.journal.release=release
        owner=self
        class Llm:
            def __init__(self):
                self.llm_engine=NS(has_unfinished_requests=lambda:False,
                    abort_request=lambda *args,**kwargs:owner.order.append('abort_owned_request'))
                self.index=0
            def reset_prefix_cache(self,**kwargs):return True
            def enqueue(self,*args,**kwargs):
                self.index+=1;self.request='actual-cpu-'+str(self.index);return [self.request]
            def wait_for_completion(self,**kwargs):
                if self.index==owner.fail_request:raise RuntimeError('actual CPU request failure')
                if self.index==3:
                    owner.assertTrue(owner.released)
                    owner.assertFalse(f.client.snapshot()['ownership_retained'])
                    owner.assertEqual(len(owner.journal.scopes),56)
                    owner.order.append('post_release_health')
                with torch.inference_mode():
                    hidden=f.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]));f.f.model.logits(hidden)
                return [NS(request_id=self.request,outputs=[NS(token_ids=[7])])]
            def collective_rpc(self,callback,args,timeout):
                owner.order.append(callback.__name__)
                return [callback(f.f.worker,*args)]
        self.llm=Llm()
        self.enterContext(patch.object(serving_context,'read_enqueued_identity',side_effect=lambda llm,ids,**kw:
            dict(internal_request_id=ids[0],external_request_id=ids[0])))
        self.report=dict(status='STARTED',healthy_before=[7],C1=False,C2=False,C3=False,E5=False)
        self.out=f.folder/'probe';self.out.mkdir()

    def run_drive(self):
        self.probe.drive(self.llm,self.fixture.client,prompt={'prompt_token_ids':[1,2,3]},params=object(),
            out=self.out,report=self.report,source=self.source)

    def test_actual_public_driver_releases_before_same_worker_health(self):
        self.run_drive();self.assertEqual(self.report['status'],self.probe.SUCCESS,self.report.get('error'))
        self.assertEqual(self.order,['remove_owned_binding','finish_journal','journal_stop','journal_snapshot',
            'journal_release','post_release_health'])
        self.assertEqual(self.report['actual_calls_per_request'],[28,28])
        self.assertEqual(len(self.journal.scopes),56);self.assertEqual(len(self.fixture.f.api_calls),84)
        self.assertTrue((self.out/'input_journal/observation-01.json').is_file())
        self.assertTrue((self.out/'input_journal/released-01.json').is_file())
        self.assertFalse(any(self.report[k] for k in ('C1','C2','C3','E5')))

    def test_failed_actual_request_keeps_primary_and_cleans_without_health_or_second_capture(self):
        self.fail_request=2;self.run_drive()
        self.assertEqual(self.report['status'],'REFUSED_OR_FAILED')
        self.assertIn('actual CPU request failure',self.report['error'])
        self.probe.finish_failed_run(self.llm,self.fixture.client,self.fixture.client.snapshot()['worker'],
            self.source,self.out,self.report)
        self.assertTrue(self.released);self.assertEqual(len(self.journal.scopes),28)
        self.assertEqual(self.fixture.f.model.calls,1)
        self.assertNotIn('post_release_health',self.order)
        self.assertFalse(self.report['failure_journal_cleanup']['errors'])
        self.assertIn('actual CPU request failure',self.report['error'])
        failed=self.report['requests'][1]
        self.assertFalse(failed['completed']);self.assertEqual(failed['phase'],'observe_repeat')
        self.assertEqual(failed['mapping']['internal_request_id'],'actual-cpu-2')
        self.assertEqual(failed['stage'],'PUBLIC_WINDOW_ARMED')
        saved=json.loads((self.out/'result.json').read_bytes())
        self.assertEqual(saved['requests'][1],failed)

    def test_failed_release_original_is_preserved_then_owned_retry_without_another_request(self):
        self.fail_release=True;self.run_drive()
        self.assertEqual(self.report['status'],'REFUSED_OR_FAILED')
        first=self.probe.common.sha(self.out/'input_journal/cleanup-01.json')
        self.fail_release=False
        self.probe.finish_failed_run(self.llm,self.fixture.client,self.fixture.client.snapshot()['worker'],
            self.source,self.out,self.report)
        self.assertTrue(self.released);self.assertEqual(self.llm.index,2)
        self.assertEqual(self.probe.common.sha(self.out/'input_journal/cleanup-01.json'),first)
        self.assertTrue(self.report['journal_cleanup']['errors'])
        self.assertFalse(self.report['failure_journal_cleanup']['errors'])
        self.assertTrue((self.out/'input_journal/released-02.json').is_file())
        self.assertNotIn('post_release_health',self.order)

    def test_binding_or_public_owner_prevents_journal_retirement(self):
        with self.assertRaisesRegex(RuntimeError,'BINDING_MUST_BE_REMOVED'):
            self.probe.finish_journal(self.fixture.f.worker,self.fixture.f.worker.demoncat_worker_identity()['worker'],
                self.source,str(self.out))
        self.assertFalse(self.order)
        blocked=NS(snapshot=lambda:dict(ownership_retained=True))
        self.probe.finish_failed_run(self.llm,blocked,{},self.source,self.out,self.report)
        self.assertEqual(self.report['failure_journal_cleanup_withheld'],'PUBLIC_OWNERSHIP_RETAINED')
        self.assertEqual(self.order,['snapshot_failed_owner'])
        refusal=self.report['failure_journal_snapshot']
        self.assertEqual(refusal['status'],'READ_ONLY_FAILURE_SNAPSHOT_REFUSED')
        self.assertIn('HELD_ACTUAL_WORKER_OR_MODEL_CHANGED',refusal['errors'][0]['error'])

    def readonly_native_state_doubles(self,**changes):
        """Production snapshot loop + actual ctypes return-code state/event seam."""
        options=dict(active_calls=0,active_getters=0,wrong_owner=False,query_error=False,drift=False,drifted=False)
        options.update(changes);journal=self.journal;config=journal.containers.config
        def container_state():
            value=J.record(Q.instance(Q.State))
            value.update(installed=1,owner_hi=config.owner_hi,owner_lo=config.owner_lo,generation=config.generation,
                owner_thread=config.thread,scope_token=self.control['scope_token'],active_calls=0,live_registrations=1)
            value['arrays'].update(active_getters=options['active_getters'],acquired_arrays=int(options['drifted']))
            return value
        def state(address):
            if options['query_error']:raise RuntimeError('explicit Native state query refused')
            value=Q.instance(J.State)
            value.installed=1;value.pid=os.getpid();value.thread=config.thread
            value.owner_hi=config.owner_hi;value.owner_lo=config.owner_lo+int(options['wrong_owner'])
            value.generation=config.generation;value.active_calls=options['active_calls'];value.pending_executors=1
            value.retained_events=1;value.total_events=1;value.accepted_windows=len(journal.scopes)
            C.memmove(address,C.byref(value),C.sizeof(value));return 0
        def event(index,address):
            self.order.append('readonly_native_event')
            self.assertEqual(index,0);value=Q.instance(J.Event)
            value.sequence=1;value.operation=2;value.phase=1;value.executor=991
            C.memmove(address,C.byref(value),C.sizeof(value));return 0
        def container_snapshot():
            result=dict(state=container_state(),events=[],explicit_cpu_container_snapshot=True)
            if options['drift']:options['drifted']=True
            return result
        journal.functions.update(state=state,event=event)
        journal.containers.current=lambda:self.order.append('readonly_mapped_identity')
        journal.containers.query=container_state;journal.containers.snapshot=container_snapshot
        journal.containers.manifest_sha256='0'*64;journal.containers.mapping={'explicit_cpu_identity':True}
        journal.sdk_tags={};journal.limits=None;journal.lifecycle=[];journal.first_lifecycle_error=None;journal.lifecycle_overflow=0
        journal.snapshot=J.NativeFiaInputJournal.snapshot.__get__(journal)
        return options

    def fail_with_pending(self,**state):
        self.control['end_status']=7
        def restore():
            self.control['end_status']=0;self.fixture.driver.stop()
        self.addCleanup(restore)
        self.run_drive();self.readonly_native_state_doubles(**state)
        self.assertTrue(self.fixture.client.snapshot()['ownership_retained']);self.assertIsNotNone(self.journal.active)
        self.assertEqual(len(self.journal.scopes),1)
        primary=self.report['error']
        self.probe.finish_failed_run(self.llm,self.fixture.client,self.fixture.client.snapshot()['worker'],
            self.source,self.out,self.report)
        self.assertEqual(self.report['error'],primary)
        self.assertTrue(self.fixture.client.snapshot()['ownership_retained']);self.assertIsNotNone(self.journal.active)
        self.assertFalse(self.released)
        self.assertNotIn('journal_stop',self.order);self.assertNotIn('journal_release',self.order)
        self.assertNotIn('post_release_health',self.order)
        return self.report['failure_journal_snapshot']

    def test_actual_core_pending_owner_saves_readonly_events_without_cleanup(self):
        result=self.fail_with_pending()
        self.assertEqual(result['status'],'READ_ONLY_PARTIAL_FAILURE_SNAPSHOT_RECORDED_OWNER_RETAINED')
        saved=json.loads((self.out/'input_journal/failure-observation-01.json').read_bytes())
        self.assertEqual(saved['events'][0]['executor'],991);self.assertEqual(saved['state']['pending_executors'],1)
        self.assertEqual(saved['containers']['state']['live_registrations'],1)
        self.assertNotEqual(saved['public_scopes'][0]['status'],'RETURNED')
        self.assertFalse(result['cleanup_performed'] or result['ownership_release_confirmed'] or result['qualified'])
        self.assertFalse((self.out/'input_journal/released-01.json').exists())
        before=list(self.order);pin=self.probe.common.sha(self.out/'input_journal/failure-observation-01.json')
        self.probe.finish_failed_run(self.llm,self.fixture.client,self.fixture.client.snapshot()['worker'],
            self.source,self.out,self.report)
        self.assertEqual(self.order,before);self.assertEqual(self.probe.common.sha(self.out/'input_journal/failure-observation-01.json'),pin)

    def test_active_native_call_saves_refusal_without_reading_events(self):
        result=self.fail_with_pending(active_calls=1)
        self.assertEqual(result['status'],'READ_ONLY_FAILURE_SNAPSHOT_REFUSED')
        self.assertIn('STILL_ACTIVE',result['errors'][0]['error']);self.assertNotIn('readonly_native_event',self.order)
        self.assertTrue((self.out/'input_journal/failure-snapshot-01.json').exists())
        self.assertFalse((self.out/'input_journal/failure-observation-01.json').exists())

    def test_active_metadata_getter_saves_refusal_without_reading_events(self):
        result=self.fail_with_pending(active_getters=1)
        self.assertIn('STILL_ACTIVE',result['errors'][0]['error']);self.assertNotIn('readonly_native_event',self.order)

    def test_query_failure_does_not_invent_zero_state(self):
        result=self.fail_with_pending(query_error=True)
        self.assertIn('Native state query refused',result['errors'][0]['error'])
        self.assertNotIn('state_before',result);self.assertNotIn('readonly_native_event',self.order)

    def test_foreign_native_owner_cannot_be_read_as_ours(self):
        result=self.fail_with_pending(wrong_owner=True)
        self.assertIn('OWNER_CHANGED',result['errors'][0]['error']);self.assertNotIn('readonly_native_event',self.order)

    def test_container_full_state_drift_refuses_snapshot_not_cleanup(self):
        result=self.fail_with_pending(drift=True)
        self.assertIn('FULL_STATE_NOT_QUIESCENT',result['errors'][0]['error'])
        self.assertFalse((self.out/'input_journal/failure-observation-01.json').exists())
        self.assertTrue((self.out/'input_journal/failure-snapshot-01.json').exists())

    def test_secondary_host_receipt_write_error_does_not_replace_api_failure(self):
        self.fail_request=2;save=self.probe.common.save
        def fail_host(path,data):
            if path.name=='host_receipt.json':raise OSError('secondary host receipt disk error')
            return save(path,data)
        with patch.object(self.probe.common,'save',side_effect=fail_host):
            try:self.run_drive()
            except OSError as error:self.probe.note_failure(self.report,error,'probe_run')
        self.assertIn('actual CPU request failure',self.report['error'])
        self.assertIn('secondary host receipt disk error',self.report['secondary_errors'][0]['error'])
        self.probe.finish_failed_run(self.llm,self.fixture.client,self.fixture.client.snapshot()['worker'],
            self.source,self.out,self.report)
        self.assertTrue(self.released)


if __name__=='__main__':unittest.main()
