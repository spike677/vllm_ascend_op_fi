"""Real CPU objects and public cleanup; fake metadata is never NPU evidence."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from demoncat.runtime.vllm_ascend.worker_extension import attention28_prepared_input as P
import test_attention28_stock_census as prior


class PreparedInputOwnership(unittest.TestCase):
    def setUp(self):
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(P,'EVIDENCE_ROOT',self.root))
        self.owner=dict(pid=os.getpid(),native_tid=threading.get_native_id(),python_thread_ident=threading.get_ident(),process_start_ticks='44')
        self.enterContext(patch.object(P,'identity',side_effect=lambda:dict(self.owner,native_tid=threading.get_native_id(),python_thread_ident=threading.get_ident())))
        self.env={};self.enterContext(patch.object(P.os,'environ',self.env))
        self.model=object();self.worker=NS(model_runner=NS(model=self.model))
        build=dict(status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',variant='tnd_prefill_denominator_replay',
            tnd28_compiled_input_observer=dict(schema='demoncat.tnd28.compiled_input_source.v1',observer_only=True,fault_authorized=False))
        self.path=self.root/'build.json';self.path.write_text(json.dumps(build));self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.capture=P.PreparedInputCapture(self.worker,build_completion=self.path,build_sha256=self.sha,evidence_dir=self.root/'owned')
        base=torch.arange(24,dtype=torch.float16).reshape(3,8)
        self.actual=dict(query=base[:,:4],key=base[:,:2],value=base[:,6:],atten_mask=torch.ones(3,3,dtype=torch.int8).triu(1))
        self.row=dict(module_object_id=20,module_path='actual.module',request_id='r',scheduler={'internal_request_id':'r'},
            native_census_scope=dict(owner='a'*32,generation=1,worker_object_id=id(self.worker),runner_object_id=id(self.worker.model_runner),
                model_object_id=id(self.model),phase='observe'))
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.capture.active is not None:
            self.worker.model_runner.model=self.model
            self.env.update(self.capture.active)
            self.capture.stop()

    def enter(self):return self.capture.enter('b'*32,self.row,self.actual,{'explicit_cpu_only_fixture':True})

    def test_actual_objects_retained_until_cleanup_and_zero_native_remains_zero(self):
        before=self.actual['value'].clone();result=self.enter()
        self.assertTrue(all(a is b for a,b in zip(self.capture.refs,self.actual.values())))
        output=(self.actual['query']+1,torch.empty(0));self.capture.returned(output)
        self.assertIs(self.capture.refs[-2],output[0]);self.capture.stop()
        self.assertEqual(self.env,{});self.assertIsNone(self.capture.refs)
        self.assertTrue(result['window_complete']);self.assertFalse(result['native_records_present'] or result['actual_native_observed'] or result['prepared_input_join'])
        self.assertTrue(torch.equal(before,self.actual['value']))
        with self.assertRaisesRegex(RuntimeError,'SINGLE_CURRENT_CALL'):self.enter()

    def test_foreign_environment_preserves_refs_and_retry_removes_only_owned_entries(self):
        self.enter();self.env['DCAT28_INPUT_OWNER']='foreign';self.env['UNRELATED']='preserve'
        with self.assertRaisesRegex(RuntimeError,'FOREIGN_ENVIRONMENT'):self.capture.stop()
        self.assertIsNotNone(self.capture.refs);self.assertTrue(self.capture.active)
        self.env['DCAT28_INPUT_OWNER']='a'*32;self.capture.stop()
        self.assertEqual(self.env,{'UNRELATED':'preserve'});self.assertIn('FOREIGN_ENVIRONMENT',self.capture.first_error)

    def test_after_delete_failure_retires_only_observed_absent_entry_then_retries(self):
        class After(dict):
            failed=False
            def __delitem__(self,key):
                super().__delitem__(key)
                if not self.failed:self.failed=True;raise OSError('after removal')
        self.env=After();P.os.environ=self.env;self.enter()
        with self.assertRaisesRegex(OSError,'after removal'):self.capture.stop()
        self.assertNotIn('DCAT28_INPUT_MODE',self.capture.active);self.assertIsNotNone(self.capture.refs)
        self.capture.stop();self.assertEqual(self.env,{});self.assertIn('after removal',self.capture.first_error)

    def test_partial_environment_start_retains_only_actual_owned_entries(self):
        class Partial(dict):
            def __setitem__(self,key,value):
                if key=='DCAT28_INPUT_MODULE':raise MemoryError('partial write')
                return super().__setitem__(key,value)
        self.env=Partial();P.os.environ=self.env
        with self.assertRaises(MemoryError):self.enter()
        self.assertEqual(set(self.capture.active),set(self.env));self.assertIsNotNone(self.capture.refs)
        self.capture.stop();self.assertEqual(self.env,{})

    def test_foreign_model_or_thread_cannot_release(self):
        self.enter();self.worker.model_runner.model=object()
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_CHANGED'):self.capture.stop()
        self.worker.model_runner.model=self.model;errors=[]
        def other():
            try:self.capture.stop()
            except RuntimeError as e:errors.append(str(e))
        thread=threading.Thread(target=other);thread.start();thread.join()
        self.assertIn('IDENTITY_CHANGED',errors[0]);self.assertIsNotNone(self.capture.refs)

    def test_build_bytes_cannot_change_or_old_package_mint_observer(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'BUILD_IDENTITY'):P.PreparedInputCapture(self.worker,
            build_completion=self.path,build_sha256=self.sha,evidence_dir=self.root/'new')


class OnePublicObservation(unittest.TestCase):
    def setUp(self):
        self.f=prior.PublicStockCensusContracts();self.f.setUp();self.addCleanup(self.f.doCleanups)

    def test_one_actual_request_keeps_all_28_calls_then_public_disable_releases(self):
        f=self.f
        capture=NS(used=False,active=False,stops=0)
        def enter(token,row,actual,api):capture.used=True;capture.active=True;return dict(explicit_cpu_fixture=True)
        def stop():capture.active=False;capture.stops+=1
        capture.enter=enter;capture.stop=stop;capture.returned=lambda output:None
        f.driver.prepared_input_capture=capture
        f.f.start(f.client);state=f.f.request(f.client,0)
        calls=[c for row in state['evidence']['requests'] for forward in row['forwards'] for c in forward['calls']]
        self.assertEqual(len(calls),28);self.assertEqual(f.native['begins'],1);self.assertEqual(len(f.f.api_calls),28)
        self.assertEqual(sum('prepared_input_capture' in c['native_census'] for c in calls),1)
        self.assertEqual(sum(c['native_census']['status']=='UNOBSERVED_OUTSIDE_SINGLE_PREPARED_INPUT_WINDOW' for c in calls),27)
        for name in ('disable','unregister'):f.f.ok(getattr(f.client,name)())
        self.assertEqual(f.client.detach()['status'],'DETACHED')
        self.assertFalse(f.client.snapshot()['ownership_retained']);self.assertIsNone(f.driver.refs);self.assertFalse(capture.active)
        with torch.inference_mode():f.f.model(torch.tensor([1,2,3]),torch.tensor([0,1,2]))
        self.assertEqual(len(f.f.api_calls),56);self.assertEqual(f.native['begins'],1)


if __name__=='__main__':unittest.main()
