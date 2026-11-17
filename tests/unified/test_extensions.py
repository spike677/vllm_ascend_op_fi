import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import demoncat
from fault_injection.unified.correlation import InvocationCorrelator,InvocationEvent
from fault_injection.unified.build_recipes import plan_fa_build
from fault_injection.unified.fia_entry import FIAModelEntry
from fault_injection.unified.worker import WorkerControl


class Extensions(unittest.TestCase):
    def event(self,event_id,parent=None,**extra):
        values=dict(event_id=event_id,parent_id=parent,kind='kernel',process=1,worker='w',rank=0,device=0,
                    thread=2,request_id='r',generation=1,source='runtime')
        values.update(extra);return InvocationEvent(**values)
    def test_out_of_order_and_scope_correlation(self):
        graph=InvocationCorrelator();graph.record(self.event('k','n'));graph.record(self.event('n',kind='native'))
        graph.record(self.event('other','n',rank=1));graph.record(self.event('mock','n',source='mock'))
        graph.record(self.event('lost','uncaptured'))
        rows={r['target']:r for r in graph.report()['edges']}
        self.assertEqual(rows['k']['status'],'OBSERVED');self.assertEqual(rows['other']['status'],'REJECTED')
        self.assertEqual(rows['mock']['status'],'MOCK');self.assertEqual(rows['lost']['status'],'NOT_VISIBLE')
        self.assertFalse(graph.report()['npu_qualified'])
    def test_duplicate_cycle_and_generation_refusal(self):
        graph=InvocationCorrelator();graph.record(self.event('a','b'))
        with self.assertRaisesRegex(ValueError,'CYCLIC'):graph.record(self.event('b','a'))
        graph.record(self.event('b',generation=2))
        self.assertEqual(graph.report()['edges'][0]['status'],'REJECTED')
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):graph.record(self.event('a'))
    def test_missing_sdk_and_version_are_not_build_success(self):
        with tempfile.TemporaryDirectory() as folder:
            paths={k:Path(folder)/k for k in ('sdk','source','policy_receipt','dependencies','work')}
            report=plan_fa_build(cann='9.1',**paths)
            self.assertEqual(report['status'],'DEPENDENCY_MISSING');self.assertFalse(report['compiled'])
            self.assertIn(str(paths['sdk'].resolve()),report['command'])
            self.assertEqual(plan_fa_build(cann='8.5.1',**paths)['status'],'NOT_IMPLEMENTED')
            with self.assertRaisesRegex(ValueError,'CANN'):plan_fa_build(cann='future',**paths)
    def test_fia_mock_keeps_original_objects_and_distinct_policy_without_qualification(self):
        from test_attention28_prefill_boundary import fixture
        actual,metadata,scheduler=fixture()
        seen=[]
        def execute(inputs,policy):
            self.assertIs(inputs,actual);seen.append(policy['requested_mode']);return inputs['query']
        entry=FIAModelEntry('28.fia.denominator_replay',mock_executor=execute)
        for phase in ('baseline','fault','off'):
            self.assertIs(entry.invoke('npu_fused_infer_attention_score',actual,metadata=metadata,scheduler=scheduler,phase=phase),actual['query'])
        self.assertEqual(seen,['coupled_m128','coupled_replay_m128','off'])
        self.assertTrue(all(e['source']=='MOCK' and not e['npu_qualified'] for e in entry.events))
        with self.assertRaisesRegex(RuntimeError,'TND_PAGE16'):
            entry.invoke('npu_fused_infer_attention_score',dict(actual,input_layout='TND'),metadata=metadata,scheduler=scheduler)
    def test_worker_contract_uses_same_session_and_structured_results(self):
        model=torch.nn.RMSNorm(4).eval()
        worker=WorkerControl(model,'18.api_half',demoncat.example_config('18.api_half'))
        self.assertTrue(worker.discover(demoncat.Request(args=(torch.ones(1,4),))).ok)
        for name in ('arm','inject','verify','disarm','restore','audit','cleanup'):
            result=worker.dispatch(name);self.assertTrue(result['ok'],result);json.dumps(result,allow_nan=False)
        self.assertEqual(worker.dispatch('__dict__')['code'],'UNKNOWN_OPERATION')
    def test_cli_and_import_do_not_initialize_torch(self):
        code="import sys,demoncat; assert 'torch' not in sys.modules; print(len(demoncat.registry.definitions()))"
        result=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
        self.assertEqual(result.stdout.strip(),'17')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'invalid.json';path.write_text('{}')
            result=subprocess.run([sys.executable,'-m','demoncat','validate',str(path)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertEqual(json.loads(result.stdout)['error']['code'],'INVALID_CONFIG')
