"""New fused-input variant CPU contracts, not Worker/NPU qualification."""
import unittest
import torch

from fault_injection.core import UnsupportedCall
from fault_injection.rope_census import NaturalRoPETrace
from fault_injection.rope_fused22 import FusedWrongRowSession
from test_rope_shared_triton import Model,JIT,triton_reader


class FusedWrongRowTests(unittest.TestCase):
    def setUp(self):
        JIT.skip_launch=False;JIT.replace_position=False
        self.model=Model().eval();self.positions=torch.arange(4);self.x=torch.arange(16.).reshape(4,4)
    def runtime(self,policy='exact'):
        return NaturalRoPETrace(self.model,triton_reader,{'layers.0.rope':self.model.layers[0].rope},{},
            simulation=True,outer_operator='rope_shared_triton_fixture::opaque',triton_kernel=JIT,
            position_policy=policy)
    def session(self,**kwargs):
        return FusedWrongRowSession(self.model,simulation=True,_runtime=self.runtime(),**kwargs)
    def bind(self,session,request='actual-a'):
        session.bind_request(request,self.positions,list(range(4)))
    def healthy(self):
        expected=self.model(self.x,self.positions)
        with self.session(arm='baseline') as baseline:
            self.bind(baseline)
            observed=self.model(self.x,self.positions)
        self.assertTrue(torch.equal(expected,observed))
        return expected,baseline.healthy_calls
    def test_all_shared_sites_real_wrong_row_reader_consumers_and_off(self):
        baseline,refs=self.healthy();cache=self.model.layers[0].rope.cos_sin_cache.clone()
        trace=self.runtime('exact_or_last_from_first')
        with trace,self.session(references=refs) as session:
            results=[]
            for request in ('actual-f1','actual-f2'):
                self.bind(session,request);trace.set_request(request,self.positions,list(range(4)),{})
                results.append(self.model(self.x,self.positions));trace.end_request()
        report=trace.report();self.assertTrue(report['complete'],report['errors'])
        self.assertTrue(all(row['independent_position_transform']=='last_from_first' for row in report['readers']))
        self.assertTrue(all(row['root_positions_unchanged'] and row['complete_cache_unchanged'] for row in report['readers']))
        self.assertTrue(all(any(c['returned'] for c in row['consumers']) for row in report['readers']))
        self.assertEqual(len(session.events),6);self.assertTrue(all(e['applied'] for e in session.events))
        self.assertEqual({e['path'] for e in session.events},{'layers.0.rope','layers.1.rope','layers.2.rope'})
        self.assertTrue(torch.equal(results[0],results[1]));self.assertFalse(torch.equal(results[0],baseline))
        self.assertTrue(torch.equal(results[0][:-1],baseline[:-1]))
        self.assertTrue(torch.equal(self.model(self.x,self.positions),baseline))
        self.assertTrue(torch.equal(self.positions,torch.arange(4)))
        self.assertTrue(torch.equal(self.model.layers[0].rope.cos_sin_cache,cache))
        self.assertTrue(session.events[0]['same_complete_inputs_as_healthy'])
        self.assertTrue(session.events[1]['propagated_inputs_possible'])
        self.assertFalse(session.report()['c2']);self.assertFalse(session.report()['c3'])
    def test_missing_reference_refuses_before_actual_rope(self):
        with self.session() as session:
            self.bind(session)
            with self.assertRaisesRegex(UnsupportedCall,'HEALTHY_SITE_REFERENCE_REQUIRED'):
                self.model(self.x,self.positions)
        self.assertEqual(session.events[0]['status'],'observed')
    def test_unselected_calls_pass_original_position_object_and_output(self):
        baseline,refs=self.healthy();trace=self.runtime('exact_or_last_from_first')
        with trace,self.session(references=refs,admission=lambda event:False) as session:
            self.bind(session);trace.set_request('actual-a',self.positions,list(range(4)),{})
            result=self.model(self.x,self.positions);trace.end_request()
        self.assertTrue(trace.report()['complete'],trace.errors)
        self.assertTrue(all(r['independent_position_transform']=='identity' for r in trace.rows))
        self.assertTrue(torch.equal(result,baseline));self.assertTrue(all(not e['applied'] for e in session.events))
    def test_non_target_producer_change_refuses(self):
        _,refs=self.healthy()
        with self.session(references=refs) as session:
            self.bind(session);changed=self.x.clone();changed[0,0]+=1
            with self.assertRaisesRegex(UnsupportedCall,'NON_TARGET_INPUT_OR_CACHE_CHANGED'):
                self.model(changed,self.positions)
    def test_fault_requires_explicit_cpu_runtime_fixture(self):
        with self.assertRaisesRegex(UnsupportedCall,'EXPLICIT_CPU_ROPE_FIXTURE_REQUIRED'):
            FusedWrongRowSession(self.model,simulation=True)


if __name__=='__main__':unittest.main()
