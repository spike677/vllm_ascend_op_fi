import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from experiments.worker_core_v11_20261118.dsv4_observe_probe import observe
from experiments.worker_core_v11_20261118.dsv4_probe_targeted import region, overlap, options_checked
from experiments.worker_core_v11_20261118.dsv4_noise_capture import capture

class Router(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.arange(24.).reshape(6,4)/24)
    def forward(self,x):
        scores=torch.nn.functional.linear(x,self.weight)
        values,indices=torch.topk(scores,2,dim=-1)
        combined=values.sum(-1,keepdim=True)
        return combined+1,indices

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.router=Router()
    def forward(self,x): return self.router(x)
    def compute_logits(self,x): return x

class Runner:
    def __init__(self):
        self.model=Model(); self.input_batch=NS(req_ids=['r'],num_computed_tokens_cpu=[0])
        self.query_start_loc_cpu=[0,3];self.requests={}
    def execute_model(self,scheduler_output,*,x):
        return self.model(x)
    def sample_tokens(self): return NS(sampled_token_ids=[[1]])

class TargetTests(unittest.TestCase):
    def worker(self):return NS(rank=0,model_runner=Runner())
    def run_trace(self,config=None):
        w=self.worker();x=torch.arange(12.).reshape(3,4);s=NS(num_scheduled_tokens={'r':3},scheduled_new_reqs=[NS(req_id='r')])
        baseline=w.model_runner.execute_model(s,x=x)
        observe(w,'start',{'target_layers':['router'],'targeted_observation':config or {}})
        actual=w.model_runner.execute_model(s,x=x)
        result=observe(w,'stop')
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(baseline,actual)))
        self.assertNotIn('execute_model',vars(w.model_runner))
        return result['targeted_observation']
    def test_direct_linear_topk_consumer_and_float_snapshot(self):
        trace=self.run_trace()
        self.assertFalse(trace['errors']);self.assertFalse(trace['truncated'])
        topk=next(x for x in trace['operations'] if 'topk' in x['operator'])
        self.assertTrue(topk['edges'])
        self.assertTrue(topk['inputs'][0]['sample_values'])
        self.assertTrue(topk['consumer_candidate'])
        self.assertTrue(topk['completion']['cpu']['complete'])
        self.assertEqual(topk['consumer_semantics'],'SOURCE_REVIEW_REQUIRED')
        self.assertTrue(any(x.get('sample_reason')=='PARAMETER_STORAGE_NOT_CAPTURED'
                            for op in trace['operations'] for x in op['inputs']))
    def test_budget_exhaustion_records_missing_without_altering_forward(self):
        trace=self.run_trace({'float_budget':1,'integer_budget':1})
        self.assertTrue(trace['truncated']);self.assertFalse(trace['errors'])
    def test_operation_bound(self):
        trace=self.run_trace({'max_operations':1})
        self.assertEqual(len(trace['operations']),1);self.assertTrue(trace['truncated'])
    def test_actual_inference_mode_tensor_lineage(self):
        with torch.inference_mode():
            trace=self.run_trace()
        self.assertFalse(trace['errors'])
        topk=next(x for x in trace['operations'] if 'topk' in x['operator'])
        self.assertTrue(topk['edges'])
        self.assertEqual(topk['edges'][0]['version_evidence'],'observed_schema_writes_only')
    def test_layer_and_config_rejected_before_install(self):
        w=self.worker()
        for opts in ({'targeted_observation':{}},
                     {'target_layers':['router'],'targeted_observation':{'float_budget':1048577}},
                     {'target_layers':['router'],'targeted_observation':{'capture_float':True}}):
            with self.assertRaises(ValueError):observe(w,'start',opts)
            self.assertNotIn('execute_model',vars(w.model_runner))
    def test_active_capture_refuses_targeted_start(self):
        w=self.worker();capture(w,'start')
        with self.assertRaisesRegex(ValueError,'CAPTURE_MUST_BE_IDLE'):
            observe(w,'start',{'target_layers':['router'],'targeted_observation':{}})
        capture(w,'abort')
    def test_strided_envelope_not_claimed_exact(self):
        x=torch.arange(16.).reshape(4,4)
        self.assertFalse(overlap(region(x),region(x[:,::2])))
        self.assertTrue(overlap(region(x),region(x[1:3])))
    def test_event_failure_missing_not_success_and_result_preserved(self):
        with patch('experiments.worker_core_v11_20261118.dsv4_probe_targeted.completed_event',side_effect=RuntimeError('event')):
            trace=self.run_trace()
        self.assertTrue(trace['errors']);self.assertFalse(trace['operations'])
    def test_view_only_not_consumer(self):
        trace=self.run_trace()
        view=next(x for x in trace['operations'] if x['operator']=='aten::t')
        self.assertFalse(view['consumer_candidate'])
    def test_text_extraction_determinism_size_and_verify(self):
        import tempfile,json
        from pathlib import Path
        from tools.extract_dsv4_targeted_record import pages,verify,canonical
        trace=self.run_trace()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'raw.json';out=Path(folder)/'text.jsonl'
            p.write_text(json.dumps(dict(rank=0,targeted_observation=trace)),encoding='utf-8')
            a=pages(p);self.assertEqual(a,pages(p))
            self.assertTrue(all(len(canonical(x))+1<=4096 for x in a))
            out.write_bytes(b'\n'.join(canonical(x) for x in a)+b'\n')
            self.assertTrue(verify(out)['verified'])

if __name__=='__main__':unittest.main()

