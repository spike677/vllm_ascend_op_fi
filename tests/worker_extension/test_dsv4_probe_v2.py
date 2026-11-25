import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from experiments.worker_core_v11_20261118.dsv4_probe_support import MethodLease, ProbeRefused, callable_identity, deployment_identity
from experiments.worker_core_v11_20261118.dsv4_observe_probe import observe
from experiments.worker_core_v11_20261118.dsv4_noise_capture import capture
from experiments.worker_core_v11_20261118.dsv4_probe_extension import Dsv4ProbeExtension

class Model:
    def compute_logits(self,x,*,gain=1):return x*gain
class Runner:
    def __init__(self):
        self.model=Model();self.input_batch=NS(req_ids=['r'],num_computed_tokens_cpu=[8]);self.requests={}
    def execute_model(self,scheduler_output,*,marker=None):return marker
    def sample_tokens(self):return NS(sampled_token_ids=[[1]])
class ProbeTests(unittest.TestCase):
    def worker(self):return NS(rank=0,model_runner=Runner())
    def test_class_method_restored_and_idle_structured(self):
        w=self.worker();self.assertFalse(observe(w,'status')['installed'])
        observe(w,'start');observe(w,'stop');self.assertNotIn('execute_model',vars(w.model_runner))
        self.assertFalse(capture(w,'status')['installed'])
    def test_reviewed_instance_callable_exact_restoration(self):
        w=self.worker();r=w.model_runner;r.execute_model=r.execute_model;old=r.execute_model
        receipt=callable_identity(r,'execute_model')
        observe(w,'start',{'approved_baselines':{'execute_model':receipt['fingerprint']}})
        observe(w,'stop');self.assertIs(r.execute_model,old)
    def test_unknown_instance_and_wrapped_refused(self):
        w=self.worker();w.model_runner.execute_model=lambda *a,**k:1
        with self.assertRaisesRegex(ProbeRefused,'BASELINE_REVIEW'):observe(w,'start')
        self.assertFalse(hasattr(w,'_dcat_dsv4_observer'))

    def test_review_receipt_cannot_be_reused_for_another_wrapped_object(self):
        import functools
        w=self.worker();r=w.model_runner;original=r.execute_model
        def factory():
            @functools.wraps(original)
            def wrapper(*args,**kwargs):return original(*args,**kwargs)
            return wrapper
        r.execute_model=factory();receipt=callable_identity(r,'execute_model')
        r.execute_model=factory()  # same source/qualname/__wrapped__, different live object
        with self.assertRaisesRegex(ProbeRefused,'BASELINE_REVIEW'):
            observe(w,'start',{'approved_baselines':{'execute_model':receipt['fingerprint']}})
        receipt=callable_identity(r,'execute_model');actual=r.execute_model
        observe(w,'start',{'approved_baselines':{'execute_model':receipt['fingerprint']}})
        observe(w,'stop');self.assertIs(r.execute_model,actual)
    def test_duplicate_does_not_stack(self):
        w=self.worker();observe(w,'start');fn=w.model_runner.execute_model
        with self.assertRaisesRegex(ProbeRefused,'ALREADY_OWNED'):observe(w,'start')
        self.assertIs(fn,w.model_runner.execute_model);observe(w,'stop')

    def test_budget_can_be_bounded_without_restart(self):
        w=self.worker()
        with self.assertRaisesRegex(ProbeRefused,'BUDGET'):observe(w,'start',{'integer_budget':4194305})
        self.assertNotIn('execute_model',vars(w.model_runner))
        observe(w,'start',{'integer_budget':4194304,'max_calls':16384})
        self.assertEqual(w._dcat_dsv4_observer['integer_budget_remaining'],4194304)
        observe(w,'stop')
    def test_partial_install_rolls_back_only_own(self):
        w=self.worker();original=MethodLease.install
        def install(lease,fn):
            if lease.name=='sample_tokens':raise RuntimeError('injected install failure')
            return original(lease,fn)
        with patch.object(MethodLease,'install',install):
            with self.assertRaisesRegex(RuntimeError,'install failure'):capture(w,'start')
        self.assertNotIn('compute_logits',vars(w.model_runner.model));self.assertFalse(hasattr(w,'_dcat_noise_capture'))
    def test_foreign_replacement_not_removed(self):
        w=self.worker();capture(w,'start');foreign=lambda *a:0;w.model_runner.sample_tokens=foreign
        with self.assertRaisesRegex(ProbeRefused,'OWNER_CHANGED'):capture(w,'abort')
        self.assertIs(w.model_runner.sample_tokens,foreign)
        self.assertIn('compute_logits',vars(w.model_runner.model))
    def test_call_args_return_exception_identity_preserved(self):
        w=self.worker();r=w.model_runner;error=RuntimeError('original')
        observe(w,'start');obj=object();scheduler=NS(num_scheduled_tokens={},scheduled_new_reqs=[])
        self.assertIs(r.execute_model(scheduler_output=scheduler,marker=obj),obj);observe(w,'stop')
        class Broken(Model):
            def compute_logits(self,*args,**kwargs):raise error
        r.model=Broken();capture(w,'start')
        with self.assertRaises(RuntimeError) as got:r.model.compute_logits(torch.ones(1,6),gain=2)
        self.assertIs(got.exception,error);capture(w,'abort')
    def test_kwargs_rpc_and_samples_full_arrays(self):
        w=self.worker()
        call=lambda **kw:Dsv4ProbeExtension.demoncat_dsv4_probe(w,**kw)
        self.assertEqual(call(tool='capture',action='start',options={'phase':'fault'})['exit_code'],0)
        x=torch.arange(6.).reshape(1,6)
        for i in range(4):
            self.assertTrue(torch.equal(w.model_runner.model.compute_logits(x,gain=2),x*2));w.model_runner.sample_tokens()
        result=call(tool='capture',action='stop')['result']
        self.assertTrue(result['capture_valid']);self.assertTrue(result['sample_join_proven'])
        self.assertEqual(result['steps'][0]['logits'],[0.,2.,4.,6.,8.,10.]);self.assertEqual(result['phase'],'fault')
        self.assertFalse(call(tool='capture',action='status')['result']['installed'])
    def test_d0_missing_ep_is_not_defaulted(self):
        w=self.worker();w.parallel_config=NS(tensor_parallel_size=2)
        result=deployment_identity(w)
        self.assertIsNone(result['parallel_configs'][0]['values']['enable_expert_parallel'])
        self.assertFalse(result['baseline_approved'])
    def test_nonfinite_and_missing_steps_return_evidence_without_changing_result(self):
        w=self.worker();capture(w,'start');x=torch.tensor([[float('inf'),1,2,3,4,5.]])
        self.assertTrue(torch.isinf(w.model_runner.model.compute_logits(x)[0,0]))
        result=capture(w,'stop');self.assertFalse(result['capture_valid']);self.assertTrue(result['capture_removed'])
        self.assertEqual(result['steps'][0]['logits'][0],'+inf')

    def test_dispatch_combine_evidence_is_not_invented_ep_join(self):
        from experiments.worker_core_v11_20261118.dsv4_probe_journal import CallJournal
        from fault_injection.sites.dsv4_observer import empty_report
        lib=torch.library.Library('dcat_probe_test','DEF')
        lib.define('moe_distribute_dispatch_v2(Tensor x, Tensor expert_ids) -> (Tensor, Tensor)')
        lib.define('moe_distribute_combine_v2(Tensor x, Tensor expand_idx) -> Tensor')
        lib.impl('moe_distribute_dispatch_v2',lambda x,expert_ids:(x.clone(),expert_ids.clone()),'CPU')
        lib.impl('moe_distribute_combine_v2',lambda x,expand_idx:x.clone(),'CPU')
        w=self.worker();r=w.model_runner;r.query_start_loc_cpu=[0,2]
        scheduler=NS(num_scheduled_tokens={'r':2})
        state=dict(report=empty_report(),calls=[],dispatches=[],steps=1,max_calls=10,
                   integer_budget_remaining=100,truncated=False,errors=[])
        x=torch.ones(2,4);idx=torch.tensor([1,0])
        with CallJournal(state,r,scheduler,0,{}, {},requests=['r'],phase='prefill'):
            y,j=torch.ops.dcat_probe_test.moe_distribute_dispatch_v2(x,idx)
            z=torch.ops.dcat_probe_test.moe_distribute_combine_v2(y,j)
        self.assertTrue(torch.equal(z,x));self.assertFalse(state['errors'])
        self.assertEqual(len(state['calls']),2)
        combine=state['calls'][1]
        self.assertEqual(combine['dispatch_candidates'],[1])
        self.assertEqual(combine['input_rows']['rows'][1]['token_position'],9)
        self.assertEqual(combine['combine_output_row_join'],'COMBINE_OUTPUT_ROW_JOIN_UNPROVEN')
        self.assertIsNone(combine['valid_output_rows']);self.assertIsNone(combine['padding'])

    def test_observation_failure_preserves_operator_return(self):
        from experiments.worker_core_v11_20261118.dsv4_probe_journal import CallJournal
        from fault_injection.sites.dsv4_observer import empty_report
        w=self.worker();s=NS(num_scheduled_tokens={'r':1})
        state=dict(report=empty_report(),calls=[],dispatches=[],steps=1,max_calls=10,
                   integer_budget_remaining=100,truncated=False,errors=[])
        mode=CallJournal(state,w.model_runner,s,0,{}, {},requests=['r'],phase='decode')
        x=torch.ones(2,2)
        with patch.object(mode,'record',side_effect=RuntimeError('diagnostic failed')):
            with mode:out=x@x
        self.assertTrue(torch.equal(out,torch.full((2,2),2.)))
        self.assertEqual(state['errors'][0]['stage'],'operator_observation')

    def test_quant_weights_do_not_consume_route_index_budget(self):
        from experiments.worker_core_v11_20261118.dsv4_probe_journal import CallJournal
        state=dict(integer_budget_remaining=2,truncated=False)
        mode=CallJournal(state,None,None,0,{}, {},requests=[],phase='decode')
        row=list(mode.tensor(torch.ones(100,dtype=torch.int8),'weight',capture_indices=True))[0]
        self.assertNotIn('values',row);self.assertFalse(state['truncated'])
        row=list(mode.tensor(torch.ones(3,dtype=torch.int32),'route',capture_indices=True))[0]
        self.assertIsNone(row['values']);self.assertTrue(state['truncated'])

    def test_rpc_identity_extract_and_capture_review_pipeline(self):
        import json,tempfile
        from pathlib import Path
        from tools.extract_dsv4_v030_record import pages
        from tools.extract_registration_record import canonical
        from fault_injection.dsv4_noise_v030 import calibrate
        from test_dsv4_noise_v030 import actual
        from test_dsv4_noise import data
        w=self.worker();reply=Dsv4ProbeExtension.demoncat_dsv4_probe(w,tool='identity',action='read')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'identity.json';p.write_text(json.dumps([reply]))
            rows=pages('identity',p);self.assertEqual(rows,pages('identity',p))
            self.assertTrue(all(len(canonical(x))+1<=4096 for x in rows))
        capture(w,'start');x=torch.arange(6.).reshape(1,6)
        for unused in range(4):w.model_runner.model.compute_logits(x);w.model_runner.sample_tokens()
        result=capture(w,'stop');self.assertTrue(result['capture_valid'])
        fixture=data()
        for run in fixture:
            for rank in run['ranks']:
                # Real tool arrays, identical repeated CPU inputs; not V4/NPU evidence.
                run['ranks'][rank]=result['steps']
        summary=calibrate(actual(),fixture)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'noise.json';p.write_text(json.dumps(summary))
            rows=pages('noise',p);self.assertEqual(len(rows),8)
            self.assertTrue(all(len(canonical(x))+1<=4096 for x in rows))

if __name__=='__main__':unittest.main()
