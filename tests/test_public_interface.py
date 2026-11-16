"""Public model-only selection, provider binding and lifecycle regression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from fault_injection import (attach, inspect_model, register_runtime, unregister_runtime,
                             registered_runtimes, UnsupportedCall)
from fault_injection.fia import FiaBackend


class PublicInterface(unittest.TestCase):
    def test_public_module_selection_and_persisted_scope_are_not_msprobe(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8)).eval()
        x=torch.randn(2,8); baseline=model(x)
        with tempfile.TemporaryDirectory() as directory:
            for arm in ('baseline','fault'):
                with attach(model,'18.api_half',arm=arm,evidence_dir=directory) as session:
                    result=model(x)
                    self.assertTrue(torch.equal(result,baseline*(.5 if arm=='fault' else 1)))
                    report=session.assert_coverage()
                    self.assertEqual(report['fault'],'18.api_half')
                    self.assertFalse(report['npu_validation_claimed'])
                self.assertTrue(torch.equal(model(x),baseline))
            paths=list(Path(directory).glob('session-*.json'))
            self.assertEqual(len(paths),2)
            for p in paths:
                receipt=json.loads(p.read_text(encoding='utf-8'))
                self.assertTrue(receipt['closed'])
                self.assertIn('not msprobe',receipt['report_scope'])

    def test_structure_discovery_handles_non_attention_model_without_hooks(self):
        model=torch.nn.Sequential(torch.nn.Linear(8,8),torch.nn.RMSNorm(8)).eval()
        x=torch.randn(2,8); before=model(x)
        result=inspect_model(model)
        self.assertEqual(result['module_candidates']['18.api_half'][0]['path'],'1')
        self.assertEqual(result['linear_candidates'][0]['path'],'0')
        self.assertFalse(result['native']['candidates'][0]['native_slot_verified'])
        self.assertFalse(model._forward_hooks)
        self.assertTrue(torch.equal(before,model(x)))
        json.dumps(result)

    def test_native_provider_factory_binds_actual_model_but_no_callbacks_is_not_pass(self):
        model=torch.nn.Linear(4,4).eval(); factory_models=[]
        class Runtime:
            def install(self,*args):return lambda:None
            def check(self,*args):raise AssertionError('No native call expected')
            def execute(self,*args):raise AssertionError('No native call expected')
            def finish(self,*args):raise AssertionError('No native call expected')
            def abort(self,*args):raise AssertionError('No native call expected')
        def factory(actual):
            factory_models.append(actual)
            return Runtime()
        register_runtime('20.buffer_substitute',factory)
        try:
            self.assertIn('20.buffer_substitute',registered_runtimes())
            with self.assertRaises(ValueError):register_runtime('20.buffer_substitute',factory)
            with self.assertRaises(ValueError):unregister_runtime('20.buffer_substitute',lambda:None)
            with attach(model,'20.buffer_substitute') as session:
                model(torch.ones(1,4))
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
                self.assertFalse(session.report()['native_fault_hit_proven'])
            self.assertEqual(factory_models,[model])
        finally:unregister_runtime('20.buffer_substitute',factory)
        with self.assertRaises(UnsupportedCall):
            with attach(model,'20.buffer_substitute'):pass

    def test_registry_does_not_accept_ignored_or_evidence_only_provider(self):
        for fault in ('28.fia.denominator_replay','8.restore_offset','18.api_half'):
            with self.assertRaises(ValueError):register_runtime(fault,lambda m:object())
        for fault in ('8.restore_offset','22.wrong_row','25.reduce_scatter','41.tail_column'):
            with self.assertRaises(UnsupportedCall):attach(torch.nn.Identity(),fault)

    def test_registered_quant_runtime_is_selected_without_layer_paths(self):
        from test_quant_auto import CPUProvider, AscendW8A8DynamicLinearMethod
        class Unit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('weight',torch.ones(3,3,dtype=torch.int8))
                self.register_buffer('bias',torch.zeros(3,dtype=torch.float16))
                self.quant_method=AscendW8A8DynamicLinearMethod()
                self.tp_size,self.tp_rank=2,0
            def forward(self,x):return self.quant_method.apply(self,x)
        model=torch.nn.Sequential(Unit(),Unit()).eval(); runtimes=[]
        def factory(actual):
            runtime=CPUProvider(actual);runtimes.append(runtime);return runtime
        register_runtime('quant.batch_shared_scale',factory)
        try:
            with attach(model,'quant.batch_shared_scale') as session:
                model(torch.ones(2,3,dtype=torch.float16))
                report=session.assert_coverage()
                self.assertEqual(len(report['events']),2)
                self.assertFalse(report['npu_validation_claimed'])
                self.assertTrue(all(not e['native_hit_proven'] for e in report['events']))
            self.assertIs(runtimes[0].model,model)
            self.assertEqual(len(runtimes[0].arms),2)
        finally:unregister_runtime('quant.batch_shared_scale',factory)

    def test_global_session_rejection_does_not_break_outer_restore(self):
        model=torch.nn.RMSNorm(8).eval(); x=torch.ones(1,8); before=model(x)
        with attach(model,'18.api_half') as outer:
            with self.assertRaises(RuntimeError):
                with attach(model,'18.api_half'):pass
            self.assertTrue(torch.equal(model(x),before*.5))
            outer.assert_coverage()
        self.assertTrue(torch.equal(model(x),before))
        with attach(model,'18.api_half',arm='baseline') as session:
            model(x);session.assert_coverage()

    def test_failed_model_saved_and_restored_without_success_upgrade(self):
        class Failing(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.RMSNorm(8)
            def forward(self,x):
                self.norm(x);raise RuntimeError('expected model failure')
        model=Failing().eval()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'expected model failure'):
                with attach(model,'18.api_half',evidence_dir=directory) as session:model(torch.ones(1,8))
            receipt=json.loads(next(Path(directory).glob('session-*.json')).read_text(encoding='utf-8'))
            self.assertIn('expected model failure',receipt['session_error'])
            with self.assertRaises((UnsupportedCall,RuntimeError)):session.assert_coverage()
        self.assertFalse(model._forward_hooks)

    def test_legal_fia_plan_never_uses_replay_and_rejects_wrong_host_key(self):
        import os
        for arm,m,key in (('baseline',512,'109000000000000000'),('fault',128,'109000000000400000')):
            backend=FiaBackend(arm=arm,mechanism='28.fia.legal_tiling')
            mode,actual_m,actual_key=backend.native_plan()
            self.assertNotIn('replay',mode)
            self.assertEqual((actual_m,actual_key),(m,key))
            row=dict(token='x',owner_pid=os.getpid(),owner_starttime_ticks='123',mode=mode,
                     eligible=True,selected_m=m,tiling_key=key,serializer_roundtrip_matches=True,checks={'shape':True})
            backend._host_receipt([row],'x','123')
            row['tiling_key']='119000000000400000'
            with self.assertRaises(UnsupportedCall):backend._host_receipt([row],'x','123')

    def test_wrong_thread_close_can_be_retried_by_owner(self):
        import threading
        errors=[]; model=torch.nn.RMSNorm(8).eval(); x=torch.ones(1,8); before=model(x)
        with attach(model,'18.api_half') as session:
            def wrong_thread():
                try:session.close()
                except RuntimeError as exc:errors.append(str(exc))
            worker=threading.Thread(target=wrong_thread);worker.start();worker.join()
            self.assertEqual(len(errors),1)
            self.assertFalse(session.report()['closed'])
            self.assertTrue(torch.equal(model(x),before*.5))
        self.assertTrue(torch.equal(model(x),before))

    def test_report_failure_preserves_model_error_and_restores(self):
        from fault_injection.session import ModelFaultSession
        model=torch.nn.RMSNorm(8).eval(); x=torch.ones(1,8); before=model(x)
        with self.assertRaisesRegex(RuntimeError,'model exploded.*report unavailable'):
            with patch.object(ModelFaultSession,'_save_report',side_effect=OSError('report unavailable')):
                with attach(model,'18.api_half',evidence_dir='unused'):
                    model(x)
                    raise ValueError('model exploded')
        self.assertTrue(torch.equal(model(x),before))

    def test_nonfinite_provider_report_never_creates_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model=torch.nn.RMSNorm(8).eval()
            with self.assertRaises(ValueError):
                with attach(model,'18.api_half',evidence_dir=directory) as session:
                    model(torch.ones(1,8))
                    original=session.adapter.report
                    session.adapter.report=lambda:dict(original(),untrusted_provider_value=float('nan'))
            self.assertEqual(list(Path(directory).iterdir()),[])
            self.assertFalse(model._forward_hooks)


if __name__=='__main__':unittest.main()
