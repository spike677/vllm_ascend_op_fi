"""Automatic binding/lifecycle tests only; fake provider never claims NPU hits."""
import hashlib
import importlib.util
import threading
import unittest
from unittest.mock import patch

from fault_injection.core import UnsupportedCall
from fault_injection.quantization import QuantizationSession, inspect_quantization, PinnedQuantRuntime


class AscendW8A8DynamicLinearMethod:
    def __init__(self):
        self.calls = 0
        self.error = False

    def apply(self, module, x):
        self.calls += 1
        if self.error:
            raise ValueError('original failure')
        return x + module.bias


class Wrapper:
    def __init__(self, inner): self.quant_method=inner
    def apply(self,module,x): return self.quant_method.apply(module,x)


class CPUProvider:
    native=False
    test_only=True
    def __init__(self, model):
        self.model=model; self.arms=[]; self.finishes=[]
        self.mapping_error=False; self.healthy_error=False; self.cleanup_error=False
    @staticmethod
    def input_sha256(x):
        import torch
        return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def context(self, call):
        b=call.input.shape[0]
        return dict(model=self.model,rank=call.module.tp_rank,world_size=2,
                    request_ids=[f'request{i}' for i in range(b)],counts=[1]*b,
                    row_request_indices=[0]*b if self.mapping_error else list(range(b)),
                    row_token_positions=[17]*b,pure_decode=True,healthy_run_id='cpu-fixture-only',
                    baseline_input_sha256='0'*64 if self.healthy_error else self.input_sha256(call.input))
    def arm(self, call, fault): self.arms.append((call.path,fault))
    def finish(self,*,fault,failed=False):
        self.finishes.append((fault,failed))
        if self.cleanup_error:raise RuntimeError('cleanup failed')
        return dict(test_only=True, native_execute_hits=None)


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU torch required')
class QuantAutoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch=torch
        class Unit(torch.nn.Module):
            def __init__(self, method):
                super().__init__()
                self.register_buffer('weight',torch.ones(3,3,dtype=torch.int8))
                self.register_buffer('bias',torch.zeros(3,dtype=torch.float16))
                self.quant_method=method;self.tp_size=2;self.tp_rank=0
            def forward(self,x):return self.quant_method.apply(self,x)
        class Model(torch.nn.Module):
            def __init__(self,shared=False,wrapped=False):
                super().__init__()
                first=AscendW8A8DynamicLinearMethod()
                second=first if shared else AscendW8A8DynamicLinearMethod()
                self.unusual=torch.nn.ModuleDict({'projection_alpha':Unit(Wrapper(first) if wrapped else first),
                                                 'projection_beta':Unit(Wrapper(second) if wrapped else second)})
                self.skip=False
            def forward(self,x):
                x=self.unusual['projection_alpha'](x)
                return x if self.skip else self.unusual['projection_beta'](x)
        cls.Model=Model

    def x(self):return self.torch.ones(2,3,dtype=self.torch.float16)

    def test_discovery_has_no_layer_name_or_model_family_requirement(self):
        model=self.Model(wrapped=True)
        found=inspect_quantization(model)
        self.assertEqual([x['path'] for x in found['candidates']],
                         ['unusual.projection_alpha','unusual.projection_beta'])
        self.assertFalse(found['native_binding_proven'])

    def test_missing_provider_is_not_claimed_supported(self):
        model=self.Model()
        with self.assertRaises(UnsupportedCall),QuantizationSession(model):pass
        self.assertNotIn('apply',vars(model.unusual['projection_alpha'].quant_method))

    def test_every_actual_call_shared_method_and_exact_restoration(self):
        model=self.Model(shared=True); runtime=CPUProvider(model)
        baseline=model(self.x())
        with QuantizationSession(model,runtime=runtime) as session:
            for _ in range(3):self.assertTrue(self.torch.equal(model(self.x()),baseline))
            report=session.assert_coverage()
            self.assertEqual(len(report['events']),6)
            self.assertTrue(all(not e['native_hit_proven'] for e in report['events']))
            self.assertEqual(len(runtime.arms),6)
        method=model.unusual['projection_alpha'].quant_method
        self.assertNotIn('apply',vars(method))
        self.assertTrue(self.torch.equal(model(self.x()),baseline))

    def test_baseline_uses_same_provider_with_fault_disabled(self):
        model=self.Model(); runtime=CPUProvider(model)
        with QuantizationSession(model,arm='baseline',runtime=runtime):model(self.x())
        self.assertEqual([fault for _,fault in runtime.arms],[False,False])

    def test_outside_model_and_other_model_are_not_armed(self):
        model=self.Model(shared=True);other=self.Model()
        shared=model.unusual['projection_alpha'].quant_method
        other.unusual['projection_alpha'].quant_method=shared
        runtime=CPUProvider(model)
        with QuantizationSession(model,runtime=runtime) as session:
            model.unusual['projection_alpha'](self.x())
            other(self.x())
            self.assertEqual(runtime.arms,[])
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_scheduler_mapping_and_healthy_input_are_checked_before_arming(self):
        for field in ('mapping_error','healthy_error'):
            model=self.Model();runtime=CPUProvider(model);setattr(runtime,field,True)
            with QuantizationSession(model,runtime=runtime) as session:
                with self.assertRaises(UnsupportedCall):model(self.x())
                self.assertEqual(runtime.arms,[])
                self.assertEqual(session.report()['events'][0]['status'],'unsupported')

    def test_invalid_tensor_geometry_passthrough_is_not_coverage(self):
        model=self.Model();runtime=CPUProvider(model)
        with QuantizationSession(model,unsupported='passthrough',runtime=runtime) as session:
            model(self.torch.ones(2,2,3,dtype=self.torch.float16))
            self.assertTrue(all(e['status']=='passthrough' for e in session.events))
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        self.assertEqual(runtime.arms,[])

    def test_missing_branch_in_later_forward_stays_uncovered(self):
        model=self.Model();runtime=CPUProvider(model)
        with QuantizationSession(model,runtime=runtime) as session:
            model(self.x());model.skip=True;model(self.x())
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_original_failure_attempts_cleanup_and_restores_binding(self):
        model=self.Model();runtime=CPUProvider(model)
        method=model.unusual['projection_alpha'].quant_method;method.error=True
        with self.assertRaisesRegex(ValueError,'original failure'),QuantizationSession(model,runtime=runtime):
            model(self.x())
        self.assertEqual(runtime.finishes,[(True,True)])
        self.assertNotIn('apply',vars(method))

    def test_cleanup_failure_preserves_original_error(self):
        model=self.Model();runtime=CPUProvider(model);runtime.cleanup_error=True
        model.unusual['projection_alpha'].quant_method.error=True
        with self.assertRaisesRegex(RuntimeError,'original failure.*cleanup failed'),QuantizationSession(model,runtime=runtime):
            model(self.x())

    def test_runtime_weight_change_rejected(self):
        model=self.Model();runtime=CPUProvider(model)
        with QuantizationSession(model,runtime=runtime):
            model.unusual['projection_alpha'].weight=model.unusual['projection_alpha'].weight.float()
            with self.assertRaises(UnsupportedCall):model(self.x())
        self.assertEqual(runtime.arms,[])

    def test_install_failure_rolls_back_and_releases_lease(self):
        model=self.Model();runtime=CPUProvider(model)
        method=model.unusual['projection_alpha'].quant_method
        original=model.register_forward_hook
        def fail(*args,**kwargs):raise RuntimeError('hook install failure')
        model.register_forward_hook=fail
        with self.assertRaisesRegex(RuntimeError,'hook install failure'),QuantizationSession(model,runtime=runtime):pass
        self.assertNotIn('apply',vars(method))
        self.assertEqual(len(model._forward_pre_hooks),0)
        model.register_forward_hook=original
        with QuantizationSession(model,runtime=runtime):model(self.x())

    def test_provider_declaration_cannot_claim_native_proof(self):
        model=self.Model();runtime=CPUProvider(model)
        runtime.native=True;runtime.test_only=False
        with patch('fault_injection.quantization.threading.active_count',return_value=1):
            with QuantizationSession(model,runtime=runtime) as session:
                model(self.x())
                self.assertTrue(all(e['provider_native_declared'] for e in session.events))
                self.assertTrue(all(not e['native_execution_proven'] and not e['native_hit_proven'] for e in session.events))

    def test_unverified_direct_native_constructor_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedCall,'verified native loader'):
            PinnedQuantRuntime(object(),lambda call:{}, {})

    def test_tp1_requires_explicit_deployment_admission(self):
        from fault_injection.quantization import _discover
        model=self.Model()
        for m in model.unusual.values():m.tp_size=1;m.tp_rank=0
        self.assertEqual(_discover(model)[0],[])
        self.assertEqual(len(_discover(model,(1,))[0]),2)
        with self.assertRaises(UnsupportedCall),QuantizationSession(model,runtime=CPUProvider(model)):pass

    def test_tp1_row_mapping_does_not_inherit_tp2_or_foreign_rank(self):
        from fault_injection.quantization import QuantizationCall,_validate_rows
        model=self.Model();module=model.unusual['projection_alpha'];module.tp_size=1
        call=QuantizationCall('automatic',module,self.x(),model,0)
        context=CPUProvider(model).context(call);context['world_size']=1
        with self.assertRaises(UnsupportedCall):_validate_rows(call,context)
        self.assertEqual(_validate_rows(call,context,(1,))['rank'],0)
        for size,rank in ((2,0),(1,1),(1,True)):
            with self.subTest(size=size,rank=rank),self.assertRaises(UnsupportedCall):
                _validate_rows(call,dict(context,world_size=size,rank=rank),(1,))

    def test_partial_arm_failure_cleans_up_and_poison_prevents_rearm(self):
        model=self.Model();runtime=CPUProvider(model)
        original_arm=runtime.arm
        def half_arm(call,fault):
            original_arm(call,fault)
            raise ValueError('arm failed after changing native state')
        runtime.arm=half_arm
        with QuantizationSession(model,runtime=runtime) as session:
            with self.assertRaisesRegex(ValueError,'arm failed'):model(self.x())
            self.assertEqual(runtime.finishes,[(True,True)])
            self.assertTrue(session.report()['session_poisoned'])
            with self.assertRaisesRegex(RuntimeError,'poisoned'):model(self.x())
            self.assertEqual(len(runtime.arms),1)

    def test_partial_arm_cleanup_failure_preserves_both_errors(self):
        model=self.Model();runtime=CPUProvider(model);runtime.cleanup_error=True
        def half_arm(call,fault):raise ValueError('partial arm')
        runtime.arm=half_arm
        with QuantizationSession(model,runtime=runtime) as session:
            with self.assertRaisesRegex(RuntimeError,'partial arm.*cleanup failed'):model(self.x())
            self.assertTrue(session.report()['session_poisoned'])

    def test_recursive_rejection_does_not_disarm_outer_forward(self):
        model=self.Model();runtime=CPUProvider(model)
        original=model.forward
        def catches_recursion(x,nested=False):
            if not nested:
                try:model(x,nested=True)
                except UnsupportedCall:pass
            return original(x)
        model.forward=catches_recursion
        with QuantizationSession(model,runtime=runtime) as session:
            model(self.x())
            self.assertEqual(len(runtime.arms),2)
            report=session.report()
            self.assertEqual(report['forwards'],[dict(index=0,status='returned')])
            self.assertEqual(len(report['rejected_root_calls']),1)
            self.assertFalse(session._local.inside)
            self.assertFalse(session._root_entries)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_failure_after_all_quant_calls_does_not_claim_full_coverage(self):
        model=self.Model();runtime=CPUProvider(model)
        original=model.forward
        def fail_after_quant(x):
            original(x)
            raise ValueError('after quantization')
        model.forward=fail_after_quant
        with QuantizationSession(model,runtime=runtime) as session:
            with self.assertRaisesRegex(ValueError,'after quantization'):model(self.x())
            self.assertEqual(len(runtime.arms),2)
            self.assertTrue(all(e['status']=='routed' for e in session.events))
            self.assertEqual(session.report()['forwards'][0]['status'],'no_output_or_exception')
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_raising_forward_hook_with_output_marks_failed_forward(self):
        model=self.Model();runtime=CPUProvider(model)
        def fail(module,args,output):raise ValueError('hook after output')
        handle=model.register_forward_hook(fail)
        try:
            with QuantizationSession(model,runtime=runtime) as session:
                with self.assertRaisesRegex(ValueError,'hook after output'):model(self.x())
                self.assertEqual(len(runtime.arms),2)
                self.assertEqual(session.report()['forwards'][0]['status'],'no_output_or_exception')
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:handle.remove()

    def test_existing_pre_hook_failure_cannot_reuse_previous_forward_coverage(self):
        model=self.Model();runtime=CPUProvider(model)
        fail=[False]
        def earlier_hook(module,args):
            if fail[0]:raise ValueError('earlier pre hook')
        handle=model.register_forward_pre_hook(earlier_hook)
        try:
            with QuantizationSession(model,runtime=runtime) as session:
                model(self.x());session.assert_coverage();fail[0]=True
                with self.assertRaisesRegex(ValueError,'earlier pre hook'):model(self.x())
                self.assertEqual(len(runtime.arms),2)
                self.assertEqual([f['status'] for f in session.report()['forwards']],['returned','no_output_or_exception'])
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:handle.remove()

    def test_concurrent_root_is_rejected_and_owner_can_continue(self):
        model=self.Model();runtime=CPUProvider(model);errors=[]
        def concurrent():
            try:model(self.x())
            except BaseException as error:errors.append(error)
        with QuantizationSession(model,runtime=runtime) as session:
            thread=threading.Thread(target=concurrent)
            thread.start();thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors),1)
            self.assertIsInstance(errors[0],UnsupportedCall)
            self.assertEqual(runtime.arms,[])
            model(self.x())
            self.assertEqual(len(runtime.arms),2)
            self.assertEqual(len(session.report()['rejected_root_calls']),1)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_existing_exception_context_does_not_invalidate_successful_forward(self):
        model=self.Model();runtime=CPUProvider(model)
        with QuantizationSession(model,runtime=runtime) as session:
            try:raise ValueError('already handled')
            except ValueError:
                model(self.x())
                session.assert_coverage()
        self.assertEqual(session.report()['forwards'][0]['status'],'returned')


if __name__=='__main__':unittest.main()
