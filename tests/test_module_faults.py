import unittest
import torch
from fault_injection.module_faults import ModuleFaultSession
from fault_injection.core import UnsupportedCall


class Patches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unusual=torch.nn.Conv3d(3,2,(2,16,16),stride=(2,16,16))
        self.elsewhere=torch.nn.Conv3d(3,4,(2,16,16),stride=(2,16,16))
    def forward(self,x):return self.unusual(x),self.elsewhere(x)


class ModuleFaults(unittest.TestCase):
    def test_conv_real_native_padding_all_modules_and_restore(self):
        model=Patches().eval()
        x=torch.arange(3*2*16*16,dtype=torch.float32).reshape(1,3,2,16,16)/100
        before=dict(zip(('unusual','elsewhere'),model(x)))
        weights={k:m.weight.detach().clone() for k,m in model.named_children()}
        with ModuleFaultSession(model,'17.spatial_padding') as s:
            for _ in range(2):
                for (key,m),out in zip(model.named_children(),model(x)):
                    self.assertEqual(out.shape,before[key].shape)
                    self.assertFalse(torch.equal(out,before[key]))
                    expected=torch.ops.aten.convolution.default(x,m.weight,m.bias,[2,16,16],[0,7,7],[1,1,1],False,[0,0,0],1)
                    self.assertTrue(torch.equal(out,expected))
            self.assertEqual(s.assert_coverage()['coverage'],{'unusual':2,'elsewhere':2})
        for key,m in model.named_children():
            self.assertTrue(torch.equal(m(x),before[key]))
            self.assertTrue(torch.equal(weights[key],m.weight))

    def test_conv_reject_and_exception_restore(self):
        model=torch.nn.Conv3d(1,1,3).eval(); original=torch.nn.functional.conv3d
        with self.assertRaises(UnsupportedCall):
            with ModuleFaultSession(model,'17.spatial_padding'):
                model(torch.ones(1,1,5,5,5))
        self.assertIs(torch.nn.functional.conv3d,original)
        self.assertFalse(model._forward_hooks)

    def test_conv_legal_geometry_varies_batch_channels_and_temporal_kernel(self):
        for batch,inputs,outputs,temporal in ((3,5,7,1),(3,2,4,3)):
            with self.subTest(batch=batch,inputs=inputs,outputs=outputs,temporal=temporal):
                kernel=(temporal,16,16)
                model=torch.nn.Conv3d(inputs,outputs,kernel,stride=kernel).eval()
                x=torch.arange(batch*inputs*temporal*16*16,dtype=torch.float32).reshape(
                    batch,inputs,temporal,16,16)/1000
                baseline=model(x)
                weights=model.weight.detach().clone()
                expected=torch.ops.aten.convolution.default(x,model.weight,model.bias,
                    list(kernel),[0,7,7],[1,1,1],False,[0,0,0],1)
                with ModuleFaultSession(model,'17.spatial_padding') as session:
                    actual=model(x)
                    self.assertEqual(tuple(actual.shape),(batch,outputs,1,1,1))
                    self.assertEqual(actual.shape,baseline.shape)
                    self.assertTrue(torch.equal(actual,expected))
                    self.assertFalse(torch.equal(actual,baseline))
                    self.assertEqual(session.assert_coverage()['coverage'],{'<root>':1})
                self.assertTrue(torch.equal(model.weight,weights))
                self.assertTrue(torch.equal(model(x),baseline))

    def test_no_false_zero_coverage_and_no_other_model_effect(self):
        a=torch.nn.Conv3d(1,1,(1,16,16),stride=(1,16,16)).eval()
        b=torch.nn.Conv3d(1,1,(1,16,16),stride=(1,16,16)).eval()
        x=torch.ones(1,1,1,16,16); before=b(x)
        with ModuleFaultSession(a,'17.spatial_padding') as s:
            self.assertTrue(torch.equal(b(x),before))
            with self.assertRaises(UnsupportedCall):s.assert_coverage()

    def test_rms_return_gain_baseline_and_off(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8),torch.nn.RMSNorm(8)).eval()
        x=torch.randn(2,8); baseline=model(x)
        with ModuleFaultSession(model,'18.api_half',arm='baseline') as s:
            self.assertTrue(torch.equal(model(x),baseline));s.assert_coverage()
        with ModuleFaultSession(model,'18.api_half') as s:
            y=model(x)
            first=torch.nn.functional.rms_norm(x,(8,),model[0].weight,model[0].eps)*.5
            expected=torch.nn.functional.rms_norm(first,(8,),model[1].weight,model[1].eps)*.5
            self.assertTrue(torch.equal(y,expected));s.assert_coverage()
        self.assertTrue(torch.equal(model(x),baseline))

    def test_shared_module_in_other_model_does_not_match(self):
        shared=torch.nn.RMSNorm(8)
        a=torch.nn.Sequential(shared).eval(); b=torch.nn.Sequential(shared).eval()
        x=torch.randn(2,8); expected=b(x)
        with ModuleFaultSession(a,'18.api_half') as s:
            self.assertTrue(torch.equal(b(x),expected))
            with self.assertRaises(UnsupportedCall):s.assert_coverage()
            self.assertTrue(torch.equal(a(x),expected*.5));s.assert_coverage()

    def test_later_forward_skipping_candidate_is_not_full_coverage(self):
        class Conditional(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first=torch.nn.RMSNorm(8)
                self.second=torch.nn.RMSNorm(8)
            def forward(self,x,skip=False):
                x=self.first(x)
                return x if skip else self.second(x)
        model=Conditional().eval();x=torch.randn(2,8)
        with ModuleFaultSession(model,'18.api_half') as session:
            model(x)
            session.assert_coverage()
            model(x,skip=True)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
            rows=session.report()['forwards']
            self.assertTrue(rows[0]['all_discovered_candidates_covered'])
            self.assertFalse(rows[1]['all_discovered_candidates_covered'])
            self.assertEqual(rows[1]['coverage'],{'first':1,'second':0})

    def test_recursive_rejection_does_not_clear_outer_scope(self):
        class CatchesRecursion(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.RMSNorm(8)
            def forward(self,x,nested=False):
                if not nested:
                    try:self(x,nested=True)
                    except UnsupportedCall:pass
                return self.norm(x)
        model=CatchesRecursion().eval();x=torch.randn(2,8)
        expected=model.norm(x)*.5
        with ModuleFaultSession(model,'18.api_half') as session:
            # The nested root pre-hook fails, but outer model computation must
            # retain its original scope and still route the subsequent norm.
            self.assertTrue(torch.equal(model(x),expected))
            report=session.report()
            self.assertEqual(report['coverage'],{'norm':1})
            self.assertEqual(len(report['forwards']),1)
            self.assertEqual(len(report['rejected_root_calls']),1)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_failed_forward_cannot_reuse_previous_coverage(self):
        class FailsAfterNorm(torch.nn.Module):
            def __init__(self):super().__init__();self.norm=torch.nn.RMSNorm(8)
            def forward(self,x,fail=False):
                result=self.norm(x)
                if fail:raise ValueError('intentional model failure')
                return result
        model=FailsAfterNorm().eval();x=torch.randn(2,8)
        with ModuleFaultSession(model,'18.api_half') as session:
            model(x)
            with self.assertRaises(ValueError):model(x,fail=True)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
            self.assertEqual(session.report()['forwards'][-1]['status'],'no_output_or_exception')

    def test_unknown_custom_forward_is_not_selected_by_class_name(self):
        class CustomRMSNorm(torch.nn.RMSNorm):
            def forward(self,x):return x
        with self.assertRaises(UnsupportedCall):
            with ModuleFaultSession(torch.nn.Sequential(CustomRMSNorm(8)).eval(),'18.api_half'):pass

    def test_instance_monkeypatch_and_custom_conv_implementation_rejected(self):
        from types import MethodType
        norm=torch.nn.RMSNorm(8).eval()
        norm.forward=MethodType(lambda self,x:x,norm)
        with self.assertRaises(UnsupportedCall):
            with ModuleFaultSession(norm,'18.api_half'):pass
        class CustomConv(torch.nn.Conv3d):
            def _conv_forward(self,input,weight,bias):return input
        conv=CustomConv(1,1,(1,16,16),stride=(1,16,16)).eval()
        with self.assertRaises(UnsupportedCall):
            with ModuleFaultSession(conv,'17.spatial_padding'):pass

    def test_cleanup_aggregates_handle_errors_and_restores_remaining_resources(self):
        model=Patches().eval()
        original=torch.nn.functional.conv3d
        session=ModuleFaultSession(model,'17.spatial_padding')
        session.__enter__()
        class RaisingHandle:
            def __init__(self,handle,message):self.handle,self.message,self.first=handle,message,True
            def remove(self):
                self.handle.remove()
                if self.first:
                    self.first=False
                    raise RuntimeError(self.message)
        session.handles[0]=RaisingHandle(session.handles[0],'first cleanup fixture')
        session.handles[-1]=RaisingHandle(session.handles[-1],'last cleanup fixture')
        with self.assertRaisesRegex(RuntimeError,'last cleanup fixture.*first cleanup fixture'):
            session.close()
        self.assertIs(torch.nn.functional.conv3d,original)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for _,m in model.named_modules()))
        self.assertFalse(session.active)
        self.assertFalse(session.closed)
        self.assertEqual(len(session.report()['cleanup_errors']),2)
        with self.assertRaisesRegex(RuntimeError,'session is active'):
            ModuleFaultSession(Patches().eval(),'17.spatial_padding').__enter__()
        session.close()
        self.assertTrue(session.closed)
        with ModuleFaultSession(Patches().eval(),'17.spatial_padding'):pass

    def test_cleanup_error_preserves_body_exception(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8)).eval()
        with self.assertRaisesRegex(RuntimeError,'original body failure.*cleanup fixture'):
            with ModuleFaultSession(model,'18.api_half') as session:
                handle=session.handles[0]
                class RaisingHandle:
                    first=True
                    def remove(self):
                        handle.remove()
                        if self.first:
                            self.first=False
                            raise RuntimeError('cleanup fixture')
                session.handles[0]=RaisingHandle()
                raise ValueError('original body failure')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for _,m in model.named_modules()))
        self.assertFalse(session.closed)
        session.close()

    def test_raising_forward_hook_with_output_is_a_failed_forward(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8)).eval()
        def fail_after_output(module,args,output):
            raise ValueError('post output failure')
        handle=model.register_forward_hook(fail_after_output)
        try:
            with ModuleFaultSession(model,'18.api_half') as session:
                with self.assertRaisesRegex(ValueError,'post output failure'):model(torch.randn(2,8))
                self.assertEqual(session.report()['coverage'],{'0':1})
                self.assertEqual(session.report()['forwards'][0]['status'],'no_output_or_exception')
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:handle.remove()

    def test_existing_exception_context_is_not_a_failed_forward(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8)).eval()
        with ModuleFaultSession(model,'18.api_half') as session:
            try:raise ValueError('already handled')
            except ValueError:
                model(torch.randn(2,8))
                session.assert_coverage()
        self.assertEqual(session.report()['forwards'][0]['status'],'returned')

    def test_existing_pre_hook_failure_does_not_reuse_previous_coverage(self):
        model=torch.nn.Sequential(torch.nn.RMSNorm(8)).eval();fail=[False]
        def earlier_hook(module,args):
            if fail[0]:raise ValueError('earlier pre hook')
        handle=model.register_forward_pre_hook(earlier_hook)
        try:
            with ModuleFaultSession(model,'18.api_half') as session:
                model(torch.randn(2,8));session.assert_coverage();fail[0]=True
                with self.assertRaisesRegex(ValueError,'earlier pre hook'):model(torch.randn(2,8))
                self.assertEqual([f['status'] for f in session.report()['forwards']],['returned','no_output_or_exception'])
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:handle.remove()


if __name__=='__main__':unittest.main()
