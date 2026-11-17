"""CPU contracts only; actual pretrained/NPU qualification stays separate."""
import unittest
from unittest.mock import patch
import torch

from fault_injection.conv3d_natural import PatchConv3dSession
from fault_injection.core import UnsupportedCall
from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_natural import NaturalTensorGroupProvider


class ImagePath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch=torch.nn.Conv3d(3,4,(2,16,16),stride=(2,16,16))
        self.second=torch.nn.Conv3d(3,4,(2,16,16),stride=(2,16,16))
    def forward(self,x):
        return self.patch(x).view(x.shape[0],4)+self.second(x).view(x.shape[0],4)


class NaturalConv3dTests(unittest.TestCase):
    def inputs(self):return torch.arange(2*3*2*16*16.).reshape(2,3,2,16,16)/1000
    def provider(self,model,x):
        return NaturalTensorGroupProvider(model.eval(),'17.spatial_padding',replay=lambda m,phase:m(x),
            synchronize=lambda:None,simulation=True,workload_sha256='a'*64,
            environment_sha256='b'*64,model_weights_sha256='c'*64,model_source_sha256='d'*64)
    def test_real_native_arguments_all_targets_c2_c3_and_exact_off(self):
        model=ImagePath().eval();x=self.inputs();baseline=model(x).detach().clone()
        provider=self.provider(model,x)
        self.assertEqual(provider.qualify()['candidate_set']['eligible'],2)
        c2=provider.run();self.assertTrue(c2['contract_satisfied'],c2)
        self.assertEqual(c2['hits'],2)
        proof=provider.group.c2_proof()
        c3=provider.run('every_match',forwards=2,c2_proof=proof)
        self.assertTrue(c3['contract_satisfied'],c3);self.assertEqual(c3['hits'],4)
        self.assertTrue(torch.equal(model(x),baseline))
        for phase in c3['phase_rows']:
            for event in (phase['mechanism'] or {}).get('events',[]):
                self.assertEqual(event['native_calls'],1)
                self.assertEqual(event['padding'],[0,7,7])
                self.assertTrue(event['inputs_unchanged']);self.assertTrue(event['consumers'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertFalse(c3['public_qualification_granted'])
    def test_kernel3_and_patch14_are_refused_without_output_repair(self):
        for kernel in ((3,3,3),(2,14,14)):
            model=torch.nn.Sequential(torch.nn.Conv3d(3,4,kernel,stride=kernel)).eval()
            with self.subTest(kernel=kernel),self.assertRaisesRegex(UnsupportedCall,'ORIGINAL_ZERO_PADDED'):
                with PatchConv3dSession(model,simulation=True,arm='baseline'):
                    model(torch.ones(1,3,*kernel))
    def test_unobserved_structural_conv_cannot_reduce_denominator(self):
        model=ImagePath().eval();model.unused=torch.nn.Conv3d(3,4,(2,16,16),stride=(2,16,16))
        with self.assertRaisesRegex(RuntimeError,'STRUCTURAL_DENOMINATOR_INCOMPLETE'):
            self.provider(model,self.inputs()).qualify()
    def test_changed_fault_input_cannot_borrow_healthy_output_comparison(self):
        model=ImagePath().eval();x=self.inputs()
        with PatchConv3dSession(model,simulation=True,arm='baseline') as healthy:model(x)
        with self.assertRaisesRegex(UnsupportedCall,'SAME_ACTUAL_CONV_INPUT'):
            with PatchConv3dSession(model,simulation=True,references=healthy.healthy_calls):model(x+1)
    def test_statistics_pause_excludes_snapshots_and_keeps_native_module_return(self):
        from test_tensor_statistics_window import GuardWindow
        model=ImagePath().eval();provider=self.provider(model,self.inputs())
        provider.qualify();provider.run();capture=GuardWindow();native_returns=[]
        original_cpu=PatchConv3dSession.cpu
        def checked_cpu(session,value):
            self.assertFalse(capture.active,'Conv diagnostics leaked into raw capture')
            return original_cpu(session,value)
        handle=model.patch.register_forward_hook(lambda *args:native_returns.append(capture.active))
        try:
            with patch.object(PatchConv3dSession,'cpu',checked_cpu):
                result=provider.collect_statistics_pair(capture)
        finally:handle.remove()
        self.assertEqual(native_returns,[True,True,True,False])
        self.assertTrue(result['baseline_exact'] and result['off_exact'])


if __name__ == '__main__':unittest.main()
