"""CPU execution evidence only; no actual Worker/NPU/ACLNN qualification."""
import unittest
import torch
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack

from fault_injection.vision17_trace import Vision17Trace
from fault_injection.vision17_padding import Vision17PaddingSession


class CustomPatch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.arange(2*3*2*16*16.).reshape(2,3,2,16,16)/10000)
        self.bias=torch.nn.Parameter(torch.ones(2))
        self.stride=(2,16,16);self.padding=(0,0,0);self.dilation=(1,1,1)
        self.groups=1;self.padding_mode='zeros'
    def forward(self,x):
        return torch.nn.functional.conv3d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.unusual_name=CustomPatch();self.ignore_output=False;self.fail=False;self.unknown_reader=False
    def forward(self,x):
        projected=self.unusual_name(x)
        if self.fail:raise RuntimeError('actual encoder failure')
        if self.ignore_output:return torch.zeros_like(projected)+3
        if self.unknown_reader:projected.relu_()
        return projected.view(x.shape[0],2).clone()+3


class VisionTraceTests(unittest.TestCase):
    def setUp(self):
        self.model=Encoder().eval();self.x=torch.arange(2*3*2*16*16.).reshape(2,3,2,16,16)/1000
    def tearDown(self):
        self.assertEqual(_get_current_dispatch_mode_stack(),[])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def test_independent_lower_trace_observes_fault_arguments_and_real_consumer(self):
        with Vision17Trace(self.model,simulation=True) as baseline:
            with Vision17PaddingSession(self.model,simulation=True,arm='baseline') as adapter:
                before=self.model(self.x)
        with Vision17Trace(self.model,simulation=True) as fault:
            with Vision17PaddingSession(self.model,simulation=True,references=adapter.healthy_calls):
                after=self.model(self.x)
        b,f=baseline.report(),fault.report()
        self.assertTrue(b['torch_dispatch_chain_complete'] and f['torch_dispatch_chain_complete'])
        self.assertEqual(b['observed_calls'][0]['actual_geometry']['padding'],[0,0,0])
        self.assertEqual(f['observed_calls'][0]['actual_geometry']['padding'],[0,7,7])
        self.assertFalse(torch.equal(before,after))
        self.assertEqual(f['observed_calls'][0]['consumer']['operator'],'aten.add.Tensor')
        self.assertEqual(len(f['observed_calls'][0]['lineage_edges']),2)
        self.assertEqual(f['held_lineage_tensors'],0)
        self.assertFalse(f['native_qualification']);self.assertFalse(f['aclnn_launch_arguments'])
        self.assertEqual(f['qualified_targets'],[])
    def test_equal_shaped_unrelated_tensor_is_not_a_consumer(self):
        self.model.ignore_output=True
        with Vision17Trace(self.model,simulation=True) as trace:self.model(self.x)
        report=trace.report()
        self.assertFalse(report['torch_dispatch_chain_complete'])
        self.assertIsNone(report['observed_calls'][0]['consumer'])
    def test_failed_actual_encoder_cannot_claim_completed_scope(self):
        self.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'actual encoder failure'):
            with Vision17Trace(self.model,simulation=True) as trace:self.model(self.x)
        self.assertFalse(trace.report()['torch_dispatch_chain_complete'])
        self.assertFalse(trace.report()['forwards'][0]['returned'])
    def test_unknown_first_reader_cannot_be_skipped_for_later_add(self):
        self.model.unknown_reader=True
        with Vision17Trace(self.model,simulation=True) as trace:self.model(self.x)
        report=trace.report();consumer=report['observed_calls'][0]['consumer']
        self.assertEqual(consumer['operator'],'aten.relu_.default')
        self.assertFalse(consumer['supported_numerical_consumer'])
        self.assertFalse(report['torch_dispatch_chain_complete'])
    def test_no_calls_are_not_success(self):
        with Vision17Trace(self.model,simulation=True) as trace:pass
        self.assertFalse(trace.report()['torch_dispatch_chain_complete'])
    def test_changed_weight_owner_path_is_refused(self):
        trace=Vision17Trace(self.model,simulation=True)
        self.model.unusual_name=CustomPatch()
        with self.assertRaisesRegex(RuntimeError,'REGISTRATION_CHANGED'):
            with trace:self.model(self.x)
    def test_foreign_dispatch_scope_is_not_popped_by_cleanup(self):
        class Foreign(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):return func(*args,**(kwargs or {}))
        trace=Vision17Trace(self.model,simulation=True).__enter__()
        foreign=Foreign();foreign.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError,'FOREIGN_DISPATCH_SCOPE'):trace.close()
            self.assertIs(_get_current_dispatch_mode_stack()[-1],foreign)
            self.assertTrue(trace.hooks)
        finally:
            foreign.__exit__(None,None,None);trace.close()


if __name__=='__main__':unittest.main()
