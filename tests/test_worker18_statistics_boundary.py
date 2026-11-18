"""Torch hook-order and diagnostic-pause contracts, not real msprobe proof."""
from contextlib import contextmanager
import unittest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fault_injection.rmsnorm_return import RMSNormReturnSession


class Worker18StatisticsBoundaryTests(unittest.TestCase):
    def run_arm(self,arm):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.RMSNorm(4);self.consumer=torch.nn.Linear(4,2,bias=False)
                with torch.no_grad():self.consumer.weight.fill_(1)
            def forward(self,x):return self.consumer(self.norm(x))*2
        model=Model().eval();observed={};active=[True];multiplies=[]
        @contextmanager
        def diagnostics():
            before=active[0];active[0]=False
            try:yield
            finally:active[0]=before
        def record_before(module,args,output):
            self.assertTrue(active[0]);observed['prepended_module_output']=output.detach().clone()
        def record_consumer(module,args):
            self.assertTrue(active[0]);observed['actual_consumer_input']=args[0].detach().clone()
        class Operations(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                if str(func)=='aten.mul.Tensor':multiplies.append(active[0])
                return func(*args,**(kwargs or {}))
        with RMSNormReturnSession(model,arm=arm,simulation=True,diagnostics_context=diagnostics):
            # This is the reviewed msprobe hook order, not a fake dump file.
            handles=[model.norm.register_forward_hook(record_before,prepend=True),
                model.consumer.register_forward_pre_hook(record_consumer)]
            try:
                with Operations():observed['output']=model(torch.arange(1.,9.).reshape(2,4)).detach()
            finally:
                for handle in handles:handle.remove()
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()))
        return observed,multiplies

    def test_module_prepend_observes_original_return_consumer_observes_effective_return(self):
        b,_=self.run_arm('baseline');f,_=self.run_arm('fault')
        self.assertTrue(torch.equal(b['prepended_module_output'],f['prepended_module_output']))
        self.assertTrue(torch.equal(b['actual_consumer_input'],b['prepended_module_output']))
        self.assertTrue(torch.equal(f['actual_consumer_input'],f['prepended_module_output']*.5))
        self.assertTrue(torch.equal(f['output'],b['output']*.5))

    def test_owned_proxy_arithmetic_pauses_both_arms_real_model_operation_stays_observable(self):
        _,baseline=self.run_arm('baseline');_,fault=self.run_arm('fault')
        self.assertEqual(baseline,fault)
        self.assertIn(False,fault)
        self.assertIs(fault[-1],True)  # The model's final multiply remains captured.


if __name__=='__main__':unittest.main()
