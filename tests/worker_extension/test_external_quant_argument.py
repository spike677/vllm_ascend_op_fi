"""Synthetic dispatcher fixture, not an NPU quantization accuracy test."""
import unittest
from types import SimpleNamespace as NS
import torch
from fault_injection.engines.region_lease import RegionLeases
from experiments.worker_core_v11_20261118.external_quant_argument_example import ExternalQuantArgumentSite


class QuantArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=torch.library.Library('npu','FRAGMENT')
        cls.library.define('npu_dynamic_quant(Tensor self, Tensor? smooth_scales=None) -> (Tensor, Tensor)')
        cls.library.impl('npu_dynamic_quant',lambda x,smooth_scales=None:
            ((x if smooth_scales is None else x*smooth_scales).to(torch.int8),torch.ones(1)),
            'CompositeExplicitAutograd')
    @classmethod
    def tearDownClass(cls):cls.library._destroy()
    def test_real_argument_forwarding_and_readonly_original_two_owners(self):
        class Quant(torch.nn.Module):
            def forward(self,x):
                q,s=torch.ops.npu.npu_dynamic_quant(x,smooth_scales=torch.ones_like(x))
                return q.to(torch.float32)*s
        model=Quant();x=torch.full((2,4),4.,dtype=torch.bfloat16)
        for owner in ('a','b'):
            leases=RegionLeases(owner,1,stream=lambda:0,complete=lambda s:None)
            site=ExternalQuantArgumentSite(model,NS(input_batch=NS(req_ids=['r'])),fault=True,region_leases=leases)
            site.install();out=model(x);site.finish_forward();site.close()
            self.assertEqual(out.tolist(),[[4]*4,[3,3,4,4]])
            self.assertTrue(torch.all(x==4));self.assertEqual(leases.count(),0)
            self.assertTrue(site.events[0]['result_forwarded_unmodified'])
            self.assertEqual(site.events[0]['changed_elements'],2)
    def test_wrong_dtype_refused_before_lease(self):
        class Quant(torch.nn.Module):
            def forward(self,x):return torch.ops.npu.npu_dynamic_quant(x)
        leases=RegionLeases('a',1,stream=lambda:0,complete=lambda s:None)
        site=ExternalQuantArgumentSite(Quant(),NS(input_batch=NS(req_ids=['r'])),fault=True,region_leases=leases)
        site.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'ARGUMENT_LAYOUT'):site.model(torch.ones(2,4))
            self.assertEqual(leases.count(),0)
        finally:site.close()


if __name__=='__main__':unittest.main()
