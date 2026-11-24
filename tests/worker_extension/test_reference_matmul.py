import types
import unittest
import torch
from reference_adapters.matmul_order import reference, ReverseSplitK, EarlyBiasRound
from fault_injection.engines.region_lease import RegionLeases


class Model(torch.nn.Module):
    def __init__(self,bias):
        super().__init__();self.proj=torch.nn.Linear(257,16,bias=bias).to(torch.bfloat16)
    def forward(self,x):return self.proj(x)+1


class MatmulTests(unittest.TestCase):
    def test_tail_key_not_omitted_and_bias_requires_natural_input(self):
        x=torch.ones(257);w=torch.ones(257,8)
        normal,fault=reference(x,w)
        self.assertTrue(torch.equal(normal,torch.full_like(normal,257)))
        self.assertTrue(torch.equal(normal,fault))
        with self.assertRaisesRegex(RuntimeError,'REAL_BIAS'):reference(x,w,early_round=True)

    def test_real_bf16_producer_consumer_and_two_scopes(self):
        torch.manual_seed(1)
        for cls in (ReverseSplitK,EarlyBiasRound):
            inner=Model(cls is EarlyBiasRound)
            model=inner
            if cls is EarlyBiasRound:
                model=torch.nn.Module();model.visual=inner
            x=torch.randn(2,257,dtype=torch.bfloat16);baseline=inner(x)
            weights={k:v.clone() for k,v in inner.state_dict().items()}
            for generation in (1,2):
                leases=RegionLeases('owner',generation,stream=lambda:0,complete=lambda s:None)
                runner=types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['q']))
                site=cls(model,runner,fault=True,region_leases=leases)
                site.install();inner(x);site.finish_forward();site.close()
                self.assertEqual(len(site.events),1)
                e=site.events[0]
                self.assertTrue(e['control_passed'] and e['consumed'] and e['restored'])
                self.assertFalse(e['quantized_operand_recomputed'])
                self.assertEqual(e['before_sha256'],e['restored_sha256'])
                self.assertTrue(torch.equal(inner(x),baseline))
                self.assertTrue(all(torch.equal(v,weights[k]) for k,v in inner.state_dict().items()))
