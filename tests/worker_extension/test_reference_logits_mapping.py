import types
import unittest
import torch
from reference_adapters.logits_mapping import LogitsColumnsRotate
from fault_injection.engines.region_lease import RegionLeases


class Model(torch.nn.Module):
    def compute_logits(self, x):
        return x.clone()


class LogitsTests(unittest.TestCase):
    def test_real_softmax_preserves_prior_legal_updates(self):
        model=Model();x=torch.tensor([[1., 2., 3., 4.]])
        runner=types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=['q']))
        for generation in (1,2):
            leases=RegionLeases('owner',generation,stream=lambda:0,complete=lambda s:None)
            site=LogitsColumnsRotate(model,runner,fault=True,region_leases=leases)
            site.install()
            for step in range(4):
                logits=model.compute_logits(x)
                site.begin_sample()
                logits.div_(.7)  # A legitimate sampler write, before the short lease.
                expected=logits.clone()
                output=logits.softmax(-1)
                site.finish_sample()
                self.assertTrue(torch.equal(logits,expected))
                self.assertTrue(torch.equal(output,torch.roll(expected,1,-1).softmax(-1)))
                self.assertGreater(site.events[-1]['distribution_l1'],0)
            site.close()
            self.assertEqual(leases.count(),0)
            self.assertTrue(torch.equal(model.compute_logits(x),x))

    def test_missing_real_producer_refused(self):
        site=LogitsColumnsRotate(Model(),None,fault=True,
            region_leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None))
        with self.assertRaisesRegex(RuntimeError,'NOT_FROM_REAL_MODEL'):site.begin_sample()
