import unittest
from types import SimpleNamespace as NS
import torch
from fault_injection.engines.region_lease import RegionLeases
from experiments.worker_core_v11_20261118.external_input_examples import (
    ExternalModuleInputSite, ExternalDispatchInputSite)


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.quant_method=object();self.input_size=4;self.output_size=4
        self.weight=torch.nn.Parameter(torch.eye(4,dtype=torch.bfloat16))
    def forward(self,x):return x @ self.weight


class Model(torch.nn.Module):
    def __init__(self):super().__init__(); self.proj=Projection()
    def forward(self,x):return self.proj(x)+torch.ones_like(x)


class InputSiteTests(unittest.TestCase):
    def exercise(self, factory):
        model=Model(); x=torch.arange(1,13,dtype=torch.bfloat16).reshape(3,4)
        original=x.clone(); baseline=model(x)
        for owner in ('a','b'):
            leases=RegionLeases(owner,1,stream=lambda:0,complete=lambda s:None)
            site=factory(model,NS(input_batch=NS(req_ids=['r'])),fault=True,region_leases=leases)
            site.install()
            for _ in range(2):
                out=model(x); site.finish_forward()
                expected=baseline.clone(); expected[-1,:2]=x[-1,:2]*.75+1
                self.assertTrue(torch.equal(out,expected));self.assertTrue(torch.equal(x,original))
                event=site.events[-1]
                self.assertEqual(event['changed_elements'],2)
                self.assertEqual(event['changed'],1)
                self.assertTrue(event['consumed'] and event['restored'])
                self.assertEqual(event['before_sha256'],event['restored_sha256'])
                self.assertEqual(leases.count(),0)
            site.close();self.assertTrue(torch.equal(model(x),baseline))
    def test_module_real_input_consumer_two_owners(self):self.exercise(ExternalModuleInputSite)
    def test_dispatch_operand_two_owners(self):self.exercise(ExternalDispatchInputSite)
    def test_no_candidate_refused(self):
        site=ExternalModuleInputSite(torch.nn.Identity(),NS(),fault=True,
            region_leases=RegionLeases('a',1,stream=lambda:0,complete=lambda s:None))
        with self.assertRaisesRegex(RuntimeError,'NO_PROJECTION_INPUT'):site.install()
        site.close()
    def test_embedding_weight_and_quant_method_are_not_projection_proof(self):
        embedding=torch.nn.Embedding(12,4);embedding.quant_method=object()
        site=ExternalModuleInputSite(embedding,NS(),fault=True,
            region_leases=RegionLeases('a',1,stream=lambda:0,complete=lambda s:None))
        with self.assertRaisesRegex(RuntimeError,'NO_PROJECTION_INPUT'):site.install()
        site.close()
    def test_zero_change_is_applied(self):
        model=Model();leases=RegionLeases('a',1,stream=lambda:0,complete=lambda s:None)
        site=ExternalModuleInputSite(model,NS(input_batch=NS(req_ids=['r'])),fault=True,region_leases=leases)
        site.install();model(torch.zeros(2,4,dtype=torch.bfloat16));site.finish_forward();site.close()
        self.assertTrue(site.events[0]['applied']);self.assertEqual(site.events[0]['changed'],0)
    def test_dispatch_preserves_alpha_and_non_target_rows(self):
        class Add(torch.nn.Module):
            def forward(self,x):return torch.add(x,torch.ones_like(x),alpha=2)
        model=Add();leases=RegionLeases('a',1,stream=lambda:0,complete=lambda s:None)
        site=ExternalDispatchInputSite(model,NS(input_batch=NS(req_ids=['r'])),fault=True,region_leases=leases)
        site.install();x=torch.ones(2,4,dtype=torch.bfloat16);out=model(x);site.finish_forward();site.close()
        self.assertEqual(out.tolist(),[[3]*4,[2.75,2.75,3,3]])


if __name__=='__main__':unittest.main()
