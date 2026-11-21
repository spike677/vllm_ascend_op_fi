import unittest
from types import SimpleNamespace
import torch
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.sites.norm18 import Norm18Site


class NormTests(unittest.TestCase):
    def test_actual_strided_output_scaled_consumer_and_restore(self):
        runner=SimpleNamespace(input_batch=SimpleNamespace(req_ids=['real']))
        leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:None)
        site=Norm18Site(None,runner,fault=True,region_leases=leases)
        site.active=True;site.forward_calls.append({})
        x=torch.arange(1,25,dtype=torch.bfloat16).reshape(4,6)[:,::2];before=x.clone()
        site.output('fused.q_norm',x)
        with site.mode:result=x.clone()
        self.assertEqual(result[-1,0],before[-1,0]*.5)
        self.assertTrue(torch.equal(x,before));self.assertEqual(leases.count(),0)
        self.assertEqual(site.forward_calls,[{'fused.q_norm':1}])
        self.assertTrue(site.events[0]['consumed'] and site.events[0]['restored'])
        self.assertEqual(site.events[0]['request_ids'],['real'])

    def test_zero_element_counts_application_separately_from_change(self):
        runner=SimpleNamespace(input_batch=SimpleNamespace(req_ids=['real']))
        site=Norm18Site(None,runner,fault=True,region_leases=RegionLeases('cpu',1,stream=lambda:0,complete=lambda s:None))
        site.active=True;site.forward_calls.append({});x=torch.zeros(2,3,dtype=torch.bfloat16)
        site.output('norm',x)
        with site.mode:x.clone()
        self.assertTrue(site.events[0]['applied']);self.assertEqual(site.events[0]['changed'],0)
