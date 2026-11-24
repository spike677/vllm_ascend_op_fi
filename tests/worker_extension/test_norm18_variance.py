import unittest
from types import SimpleNamespace as NS
import torch
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.sites.norm18_variance import Variance18Site
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report


class VarianceTests(unittest.TestCase):
    def site(self, fault=True):
        site = Variance18Site(None, NS(input_batch=NS(req_ids=['real'])), fault=fault,
            region_leases=RegionLeases('owner', 1, stream=lambda:0, complete=lambda s:None))
        site.active=True; site.forward_calls.append({})
        return site

    def test_entire_output_consumed_and_byte_restored(self):
        site=self.site(); x=torch.arange(1,25,dtype=torch.bfloat16).reshape(4,6); before=x.clone()
        site.output('norm',x)
        with site.mode: result=x+1
        self.assertTrue(torch.equal(result,before*.5+1))
        self.assertTrue(torch.equal(x.view(torch.uint8),before.view(torch.uint8)))
        e=site.events[0]
        self.assertEqual((e['changed'],e['changed_elements']),(1,24))
        self.assertTrue(e['consumed'] and e['restored']); self.assertEqual(site.region_leases.count(),0)

    def test_zero_still_applied(self):
        site=self.site();x=torch.zeros(2,3,dtype=torch.bfloat16);site.output('norm',x)
        with site.mode:x.clone()
        e=site.events[0]; self.assertTrue(e['applied']);self.assertEqual(e['changed_elements'],0)
        self.assertEqual(e['changed'],0)

    def test_dense_transpose_is_full_consumer_but_view_is_not(self):
        site=self.site();x=torch.ones(2,3,dtype=torch.bfloat16);site.output('norm',x)
        with site.mode:
            view=x.T
            self.assertEqual(site.pending,1)
            value=view.clone()
        self.assertTrue(torch.equal(value,torch.full_like(value,.5)));self.assertEqual(site.pending,0)

    def test_partial_reader_and_mutation_are_not_false_proof(self):
        for mutate in (False,True):
            site=self.site();x=torch.ones(2,3,dtype=torch.bfloat16);site.output('norm',x)
            with self.assertRaisesRegex(RuntimeError,'NORM_(PARTIAL_REGION_READER|READER_WRITES)'):
                with site.mode:
                    x.add_(1) if mutate else x[:1].clone()
            self.assertEqual(site.pending,1)
            self.assertFalse(site.events[0]['consumed'])
            with site.mode:x.clone()  # finish original protected value, not a blind failure restore

    def test_noncontiguous_output_refused_before_write(self):
        site=self.site();x=torch.ones(2,6,dtype=torch.bfloat16)[:,::2]
        with self.assertRaisesRegex(RuntimeError,'DENSE_BF16'):site.output('norm',x)
        self.assertEqual(site.region_leases.count(),0)

    def test_no_old_qualification_inherited_and_no_user_gain(self):
        self.assertEqual(FaultConfig(18,'variance_over4').mechanism,'18.variance_over4')
        self.assertFalse(ordinary_route_report('18.variance_over4')['can_prepare'])
        with self.assertRaises(TypeError):FaultConfig(18,'variance_over4',gain=5)

    def test_rms_proxy_and_epsilon_limit_are_explicit(self):
        x=torch.tensor([[1.,2.,3.]],dtype=torch.float64);s=x.square().mean(-1,keepdim=True)
        norm=x/torch.sqrt(s)
        self.assertTrue(torch.equal(norm*.5,x/torch.sqrt(4*s)))
        eps=1.
        self.assertFalse(torch.equal(x/torch.sqrt(s+eps)*.5,x/torch.sqrt(4*s+eps)))
        self.assertFalse(Variance18Site.__doc__ == 'exact variance kernel')

    def test_off_leaves_output_unchanged(self):
        site=self.site(False);x=torch.ones(2,3,dtype=torch.bfloat16);site.output('norm',x)
        with site.mode:y=x.clone()
        self.assertTrue(torch.equal(x,y));self.assertEqual(site.region_leases.count(),0)
        self.assertFalse(site.events[0]['applied'])
