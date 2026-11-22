import unittest
import torch
from fault_injection.engines.attention import recompute_attention_row
from fault_injection.sites.paged33 import partition_pair,Paged33Site


class PagedTests(unittest.TestCase):
    def test_full_key_set_and_float32_reference(self):
        torch.manual_seed(33)
        q=torch.randn(4,16);k=torch.randn(259,2,16);v=torch.randn(259,2,12)
        native=recompute_attention_row(q,k,v,scale=.25,block_size=128)
        output,info=partition_pair(q,k,v,scale=.25,native=native)
        kk=k.repeat_interleave(2,1);vv=v.repeat_interleave(2,1)
        reference=torch.einsum('hk,khd->hd',torch.softmax(torch.einsum('hd,khd->hk',q,kk)*.25,-1),vv)
        torch.testing.assert_close(output,reference,rtol=1e-5,atol=1e-7)
        self.assertEqual(info['key_visits'],259);self.assertFalse(info['denominator_replay'])
        self.assertEqual(info['native_equal_control_elements'],48)

    def test_native_reference_difference_is_never_injected(self):
        torch.manual_seed(2)
        q=torch.randn(2,8);k=torch.randn(133,1,8);v=torch.randn(133,1,8)
        native=torch.full_like(q,100.)
        output,info=partition_pair(q,k,v,scale=.25,native=native)
        self.assertTrue(torch.equal(output,native));self.assertEqual(info['selected_changed_elements'],0)

    def test_prefill_and_single_block_not_eligible(self):
        site=object.__new__(Paged33Site)
        self.assertFalse(site.eligible_call(256,256)[0]);self.assertFalse(site.eligible_call(1,100)[0])
        self.assertTrue(site.eligible_call(1,256)[0])
