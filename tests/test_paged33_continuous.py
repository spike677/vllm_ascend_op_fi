import unittest
from types import SimpleNamespace
import torch
from fault_injection.deployments.paged33_continuous import pack_partial_pages,ContinuousPaged33Runtime
from fault_injection.core import UnsupportedCall


class PartialPagesTests(unittest.TestCase):
    def test_exact_valid_inverse_and_zero_tail_across_requests(self):
        for length in (128,129,255,256,257):
            x=torch.arange(2*3*length*4).reshape(2,3,length,4)
            pages,mapping,tail=pack_partial_pages(x)
            for i,ids in enumerate(mapping):
                inv=pages[ids[0]:ids[-1]+1].permute(1,0,2,3).reshape(3,-1,4)
                self.assertTrue(torch.equal(inv[:,:length],x[i]));self.assertEqual(int((inv[:,length:]!=0).sum()),0)
            self.assertFalse(set(mapping[0])&set(mapping[1]));self.assertEqual(tail,(-length)%128)

    def test_invalid_lengths_and_capacity_are_rejected(self):
        with self.assertRaises(UnsupportedCall):pack_partial_pages(torch.empty(2,3,0,4))
        for count in (0,9,True):
            with self.assertRaises(UnsupportedCall):ContinuousPaged33Runtime(object(),bundle='unused',evidence_dir='unused',decode_steps=count)

    def test_stock_followup_requires_closed_completed_fault_scope(self):
        runtime=ContinuousPaged33Runtime.__new__(ContinuousPaged33Runtime)
        runtime.stock_started=False;runtime.previous_length=16383;runtime.calls=[{}]
        runtime.route=SimpleNamespace(_closed=False);runtime.pending=None
        with self.assertRaises(UnsupportedCall):
            with runtime.stock_followups():pass
        runtime.route._closed=True;runtime.previous_length=16384
        with self.assertRaises(UnsupportedCall):
            with runtime.stock_followups():pass


if __name__=='__main__':unittest.main()
