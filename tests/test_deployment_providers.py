import unittest
from fault_injection.core import UnsupportedCall
from fault_injection.deployments.residual26 import allocation_pair,Residual26Runtime
from fault_injection.deployments.paged33 import pack_pages

class AllocationGateTests(unittest.TestCase):
    def snapshot(self,state='active_allocated',gap=0):
        return [dict(address=1000,blocks=[dict(size=512,state='active_allocated'),
            dict(address=1512+gap,size=512,state=state)])]
    def test_distinct_live_blocks_and_padding(self):
        pair=allocation_pair(self.snapshot(),1000,1512,480,128)
        self.assertEqual((pair['padding'],pair['shift']),(32,64))
    def test_free_or_nonadjacent_or_same_allocation_is_rejected(self):
        for snapshot,residual,output in ((self.snapshot('inactive'),1000,1512),
            (self.snapshot(gap=512),1000,2024),(self.snapshot(),1000,1200)):
            with self.assertRaises(UnsupportedCall):allocation_pair(snapshot,residual,output,128,128)
    def test_tensor_must_fit_actual_blocks(self):
        with self.assertRaises(UnsupportedCall):allocation_pair(self.snapshot(),1000,1512,513,128)
        with self.assertRaises(UnsupportedCall):allocation_pair(self.snapshot(),1000,1512,480,513)

class PageMappingTests(unittest.TestCase):
    def test_every_request_head_and_key_survives_page_mapping(self):
        import torch
        # Distinct row/head/key values catch accidental B/H interleaving.
        x=torch.arange(3*2*12*4).reshape(3,2,12,4)
        pages=pack_pages(x,4)
        for row in range(3):
            restored=pages[row*3:(row+1)*3].permute(1,0,2,3).reshape(2,12,4)
            self.assertTrue(torch.equal(restored,x[row]))

class ResidualForwardLifetimeTests(unittest.TestCase):
    def runtime(self):
        runtime=Residual26Runtime.__new__(Residual26Runtime)
        runtime.retained={123:object()};runtime.forward_active=False;runtime.forward_index=-1
        runtime.pending=lambda:0;runtime.error=None;runtime.verify_completion=lambda:None
        return runtime

    def test_prefill_residual_cannot_be_selected_in_next_decode(self):
        r=self.runtime();r.begin_forward(None,())
        self.assertNotIn(123,r.retained)
        r.retained[123]=object()  # Actual producer in this forward.
        r.finish_forward(None,(),object())
        r.begin_forward(None,())
        self.assertNotIn(123,r.retained)
        self.assertEqual(r.forward_index,1)

    def test_failed_forward_does_not_leave_stale_producer(self):
        r=self.runtime();r.begin_forward(None,());r.retained[123]=object();r.error='native failure'
        with self.assertRaises(RuntimeError):r.finish_forward(None,(),None)
        self.assertFalse(r.forward_active);self.assertEqual(r.retained,{})

    def test_pending_consumer_prevents_next_forward(self):
        r=self.runtime();r.pending=lambda:1
        with self.assertRaises(RuntimeError):r.begin_forward(None,())
        self.assertEqual(r.forward_index,-1)

if __name__=='__main__':unittest.main()
