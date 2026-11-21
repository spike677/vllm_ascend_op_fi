import unittest
import torch
from fault_injection.engines.dynamic_quant import shared_dynamic_quant
from fault_injection.engines.deferred_quant_reference import DeferredQuantReference


class DeferredReferenceTests(unittest.TestCase):
    def queue(self,**kw):
        self.fences=[];self.stream=0
        return DeferredQuantReference(stream=lambda:self.stream,complete=self.fences.append,**kw)

    def test_snapshots_survive_live_buffer_reuse_and_one_fence_covers_calls(self):
        queue=self.queue(sample_limit=16)
        for i in range(3):
            x=torch.arange(1,65,dtype=torch.float32).reshape(4,16)+i
            q,s=shared_dynamic_quant(x,arithmetic_device='cpu');queue.enqueue(str(i),x,q,s)
            x.fill_(999);q.zero_();s.zero_()
        self.assertEqual(self.fences,[])
        rows=queue.flush();self.assertEqual(self.fences,[0]);self.assertEqual(queue.pending(),0)
        self.assertTrue(all(r['reference_exact'] and r['scale_reference_exact'] for r in rows))

    def test_bad_scale_and_nonboundary_q_are_not_hidden_by_defer(self):
        for kind in ('scale','q'):
            queue=self.queue();x=torch.tensor([[1.,2.,3.],[4.,5.,6.]])
            q,s=shared_dynamic_quant(x,arithmetic_device='cpu')
            if kind=='scale':s[0]=s[0]*1.001
            else:q[0,0]+=3
            queue.enqueue('bad',x,q,s)
            with self.assertRaises(ValueError):queue.flush()
            self.assertEqual(queue.pending(),1)

    def test_stream_budget_and_duplicate_refuse_without_discarding_snapshot(self):
        queue=self.queue(max_calls=1);x=torch.ones(2,4);q,s=shared_dynamic_quant(x)
        queue.enqueue('one',x,q,s)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):queue.enqueue('one',x,q,s)
        with self.assertRaisesRegex(ValueError,'BUDGET'):queue.enqueue('two',x,q,s)
        self.stream=1
        with self.assertRaisesRegex(ValueError,'STREAM'):queue.flush()
        self.stream=0;self.assertEqual(len(queue.flush()),1)

    def test_budget_reserves_packing_peak_not_only_retained_samples(self):
        x=torch.ones(2,4);q,s=shared_dynamic_quant(x)
        retained=8*(4+1+8)+2*4+4
        queue=self.queue(max_bytes=retained)
        with self.assertRaisesRegex(ValueError,'BUDGET'):queue.enqueue('one',x,q,s)
        self.assertEqual(queue.pending(),0)
        queue=self.queue(max_bytes=2*retained)
        queue.enqueue('one',x,q,s);self.assertEqual(len(queue.flush()),1)
