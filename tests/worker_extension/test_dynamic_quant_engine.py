import unittest
import torch
from fault_injection.engines.dynamic_quant import shared_dynamic_quant


class SharedQuantEngineTests(unittest.TestCase):
    def test_elementwise_cpu_reference_and_input_immutable(self):
        # Scale 1 gives exact, independently enumerated ties-to-even values.
        x=torch.tensor([[127., -127., .5, 1.5], [2.5,-1.5,-.5,0.]])
        before=x.clone(); q,s=shared_dynamic_quant(x)
        self.assertTrue(torch.equal(q,torch.tensor([[127,-127,0,2],[2,-2,0,0]],dtype=torch.int8)))
        self.assertTrue(torch.equal(s,torch.ones(2)))
        self.assertTrue(torch.equal(x,before))
        self.assertTrue(q.is_contiguous() and s.is_contiguous())

    def test_zero_and_nonfinite(self):
        q,s=shared_dynamic_quant(torch.zeros(3,4))
        self.assertEqual(q.count_nonzero().item(),0)
        self.assertEqual(s.count_nonzero().item(),0)
        for value in (float('inf'),float('nan')):
            with self.assertRaisesRegex(ValueError,'NONFINITE'):
                shared_dynamic_quant(torch.tensor([[value]]))

    def test_explicit_portable_backend_matches_independent_rounding_reference(self):
        import numpy as np
        generator=torch.Generator().manual_seed(42)
        for maximum in (.001,.1,.3,.7,1.13,3.14,8.125,2.015625):
            x=(torch.randn(88,1024,generator=generator)*maximum).to(torch.bfloat16)
            q,s=shared_dynamic_quant(x,arithmetic_device='cpu')
            a=x.float().numpy();scale=np.max(np.abs(a))/np.float32(127)
            expected=np.clip(np.rint(a/scale),-127,127).astype(np.int8)
            self.assertTrue(np.array_equal(q.numpy(),expected))
            self.assertTrue(np.array_equal(s.numpy(),np.full(88,scale,dtype=np.float32)))
        with self.assertRaisesRegex(ValueError,'UNKNOWN_REFERENCE'):
            shared_dynamic_quant(torch.ones(2,2),arithmetic_device='typo')

if __name__=='__main__': unittest.main()
