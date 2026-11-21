import unittest
import numpy as np
import torch
from fault_injection.engines.quant_reference import compare_samples,sampled_reference
from fault_injection.engines.dynamic_quant import shared_dynamic_quant,compensated_scale


class ReferenceTests(unittest.TestCase):
    def compare(self,x,q,scale=1):
        return compare_samples(x,np.array(q,dtype=np.int8),[scale],127,total_elements=len(x))

    def test_half_integer_only_not_general_one_code_tolerance(self):
        r=self.compare([.5,1.5,-.5],[1,1,-1])
        self.assertEqual(r['q_sample_differences'],3)
        self.assertFalse(r['reference_exact']);self.assertTrue(r['reference_passed'])
        with self.assertRaisesRegex(ValueError,'NON_BOUNDARY'):self.compare([.4],[1])
        with self.assertRaisesRegex(ValueError,'NON_BOUNDARY'):self.compare([.5],[2])

    def test_scale_must_be_exact(self):
        with self.assertRaisesRegex(ValueError,'SCALE'):self.compare([.5],[0],1.0001)

    def test_sampling_is_bounded_and_zero_input_is_defined(self):
        for x in (torch.arange(3000,dtype=torch.float32).reshape(3,1000),torch.zeros(3,1000)):
            q,s=shared_dynamic_quant(x)
            r=sampled_reference(x,q,s,sample_limit=32)
            self.assertLessEqual(r['sampled_elements'],32)
            self.assertEqual(r['total_elements'],3000)
            self.assertTrue(r['reference_passed'])
    def test_compensation_corrects_one_ulp_for_bf16_normal_magnitudes(self):
        # Exhaustive BF16 significands across the exponent range exercised by
        # activation quantization, with independently simulated +/-1 ULP errors.
        bits=torch.arange(0x2000,0x6000,dtype=torch.int32).to(torch.int16)
        values=bits.view(torch.bfloat16).float()
        ref=torch.from_numpy(values.numpy()/np.float32(127))
        for direction in (-float('inf'),float('inf')):
            approximate=torch.nextafter(ref,torch.full_like(ref,direction))
            self.assertTrue(torch.equal(compensated_scale(values,approximate),ref))
