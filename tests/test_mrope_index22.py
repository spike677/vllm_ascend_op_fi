import unittest
import torch
from fault_injection.mrope_index22 import call_mrope_index22
from fault_injection.core import UnsupportedCall

class MropeIndexBoundary(unittest.TestCase):
    def setUp(self):
        # Three distinct axes, partial rotary width64/head256, real cache index.
        self.cache=torch.arange(16*64,dtype=torch.float32).reshape(16,64).to(torch.bfloat16)
        self.positions=torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
        self.calls=0
    def original(self,cache,indices):
        self.calls+=1
        return torch.ops.aten.index.Tensor(cache,indices)
    def call(self,arm='fault',**changes):
        kw=dict(request_positions=self.positions,expected_positions=self.positions.tolist(),
                arm=arm,admit=lambda:True,simulation=True)
        kw.update(changes)
        return call_mrope_index22(self.original,self.cache,self.positions,**kw)
    def test_real_index_effect_and_off_restore(self):
        p=self.positions.clone();c=self.cache.clone()
        b,br=self.call('baseline');f,fr=self.call();o,orr=self.call('off')
        self.assertEqual(self.calls,3);self.assertTrue(torch.equal(b,o))
        self.assertTrue(torch.equal(f[:,-1],b[:,0]));self.assertTrue(torch.equal(f[:,:-1],b[:,:-1]))
        self.assertTrue(torch.equal(self.positions,p));self.assertTrue(torch.equal(self.cache,c))
        self.assertGreater(fr['changed_cache_elements'],0);self.assertEqual(br['changed_cache_elements'],0)
        self.assertFalse(fr['fused_qk_effect_verified']);self.assertFalse(fr['output_rewritten'])
    def test_nonoverlapping_three_axis_runner_prefix_and_float_cache(self):
        storage=torch.zeros((3,32),dtype=torch.int64);storage[:,:4].copy_(self.positions)
        self.positions=storage[:,:4];self.cache=self.cache.float()
        self.assertFalse(self.positions.is_contiguous())
        before=storage.clone();b,_=self.call('baseline');f,report=self.call()
        self.assertTrue(torch.equal(before,storage));self.assertTrue(torch.equal(f[:,:-1],b[:,:-1]))
        self.assertTrue(torch.equal(f[:,-1],b[:,0]));self.assertGreater(report['changed_cache_elements'],0)
    def test_overlapping_axis_view_refused(self):
        self.positions=torch.tensor([0,1,2,3]).expand(3,4)
        with self.assertRaisesRegex(UnsupportedCall,'THREE_AXIS'):self.call()
        self.assertEqual(self.calls,0)
    def test_wrong_request_object_refused(self):
        with self.assertRaises(UnsupportedCall):self.call(request_positions=self.positions.clone())
        self.assertEqual(self.calls,0)
    def test_partial_request_mapping_refused(self):
        with self.assertRaises(UnsupportedCall):self.call(expected_positions=[[0,1,2,3]]*3)
        self.assertEqual(self.calls,0)
    def test_missing_admission_refuses_before_original(self):
        with self.assertRaises(UnsupportedCall):self.call(admit=lambda:False)
        self.assertEqual(self.calls,0)
    def test_unknown_device_is_not_qualified(self):
        with self.assertRaises(UnsupportedCall):self.call(simulation=False)
        self.assertEqual(self.calls,0)
    def test_out_of_range_refused(self):
        self.positions[2,-1]=16
        with self.assertRaises(UnsupportedCall):self.call()
        self.assertEqual(self.calls,0)
    def test_fault_does_not_write_native_return_to_pass(self):
        def wrong(cache,indices):return cache[self.positions].clone()
        with self.assertRaisesRegex(UnsupportedCall,'ACTUAL_INDEX_RESULT'):
            call_mrope_index22(wrong,self.cache,self.positions,request_positions=self.positions,
                expected_positions=self.positions.tolist(),arm='fault',admit=lambda:True,simulation=True)
    def test_no_effect_identical_cache_refused(self):
        self.cache.fill_(1)
        with self.assertRaises(UnsupportedCall):self.call()
        self.assertEqual(self.calls,0)

if __name__=='__main__':unittest.main()
