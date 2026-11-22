import unittest,torch
from fault_injection.engines.paged33_native_control import split_fuse_control,agreement
class ControlTests(unittest.TestCase):
 def test_key_set_guard_and_output_shape(self):
  torch.manual_seed(41);q=torch.randn(1,4,8).bfloat16();k=torch.randn(13,2,8).bfloat16();v=torch.randn_like(k)
  c=split_fuse_control(q,k,v,scale=.25,split_spans=[(0,8),(8,13)])
  self.assertEqual(c.shape,q.shape);self.assertTrue(torch.isfinite(c).all())
  with self.assertRaisesRegex(ValueError,'GAP'):split_fuse_control(q,k,v,scale=.25,split_spans=[(0,8),(7,13)])
 def test_signed_zero_is_not_bitwise_equal(self):
  a=torch.tensor([0.],dtype=torch.bfloat16);b=-a
  self.assertEqual(agreement(a,b)['bitwise_equal'],0)

 def test_bf16_operands_preserved_at_fp32_output_boundary(self):
  from unittest.mock import patch
  q=torch.randn(1,2,8).bfloat16();k=torch.randn(13,1,8).bfloat16();v=torch.randn_like(k);calls=[]
  def mm(a,b,*,out_dtype):
   calls.append((a.dtype,b.dtype,out_dtype));return a.float()@b.float()
  with patch.object(torch,'mm',mm):c=split_fuse_control(q,k,v,scale=.25,split_spans=[(0,8),(8,13)],matmul_mode='bf16_out_fp32')
  self.assertEqual(c.dtype,torch.bfloat16);self.assertEqual(calls,[(torch.bfloat16,torch.bfloat16,torch.float32)]*4)
