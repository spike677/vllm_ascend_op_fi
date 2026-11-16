"""Mask semantic checks on CPU; no native device call."""
import importlib.util
import unittest
from types import SimpleNamespace
from fault_injection import AttentionCall, UnsupportedCall
from fault_injection.fia import FiaBackend


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch required')
class Masks(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.q = torch.zeros(2, 4, 8, 4)
        self.blocked = torch.ones(8, 8, dtype=torch.bool).triu(1)

    def convert(self, mask):
        return FiaBackend.causal_mask(AttentionCall('any', SimpleNamespace(), self.q,
                                    self.q, self.q, mask))

    def test_none_uses_explicit_causal_semantics(self):
        self.assertTrue(self.torch.equal(self.convert(None), self.blocked))

    def test_bool_allowed_converts_to_native_blocked(self):
        self.assertTrue(self.torch.equal(self.convert(~self.blocked), self.blocked))

    def test_four_dimensional_additive_mask(self):
        t = self.torch
        mask = t.zeros(2,1,8,8).masked_fill(self.blocked, t.finfo(t.float32).min)
        self.assertTrue(t.equal(self.convert(mask), self.blocked))

    def test_padding_and_bias_and_wrong_polarity_rejected(self):
        t = self.torch
        padding = (~self.blocked).clone(); padding[:,0] = False
        bias = t.zeros(8,8).masked_fill(self.blocked,float('-inf')); bias[4,0] = .1
        for mask in (padding,bias,self.blocked,t.ones(8,8,dtype=t.int32)):
            with self.assertRaises(UnsupportedCall): self.convert(mask)

    def test_post_rope_full_prefill_positions_and_cache_offsets(self):
        t = self.torch
        def call(pos):
            return AttentionCall('any',SimpleNamespace(),self.q,self.q,self.q,None,
                                 kwargs={'position_ids':pos,'use_cache':True})
        FiaBackend.validate_positions(call(t.arange(8).reshape(1,8)))
        for positions in (t.arange(8).reshape(1,8)+1, t.zeros(2,8,dtype=t.int64), t.arange(8).float()):
            with self.assertRaises(UnsupportedCall): FiaBackend.validate_positions(call(positions))


if __name__=='__main__': unittest.main()
