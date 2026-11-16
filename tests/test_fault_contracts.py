import unittest
import os
from dataclasses import replace
from types import SimpleNamespace as NS
from fault_injection import AttentionCall, UnsupportedCall, attach, list_faults
from fault_injection.fia import FiaBackend


def tensor(shape, dtype='torch.float16', device='npu'):
    return NS(shape=shape, dtype=dtype, device=NS(type=device))


class Contracts(unittest.TestCase):
    def call(self, q=(8,16,2048,128), k=(8,8,2048,128), v=(8,8,2048,128)):
        return AttentionCall('arbitrary.blocks.37.attention', NS(training=False, is_causal=True),
                             tensor(q), tensor(k), tensor(v), None, scaling=128**-.5)

    def test_distinct_v_dimension_is_described_but_kernel_rejects(self):
        call=self.call(q=(8,16,2048,192), k=(8,8,2048,192), v=(8,8,2048,128))
        self.assertEqual(call.geometry()['qk_dim'],192)
        self.assertEqual(call.geometry()['value_dim'],128)
        self.assertFalse(FiaBackend().check(call).supported)

    def test_registry_does_not_imply_runtime_coverage(self):
        self.assertGreater(len(list_faults()),10)
        with self.assertRaises(UnsupportedCall): attach(object(),'15.alias_detach')
        with self.assertRaises(ValueError): attach(object(),'28')

    def test_packed_tokens_are_not_batch(self):
        with self.assertRaises(UnsupportedCall): self.call(q=(16384,16,128)).geometry()

    def test_decode_not_pretended_prefill(self):
        self.assertFalse(FiaBackend().check(self.call(q=(8,16,1,128))).supported)

    def test_unknown_mask_cache_and_packing_semantics_rejected(self):
        for kwargs in ({'is_causal':False}, {'past_key_value':object()},
                       {'cu_seq_lens_q':[0,128,2048]}, {'sink':object()}, {'unrecognized':None}):
            self.assertFalse(FiaBackend().check(replace(self.call(),kwargs=kwargs)).supported)

    def test_host_receipt_cannot_substitute_a_different_call(self):
        backend=FiaBackend()
        row=dict(token='a',owner_pid=os.getpid(),owner_starttime_ticks='123',
                 mode='coupled_replay_m128',eligible=True,selected_m=128,
                 tiling_key='119000000000400000',serializer_roundtrip_matches=True,checks={'shape':True})
        self.assertEqual(backend._host_receipt([row],'a','123'),row)
        for rows in ([],[row,row],[dict(row,mode='off')],[dict(row,checks={'shape':False})],
                     [dict(row,tiling_key='109000000000400000')]):
            with self.assertRaises(UnsupportedCall):backend._host_receipt(rows,'a','123')


if __name__=='__main__': unittest.main()
