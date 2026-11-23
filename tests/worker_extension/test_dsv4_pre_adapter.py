import unittest
from fault_injection.sites.dsv4_pre_adapter import prepare_site

class Dsv4PreparationTests(unittest.TestCase):
    def event(self,op):return dict(operator=op,source_fingerprints={'model':'a'*64},scheduler_rows=['request-real'],consumer_join={'input':'observed'})
    def call(self,m,o):return prepare_site(m,o,{'model':'a'*64})
    def test_norm_pair_preserves_routing_view(self):
        o=self.event('npu_rms_norm_cast');o['outputs']=[dict(dtype=d,shape=[2,8]) for d in ('torch.bfloat16','torch.float32')]
        p=self.call('18',o);self.assertEqual(p['targets'],[0,1]);self.assertFalse(p['public_enabled'])
        o['outputs'].pop()
        with self.assertRaisesRegex(ValueError,'PAIR'):self.call('18',o)
    def test_partial_rotary_actual_tail_slice(self):
        o=self.event('inplace_partial_rotary_mul');o.update(inputs=[dict(shape=[2,1,4,16])],partial_slice=[8,16],rotary_mode='interleave',cos_sin_row_join=True)
        self.assertEqual(self.call('22',o)['partial_slice'],[8,16])
        o['cos_sin_row_join']=False
        with self.assertRaisesRegex(ValueError,'ROW_JOIN'):self.call('22',o)
    def test_linear_quantized_needs_arithmetic_contract(self):
        for op in ('MatMulV2','QuantMatmulWeightNz'):
            o=self.event(op);o['outputs']=[dict(dtype='torch.bfloat16',format='ND')]
            self.assertFalse(self.call('39op',o)['qualification_granted'])
            with self.assertRaisesRegex(ValueError,'ARITHMETIC'):self.call('39mr',o)
            o['matmul_contract']={k:True for k in ('logical_k_join','weight_layout_decoded','scales_joined','bias_joined')}
            self.assertEqual(self.call('39mr',o)['method'],'reference_recompute')
    def test_sparse_runtime_layout_and_key_set_required(self):
        o=self.event('npu_sparse_attn_sharedkv')
        with self.assertRaisesRegex(ValueError,'LAYOUT'):self.call('28',o)
        o.update(kv_layout='PA_ND',compressed_cache_decoded=True,legal_key_indices=[0,2,4],causal_request_join=True,softmax_scale=0.125)
        self.assertEqual(self.call('28',o)['legal_key_indices'],[0,2,4])
        o['legal_key_indices']=[0,2,2]
        with self.assertRaisesRegex(ValueError,'KEY_SET'):self.call('28',o)
    def test_no_request_consumer_or_source_no_plan(self):
        o=self.event('RmsNorm')
        for k in ('scheduler_rows','consumer_join','source_fingerprints'):
            v=o[k];o[k]=None
            with self.assertRaises(ValueError):self.call('18',o)
            o[k]=v
if __name__=='__main__':unittest.main()
