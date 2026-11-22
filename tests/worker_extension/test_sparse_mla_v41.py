import unittest
import torch
from fault_injection.sites.sparse_mla_v41 import from_operator_call
from fault_injection.engines.attention import recompute_attention_row


class SparseTests(unittest.TestCase):
    def test_real_config_slice_preserves_topology_and_quant_modules(self):
        import json
        from pathlib import Path
        from fault_injection.sites.deepseek_v41_metadata import StructureFixture
        data=json.loads((Path(__file__).parents[1]/'fixtures/dsv41_sparse_metadata.json').read_text())
        self.assertEqual(data['config_sha256'],'43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7')
        f=StructureFixture(data['config'],data['quant'],data['original_layer_ids'])
        self.assertEqual([f.layers[str(i)].compression_ratio for i in (0,2,20)],[0,2,1])
        for i in (0,2,20):
            leaves={x.actual_path:x for x in f.layers[str(i)].leaves}
            self.assertEqual(leaves[f'layers.{i}.attn.wq_b'].quant_types,('W8A8_DYNAMIC',))
            self.assertEqual(leaves[f'layers.{i}.ffn.experts.0.w1'].quant_types,('W4A8_DYNAMIC',))
        self.assertEqual(f.config['text_config']['head_dim'],512)
        with self.assertRaisesRegex(RuntimeError,'NOT_MODEL_INFERENCE'):f()

    def call(self):
        torch.manual_seed(123)
        q=torch.randn(2,2,512)*0.02
        cache=torch.randn(3,4,1,512)*0.05
        a=dict(layout_q='TND',layout_kv='PA_BBND',topk_value_mode=1,
            cu_seqlens_q=torch.tensor([0,1,2]),seqused_ori_kv=torch.tensor([3,5]),
            ori_kv=cache,ori_block_table=torch.tensor([[2,0],[1,0]]),ori_mask_mode=4,
            ori_win_left=2,ori_win_right=0,sinks=torch.tensor([0.3,-0.2]),softmax_scale=0.1,
            cmp_ratio=0,cmp_kv=None)
        return q,a

    def test_sink_and_replay_match_full_softmax_reference(self):
        q,a=self.call();r=from_operator_call(q,a,row=1,request_ids=['a','b'])
        self.assertEqual(r.request_id,'b')
        self.assertEqual(r.selected,(('ori',2,1,2),('ori',3,1,3),('ori',4,0,0)))
        scores=torch.einsum('hd,khd->hk',r.query,r.keys.expand(-1,2,-1))*r.scale
        logits=torch.cat([scores,r.sinks[:,None]],-1)
        expected=torch.einsum('hk,khd->hd',logits.softmax(-1)[:,:-1],r.keys.expand(-1,2,-1))
        torch.testing.assert_close(r.recompute(block_size=2),expected,rtol=2e-6,atol=1e-7)
        # Replay first two real keys in the denominator only; sink remains single.
        expanded=torch.cat([logits,scores[:,:2]],-1)
        reference=torch.einsum('hk,khd->hd',expanded.softmax(-1)[:,:3],r.keys.expand(-1,2,-1))
        torch.testing.assert_close(r.recompute(block_size=2,replay_blocks=(0,)),reference,rtol=2e-6,atol=1e-7)

    def test_sparse_sentinels_compressed_namespace_and_causal_boundary(self):
        q,a=self.call()
        a.update(cmp_kv=a['ori_kv'].clone(),cmp_block_table=torch.tensor([[1],[2]]),
            cmp_ratio=2,cmp_mask_mode=3,seqused_cmp_kv=torch.tensor([1,2]),cmp_residual_kv=torch.tensor([1,1]),
            cmp_sparse_indices=torch.tensor([[[0,-1,1]],[[1,0,99]]]))
        r=from_operator_call(q,a,row=1,request_ids=['a','b'])
        self.assertEqual(r.selected[-2:],(('cmp',1,2,1),('cmp',0,2,0)))
        a['cmp_sparse_indices'][1,0,2]=1
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):from_operator_call(q,a,row=1,request_ids=['a','b'])

    def test_reject_unproven_pages_packed_cache_and_sink(self):
        for key in ('page','dtype','sink','request'):
            q,a=self.call();ids=['a','b']
            if key=='page':a['ori_block_table'][1,1]=30
            elif key=='dtype':a['ori_kv']=a['ori_kv'].to(torch.int8)
            elif key=='sink':a['sinks']=None
            else:ids=['a','a']
            with self.subTest(key=key),self.assertRaises(ValueError):from_operator_call(q,a,row=1,request_ids=ids)

    def test_large_sink_stable_and_shape_rejected(self):
        q,a=self.call();a['sinks'].fill_(1000)
        r=from_operator_call(q,a,row=1,request_ids=['a','b'])
        self.assertTrue(torch.equal(r.recompute(block_size=1),torch.zeros_like(r.query)))
        with self.assertRaisesRegex(ValueError,'SINKS'):
            recompute_attention_row(r.query,r.keys,r.keys,scale=.1,block_size=1,sinks=torch.ones(1))
