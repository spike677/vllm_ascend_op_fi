import unittest
from fault_injection.sites.deepseek_v41_metadata import StructureFixture, inspect_metadata, site_requirements
from tools.fetch_model_metadata import allowed


class MetadataTests(unittest.TestCase):
    def test_download_never_allows_weights_or_parent_traversal(self):
        for p in ('weights.safetensors','optional/quarot.safetensors','pytorch_model.bin','../config.json'):
            self.assertFalse(allowed(p))
        self.assertTrue(allowed('quant_model_weights.safetensors.index.json'))
        self.assertTrue(allowed('tokenizer.model'))

    def test_original_layer_ids_and_quantization_are_preserved(self):
        c=dict(model_type='deepseek_v41',text_config=dict(num_hidden_layers=3,
            compress_ratios=[0,2,1],num_attention_heads=64,head_dim=512,n_routed_experts=384,num_experts_per_tok=6))
        q={'layers.2.attn.wq_b.weight':'W8A8_DYNAMIC', 'layers.2.ffn.experts.0.w1.weight':'W4A8_DYNAMIC',
           'mtp.0.attn.wq_b.weight':'FLOAT', 'layers.1.engram.embed.quant_type':'INT8_SYM'}
        f=StructureFixture(c,q,(0,2))
        self.assertEqual(f.layers['2'].compression_ratio,1)
        self.assertEqual({x.actual_path for x in f.layers['2'].leaves},{'layers.2.attn.wq_b','layers.2.ffn.experts.0.w1'})
        self.assertEqual(len(inspect_metadata(c,q)['exceptions']),2)
        self.assertFalse(inspect_metadata(c,q)['qualification_granted'])
        with self.assertRaisesRegex(RuntimeError,'NOT_MODEL_INFERENCE'): f()
        with self.assertRaises(ValueError): StructureFixture(c,q,(0,3))
        self.assertIn('actual expert placement',site_requirements()['8'])
