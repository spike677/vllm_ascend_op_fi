"""Tiny-depth structural model with real public config and meta dummy weights."""
import copy,json,unittest
from pathlib import Path
try:
    import torch
except ImportError:
    torch=None
from demoncat.runtime.vllm_ascend.worker_extension.v026_inspection import discover_qwen35_sites

@unittest.skipIf(torch is None,'optional CPU structural fixture requires torch')
class Qwen35StructureTests(unittest.TestCase):
    def fixture(self, quantized=False):
        config=json.loads((Path(__file__).parent/'fixtures/qwen38_public_config.json').read_bytes())
        config=copy.deepcopy(config);text=config['text_config'];text['num_hidden_layers']=4;text['layer_types']=text['layer_types'][:4]
        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.empty(text['hidden_size'],device='meta',dtype=torch.bfloat16))
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.empty((text['hidden_size'],text['linear_num_value_heads']*text['linear_value_head_dim']//2),device='meta',dtype=torch.int8 if quantized else torch.bfloat16),requires_grad=False)
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__();self.num_heads=text['num_attention_heads']//2;self.num_kv_heads=text['num_key_value_heads']//2;self.head_dim=text['head_dim']
                self.rotary_emb=torch.nn.Module();self.rotary_emb.rotary_dim=int(text['head_dim']*text['partial_rotary_factor']);self.rotary_emb.mrope_section=text['rope_parameters']['mrope_section'];self.rotary_emb.mrope_interleaved=text['rope_parameters']['mrope_interleaved']
        class Decoder(torch.nn.Module):
            def __init__(self,kind):
                super().__init__();self.layer_type=kind;self.input_layernorm=Norm()
                if kind=='full_attention':self.self_attn=Attention()
                else:self.linear_attn=torch.nn.Module();self.linear_attn.out_proj=Linear()
        model=torch.nn.Module();model.unusual_container=torch.nn.ModuleList([Decoder(k) for k in text['layer_types']])
        return model,config,dict(decoder_type=Decoder,norm_type=Norm,attention_type=Attention,linear_type=Linear,tp=2)

    def test_actual_objects_discovered_without_layer_names(self):
        m,c,types=self.fixture();rows=discover_qwen35_sites(m,c,**types)
        self.assertEqual(sum(r['mechanism']=='18' for r in rows),4)
        self.assertEqual(sum(r['mechanism']=='22' for r in rows),1)
        self.assertEqual(sum(r['mechanism']=='39' for r in rows),3)
        self.assertIs(rows[0]['module'],m.unusual_container[0].input_layernorm)
        self.assertTrue(all(r['path'].startswith('unusual_container.') for r in rows))

    def test_quantized_projection_is_not_bf16_candidate(self):
        m,c,types=self.fixture(quantized=True);rows=discover_qwen35_sites(m,c,**types)
        self.assertTrue(all(r['status'].startswith('INAPPLICABLE') for r in rows if r['mechanism']=='39'))

    def test_wrong_fused_geometry_refused(self):
        m,c,types=self.fixture();m.unusual_container[3].self_attn.rotary_emb.rotary_dim=128
        with self.assertRaisesRegex(ValueError,'MROPE_GEOMETRY'):discover_qwen35_sites(m,c,**types)

    def test_missing_or_wrong_order_not_selected(self):
        m,c,types=self.fixture();m.unusual_container[0].layer_type='full_attention'
        with self.assertRaisesRegex(ValueError,'LAYER_TYPE_ORDER'):discover_qwen35_sites(m,c,**types)

if __name__=='__main__':unittest.main()
