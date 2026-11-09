"""Registered FIA attention inside the actual HF Qwen3; no model math rewrite.

Both arms use this same route. Aliases expose actual Q/K/V/FIA values without
changing their values. Alias/no-alias logits neutrality is verified separately.
"""
import torch

ROUTE = 'fault28_fia_fullgraph'
FIA = torch.ops.npu.npu_fused_infer_attention_score.default


def attention(module, query, key, value, attention_mask, dropout=0.0,
              scaling=None, fault28_observations=None, **kwargs):
    batch, heads, length, dim = query.shape
    assert batch in (1, 8) and heads == 16 and length == 2048 and dim == 128
    assert tuple(key.shape) == tuple(value.shape) == (batch, 8, length, dim)
    assert dropout == 0.0 and scaling == 128 ** -0.5
    assert attention_mask.dtype == torch.bool and tuple(attention_mask.shape) == (2048, 2048)
    q, k, v = query.contiguous(), key.contiguous(), value.contiguous()
    output, _ = FIA(q, k, v, atten_mask=attention_mask,
        actual_seq_lengths=[length] * batch, actual_seq_lengths_kv=[length] * batch,
        num_heads=16, num_key_value_heads=8, scale=scaling, input_layout='BNSD',
        pre_tokens=2147483647, next_tokens=0, sparse_mode=2, inner_precise=0)
    if fault28_observations is not None:
        fault28_observations.extend((q, k, v, output))
    return output.transpose(1, 2), None


def install(model):
    from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
    assert model.config.num_hidden_layers == 28
    assert model.config.num_attention_heads == 16 and model.config.num_key_value_heads == 8
    assert model.config.head_dim == 128 and set(model.config.layer_types) == {'full_attention'}
    ALL_ATTENTION_FUNCTIONS.register(ROUTE, attention)
    model.config._attn_implementation = ROUTE
    for index, layer in enumerate(model.model.layers):
        assert layer.self_attn.layer_idx == index and layer.self_attn.config is model.config


class FullModelGraph(torch.nn.Module):
    def __init__(self, model, aliases=True):
        super().__init__()
        self.model = model
        self.aliases = aliases

    def forward(self, input_ids, position_ids, causal_mask):
        observations = [] if self.aliases else None
        reply = self.model(input_ids=input_ids, position_ids=position_ids,
            attention_mask={'full_attention': causal_mask}, use_cache=False,
            logits_to_keep=1, return_dict=True, fault28_observations=observations)
        if not self.aliases:
            return (reply.logits,)
        return (reply.logits, *observations)
