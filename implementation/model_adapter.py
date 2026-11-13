"""Actual eager Qwen3 FIA route, resolving the public API after msprobe hooks."""
import json
import time
import torch
import torch_npu

ROUTE = 'fault28_fia_eager'
STATE = {'forward': None, 'calls': []}


def native_fia(layer, q, k, v, mask, scaling):
    assert not torch.compiler.is_compiling()
    index = len(STATE['calls'])
    record = {'schema': 'fault28.eager.api_marker.v1', 'event': 'enter',
              'api_index': index, 'forward': STATE['forward'], 'layer': layer,
              'api': 'torch_npu.npu_fused_infer_attention_score',
              'input_shapes': [list(t.shape) for t in (q,k,v)],
              'input_strides': [list(t.stride()) for t in (q,k,v)],
              'input_dtypes': [str(t.dtype) for t in (q,k,v)], 'time': time.time()}
    torch.npu.synchronize()
    print(json.dumps(record), flush=True)
    # Do not cache torch.ops or the Python callable before PrecisionDebugger
    # initializes: the native public API must itself appear in the raw dump.
    out, lse = torch_npu.npu_fused_infer_attention_score(q,k,v,
        atten_mask=mask,actual_seq_lengths=[q.shape[2]]*q.shape[0],
        actual_seq_lengths_kv=[k.shape[2]]*k.shape[0],num_heads=16,
        num_key_value_heads=8,scale=scaling,input_layout='BNSD',
        pre_tokens=2147483647,next_tokens=0,sparse_mode=2,inner_precise=0)
    torch.npu.synchronize()
    done = dict(record,event='exit',output_shape=list(out.shape),lse_shape=list(lse.shape),time=time.time())
    STATE['calls'].append(done)
    print(json.dumps(done), flush=True)
    return out


def attention(module, query, key, value, attention_mask, dropout=0.0,
              scaling=None, fault28_observations=None, **kwargs):
    assert not torch.compiler.is_compiling()
    b,n,s,d = query.shape
    assert b in (1,8) and (n,s,d)==(16,2048,128)
    assert tuple(key.shape)==tuple(value.shape)==(b,8,s,d)
    assert dropout==0 and scaling==128**-0.5
    assert attention_mask.dtype==torch.bool and tuple(attention_mask.shape)==(s,s)
    q,k,v = query.contiguous(),key.contiguous(),value.contiguous()
    out = native_fia(module.layer_idx,q,k,v,attention_mask,scaling)
    if fault28_observations is not None:
        fault28_observations.extend((q,k,v,out))
    return out.transpose(1,2),None


def install(model):
    from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS
    assert model.config.num_hidden_layers==28 and model.config.num_attention_heads==16
    assert model.config.num_key_value_heads==8 and model.config.head_dim==128
    ALL_ATTENTION_FUNCTIONS.register(ROUTE,attention)
    model.config._attn_implementation = ROUTE
    for i, layer in enumerate(model.model.layers):
        assert layer.self_attn.layer_idx==i and layer.self_attn.config is model.config


class FullModelEager(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, positions, mask):
        assert not torch.compiler.is_compiling()
        observations = []
        reply = self.model(input_ids=ids,position_ids=positions,
            attention_mask={'full_attention':mask},use_cache=False,logits_to_keep=1,
            return_dict=True,fault28_observations=observations)
        # Keep the exact live native-API aliases for the independent complete
        # array observer after debugger.stop(). The model wrapper returns its
        # actual logits, avoiding a second module-level statistics traversal
        # over every intermediate alias. No API/module hook is filtered out.
        STATE['observations'] = observations
        return reply.logits
