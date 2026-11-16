"""Small real architecture CPU prefill/decode checks, no downloads or NPU calls."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from fault_injection import inspect_model
from fault_injection.attention import AttentionSession, _resolve
from fault_injection.core import Decision
from fault_injection.fia import FiaBackend


def make_model(name):
    import torch
    import transformers as hf
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                  attention_dropout=0.0, pad_token_id=0)
    spec = {
        'qwen3_dense': (hf.Qwen3Config, hf.Qwen3ForCausalLM, dict(head_dim=8)),
        'qwen3_moe': (hf.Qwen3MoeConfig, hf.Qwen3MoeForCausalLM,
                      dict(head_dim=8, moe_intermediate_size=16, num_experts=4, num_experts_per_tok=2)),
        'glm': (hf.GlmConfig, hf.GlmForCausalLM, dict(head_dim=8)),
        'deepseek_v3_mla_moe': (hf.DeepseekV3Config, hf.DeepseekV3ForCausalLM,
            dict(num_key_value_heads=4, q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8,
                 qk_rope_head_dim=4, v_head_dim=8, moe_intermediate_size=16, n_shared_experts=1,
                 n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=0)),
        'deepseek_v2_mla_moe': (hf.DeepseekV2Config, hf.DeepseekV2ForCausalLM,
            dict(num_key_value_heads=4, q_lora_rank=None, kv_lora_rank=8, qk_nope_head_dim=8,
                 qk_rope_head_dim=4, v_head_dim=8, moe_intermediate_size=16, n_shared_experts=1,
                 n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, first_k_dense_replace=0)),
        'llama': (hf.LlamaConfig, hf.LlamaForCausalLM, {}),
    }
    config_class, model_class, extra = spec[name]
    torch.manual_seed(17)
    config = config_class(**(common | extra))
    config._attn_implementation = 'eager'
    return model_class(config).eval()


class TransparentCpuBackend:
    def __init__(self):
        self.calls = []

    def check(self, call):
        call.geometry()
        return Decision(True, 'CPU transparent route only; not a native fault provider')

    def run(self, call):
        resolved, _ = _resolve(call.module)
        self.calls.append(call)
        return resolved[0].get_interface('eager', resolved[1])(
            call.module, call.query, call.key, call.value, call.attention_mask,
            dropout=call.dropout, scaling=call.scaling, **call.kwargs)


def exercise(name):
    import torch
    model = make_model(name)
    inventory = inspect_model(model)
    identities = inventory['structure']['architecture_identity']
    roles = inventory['structure']['modules']
    executed = Counter()
    hooks = []
    for path, module in model.named_modules():
        def observed(_, args, output, path=path):
            executed[path] += 1
        hooks.append(module.register_forward_hook(observed))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [0, 0, 7, 8, 9, 10]])
    mask = (ids != 0).long()
    next_ids = torch.tensor([[11], [12]])
    decode_mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.long)), dim=1)
    with torch.inference_mode():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=True)
        baseline_decode = model(input_ids=next_ids, attention_mask=decode_mask,
                                past_key_values=baseline.past_key_values, use_cache=True)
        backend = TransparentCpuBackend()
        with AttentionSession(model, backend) as session:
            routed = model(input_ids=ids, attention_mask=mask, use_cache=True)
            routed_decode = model(input_ids=next_ids, attention_mask=decode_mask,
                                  past_key_values=routed.past_key_values, use_cache=True)
            coverage = session.assert_coverage()
        off = model(input_ids=ids, attention_mask=mask, use_cache=False)
    for hook in hooks:
        hook.remove()
    equal = dict(prefill=torch.equal(baseline.logits, routed.logits),
                 cached_decode=torch.equal(baseline_decode.logits, routed_decode.logits),
                 off=torch.equal(baseline.logits, off.logits))
    finite = all(bool(torch.isfinite(x).all()) for x in
                 (baseline.logits, baseline_decode.logits, routed.logits, routed_decode.logits, off.logits))
    actual_roles = [row for row in roles if row['canonical_roles'] and executed[row['path']] > 0]
    calls = [dict(module_type=type(c.module).__module__ + '.' + type(c.module).__name__,
                  geometry=c.geometry(), dtype=str(c.query.dtype), device=c.query.device.type,
                  native_fia_check=vars(FiaBackend().check(c))) for c in backend.calls]
    expected_layers = model.config.num_hidden_layers
    if not (all(equal.values()) and finite and len(calls) == 2 * expected_layers
            and all(not c['native_fia_check']['supported'] for c in calls)):
        raise AssertionError('CPU model adaptation contract failed: ' + name)
    return dict(model=name, architecture=identities, config=model.config.to_dict(),
                status='CPU_VERIFIED', model_structure='MODEL_STRUCTURE_VERIFIED_CPU',
                module_roles='MODULE_ROLE_VERIFIED_CPU', operator_path='CPU_SHARED_ATTENTION_OBSERVED',
                native_provider='UNVERIFIED', model_fault='UNVERIFIED',
                recovery='CPU_TRANSPARENT_ROUTE_RECOVERY_VERIFIED',
                transparent_exact=equal, finite=finite, requests=2,
                role_counts=dict(Counter(role for row in actual_roles for role in row['canonical_roles'])),
                roles=[dict(path=r['path'], aliases=r['aliases'], canonical_roles=r['canonical_roles'],
                            module_type=r['module_type'], executed_forward_calls=executed[r['path']])
                       for r in actual_roles],
                attention_candidates=len(inventory['candidates']), attention_calls=calls,
                observation_scope='Random initialized actual Transformers CPU models; no weights download, native injection or quality acceptance',
                actual_npu_calls=0)


def main():
    import torch
    import transformers
    p = argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if transformers.__version__ != '5.5.4':
        raise RuntimeError('This reproducible model matrix requires transformers==5.5.4')
    torch.set_num_threads(1)
    rows = [exercise(n) for n in ('qwen3_dense', 'qwen3_moe', 'glm', 'deepseek_v2_mla_moe', 'deepseek_v3_mla_moe', 'llama')]
    output = dict(schema='demoncat.overnight.model_matrix.v1', actual_npu_calls=0,
                  packages={n:importlib.metadata.version(n) for n in ('torch','transformers')},
                  models=rows, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    print(json.dumps(dict(status='PASS_CPU_ONLY', models=[r['model'] for r in rows], actual_npu_calls=0)))


if __name__ == '__main__':
    main()
