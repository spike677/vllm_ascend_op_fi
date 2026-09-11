"""Versioned structural rules propose roles; they never grant execution proof."""
from __future__ import annotations

REGISTRY = {
    'qwen2': ('qwen_dense_v1', 'Qwen2ForCausalLM', 'Qwen2'),
    'qwen3': ('qwen_dense_v1', 'Qwen3ForCausalLM', 'Qwen3'),
    'qwen2_moe': ('qwen_moe_v1', 'Qwen2MoeForCausalLM', 'Qwen2Moe'),
    'qwen3_moe': ('qwen_moe_v1', 'Qwen3MoeForCausalLM', 'Qwen3Moe'),
    'deepseek_v2': ('deepseek_mla_v1', 'DeepseekV2ForCausalLM', 'DeepseekV2'),
    'deepseek_v3': ('deepseek_mla_v1', 'DeepseekV3ForCausalLM', 'DeepseekV2'),
    'deepseek_v4': ('deepseek_v4_v1', 'DeepseekV4ForCausalLM', 'DeepseekV4'),
}
LINEAR_CLASSES = {'QKVParallelLinear', 'RowParallelLinear', 'MergedColumnParallelLinear',
                  'ColumnParallelLinear', 'ReplicatedLinear', 'AscendQKVParallelLinear',
                  'AscendRowParallelLinear', 'AscendMergedColumnParallelLinear',
                  'AscendColumnParallelLinear', 'AscendReplicatedLinear'}
QWEN_ATTENTION = {'qkv_proj': ['ATTN_QKV'], 'q_proj': ['ATTN_Q'],
                  'k_proj': ['ATTN_K'], 'v_proj': ['ATTN_V'], 'o_proj': ['ATTN_OUT']}
MLP = {'gate_up_proj': ['MLP_GATE_UP'], 'gate_proj': ['MLP_GATE'],
       'up_proj': ['MLP_UP'], 'down_proj': ['MLP_DOWN']}
MLA = {'q_a_proj': ['ATTN_Q_A'], 'q_b_proj': ['ATTN_Q_B'], 'q_proj': ['ATTN_Q'],
       'kv_a_proj_with_mqa': ['ATTN_KV_A'], 'kv_b_proj': ['ATTN_KV_B'],
       'fused_qkv_a_proj': ['ATTN_Q_A', 'ATTN_KV_A'], 'o_proj': ['ATTN_OUT']}
V4 = {'fused_wqa_wkv': ['ATTN_Q_A', 'ATTN_KV_A'], 'wq_a': ['ATTN_Q_A'],
      'wq_b': ['ATTN_Q_B'], 'wkv': ['ATTN_KV_A'], 'wo_a': ['ATTN_OUT_A'], 'wo_b': ['ATTN_OUT_B']}


def select_adapter(descriptor):
    config = descriptor.get('config', {})
    rule = REGISTRY.get(config.get('model_type'))
    if rule is None:
        return {'status': 'UNSUPPORTED_ARCHITECTURE', 'adapter_id': None}
    adapter, arch, prefix = rule
    actual = descriptor.get('model_class', '').rsplit('.', 1)[-1]
    # vLLM deliberately implements V3 using the V2 runtime class in some builds.
    allowed = {arch} | ({'DeepseekV2ForCausalLM'} if config['model_type'] == 'deepseek_v3' else set())
    if config.get('architectures') != [arch] or actual not in allowed:
        return {'status': 'IDENTITY_IMPLEMENTATION_MISMATCH', 'adapter_id': adapter}
    return {'status': 'IDENTIFIED', 'adapter_id': adapter, 'class_prefix': prefix,
            'family': 'qwen' if adapter.startswith('qwen') else 'deepseek',
            'qualification': ('CURRENT_EPOCH_IDENTITY_REQUIRED' if adapter == 'qwen_dense_v1'
                              else 'READONLY_UNTIL_VARIANT_LIVE_VALIDATED')}


def structural_targets(descriptor, modules):
    adapter = select_adapter(descriptor)
    if adapter['status'] != 'IDENTIFIED':
        return adapter, [], [{'reason': adapter['status']}]
    prefix = adapter['class_prefix']
    lookup = {m['module_name']: m for m in modules}
    targets, gaps = [], []
    for row in modules:
        name, cls = row['module_name'], row['module_class'].rsplit('.', 1)[-1]
        parent = lookup.get(row.get('parent_name'), {})
        pc = parent.get('module_class', '').rsplit('.', 1)[-1]
        leaf = name.rsplit('.', 1)[-1]
        roles, scope = [], 'projection_input'
        # Both parent implementation and its position are required; indexer/compressor
        # children do not inherit the main attention's projection role.
        main_attn = row.get('parent_name', '').endswith(('.self_attn', '.attn'))
        if pc.startswith(prefix) and ('Attention' in pc or 'MLA' in pc) and main_attn:
            roles = (QWEN_ATTENTION if adapter['family'] == 'qwen' else
                     V4 if adapter['adapter_id'] == 'deepseek_v4_v1' else MLA).get(leaf, [])
        elif pc in ({prefix + 'MLP'} | ({'Qwen2MLP'} if adapter['adapter_id'] == 'qwen_dense_v1' else set())) and parent.get('module_name', '').endswith(('.mlp', '.shared_experts')):
            roles = MLP.get(leaf, [])
            if parent['module_name'].endswith('.shared_experts'):
                roles = ['SHARED_' + r for r in roles]
        elif cls in {prefix + 'SparseMoeBlock', prefix + 'MoE'} and leaf == 'mlp':
            roles, scope = ['MOE_BLOCK_INPUT'], 'outer_moe_input_not_expert_projection'
        elif any(part in ('compressor', 'indexer', 'experts') for part in name.split('.')):
            if leaf in set(V4) | set(MLP) | {'fused_wkv_wgate', 'experts'}:
                gaps.append({'module_name': name, 'layer': row.get('layer_index'),
                             'reason': 'FUSED_OR_AUXILIARY_BOUNDARY_REQUIRES_SEPARATE_CONTRACT'})
        if roles:
            mro = {c.rsplit('.', 1)[-1] for c in row.get('class_mro', [])}
            class_ok = bool(mro & LINEAR_CLASSES) if scope == 'projection_input' else True
            targets.append({**row, 'roles': roles, 'scope': scope,
                            'class_contract': class_ok,
                            'role_proof': [adapter['adapter_id'], pc, cls, leaf],
                            'semantic_args': ['x', 'input_', 'hidden_states', 'intermediate_states']})
    return adapter, targets, gaps


def expected_role_gaps(descriptor, modules, targets, selected_roles):
    """Expected sets come from the adapter, not runner layer-count multiplication."""
    adapter = select_adapter(descriptor)
    layers = descriptor.get('config', {}).get('num_hidden_layers')
    if type(layers) is not int or layers <= 0:
        return [{'reason': 'CONFIG_LAYER_COUNT_UNKNOWN'}]
    gaps = []
    if descriptor.get('owned_layers') != list(range(layers)):
        gaps.append({'reason': 'OWNED_LAYER_SET_MISMATCH_OR_UNSUPPORTED_PP'})
    if adapter.get('adapter_id') == 'qwen_dense_v1':
        # This dense rule explicitly expects the four packed projection boundaries.
        for layer in range(layers):
            for role in selected_roles:
                found = [t for t in targets if t['layer_index'] == layer and role in t['roles']]
                if len(found) != 1:
                    gaps.append({'layer': layer, 'role': role, 'reason': 'EXPECTED_ROLE_MISSING_OR_DUPLICATE'})
    else:
        gaps.append({'reason': 'VARIANT_COVERAGE_CONTRACT_NOT_LIVE_VALIDATED'})
    return gaps
