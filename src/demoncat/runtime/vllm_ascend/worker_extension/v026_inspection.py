"""Read-only 0.26 source and live-request audit, separate from 0.18 admission.

No report, config name or source pin enables an ordinary task. Binding and NPU
qualification remain pending until the new execution observer is integrated.
"""
import hashlib
import json
from pathlib import Path
from .prerequisites import assessment, expectation, observed_field, _MISSING

PROFILE = Path(__file__).with_name('v026_source_review.json')


def inspect_sources(roots):
    checks = []
    for row in json.loads(PROFILE.read_bytes())['files']:
        root = roots.get(row['repository'])
        path = None if root is None else Path(root) / row['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else _MISSING
        checks.append(expectation('deployment_identity', row['repository'] + '/' + row['path'],
                                  row['sha256'], actual, 'V026_SOURCE_NOT_REVIEWED'))
    return assessment(checks)


def parallel_assessment(runner, *, pcp_group, dcp_group, tp_group, expected_tp, source_profile="v026"):
    if type(expected_tp) is not int or expected_tp not in (1, 2):
        raise ValueError('V026_ONLY_TP1_TP2_REVIEWED')
    if source_profile not in ('v026','v026_installed_c9c9432b','preferred_a2_85a2bc5e'):
        raise ValueError('NEWSTACK_PARALLEL_PROFILE_NOT_REVIEWED')
    checks = []
    def check(path, expected, actual):
        checks.append(expectation('mechanism_compatibility', path, expected, actual,
                                  'V026_PARALLEL_FACT_MISSING_OR_DIFFERENT'))
    for name, expected in (('prefill_context_parallel_size', 1), ('decode_context_parallel_size', 1),
                           ('tensor_parallel_size', expected_tp), ('pipeline_parallel_size', 1),
                           ('data_parallel_size', 1)):
        path = 'vllm_config.parallel_config.' + name
        check(path, expected, observed_field(runner, path))
    # Runner's dcp_size catches exceptions and defaults to 1 in upstream. It is
    # not sufficient: corroborate both config and initialized group objects.
    for name, group, size in (('pcp', pcp_group, 1), ('dcp', dcp_group, 1), ('tp', tp_group, expected_tp)):
        check(name + '_group.world_size', size, observed_field(group, 'world_size'))
    check('dcp_size', 1, observed_field(runner, 'dcp_size'))
    if source_profile in ('v026','v026_installed_c9c9432b'):
        check('use_dcp', False, observed_field(runner, 'use_dcp'))
    else:
        # This pinned 0.23 runner has no use_dcp; its source-defined use_cp
        # property is pcp_size*dcp_size>1. Both sizes and actual groups required.
        check('pcp_size', 1, observed_field(runner, 'pcp_size'))
        check('use_cp', False, observed_field(runner, 'use_cp'))
    return assessment(checks)


def inspect_single_chunk(runner, scheduler_output, expected_prompts):
    """Called after _update_states, before model forward, on actual CPU mirrors.

    This is a necessary mapping check, not visual-path exclusion or permission.
    No copy of old request state and no config-only single-chunk proof suffices.
    """
    checks = []
    def check(path, expected, actual):
        checks.append(expectation('actual_call', path, expected, actual,
                                  'V026_FRESH_TEXT_SINGLE_CHUNK_UNPROVEN'))
    if not isinstance(expected_prompts, dict) or not expected_prompts or any(
        not isinstance(k, str) or not isinstance(v, (list, tuple)) or not v
        or any(type(token) is not int or token < 0 for token in v) for k, v in expected_prompts.items()):
        raise ValueError('EXACT_NONEMPTY_INTERNAL_REQUEST_PROMPTS_REQUIRED')
    batch = observed_field(runner, 'input_batch')
    ids = observed_field(batch, 'req_ids')
    indexes = observed_field(batch, 'req_id_to_index')
    computed = observed_field(batch, 'num_computed_tokens_cpu')
    requests = observed_field(runner, 'requests')
    scheduled = observed_field(scheduler_output, 'num_scheduled_tokens')
    expected_ids = set(expected_prompts)
    check('input_batch.request_set', sorted(expected_ids), sorted(ids) if type(ids) is list and all(type(x) is str for x in ids) else _MISSING)
    check('input_batch.unique_ids', len(expected_ids), len(ids) if type(ids) is list else _MISSING)
    check('scheduler.num_scheduled_tokens', {k: len(v) for k,v in expected_prompts.items()}, scheduled)
    check('scheduler.total_num_scheduled_tokens', sum(map(len, expected_prompts.values())),
          observed_field(scheduler_output, 'total_num_scheduled_tokens'))
    for name in ('scheduled_encoder_inputs', 'scheduled_spec_decode_tokens'):
        check('scheduler.'+name, {}, observed_field(scheduler_output, name))
    check('chunked_prefill.enabled', True, observed_field(runner, 'vllm_config.scheduler_config.enable_chunked_prefill'))
    budget = observed_field(runner, 'vllm_config.scheduler_config.max_num_batched_tokens')
    check('chunked_prefill.sufficient_budget', True,
          type(budget) is int and budget >= sum(map(len, expected_prompts.values())) if budget is not _MISSING else _MISSING)
    for request_id, prompt in expected_prompts.items():
        req = requests.get(request_id, _MISSING) if isinstance(requests, dict) else _MISSING
        check(request_id+'.prompt_token_ids', list(prompt), observed_field(req, 'prompt_token_ids'))
        check(request_id+'.mm_features', [], observed_field(req, 'mm_features'))
        index = indexes.get(request_id) if isinstance(indexes, dict) else None
        valid = type(index) is int and type(ids) is list and 0 <= index < len(ids) and ids[index] == request_id
        check(request_id+'.row_mapping', True, valid)
        value = _MISSING
        if valid and computed is not _MISSING:
            try:
                raw = computed[index]
                # numpy CPU scalar is accepted only if its integer dtype is proven.
                value = raw if type(raw) is int else int(raw) if getattr(getattr(raw, 'dtype', None), 'kind', None) in ('i','u') else _MISSING
            except (IndexError, TypeError, ValueError):
                pass
        check(request_id+'.num_computed_tokens', 0, value)
    result = assessment(checks)
    result.update(visual_execution_excluded=False, model_input_association_verified=False, source_binding_required=True,
                  qualification_granted=False, public_admission_changed=False)
    return result


def describe_qwen35_structure(config, quantization):
    """Config-driven candidate roles. Actual calls and types must still match."""
    text = config['text_config']
    layers = text['layer_types']
    if config['model_type'] != 'qwen3_5' or len(layers) != text['num_hidden_layers'] or any(
            kind not in ('linear_attention', 'full_attention') for kind in layers):
        raise ValueError('QWEN35_LAYER_STRUCTURE_INCONSISTENT')
    full = [i for i,kind in enumerate(layers) if kind == 'full_attention']
    linear = [i for i,kind in enumerate(layers) if kind == 'linear_attention']
    rope = text['rope_parameters']
    counts = {name: sum(value == name for value in quantization.values()) for name in ('W8A8_DYNAMIC','FLOAT')}
    outputs = {key: value for key,value in quantization.items() if '.linear_attn.out_proj.weight' in key and key.endswith('.weight')}
    return dict(schema='demoncat.qwen35.structure_candidates.v1', full_attention_layers=full,
        linear_attention_layers=linear, hidden_size=text['hidden_size'], head_dim=text['head_dim'],
        rotary_dim=int(text['head_dim']*text['partial_rotary_factor']),
        mrope_section=rope['mrope_section'], mrope_interleaved=rope['mrope_interleaved'],
        quantization_counts=counts, linear_out_proj_quantization=outputs,
        candidates={'18':['AscendGemmaRMSNorm', 'triton_split_qkv_rmsnorm_mrope'],
                    '39':['linear_attn.out_proj'], '22':['triton_split_qkv_rmsnorm_mrope'],
                    'Batch':['DynamicQuant + QuantBatchMatmulV3'],
                    '17':['visual.patch_embed.proj'], '41':['full_attention KV write']},
        actual_calls_observed=False, qualification_granted=False, execution_authorized=False)


def discover_qwen35_sites(model, config, *, decoder_type, norm_type, attention_type, linear_type, tp):
    """Walk real module objects; names are output evidence, never user selectors.

    Types must be imported by the trusted, source-checked deployment adapter.
    Discovery alone does not install hooks or authorize a mechanism.
    """
    if type(tp) is not int or tp not in (1, 2):raise ValueError('QWEN35_TP1_TP2_ONLY')
    text=config['text_config'];describe_qwen35_structure(config,{})
    named=list(model.named_modules());paths={id(module):name for name,module in named}
    blocks=[(name,module) for name,module in named if isinstance(module,decoder_type)]
    if len(blocks)!=text['num_hidden_layers']:raise ValueError('ACTUAL_DECODER_COUNT_DIFFERS')
    rows=[]
    for index,(path,block) in enumerate(blocks):
        kind=getattr(block,'layer_type',None)
        if kind!=text['layer_types'][index]:raise ValueError('ACTUAL_LAYER_TYPE_ORDER_DIFFERS')
        norm=getattr(block,'input_layernorm',None)
        if not isinstance(norm,norm_type) or tuple(norm.weight.shape)!=(text['hidden_size'],):
            raise ValueError('ACTUAL_GEMMA_NORM_GEOMETRY_DIFFERS')
        rows.append(dict(mechanism='18',path=paths[id(norm)],module=norm,boundary='AscendGemmaRMSNorm.forward_oot',
                         status='CANDIDATE_REQUIRES_ACTUAL_GEMMA_API_ADAPTATION'))
        if kind=='full_attention':
            attention=getattr(block,'self_attn',None)
            if not isinstance(attention,attention_type):raise ValueError('ACTUAL_FULL_ATTENTION_TYPE_DIFFERS')
            rope=getattr(attention,'rotary_emb',None)
            if (getattr(attention,'num_heads',None)!=text['num_attention_heads']//tp
                or getattr(attention,'num_kv_heads',None)!=text['num_key_value_heads']//tp
                or getattr(attention,'head_dim',None)!=text['head_dim']
                or getattr(rope,'rotary_dim',None)!=int(text['head_dim']*text['partial_rotary_factor'])
                or getattr(rope,'mrope_section',None)!=text['rope_parameters']['mrope_section']
                or getattr(rope,'mrope_interleaved',None)!=text['rope_parameters']['mrope_interleaved']):
                raise ValueError('ACTUAL_FUSED_MROPE_GEOMETRY_DIFFERS')
            rows.append(dict(mechanism='22',path=paths[id(attention)],module=attention,
                boundary='triton_split_qkv_rmsnorm_mrope',status='CANDIDATE_REQUIRES_FUSED_API_ADAPTATION'))
        else:
            attention=getattr(block,'linear_attn',None);projection=getattr(attention,'out_proj',None)
            if not isinstance(projection,linear_type) or id(projection) not in paths:
                raise ValueError('ACTUAL_LINEAR_OUTPUT_PROJECTION_MISSING')
            weight=getattr(projection,'weight',None)
            bf16=str(getattr(weight,'dtype',None))=='torch.bfloat16'
            rows.append(dict(mechanism='39',path=paths[id(projection)],module=projection,
                boundary='linear_attn.out_proj',status='CANDIDATE_REQUIRES_MATMULV2_CALL' if bf16
                else 'INAPPLICABLE_TO_BF16_MATMULV2_WEIGHT_DTYPE',weight_dtype=str(getattr(weight,'dtype',None))))
    return rows
