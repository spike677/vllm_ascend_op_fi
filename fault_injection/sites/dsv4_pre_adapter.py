"""Closed V4-0731 adapter preparation; no hook, permit or execution grant.

Only actual typed operator observations and explicit consumer/scheduler joins
may produce a plan. Cache decoding is intentionally absent until D1 evidence.
"""
import math

SUPPORTED=('18','22','39op','39mr','28')

def prepare_site(mechanism, observation, source_fingerprints):
    if mechanism not in SUPPORTED:raise ValueError('DSV4_MECHANISM_NOT_PREPARED')
    if not source_fingerprints or any(type(x) is not str or len(x)!=64 for x in source_fingerprints.values()):
        raise ValueError('DSV4_SOURCE_FINGERPRINT_REQUIRED')
    if observation.get('source_fingerprints')!=source_fingerprints:raise ValueError('DSV4_SOURCE_DRIFT')
    if not observation.get('scheduler_rows') or not observation.get('consumer_join'):
        raise ValueError('DSV4_REAL_REQUEST_AND_CONSUMER_JOIN_REQUIRED')
    op=observation.get('operator');inputs=observation.get('inputs',[]);outputs=observation.get('outputs',[])
    plan=dict(mechanism=mechanism,operator=op,qualification_granted=False,public_enabled=False,
              source_fingerprints=dict(source_fingerprints),scheduler_rows=observation['scheduler_rows'],consumer_join=observation['consumer_join'])
    if mechanism=='18':
        if op=='npu_rms_norm_cast':
            if (len(outputs)!=2 or outputs[0]['dtype']!='torch.bfloat16' or outputs[1]['dtype']!='torch.float32'
                    or outputs[0]['shape']!=outputs[1]['shape']):raise ValueError('DSV4_RMS_CAST_PAIR_REQUIRED')
            plan.update(method='region_rewrite',targets=[0,1],reason='BF16 activation and exact FP32 routing consumer are separate outputs; preserve pair semantics')
        elif op=='RmsNorm' and outputs and outputs[0]['dtype']=='torch.bfloat16':
            plan.update(method='region_rewrite',targets=[0])
        else:raise ValueError('DSV4_NORM_ROUTE_UNSUPPORTED')
    elif mechanism=='22':
        if op!='inplace_partial_rotary_mul' or not inputs:raise ValueError('DSV4_ROTARY_ROUTE_UNSUPPORTED')
        start,end=observation.get('partial_slice',(None,None))
        shape=inputs[0]['shape']
        if (not shape or type(start) is not int or type(end) is not int or not 0<=start<end<=shape[-1]
                or (end-start)%2 or observation.get('rotary_mode')!='interleave'):
            raise ValueError('DSV4_ROTARY_SLICE_UNPROVEN')
        if observation.get('cos_sin_row_join') is not True:raise ValueError('DSV4_ROTARY_ROW_JOIN_UNPROVEN')
        plan.update(method='index_perturb',partial_slice=[start,end],inplace_input=0)
    elif mechanism in ('39op','39mr'):
        if op not in ('MatMulV2','QuantMatmulWeightNz'):raise ValueError('DSV4_LINEAR_ROUTE_UNSUPPORTED')
        if not outputs or outputs[0].get('dtype')!='torch.bfloat16' or outputs[0].get('format')!='ND':
            raise ValueError('DSV4_LINEAR_BF16_ND_REQUIRED')
        plan.update(method='region_rewrite' if mechanism=='39op' else 'reference_recompute',targets=[0])
        if mechanism=='39mr':
            contract=observation.get('matmul_contract',{})
            if not all(contract.get(k) is True for k in ('logical_k_join','weight_layout_decoded','scales_joined','bias_joined')):
                raise ValueError('DSV4_MATMUL_ARITHMETIC_JOIN_UNPROVEN')
            plan['matmul_contract']=dict(contract)
    else:
        if op not in ('npu_sparse_attn_sharedkv','SparseFlashMla'):raise ValueError('DSV4_SPARSE_ROUTE_UNSUPPORTED')
        layout=observation.get('kv_layout');keys=observation.get('legal_key_indices')
        if layout not in ('PA_ND','PA_BBND') or observation.get('compressed_cache_decoded') is not True:
            raise ValueError('DSV4_COMPRESSED_LAYOUT_UNPROVEN')
        if (type(keys) is not list or not keys or any(type(k) is not int or k<0 for k in keys)
                or len(set(keys))!=len(keys) or observation.get('causal_request_join') is not True):
            raise ValueError('DSV4_SPARSE_VALID_KEY_SET_UNPROVEN')
        scale=observation.get('softmax_scale')
        if type(scale) not in (float,int) or not math.isfinite(scale) or scale<=0:raise ValueError('DSV4_SCALE_UNPROVEN')
        plan.update(method='reference_recompute',kv_layout=layout,legal_key_indices=list(keys),softmax_scale=scale)
    return plan
