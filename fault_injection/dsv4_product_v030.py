"""Closed V4-0.30 D-product preparation. Not a public registration or enable route.

Called by future trusted D binding only after D1 source/site review. Existing
Qwen binding/request limits and exact canary are deliberately not modified.
"""
from copy import deepcopy
from fault_injection.product_contract import request_range

def require(ok, code):
    if not ok:raise ValueError(code)

def request_policy(facts, deployment):
    result=request_range(facts,max_context=4096,visual=False)
    problems=result['reasons']
    n=facts.get('prompt_length')
    if type(n) is not int or not 2500<=n<=4092:problems.append('D030_PROMPT_OUTSIDE_2500_4092')
    if facts.get('max_tokens')!=4 or type(facts.get('max_tokens')) is not int:problems.append('D030_FOUR_TOKENS_REQUIRED')
    if type(n) is int:
        if type(deployment.get('max_model_len')) is not int or n+4>deployment['max_model_len']:problems.append('D030_DEPLOYMENT_CONTEXT_TOO_SHORT')
        if type(deployment.get('max_num_batched_tokens')) is not int or n>deployment['max_num_batched_tokens']:problems.append('D030_SINGLE_CHUNK_BUDGET_UNPROVEN')
    return dict(eligible=not problems,reasons=problems,qualification_granted=False)

def validate_group(rows):
    require(type(rows) is list and len(rows)==8,'D030_EIGHT_RUNTIME_RANKS_REQUIRED')
    require({r.get('global_rank') for r in rows}==set(range(8)),'D030_RANK_SET_MISMATCH')
    for row in rows:
        rank=row['global_rank']
        require(type(rank) is int,'D030_RANK_TYPE')
        for name in ('tp','ep','mc2'):
            group=row.get(name)
            require(type(group) is dict,'D030_RUNTIME_GROUP_MISSING:'+name)
            ranks=group.get('members');local=group.get('local_rank')
            require(type(ranks) is list and ranks and all(type(x) is int for x in ranks)
                    and len(set(ranks))==len(ranks) and set(ranks)<=set(range(8)),
                    'D030_GROUP_MEMBERS_UNPROVEN:'+name)
            require(type(local) is int and 0<=local<len(ranks) and ranks[local]==rank,'D030_LOCAL_RANK_MISMATCH:'+name)
            require(type(group.get('source')) is str and bool(group['source']),'D030_GROUP_SOURCE_MISSING:'+name)
            if name=='tp':require(len(ranks)==8,'D030_TP8_REQUIRED')
            else:require(len(ranks)>1,'D030_EP_RUNTIME_REQUIRED:'+name)
            for peer in ranks:
                other=next(x for x in rows if x['global_rank']==peer).get(name,{})
                require(other.get('members')==ranks,'D030_GROUP_MEMBERS_DISAGREE:'+name)
    return deepcopy(rows)

def prepare_route(identity, runtime_groups):
    require(identity.get('vllm') in ('0.30.0','0.30.0+empty')
            and identity.get('vllm_ascend')=='0.30.0'
            and identity.get('cann')=='9.2.1','D030_SOFTWARE_IDENTITY')
    require(identity.get('tp')==8 and type(identity.get('tp')) is int
            and identity.get('ep') is True and identity.get('dp')==1,'D030_TP8_EP_DP1_REQUIRED')
    for field in ('model_config_sha256','quant_model_description_sha256','runtime_source_sha256'):
        value=identity.get(field)
        require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'D030_IDENTITY_MISSING:'+field)
    indexes=identity.get('safetensors_index_sha256')
    require(type(indexes) is dict and indexes and all(type(k) is str and k.endswith('.safetensors.index.json')
        and type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v) for k,v in indexes.items()),'D030_INDEX_IDENTITY_REQUIRED')
    groups=validate_group(runtime_groups)
    return dict(schema='dcat.dsv4.v030.product.preparation.v1',identity=deepcopy(identity),
                runtime_groups=groups,mode='all',request_range=dict(min_prompt=2500,max_prompt=4092,max_tokens=4,max_context=4096,single_chunk=True,max_num_seqs=1),
                noise_rule='demoncat.dsv4_0731.v030.noise.v1',executable=False,
                blockers=['D1_ROUTER_SITE_AND_REQUEST_JOIN_REVIEW','D030_BINDING_AND_AUTOMATIC_OBSERVER_INTEGRATION',
                          'NOISE_CANARY_LIFECYCLE_INTEGRATION','A3_TWO_OWNER_ACCEPTANCE'],
                qualification_granted=False)

def enable_prepared_route(_):
    # No boolean supplied by a caller may bypass pending source/lifecycle review.
    raise ValueError('D030_PRODUCT_ROUTE_NOT_YET_REVIEWED')

