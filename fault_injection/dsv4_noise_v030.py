"""Separate 0.30 V4-0731 noise review; frozen 0.32 reviewer is unchanged.

The rule is fixed, not fit to fault/off results. Qwen must retain exact review.
"""
from copy import deepcopy
from itertools import combinations
import hashlib,json,math

RULE='demoncat.dsv4_0731.v030.noise.v1'
SAFETY_FACTOR=1.5
METRICS=('hidden_max_abs','logits_max_abs','top5_rank_logprob_max_abs','top5_set_distance')
TARGET=dict(model='V4-Flash-0731-W8A8',hardware_family='A3',vllm_ascend='0.30.0',cann='9.2.1',tp=8,dp=1)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def identity(combination):
    require(all(type(combination.get(k)) is type(v) and combination[k]==v for k,v in TARGET.items()),'NOISE_COMBINATION_NOT_SUPPORTED')
    require(combination.get('vllm') in ('0.30.0','0.30.0+empty'),'NOISE_VLLM_030_REQUIRED')
    require(type(combination.get('ep')) is bool and isinstance(combination.get('ep_source'),str) and bool(combination['ep_source'].strip()),'NOISE_EP_RUNTIME_SOURCE_REQUIRED')
    for key in ('model_config_sha256','quant_model_description_sha256','runtime_source_sha256'):
        require(isinstance(combination.get(key),str) and len(combination[key])==64 and all(c in '0123456789abcdef' for c in combination[key]),'NOISE_IDENTITY_MISSING:'+key)
    indexes=combination.get('safetensors_index_sha256',{})
    require(bool(indexes) and all(isinstance(k,str) and k.endswith('.safetensors.index.json') and isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for k,v in indexes.items()),'NOISE_INDEX_IDENTITY_MISSING')
    require(type(combination.get('max_model_len')) is int and 2504<=combination['max_model_len']<=4096,'NOISE_CONTEXT_RANGE')
    require(type(combination.get('max_num_batched_tokens')) is int and combination['max_num_batched_tokens']>=2500 and type(combination.get('max_num_seqs')) is int and combination['max_num_seqs']>=1,'NOISE_SCHEDULER_IDENTITY_MISSING')


def validate_run(run, *, allow_nonfinite=False):
    def valid(x):return type(x) in (float,int) and math.isfinite(x) or allow_nonfinite and x in ("+inf","-inf","nan")
    require(isinstance(run.get('process_identity'),str) and run['process_identity'],'NOISE_PROCESS_IDENTITY_REQUIRED')
    require(isinstance(run.get('sample_id'),str) and run['sample_id'],'NOISE_SAMPLE_ID_REQUIRED')
    require(type(run.get('prompt_tokens')) is int and 2500<=run['prompt_tokens']<=4092 and run.get('single_chunk') is True,'NOISE_PROMPT_SINGLE_CHUNK')
    require(isinstance(run.get('prompt_sha256'),str) and len(run['prompt_sha256'])==64 and all(c in '0123456789abcdef' for c in run['prompt_sha256']),'NOISE_PROMPT_IDENTITY_REQUIRED')
    require(len(run.get('tokens',[]))==4 and all(type(t) is int for t in run['tokens']),'NOISE_FOUR_GENERATED_TOKENS_REQUIRED')
    ranks=run.get('ranks',{})
    require(set(ranks)=={str(i) for i in range(8)},'NOISE_ALL_RANKS_REQUIRED')
    for steps in ranks.values():
        require(len(steps)==4,'NOISE_FOUR_FORWARD_OBSERVATIONS_REQUIRED')
        for step in steps:
            for field in ('hidden','logits'):
                require(isinstance(step.get(field),list) and step[field] and all(valid(x) for x in step[field]),'NOISE_FINITE_COMPLETE_ARRAY_REQUIRED:'+field)
            top=step.get('top5',[])
            require(len(top)==5 and len({x['token'] for x in top})==5 and all(type(x['token']) is int and valid(x['logprob']) for x in top),'NOISE_TOP5_REQUIRED')
            require(any(type(x['logprob']) is str for x in top) or all(top[i]['logprob']>=top[i+1]['logprob'] for i in range(4)),'NOISE_TOP5_ORDER')


def step_distance(a,b):
    result={}
    for key in ('hidden','logits'):
        require(len(a[key])==len(b[key]),'NOISE_ARRAY_SHAPE_MISMATCH')
        result[key+'_max_abs']=None if any(type(x) is str for x in a[key]+b[key]) else max(abs(x-y) for x,y in zip(a[key],b[key]))
    result['top5_rank_logprob_max_abs']=None if any(type(x['logprob']) is str for x in a['top5']+b['top5']) else max(abs(x['logprob']-y['logprob']) for x,y in zip(a['top5'],b['top5']))
    x={v['token'] for v in a['top5']};y={v['token'] for v in b['top5']}
    result['top5_set_distance']=1-len(x&y)/len(x|y)
    return result


def calibrate(combination,runs):
    identity(combination)
    require(len(runs)>=10,'NOISE_TWO_PROCESSES_K5_REQUIRED')
    for r in runs:validate_run(r)
    require(len({(r['process_identity'],r['sample_id']) for r in runs})==len(runs),'NOISE_DUPLICATE_CALIBRATION_SAMPLE')
    groups={}
    for r in runs:groups.setdefault(r['process_identity'],[]).append(r)
    require(len(groups)==2 and all(len(g)>=5 for g in groups.values()),'NOISE_TWO_PROCESSES_K5_REQUIRED')
    require(len({(r['prompt_sha256'],r['prompt_tokens']) for r in runs})==1,'NOISE_CALIBRATION_PROMPTS_DIFFER')
    require(all(r['tokens']==runs[0]['tokens'] for r in runs),'NOISE_CALIBRATION_TOKENS_UNSTABLE')
    require(runs[0]['prompt_tokens']+4<=combination['max_model_len'] and runs[0]['prompt_tokens']<=combination['max_num_batched_tokens'],'NOISE_CONTEXT_RANGE')
    bounds={str(r):[{m:0.0 for m in METRICS} for _ in range(4)] for r in range(8)}
    for a,b in combinations(runs,2):
        for rank in bounds:
            for i,(x,y) in enumerate(zip(a['ranks'][rank],b['ranks'][rank])):
                for metric,value in step_distance(x,y).items():bounds[rank][i][metric]=max(bounds[rank][i][metric],value)
    maxima=deepcopy(bounds)
    for steps in bounds.values():
        for step in steps:
            for m in step:step[m]*=SAFETY_FACTOR
    return dict(schema=RULE,all_finite=True,tokens_stable=True,safety_factor=SAFETY_FACTOR,combination=deepcopy(combination),calibration_sha256=digest(runs),
        prompt_sha256=runs[0]['prompt_sha256'],prompt_tokens=runs[0]['prompt_tokens'],tokens=runs[0]['tokens'],
        runs=len(runs),process_counts={p:len(g) for p,g in groups.items()},observed_maxima=maxima,bounds=bounds,qualification_granted=False)


def review(combination,calibration,phases):
    rule=calibrate(combination,calibration) # caller cannot submit its own tolerance
    require(set(phases)=={'normal','fault','off','health'},'NOISE_PHASE_SET')
    for phase,r in phases.items():
        validate_run(r,allow_nonfinite=phase=='fault')
        require(r['prompt_sha256']==rule['prompt_sha256'] and r['prompt_tokens']==rule['prompt_tokens'],'NOISE_UNCALIBRATED_PROMPT')
    require(len({r['process_identity'] for r in phases.values()})==1,'NOISE_PHASE_WORKERS_CHANGED')
    normal=phases['normal'];require(normal['tokens']==rule['tokens'],'NOISE_NORMAL_TOKEN_MISMATCH')
    # An unstable new normal must not silently expand calibration bounds.
    comparisons={}
    for name,run in phases.items():
        reference=calibration[0] if name=='normal' else normal
        exceeded=[];maximum={m:0.0 for m in METRICS}
        for rank in rule['bounds']:
            for step,(a,b) in enumerate(zip(reference['ranks'][rank],run['ranks'][rank])):
                for m,v in step_distance(a,b).items():
                    maximum[m]=None if v is None or maximum[m] is None else max(maximum[m],v)
                    if v is None or v>rule['bounds'][rank][step][m]:exceeded.append(dict(rank=int(rank),step=step,field=m,difference=v,bound=rule['bounds'][rank][step][m]))
        token_exact=run['tokens']==normal['tokens']
        comparisons[name]=dict(token_exact=token_exact,max_difference=maximum,exceeded=exceeded,
            within_noise=token_exact and not exceeded)
    require(comparisons['normal']['within_noise'],'NOISE_NEW_NORMAL_OUTSIDE_CALIBRATION')
    recovery=all(phases[p].get('region_bytes_restored') is True and phases[p].get('resources')==dict(leases=0,hooks=0,inflight=0) for p in ('fault','off','health'))
    return dict(schema=RULE,calibration=rule,comparisons=comparisons,strict_storage_recovery=recovery,
        off_health_passed=recovery and all(comparisons[p]['within_noise'] for p in ('off','health')),
        model_effect_visible=bool(comparisons['fault']['exceeded']) or not comparisons['fault']['token_exact'],
        qualification_granted=False)
