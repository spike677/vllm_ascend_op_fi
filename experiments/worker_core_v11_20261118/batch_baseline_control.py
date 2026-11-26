"""Three consecutive B8 rounds on a fresh service; no fault lifecycle calls."""
import hashlib,json,re,time
from pathlib import Path
from tools.product_effect_probe import canonical,compare,api_values,require
from tools.effect_rounds import same_composition,signature


def request_order(reply,trace):
    """Join actual engine IDs to API choice indices, not completion arrival order."""
    signature(trace,2)
    ordered=[]
    for rank in trace:
        ids={}
        for step in rank['steps']:
            for row in step['rows']:
                rid=row['request_id']
                match=re.fullmatch(re.escape(reply['id'])+r'-(\d+)(?:-[A-Za-z0-9]+)?',rid)
                require(match is not None,'BASELINE_API_ENGINE_ID_JOIN_UNPROVEN')
                index=int(match[1]);value=dict(index=index,request_id=rid,prompt_sha256=row['prompt_sha256'])
                require(index not in ids or ids[index]==value,'BASELINE_DUPLICATE_REQUEST_INDEX')
                ids[index]=value
        require(sorted(ids)==list(range(8)),'BASELINE_EIGHT_REQUESTS_REQUIRED')
        ordered.append([ids[i] for i in range(8)])
    require(ordered[0]==ordered[1],'BASELINE_RANK_REQUEST_JOIN_MISMATCH')
    require(sorted(c['index'] for c in reply['choices'])==list(range(8)),'BASELINE_API_CHOICES_MISSING')
    return ordered[0]


def run(ex,root):
    prompts=['Explain briefly '+t for t in ['weather','cities','plants','rivers','physics','music','bikes','oceans']]
    sampling=dict(temperature=0,seed=42,logprobs=5,max_tokens=4,min_tokens=4,ignore_eos=True,return_tokens_as_token_ids=True)
    raw=dict(schema='dcat.batch.uninjected.baselines.v1',injection_enabled=False,
        warmup_http_requests=0,round_count=3,prompts=prompts,sampling=sampling,rounds=[],comparisons=[],
        source_manifest_sha256=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest(),
        qualification_granted=False,control_calls=[])
    target=Path(root)/'batch_uninjected_baselines.json'
    def save():target.write_bytes(canonical(raw)+b'\n')
    for i in range(3):
        row=dict(index=i);raw['rounds'].append(row);save()
        raw['control_calls'].append('effect_round_start')
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('effect_round_start',{}))
        start=time.monotonic()
        try:row['output']=ex.post('/v1/completions',dict(sampling,model='acceptance',prompt=prompts))
        finally:
            row['seconds']=time.monotonic()-start
            raw['control_calls'].append('effect_round_stop')
            row['trace']=sorted(ex.collective_rpc('demoncat_acceptance_bootstrap',args=('effect_round_stop',{})),key=lambda r:r['rank'])
            save()
        row['request_order']=request_order(row['output'],row['trace']);api_values(row['output']);save()
    for i,j in ((0,1),(0,2),(1,2)):
        a,b=raw['rounds'][i],raw['rounds'][j]
        require([r['prompt_sha256'] for r in a['request_order']]==[r['prompt_sha256'] for r in b['request_order']],'BASELINE_PROMPTS_CHANGED')
        raw['comparisons'].append(dict(rounds=[i,j],same_actual_composition=same_composition(a['trace'],b['trace'],2),
            api=compare(a['output'],b['output']),per_request=[dict(index=k,
                request_ids=[a['request_order'][k]['request_id'],b['request_order'][k]['request_id']],
                api=compare(dict(choices=[next(c for c in a['output']['choices'] if c['index']==k)]),
                            dict(choices=[next(c for c in b['output']['choices'] if c['index']==k)]))) for k in range(8)]))
    raw['status']='THREE_ROUNDS_CAPTURED_NOT_A_QUALIFICATION';save()
    return raw
