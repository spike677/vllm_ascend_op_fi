"""Independent receipt review of real Batch site execution; no public grant."""
import argparse
import json
from pathlib import Path


def require(ok,reason):
    if not ok:raise ValueError(reason)


def review(report):
    require(report['status'].startswith('TWO_SCOPES'),'RUN_NOT_COMPLETE')
    warm=report['warmup']
    require(2<=len(warm)<=6 and all(r['finite'] for r in warm[-2:]),'WARMUP_NOT_FINITE')
    for rank in (0,1):
        require(warm[-1]['ranks'][rank]['output']==warm[-2]['ranks'][rank]['output'],'WARMUP_NOT_EXACT')
    require(len(report['scopes'])==2,'TWO_SCOPES_REQUIRED')
    identities=[];owners=[];scopes=[]
    for scope in report['scopes']:
        owners.append(scope['owner']);requests=scope['requests']
        require([x['phase'] for x in requests]==['normal','fault','off','health'],'PHASE_SEQUENCE')
        require(all(x['finite'] for x in requests),'NONFINITE_BATCH')
        for request in requests:
            require(len(request['ranks'])==len(request['installed'])==len(request['cleanup'])==2,'RANK_MISSING')
            ids={m['internal_request_id'] for m in request['mapping']}
            require(len(ids)==len(request['mapping'])>=2,'ACTUAL_MULTI_REQUEST_MAPPING_REQUIRED')
            for rank in (0,1):
                row=request['ranks'][rank]
                require(row['rank']==rank and not row['uncovered'],'UNOBSERVED_CALLS_OR_RANK_MISMATCH')
                require(len(row['requests'])==len(row['output'])==4,'FOUR_FORWARDS_REQUIRED')
                require(all(set(x['request_ids'])==ids for x in row['requests']),'REQUEST_MAP_MISMATCH')
                require(request['cleanup'][rank]['resources']==dict(leases=0,hooks=0,inflight=0),'RESOURCES_NOT_ZERO')
        ranks={}
        for rank in (0,1):
            n,f,o,h=[r['ranks'][rank] for r in requests]
            require(len({(r['pid'],r['model_id']) for r in (n,f,o,h)})==1,'WORKER_CHANGED')
            require(n['output']==o['output']==h['output'],'OFF_HEALTH_NOT_EXACT')
            require(n['output']!=f['output'],'NO_PROPAGATION')
            sites=set(requests[1]['installed'][rank]['sites']);events=f['events']
            require(len(sites)==f['site_count'] and {e['site'] for e in events}==sites,'SITE_SET_MISMATCH')
            for site in sites:
                require([e['method'] for e in events if e['site']==site]==['_forward_prefill']+['_forward_decode']*3,'CALL_COVERAGE_MISMATCH')
            ids={m['internal_request_id'] for m in requests[1]['mapping']}
            for e in events:
                require(set(e['request_ids'])==ids,'CALL_REQUEST_MAP_MISMATCH')
                require(all(e[x] for x in ('applied','consumed','restored','reference_exact','all_real_rows')) and not e['failed'],'CALL_NOT_COMPLETE')
                before=e['before_sha256']
                require(len(before)==2 and all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in before),'BYTE_HASH_REQUIRED')
                require(before==e['restored_sha256'],'BYTE_RESTORE_NOT_EXACT')
                require(e['qmm_calls'] in (1,2),'REAL_QMM_MISSING')
            counts=dict(observed=len(events),eligible=len(events),attempted=len(events),applied=len(events),
                changed=sum(e['changed']>0 for e in events),consumed=len(events),restored=len(events),failed=0)
            require(counts==f['coverage'] and counts['changed']>0,'COUNTS_MISMATCH_OR_NO_EFFECT')
            ranks[str(rank)]=dict(sites=len(sites),coverage=counts,off_health_exact=True,byte_restore_verified=True,
                hidden_changed=any(x['hidden']!=y['hidden'] for x,y in zip(n['output'],f['output'])),
                logits_changed=any(x['logits']!=y['logits'] for x,y in zip(n['output'],f['output'])))
        identities.append([(r['pid'],r['model_id']) for r in requests[0]['ranks']])
        scopes.append(dict(ranks=ranks,batch_size=len(requests[1]['mapping']),fault_time_ratio=requests[1]['seconds']/requests[0]['seconds'],
            tokens_changed=requests[0]['tokens']!=requests[1]['tokens']))
    require(identities[0]==identities[1] and owners[0]!=owners[1],'SAME_WORKERS_DISTINCT_SCOPES_REQUIRED')
    return dict(status='REAL_BATCH_SITE_REVIEW_PASSED',scopes=scopes,qualification_granted=False,
                public_runtime_integration='NOT_ACCEPTED_BY_THIS_REVIEW')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt');a=p.parse_args()
    print(json.dumps(review(json.loads(Path(a.receipt).read_bytes())),indent=2))
