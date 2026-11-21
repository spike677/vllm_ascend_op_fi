"""Independent receipt review; no NPU access or deployment registration."""
import argparse,json
from pathlib import Path


def review(report):
    def require(ok,code):
        if not ok:raise ValueError(code)
    require(report.get('status','').startswith('TWO_SCOPES'),'RUN_NOT_COMPLETED')
    require(len(report['scopes'])==2,'TWO_SCOPES_REQUIRED')
    summaries=[]; identities=[]; owners=[]
    for scope in report['scopes']:
        owners.append(scope['owner']); requests=scope['requests']
        require([r['phase'] for r in requests]==['normal','fault','off','health'],'PHASE_SEQUENCE')
        normal,fault,off,health=requests
        require(all(len(r['ranks'])==len(r['cleanup'])==len(r['installed'])==2 for r in requests),'EXACT_TWO_RANKS_REQUIRED')
        ranked={}
        for rank in (0,1):
            n,f,o,h=[request['ranks'][rank] for request in requests]
            require(all(len(r['output'])==len(r['requests'])==4 for r in (n,f,o,h)),'FOUR_FORWARD_RECEIPTS_REQUIRED')
            require(all(r['rank']==rank for r in (n,f,o,h)),'RANK_MISMATCH')
            require(len({(r['pid'],r['model_id']) for r in (n,f,o,h)})==1,'WORKER_CHANGED')
            require(n['output']==o['output']==h['output'],'OFF_HEALTH_NOT_EXACT')
            events=f['events'];sites=set(fault['installed'][rank]['sites'])
            require(len(sites)==f['site_count'] and {e['site'] for e in events}==sites,'SITE_SET_MISMATCH')
            require(len(events)==4*len(sites),'CALL_COUNT_MISMATCH')
            for site in sites:
                seq=[e['method'] for e in events if e['site']==site]
                require(seq==['_forward_prefill']+['_forward_decode']*3,'PREFILL_DECODE_SEQUENCE')
            require(all(e['applied'] and e['consumed'] and e['restored'] and e['non_target_exact'] for e in events),'INCOMPLETE_CALL')
            for event in events:
                before=event.get('before_sha256')
                require(isinstance(before,str) and len(before)==64 and all(c in '0123456789abcdef' for c in before),'BYTE_SNAPSHOT_HASH_REQUIRED')
                require(event.get('restored_sha256')==before,'RESTORED_BYTES_NOT_PROVEN')
                if 'actual_output_restored_sha256' in event:
                    require(event['actual_output_restored_sha256']==before,'ACTUAL_OUTPUT_BYTES_NOT_RESTORED')
            counts=dict(observed=len(events),eligible=len(events),attempted=len(events),applied=len(events),
                        changed=sum(e['changed']>0 for e in events),consumed=len(events),restored=len(events),failed=0)
            if 'coverage' in f:require(f['coverage']==counts,'REPORTED_COUNTS_MISMATCH')
            for request in requests:
                require(request['cleanup'][rank]['resources']==dict(leases=0,hooks=0,inflight=0),'RESOURCES_NOT_ZERO')
                require('mapping' in request and len(request['mapping'])==1,'ACTUAL_ENQUEUE_MAPPING_REQUIRED')
                internal=request['mapping'][0]['internal_request_id']
                require(all(r['request_ids']==[internal] for r in request['ranks'][rank]['requests']),'REQUEST_MAP_MISMATCH')
            ranked[str(rank)]=dict(sites=len(sites),coverage=counts,
                hidden_changed=any(a['hidden']!=b['hidden'] for a,b in zip(n['output'],f['output'])),
                logits_changed=any(a['logits']!=b['logits'] for a,b in zip(n['output'],f['output'])),
                off_health_exact=True,byte_restore_verified=True)
        identities.append([(r['pid'],r['model_id']) for r in normal['ranks']])
        summaries.append(dict(ranks=ranked,fault_time_ratio=fault['seconds']/normal['seconds'],
                             tokens_changed=normal['tokens']!=fault['tokens'],request_mapping='mapping' in normal))
    require(identities[0]==identities[1] and owners[0]!=owners[1],'SAME_WORKERS_DISTINCT_SCOPES_REQUIRED')
    return dict(status='REAL_SITE_REVIEW_PASSED',scopes=summaries,
                qualification_granted=False,public_runtime_integration='NOT_ACCEPTED_BY_THIS_REVIEW')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt');args=parser.parse_args()
    print(json.dumps(review(json.loads(Path(args.receipt).read_bytes())),indent=2))
