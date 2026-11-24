"""Check real page-notice alignment; read-only evidence is never injection proof."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path


def review(report):
    issues=[];ranks=report.get('ranks',[]);summaries=[];reference=None
    if report.get('status')!='CENSUS_COMPLETE_NOT_PRODUCT_ACCEPTANCE':issues.append('CENSUS_INCOMPLETE')
    if sorted(r.get('rank',-1) for r in ranks)!=[0,1]:issues.append('TWO_ACTUAL_RANKS_REQUIRED')
    for rank in ranks:
        prefix=str(rank.get('rank'))+':'
        def issue(s):issues.append(prefix+s)
        events=rank.get('page_barrier_events',[])
        if not events:issue('NO_ENGINE_PAGE_NOTICES')
        if rank.get('active_page_tickets')!=0:issue('ACTIVE_PAGE_TICKETS')
        if rank.get('page_injection_executed') is not False:issue('READ_ONLY_SCOPE_CHANGED')
        current={};executed=Counter();ended=Counter();last=0
        for e in events:
            seq=e.get('sequence');rid=e.get('request_id');generation=e.get('request_generation');pages=e.get('pages')
            if type(seq) is not int or seq<=last:issue('STALE_SEQUENCE')
            else:last=seq
            if len(set(e.get('full_attention_layers',[])))!=16:issue('FULL_ATTENTION_SCOPE')
            if not rid or not generation or not pages:issue('REQUEST_PAGE_IDENTITY_MISSING')
            if e.get('action')=='before_execute':
                if rid in current and current[rid]!=generation:issue('GENERATION_CHANGED')
                current[rid]=generation;executed[rid]+=1
            elif e.get('action')=='before_free':
                if current.get(rid)!=generation:issue('UNMATCHED_FREE')
                current.pop(rid,None);ended[rid]+=1
            else:issue('UNKNOWN_NOTICE')
        actual=Counter(rid for r in rank.get('requests',[]) for rid in r.get('ids',[]))
        if executed!=actual:issue('ENGINE_WORKER_REQUESTS_DIFFER')
        if len(actual)!=2 or any(v!=4 for v in actual.values()):issue('TWO_REQUESTS_FOUR_STEPS_REQUIRED')
        if set(ended)!=set(actual) or any(v!=1 for v in ended.values()) or current:issue('REQUEST_RELEASE_INCOMPLETE')
        if reference is not None and events!=reference:issue('RANK_NOTICES_DIFFER')
        reference=events
        summaries.append(dict(rank=rank.get('rank'),requests=len(actual),steps=sum(actual.values()),
                              releases=sum(ended.values()),notices=len(events)))
    return dict(schema='dcat.v2.kv_page_probe_review.v1',passed=not issues,issues=issues,ranks=summaries,
                scope='READ_ONLY_ENGINE_PAGE_RELEASE_NOT_INJECTION',qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();raw=a.receipt.read_bytes()
    result=review(json.loads(raw));result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
