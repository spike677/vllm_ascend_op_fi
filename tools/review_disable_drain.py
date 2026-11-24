"""Review the two-owner real-request drain stress extension, without inference."""
import argparse
import json
from pathlib import Path
from tools.review_product_mode import review


def check(data):
    base=review(data);results=[]
    for scope in data['scopes']:
        stress=scope.get('drain_stress',{})
        if stress.get('submitted')!=2 or stress.get('pending_at_disable',0)<1:
            raise ValueError('NO_OUTSTANDING_HTTP_REQUEST_AT_DISABLE')
        if len(stress.get('http_results',[]))!=2 or stress.get('status_failures')!=[]:
            raise ValueError('DRAIN_FAILED_OR_STATUS_MISMATCH')
        completed=stress.get('completed_requests',[])
        if len(completed)!=2 or len({r['request_id'] for r in completed})!=2:
            raise ValueError('DRAIN_REQUEST_SET_INCOMPLETE')
        for request in completed:
            ranks=request['ranks']
            if sorted(r['rank'] for r in ranks)!=[0,1]:raise ValueError('DRAIN_RANK_SET')
            for r in ranks:
                c=r['coverage']
                if r['request_id']!=request['request_id'] or not r['completed']:
                    raise ValueError('DRAIN_REQUEST_ASSOCIATION')
                if not (c['eligible']>0 and c['eligible']==c['applied']==c['consumed']==c['restored'] and c['failed']==0):
                    raise ValueError('DRAIN_HIT_OR_RESTORE_INCOMPLETE')
                if r['resources']!=dict(leases=0,hooks=0,inflight=0):raise ValueError('DRAIN_RESOURCES_REMAIN')
        results.append(dict(scope=scope['scope'],pending_at_disable=stress['pending_at_disable'],
                            fully_completed_after_drain=2,rank_counts_equal=True))
    return dict(status='TWO_OWNER_DRAIN_REVIEW_PASSED',base=base,stress=results,
                reproduces_yellow_mismatch=False,qualification_granted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--output');a=p.parse_args()
    result=check(json.loads(Path(a.receipt).read_bytes()));text=json.dumps(result,indent=2)+'\n'
    if a.output:Path(a.output).write_text(text)
    else:print(text)


if __name__=='__main__':main()
