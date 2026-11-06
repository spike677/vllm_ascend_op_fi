"""Read-only compact view of original completion, graph replay and profile rows."""
import argparse
import csv
import json
from pathlib import Path
import re

ROOT=Path('/home/fault28-fia-graph-20261106')


def compact(attempt):
    root=ROOT/'outputs'/attempt
    complete=json.loads((root/'completion.json').read_bytes())
    result={'attempt':attempt,'status':complete['status'],'cleanup_errors':complete.get('cleanup_errors'), 'cases':[]}
    for row in complete['cases']:
        name=row['case_id'];folder=root/name
        case=json.loads((folder/'result.json').read_bytes())
        item={'case_id':name,'status':case['status'],'calls':[{k:r.get(k) for k in
              ('call_id','target_sha256','full_sha256','inputs_unchanged')} for r in case['calls']],
              'backend_graphs':case['backend_graphs'],'error':case.get('traceback','')[-4500:]}
        host=folder/'host_events.jsonl'
        if host.exists():
            item['host']=[{k:v for k,v in h.items() if k not in ('raw_hex','typed','checks')}
                          for h in map(json.loads,host.read_text().splitlines())]
        native=[]
        for p in (root/(name+'_native_profile')).rglob('op_summary*.csv'):
            with p.open(newline='') as stream:
                selected=[r for r in csv.DictReader(stream) if
                          any('FusedInferAttentionScore' in str(v) for v in r.values())]
            native.append({'path':str(p.relative_to(root)),'rows':selected})
        item['profile']=native
        result['cases'].append(item)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]{1,40}',a.attempt)
    print(json.dumps(compact(a.attempt),indent=2))
