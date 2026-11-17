"""Read-only concise progress; no tensor files or device calls."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
if not a.attempt.replace('-','').isalnum():raise ValueError('Invalid attempt')
root=Path('/home/worker-positive-20261117')/a.attempt
out={}
for path in root.glob('*/result.json'):
    x=json.loads(path.read_text())
    out[path.parent.name]={k:x[k] for k in ('status','error','cleanup_error','finish') if k in x and k!='finish'}
    out[path.parent.name]['phases']=[dict(phase=r['phase'],returned='response' in r,
        query_status=r.get('query',{}).get('status')) for r in x.get('phases',[])]
    if 'finish' in x and isinstance(x['finish'],list):out[path.parent.name]['finish']=x['finish']
for path in root.glob('*/serving/runtime/serving_receipt.json'):
    x=json.loads(path.read_text());out['qualification']=x['qualification']
print(json.dumps(out,indent=2))
