"""Read selected lines from completed native logs, no device calls."""
import argparse,csv,json
from collections import deque
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
root=Path('/home/fault28-fia-graph-20261106/outputs')/a.attempt
for c in json.loads((root/'completion.json').read_bytes())['cases']:
 name=c['case_id'];f=root/name
 for v in (f/'loaded_libraries.json',):
  if v.is_file():print(v.name,v.read_text())
 counts={};hits=deque(maxlen=80)
 for n,line in enumerate((root/(name+'.log')).open(errors='replace'),1):
  if 'tilingkey' in line.lower() or 'tiling_key' in line.lower() or 'FIA sInnerSize_' in line:
   counts['matched']=counts.get('matched',0)+1
   hits.append({'line':n,'text':line.strip()[:1500]})
 print(json.dumps({'counts':counts,'evidence':list(hits)}))
 for v in (root/(name+'_native_profile')).rglob('op_summary*.csv'):
  rows=list(csv.DictReader(v.open()));print('profile_count',len(rows));print('profile_first',[{k:v for k,v in r.items() if k in ('Op Name','OP Type','Block Num','Task ID','Model ID','Task Type')} for r in rows[:8]])
  print(json.dumps([r for r in rows if any('attention' in str(x).lower() for x in r.values())][:5]))
