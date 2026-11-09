"""Read-only native graph launch/compile evidence discovery, not an acceptor."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
root=Path('/home/fault28-fia-graph-20261106/outputs')/a.attempt
for c in json.loads((root/'completion.json').read_bytes())['cases']:
 lines=[]
 for n,line in enumerate((root/(c['case_id']+'.log')).open(errors='replace'),1):
  low=line.lower()
  if ('fusedinferattention' in low or 'fused_infer_attention' in low) and any(t in low for t in ('launch','kernel', 'tilingkey', 'tiling_key','taskid', 'task_id','build argument')):
   if not any(t in low for t in ('staticregister','registering tiling','register_tiling','genattrinfo','paramtype','attr_desc','output_desc','input_desc')):
    lines.append({'line':n,'text':line.strip()[:2500]})
 print(json.dumps({'count':len(lines),'excerpts':lines[-95:]}))
