"""Inspect actual saved compiler metadata, no framework/device imports."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
inventory=json.loads((a.folder/'jit_compile_inventory.json').read_bytes())
for row in inventory['files']:
 if 'private_copy' not in row:continue
 path=a.folder/row['private_copy'];raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
 obj=json.loads(raw);print('FILE',path.name,'ROOT KEYS',list(obj) if isinstance(obj,dict) else type(obj).__name__)
 def visit(value,key):
  if isinstance(value,dict):
   for k,v in value.items():visit(v,key+'.'+k)
  elif isinstance(value,list):
   if len(value)<20:
    for i,v in enumerate(value):visit(v,key+'['+str(i)+']')
  elif any(t in key.lower() for t in ('tiling','key','kernel','compile','binfilename')):
   print(key, repr(value)[:650], 'length='+str(len(value)) if isinstance(value,str) else '')
 visit(obj,'$')
