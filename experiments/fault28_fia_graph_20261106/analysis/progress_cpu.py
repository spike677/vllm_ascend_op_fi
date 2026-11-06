"""Read-only completed case summary; it never launches a case or accepts fault."""
import json
from pathlib import Path
from compare_targets import load,difference
ROOT=Path('/home/fault28-fia-graph-20261106')
m=json.loads((ROOT/'setup/formal_matrix_v3.json').read_bytes())
folders={}
for name in m['case_ids']:
 root=ROOT/'outputs'/(name+'_final_a1');folder=root/name
 if not (root/'completion.json').is_file():continue
 result=json.loads((folder/'result.json').read_bytes());folders[name]=folder
 print(json.dumps({'case':name,'status':result['status'],'target_sha':result.get('calls',[{}])[-1].get('target_sha256'),
    'error':result.get('traceback','')[-900:]}))
for n in (128,256):
 for batch in (1,8):
  a=f'g_m{n}_b{batch}_r1';b=f'g_causal_m{n}_b{batch}_r1'
  if a in folders and b in folders:
   left,_=load(folders[a]);right,_=load(folders[b]);r=difference(left,right)
   print(json.dumps({'pair':[a,b],'causal_inner_width_effect_elements':r['changed_elements'],'max_abs':r['max_abs']}))
