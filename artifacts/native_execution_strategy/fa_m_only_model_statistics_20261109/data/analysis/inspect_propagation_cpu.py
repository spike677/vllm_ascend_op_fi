"""Provisional read-only numeric propagation view, never an acceptance result."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case-dir',type=Path,required=True);p.add_argument('--step',type=int,default=4);a=p.parse_args()
    result=json.loads((a.case_dir/'result.json').read_bytes());rows=[]
    for item in result['calls']:
        if item['step']!=a.step:continue
        call=json.loads((a.case_dir/item['file']).read_bytes())
        observed=call['output']['exact_numeric_observation']
        rows.append({'layer':call['layer_index'],'Attention_changed':observed.get('comparison',{}).get('changed_elements'),
                     'Attention_last_query_changed':observed.get('comparison',{}).get('last_query_changed_elements'),
                     'actual_M':call.get('actual_strategy',{}).get('parameters',{}).get('query_tile'),
                     'actual_ratio':call.get('actual_strategy',{}).get('parameters',{}).get('key_group_ratio')})
    print(json.dumps({'provisional':True,'not_acceptance':True,'step':a.step,'layers':rows},indent=2))
