"""Explain alignment refusal using only the three raw statistics JSON files."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('raw',a.source/'fault39/tools/compare_msprobe.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    b,f=module.load_raw(a.baseline),module.load_raw(a.fault)
    report={'schema':'demoncat.raw_alignment_diagnostic.v1','native_logs_read':False,
            'private_tensors_read':False,'global_first_difference_claimed':False,
            'baseline_hashes':b['hashes'],'fault_hashes':f['hashes'],'mismatches':{}}
    for field in ('structure','stack','construct'):
        if b[field]==f[field]:continue
        left,right=b[field],f[field]
        report['mismatches'][field]={'baseline_count':len(left),'fault_count':len(right)}
        if isinstance(left,list) and isinstance(right,list):
            entries=[{'index':i,'baseline':x,'fault':y} for i,(x,y) in
                     enumerate(itertools.zip_longest(left,right)) if x!=y]
            report['mismatches'][field].update(first_entries=entries[:3],different_entries=len(entries))
        elif isinstance(left,dict) and isinstance(right,dict):
            changed=[k for k in dict.fromkeys([*left,*right]) if left.get(k)!=right.get(k)]
            report['mismatches'][field].update(first_keys=changed[:3],different_keys=len(changed))
    with a.out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(report,allow_nan=False))


if __name__=='__main__':main()
