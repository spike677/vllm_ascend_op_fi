"""Independent raw-only statistics comparison, including baseline self-check."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser();parser.add_argument('--baseline-case',type=Path,required=True)
parser.add_argument('--fault-case',type=Path,required=True);parser.add_argument('--step',type=int,required=True)
parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
root=Path('/home/worker-positive-20261117')
sys.path.insert(0,str(root/'packages-a6/installed'))
source=root/'native-source-a6/fault39/tools/compare_msprobe.py'
spec=importlib.util.spec_from_file_location('independent_raw_statistics_comparator',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from demoncat_fault39.msprobe import ComparisonPolicy

def capture(case):
    parent=case/'serving/capture'
    return parent/'observation'/('step'+str(args.step))/'rank0',parent/'config.json',parent/'collector_identity.json'

b,bc,bi=capture(args.baseline_case);f,fc,fi=capture(args.fault_case)
policy=ComparisonPolicy(1e-8,1e-5)
compare=lambda l,r,lc,rc,li,ri:module.compare(l,r,policy,baseline_config=lc,fault_config=rc,
                                           baseline_collector=li,fault_collector=ri)
full=dict(comparator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          baseline=str(b),fault=str(f),baseline_self=compare(b,b,bc,bc,bi,bi),
          comparison=compare(b,f,bc,fc,bi,fi))
args.out.mkdir(parents=True,exist_ok=False)
raw=args.out/'independent_full.json';raw.write_text(json.dumps(full,indent=2,allow_nan=False)+'\n')
def compact(value):
    output=dict(value)
    if 'comparison' in output:
        output['comparison']=dict(output['comparison'])
        missing=output['comparison'].pop('unobservable_fields',[])
        output['comparison']['unobservable_fields_count_before_first_or_all']=len(missing)
        output['comparison']['unobservable_fields_preserved_in_private_full_report']=True
    return output
summary=dict(comparator_sha256=full['comparator_sha256'],baseline=full['baseline'],fault=full['fault'],
             independent_full_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
             baseline_self=compact(full['baseline_self']),comparison=compact(full['comparison']))
(args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(step=args.step,comparison=summary['comparison'].get('comparison',summary['comparison']))))
