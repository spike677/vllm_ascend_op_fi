"""Independent raw-statistics comparison; compact public summary, private report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fault39/tools'))
from compare_model_capture import compare_groups

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0)
    a=p.parse_args()
    result=compare_groups(a.baseline,a.fault,rank=a.rank)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    summary={key:result[key] for key in ('status','comparison_policy','baseline_steps','fault_steps','all_steps_aligned','blocking_steps')}
    summary['steps']=[]
    for step in result['steps']:
        c=step['report'].get('comparison',{})
        summary['steps'].append(dict(step=step['step'],status=step['status'],
            identity=c.get('identity'),difference=c.get('difference'),
            unobservable_count=len(c.get('unobservable_fields',[]))))
    summary['private_report_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=='__main__':main()
