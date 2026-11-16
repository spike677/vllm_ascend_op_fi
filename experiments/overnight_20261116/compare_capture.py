"""Independent B/F raw statistics comparison with unchanged fixed tolerances."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    a=p.parse_args();sys.path.insert(0,str(a.source/'fault39/tools'))
    from compare_model_capture import compare_groups
    result=compare_groups(a.baseline,a.fault,rank=a.rank)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    summary={k:result[k] for k in ('status','comparison_policy','baseline_steps','fault_steps','all_steps_aligned','blocking_steps')}
    summary.update(raw_statistics_published=False,native_logs_read=False,private_tensors_read=False,
                   private_report=str(a.output),private_report_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),steps=[])
    for step in result['steps']:
        c=step['report'].get('comparison',{})
        summary['steps'].append(dict(step=step['step'],status=step['status'],identity=c.get('identity'),
            difference=c.get('difference'),unobservable_count=len(c.get('unobservable_fields',[])),
            raw_evidence={k:v for k,v in step['report'].items() if k in ('baseline','fault')},
            alignment_diagnostic=step['report'].get('reason')))
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
