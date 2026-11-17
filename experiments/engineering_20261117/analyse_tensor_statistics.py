"""Locate solely from the actual B/F raw statistics, preserving refusals."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'fault39/src')]
from fault_injection.evidence import digest, independent_statistics_pair


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();capture=a.case/'statistics'
    output=dict(schema='demoncat.engineering.statistics_summary.v1',case=str(a.case),
                raw_published=False,private_tensors_read=False,native_logs_read=False)
    try:
        groups={step:list((capture/'observation'/f'step{step}').rglob('dump.json')) for step in (0,3)}
        if any(len(paths)!=1 for paths in groups.values()):raise ValueError('MISSING_OR_AMBIGUOUS_REAL_STEP')
        output['analysis']=independent_statistics_pair(comparator=ROOT/'fault39/tools/compare_msprobe.py',
            baseline=groups[0][0].parent,fault=groups[3][0].parent,
            baseline_config=capture/'config.json',fault_config=capture/'config.json',
            baseline_collector=capture/'collector_identity.json',fault_collector=capture/'collector_identity.json')
        output['actual_steps']={'baseline':0,'fault':3}
        output['status']='ANALYSIS_COMPLETED'
    except Exception as error:
        output.update(status='ANALYSIS_UNAVAILABLE',reason=repr(error))
    # Native/Provider results are deliberately absent from this analyzer.
    output['source_sha256']={str(path.relative_to(ROOT)):digest(path) for path in
        (Path(__file__),ROOT/'fault_injection/evidence.py',ROOT/'fault39/tools/compare_msprobe.py')}
    with a.out.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(status=output['status'],case=str(a.case),
        independent_status=output.get('analysis',{}).get('first_difference_status'),
        comparable=output.get('analysis',{}).get('whole_dump_comparable'))))


if __name__=='__main__':main()
