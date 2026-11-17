"""Read B/F statistics only; reports refusals without using injection receipts."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',choices=['tensor','postwrite'],required=True);a=p.parse_args()
    sys.path[:0]=[str(a.source),str(a.source/'fault39/src')]
    from fault_injection.evidence import independent_statistics_pair
    a.out.mkdir(parents=True,exist_ok=False)
    pairs=[]
    if a.kind=='tensor':
        for case in sorted(a.run.glob('tensor*')):
            capture=case/'statistics'
            pairs.append((case.name,capture,capture,0,3))
    else:
        for case in sorted(a.run.glob('*-baseline')):
            pairs.append((case.name.removesuffix('-baseline'),case/'capture',
                          case.with_name(case.name.replace('-baseline','-fault'))/'capture',0,0))
    summaries=[]
    for name,base,fault,bs,fs in pairs:
        result={'case':name,'actual_steps':{'baseline':bs,'fault':fs},'native_logs_read':False,
                'private_tensors_read':False,'raw_published':False}
        try:
            dumps=[list((c/'observation'/f'step{s}').rglob('dump.json')) for c,s in ((base,bs),(fault,fs))]
            if any(len(rows)!=1 for rows in dumps):raise ValueError('MISSING_OR_AMBIGUOUS_STEP')
            result['analysis']=independent_statistics_pair(
                comparator=a.source/'fault39/tools/compare_msprobe.py',
                baseline=dumps[0][0].parent,fault=dumps[1][0].parent,
                baseline_config=base/'config.json',fault_config=fault/'config.json',
                baseline_collector=base/'collector_identity.json',fault_collector=fault/'collector_identity.json')
            result['status']='ANALYSIS_COMPLETED'
        except Exception as error:
            result.update(status='ANALYSIS_UNAVAILABLE',reason=repr(error))
        (a.out/(name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        summary={k:v for k,v in result.items() if k!='analysis'}
        analysis=result.get('analysis',{})
        summary.update(independent_status=analysis.get('first_difference_status'),
                       comparable=analysis.get('whole_dump_comparable'),
                       first_difference=analysis.get('pair',{}).get('comparison'))
        summaries.append(summary)
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summaries,allow_nan=False))


if __name__=='__main__':main()
