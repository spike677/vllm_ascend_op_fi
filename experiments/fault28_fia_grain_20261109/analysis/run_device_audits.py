"""CPU-only official decoding and measured-column checks for completed cases."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from device_trace import audit,sha,ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suffix',default='grain_a2');a=p.parse_args()
    acceptance=json.loads((a.reports/'acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM'
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for summary in acceptance['case_summary']:
        name=summary['case_id'];folder=ROOT/'outputs'/(name+'_'+a.suffix)/name
        case=json.loads((folder/'case.json').read_bytes())
        if case['operator_variant']!='private_vendor':
            rows.append({'case_id':name,'status':'STOCK_NO_DEVICE_OBSERVER','scalar_records':0});continue
        destination=a.output/name;destination.mkdir()
        execution_file=a.reports/(name+'_audit.json')
        execution=json.loads(execution_file.read_bytes())
        receipt=destination/'sdk_receipt.json'
        command=[sys.executable,'-B',str(Path(__file__).with_name('sdk_debug_tool_cpu.py')),
                 '--file',str(folder/'private_device_debug/printf'),'--output',str(destination/'official_decode'),
                 '--receipt',str(receipt)]
        subprocess.run(command,check=True)
        report=audit(folder,execution,receipt)
        report['execution_report_sha256']=sha(execution_file)
        report['analysis_source_sha256']=sha(Path(__file__).with_name('device_trace.py'))
        with (destination/'device_columns.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        rows.append({'case_id':name,'status':report['status'],'M':report['actual_m'],
                     'actual_key':report['actual_key'],'scalar_records':report['canonical_scalar_records'],
                     'observed_columns':sorted({p['measured_columns'] for p in report['query_tiles']}),
                     'query63_columns':[p['measured_columns'] for p in report['query_tiles'] if p['query']==63],
                     'report':name+'/device_columns.json','report_sha256':sha(destination/'device_columns.json')})
        print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==23 and sum(r['status']=='PASS_MEASURED_DEVICE_SOFTMAX_COLUMNS' for r in rows)==19
    result={'schema':'fault28.fia.device_columns_matrix.v1','status':'PASS_MEASURED_FIA_SOFTMAX_COLUMN_MATRIX',
            'cases':23,'measured_private_cases':19,'rows':rows,
            'scalar_records':sum(r['scalar_records'] for r in rows),'sampled_queries':[63,255,511,767,1023,1535,2047],
            'individual_scalar_to_replay_pairing_proven':False,'native_calls_executed_by_this_analysis':0,
            'analysis_source_sha256':sha(Path(__file__))}
    with (a.output/'summary.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','cases','measured_private_cases','scalar_records')}))


if __name__=='__main__':main()
