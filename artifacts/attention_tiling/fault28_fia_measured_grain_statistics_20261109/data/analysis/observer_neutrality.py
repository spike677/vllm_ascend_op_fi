"""Verify device scalar observation did not change any recorded model output byte."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from compare_targets import load
from audit_case import ROOT


def check(suffix='grain_a2'):
    matrix=json.loads((ROOT/'setup/fp64_matrix_v1.json').read_bytes())
    rows=[]
    for name in matrix['case_ids']:
        folders=[Path('/home/fault28-fia-graph-20261106/outputs')/(name+'_fp64_a1')/name,
                 ROOT/'outputs'/(name+'_'+suffix)/name]
        results=[json.loads((p/'result.json').read_bytes()) for p in folders]
        assert all(r['status']=='CAPTURE_COMPLETED' for r in results)
        for call in (0,1,2):
            a,b=[r['calls'][call] for r in results]
            for key in ('full_sha256','target_sha256','request_sha256','shape','dtype'):
                assert a[key]==b[key],(name,call,key)
            for key in ('full_sha256','target_sha256','request_sha256','shape'):
                assert a['consumer'][key]==b['consumer'][key],(name,call,'consumer',key)
            for consumer in (False,True):
                left,lmeta=load(folders[0],call,consumer);right,rmeta=load(folders[1],call,consumer)
                assert lmeta['sha256']==rmeta['sha256'] and np.array_equal(left,right)
        raw_equal=None
        if (folders[0]/'host_events.jsonl').exists():
            raw=[json.loads((p/'host_events.jsonl').read_text().splitlines()[0]) for p in folders]
            assert raw[0]['raw_hex']==raw[1]['raw_hex'] and raw[0]['tiling_key']==raw[1]['tiling_key']
            raw_equal=True
        rows.append({'case_id':name,'graph_calls':3,'complete_attention_and_projection_targets_equal':True,
                     'all_request_device_hashes_equal':True,'public_raw_and_actual_key_equal':raw_equal,
                     'attention_sha256':results[1]['calls'][1]['target_sha256'],
                     'projection_sha256':results[1]['calls'][1]['consumer']['target_sha256']})
    return {'schema':'fault28.device_scalar_observer_neutrality.v1','status':'PASS_ALL_NUMERICAL_OUTPUTS_UNCHANGED',
            'cases':len(rows),'paired_graph_calls':len(rows)*3,'rows':rows,
            'scope':'Identical CPU FP64 collectors; original versus device-scalar-observer kernel. Complete private target bytes plus all-request synchronized hashes and actual raw/key must match.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--suffix',default='grain_a2');a=p.parse_args()
    try:report=check(a.suffix);rc=0
    except (AssertionError,OSError,KeyError,ValueError,TypeError) as e:
        report={'status':'FAIL_OBSERVER_NEUTRALITY','error':str(e)};rc=1
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','cases','paired_graph_calls','error')}))
    raise SystemExit(rc)
