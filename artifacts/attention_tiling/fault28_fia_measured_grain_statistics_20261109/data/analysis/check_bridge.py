"""Require independent completed stock/OFF bridge before numerical intervention."""
import argparse
import json
from pathlib import Path
from audit_case import audit,ROOT
from compare_targets import load,reference,ref_errors
from graph_evidence import ge_compare,mix_compare
import numpy as np


def check(repetitions,suffix):
    records=[];signatures={};ref=reference(Path('/home/fault28-causal-20261029/inputs'))
    for repeat in repetitions:
        for batch in (1,8):
            cases={key:f'g_{key}_b{batch}_r{repeat}' for key in ('stock','off')}
            folders={key:ROOT/'outputs'/(value+'_'+suffix)/value for key,value in cases.items()}
            audits={key:audit(value+'_'+suffix) for key,value in cases.items()}
            for key,a in audits.items():
                assert a['status']=='PASS_RECORDED_EXECUTION'
                if key=='off':
                    assert a['public_host'][0]['plan']['m']==512
                    assert not a['public_host'][0]['plan']['pruned']
            ge=[ge_compare(folders['stock'],folders['off'],i) for i in (0,1,2)]
            mix=[mix_compare(folders['stock'],folders['off'],i) for i in (0,1)]
            assert all(r['first_observed_difference'] is None for r in ge+mix)
            for call in (0,1,2):
                for consumer in (False,True):
                    b,bm=load(folders['stock'],call,consumer);f,fm=load(folders['off'],call,consumer)
                    assert bm['sha256']==fm['sha256'] and np.array_equal(b,f)
                    identity=(batch,consumer)
                    if identity in signatures:assert signatures[identity]==bm['sha256']
                    else:signatures[identity]=bm['sha256']
            errors=ref_errors(load(folders['stock'])[0],ref)
            assert all(r['max_abs']<=0.02 and r['relative_l2']<=0.01 for r in errors)
            records.append({'batch':batch,'repeat':repeat,'complete_attention_and_projection_bridge':True,
                            'audit':audits,'ge_statistics':ge,'statistics_mix':mix,'fp64_reference_sanity':errors})
    return {'schema':'fault28.fia_graph.bridge_gate.v1','status':'PASS_STOCK_OFF_BRIDGE',
            'repetitions':repetitions,'records':records,
            'note':'Exact complete target outputs, repeated independent processes, unchanged inputs, actual static task and public/compiler raw. No numerical fault yet.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repetitions',nargs='+',type=int,required=True)
    p.add_argument('--suffix',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=check(a.repetitions,a.suffix);rc=0
    except (AssertionError,ValueError,OSError,KeyError,TypeError,IndexError) as e:
        report={'schema':'fault28.fia_graph.bridge_gate.v1','status':'FAIL_BRIDGE','error':str(e)};rc=1
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'error':report.get('error')}))
    raise SystemExit(rc)
