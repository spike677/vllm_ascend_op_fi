"""M512 causal variant must exactly preserve the independently bridged OFF."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_case import audit,ROOT
from compare_targets import load
from graph_evidence import ge_compare,mix_compare


def check(suffix):
    rows=[];targets=[]
    for batch in (1,8):
        names=[f'g_off_b{batch}_r1',f'g_causal_base512_b{batch}_r1']
        audits=[audit(n+'_'+suffix) for n in names]
        folders=[ROOT/'outputs'/(n+'_'+suffix)/n for n in names]
        for a in audits:assert a['status']=='PASS_RECORDED_EXECUTION'
        plan=audits[1]['public_host'][0]['plan']
        assert plan['m']==512 and plan['pruned'] and not plan['target_queries_with_partial_inner_width']
        assert audits[1]['compiler_binding']['actual_template_key']=='109000000000000000'
        for call in (0,1,2):
            results=[json.loads((f/'result.json').read_bytes())['calls'][call] for f in folders]
            assert results[0]['full_sha256']==results[1]['full_sha256']
            assert results[0]['consumer']['full_sha256']==results[1]['consumer']['full_sha256']
            for consumer in (False,True):
                b,bm=load(folders[0],call,consumer);f,fm=load(folders[1],call,consumer)
                assert bm['sha256']==fm['sha256'] and np.array_equal(b,f)
        ge=[ge_compare(*folders,i) for i in (0,1,2)]
        mix=[mix_compare(*folders,i) for i in (0,1)]
        assert all(r['first_observed_difference'] is None for r in ge+mix)
        targets.append(load(folders[1])[0])
        rows.append({'batch':batch,'audit':audits,'ge':ge,'statistics_mix':mix,
                     'full_all_request_attention_and_projection_equal':True})
    assert np.array_equal(targets[0],targets[1])
    return {'schema':'fault28.fia_graph.base_variant_gate.v1','status':'PASS_M512_VARIANT_COMPATIBILITY',
            'rows':rows,'fixed_M512_cross_batch_complete_target_equal':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suffix',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=check(a.suffix);rc=0
    except (AssertionError,ValueError,OSError,KeyError,TypeError,IndexError) as e:
        report={'schema':'fault28.fia_graph.base_variant_gate.v1','status':'FAIL_BASE_VARIANT','error':str(e)};rc=1
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'error':report.get('error')}));raise SystemExit(rc)
