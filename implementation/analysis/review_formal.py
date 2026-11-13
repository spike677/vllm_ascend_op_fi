"""Three-step statistical gate followed by separate native/array mechanism checks."""
import hashlib
import json
from pathlib import Path
from compare_statistics import compare
from accept_statistics import accept
from audit_runtime import audit


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def values(folder):
    result=json.loads((folder/'result.json').read_bytes())
    assert result['status']=='CAPTURE_COMPLETED' and result['repeated_all_outputs_equal']
    return result,{(r['layer'],r['kind']):r for r in result['calls'][0]['values']}


def logits(b,f):
    import numpy as np
    arms=[]
    for folder in (b,f):
        _,v=values(folder);row=v[None,'logits'];path=folder/row['private_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['full_sha256']
        arms.append(np.fromfile(path,dtype='<f2').reshape(row['shape']).astype(np.float64))
    rows=[]
    for slot,(left,right) in enumerate(zip(*arms)):
        diff=right-left;count=int(np.count_nonzero(diff));index=np.unravel_index(int(np.argmax(np.abs(diff))),diff.shape)
        norm=float(np.linalg.norm(left));baseline=float(left[index]);error=float(abs(diff[index]))
        rows.append({'request_slot':slot,'elements':left.size,'changed_elements':count,
            'max_abs':error,'max_abs_index':list(map(int,index)),
            'B_at_max_abs':baseline,'F_at_max_abs':float(right[index]),
            'point_relative_at_max_abs':error/abs(baseline) if baseline else None,
            'relative_L2':float(np.linalg.norm(diff))/norm if norm else None,
            'B_top1':int(left[0].argmax()),'F_top1':int(right[0].argmax())})
    return {'requests':rows,'changed_requests':sum(r['changed_elements']>0 for r in rows),
            'top1_changed_requests':sum(r['B_top1']!=r['F_top1'] for r in rows),
            'max_abs':max(r['max_abs'] for r in rows),'used_to_select_statistics_first':False}


def review(baseline,fault,output,candidate):
    b,f=Path(baseline),Path(fault);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Complete generic localization before looking at any runtime or private array.
    reports=[];gates=[];layers=[]
    for step in range(3):
        r=compare(b/f'observation/step{step}/rank0/dump.json',f/f'observation/step{step}/rank0/dump.json')
        gate=accept(r);reports.append(r);gates.append(gate)
        save(out/f'independent_first_difference_step{step}.json',r)
        save(out/f'statistics_gate_step{step}.json',gate)
        nodes=[x['B'] for x in r['comparisons'] if x['B'].startswith('NPU.npu_fused_infer_attention_score.')]
        first=r['first_observed_api_difference']
        layers.append(nodes.index(first['B_node']) if first and first['B_node'] in nodes else None)
    same_field=set.intersection(*(set(d['path'] for d in g['qualifying_fields']) for g in gates))
    br,bv=values(b);fr,fv=values(f)
    runtime={name:audit(folder) for name,folder in [('baseline',b),('fault',f)]}
    for name,data in runtime.items():save(out/f'runtime_{name}.json',data)
    assert json.loads((b/'inputs.json').read_bytes())==json.loads((f/'inputs.json').read_bytes())
    layers_equal=all(l is not None and l==layers[0] for l in layers)
    first_qkv_equal=layers_equal and all(bv[layers[0],k]['full_sha256']==fv[layers[0],k]['full_sha256'] and
        bv[layers[0],k]['request_sha256']==fv[layers[0],k]['request_sha256'] for k in ('query','key','value'))
    actual={name:sorted({(x['decoded']['innerSplitParams.mBaseSize'],x['decoded']['innerSplitParams.s2BaseSize'])
                        for x in data['public_plan_unique']}) for name,data in runtime.items()}
    native_ok=actual['baseline']==[(candidate['B_M'],candidate['B_S2'])] and actual['fault']==[(candidate['F_M'],candidate['F_S2'])]
    if candidate['variant']=='online_softmax_denominator_replay':
        native_ok=native_ok and {x['actual_key'] for x in runtime['baseline']['census']}=={'109000000000400000'} and {x['actual_key'] for x in runtime['fault']['census']}=={'119000000000400000'}
    propagated=logits(b,f);save(out/'logits_comparison.json',propagated)
    accepted=all(g['accepted'] for g in gates) and bool(same_field) and layers_equal and first_qkv_equal and native_ok and propagated['changed_requests']>0
    r={'schema':'fault28.visible.fullmodel_review.v1','status':'PASS_FULLMODEL_REQUIRES_RESTART_OFF_AND_BATCH_CONTROLS' if accepted else 'NOT_ACCEPTED',
       'accepted_before_controls':accepted,'candidate_id':candidate['candidate_id'],'statistics_only_gate':[g['accepted'] for g in gates],
       'same_qualifying_fields_all_steps':sorted(same_field),'first_observed_FIA_layers':layers,
       'first_observed_FIA_complete_QKV_byte_equal':first_qkv_equal,'actual_M_S2':actual,
       'logits':propagated,'actual_steps':[0,1,2],'rtol':1e-5,'atol':1e-8,'minimum_relative_change':1e-3,
       'limits':['Independent localization used only raw statistics/config/stack/construct and collector identity.',
                 'Complete array/hash/native evidence is separate, after localization.',
                 'Top1 change is not required. No continuous-generation or task-quality acceptance.',
                 'Inner grain is decoded raw plus executed source, not an instruction-level device trace.']}
    save(out/'review.json',r);return r
