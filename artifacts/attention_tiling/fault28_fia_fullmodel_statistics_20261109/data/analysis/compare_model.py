"""Private complete all-request comparison; never used to choose msprobe first."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/fault28-fia-fullmodel-20261109')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(folder):
    result=json.loads((folder/'result.json').read_bytes())
    assert result['status']=='CAPTURE_COMPLETED' and result['repeated_all_outputs_equal']
    assert [c['call_id'] for c in result['calls']]==[0,1,2]
    first=result['calls'][0]['values']
    assert all([(v['shape'],v['full_sha256'],v['request_sha256']) for v in c['values']]==
               [(v['shape'],v['full_sha256'],v['request_sha256']) for v in first] for c in result['calls'])
    return result,{(v['layer'],v['kind']):v for v in first}


def tensor_file(folder,value):
    if value['kind']=='logits':
        path=folder/value['target_file']
        assert path.resolve().is_relative_to((folder/'private_output').resolve())
    else:
        storage=json.loads((folder.parent/'private_batch_storage.json').read_bytes())
        parent=Path(storage['host_path']);assert parent.parent==Path('/tmp/fault28-fia-fullmodel-20261109')
        path=parent/value['private_batch_relative']
        assert path.resolve().is_relative_to(parent.resolve())
    assert path.stat().st_size==2*int(np.prod(value['shape'])) and sha(path)==value['full_sha256']
    return path


def compare_values(a,b,kind):
    """Full arrays for every request; no top-k or summary substitute."""
    assert a.shape==b.shape
    rows=[]
    for slot in range(a.shape[0]):
        left,right=a[slot],b[slot];diff=left!=right;count=int(np.count_nonzero(diff))
        row={'request_slot':slot,'elements':left.size,'changed_elements':count,
             'max_abs':float(np.max(np.abs(left.astype(np.float32)-right.astype(np.float32)))) if count else 0.0}
        if kind=='attention':
            queries=np.count_nonzero(diff,axis=(0,2));changed=np.flatnonzero(queries)
            row.update(changed_queries=len(changed),first_changed_query=int(changed[0]) if len(changed) else None,
                last_changed_query=int(changed[-1]) if len(changed) else None,last_query_changed=int(queries[-1]))
        if kind=='logits':
            row.update(B_top1=int(left[0].argmax()),F_top1=int(right[0].argmax()))
        if count:
            positions=np.flatnonzero(diff)[:8]
            row['first_difference_indices']=[list(map(int,np.unravel_index(int(i),left.shape))) for i in positions]
        rows.append(row)
    return rows


def compare(baseline,fault,logits_only=False):
    br,bv=read(baseline);fr,fv=read(fault)
    inputs=[json.loads((p/'inputs.json').read_bytes()) for p in (baseline,fault)]
    assert inputs[0]==inputs[1],'Actual request/input mapping differs'
    assert sha(baseline/'loaded_parameters.json')==sha(fault/'loaded_parameters.json')
    if logits_only:
        bv={k:v for k,v in bv.items() if k==(None,'logits')}
        fv={k:v for k,v in fv.items() if k==(None,'logits')}
    assert bv.keys()==fv.keys()
    comparisons=[]
    for key in bv:
        v,w=bv[key],fv[key];assert v['shape']==w['shape'] and v['dtype']==w['dtype']=='torch.float16'
        lp,rp=tensor_file(baseline,v),tensor_file(fault,w)
        a=np.memmap(lp,dtype='<f2',mode='r',shape=tuple(v['shape']))
        b=np.memmap(rp,dtype='<f2',mode='r',shape=tuple(w['shape']))
        rows=compare_values(a,b,v['kind'])
        comparisons.append({'layer':v['layer'],'kind':v['kind'],'shape':v['shape'],
            'B_sha256':v['full_sha256'],'F_sha256':w['full_sha256'],
            'all_request_changed_elements':sum(r['changed_elements'] for r in rows),'requests':rows})
        del a,b
    attention=[r for r in comparisons if r['kind']=='attention']
    logits=next(r for r in comparisons if r['kind']=='logits')
    first=min((r['layer'] for r in attention if r['all_request_changed_elements']),default=None)
    last=min((r['layer'] for r in attention if any(q['last_query_changed'] for q in r['requests'])),default=None)
    first_inputs=[]
    if first is not None:
        first_inputs=[r for r in comparisons if r['layer']==first and r['kind'] in ('query','key','value')]
    return {'schema':'fault28.fia_fullmodel.complete_comparison.v1','status':'COMPLETE_ALL_REQUEST_COMPARISON',
        'B':str(baseline),'F':str(fault),'request_rows':inputs[0]['request_rows'],
        'B_result_sha256':sha(baseline/'result.json'),'F_result_sha256':sha(fault/'result.json'),
        'first_changed_attention_layer':first,'first_changed_last_query_attention_layer':last,
        'first_changed_attention_inputs_equal':None if first is None else len(first_inputs)==3 and all(r['all_request_changed_elements']==0 for r in first_inputs),
        'logits_changed_requests':sum(r['changed_elements']>0 for r in logits['requests']),
        'top1_changed_requests':sum(r['B_top1']!=r['F_top1'] for r in logits['requests']),
        'all_observed_elements_equal':all(r['all_request_changed_elements']==0 for r in comparisons),
        'comparisons':comparisons,'private_tensor_payload_exported':False,'logits_only':logits_only,
        'scope':'All request rows and all output elements in the first completed fullgraph call. Subsequent two calls separately verify synchronized full/request hashes. Same-batch numerical evidence, independent of statistical localization.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=compare(a.baseline,a.fault)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('status','first_changed_attention_layer','first_changed_last_query_attention_layer','logits_changed_requests','top1_changed_requests','all_observed_elements_equal')}))


if __name__=='__main__':main()
