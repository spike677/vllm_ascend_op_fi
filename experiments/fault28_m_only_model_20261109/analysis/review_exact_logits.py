"""CPU-only, offline comparison of complete private logits, independently of hits.

No Attention/KV replay and no inference. Each file is bound to the original
phase shape, complete digest and per-request digest before comparing values.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(payload):return hashlib.sha256(payload).hexdigest()


def load(folder,phase):
    file=folder/phase['private_logits_file']
    assert file.resolve().is_relative_to(folder.resolve()) and not file.is_symlink()
    data=file.read_bytes();snapshot=phase['logits'];shape=snapshot['shape']
    assert shape[1]==1 and snapshot['dtype']=='torch.float16'
    assert sha(data)==snapshot['full_sha256'] and len(data)==2*int(np.prod(shape))
    values=np.frombuffer(data,dtype='<f2').reshape(shape)
    assert np.isfinite(values).all()
    assert [sha(v.tobytes()) for v in values]==snapshot['per_batch_sha256']
    return data,values


def compare(folder):
    phases=json.loads((folder/'phase_manifest.json').read_bytes())['phases']
    assert len(phases)==10
    loaded=[load(folder,p) for p in phases]
    pairs=((0,2,'calibration_native_prefill'),(1,3,'calibration_native_decode'),
           (2,4,'native_fault_a_prefill'),(3,5,'native_fault_a_decode'),
           (2,6,'native_fault_b_prefill'),(3,7,'native_fault_b_decode'),
           (4,6,'fault_repeat_prefill'),(5,7,'fault_repeat_decode'),
           (2,8,'native_off_prefill'),(3,9,'native_off_decode'))
    rows=[]
    for left,right,name in pairs:
        a_bytes,a=loaded[left];b_bytes,b=loaded[right]
        assert a.shape==b.shape
        changed=a!=b;delta=np.abs(a.astype('float64')-b.astype('float64'))
        per_request=[]
        for index in range(a.shape[0]):
            top_a=int(a[index,0].argmax());top_b=int(b[index,0].argmax())
            per_request.append({'request_slot':index,'changed_elements':int(changed[index].sum()),
                                'max_abs':float(delta[index].max()),'mean_abs':float(delta[index].mean()),
                                'B_top1_token_id':top_a,'F_top1_token_id':top_b,'top1_changed':top_a!=top_b})
        row={'comparison':name,'B_step':left,'F_step':right,'shape':list(a.shape),
             'bitwise_equal':a_bytes==b_bytes,'changed_elements':int(changed.sum()),
             'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'per_request':per_request}
        if 'fault_' not in name or name.startswith('fault_repeat'):
            assert row['bitwise_equal'] and row['changed_elements']==0,'Logits repeat/recovery is not exact'
        # Cross-check the common runtime observer without using it to determine
        # the offline result. Counts of numerical changes preserve signed zero.
        if left==2 or left==3:
            runtime=phases[right]['logits']['exact_numeric_observation']['comparison']
            assert runtime['changed_elements']==row['changed_elements']
            assert runtime['per_request_changed_elements']==[x['changed_elements'] for x in per_request]
            assert runtime['max_abs']==row['max_abs'] and runtime['bitwise_equal']==row['bitwise_equal']
        rows.append(row)
    return {'schema':'fault28.m_only.independent_private_logits.v1','status':'PASS_COMPLETE_LOGITS_BYTES_REVIEW',
            'case_id':folder.name,'case_result_sha256':sha((folder/'result.json').read_bytes()),
            'phase_manifest_sha256':sha((folder/'phase_manifest.json').read_bytes()),
            'comparisons':rows,'private_logits_published':False,
            'limit':'Logit changes and top-1 changes do not establish answer quality or accuracy loss.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:report=compare(a.case_dir);code=0
    except (ValueError,OSError,KeyError,AssertionError,TypeError) as error:
        report={'status':'FAIL_COMPLETE_LOGITS_BYTES_REVIEW','error':str(error)};code=1
    report['analysis_source_sha256']=sha(Path(__file__).read_bytes())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','case_id','error') if k in report}))
    raise SystemExit(code)
