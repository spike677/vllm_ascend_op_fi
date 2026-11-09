"""Exact CPU-only comparisons of the already copied logical tensor bytes.

No framework imports, NPU calls, model writes or injection decisions. Only
native-baseline references are saved privately; F/OFF compare every element.
"""
import hashlib
from pathlib import Path
import numpy as np


def sha(data):return hashlib.sha256(data).hexdigest()


def compare_bytes(left,right,shape,query_axis):
    assert len(left)==len(right)==2*int(np.prod(shape))
    a=np.frombuffer(left,dtype='<f2').reshape(shape)
    b=np.frombuffer(right,dtype='<f2').reshape(shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    mask=a!=b
    per_request=mask.reshape(shape[0],-1).sum(axis=1).tolist()
    axis=query_axis%len(shape)
    others=tuple(i for i in range(len(shape)) if i!=axis)
    per_query=mask.sum(axis=others).tolist()
    max_abs=0.;absolute_sum=0.
    # Bound temporary differences to one request rather than a whole B8 tensor.
    for slot in range(shape[0]):
        delta=np.abs(a[slot].astype('float32')-b[slot].astype('float32'))
        max_abs=max(max_abs,float(delta.max()));absolute_sum+=float(delta.sum(dtype='float64'))
    last_a=np.take(a,-1,axis=axis);last_b=np.take(b,-1,axis=axis)
    samples=[]
    for row in np.argwhere(mask)[:8]:
        index=tuple(row)
        samples.append({'index':row.tolist(),'B':float(a[index]),'F':float(b[index])})
    return {'bitwise_equal':left==right,'elements':int(a.size),'changed_elements':int(mask.sum()),
            'per_request_changed_elements':per_request,'per_query_changed_elements':per_query,
            'queries_with_changes':int(np.count_nonzero(per_query)),'max_abs':max_abs,
            'mean_abs':absolute_sum/a.size,'last_query_changed_elements':int(np.count_nonzero(last_a!=last_b)),
            'last_query_B_sha256':sha(last_a.tobytes()),'last_query_F_sha256':sha(last_b.tobytes()),
            'small_samples':samples}


def observe(payload,shape,context):
    assert context['mode'] in ('calibration','native','controlled','off')
    path=Path(context['reference_file']);axis=context['query_axis']
    values=np.frombuffer(payload,dtype='<f2').reshape(shape)
    assert np.isfinite(values).all()
    last=np.take(values,-1,axis=axis)
    row={'schema':'fault28.m_only.exact_observation.v1','mode':context['mode'],
         'stage':context['stage'],'kind':context['kind'],'layer_index':context['layer_index'],
         'shape':list(shape),'dtype':'torch.float16','full_sha256':sha(payload),
         'last_query_sha256':sha(last.tobytes()),'private_reference_file':str(path),
         'numeric_data_source':'Logical bytes copied by common tensor_snapshot; CPU comparison only.'}
    if context['mode']=='calibration':
        row['comparison_status']='CALIBRATION_NO_REFERENCE'
    elif context['mode']=='native':
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(payload)
        row.update(comparison_status='BASELINE_REFERENCE_SAVED',reference_sha256=sha(payload))
    else:
        baseline=path.read_bytes()
        row.update(comparison_status='COMPLETE_ELEMENTWISE_COMPARISON',reference_sha256=sha(baseline),
                   comparison=compare_bytes(baseline,payload,shape,axis))
    return row
