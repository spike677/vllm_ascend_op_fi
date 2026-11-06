"""CPU comparison of COMPLETE private query outputs, not msprobe localization.

Only small numeric samples, exact counts and hashes are emitted. Original full
target tensors remain private. The fixed reference query positions were chosen
before any F execution, not from observed error positions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

POSITIONS=(0,63,127,255,383,511,512,767,1023,2047)


def sha(data):return hashlib.sha256(data).hexdigest()


def load(folder,call=1,consumer=False):
    result=json.loads((folder/'result.json').read_bytes())
    row=result['calls'][call]
    if row['call_id']!=call:raise ValueError('Unaligned call result')
    item=row['consumer'] if consumer else row
    raw=(folder/item['target_file']).read_bytes()
    shape=(2048,1024) if consumer else (16,2048,128)
    if sha(raw)!=item['target_sha256'] or len(raw)!=np.prod(shape)*2:
        raise ValueError('Target tensor size/SHA differs from recorded complete data')
    values=np.frombuffer(raw,dtype='<f2').reshape(shape)
    if not np.isfinite(values).all():raise ValueError('Nonfinite target output')
    return values,{'sha256':sha(raw),'elements':int(values.size),'shape':list(shape)}


def difference(a,b):
    changed=(a!=b);absolute=np.abs(a.astype('float32')-b.astype('float32'))
    coordinates=np.argwhere(changed)
    samples=[]
    for index in coordinates[:8]:
        key=tuple(index)
        samples.append({'index':index.tolist(),'B':float(a[key]),'F':float(b[key]),'abs':float(absolute[key])})
    query_axis=1 if a.ndim==3 else 0
    others=tuple(i for i in range(a.ndim) if i!=query_axis)
    perquery=changed.sum(axis=others)
    return {'changed_elements':int(changed.sum()),'total_elements':int(a.size),
            'max_abs':float(absolute.max()),'mean_abs':float(absolute.mean(dtype='float64')),
            'queries_with_changes':int(np.count_nonzero(perquery)),
            'per_query_changed_elements':perquery.tolist(),
            'last_query_changed_elements':int(perquery[-1]),'small_samples':samples}


def reference(inputs):
    manifest_path=inputs/'manifest.json';manifest=json.loads(manifest_path.read_bytes())
    source={}
    for name,row in manifest['inputs'].items():
        raw=(inputs/row['file']).read_bytes()
        if sha(raw)!=row['sha256']:raise ValueError('Source reference bytes differ')
        source[name]=np.frombuffer(raw,dtype='<f2').reshape(row['shape'])[0].astype('float64')
    q,k,v=(source[name] for name in ('query','key','value'))
    scale=np.float64(np.float32(manifest['source_sdpa_scalar_arguments']['scale']))
    result={}
    for position in POSITIONS:
        heads=[]
        for head in range(16):
            scores=k[head//2,:position+1]@q[head,position]*scale
            weights=np.exp(scores-scores.max());weights/=weights.sum()
            heads.append(weights@v[head//2,:position+1])
        result[position]=np.stack(heads)
    return result


def ref_errors(values,ref):
    result=[]
    for position,r in ref.items():
        got=values[:,position,:].astype('float64');delta=got-r
        result.append({'query':position,'elements':int(got.size),'max_abs':float(np.abs(delta).max()),
                       'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(r),1e-30))})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--fault',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=Path('/home/fault28-causal-20261029/inputs'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    baseline,bmeta=load(a.baseline);fault,fmeta=load(a.fault)
    bp,bpmeta=load(a.baseline,consumer=True);fp,fpmeta=load(a.fault,consumer=True)
    ref=reference(a.inputs)
    repeats={}
    for name,folder in [('B',a.baseline),('F',a.fault)]:
        values=[load(folder,i)[0] for i in (0,1,2)]
        projections=[load(folder,i,True)[0] for i in (0,1,2)]
        repeats[name]={'complete_target_equal_all_three':all(np.array_equal(values[0],v) for v in values[1:]),
                       'complete_projection_equal_all_three':all(np.array_equal(projections[0],v) for v in projections[1:])}
    report={'schema':'fault28.complete_private_target_comparison.v1','status':'COMPARED',
            'target':{'B':bmeta,'F':fmeta,**difference(baseline,fault)},
            'consumer':{'B':bpmeta,'F':fpmeta,**difference(bp,fp)},'repeats':repeats,
            'fp64_reference':{'positions':list(POSITIONS),'scale':'actual float32 operator scale',
                              'B':ref_errors(baseline,ref),'F':ref_errors(fault,ref)},
            'scope':'Complete target request all heads/queries/dimensions; selected reference only. No full tensor published. Independent of statistics localization.'}
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'changed':report['target']['changed_elements'],'max_abs':report['target']['max_abs'],
                      'consumer_changed':report['consumer']['changed_elements'],'output':str(a.output)}))


if __name__=='__main__':main()
