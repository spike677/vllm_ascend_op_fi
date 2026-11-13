"""Post-localization CPU-FP64 mechanism check; never a statistics input."""
import hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np

ROOT=Path('/home/fault28-reduce-replay-20261113')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(row,base,key='file'):
    p=base/row[key];assert sha(p)==row.get('sha256',row.get('full_sha256'))
    return np.memmap(p,dtype='<f2',mode='r',shape=tuple(row['shape']))
def main():
    assert json.loads((ROOT/'reports/calibration_gate.json').read_bytes())['accepted']
    state=json.loads((ROOT/'search_state.json').read_bytes())
    folders={k:ROOT/state['cases'][v]['folder'] for k,v in [('h','h_input_00'),('B','real_B'),('F','real_F')]}
    m=json.loads((folders['h']/'qkv_manifest.json').read_bytes())
    q,k,v=[load(m[name],folders['h']) for name in ('query','key','value')]
    outputs=[]
    for arm in ('B','F'):
        result=json.loads((folders[arm]/'result.json').read_bytes())
        outputs.append(load(result['calls'][0]['values'][0],folders[arm],'private_file'))
    b,f=outputs;assert np.array_equal(b[:,:,:128,:],f[:,:,:128,:])
    records=[]
    # Frozen query/head list independent of where large tensor differences occur.
    for request in range(8):
        for head in (0,7,15):
            for query in (127,128,255,511,1023,2047):
                scores=k[request,head//2,:query+1].astype(np.float64)@q[request,head,query].astype(np.float64)*float(np.float32(128**-.5))
                weights=np.exp(scores-scores.max());den=weights.sum()
                extra=weights[:128].sum() if query>=128 else 0.0
                ratio=den/(den+extra)
                left=b[request,head,query].astype(np.float64);right=f[request,head,query].astype(np.float64)
                predicted=left*ratio
                norm=float(np.linalg.norm(right))
                error=float(np.linalg.norm(right-predicted))/norm if norm else None
                records.append({'request':request,'head':head,'query':query,
                    'expected_denominator_ratio':float(ratio),'relative_L2_from_B_scaled_by_QK_derived_ratio':error,
                    'max_abs_prediction_residual':float(np.max(np.abs(right-predicted)))})
    after=[r for r in records if r['query']>=128]
    worst=max(r['relative_L2_from_B_scaled_by_QK_derived_ratio'] for r in after if r['relative_L2_from_B_scaled_by_QK_derived_ratio'] is not None)
    summary={'status':'POST_LOCALIZATION_MECHANISM_OBSERVATION','records':records,
        'samples':len(records),'q0_127_complete_output_byte_equal':True,
        'largest_sampled_relative_L2_prediction_residual':worst,
        'sampled_QK_derived_denominator_ratio_range':[min(r['expected_denominator_ratio'] for r in after),max(r['expected_denominator_ratio'] for r in after)],
        'method':'Use unchanged real Q/K on CPU FP64 to predict D/(D+D0). Compare existing NPU B and F outputs; no tensor is modified or returned to the model.',
        'used_for_independent_localization':False,'tolerance_tuned_from_this_report':False,
        'limits':['Prediction uses already rounded baseline output; scalar FP64 score evaluation differs from native matmul/softmax/PV rounding.',
                  'These sampled complete-array observations are auxiliary. Native analytic fixture is the exact formula check.']}
    with (ROOT/'reports/real_qkv_mechanism.json').open('x') as out:json.dump(summary,out,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
if __name__=='__main__':main()
