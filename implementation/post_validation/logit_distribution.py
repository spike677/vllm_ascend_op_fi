"""Post-hoc logits interpretation, separate from all preregistered gates."""
import hashlib,json,os
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import numpy as np

ROOT=Path('/home/fault28-reduce-replay-20261113')
def main():
    review=json.loads((ROOT/'reports/formal/review.json').read_bytes())
    assert review['accepted_before_controls']
    state=json.loads((ROOT/'search_state.json').read_bytes());values=[]
    for arm in ('B','F'):
        folder=ROOT/state['cases']['formal_'+arm]['folder']
        result=json.loads((folder/'result.json').read_bytes())
        row=next(r for r in result['calls'][0]['values'] if r['kind']=='logits')
        path=folder/row['private_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['full_sha256']
        values.append(np.fromfile(path,dtype='<f2').reshape(row['shape'])[:,0,:].astype(np.float64))
    rows=[]
    def softmax(x):
        weights=np.exp(x-x.max());return weights/weights.sum()
    for request,(b,f) in enumerate(zip(*values)):
        delta=f-b;offset=float(delta.mean());p,q=softmax(b),softmax(f);m=(p+q)/2
        def contribution(x):
            mask=x>0;return float(np.sum(x[mask]*np.log(x[mask]/m[mask])))
        rows.append({'request':request,'mean_logit_shift':offset,
            'max_abs_after_removing_mean_shift':float(np.max(np.abs(delta-offset))),
            'softmax_total_variation':float(np.sum(np.abs(p-q))*.5),
            'softmax_Jensen_Shannon_nats':(contribution(p)+contribution(q))*.5,
            'B_top1':int(b.argmax()),'F_top1':int(f.argmax()),
            'B_top1_probability':float(p.max()),'F_top1_probability':float(q.max())})
    report={'status':'POST_HOC_LOGIT_DISTRIBUTION_OBSERVATION','used_for_statistical_first_or_acceptance':False,
        'requests':rows,'limits':['CPU-FP64 softmax of already returned FP16 logits, not a new model invocation.',
            'Common logit offsets do not change softmax; centered differences and TV quantify the remaining change.',
            'No continuous-generation or task-quality claim. No acceptance threshold added.']}
    with (ROOT/'reports/logit_distribution.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'mean_shift_range':[min(x['mean_logit_shift'] for x in rows),max(x['mean_logit_shift'] for x in rows)],
        'centered_max_abs_range':[min(x['max_abs_after_removing_mean_shift'] for x in rows),max(x['max_abs_after_removing_mean_shift'] for x in rows)],
        'TV_range':[min(x['softmax_total_variation'] for x in rows),max(x['softmax_total_variation'] for x in rows)]}))
if __name__=='__main__':main()
