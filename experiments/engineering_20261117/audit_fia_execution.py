"""Audit bounded fresh FIA runtime receipts, separately from statistics analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(folder):
    path=folder/'run/result.json';data=json.loads(path.read_bytes())
    assert data['status']=='EXPERIMENTAL_EXECUTION_COMPLETED_NOT_LAUNCH_QUALIFIED'
    assert not data['public_worker_qualified'] and not data['native_launch_verified']
    assert all(c['finite'] and c['input_hashes_unchanged'] for c in data['calls'])
    inputs=folder/'run/inputs.json'
    return dict(result=ref(path),inputs=ref(inputs),data=data),json.loads(inputs.read_bytes())['tensors']


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('bridge','pair'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();names=('stock','off') if a.mode=='bridge' else ('baseline','fault','off-recovery')
    cases={};identities=[]
    for name in names:
        cases[name],inputs=load(a.root/('fia-'+name));identities.append(inputs)
    assert all(item==identities[0] for item in identities)
    records={k: v['data'] for k,v in cases.items()}
    if a.mode=='bridge':
        hashes={c['output_sha256'] for d in records.values() for c in d['calls']}
        assert len(hashes)==1,'STOCK_AND_OFF_COMPLETE_OUTPUT_DIFFER'
    else:
        assert records['off-recovery']['off_exact_restore'] is True
        assert all(records[k]['repeated_complete_output_stable'] for k in ('baseline','fault'))
        assert records['baseline']['completion_sha256']==records['fault']['completion_sha256']
    report=dict(schema='demoncat.fia.experimental_execution_summary.v1',mode=a.mode,
        status='EXPERIMENTAL_BRIDGE_EXACT' if a.mode=='bridge' else 'EXPERIMENTAL_BFOFF_EXECUTED',
        scope='SYNTHETIC_STANDALONE_FIA_NOT_MODEL_OR_WORKER',cases=cases,
        actual_npu_calls=sum(d['actual_npu_fia_calls'] for d in records.values()),
        complete_inputs_equal=True,stock_off_exact=True if a.mode=='bridge' else None,
        complete_bf_output_changed=(records['baseline']['calls'][-1]['output_sha256']!=
            records['fault']['calls'][-1]['output_sha256']) if a.mode=='pair' else None,
        native_launch_verified=False,public_worker_qualified=False,model_acceptance=False,
        all_eligible=False,every_match=False,
        independent_statistics='NOT_READ_BY_THIS_RUNTIME_AUDITOR')
    with a.out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','actual_npu_calls','stock_off_exact','complete_bf_output_changed')}))


if __name__=='__main__':main()
