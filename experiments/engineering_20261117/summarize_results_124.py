"""Summarize runtime receipts separately from independent raw-statistics reports."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
    for path in sorted(a.run.glob('*/result.json')):
        raw=path.read_bytes();d=json.loads(raw)
        row={'case':path.parent.name,'result_path':str(path),'result_sha256':hashlib.sha256(raw).hexdigest()}
        for k in ('status','scope','mechanism','arm','pretrained_model','public_vllm_worker','worker_extension',
                  'model_path','error','cleanup_error','exact_off','changed_elements','observed_inf','observed_nan',
                  'live_leases','cache_restored_exact','non_target_exact','source_sha256','library_sha256',
                  'all_observed_metadata_verified','healthy_before','healthy_after'):
            if k in d:row[k]=d[k]
        if 'provider' in d:
            state=d['provider'];row['provider_qualification']=state.get('qualification')
            row['provider_state']={k:v for k,v in state.items() if k not in ('qualification','events','calls','model_inventory')}
            clean=d.get('cleanup',{});row['cleanup']={k:clean.get(k) for k in
                ('registered','enabled','hook_owned','leases','inflight','recovery_confirmed')}
        if 'phase' in d:
            row['phase']=d['phase'];row['off']=d.get('off');row['qualification']=d.get('qualification')
        if 'audit' in d:
            audit=d['audit'];records=audit.get('rows',[])
            row['audit']={'status':audit.get('status'),'actual_calls':len(records),
                'operators':sorted({r['actual_operator'] for r in records}),
                'layouts':sorted({r.get('layout','') for r in records}),
                'modules':sorted({r.get('module_path','') for r in records}),
                'refusals':sorted({r.get('refusal','') for r in records}),
                'all_metadata_verified':all(r.get('metadata_binding')=='VERIFIED' for r in records),
                'all_consumers_returned':all(r.get('consumer_returned') for r in records),
                'native_launch_verified':False,'injection_enabled':False}
        if 'phases' in d:
            row['phases']=[dict(phase=r.get('phase'),advance_status=r.get('advance',{}).get('status'),
                output=r.get('output'),query=r.get('query')) for r in d['phases']]
            row['finish']=d.get('finish')
        rows.append(row)
    with a.out.open('x') as stream:json.dump(dict(raw_statistics_published=False,run=str(a.run),cases=rows),stream,indent=2,allow_nan=False)
    print(json.dumps([dict(case=r['case'],status=r.get('status'),error=r.get('error')) for r in rows]))


if __name__=='__main__':main()
