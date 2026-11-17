"""Prepare bounded standalone new-vendor cases; does not grant Worker qualification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    for name in ('source','build','schema','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phases',nargs='+',choices=('stock','off','baseline','fault','off_recovery'),required=True)
    p.add_argument('--batch',type=int,choices=(1,8),default=8)
    p.add_argument('--registered',type=Path,help='Separately derived registration-only vendor directory')
    a=p.parse_args()
    assert 1<=len(a.phases)<=8 and len(set(a.phases))==len(a.phases)
    completion=a.build/'run/evidence/completion.json'
    original_completion=completion
    if a.registered:
        completion=a.registered/'completion.json'
    metadata=json.loads(completion.read_bytes())
    assert metadata['status']=='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
    schema=a.schema/'fia_abi_schema.json'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if a.registered:
        assert metadata['parent_completion']['sha256']==digest(original_completion)
        assert metadata['derivation_scope']=='ONLY_ADD_OFFICIAL_BINARY_SUMMARY_INDEX'
    variant=metadata['variant']
    vendor='dcat_f28_legal_transformer' if variant=='legal_tiling' else 'dcat_f28_replay_transformer'
    host_vendor=(a.registered/'vendors'/vendor) if a.registered else (a.build/'run/stage/vendors'/vendor)
    assert host_vendor.is_dir()
    completion_arg='/registered/completion.json' if a.registered else '/build/run/evidence/completion.json'
    vendor_arg='/registered/vendors/'+vendor if a.registered else '/build/run/stage/vendors/'+vendor
    cases=[]
    for phase in a.phases:
        mounts=[[str(a.source),'/repo','ro'],[str(a.build),'/build','ro'],[str(a.schema),'/schema','ro']]
        if a.registered:mounts.append([str(a.registered),'/registered','ro'])
        cases.append(dict(id='fia-'+phase.replace('_','-'),image='cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8',
            timeout=900,mounts=mounts,
            env=dict(PYTHONPATH='/repo:/repo/src:/repo/fault39/src:/msprobe_packages',OMP_NUM_THREADS='4',ACLNN_CACHE_LIMIT='0'),
            command='source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexec python3 /repo/experiments/engineering_20261117/fresh_fia_qualification.py '
                '--completion '+completion_arg+' --completion-sha256 '+digest(completion)+
                ' --vendor '+vendor_arg+' --public-schema /schema/fia_abi_schema.json --schema-sha256 '+digest(schema)+
                ' --phase '+phase+' --batch '+str(a.batch)+' --out /evidence/run --experimental-qualification'+
                (' --statistics' if phase in ('baseline','fault') else '')))
    with a.out.open('x') as stream:json.dump(cases,stream,indent=2)


if __name__=='__main__':main()
