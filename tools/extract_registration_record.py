"""Read local receipts, rerun review, print bounded canonical registration text.

No state changes, registration, network access, or full tensor export.
"""
import argparse,hashlib,json,sys
from pathlib import Path

SCHEMA='dcat.reg.extract.v1'
COUNTERS=('observed','eligible','attempted','applied','changed','consumed','restored','failed')
DEPLOYMENT=('hardware_family','hardware_device_name','vllm','vllm_ascend','torch','torch_npu','cann','image_digest','model_config_sha256','quant_model_description_sha256','safetensors_index_sha256','max_model_len','max_num_batched_tokens','max_num_seqs','tp','dp','pp','eager','dtype','live_config_matches_file')


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def get(obj,*path):
    for key in path:
        if not isinstance(obj,dict):return None
        obj=obj.get(key)
    return obj

def unique(values):
    if not values:return None
    return values[0] if all(v==values[0] for v in values) else None

def tri_all(values):
    if any(v is False for v in values):return False
    if not values or any(v is None for v in values):return None
    return True

def file_id(path,raw):return [Path(path).name,len(raw),sha(raw)]

def extract(summary_path,receipt_path,*,release_manifest=None):
    import importlib.util
    path=Path(__file__).parent/'_r5'/'review_continuous_public.py'
    if not path.exists():path=Path(__file__).parent/'review_continuous_public.py'
    if sha(path.read_bytes())!='deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc':raise ValueError('R5_REVIEWER_BYTES_MISMATCH')
    spec=importlib.util.spec_from_file_location('dcat_frozen_r5_review',path);reviewer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reviewer)
    sr=Path(summary_path).read_bytes();rr=Path(receipt_path).read_bytes()
    summary=json.loads(sr);raw=json.loads(rr);claimed=summary.get('review',summary)
    error=None;reviewed=None
    try:reviewed=reviewer.review(raw,require_negative_creation=True)
    except Exception as e:error=type(e).__name__+':'+str(e)
    # Match every original review field. Extra byte/source metadata are allowed.
    agree=all(canonical(claimed.get(k))==canonical(v) for k,v in reviewed.items()) if reviewed else None
    linked=summary.get('source_receipt_sha256',claimed.get('source_receipt_sha256'))
    receipt_link=(linked==sha(rr)) if linked is not None else None
    scopes=[];identities=[];permits=[];hardware=[];conflicts=[]
    for index,s in enumerate(raw.get('scopes',[])):
        states=get(s,'requests')
        states=get(states[-1],'result','response','provider_state','rank_states') if isinstance(states,list) and states else {}
        rank_rows=[]
        for rank,state in sorted((states or {}).items(),key=lambda kv:int(kv[0])):
            use=state.get('deployment_acceptance_use',{});actual=use.get('actual',{})
            permit=get(use,'permit','source_manifest_sha256');permits.append(permit)
            identity={k:actual.get(k) for k in DEPLOYMENT};identities.append(identity)
            hardware.append([int(rank),actual.get('hardware_logical_device')])
            rows=get(state,'evidence','actual_requests') or [];by={r.get('phase'):r for r in rows};f=by.get('fault',{});c=f.get('coverage',{})
            equal=lambda phase: (by['normal'].get('output')==by[phase].get('output')) if 'normal' in by and phase in by and by['normal'].get('output') is not None else None
            counters=[c.get(k) for k in COUNTERS]
            consumer=(c['eligible']>0 and c['consumed']==c['eligible']) if all(type(c.get(k)) is int for k in ('eligible','consumed')) else None
            clean=get(s,'unregister','response','provider_state','rank_states',rank)
            zero=tri_all([(clean.get(k) in (0,False)) if clean and k in clean else None for k in ('leases','inflight','owned_hooks','registered','native_registered','enabled')])
            rank_rows.append([int(rank),counters,consumer,[f.get('bytes_restored'),zero,tri_all([equal('off'),equal('health')])]])
        missing=s.get('negative_missing_permit');creation=s.get('negative_missing_permit_creation');cleanup=s.get('cleanup')
        negative1=(s['negative_extra_advance']=='GROUP_PHASES_EXHAUSTED') if 'negative_extra_advance' in s else None
        negative2=tri_all([tri_all([r.get('can_prepare') is False for r in missing]) if missing else None,
            tri_all([get(r,'prepare','status') in ('PROVIDER_UNAVAILABLE','REJECTED') and get(r,'snapshot','ownership_retained') is False for r in creation]) if creation else None,
            tri_all([r.get('permit_remaining') is False and r.get('runtime_leases')==r.get('runtime_tenants')==0 for r in cleanup]) if cleanup else None])
        scopes.append([s.get('owner'),rank_rows,[negative1,negative2],s.get('fault_time_ratio')])
    combination={k:unique([d[k] for d in identities]) for k in DEPLOYMENT}
    for key in DEPLOYMENT:
        if identities and any(d[key]!=identities[0][key] for d in identities):conflicts.append('deployment:'+key)
    permit=unique(permits)
    if permits and any(p!=permits[0] for p in permits):conflicts.append('permit')
    package=[raw.get('release_tag'),raw.get('source_commit'),permit]
    if release_manifest is not None:
        manifest=json.loads(Path(release_manifest).read_bytes())
        # Optional explicit package file is evidence, never infer from path/default.
        if manifest.get('source_manifest_sha256')!=permit:conflicts.append('release_permit')
        else:
            package[1]=manifest.get('source_commit')
            tags={v['path'][:-7] for v in manifest.get('files',[]) if v.get('path','').endswith('.tar.gz') and v['path'].startswith('demoncatpfi-rc3-preview-')}
            package[0]=manifest.get('tag') or (next(iter(tags)) if len(tags)==1 else None)
    value=dict(schema=SCHEMA,p=package,d=[combination[k] for k in DEPLOYMENT],
        devices=sorted({tuple(v) for v in hardware}),m=raw.get('mechanism'),s=scopes,
        review=[reviewed is not None,error,agree,receipt_link,sha(Path(reviewer.__file__).read_bytes())],
        files=[file_id(summary_path,sr),file_id(receipt_path,rr)],conflicts=conflicts)
    return seal(value)

def seal(value):
    value=dict(value);value.pop('extract_sha256',None);value['extract_sha256']=sha(canonical(value))
    if len(canonical(value))+1>4096:raise ValueError('EXTRACT_EXCEEDS_4096_BYTES_NO_FIELDS_TRUNCATED')
    return value

def verify(value):
    body=dict(value);claimed=body.pop('extract_sha256',None)
    if value.get('schema')!=SCHEMA or claimed!=sha(canonical(body)):raise ValueError('EXTRACT_HASH_OR_SCHEMA_MISMATCH')
    if len(canonical(value))+1>4096:raise ValueError('EXTRACT_EXCEEDS_4096_BYTES')
    issues=[]
    if value['conflicts']:issues+=value['conflicts']
    if value['review'][:4]!=[True,None,True,True]:issues.append('LOCAL_REVIEW_OR_SUMMARY_LINK_NOT_PASSED')
    if len(value['s'])!=2 or len({s[0] for s in value['s']})!=2:issues.append('TWO_INDEPENDENT_SCOPES_REQUIRED')
    for i,s in enumerate(value['s']):
        for r in s[1]:
            c=r[1]
            if len(c)!=8 or any(type(n) is not int or n<0 for n in c) or not (c[0]>=c[1]>0 and c[1]==c[2]==c[3]==c[5]==c[6] and c[4]<=c[3] and c[7]==0):issues.append(f'COUNTERS:{i}:{r[0]}')
            if r[2] is not True or r[3]!=[True,True,True]:issues.append(f'CONSUMER_RECOVERY:{i}:{r[0]}')
        if s[2]!=[True,True]:issues.append(f'NEGATIVE:{i}')
    registration_issues=[]
    if any(v is None for v in value['p']+value['d']) or any(v[1] is None for v in value['devices']):registration_issues.append('IDENTITY_MISSING')
    actual=dict(zip(DEPLOYMENT,value['d']))
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import registration_identity_issues
    registration_issues+=registration_identity_issues(actual)
    tp=actual.get('tp')
    for i,scope in enumerate(value['s']):
        if type(tp) is not int or tp not in (1,2) or [r[0] for r in scope[1]]!=list(range(tp)):issues.append(f'RANK_SET:{i}')
    expected=['demoncatpfi-rc3-preview-20261122-r5','d4ae552df2b4fd82641bb39f234d2071dcccaed5','eff3ec54d292deec219555f32fc928278c30c23541e3ef794ef7f89f152a2fff']
    if value['p']!=expected:registration_issues.append('R5_PACKAGE_IDENTITY_MISMATCH')
    if actual.get('hardware_family')!='A3' or actual.get('hardware_device_name')!='Ascend910_9382' or actual.get('vllm')!='0.26.0' or actual.get('vllm_ascend')!='0.26.0rc1' or actual.get('cann')!=['9.1.0']:registration_issues.append('R5_COMBINATION_REQUIRES_REVIEW')
    # Valid hash syntax is not proof this is the deployed yellow weight artifact.

    return dict(consistent=not issues,issues=issues,registration_ready=not issues and not registration_issues,registration_issues=registration_issues,registration_written=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',nargs='?');p.add_argument('receipt',nargs='?');p.add_argument('--release-manifest');p.add_argument('--verify')
    a=p.parse_args()
    if a.verify:result=verify(json.loads(Path(a.verify).read_bytes()))
    else:
        if not a.summary or not a.receipt:p.error('summary and receipt required')
        result=extract(a.summary,a.receipt,release_manifest=a.release_manifest)
    raw=canonical(result)+b'\n'
    if len(raw)>2048:print('Notice: extract exceeds 2048-byte target; below 4096-byte hard limit.',file=sys.stderr)
    sys.stdout.buffer.write(raw)
if __name__=='__main__':main()
