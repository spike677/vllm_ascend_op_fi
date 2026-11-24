"""Independent receipt checks for the v2 weight example, not registration."""
import argparse,hashlib,json
from pathlib import Path

COUNTERS=('observed','eligible','attempted','applied','changed','consumed','restored','failed')


def review(data):
    issues=[];rows=[];scopes=data.get('scopes',[]);owners=[s.get('owner') for s in scopes]
    if len(scopes)!=2 or None in owners or len(set(owners))!=2:issues.append('TWO_DISTINCT_OWNERS_REQUIRED')
    if data.get('status')!='TWO_OWNER_WEIGHT_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION':issues.append('RUN_NOT_COMPLETE')
    if data.get('missing_permit',{}).get('exit_code')!=4:issues.append('MISSING_PERMIT_NEGATIVE')
    previous_workers=None
    for scope in scopes:
        result=scope.get('disable',{});cleanup=result.get('cleanup',{})
        stopped=cleanup.get('disable',[]);workers=[r.get('worker') for r in stopped]
        if len(stopped)!=2 or [w.get('rank') for w in workers if isinstance(w,dict)]!=[0,1]:issues.append('TP2_RANK_IDENTITY_REQUIRED')
        if previous_workers is not None and workers!=previous_workers:issues.append('WORKERS_CHANGED_BETWEEN_OWNERS')
        previous_workers=workers
        if not all(result.get('recovery',{}).get(k) is True for k in ('bytes_restored','resources_zero','canary_exact','restored')):
            issues.append('THREE_PART_RECOVERY_REQUIRED')
        canary=result.get('canary',{})
        if not canary.get('before') or canary.get('before')!=canary.get('after'):issues.append('CANARY_VALUES_DIFFER')
        before=canary.get('before',[])
        if len(before)!=2 or [r.get('rank') for r in before]!=[0,1] or any(
                r.get('all_finite') is not True or not r.get('output') for r in before):issues.append('CANARY_FINITE_RANK_PROOF')
        for row in stopped:
            state=row.get('response',{}).get('provider_state',{});events=state.get('events',[])
            if state.get('derived_copy')!='original_object':issues.append('DERIVED_COPY_NOT_RESTORED')
            if not events:issues.append('NO_WRITE_EVENTS')
            for e in events:
                if any(type(e.get(k)) is not int or e[k]<0 for k in COUNTERS):issues.append('COUNTER_SCHEMA');continue
                if not (e['observed']>=e['eligible']==e['attempted']==e['applied']==e['consumed']==e['restored']>0
                        and 0<=e['changed']<=e['applied'] and e['failed']==0):issues.append('PERSISTENT_COUNTER_RELATION')
                if e.get('restore_kind')!='BYTE_EXACT_SAME_ALLOCATION_GENERATION':issues.append('ROLLBACK_IS_NOT_ACCEPTANCE')
            readers=state.get('readers',[])
            if (type(state.get('reader_observations')) is not int or state['reader_observations']!=len(readers)
                    or len(readers)<4 or any(r.get('consumed_changed_scale') is not True
                        or r.get('device_read_complete') is not True or not r.get('request_id') for r in readers)):
                issues.append('ACTUAL_READER_EVIDENCE')
            rows.append(dict(owner=scope.get('owner'),rank=row.get('worker',{}).get('rank'),site=state.get('selected_site'),
                counters={k:sum(e.get(k,0) for e in events) for k in COUNTERS},
                changed_elements=sum(e.get('changed_elements',0) for e in events),reader_observations=len(readers)))
        audit=scope.get('cleanup',[])
        if len(audit)!=2 or {r.get('rank') for r in audit}!={0,1} or any(
            type(r.get(k)) is not int or r[k]!=0 for r in audit for k in ('leases','owners')):issues.append('RESOURCES_NOT_ZERO')
        if scope.get('detach',{}).get('status')!='PRODUCT_DETACHED':issues.append('DETACH_MISSING')
    return dict(schema='dcat.persistent_example.review.v1',passed=not issues,issues=sorted(set(issues)),ranks=rows,
        qualification_granted=False,limits=['WEIGHT_SCALE_EXAMPLE_ONLY','NO_KV_ACCEPTANCE',
        'DEDICATED_NETWORK_NONE_SOLE_HTTP_SUBMITTER','NO_REAL_CANCEL_ACCEPTANCE','NO_A3_QUALIFICATION'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.receipt.read_bytes();result=review(json.loads(raw));result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],issues=result['issues'],receipt_sha256=result['receipt_sha256'])))
    raise SystemExit(0 if result['passed'] else 1)
