"""Independent KV example review. No deployment registration is granted."""
import argparse,hashlib,json
from pathlib import Path
from tools.review_persistent_example import review as weight_review


def review(data,*,require_negatives=False):
    # Reuse the same owner/Worker/canary/resource checks, without treating KV
    # as a weight-scale proof. Translate only those weight-specific fields.
    import copy
    proxy=copy.deepcopy(data);proxy['status']='TWO_OWNER_WEIGHT_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION'
    issues=[];details=[]
    if data.get('status')!='TWO_OWNER_KV_VALUE_CELL_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION':issues.append('KV_RUN_INCOMPLETE')
    for scope,converted in zip(data.get('scopes',[]),proxy.get('scopes',[])):
        if scope.get('fault_hidden_changed') is not True:issues.append('KV_MODEL_EFFECT_NOT_OBSERVED')
        states=scope.get('disable',{}).get('cleanup',{}).get('disable',[])
        converted_states=converted.get('disable',{}).get('cleanup',{}).get('disable',[])
        for row,target in zip(states,converted_states):
            s=row.get('response',{}).get('provider_state',{});t=target['response']['provider_state']
            sites=s.get('selected_sites',[]);events=s.get('events',[]);readers=s.get('readers',[])
            if len(set(sites))!=16 or sorted(e.get('name','') for e in events)!=sorted(sites):issues.append('KV_16_REAL_WRITE_SITES_REQUIRED')
            if any(e.get('restore_kind')!='BYTE_EXACT_SAME_REQUEST_GENERATION_BEFORE_PAGE_FREE' for e in events):issues.append('KV_REQUEST_RELEASE_RESTORE_UNPROVEN')
            if (len(readers)<48 or set(r.get('site') for r in readers)!=set(sites)
                    or any(r.get('changed_cache_value_read') is not True or r.get('device_read_complete') is not True
                           or not r.get('request_id') or 'fused_infer_attention_score' not in r.get('consumer','') for r in readers)):
                issues.append('KV_ACTUAL_FIA_READER_REQUIRED')
            if len({r.get('request_id') for r in readers})!=1:issues.append('KV_SERIAL_REQUEST_ASSOCIATION')
            if s.get('persistent_leases')!=0:issues.append('KV_PAGE_LEASE_REMAINS')
            # Restore tags are checked above; only provide the generic checker
            # its own vocabulary. Never rewrite the original receipt bytes.
            t['derived_copy']='original_object'
            for e in t.get('events',[]):e['restore_kind']='BYTE_EXACT_SAME_ALLOCATION_GENERATION'
            for r in t.get('readers',[]):r['consumed_changed_scale']=r.get('changed_cache_value_read')
            details.append(dict(owner=scope.get('owner'),rank=row.get('worker',{}).get('rank'),sites=len(sites),
                                writes=len(events),readers=len(readers),changed_elements=sum(e.get('changed_elements',0) for e in events)))
    generic=weight_review(proxy);issues.extend(generic['issues'])
    cancel_checked=False
    if require_negatives:
        for scope in data.get('scopes',[]):
            if scope.get('range_negative',{}).get('exit_code')!=4:issues.append('KV_RANGE_NEGATIVE_MISSING')
            audits=scope.get('page_audit',[])
            if (len(audits)!=2 or [r.get('rank') for r in audits]!=[0,1]
                    or any(r.get('tickets')!=0 or r.get('leases')!=0 for r in audits)
                    or audits[0].get('events')!=audits[1].get('events')):issues.append('KV_POSITIVE_PAGE_AUDIT')
            before=scope.get('enable',{}).get('ranks',[])
            after=scope.get('disable',{}).get('cleanup',{}).get('disable',[])
            if len(before)!=2 or len(after)!=2:issues.append('KV_OWNER_RANK_CHAIN_MISSING')
            else:
                for b,a in zip(before,after):
                    ticket=a.get('response',{}).get('ticket',{});bs=b.get('response',{}).get('provider_state',{});ast=a.get('response',{}).get('provider_state',{})
                    if (ticket.get('owner')!=scope.get('owner') or ticket.get('worker')!=a.get('worker')
                            or b.get('worker')!=a.get('worker') or not bs.get('owner')
                            or bs.get('owner')!=ast.get('owner') or bs.get('generation')!=ast.get('generation')):
                        issues.append('KV_OWNER_RANK_CHAIN_MISMATCH')
        c=data.get('cancel_scope',{});audit=c.get('audit',[]);rid=c.get('http_id');endings=[]
        if not rid or c.get('actual_abort_proven') is not True or len(audit)!=2:issues.append('KV_REAL_HTTP_ABORT_MISSING')
        else:
            for r in audit:
                end=[e for e in r.get('events',[]) if e.get('action')=='before_free' and e.get('request_id','').startswith(rid)]
                proofs=[e for e in r.get('restore_proofs',[]) if e.get('payload',{}).get('request_id','').startswith(rid)]
                if (len(end)!=1 or end[0].get('request_status')!='FINISHED_ABORTED'
                        or len(proofs)!=1 or proofs[0]['payload']!={k:v for k,v in end[0].items() if k!='action'}
                        or proofs[0]['result'].get('bytes_restored') is not True
                        or proofs[0]['result'].get('request_leases')!=0
                        or proofs[0]['result'].get('read_only_no_injection') is not False
                        or r.get('tickets')!=0 or r.get('leases')!=0):issues.append('KV_ABORT_PAGE_RECOVERY_UNPROVEN')
                endings.append(end)
            if [r.get('rank') for r in audit]!=[0,1] or endings[0]!=endings[1]:issues.append('KV_ABORT_RANKS_MISMATCH')
        if (not all(c.get('disable',{}).get('recovery',{}).get(k) is True for k in ('bytes_restored','resources_zero','canary_exact','restored'))
                or c.get('detach',{}).get('status')!='PRODUCT_DETACHED'):issues.append('KV_ABORT_THREE_PART_RECOVERY')
        cancel_checked=not any(i.startswith('KV_') and ('ABORT' in i or 'NEGATIVE' in i or 'AUDIT' in i) for i in issues)
    return dict(schema='dcat.persistent_kv_example.review.v1',passed=not issues,issues=sorted(set(issues)),ranks=details,
                cancellation_and_range_reviewed=cancel_checked,
                qualification_granted=False,limits=['A2_TP2_QWEN_COMMUNITY_W8A8_ONLY','B1_FOUR_TOKENS_16_FULL_ATTENTION_LAYERS',
                'ONE_VALUE_CELL_PER_LAYER_REQUEST','NO_MLA_OR_PREFIX_CACHE']+([] if cancel_checked else ['NO_REAL_CANCEL_ACCEPTANCE']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-negatives',action='store_true')
    a=p.parse_args();raw=a.receipt.read_bytes();r=review(json.loads(raw),require_negatives=a.require_negatives);r['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)
