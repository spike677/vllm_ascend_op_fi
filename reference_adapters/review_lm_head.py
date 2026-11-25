"""Independent persistent-row review: compare normal/fault first-token logits."""
import argparse,hashlib,json,re
from pathlib import Path
# Independent fixed contract; reviewing receipts must not load torch/NPU or the
# provider implementation merely to obtain the two names and deployment IDs.
VARIANTS=('row12_lm_head_collapse','row12_lm_head_swap')
TOKEN_IDS=(10,11,12)


def review(d):
    issues=[];effects=[];workers_before=None;native_owners=set()
    mode=d.get('mode','').removeprefix('persistent_').removesuffix('_example')
    if mode not in VARIANTS:issues.append('UNKNOWN_VARIANT')
    expected='TWO_OWNER_'+mode.upper()+'_EXAMPLE_COMPLETE_NOT_RELEASE_QUALIFICATION'
    if d.get('status')!=expected:issues.append('RUN_NOT_COMPLETE')
    scopes=d.get('scopes',[])
    if len(scopes)!=2 or len({s.get('owner') for s in scopes})!=2:issues.append('TWO_OWNERS_REQUIRED')
    if d.get('missing_permit',{}).get('exit_code')!=4:issues.append('MISSING_PERMIT_NEGATIVE')
    if d.get('observer_cleanup')!=[dict(rank=r,status='EXAMPLE_OBSERVER_REMOVED') for r in (0,1)]:
        issues.append('OBSERVER_CLEANUP_REQUIRED')
    for s in scopes:
        disabled=s.get('disable',{});recovery=disabled.get('recovery',{})
        if not all(recovery.get(k) is True for k in ('bytes_restored','resources_zero','canary_exact','restored')):
            issues.append('THREE_PART_RECOVERY_REQUIRED')
        for negative in ('range_negative','duplicate_enable'):
            if s.get(negative,{}).get('exit_code')!=4:issues.append('NEGATIVE_REQUIRED:'+negative)
        canary=disabled.get('canary',{});before=canary.get('before',[])
        if not before or before!=canary.get('after'):issues.append('CANARY_NOT_BYTE_EXACT')
        fault=s.get('fault_hidden',[])
        if [r.get('rank') for r in before]!=[0,1] or [r.get('rank') for r in fault]!=[0,1]:
            issues.append('RANK_SET');continue
        effect_ranks=0
        for b,f in zip(before,fault):
            bs=b.get('output',{}).get('logits_observations',[])
            fs=f.get('output',{}).get('logits_observations',[])
            if len(bs)!=4 or len(fs)!=4 or b.get('all_finite') is not True or f.get('all_finite') is not True:
                issues.append('FINITE_FOUR_STEPS');continue
            for row in bs+fs:
                if row.get('tokens')!=list(TOKEN_IDS) or not row.get('weight_readers'):
                    issues.append('WEIGHT_READER_OR_FIXED_TOKENS_MISSING')
                if row.get('target_logits') is not None and row.get('sampler_consumed') is not True:
                    issues.append('ACTUAL_SAMPLER_MISSING')
            if bs[0]['hidden_sha256']!=fs[0]['hidden_sha256']:issues.append('FIRST_HIDDEN_CHANGED')
            base,changed=bs[0].get('target_logits'),fs[0].get('target_logits')
            if base is None:
                if changed is not None:issues.append('DRIVER_RANK_CHANGED')
                continue
            if not base or len(base)!=len(changed or []):issues.append('TARGET_LOGITS_SHAPE');continue
            expected_rows=[[row[0]]*3 if mode.endswith('collapse') else [row[1],row[0],row[2]] for row in base]
            if changed!=expected_rows:issues.append('TARGET_LOGITS_NOT_EXACT_NAMED_RULE')
            if bs[0]['target_probabilities']==fs[0]['target_probabilities']:issues.append('NO_DISTRIBUTION_EFFECT')
            else:effect_ranks+=1
        if effect_ranks==0:issues.append('NO_SAMPLER_DISTRIBUTION_PROOF')
        stopped=disabled.get('cleanup',{}).get('disable',[])
        workers=[r.get('worker') for r in stopped]
        if len(workers)!=2 or [w.get('rank') for w in workers if isinstance(w,dict)]!=[0,1]:issues.append('WORKER_IDENTITIES')
        if workers_before is not None and workers!=workers_before:issues.append('WORKERS_CHANGED')
        workers_before=workers;owners=0;request_ids=[]
        http=s.get('fault_http',{});http_id=http.get('id')
        if (not isinstance(http_id,str) or not http_id or http.get('usage',{}).get('completion_tokens')!=4
                or not http.get('choices') or any(c.get('finish_reason')!='length' for c in http['choices'])):
            issues.append('HTTP_FOUR_TOKEN_COMPLETION_REQUIRED')
        for index,row in enumerate(stopped):
            state=row.get('response',{}).get('provider_state',{});adapter=state.get('row_adapter',{})
            events=state.get('events',[])
            # Runtime intentionally generates a separate random provider owner.
            # Bind it to the group through stable ticket/worker/generation and
            # the enable -> status -> disable records, not string equality.
            ticket=row.get('ticket',{});local_owner=state.get('owner')
            if (ticket.get('owner')!=s.get('owner') or ticket.get('worker')!=row.get('worker')
                    or ticket.get('generation')!=state.get('generation')
                    or not isinstance(local_owner,str) or not local_owner or local_owner in native_owners):
                issues.append('OWNER_MISMATCH')
            native_owners.add(local_owner)
            for phase in ('enable','status'):
                peers=s.get(phase,{}).get('ranks',[])
                if index>=len(peers):issues.append('OWNER_CHAIN_MISSING');continue
                peer=peers[index];ps=peer.get('response',{}).get('provider_state',{})
                if (peer.get('worker')!=row.get('worker') or peer.get('ticket')!=ticket
                        or ps.get('owner')!=local_owner or ps.get('generation')!=state.get('generation')):
                    issues.append('OWNER_CHAIN_MISMATCH')
            readers=state.get('readers',[]);ids={r.get('request_id') for r in readers}
            if len(readers)!=4 or len(ids)!=1 or any(r.get('weight_read') is not True for r in readers):
                issues.append('FOUR_ACTUAL_READERS_REQUIRED')
            else:
                rid=next(iter(ids));request_ids.append(rid)
                # 0.26 API request -> single completion index -> scheduler suffix.
                if not isinstance(rid,str) or not re.fullmatch(re.escape(http_id or '')+r'-0-[0-9a-f]+',rid):
                    issues.append('HTTP_INTERNAL_REQUEST_ASSOCIATION')
            if adapter.get('owns_rows') is True:
                owners+=1
                if len(events)!=1:issues.append('ONE_PERSISTENT_WRITE_LEASE_REQUIRED')
                for e in events:
                    if not all(e.get(k)==1 for k in ('observed','eligible','attempted','applied','consumed','restored')):
                        issues.append('PERSISTENT_WRITE_COUNTS')
                    if e.get('changed')!=1 or e.get('changed_elements',0)<=0 or e.get('failed')!=0:
                        issues.append('PERSISTENT_WRITE_EFFECT')
                    if e.get('restore_kind')!='BYTE_EXACT_SAME_ALLOCATION_GENERATION':issues.append('NOT_SUCCESSFUL_BYTE_RESTORE')
                if state.get('reader_observations')!=4:issues.append('PREFILL_DECODE_READERS')
            elif adapter.get('owns_rows') is False:
                if events or adapter.get('nonowner_guard_reads')!=4:issues.append('NONOWNER_GUARD')
            else:issues.append('OWNERSHIP_NOT_PROVEN')
        if owners!=1:issues.append('EXACTLY_ONE_VOCAB_SHARD_OWNER_REQUIRED')
        if len(request_ids)!=2 or len(set(request_ids))!=1:issues.append('RANK_REQUEST_ASSOCIATION')
        cleanup=s.get('cleanup',[])
        if len(cleanup)!=2 or any(r.get(k)!=0 for r in cleanup for k in ('leases','owners')):issues.append('RESOURCES_REMAIN')
        effects.append(dict(owner=s.get('owner'),sampler_ranks=effect_ranks,weight_owner_ranks=owners))
    return dict(schema='dcat.reference.lm_head.review.v1',passed=not issues,issues=sorted(set(issues)),
        variant=mode,semantics='ORIGINAL_DESCRIBED_WEIGHT_VECTOR_CONSTRUCTION',effects=effects,
        qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.input.read_bytes();r=review(json.loads(raw));r['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps(r))
    raise SystemExit(0 if r['passed'] else 1)
