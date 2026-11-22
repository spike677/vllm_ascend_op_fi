"""Independent receipt-only review; no injection logs, model tensors or grants."""
import hashlib,json
from fault_injection.continuous_contract import review_rank_coverage


def need(ok,reason):
    if not ok:raise ValueError(reason)


def review(report, *, require_negative_creation=False):
    need(report.get('acceptance_scope')=='PUBLIC_WORKER_RUNTIME','PUBLIC_RUNTIME_RECEIPT_REQUIRED')
    scopes=report.get('scopes',[]);need(len(scopes)==2,'TWO_SCOPES_REQUIRED')
    result=[];members=None;owners=[]
    for index,s in enumerate(scopes):
        owners.append(s['owner'])
        prepare=s['prepare'];group=prepare['group']
        need(group['owner']==s['owner'] and group['injection_policy']=='ALL_RANK_EVERY_ELIGIBLE_CALL','GROUP_SCOPE_CHANGED')
        need([m['rank'] for m in group['members']]==list(range(group['tp'])),'ACTUAL_RANK_SET_MISSING')
        if members is None:members=group['members']
        need(members==group['members'],'WORKERS_CHANGED_BETWEEN_SCOPES')
        for verb in ('prepare','register','check','enable','finish','unregister'):
            need(s[verb]['status']=='ACTION_COMPLETED','ACTION_NOT_COMPLETED:'+verb)
        need(s['detach']['status']=='DETACHED' and s['snapshot']['ownership_retained'] is False,'HOST_OWNERSHIP_REMAIN')
        need(s['negative_extra_advance']=='GROUP_PHASES_EXHAUSTED','EXTRA_ADVANCE_NOT_REFUSED')
        need(len(s['negative_missing_permit'])==len(members) and all(r['can_prepare'] is False for r in s['negative_missing_permit']),
             'MISSING_PERMIT_NOT_REFUSED')
        creation=s.get('negative_missing_permit_creation',[])
        if require_negative_creation or creation:
            need([r['rank'] for r in creation]==list(range(len(members))),'NEGATIVE_CREATION_RANKS_MISSING')
            for r in creation:
                need(r['prepare']['status'] in ('PROVIDER_UNAVAILABLE','REJECTED')
                    and r['snapshot']['ticket'] is None and r['snapshot']['ownership_retained'] is False
                    and r['detach']['status']=='DETACHED','UNPERMITTED_CREATION_NOT_REFUSED_OR_CLEANED')
            cleanup=s.get('cleanup',[])
            need([r['rank'] for r in cleanup]==list(range(len(members))) and all(
                r['permit_remaining'] is False and r['runtime_leases']==r['runtime_tenants']==0 for r in cleanup),
                'FINAL_BINDING_PERMIT_OR_RUNTIME_RESOURCES_REMAIN')
        warm=[w for w in report['warmup'] if w['scope']==index]
        need(2<=len(warm)<=6 and len(warm[-1]['ranks'])==len(members),'WARMUP_BUDGET_OR_RANKS')
        for rank in range(len(members)):
            wr=warm[-1]['ranks'][rank];observations=wr['observations']
            need(wr['warmup_valid'] is True and len(observations)>=2,'WARMUP_NOT_CONFIRMED')
            need(all(r['all_finite'] for r in observations[-2:]) and observations[-1]['output']==observations[-2]['output'],
                 'WARMUP_NOT_FINITE_EXACT')
        need([r['phase'] for r in s['requests']]==['normal','fault','off','health'],'PHASE_SEQUENCE')
        ranks={};effects={};actuals={}
        for rank in range(len(members)):
            final=s['requests'][-1]['result']['response']['provider_state']['rank_states'][str(rank)]
            use=final['deployment_acceptance_use'];actuals[rank]=use['actual']
            need(use['scope_valid'] is True and use['registration_source'] in ('BLUE_COMBINATION','YELLOW_DEDICATED_INSTANCE'),
                 'DEPLOYMENT_SCOPE_INVALID')
            need(final['complete'] is True and final['next_phase']=='complete','RANK_NOT_COMPLETE')
            rows=final['evidence']['actual_requests'];need([r['phase'] for r in rows]==['normal','fault','off','health'],'RANK_PHASE_SEQUENCE')
            n,f,o,h=rows
            if report['mechanism']=='quant.batch_shared_scale':
                need(all(row['quant_reference'].get('mode','immediate')==report.get('reference_mode','immediate') for row in rows),
                    'QUANT_REFERENCE_MODE_IDENTITY_MISMATCH')
            ids=[]
            for hostrow,row in zip(s['requests'],rows):
                need(hostrow['result']['status']=='QUERY_OK','REQUEST_QUERY_FAILED')
                actual_ids=[m['internal_request_id'] for m in hostrow['mapping']]
                need(set(row['request_ids'])==set(actual_ids) and len(row['output'])==4,'INTERNAL_ID_OR_FOUR_TOKEN_MAPPING')
                if not (report['mechanism']=='39.output_proxy' and row['phase']=='fault'):
                    need(row['all_finite'] is True and all(x['finite'] is True for x in row['output']),'NONFINITE_OUTPUT')
                need(row['resources']==dict(leases=0,hooks=0,inflight=0),'REQUEST_RESOURCES_REMAIN')
                ids.extend(row['request_ids'])
            need(len(ids)==len(set(ids)),'REQUEST_IDS_REUSED')
            need(n['output']==o['output']==h['output'] and n['output']!=f['output'],'MODEL_EFFECT_OR_EXACT_RECOVERY_MISSING')
            clean=s['unregister']['response']['provider_state']['rank_states'][str(rank)]
            need(all(clean[k] in (False,0) for k in ('leases','inflight','owned_hooks','registered','native_registered','enabled')),
                'FINAL_OWNED_RESOURCES_REMAIN')
            if report['mechanism'] in ('39.output_proxy','39.matmul_range','18.api_half','22.wrong_row','8.restore_offset'):
                calls=f.get('actual_forward_calls',[])
                need(len(calls)==4 and all(calls) and all(isinstance(n,int) and n>0 for c in calls for n in c.values())
                    and sum(sum(c.values()) for c in calls)==f['coverage']['observed']
                    and len(set().union(*(set(c) for c in calls)))==f['site_count'],'SITE_ACTUAL_CALL_COVERAGE_MISMATCH')
            elif report['mechanism']=='17.spatial_padding':
                for row in rows:
                    effect=row.get('vision_effect',{})
                    need(effect.get('encoder_calls')==1 and effect.get('embedding_consumers',0)>=1,
                        'VISION17_REAL_ENCODER_CONSUMER_REQUIRED_EACH_PHASE')
                    need(row.get('restore_kind')=='CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE',
                        'VISION17_ARGUMENT_RESTORE_KIND_REQUIRED')
                    padding=[0,7,7] if row['phase']=='fault' else [0,0,0]
                    actual=effect.get('actual_padding',[])
                    need(len(actual)==row['coverage']['observed'] and bool(actual)
                        and all(p==padding for p in actual),'VISION17_ACTUAL_PADDING_MISMATCH')
                    need(bool(effect.get('consumer_operators')),'VISION17_VALUE_CONSUMER_MISSING')
                need(f['coverage']['observed']==f['site_count']
                    and f['vision_effect']['changed_outputs']==f['coverage']['changed']>0,
                    'VISION17_PATCH_CALL_COVERAGE_MISMATCH')
            else:need(f['coverage']['observed']==4*f['site_count'],'SITE_FORWARD_COVERAGE_MISMATCH')
            ranks[rank]=dict(counts=f['coverage'],observation_complete=f.get('observation_complete',False),
                same_stream_verified=f['same_stream_verified'],bytes_restored=f['bytes_restored'],
                off_exact=True,health_exact=True,uncovered_sites=f.get('uncovered_sites'),resources=f['resources'],
                inapplicable_reasons=f.get('inapplicable_reasons',[]))
            effects[rank]=dict(coverage=f['coverage'],site_count=f['site_count'],quant_reference=f.get('quant_reference',{}),
                hidden_changed=any(a['hidden']!=b['hidden'] for a,b in zip(n['output'],f['output'])),
                logits_changed=any(a['logits']!=b['logits'] for a,b in zip(n['output'],f['output'])),
                recovered=True,events_sha256=f['events_sha256'])
            if report['mechanism']=='8.restore_offset':
                moe=f['moe_effect'];need(moe['actual_producer_joins']==moe['native_combine_calls']==f['coverage']['eligible']
                    and moe['request_rows_verified']>=f['coverage']['eligible'],'MOE_CURRENT_PRODUCER_CONSUMER_MAPPING_MISSING')
                effects[rank]['moe_effect']=moe
            if report['mechanism']=='33.paged_decode':
                effect=f['partition_effect'];need(effect['calls']==f['coverage']['eligible'] and effect['selected_changed_elements']>0,'PAGED_PARTITION_EFFECT_MISSING')
                need(all(p['same_legal_key_set'] and not p['denominator_replay'] and p['nonselected_exact'] and p['key_visits']==p['page_mapping']['logical_keys'] for p in effect['pairs']),'PAGED_KEY_SET_OR_CONTROL_MISMATCH')
                effects[rank]['partition_effect']={k:v for k,v in effect.items() if k!='pairs'}
            if report['mechanism']=='39.matmul_range':
                effect=f['matmul_effect'];need(effect['reference_controls']==f['coverage']['eligible'] and effect['rules']==['omit_final_quarter'],'MATMUL_RANGE_REFERENCE_MISSING')
                effects[rank]['matmul_effect']=effect
            if report['mechanism']=='17.spatial_padding':effects[rank]['vision_effect']=f['vision_effect']
            effects[rank]['projection_effect']=f.get('projection_effect')
            if report['mechanism']=='22.wrong_row':
                rot=f['rotary_effect'];need(rot['rows_changed']>0,'ROTARY_ROWS_UNCHANGED')
                before=n['rotary_effect']['qk_output_fingerprints'];after=rot['qk_output_fingerprints']
                need(len(before)==len(after)==f['coverage']['applied'] and [r['site'] for r in before]==[r['site'] for r in after], 'ROTARY_QK_CALL_ALIGNMENT')
                changed=sum(a['q']!=b['q'] or a['k']!=b['k'] for a,b in zip(before,after))
                need(changed>0,'ROTARY_QK_OUTPUT_UNCHANGED')
                effects[rank]['rotary_effect']=dict(rows_changed=rot['rows_changed'],position_rows_changed=rot['position_rows_changed'],qk_changed_calls=changed)
            effects[rank]['structural_candidates_without_call']=f.get('structural_candidates_without_call',[])
            effects[rank]['fault_all_finite']=f['all_finite']
            quant=f.get('quant_reference',{});sample_count=quant.get('sampled_elements',0)
            effects[rank]['sampled_q_difference_fraction']=(quant.get('q_sample_differences',0)/sample_count if sample_count else None)
        coverage=review_rank_coverage(range(len(members)),ranks)
        need(coverage['passed'],'COVERAGE_REVIEW_FAILED:'+str(coverage['failures']))
        result.append(dict(owner=s['owner'],rank_effects=effects,deployment=actuals,fault_time_ratio=s['fault_time_ratio'],
            request_plan=s['requests'][-1]['result']['response']['provider_state']['request_plan'],
            recovered=True,resources_zero=True))
    need(owners[0]!=owners[1],'INDEPENDENT_OWNER_REQUIRED')
    return dict(schema='demoncat.continuous.public_review.v1',mechanism=report['mechanism'],
        arithmetic_device=report.get('arithmetic_device'),reference_mode=report.get('reference_mode','immediate'),passed=True,scopes=result,
        negative_creation_verified=all(bool(s.get('negative_missing_permit_creation')) for s in scopes),
        acceptance_scope='PUBLIC_RUNTIME_SOURCE_REVIEW_STILL_REQUIRED',qualification_granted=False)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--output')
    p.add_argument('--require-negative-creation',action='store_true');a=p.parse_args()
    raw=Path(a.receipt).read_bytes();result=review(json.loads(raw),require_negative_creation=a.require_negative_creation)
    result['source_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    result['reviewer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if len(text.encode('utf-8'))>60000:raise ValueError('TEXT_RETURN_LIMIT_EXCEEDED')
    if a.output:Path(a.output).write_text(text)
    else:print(text)
