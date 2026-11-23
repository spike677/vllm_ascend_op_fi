"""Independent review of raw product-facade receipts; grants no deployment."""
import argparse
import hashlib
import json
from pathlib import Path


def review_batch_requests(requests,need,expected_tp=2):
    """Rebuild unique rank/frame counts, never add per-request participation."""
    from fault_injection.product_contract import COUNTERS
    frames={};members={};effects=[];sizes=set();negative=False
    need(len(requests)>=4,'BATCH_PRODUCT_REQUEST_MATRIX_MISSING')
    for row in requests:
        result=row['result'];ranks=sorted(result['ranks'],key=lambda r:r['rank'])
        need([r['rank'] for r in ranks]==list(range(expected_tp)),'EXACT_REQUESTED_RANK_SET_REQUIRED')
        need(all(r['prompt_sha256']==ranks[0]['prompt_sha256'] for r in ranks),'RANK_USER_INPUTS_DIFFER')
        for r in ranks:
            need(r['request_id']==result['request_id'] and r['completed'],'REQUEST_ASSOCIATION_INCOMPLETE')
            need(r['coverage_unit']=='REQUEST_PARTICIPATION_IN_ACTUAL_BATCH_CALLS','BATCH_COVERAGE_UNIT_UNDEFINED')
            need(r['resources']==dict(leases=0,hooks=0,inflight=0) and r['bytes_restored'],'REQUEST_RESOURCES_REMAIN')
            sums=dict.fromkeys(COUNTERS,0)
            for f in r['frames']:
                c=f['coverage'];ids=f['request_ids'];scheduled=f['scheduled_counts']
                need(r['request_id'] in ids and set(ids)==set(scheduled),'BATCH_FRAME_REQUEST_MAPPING')
                need(c['failed']==0 and 0<=c['changed']<=c['applied']<=c['observed'],'BATCH_FRAME_COUNTER_INVALID')
                if f['eligible']:
                    need(2<=len(ids)<=8 and all(v==1 for v in scheduled.values()) and f['reason'] is None,'BATCH_NOT_REAL_MULTI_REQUEST_DECODE')
                    need(c['eligible']>0 and c['eligible']==c['attempted']==c['applied']==c['consumed']==c['restored'],'ELIGIBLE_COVERAGE_INCOMPLETE')
                    sizes.add(len(ids))
                else:need(c['eligible']==c['attempted']==c['applied']==c['consumed']==c['restored']==0 and f['reason'],'INELIGIBLE_BATCH_WAS_INJECTED')
                for k in COUNTERS:sums[k]+=c[k]
                key=(r['rank'],f['index'])
                need(key not in frames or frames[key]==f,'SHARED_BATCH_FRAME_CONTRADICTORY')
                frames[key]=f;members.setdefault(key,set()).add(r['request_id'])
            need(sums==r['coverage'],'REQUEST_BATCH_COUNTER_SUM_MISMATCH')
            for site in r['sites']:
                need(site['eligible']==site['consumed']==site['restored']==site['reference_passed'],'QMM_REFERENCE_OR_CONSUMER_PROOF_MISSING')
            need(sum(t['eligible'] for t in r['sites'])==sums['eligible'],'SITE_BATCH_COUNTER_SUM_MISMATCH')
            if row['expected_injection']:
                need(r['status']=='OBSERVED' and not r['reasons'] and sums['eligible']>0 and r['observed_model_steps']==4,'USER_CONCURRENT_REQUEST_NOT_INJECTED')
            else:
                need(r['status']=='NOT_INJECTED' and r['reasons'] and sums['applied']==0,'RANGE_NEGATIVE_FAILED')
                if row['scenario']=='single':
                    need('DECODE_BATCH_SIZE_ONE' in r['reasons'],'B1_SKIP_REASON_MISSING');negative=True
            effects.append(dict(rank=r['rank'],request_id=r['request_id'],coverage=sums,reasons=r['reasons']))
    totals={r:dict.fromkeys(COUNTERS,0) for r in range(expected_tp)}
    for key,f in frames.items():
        need(members[key]==set(f['request_ids']),'BATCH_FRAME_PARTICIPANTS_MISSING')
        for peer in range(expected_tp):
            other=frames.get((peer,key[1]))
            need(other is not None and all(other[k]==f[k] for k in ('request_ids','scheduled_counts','eligible','reason')),'RANK_BATCH_MAPPING_DIFFERENT')
        for k in COUNTERS:totals[key[0]][k]+=f['coverage'][k]
    need(sizes and negative,'REAL_BATCH_AND_B1_NEGATIVE_REQUIRED')
    return dict(request_participation=effects,unique_batch_calls_by_rank=totals,actual_eligible_batch_sizes=sorted(sizes))


def review(data, *, expected_tp=2, hardware_scope="A2/0.26/Qwen3.8-community-W8A8/TP2"):
    if type(expected_tp) is not int or expected_tp not in (1,2):raise ValueError("PRODUCT_REVIEW_TP_UNSUPPORTED")
    def need(value, code):
        if not value: raise ValueError(code)
    need(data.get('status') == 'HTTP_PRODUCT_TWO_SCOPES_PENDING_REVIEW', 'PRODUCT_RUN_INCOMPLETE')
    need(data.get('entry')=='VLLM_HTTP_API' and data.get('host_per_request_hooks') is False, 'WORKER_AUTOMATIC_HTTP_EVIDENCE_REQUIRED')
    scopes = data.get('scopes', [])
    need(len(scopes) == 2, 'TWO_SCOPES_REQUIRED')
    workers = None
    previous_owners = set()
    summaries = []
    for scope in scopes:
        requests = scope['requests']
        need(len(requests) >= 4, 'THREE_USER_REQUESTS_AND_NEGATIVE_REQUIRED')
        effects = []
        hashes = []
        batch_report=review_batch_requests(requests,need,expected_tp) if data['mechanism']=='quant.batch_shared_scale' else None
        for index, row in ([] if batch_report else enumerate(requests)):
            result = row['result']
            ranks = sorted(result['ranks'], key=lambda r:r['rank'])
            need([r['rank'] for r in ranks] == list(range(expected_tp)), 'EXACT_TP2_RANK_SET_REQUIRED')
            for rank in ranks:
                need(rank['request_id'] == result['request_id'] and rank['completed'], 'REQUEST_ASSOCIATION_INCOMPLETE')
                c = rank['coverage']
                need(c['failed']==0 and rank['bytes_restored'], 'HIT_RECOVERY_FAILED')
                need(rank['resources']==dict(leases=0,hooks=0,inflight=0), 'REQUEST_RESOURCES_REMAIN')
                if index < 3:
                    need(rank['status']=='OBSERVED' and not rank['reasons'], 'USER_REQUEST_SKIPPED')
                    need(c['eligible']>0 and c['eligible']==c['attempted']==c['applied']==c['consumed']==c['restored'], 'ELIGIBLE_COVERAGE_INCOMPLETE')
                    need(rank['observed_model_steps']==4, 'PREFILL_AND_DECODE_REQUIRED')
                    need(rank['prompt_sha256'] is not None, 'ACTUAL_USER_INPUT_IDENTITY_REQUIRED')
                    need(all(not s['eligible'] or s['consumer_operator'] and s['consumed'] and s['restored']
                             and s['before_sha256'] is not None and s['before_sha256']==s['restored_sha256'] for s in rank['sites']),
                         'ACTUAL_CONSUMER_OR_BYTE_PROOF_MISSING')
                else:
                    need(rank['status']=='NOT_INJECTED' and c['applied']==0 and rank['reasons'], 'RANGE_NEGATIVE_FAILED')
                if index<3 and data['mechanism']=='39.matmul_range':
                    for site in rank['sites']:
                        if site['eligible']:
                            need(site.get('full_range_control_passed') is True and site.get('non_target_exact') is True and site.get('range_reference',{}).get('rule')=='omit_final_quarter','PRODUCT_MATMUL_CONTROL_MISSING')
                if index<3 and data['mechanism']=='33.paged_decode':
                    for site in rank['sites']:
                        if site['eligible']:
                            ref=site.get('partition_reference',{});agreement=ref.get('control_agreement',{})
                            need(agreement.get('elements',0)>0 and agreement.get('bitwise_equal',0)/agreement['elements']>=.99 and ref.get('same_legal_key_set') is True and ref.get('denominator_replay') is False and ref.get('replacement_policy')=='WHOLE_TARGET_ROW_NO_VALUE_SELECTION','PRODUCT_PAGED_CONTROL_MISSING')
                sites=rank['sites']
                nonfinite={field:(sum(s[field] for s in sites) if sites and all(type(s.get(field)) is int for s in sites) else None)
                           for field in ('nonfinite_before','nonfinite_after')}
                effects.append(dict(rank=rank['rank'], request_index=index, coverage=c,
                                    changed_elements=rank['changed_elements'], reasons=rank['reasons'],
                                    consumers=sorted({s['consumer_operator'] for s in sites if s.get('consumer_operator')}),
                                    **nonfinite))
            if index < 3:
                need(all(r['prompt_sha256']==ranks[0]['prompt_sha256'] for r in ranks), 'RANK_USER_INPUTS_DIFFER')
                hashes.append(ranks[0]['prompt_sha256'])
        if batch_report is None:need(len(set(hashes))==3, 'DISTINCT_USER_PROMPTS_REQUIRED')
        stopped = scope['disable']
        before, after = stopped['canary']['before'], stopped['canary']['after']
        need([r['rank'] for r in before]==list(range(expected_tp)) and [r['rank'] for r in after]==list(range(expected_tp)), 'CANARY_ALL_RANK_REQUIRED')
        for b,a in zip(before,after):
            if data['mechanism']=='quant.batch_shared_scale':
                from demoncat.runtime.vllm_ascend.worker_extension.batch_canary import stable
                need(stable(b['observations'],3) and stable(a['observations'],2,b['observations'][-1]),'BATCH_SERIAL_CANARY_3_BEFORE_2_AFTER_REQUIRED')
            bs, az = b['observations'], a['observations'][-1]
            need(len(bs)>=2 and all(x['all_finite'] for x in bs[-2:]), 'FINITE_STABLE_CANARY_REQUIRED')
            need(bs[-1]['output']==bs[-2]['output']==az['output'] and az['all_finite'], 'CANARY_FULL_OUTPUT_NOT_EXACT')
        for action in ('disable','unregister','detach'):
            replies = stopped['cleanup'][action]
            need(len(replies)==expected_tp and all(x['status']==('DETACHED' if action=='detach' else 'ACTION_COMPLETED') for x in replies), 'RUNTIME_RELEASE_FAILED')
        for released, detached in zip(stopped['cleanup']['unregister'], stopped['cleanup']['detach']):
            state = released.get('response', {}).get('provider_state', {})
            need(all(type(state.get(k)) is int and state[k] == 0 for k in ('leases','owned_hooks','inflight'))
                 and state.get('registered') is False and state.get('replay_pending') is False
                 and state.get('recovery', {}).get('resources_zero') is True
                 and detached.get('ownership_retained') is False, 'RUNTIME_RESOURCE_COUNTERS_NOT_ZERO')
        identities=[x['worker'] for x in stopped['cleanup']['unregister']]
        owners={x['response']['provider_state'].get('owner') for x in stopped['cleanup']['unregister']}
        need(None not in owners and not previous_owners.intersection(owners), 'SCOPES_REUSE_RUNTIME_OWNER')
        previous_owners.update(owners)
        if workers is None: workers=identities
        need(workers==identities, 'WORKERS_CHANGED_BETWEEN_SCOPES')
        need(all(r['leases']==r['owners']==0 for r in scope['cleanup']), 'OWNER_OR_LEASE_REMAINS')
        need(stopped['recovery']==dict(bytes_restored=True,resources_zero=True,canary_exact=True,restored=True), 'RESTORATION_SUMMARY_CONTRADICTORY')
        need(scope['detach']['status']=='PRODUCT_DETACHED', 'BINDING_NOT_DETACHED')
        summaries.append(dict(scope=scope['scope'], effects=effects, batch=batch_report, canary_exact=True, all_resources_zero=True))
    if data['mechanism']=='17.spatial_padding':
        need(len(set(data.get('image_pixel_sha256', [])))>=3, 'CANARY_PLUS_TWO_USER_IMAGES_REQUIRED')
    return dict(status='PRODUCT_BLUE_REVIEW_PASSED', mechanism=data['mechanism'],
                source_manifest_sha256=data['source_manifest_sha256'], scopes=summaries,
                product_entry=data.get('product_entry','ProductClient direct constructor'),
                between_request_cache_reset=data.get('between_request_cache_reset'),
                deployment_config_sha256=data.get('deployment_config_sha256'),
                hardware_scope=hardware_scope,
                yellow_main_project_integration_verified=False, product_closed=False, qualification_granted=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('receipt');parser.add_argument('--output')
    parser.add_argument('--tp',type=int,choices=(1,2),default=2)
    args=parser.parse_args();raw=Path(args.receipt).read_bytes()
    result=review(json.loads(raw),expected_tp=args.tp,hardware_scope="CALLER_DECLARED_TP"+str(args.tp)+"_NOT_IDENTITY_ATTESTATION");result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    result['reviewer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    else:print(text)


if __name__=='__main__':main()
