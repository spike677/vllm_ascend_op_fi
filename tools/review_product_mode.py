"""Independent review of raw product-facade receipts; grants no deployment."""
import argparse
import hashlib
import json
from pathlib import Path


def review(data):
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
        for index, row in enumerate(requests):
            result = row['result']
            ranks = sorted(result['ranks'], key=lambda r:r['rank'])
            need([r['rank'] for r in ranks] == [0,1], 'EXACT_TP2_RANK_SET_REQUIRED')
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
                sites=rank['sites']
                nonfinite={field:(sum(s[field] for s in sites) if sites and all(type(s.get(field)) is int for s in sites) else None)
                           for field in ('nonfinite_before','nonfinite_after')}
                effects.append(dict(rank=rank['rank'], request_index=index, coverage=c,
                                    changed_elements=rank['changed_elements'], reasons=rank['reasons'],
                                    consumers=sorted({s['consumer_operator'] for s in sites if s.get('consumer_operator')}),
                                    **nonfinite))
            if index < 3:
                need(ranks[0]['prompt_sha256']==ranks[1]['prompt_sha256'], 'RANK_USER_INPUTS_DIFFER')
                hashes.append(ranks[0]['prompt_sha256'])
        need(len(set(hashes))==3, 'DISTINCT_USER_PROMPTS_REQUIRED')
        stopped = scope['disable']
        before, after = stopped['canary']['before'], stopped['canary']['after']
        need([r['rank'] for r in before]==[0,1] and [r['rank'] for r in after]==[0,1], 'CANARY_ALL_RANK_REQUIRED')
        for b,a in zip(before,after):
            bs, az = b['observations'], a['observations'][-1]
            need(len(bs)>=2 and all(x['all_finite'] for x in bs[-2:]), 'FINITE_STABLE_CANARY_REQUIRED')
            need(bs[-1]['output']==bs[-2]['output']==az['output'] and az['all_finite'], 'CANARY_FULL_OUTPUT_NOT_EXACT')
        for action in ('disable','unregister','detach'):
            replies = stopped['cleanup'][action]
            need(len(replies)==2 and all(x['status']==('DETACHED' if action=='detach' else 'ACTION_COMPLETED') for x in replies), 'RUNTIME_RELEASE_FAILED')
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
        summaries.append(dict(scope=scope['scope'], effects=effects, canary_exact=True, all_resources_zero=True))
    if data['mechanism']=='17.spatial_padding':
        need(len(set(data.get('image_pixel_sha256', [])))>=3, 'CANARY_PLUS_TWO_USER_IMAGES_REQUIRED')
    return dict(status='PRODUCT_BLUE_REVIEW_PASSED', mechanism=data['mechanism'],
                source_manifest_sha256=data['source_manifest_sha256'], scopes=summaries,
                product_entry=data.get('product_entry','ProductClient direct constructor'),
                between_request_cache_reset=data.get('between_request_cache_reset'),
                deployment_config_sha256=data.get('deployment_config_sha256'),
                hardware_scope='A2/0.26/Qwen3.8-community-W8A8/TP2',
                yellow_main_project_integration_verified=False, product_closed=False, qualification_granted=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('receipt');parser.add_argument('--output')
    args=parser.parse_args();raw=Path(args.receipt).read_bytes()
    result=review(json.loads(raw));result['receipt_sha256']=hashlib.sha256(raw).hexdigest()
    result['reviewer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    else:print(text)


if __name__=='__main__':main()
