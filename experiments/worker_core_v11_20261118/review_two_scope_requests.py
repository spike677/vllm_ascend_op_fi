"""Offline reader for actual ordinary 18/39 request records, no execution logs.

Checks raw request/effect/native receipts instead of copying top-level PASS.
Does not certify the main project, another deployment, or all model targets.
"""
import argparse,hashlib,json
from pathlib import Path

def need(v,why):
    if not v:raise ValueError(why)

def review(data,mechanism):
    scopes=data['scopes'];need(len(scopes)==2,'EXACT_TWO_SCOPES')
    need(scopes[0]['worker']==scopes[1]['worker'],'WORKER_CHANGED')
    worker=scopes[0]['worker']
    need(all(worker.get(k) for k in ('model_instance_id','worker_start_identity','process_start_identity','pid')),'WORKER_IDENTITY_MISSING')
    need(worker.get('rank')==0 and worker.get('vllm_ascend_version')=='0.18.0','READER_COMBINATION_UNSUPPORTED')
    need(scopes[0]['owner']!=scopes[1]['owner'],'OWNER_REUSED')
    results=[];allids=set()
    for s in scopes:
        records=s['requests'];need(len(records)==4,'FOUR_REAL_REQUESTS')
        rows=[r['observation'] for r in records]
        need([r['phase'] for r in rows]==['normal','fault','off','health'],'PHASE_ORDER')
        ids=[r.get('request_id',r.get('actual_internal_request_id')) for r in rows]
        need(len(set(ids))==4 and not allids.intersection(ids),'REQUEST_ID_REUSED');allids.update(ids)
        for record,rid in zip(records,ids):
            need(record['completed'] and record['mapping']['internal_request_id']==rid,'REQUEST_COMPLETION_MAPPING')
        n,f,o,h=rows;need(all(r['input_sha256']==n['input_sha256'] for r in rows),'MATCHED_INPUTS')
        if mechanism=='18.api_half':
            need(n['output']==o['output']==h['output'],'NUMERIC_RECOVERY')
            normal=n['mechanism']['events'];fault=f['mechanism']['events']
            need(len(normal)==len(fault)==1,'ONE_TARGET')
            b,e=normal[0],fault[0]
            need(not b['applied'] and e['applied'] and e['gain']==.5,'RMS_HALF_WRITE')
            need(e['actual']==b['actual'] and e['path']==b['path'],'ACTUAL_TARGET_CHANGED')
            need(e['original_execution_verified'] and e['return_readback_exact'] and e['output_changed'] and e['consumers'],'RMS_RETURN_AND_CONSUMER')
            need(o['mechanism'] is None and h['mechanism'] is None,'OFF_OBSERVER_REMAINS')
            effect=e['effect']['values'];need(effect['changed_elements']>0 and effect['before_nonfinite']==effect['after_nonfinite']==0,'FINITE_EFFECT')
            evidence=dict(selected_path=e['path'],changed_elements=effect['changed_elements'],max_abs_diff=effect['max_abs_diff_finite'],hidden_changed=n['output']['hidden']!=f['output']['hidden'],logits_changed=n['output']['logits']!=f['output']['logits'])
        else:
            need(mechanism=='39.output_proxy','READER_SCOPE')
            need(n['hidden_logits_sha256']==o['hidden_logits_sha256']==h['hidden_logits_sha256'],'NUMERIC_RECOVERY')
            need(n['report']['applied']==o['report']['applied']==0 and f['report']['applied']==1 and h['report'] is None,'ONE_NATIVE_WRITE')
            report=f['report'];effect=report['effect']
            need(report['recovered_storage_exact'] and report['consumers'] and effect['changed_only_selected'] and effect['changed_elements']==effect['finite_to_positive_inf']==1,'NATIVE_EFFECT')
            for row in (n,f):
                lease=row['report']['native_lease_receipt']
                need(lease['closed'] and lease['first_failure'] is None,'LEASE_CLEAN')
                need([e['operation'] for e in lease['events']]==['open','apply','restore','release','live'],'NATIVE_LIFECYCLE')
                need(all(e['return_code']==0 and e['returned'] and e['error'] is None for e in lease['events']),'NATIVE_RETURN')
                need(lease['live_readbacks'][-1]['native_live'] is False,'NATIVE_RESOURCE_REMAINS')
            evidence=dict(changed_elements=effect['changed_elements'],finite_to_positive_inf=effect['finite_to_positive_inf'],hidden_logits_changed=n['hidden_logits_sha256']!=f['hidden_logits_sha256'])
        cleanup=s['resource_cleanup'];need(all(cleanup[k]==0 for k in ('owned_hooks','inflight')),'RESOURCE_REMAINS')
        need(cleanup['leases' if mechanism=='18.api_half' else 'native_leases']==0 and s['ownership_retained'] is False,'OWNER_RETAINED')
        output=[r['output'] for r in records]
        need(output[0]['tokens']==output[2]['tokens']==output[3]['tokens'],'TOKEN_RECOVERY')
        need(evidence.get('hidden_logits_changed', evidence.get('hidden_changed') and evidence.get('logits_changed')),'PROPAGATION_MISSING')
        evidence.update(exact_off=True,exact_health=True,resources_zero=True,token_changed=output[0]['tokens']!=output[1]['tokens'],text_changed=output[0]['text']!=output[1]['text'])
        results.append(evidence)
    return dict(status='TWO_SAME_WORKER_SCOPES_INDEPENDENTLY_REVIEWED',mechanism=mechanism,source_manifest_sha256=data['source']['manifest_sha256'],
        combination_source='reviewed case deployment, plus actual worker runtime identity; not inferred from effect',
        combination=dict(hardware='Ascend910B4/A2',cann='8.5.1',vllm='0.18',model='Qwen3-0.6B BF16',tp=1),
        same_worker=True,independent_owners=True,scope_count=2,actual_requests=8,scopes=results,
        review_scope='raw request records, complete output hashes, effect and native lifecycle receipts; not all targets or main project',qualification_granted=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--mechanism',choices=['18.api_half','39.output_proxy'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.input.read_bytes();result=review(json.loads(raw),a.mechanism)
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(),reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['status'])
if __name__=='__main__':main()
