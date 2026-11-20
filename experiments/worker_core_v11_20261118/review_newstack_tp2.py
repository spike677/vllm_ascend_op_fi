"""Offline review of warmed 0.26 ordinary TP2 receipts, not probe success logs."""
import argparse,hashlib,json
from pathlib import Path
PHASES=['normal','fault','off','health']
def need(ok,why):
    if not ok:raise ValueError(why)
def review(report,hosts):
    need(report['mechanism'] in ('18.api_half','22.wrong_row'),'MECHANISM')
    need(report.get('validation_scope')=='WARMED_WORKERS_ONLY_COLD_FIRST_REQUEST_NOT_QUALIFIED','WARMED_SCOPE')
    controls=report.get('preflight_controls',[]);need(len(controls)==3,'THREE_NO_FAULT_CONTROLS')
    for rank in range(2):
        need(controls[1]['rank_rows'][rank]['row']['output']==controls[2]['rank_rows'][rank]['row']['output'],'WARMED_BASELINE_NOT_EXACT')
        for c in controls:
            need(c['rank_rows'][rank]['fault_calls']==0,'CONTROL_INJECTED')
            need(all(c['cleanup'][rank][k]==0 for k in ('owned_hooks','owned_wrappers','leases','inflight')),'CONTROL_RESOURCES')
    scopes=report['scopes'];need(len(scopes)==len(hosts)==2,'TWO_SCOPES_REQUIRED')
    need(scopes[0]['group']['members']==scopes[1]['group']['members'] and scopes[0]['group']['owner']!=scopes[1]['group']['owner'],'SAME_WORKERS_DISTINCT_OWNERS')
    result=[];allids=[]
    for scope,host in zip(scopes,hosts):
        need(host['group']==scope['group'] and not host['ownership_retained'],'GROUP_OWNERSHIP_NOT_RELEASED')
        need(not host['failure'],'HOST_FAILURE')
        need([r['phase'] for r in scope['requests']]==PHASES,'FOUR_COMPLETED_HOST_REQUESTS')
        ids=[r['mapping']['internal_request_id'] for r in scope['requests']];allids+=ids
        states=scope['final_states'];need(set(states)=={'0','1'},'TWO_FINAL_RANKS')
        selected=host['role_decision']['selected_rank'];need(selected in (0,1),'AUTOMATIC_RANK')
        if report['mechanism']=='18.api_half':need(selected==0,'REPLICATED_RMS_DEFAULT_RANK')
        ranks=[]
        for rank in ('0','1'):
            state=states[rank];serving=state['serving'];rows=serving['actual_requests']
            need([r['phase'] for r in rows]==PHASES and [r['request_id'] for r in rows]==ids,'RANK_REQUEST_MAPPING')
            n,f,o,h=rows
            need(n['input']==f['input']==o['input']==h['input'],'COMPLETE_INPUTS_CHANGED')
            need(n['output']==o['output']==h['output'],'OFF_HEALTH_NOT_EXACT')
            need(serving['complete'] and serving['recovered'] and not serving['failure'],'RANK_NOT_COMPLETE')
            for row in rows:
                trace=row['request_trace'];need(trace['status']=='OBSERVED' and trace['visual_calls']==0 and trace['visual_execution_excluded'] and trace['model_input_association_verified'],'ACTUAL_TEXT_REQUEST_UNPROVEN')
                need(row['device_input_bytes_equality']=='COMPLETE_DEVICE_READBACK_VERIFIED','DEVICE_INPUT_BYTES_UNPROVEN')
            events=f['mechanism']['events'];need(len(events)==1,'ONE_SELECTED_CALL')
            e=events[0];injector=int(rank)==selected
            need(bool(e['applied'])==injector and e['return_readback_exact'] and e['consumers'],'TARGET_OR_CONSUMER_UNPROVEN')
            need(e['actual']==n['mechanism']['events'][0]['actual'],'LOCAL_INPUT_WEIGHT_CHANGED')
            if injector:need(e['output_changed'],'NO_LOCAL_EFFECT')
            else:need(f['guard_boundary_unchanged'] and not e['output_changed'],'GUARD_WROTE')
            if report['mechanism']=='22.wrong_row':
                a=e['fused_reader'];b=n['mechanism']['events'][0]['fused_reader']
                need(a['qkv_sha256']==b['qkv_sha256'],'FUSED_INPUT_CHANGED')
                need(all(a[k]==b[k] for k in ('fused_q_non_target_sha256','fused_k_non_target_sha256')),'OTHER_QK_ROWS_CHANGED')
                need(any(a[k]!=b[k] for k in ('fused_q_last_sha256','fused_k_last_sha256'))==injector,'QK_ROW_EFFECT_UNPROVEN')
                need(a['changed_cache_elements']>0 if injector else a['changed_cache_elements']==0,'CACHE_EFFECT_UNPROVEN')
                need(a['original_positions_unchanged'] and a['original_cache_unchanged'] and a['value_and_gate_preserved'],'SOURCE_OR_V_GATE_CHANGED')
            finished=[v for v in host['ranks'][rank]['history'] if v['action']=='finish']
            need(len(finished)==1,'FINISH_RECEIPT_REQUIRED');clean=finished[0]['response']['provider_state']
            need(all(clean[k]==0 for k in ('owned_hooks','owned_wrappers','leases','inflight')) and not clean['replay_pending'],'OWNED_RESOURCES_REMAIN')
            need(n['output']['hidden']!=f['output']['hidden'],'RANK_PROPAGATION_NOT_OBSERVED')
            ranks.append(dict(rank=int(rank),injector=injector,path=e['path'],local_effect=bool(e['output_changed']),hidden_changed=True,
                logits_changed=n['output']['logits']!=f['output']['logits'],off_health_exact=True,owned_resources_zero=True,
                row_effect=e.get('effect'),normal=n['output'],fault=f['output']))
        result.append(dict(index=scope['index'],ranks=ranks,generated_tokens_changed=sum(a['output']['tokens']!=scope['requests'][0]['output']['tokens'] for a in scope['requests'][1:2])))
    need(len(set(allids))==8,'FRESH_REQUEST_IDS_REQUIRED')
    return dict(status='TWO_WARMED_TP2_SCOPES_INDEPENDENTLY_REVIEWED',mechanism=report['mechanism'],scopes=result,
        qualification_granted=False,scope='A2 / 0.26 installed c9c9432b / CANN9.1 / Qwen3.8 community W8A8 / TP2; warmed workers only',
        limits=['Cold first request differs in no-fault controls; root cause not uniquely established','No A3 or yellow artifact qualification','Cross-rank hidden/logits propagation, no unique collective attribution'])
def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=a.directory
    files=[r/'result.json',r/'scope0_host.json',r/'scope1_host.json']
    try:
        output=review(json.loads(files[0].read_bytes()),[json.loads(x.read_bytes()) for x in files[1:]])
    except Exception as e:output=dict(status='REJECTED',error=repr(e),qualification_granted=False)
    output['receipts']=[dict(file=x.name,bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in files if x.exists()]
    (r/'independent-review.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output));return 0 if output['status']!='REJECTED' else 2
if __name__=='__main__':raise SystemExit(main())
