"""Independent TP2 raw-request checks. No runtime checks/PASS copied as authority."""
import argparse,hashlib,json
from pathlib import Path
from experiments.worker_core_v11_20261118.review_two_scope_requests import need

def review(d,hosts):
    scopes=d['scopes'];need(len(scopes)==2,'TWO_SCOPES');a,b=scopes
    need(a['group']['members']==b['group']['members'] and a['group']['owner']!=b['group']['owner'],'SAME_WORKERS_FRESH_OWNERS')
    results=[];seen=set()
    for index,scope in enumerate(scopes):
        host=hosts[index]
        need(host['group']==scope['group'] and not host['ownership_retained'],'HOST_SCOPE_CHANGED')
        finish=[r for r in host['history'] if r['action']=='finish']
        need(len(finish)==1 and finish[0]['status']=='ACTION_COMPLETED','FINISH_MISSING')
        cleaned=finish[0]['response']['provider_state']['rank_states']
        need(scope['ownership_retained'] is False and scope['last_action']['status']=='DETACHED','OWNER_REMAINS')
        states=scope['final_states'];need(set(states)=={'0','1'},'BOTH_RANKS_REQUIRED');rankresults={};ids=[]
        for rank in ('0','1'):
            state=states[rank];rows=state['serving']['actual_requests'];need([r['phase'] for r in rows]==['normal','fault','off','health'],'FOUR_PHASES')
            n,f,o,h=rows;ids.append([r['request_id'] for r in rows]);need(all(r['input_sha256']==n['input_sha256'] for r in rows),'SAME_INPUT')
            need(n['output']==o['output']==h['output'],'RANK_NUMERIC_RECOVERY')
            e=f['mechanism']['events'];need(len(e)==1,'ONE_ACTUAL_TARGET');e=e[0];bn=n['mechanism']['events'][0]
            need(e['actual']==bn['actual'] and e['path']==bn['path'],'TARGET_CHANGED')
            need(e['original_execution_verified'] and e['return_readback_exact'] and e['consumers'],'ACTUAL_BOUNDARY_REQUIRED')
            need(e['applied']==(rank=='0') and e['output_changed']==(rank=='0'),'INJECTOR_GUARD_SEPARATION')
            if rank=='1':need(f['guard_boundary_unchanged'] is True,'GUARD_CHANGED')
            need(n['output']['hidden']!=f['output']['hidden'] and n['output']['logits']!=f['output']['logits'],'RANK_PROPAGATION_MISSING')
            need(all(cleaned[rank][k]==0 for k in ('owned_hooks','leases','inflight','replay_pending')),'RANK_RESOURCES')
            proof=state['multi_rank'];need(proof['selected_rank']==0 and proof['role']==('injector' if rank=='0' else 'guard'),'ROLE_READBACK')
            rankresults[rank]=dict(path=e['path'],applied=e['applied'],boundary_unchanged=not e['output_changed'],hidden_changed=True,logits_changed=True,off_health_exact=True,resources_zero=True)
        need(ids[0]==ids[1] and len(set(ids[0]))==4 and not seen.intersection(ids[0]),'RANK_REQUEST_MAPPING');seen.update(ids[0])
        outputs=[r['output']['tokens'] for r in scope['requests']];need(outputs[0]==outputs[2]==outputs[3],'TOKENS_RECOVERY')
        results.append(dict(ranks=rankresults,token_changed=outputs[0]!=outputs[1],rank_request_alignment=True))
    return dict(status='TP2_RMS18_TWO_SCOPES_INDEPENDENTLY_REVIEWED',source_manifest_sha256=d['source_manifest_sha256'],combination='BLUE_COMBINATION A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B BF16/TP2',scopes=results,scope_count=2,request_count=8,rank_request_observations=16,collective_boundary_observed=False,limitation='Guard rank hidden/logits propagation observed; unique all-reduce attribution not established. Only Ascend use_all_gather=True path validated.',qualification_granted=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();v=review(json.loads(a.input.read_bytes()),[json.loads((a.input.parent/('scope'+str(i)+'_host.json')).read_bytes()) for i in range(2)]);v.update(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.out.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(v['status'])
