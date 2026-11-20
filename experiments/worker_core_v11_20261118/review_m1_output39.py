"""Independent actual TP2 native return/guard/recovery reader."""
import argparse,hashlib,json
from pathlib import Path

def need(v,why):
    if not v:raise ValueError(why)

def lease_check(v):
    need(v['closed'] and v['first_failure'] is None,'LEASE_UNCLEAN')
    need([r['operation'] for r in v['events']]==['open','apply','restore','release','live'],'NATIVE_LIFECYCLE')
    need(all(r['return_code']==0 and r['returned'] and r['error'] is None for r in v['events']),'NATIVE_ERROR')
    need(v['live_readbacks'][-1]['native_live'] is False,'LIVE_LEASE')

def review(d,hosts):
    need(d['mechanism']=='39.output_proxy' and len(d['scopes'])==len(hosts)==2,'EXACT_SCOPE')
    a,b=d['scopes'];need(a['group']['members']==b['group']['members'] and a['group']['owner']!=b['group']['owner'],'SAME_WORKERS_TWO_OWNERS')
    results=[];seen=set()
    for s,h in zip(d['scopes'],hosts):
        need(h['group']==s['group'] and not h['ownership_retained'] and not s['ownership_retained'],'RELEASED_GROUP')
        finishes=[r for r in h['history'] if r['action']=='finish'];need(len(finishes)==1 and finishes[0]['status']=='ACTION_COMPLETED','ACTUAL_FINISH')
        clean=finishes[0]['response']['provider_state']['rank_states'];states=s['final_states'];need(set(states)=={'0','1'},'BOTH_RANKS')
        selected=states['0']['multi_rank']['selected_rank'];need(selected in (0,1),'AUTOMATIC_RANK')
        rowsum={};ids=[]
        for rank,state in states.items():
            rows=state['evidence']['actual_requests'];need([r['phase'] for r in rows]==['normal','fault','off','health'],'FOUR_PHASES')
            n,f,o,hrow=rows;ids.append([r['actual_internal_request_id'] for r in rows]);need(all(r['input_sha256']==n['input_sha256'] for r in rows),'MATCHED_INPUT')
            need(n['hidden_logits_sha256']==o['hidden_logits_sha256']==hrow['hidden_logits_sha256'] and n['model_output_sha256']==o['model_output_sha256']==hrow['model_output_sha256'],'NUMERIC_RECOVERY')
            normal=n['report'];e=f['report'];off=o['report'];need(hrow['report'] is None and normal['applied']==off['applied']==0,'HEALTH_OR_OFF_INJECTED')
            for r in (normal,e,off):need(r['recovered_storage_exact'] and r['consumers'],'RETURN_CONSUMER_AND_RESTORE')
            for key in ('module_path','module_object','operator','ordinal'):
                need(normal['target'][key]==e['target'][key]==off['target'][key],'TARGET_CHANGED')
            need(normal['selected_return_sha256']==e['selected_return_sha256']==off['selected_return_sha256'],'SELECTED_RETURN_NOT_RESTORED')
            lease_check(normal['native_lease_receipt']);injector=rank==str(selected)
            proof=state['multi_rank'];need(proof['selected_rank']==selected and proof['role']==('injector' if injector else 'guard'),'ROLE_DISAGREEMENT')
            need(e['applied']==int(injector),'INJECTOR_GUARD_SEPARATION')
            if injector:
                lease_check(e['native_lease_receipt']);effect=e['effect']
                need(effect['changed_only_selected'] and effect['changed_elements']==effect['finite_to_positive_inf']==1 and effect['before_sha256']==normal['selected_return_sha256'],'NATIVE_EFFECT')
            else:need(e['guard_boundary_unchanged'] and e['effect'] is None and e['native_lease_receipt'] is None,'GUARD_WROTE')
            need(n['hidden_logits_sha256']!=f['hidden_logits_sha256'],'RANK_PROPAGATION')
            need(all(clean[rank][k]==0 for k in ('owned_hooks','leases','inflight','replay_pending')),'RANK_RESOURCE_REMAINS')
            rowsum[rank]=dict(path=e['target']['module_path'],applied=injector,guard_boundary_unchanged=not injector,combined_hidden_logits_changed=True,logits_changed=n['model_output_sha256']!=f['model_output_sha256'],off_health_exact=True,resources_zero=True)
        need(ids[0]==ids[1] and len(set(ids[0]))==4 and not seen.intersection(ids[0]),'REQUEST_ALIGNMENT');seen.update(ids[0])
        tokens=[r['output']['tokens'] for r in s['requests']];need(tokens[0]==tokens[2]==tokens[3],'TOKEN_RECOVERY')
        results.append(dict(ranks=rowsum,token_changed=tokens[0]!=tokens[1]))
    return dict(status='TP2_OUTPUT39_TWO_SCOPES_INDEPENDENTLY_REVIEWED',source_manifest_sha256=d['source_manifest_sha256'],combination='BLUE_COMBINATION A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B BF16/TP2',scopes=results,request_count=8,rank_request_observations=16,scope_count=2,qualification_granted=False,limits=['One selected output proxy, not matmul_range OOB.', 'Combined full hidden/logits and separate logits hashes; no separate hidden-only digest retained.', 'No unique collective-boundary attribution.', 'Only actual use_all_gather=True path accepted.'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    v=review(json.loads(a.input.read_bytes()),[json.loads((a.input.parent/('scope'+str(i)+'_host.json')).read_bytes()) for i in range(2)])
    v.update(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.out.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(v['status'])
