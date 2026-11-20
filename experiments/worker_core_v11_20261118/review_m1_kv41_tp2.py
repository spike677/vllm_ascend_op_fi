"""Independent receipt review; no provider imports, NPU calls or hit-log authority."""
import hashlib,json
from pathlib import Path
PHASES=['normal','normal_repeat','lossless','fault','off','health']
def require(ok,why):
    if not ok:raise ValueError(why)
def review(report,hosts):
    require(report['mechanism']=='41.tail_column' and report['explicit_group_recovery'] is True,'EXPLICIT_41_RECOVERY')
    scopes=report['scopes'];require(len(scopes)==len(hosts)==2,'TWO_SCOPES')
    require(scopes[0]['group']['members']==scopes[1]['group']['members'] and scopes[0]['group']['owner']!=scopes[1]['group']['owner'],'SAME_WORKERS_DISTINCT_OWNERS')
    summary=[]
    for scope,host in zip(scopes,hosts):
        require(host['group']==scope['group'] and host['ownership_retained'] is False and scope['ownership_retained'] is False,'RELEASED_CURRENT_GROUP')
        actions=host['history'];require([h['action'] for h in actions].count('recover')==1,'ONE_EXPLICIT_GROUP_RECOVER')
        require(actions[-1]['action']=='detach' and actions[-1]['status']=='DETACHED','REAL_DETACH')
        finish=next(h for h in actions if h['action']=='finish')
        require(finish['status']=='ACTION_COMPLETED','FINISH_FAILED')
        final=finish['response']['provider_state']['rank_states']
        states=scope['final_states'];require(set(states)==set(final)=={'0','1'},'TWO_RANKS')
        requests=scope['requests'];require([r['phase'] for r in requests]==PHASES,'SIX_REAL_PHASES')
        selected=host['role_decision']['selected_rank'];require(selected==0,'AUTOMATIC_FIRST_REPLICA_RANK')
        rank_results=[];aligned=[]
        for rank,state in states.items():
            require(state['next_phase']=='complete' and state['complete'],'COMPLETE_STATE')
            require(all(final[rank][k]==0 for k in ('owned_hooks','leases','inflight','replay_pending')),'RESOURCE_REMAINS')
            rows=state['evidence']['actual_requests'];require([r['phase'] for r in rows]==PHASES,'RANK_PHASES')
            ids=[r['request_id'] for r in rows];aligned.append(ids)
            require(ids==[r['mapping']['internal_request_id'] for r in requests],'ACTUAL_REQUEST_MAP')
            normal,fault,off,health=rows[0],rows[3],rows[4],rows[5]
            require(normal['model_output_sha256']==off['model_output_sha256']==health['model_output_sha256'],'RANK_MODEL_RECOVERY')
            effect=fault['effect'];injector=int(rank)==selected
            require(fault['new_fault_hits']==int(injector),'SINGLE_RANK_HIT')
            if injector:
                require(effect['native_applied'] and effect['readback_exact'] and effect['original_calls']==1 and effect['consumer_returned'],'ACTUAL_NATIVE_CONSUMER')
                require(effect['changed_elements']==effect['selected_elements']==8 and effect['restored'] and effect['released'],'EXACT_NATIVE_EFFECT_AND_RELEASE')
                require(effect['boundary']['controlled_legal_address_proxy'] and not effect['boundary']['native_out_of_bounds_claim'],'PROXY_BOUNDARY')
            else:require(effect['before']==effect['after'] and effect['unchanged'] and effect['native_writes']==0,'GUARD_MODIFIED')
            traces=[]
            for request in requests:
                per=request['trace'];require({t['rank'] for t in per}=={0,1} and len(per)==2,'EXACT_TRACE_RANKS')
                trace=next(t['rows'] for t in per if t['rank']==int(rank))
                require([t['kind'] for t in trace]==['hidden','logits','hidden','logits'] and all(t['finite'] for t in trace),'FULL_FINITE_PREFILL_DECODE')
                traces.append(trace)
            require(traces[0]==traces[4]==traces[5],'FULL_TRACE_RECOVERY')
            hidden=traces[0][2]['sha256']!=traces[3][2]['sha256'];logits=traces[0][3]['sha256']!=traces[3][3]['sha256']
            require(hidden and logits,'PROPAGATION_NOT_OBSERVED')
            rank_results.append(dict(rank=int(rank),injector=injector,native_hits=fault['new_fault_hits'],changed_kv_elements=effect['changed_elements'] if injector else 0,hidden_changed=hidden,logits_changed=logits,off_health_exact=True,resources_zero=True))
        require(aligned[0]==aligned[1],'CROSS_RANK_REQUEST_ALIGNMENT')
        summary.append(dict(scope=scope['index'],ranks=rank_results,requests=6,explicit_recover=True))
    return dict(status='BLUE_TP2_TWO_SCOPE_MECHANISM_ACCEPTED',combination=report['combination'],source_manifest_sha256=report['source_manifest_sha256'],scopes=summary,qualification_granted=False,limits=['controlled legal KV tail address proxy; not real out-of-bounds','no direct collective boundary attribution','A2 old 0.18 combination only; no A3 or current-model qualification'])
def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument('folder',type=Path);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    paths=[args.folder/n for n in ('result.json','scope0_host.json','scope1_host.json')]
    raw=[p.read_bytes() for p in paths];values=[json.loads(r) for r in raw]
    result=review(values[0],values[1:]);result['receipts']=[dict(file=p.name,bytes=len(r),sha256=hashlib.sha256(r).hexdigest()) for p,r in zip(paths,raw)]
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
if __name__=='__main__':main()
