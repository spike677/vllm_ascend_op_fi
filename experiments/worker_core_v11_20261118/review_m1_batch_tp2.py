"""Standalone raw-receipt review; no provider import or driver PASS authority."""
import hashlib,json
from pathlib import Path
PHASES=['normal','reference','lossless','calibration_off','fault','off']
def need(ok,why):
    if not ok:raise ValueError(why)
def review(report):
    scopes=report['scopes'];need(len(scopes)==2,'TWO_SCOPES_REQUIRED')
    need(scopes[0]['owner']!=scopes[1]['owner'],'INDEPENDENT_OWNERS_REQUIRED')
    first_workers=None;seen=set();result=[]
    for scope in scopes:
        states=scope['final_states'];need(set(states)=={'0','1'},'ALL_RANKS_REQUIRED')
        workers={rank:state['multi_rank']['worker'] for rank,state in states.items()}
        need(first_workers is None or first_workers==workers,'WORKERS_RESTARTED');first_workers=workers
        phases=scope['phases'];need([x['phase'] for x in phases]==PHASES,'ACTUAL_BATCH_PHASES')
        ids=[x['ids'] for x in phases]
        for phase in phases:
            need(len(phase['ids'])==len(set(phase['ids']))==8 and not seen.intersection(phase['ids']),'FRESH_B8_IDS')
            seen.update(phase['ids']);need(set(phase['ids'])=={x['internal_request_id'] for x in phase['mapping']},'HOST_MAPPING')
        need(scope['explicit_recover']['next_phase']=='off','UNIFIED_RECOVER_TO_OFF')
        need(not scope['ownership_retained'] and scope['last_action']['status']=='DETACHED','OWNER_REMAINS')
        need(len(scope['removed'])==2 and all(x['status']=='DEPLOYMENT_REMOVED' for x in scope['removed']),'BINDING_REMAINS')
        need(scope['health_before']==scope['health_after'],'EXTERNAL_HEALTH_NOT_EXACT')
        health=scope['health_before']['rank_hashes'];need(len(health)==2,'ALL_RANK_HEALTH')
        for h in health:
            need(h['owned_hooks']==0 and h['complete'],'HEALTH_HOOKS_REMAIN')
            need([x['kind'] for x in h['rows']]==['hidden','logits','hidden','logits'] and all(x['finite'] for x in h['rows']),'FINITE_FULL_HEALTH')
        ranks=[]
        for rank,state in states.items():
            need(state['complete'] and state['next_phase']=='complete','INCOMPLETE')
            need(state['leases']==state['inflight']==0 and not state['replay_pending'] and not state['native_registered'] and not state['enabled'],'RANK_RESOURCES_REMAIN')
            v=state['serving'];need(all(v[k]==0 for k in ('owned_api_patches','owned_hooks','owned_method_patches','owned_normal_readback_bytes','owned_row_modes')),'OWNED_RESOURCES_REMAIN')
            rows=v['rows'];need([x['phase'] for x in rows]==PHASES,'RANK_BATCH_PHASES')
            need([x['decode']['decode']['request_ids'] for x in rows]==ids,'RANK_IDS_UNALIGNED')
            n,f,o=rows[0],rows[4],rows[5];selected=state['multi_rank']['selected_rank']
            need(selected==0 and state['multi_rank']['role']==('injector' if rank=='0' else 'guard'),'AUTO_ROLE_DIFFERS')
            need(n['projection']['hashes']==o['projection']['hashes'],'Q_SCALE_OFF_NOT_EXACT')
            need(all(row['projection']['hashes']['input']==n['projection']['hashes']['input'] for row in rows),'CURRENT_INPUT_CHANGED')
            for row in rows:
                need(row['decode']['hidden_finite'] and row['decode']['logits_finite'] and row['decode']['returned'] and row['decode']['logits_returned'],'NONFINITE_OR_INCOMPLETE_RETURN')
            need(all(n['decode'][k]==o['decode'][k] for k in ('hidden_sha256','logits_sha256')),'OFF_FULL_RETURN_NOT_EXACT')
            hidden=n['decode']['hidden_sha256']!=f['decode']['hidden_sha256'];logits=n['decode']['logits_sha256']!=f['decode']['logits_sha256']
            need(hidden and logits,'RANK_PROPAGATION_NOT_OBSERVED')
            receipts=v['native_receipts'];expected=['reference','lossless','calibration_off']+(['fault'] if rank=='0' else [])
            need([x['phase'] for x in receipts]==expected,'ACTUAL_NATIVE_PHASES')
            changed_q=changed_scale=0
            if rank=='0':
                c=receipts[-1]['completion'];core=c['core'];call=c['call']
                need(c['status']==0 and c['admitted']==1 and core['mode']==3 and core['error']==core['last_error']==0,'NATIVE_FAULT_NOT_COMPLETED')
                need(all(core[k]==1 for k in ('input_before_captured','input_unchanged','publication_exact','published','reference_effect_compared','released')),'NATIVE_EFFECT_OR_RELEASE_UNPROVEN')
                need(core['unsafe_allocation_address']==core['unsafe_allocation_bytes']==0,'UNSAFE_ALLOCATION_REMAINS')
                for name in ('reference','effect'):
                    child=core[name]
                    need(child['prepare_original_calls']==child['execute_original_calls']==1 and child['prepare_status']==child['execute_status']==0,'ACTUAL_CHILD_EXECUTION')
                    need(all(child[k]==1 for k in ('drained','prepared','readback','returned')),'CHILD_NOT_COMPLETE')
                effect=core['effect']['plan'];ref=core['reference']['plan']
                need(call['input']['rows']==ref['input']['rows']==8 and effect['input']['rows']==1,'BATCH_SCALE_REDUCTION_GEOMETRY')
                need(effect['input']['columns']==8*call['input']['columns'] and effect['input']['address']==call['input']['address'],'SAME_INPUT_FLATTENED')
                change=v['fault_effect'];changed_q=change['quant']['changed_elements'];changed_scale=change['scale']['changed_elements']
                need(changed_q==core['changed_quant_bytes']>0 and changed_scale>0 and core['changed_scale_bytes']>0,'NUMERICAL_QUANT_SCALE_EFFECT_MISSING')
                need(all(n['projection']['hashes'][k]!=f['projection']['hashes'][k] for k in ('quant','scale')),'FAULT_HASHES_UNCHANGED')
                need(v['actual_fault_hits']==1,'EXACT_ONE_SELECTED_RANK_HIT')
            else:
                need(f['projection']['hashes']==n['projection']['hashes'] and v['actual_fault_hits']==0,'GUARD_LOCAL_MODIFICATION')
            ranks.append(dict(rank=int(rank),native_hits=int(rank=='0'),changed_quant_elements=changed_q,changed_scale_elements=changed_scale,hidden_changed=hidden,logits_changed=logits,off_health_exact=True,resources_zero=True,selected_site=v['selected_site']['path']))
        result.append(dict(scope=scope['index'],batches=6,requests_per_batch=8,explicit_recover=True,ranks=ranks,
            generated_sequences_changed=sum(x!=y for x,y in zip(phases[0]['tokens'],phases[4]['tokens']))))
    return dict(status='BLUE_TP2_BATCH_TWO_SCOPE_ACCEPTED',combination=report['combination'],source_manifest_sha256=report['source_manifest_sha256'],scopes=result,qualification_granted=False,limits=['A2/CANN8.5.1/0.18/Qwen3-0.6B W8A8 TP2 only','first decode B8, one automatically selected projection/rank','cross-rank hidden/logits observed; no unique all-reduce boundary attribution','no A3, Qwen3.8 or new-stack qualification'])
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    raw=a.receipt.read_bytes();r=review(json.loads(raw));r['receipt']=dict(file=a.receipt.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());a.out.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(r['status'])
if __name__=='__main__':main()
