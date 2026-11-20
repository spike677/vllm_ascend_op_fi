"""Stdlib-only review of the two original ordinary KV41 Host scopes.

No provider imports, NPU calls or statistics-first-difference claims. Native
readback receipts and semantic writer/cache/consumer joins are reviewed as such.
"""
import argparse
import hashlib
import json
from pathlib import Path

PHASES=['normal','normal_repeat','lossless','fault','off','health']

def review(folder):
    files={name:json.loads((folder/name).read_bytes()) for name in
           ('result.json','scope0-host.json','scope1-host.json','controller.json')}
    result,control=files['result.json'],files['controller.json']
    issues=[];checks=0
    def check(value,reason):
        nonlocal checks
        checks+=1
        if not value:issues.append(reason)
    scopes=result.get('scopes',[])
    check(len(scopes)==2,'TWO_SCOPES_REQUIRED')
    if len(scopes)!=2:return dict(consistent=False,issues=issues,checks=checks)
    check(scopes[0]['worker']==scopes[1]['worker'],'SAME_LIVE_WORKER_AND_MODEL')
    check(scopes[0]['owner']!=scopes[1]['owner'],'INDEPENDENT_RUNTIME_OWNERS')
    c=control['cases'][0]
    check(c['exit_code']==0 and c['image_id']=='sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0','IMAGE_AND_EXIT')
    check(c['gate']['idle'] is True and not c['gate']['blockers'],'ACTUAL_PRECASE_GATE')
    check(c['source_manifests'][0]['manifest_sha256']==result['source_manifest_sha256'],'SOURCE_MANIFEST_JOIN')
    check(c['cleanup']['status']=='OWNED_CONTAINER_REMOVED','OWNED_CONTAINER_CLEANUP_ONLY')
    records=[];native_owners=[];permits=[]
    expected_actions=['attach','prepare','register','check','enable']+['advance','query']*6+['finish','disable','unregister','detach']
    for index,s in enumerate(scopes):
        label='scope'+str(index);h=files[label+'-host.json'];rows=s['requests'];state=s['final_state']
        check([v['phase'] for v in rows]==PHASES,label+':ACTUAL_PHASES')
        check([v['action'] for v in h['history']]==expected_actions,label+':NO_QUERY_COLLECT_OR_HIDDEN_ACTIONS')
        check(h['status']=='DETACHED' and h['ownership_retained'] is False and h['ticket'] is None,label+':HOST_OWNER_RELEASED')
        check(h['owner']==s['owner'] and h['worker']==s['worker'],label+':HOST_IDENTITY_JOIN')
        check(s['binding_removed']['status']=='SERVING_DEPLOYMENT_REMOVED',label+':BINDING_REMOVED')
        queries=[v['response'] for v in h['history'] if v['action']=='query']
        check(len(queries)==6,label+':SIX_QUERY_RECEIPTS')
        if len(rows)!=6 or len(queries)!=6:
            check(False,label+':INCOMPLETE_ROWS');continue
        baseline=rows[0]['worker']['model_output_sha256']
        ids=[]
        for n,(row,q) in enumerate(zip(rows,queries)):
            phase=row['phase'];w=row['worker'];rid=w['request_id'];ids.append(rid)
            check(q['status']=='QUERY_OK' and q['recovery_required'] is False,label+':QUERY_'+phase)
            ev=q['provider_state']['evidence']
            check(len(ev['actual_requests'])==n+1 and ev['actual_requests'][-1]==w,label+':CALLBACK_COMPLETED_'+phase)
            check(w['phase']==phase and rid==row['mapping']['internal_request_id'],label+':ENQUEUE_JOIN_'+phase)
            check(w['model_output_sha256']==baseline if phase!='fault' else w['model_output_sha256']!=baseline,label+':ROOT_OUTPUT_'+phase)
            if phase!='fault':
                check(row['output']['tokens']==result['healthy_before']['tokens'] and row['output']['text']==result['healthy_before']['text'],label+':TOKENS_TEXT_'+phase)
            if phase!='health':check(w['new_fault_hits']==int(phase=='fault'),label+':SINGLE_HIT_'+phase)
            else:check(w['mechanism_hooks_present'] is False and w['input_prefixes_exact'] is True,label+':UNHOOKED_HEALTH')
            audit=q['provider_state']['deployment_acceptance_use']
            check(audit['registration_source']=='BLUE_COMBINATION' and audit['scope_valid'] is True
                  and audit['accepted_record_created'] is False and audit['permit']['source_manifest_sha256']==result['source_manifest_sha256'],label+':PERMIT_'+phase)
        check(len(set(ids))==6,label+':FRESH_REQUEST_IDS')
        effect=rows[3]['worker']['effect'];native_owners.append(effect['owner'])
        check(effect['native_applied'] and effect['readback_exact'] and effect['restored'] and effect['released']
              and effect['consumer_returned'] and effect['original_calls']==1,label+':NATIVE_EFFECT_CONSUMER_RESTORE')
        geometry=effect['boundary'];shape=geometry['cache']['shape'];heads,dim=shape[-2:]
        tails=[h*dim+d for h in range(heads) for d in range(dim-2,dim)]
        check(effect['changed_elements']==effect['selected_elements']==2*heads and geometry['relative_indices']==tails,label+':ACTUAL_TAIL_GEOMETRY')
        check(geometry['span_elements']==heads*dim and geometry['span_element_offset']==geometry['slot']*heads*dim
              and geometry['allocation_element_indices']==[geometry['span_element_offset']+v for v in tails],label+':SLOT_TO_SPAN_JOIN')
        meta=geometry['observational_metadata'];writer=meta['writer']
        check(writer['returned'] and writer['slots']==meta['slots']==[geometry['slot']],label+':ACTUAL_WRITER_SLOT')
        check(meta['value']['storage_identity']==writer['value_cache']['storage_identity']==geometry['cache']['storage_identity']
              and meta['layout']=='TND' and meta['scheduler']['request_ids']==[ids[3]],label+':CACHE_CONSUMER_REQUEST_JOIN')
        check(not geometry['native_out_of_bounds_claim'] and geometry['controlled_legal_address_proxy'],label+':CONTROLLED_MECHANISM_LIMIT')
        lossless=rows[2]['worker']['effect']
        check(lossless['changed_elements']==0 and not lossless['native_applied'] and lossless['restored'] and lossless['released'],label+':LOSSLESS')
        check(state['owned_hooks']==state['leases']==state['inflight']==0 and not state['owned_api_state']['owned_links_present'],label+':RESOURCE_ZERO')
        check(state['evidence']['matched_off_and_health_confirmed'] and not state['evidence']['failures'],label+':HEALTH_COMPLETE_NO_FAILURE')
        check(all(s['healthy_after'][k]==result['healthy_before'][k] for k in ('tokens','text')),label+':POST_UNREGISTER_HEALTH')
        audit=state['deployment_acceptance_use'];permits.append(audit['permit']['permit_id'])
        records.append(dict(scope=index,owner=s['owner'],native_owner=effect['owner'],selected_module=state['evidence']['selection']['module_path'],
            actual_calls_per_request=[v['worker'].get('actual_calls') for v in rows],fault_hits=1,changed_cache_elements=effect['changed_elements'],
            original_selected_consumer_calls=effect['original_calls'],normal_hidden_sha256=baseline,fault_hidden_sha256=rows[3]['worker']['model_output_sha256'],
            off_hidden_sha256=rows[4]['worker']['model_output_sha256'],health_hidden_sha256=rows[5]['worker']['model_output_sha256'],
            generated_tokens={v['phase']:v['output']['tokens'] for v in rows},tokens_changed=rows[3]['output']['tokens']!=rows[0]['output']['tokens'],
            text_changed=rows[3]['output']['text']!=rows[0]['output']['text'],source=audit['registration_source'],permit=audit['permit'],
            deployment=audit['actual'],resources=dict(owned_hooks=state['owned_hooks'],leases=state['leases'],inflight=state['inflight'],api_links=state['owned_api_state']['owned_links_present']),
            limitations=dict(logits='NOT_CAPTURED',msprobe_E5='NOT_ASSESSED',FIA_native_launch_metadata_verified=meta['native_launch_verified'],FIA_tiling_fault_claim=False)))
    check(len(set(native_owners))==2 and len(set(permits))==2,'INDEPENDENT_NATIVE_OWNERS_AND_PERMITS')
    return dict(schema='demoncat.ordinary41.independent_review.v1',consistent=not issues,checks=checks,issues=issues,
        source_commit=result['source_commit'],source_manifest_sha256=result['source_manifest_sha256'],
        native_sha256=result['native_sha256'],build_fingerprint=result['build_fingerprint'],same_worker_two_owners=True,
        scopes=records,originals={name:dict(bytes=(folder/name).stat().st_size,sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest()) for name in files},
        runtime_acceptance_granted=False,originals_modified=False,raw_statistics_submitted=False,
        verdict='ACCEPT_BLUE_DECLARED_ORDINARY_KV41_SCOPE' if not issues else 'FAILED_INDEPENDENT_REVIEW')

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=review(a.folder);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('consistent','checks','issues')},ensure_ascii=False))
    return 0 if result['consistent'] else 2
if __name__=='__main__':raise SystemExit(main())
