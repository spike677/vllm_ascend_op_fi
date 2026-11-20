"""Independent stdlib review of two current Worker research scopes.
No torch, DSO load, device, policy evaluation or injector import. Original raw
records and registered kernel bytes are rejoined, not trusted from hit flags.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import review_tnd28_prepared_input as P
import arch22_normal_bridge_evidence as E
PHASES=['observe','observe_repeat','baseline','baseline_repeat','fault','off','health']
need=P.need

def values(row):return [dict(input=f['input'],hidden=f['hidden'],logits=f['logits']) for f in row['forwards']]
def calls(row):return [c for f in row['forwards'] for c in f['calls']]
def scoped_file(run,name):
    path=Path(name);relative=path.relative_to('/evidence/probe')
    return E.child(run,relative.as_posix())

def verify_history(host):
    need(host['protocol']=='EXPLICIT_TWO_STAGE_RESEARCH_NOT_PUBLIC_RUNTIME' and host['ownership_retained'] is False,'HOST_NOT_RELEASED')
    history=host['history'];expected=['prepare','register','check','enable']+['advance','query','collect']*7+['finish','unregister','detach']
    need([h['action'] for h in history]==expected,'RESEARCH_HISTORY_PHASE_ORDER')
    state=history[-1]['response']['provider_state'];rows=state['evidence']['requests']
    need(state['serving']['next_phase']=='complete' and not state['registered'] and not state['binding_reserved']
        and all(state[k]==0 for k in ('owned_hooks','owned_api_wrappers','inflight','leases')),'FINAL_RESOURCES_REMAIN')
    need([r['phase'] for r in rows]==PHASES and len({r['request_id'] for r in rows})==7,'REQUEST_PHASE_OR_ID_REUSED')
    need(E.digest(rows)==state['evidence']['actual_requests_sha256'],'REQUEST_ROWS_HASH_DIFFERS')
    for i,row in enumerate(rows):
        advance,query,collect=history[4+3*i:7+3*i]
        need(advance['request_id']==row['request_id'] and collect['response']['provider_state']['evidence']['requests']==rows[:i+1],
             'ORIGINAL_REQUEST_HISTORY_DISAGREES')
    reference=values(rows[0])
    for i in (1,2,3,5,6):need(values(rows[i])==reference,'COMPLETE_NORMAL_OFF_HEALTH_VALUES_DIFFER:'+PHASES[i])
    need(all(len(calls(row))==28 for row in rows[:6]) and not calls(rows[6]),'ACTUAL_MODEL_CALL_COUNT_OR_HEALTH_HOOKS')
    need(values(rows[4])!=reference,'FAULT_NO_MODEL_EFFECT')
    for left,right in zip(values(rows[4]),reference):need(left['input']==right['input'],'FAULT_ROOT_INPUT_CHANGED')
    target=state['evidence']['selection']['target_key'];selected=[]
    for index in (2,3,4,5):
        hits=[c for c in calls(rows[index]) if c['native'] is not None]
        need(len(hits)==1 and hits[0]['target_key']==target and hits[0]['original_calls']==1 and hits[0]['original_returned'],
             'ACTUAL_SELECTED_CALL_ID_OR_COUNT')
        selected.append(hits[0])
    baseline,repeat,fault,off=selected
    need(all(c['input']==baseline['input'] for c in selected),'ACTUAL_FIA_INPUTS_DIFFER')
    need(repeat['output']==off['output']==baseline['output'] and fault['output']!=baseline['output'],'FIA_EFFECT_OR_RECOVERY')
    need(not state['native_state']['scope_active'] and not state['native_state']['scope_owned']
        and all(state['native_state'].get(k)==0 for k in ('environment_owned','pending_drains','pending_references')),'NATIVE_RESOURCES_REMAIN')
    return rows,selected

def review(a):
    result=P.json_file(a.run/'result.json',128*1024**2)
    need(result['entry_protocol']=='EXPLICIT_TWO_STAGE_RESEARCH_NOT_PUBLIC_RUNTIME' and len(result['scopes'])==2,'TWO_RESEARCH_SCOPES_REQUIRED')
    build,files,metadata,abi=P.bundle(a.build,a.build_sha256,a.vendor,a.source,a.source_manifest_sha256)
    need(E.sha(a.libraries)==a.libraries_sha256,'EXACT_LIBRARY_MANIFEST_REQUIRED')
    libraries=P.json_file(a.libraries);entries={r['recorded_path']:r for r in libraries['files']}
    need(len(entries)==len(libraries['files'])<=32,'LIBRARY_MANIFEST_DUPLICATE_OR_BOUND')
    runtime=P.lines(a.run/'runtime/runtime.jsonl',128*1024**2);summaries=[];owners=[];references=[]
    for index,scope in enumerate(result['scopes']):
        hostpath=a.run/f'scope{index}/host_receipt.json';host=P.json_file(hostpath,1024**3)
        original_bytes=0;original_hashes=[]
        for ordinal,entry in enumerate(host['history']):
            path=a.run/f'scope{index}'/('%03d-%s.json'%(ordinal,entry['action']))
            original_bytes+=path.stat().st_size
            need(original_bytes<=1024**3,'SCOPE_ORIGINAL_RPC_BYTE_BOUND')
            need(P.json_file(path,64*1024**2)==entry,'ORIGINAL_RPC_AND_HOST_HISTORY_DIFFER')
            original_hashes.append(dict(file=path.name,bytes=path.stat().st_size,sha256=E.sha(path)))
        rows,selected=verify_history(host);owners.append(host['owner']);references.append(values(rows[4]))
        need(host['worker']==result['worker']==scope['setup']['worker'],'WORKER_CHANGED')
        need(scope['resources_zero'] is True and len(scope['requests'])==7,'SCOPE_NOT_COMPLETE')
        for r,h in zip(rows,scope['requests']):need(r['phase']==h['phase'] and r['request_id']==h['mapping']['internal_request_id'],'HOST_WORKER_REQUEST_MAPPING')
        native=[]
        for phase,call in zip(('baseline','baseline_repeat','fault','off'),selected):
            capture=call['native']['input_capture'];public=P.json_file(scoped_file(a.run,capture['public_artifact']))
            public['_return']=P.json_file(scoped_file(a.run,capture['returned_artifact']))
            events=P.lines(scoped_file(a.run,capture['native_artifact']),1024**2)
            need(public['owner']==host['owner'] and public['request_id']==rows[PHASES.index(phase)]['request_id']
                and public['build_sha256']==a.build_sha256,'NATIVE_REQUEST_OWNER_OR_BUILD')
            current=P.current_values({'binding_removal':{'input_window':capture}},public)
            need(current['observed'],'FRESH_VALUES_MISSING')
            typed=P.protocol(public,public['_return'],events)
            for stage,data in typed.items():
                if 'function' not in data:continue
                function=data['function'];entry,_=P.function_library_entry(function,entries);path=Path(entry['path'])
                need(path.stat().st_size==entry['bytes']==function['file_bytes'] and E.sha(path)==entry['sha256'],'MAPPED_LIBRARY_CHANGED:'+stage)
                if stage in ('prepare_enter','execute_enter'):
                    need(any(name.endswith('/libcust_opapi.so') and item['sha256']==entry['sha256'] for name,item in files.items()),'WRONG_CURRENT_OPAPI')
                if stage in ('workspace_enter','fia_execute_enter'):need(entry['sha256']==P.NNOPBASE_SHA,'UNREVIEWED_NNOPBASE')
            actual=P.runtime_join(a.run,runtime,public,events,abi,build['kernel_binary_sha256'])
            plan=capture['native_typed_plan_return'];fault=phase=='fault'
            need(plan['fault_authorized'] is fault and bool(plan['replay_parent_committed']) is fault
                and plan['copy_launch_read']==1 and plan['first_error']==plan['terminal_uncertain']==0,'ACTUAL_NATIVE_FINAL_GATE')
            if fault:need(plan['replay_requested']==plan['host_completed']==1 and plan['host_gate_checks']>0,'TWO_STAGE_HOST_NOT_OBSERVED')
            key=actual['launches'][1]['tiling_key'];expected='119000000030400003' if fault else '5000000000000200103'
            need(key==expected and int(key) in build['compiled_keys'],'ACTUAL_KEY_NOT_REVIEWED_VARIANT')
            need(capture['window_complete'] is True and capture['research_policy_released'] is True,'OWNED_POLICY_NOT_RELEASED')
            native.append(dict(phase=phase,actual=actual,current_values=current,native_final_plan=plan,
                public_sha256=E.sha(scoped_file(a.run,capture['public_artifact'])),native_sha256=E.sha(scoped_file(a.run,capture['native_artifact']))))
        summaries.append(dict(owner=host['owner'],host_sha256=E.sha(hostpath),original_rpc=original_hashes,requests=[r['request_id'] for r in rows],
            native=native,root_hidden_logits_fault_changed=True,normal_off_health_full_hash_equal=True,resources_zero=True,
            selected_module=selected[0]['module_path'],tokens_changed=scope['requests'][4]['output_tokens']!=scope['requests'][2]['output_tokens']))
    need(len(set(owners))==2 and references[0]==references[1],'SECOND_SCOPE_OWNER_OR_FAULT_REPEAT_DIFFERS')
    need(len({s['setup']['pid'] for s in result['scopes']})==len({s['setup']['model_object_id'] for s in result['scopes']})==1,'SAME_WORKER_MODEL_REQUIRED')
    return summaries

def main():
    p=argparse.ArgumentParser()
    for name in ('run','build','vendor','source','libraries','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('build-sha256','source-manifest-sha256','libraries-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=dict(status='REJECTED_OR_INCOMPLETE',issues=[],public_qualification_granted=False,reader_sha256=E.sha(__file__),
        scope='EXPLICIT_RESEARCH_B3_B4_B5_NOT_PUBLIC_WORKER_QUALIFICATION',limitations=['Slice POD not decoded','production nnopbase full source equivalence not proved','no C2/C3 or all-target coverage claim'])
    sys.addaudithook(E.prohibit_runtime)
    try:r.update(scopes=review(a),status='TWO_RESEARCH_SCOPES_NATIVE_EFFECT_AND_FULL_RECOVERY_REVIEWED')
    except Exception as error:r['issues'].append(repr(error))
    with a.out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=r['status'],issues=r['issues'])));return 2 if r['issues'] else 0
if __name__=='__main__':raise SystemExit(main())
