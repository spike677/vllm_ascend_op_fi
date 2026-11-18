"""Independent all-object normal review. Reads bytes only; never grants F."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import arch22_normal_bridge_evidence as E
import review_arch22_normal_bridge as N
from review_arch22_s2_single import request_lineage,decode_cache_rows


def phase_capture(run,phase,*,group=False):
    if not group:return phase['finish']['capture']
    name=phase.get('phase','UNKNOWN');finish=phase.get('finish',{})
    E.need(type(finish) is dict and finish.get('closed') is True
        and type(finish.get('capture_artifact')) is dict,'GROUP_INCOMPLETE_NORMAL_PHASE:'+str(name)+'.finish.capture_artifact')
    reference=finish['capture_artifact']
    E.need({'schema','path','sha256','bytes','payload_identity'}<=set(reference),
           'GROUP_INCOMPLETE_NORMAL_PHASE:'+str(name)+'.finish.capture_artifact.fields')
    E.need(name in N.PHASES and finish['closed'] is True and 'capture' not in finish
        and reference['schema']=='demoncat.paged33.group_normal_capture_ref.v1'
        and reference['path']=='/evidence/run/'+name+'/runtime_result.json'
        and type(reference['bytes']) is int and 0<reference['bytes']<=128*1024**2,
        'GROUP_NORMAL_COMPLETE_BOUNDED_CAPTURE_REFERENCE_REQUIRED')
    path=E.child(run,name+'/runtime_result.json')
    E.need(path.stat().st_size==reference['bytes'] and E.sha(path)==reference['sha256'],
           'GROUP_NORMAL_CAPTURE_ARTIFACT_BYTES_CHANGED')
    capture=E.read(path)
    def identity(value):return {key:value[key] for key in ('shape','dtype','bytes','sha256')}
    payloads={key:identity(capture[key]) for key in ('hidden','logits_hidden','logits')}
    payloads['attention']={path:{kind:identity(value) for kind,value in rows.items()} for path,rows in capture['attention'].items()}
    E.need(payloads==reference['payload_identity'],'GROUP_NORMAL_RPC_PAYLOAD_REFERENCE_DIFFERS')
    return capture


def indexed_cost(capture,phase,runtime,run):
    import math
    cost=capture['runtime_validation_cost'];index=cost['index'];size=index['prefix_bytes']
    path=E.child(run,'runtime/runtime.jsonl')
    E.need(type(size) is int and 0<size<=path.stat().st_size<=128*1024**2,'GROUP_INDEX_PREFIX_BOUND')
    with path.open('rb') as stream:raw=stream.read(size)
    rows=[json.loads(line) for line in raw.splitlines()]
    scope=capture['target_scope'];seconds=cost['indexed_history_validation_seconds'];elapsed=phase['elapsed_seconds']
    E.need(raw.endswith(b'\n') and hashlib.sha256(raw).hexdigest()==index['prefix_sha256']
        and rows==runtime[:len(rows)] and index['parsed_events']==len(rows)
        and index['owner']==[scope['pid'],scope['python_thread_ident'],scope['native_tid']]
        and index['schema']=='demoncat.attention.owned_runtime_index.v1'
        and index['complete_old_prefix_rehashed'] is True and index['raw_log_modified'] is index['qualification_granted'] is False
        and cost['raw_log_kept_complete'] is cost['all_prior_registration_conflicts_retained'] is True
        and type(index['refreshes']) is int and index['refreshes']>=28
        and all(type(n) in (int,float) and math.isfinite(n) and n>0 for n in (elapsed,seconds,index['validation_seconds']))
        and index['validation_seconds']<=seconds<=elapsed,'GROUP_ACTUAL_INDEX_PREFIX_OR_COST_DIFFERS')
    return dict(elapsed_seconds=elapsed,indexed_history_validation_seconds=seconds,prefix_bytes=size)


def target_set(capture,audit,*,expected_generation=1):
    scope=capture['target_scope'];rows=scope['targets'];paths=[r['path'] for r in rows]
    E.need(scope['schema']=='demoncat.paged33.all_normal_targets.v1' and len(rows)==28
        and len(set(paths))==len({r['module_object_id'] for r in rows})==28
        and set(paths)==set(audit)==set(capture['all_decode_boundaries'])
        and scope['sha256']==E.digest(rows) and type(scope['generation']) is int and scope['generation']==expected_generation
        and all(type(scope[key]) is int and scope[key]>0 for key in ('owner_object_id','worker_object_id',
                'model_object_id','pid','python_thread_ident','native_tid'))
        and all(type(row[key]) is int and row[key]>0 for row in rows for key in ('module_object_id','impl_object_id'))
        and scope['normal_only'] is True and all(scope[k] is False for k in ('serializable_authority','C1','C2','C3')),
        'ALL_NORMAL_ACTUAL_TARGET_SET_INCOMPLETE_OR_CHANGED')
    for row in rows:
        path=row['path'];saved=capture['all_decode_boundaries'][path];identity=saved['logical_input_binding']['identity']
        E.need(row['module_object_id']==audit[path]['module_object_id']==identity['module_object_id']
            and row['impl_object_id']==audit[path]['impl_object_id']==identity['impl_object_id']
            and all(scope[k]==identity[k] for k in ('owner_object_id','generation','worker_object_id','model_object_id',
                'pid','python_thread_ident','native_tid')),'ALL_NORMAL_TARGET_SCOPE_TO_REAL_CALL_DIFFERS')
    return paths


def compare(left,right,helper,*,relocated=False):
    E.need(set(left['targets'])==set(right['targets']) and len(left['targets'])==28,'ALL_NORMAL_PAIR_TARGET_SET_DIFFERS')
    payloads=helper.differences.compare(left['payloads'],right['payloads'],N.LAYERS)
    targets={}
    for path in left['targets']:
        result=N.compare_pair(dict(boundary=left['targets'][path]),dict(boundary=right['targets'][path]),helper,
                              allow_page_relocation=relocated,payload_comparison=payloads)
        targets[path]={k:v for k,v in result.items() if k!='all_complete_payloads'}
    return dict(targets=targets,all_complete_payloads=payloads,exact=all(r['exact'] for r in targets.values()),
        normal_repeat_valid=all(r.get('normal_repeat_valid',r['exact']) for r in targets.values()))


def actual_output_join(capture):
    """Join rehashed full payloads to final model and original API returns."""
    decode=capture['natural_request']['forwards'][1]
    E.need(capture['complete_tensor_readback_during_statistics'] is False
        and all(all(capture[name][key]==decode[name][key] for key in ('shape','dtype','sha256','finite'))
                for name in ('hidden','logits'))
        and all(capture['hidden'][key]==capture['logits_hidden'][key] for key in ('shape','dtype','bytes','sha256','finite')),
        'ALL_NORMAL_ACTUAL_FINAL_MODEL_PAYLOAD_JOIN_DIFFERS')
    for call in decode['calls']:
        value=capture['attention'][call['module_path']]['output'];actual=call['output']
        E.need(type(actual) is list and len(actual)==2
            and all(actual[0][key]==value[key] for key in ('shape','dtype','sha256','finite')),
            'ALL_NORMAL_ACTUAL_API_TO_FULL_LAYER_PAYLOAD_DIFFERS')


def arm(run,expected,kind,source,helper,bundle,observer,checks,symbols):
    run=Path(run).resolve(strict=True);E.need(E.sha(run/'result.json')==expected,'ALL_NORMAL_RESULT_SHA_DIFFERS')
    data=E.read(run/'result.json');failures=N.original_failures(data)
    checks.require(kind+'.failures',not failures,'ORIGINAL_RUN_FAILURES_PRESERVED')
    N.check_execution(data,kind,source,helper,bundle,observer)
    E.need(data['native_target_scope']==('ALL_DISCOVERED_OBJECTS_GROUP_PACKAGE_NORMAL_ONLY' if E.variant(source)==E.GROUP_VARIANT else 'ALL_DISCOVERED_OBJECTS_NORMAL_ONLY')
        and [p['phase'] for p in data['phases']]==list(N.PHASES),'ALL_NORMAL_FIVE_PHASE_SCOPE_REQUIRED')
    runtime,raw=helper.boundary.read_runtime(run)
    E.need(not any(r['event']=='capture_overflow' for r in runtime),'ALL_NORMAL_RUNTIME_OVERFLOW')
    tokens=data['warm_healthy']['prompt_token_ids'];values={};summary=[];all_tokens=[];identities=[];owners=[];requests=[]
    E.need(len(tokens)==1024 and hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()==data['prompt']['token_sha256'],
           'ALL_NORMAL_ACTUAL_PROMPT_HASH_DIFFERS')
    group=E.variant(source)==E.GROUP_VARIANT
    if group:
        for p in data['phases']:
            E.need(p.get('finish',{}).get('closed') is True and
                type(p.get('finish',{}).get('capture_artifact',{}).get('bytes')) is int,
                'GROUP_INCOMPLETE_NORMAL_PHASE:'+str(p.get('phase'))+'.finish.capture_artifact')
        E.need(sum(p['finish']['capture_artifact']['bytes'] for p in data['phases'])<=640*1024**2,
               'GROUP_NORMAL_TOTAL_CAPTURE_BYTES_BOUND')
    for original_phase in data['phases']:
        capture=phase_capture(run,original_phase,group=group)
        phase=dict(original_phase,finish=dict(original_phase['finish'],capture=capture))
        name=phase['phase'];key=kind+'.'+name;private=kind=='baseline' and name!='healthy_after'
        N.check_cleanup(phase['finish'],private)
        E.need(phase['finish']['cleanup']['logical_bindings_released'] is True
            and phase['finish']['cleanup']['normal_only'] is True
            and E.read(run/name/'runtime_result.json')==capture,'ALL_NORMAL_CAPTURE_OR_OWNED_INPUT_CLEANUP_DIFFERS')
        E.need(phase['setup']['source']==capture['audit']['source'] and phase['setup']['candidate_count']==28
            and phase['setup']['stock_read_only'] is (not private) and phase['setup']['core_methods_replaced'] is False
            and phase['setup']['module_sha256']==source['pins'][E.WORKER+'paged33_tnd_probe.py']
            and phase['setup']['all_normal_session_sha256']==source['pins'][E.WORKER+('paged33_group_normal.py' if E.variant(source)==E.GROUP_VARIANT else 'paged33_all_normal.py')],
            'ALL_NORMAL_INSTALLED_SESSION_SOURCE_DIFFERS')
        # A read-only view maps the producer's natural_request into the shared
        # request validator. It supplies no missing scheduler or call fields.
        lineage=request_lineage(dict(mapping=phase['mapping'],phase=name,worker=capture['natural_request'],
            completed=phase['finish']['closed'],response=phase['response']),tokens,'npu:0')
        prefill,decode=capture['natural_request']['forwards']
        lineage['writer_consumer']=decode_cache_rows(prefill['scheduler'],decode['scheduler'],decode['calls'],device='npu:0')
        E.need(capture['audit']['rows']==capture['natural_request']['forwards'][1]['calls'],
               'ALL_NORMAL_ACTUAL_DECODE_AND_CAPTURE_ROWS_DIFFER')
        requests.append(lineage['internal_request_id'])
        audit=helper.bridge.boundary_rows(capture,phase,tokens);paths=target_set(capture,audit)
        payloads=N.payloads_for_phase(run,name,capture,helper);actual_output_join(capture)
        current={};phase_summary=dict(phase=name,request=lineage,targets={})
        if group:phase_summary['runtime_cost']=indexed_cost(capture,phase,runtime,run)
        for ordinal,path in enumerate(paths):
            saved=capture['all_decode_boundaries'][path]
            N.check_actual_inputs(saved);N.check_live_input(saved,audit[path]);thread=N.captured_thread_identity(saved)
            identity=tuple(saved[k] for k in ('pid','worker_object_id','model_object_id','rank'))+(thread['python_thread_ident'],thread['native_tid'])
            identities.append(identity)
            boundary=helper.boundary.first_boundary(run,phase,audit[path],difference_reader=helper.differences,
                target_path=path,relative_folder='boundaries/'+str(ordinal))
            boundary['page_relocation']=N.page_relocation_evidence(saved,boundary,audit[path],target_path=path);current[path]=boundary
            token=saved['native_call_id'];all_tokens.append(token)
            resolution=N.one_launch(run,runtime,token,saved['pid'],helper,native_tid=thread['native_tid'],symbols=symbols,
                kernel_sha=bundle['completion']['kernel_binary_sha256'] if kind=='baseline' else None)
            E.need(saved['runtime_observation']['call_id']==token,'ALL_NORMAL_CAPTURE_RUNTIME_SCOPE_DIFFERS')
            phase_summary['targets'][path]=dict(native_call_id=token,input_sha256=boundary['input_sha256'],
                query_sha256=boundary['query']['sha256'],output_sha256=[r['sha256'] for r in boundary['outputs']],
                launches=resolution['launches'],original_runtime_observation=saved['runtime_observation'])
        if private:
            native=N.native_phase(run,phase,audit,runtime,capture['all_decode_boundaries'][paths[0]],helper,bundle,checks,symbols)
            E.need(native['independently_replayed']==28,'ALL_NORMAL_28_FINAL_PRIVATE_PODS_REQUIRED')
            phase_summary['native_replay']=native
        else:E.need(capture['native_calls']==[] and all(row['native'] is None for row in audit.values()),
                    'ALL_NORMAL_PRIVATE_POLICY_IN_STOCK_SCOPE')
        E.need(phase['response']['outputs']==data['warm_healthy']['outputs'],'ALL_NORMAL_HEALTHY_GENERATION_CHANGED')
        owners.append(capture['target_scope']['targets'])
        summary.append(phase_summary);values[name]=dict(payloads=payloads,targets=current)
    E.need(len(requests)==len(set(requests))==5 and len(set(identities))==1 and all(x==owners[0] for x in owners),
           'ALL_NORMAL_SAME_WORKER_TARGETS_OR_FRESH_REQUESTS_CHANGED')
    E.need(len(all_tokens)==len(set(all_tokens))==140 and Counter(r['call_id'] for r in runtime if r['event']=='begin')
        ==Counter(r['call_id'] for r in runtime if r['event']=='end')==Counter(all_tokens),'ALL_NORMAL_140_OWNED_SCOPE_CENSUS')
    library=N.check_coexistence(data,source,bundle,identities[0][0]) if kind=='baseline' else None
    repeats={}
    for name in N.PHASES[1:]:
        repeats[name]=compare(values[N.PHASES[0]],values[name],helper,relocated=True)
        checks.require(kind+'.repeat.'+name,repeats[name]['normal_repeat_valid'],'ALL_NORMAL_WITHIN_ARM_REPEAT_DIFFERS')
    E.need(data['healthy_after_owned_observers_removed']['outputs']==data['warm_healthy']['outputs']
        and data['healthy_after_owned_observers_removed']['prompt_token_ids']==tokens,'ALL_NORMAL_POST_CLEAN_TOKEN_HEALTH_DIFFERS')
    E.need(E.sha(run/'result.json')==expected and E.sha(run/'runtime/runtime.jsonl')==raw['sha256'],'ALL_NORMAL_ORIGINAL_CHANGED_DURING_REVIEW')
    return dict(data=data,values=values,summary=dict(result_path=str(run/'result.json'),result_sha256=expected,
        original_failures=failures,phases=summary,within_run=repeats,runtime=raw,library_coexistence=library,
        actual_calls=280,decode_calls=140,private_normal_calls=112 if kind=='baseline' else 0,
        post_clean_scope='ORDINARY_REQUEST_TOKENS_TEXT_ONLY'))


def review(args,*,build_variant=E.ALL_VARIANT):
    checks=N.Checks();report=dict(schema='demoncat.paged33.all_normal_offline_review.v1',
        status='INCOMPLETE_OR_REJECTED_ALL_NORMAL_NO_F',fault_authorized=False,C1=False,C2=False,C3=False,
        E5='NOT_ASSESSED',actual_npu_calls=0,original_results_modified=False,arms={},cross_arm={},registered_elfs={})
    for kind in ('stock','baseline'):
        def original(kind=kind):
            path=Path(getattr(args,kind+'_run'))/'result.json';expected=getattr(args,kind+'_sha256')
            E.need(E.sha(path)==expected,'ALL_NORMAL_ORIGINAL_RESULT_SHA_DIFFERS')
            data=E.read(path)
            return dict(result_path=str(path),result_sha256=expected,original_status=data.get('status'),
                        original_failures=N.original_failures(data))
        report['arms'][kind]=checks.run(kind+'.original',original)
    def context():
        source=E.frozen_source(args.source,args.source_manifest_sha256,build_variant=build_variant)
        E.need(E.sha(Path(__file__))==source['pins'][E.EXPERIMENT+Path(__file__).name],'ALL_NORMAL_REVIEWER_SOURCE_DIFFERS')
        helper=E.helpers(source);bundle=E.build_bundle(args.build_root,source,helper,args.completion_sha256)
        observer=E.observer_bundle(args.observer,source,args.observer_receipt_sha256)
        return source,helper,bundle,observer
    loaded=checks.run('source_build_observer',context)
    if loaded is not None:
        source,helper,bundle,observer=loaded
        report['source']=dict(manifest_sha256=source['manifest_sha256'],required_sources=source['pins'])
        report['build']=bundle['summary'];arms={}
        for kind in ('stock','baseline'):
            result=checks.run(kind,lambda kind=kind:arm(getattr(args,kind+'_run'),getattr(args,kind+'_sha256'),kind,
                source,helper,bundle,observer,checks,report['registered_elfs']))
            if result is not None:arms[kind]=result;report['arms'][kind]=result['summary']
        if len(arms)==2:
            checks.require('paired_execution',arms['stock']['data']['execution_contract']==arms['baseline']['data']['execution_contract'],
                           'ALL_NORMAL_PAIRED_SOURCE_MODEL_ENVIRONMENT_DIFFERS')
            for phase in N.PHASES:
                value=checks.run('cross.'+phase,lambda phase=phase:compare(arms['stock']['values'][phase],arms['baseline']['values'][phase],helper))
                report['cross_arm'][phase]=value
                checks.require('cross.'+phase,value is not None and value['exact'],'ALL_NORMAL_RAW_TARGET_INPUT_OR_COMPLETE_OUTPUT_DIFFERS')
            if not checks.issues:report.update(status='SOURCE_BOUND_28_TARGET_NORMAL_BRIDGE_RECORDED_NO_F_AUTHORITY',
                complete_payload_pairs=435,per_target_input_output_pairs=140,normal_payloads_exact=True)
    report.update(checks=checks.checks,issues=checks.issues,limitations=[
        'Only this fresh package, model, prompt, real TP1 Worker and actual TND Q1/KV1025/page16 calls are covered.',
        'No F, C1/C2/C3, live C2Proof or E5 authority. No first-layer certificate is inherited.',
        'Stock selected ACLNN owner is unobserved; its actual registered ELF/key/symbol/launch bytes are verified without guessing a POD ABI.',
        'Mask is hash-only; read-before complete logical KV hashes are reconciled with rehashed complete matching post-consumer files.',
        'Within-arm physical page relocation is explicit; cross-arm same-phase raw inputs remain strict. Checkpoints do not prove absence of intervening writes.',
        'The final ordinary post-clean request proves tokens/text only.'])
    return report


def main(*,build_variant=E.ALL_VARIANT):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','build-root','observer','stock-run','baseline-run','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args();sys.dont_write_bytecode=True;E.need(not args.out.exists(),'FRESH_ALL_NORMAL_REVIEW_REQUIRED')
    numeric=N.initialize_cpu_byte_reader();sys.addaudithook(E.prohibit_runtime)
    result=review(args,build_variant=build_variant);result['cpu_byte_reader']=numeric
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=result['checks'],issues=len(result['issues']),sha256=E.sha(args.out))))
    return 0 if not result['issues'] else 2

if __name__=='__main__':raise SystemExit(main())
