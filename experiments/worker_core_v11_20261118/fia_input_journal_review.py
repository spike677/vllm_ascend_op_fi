"""Pure-stdlib original-record review. Never imports an injector/collector.

GWS arguments and creator snapshots are not internal executor input tensors.
Even a complete journal does not establish Slice operands, numerical equality,
live qualification or first-difference evidence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

TENSOR_INDICES=(0,3,4,*range(7,23),24,25,26,40,41)
CONTAINER_INDICES=(1,2,5,6,23)
SCALAR_INDICES=(27,29,30,32,33,34,35,36,38,39)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path,expected=None):
    path=Path(path);before=path.stat()
    if path.is_symlink() or not 0<before.st_size<=64*1024**2:raise ValueError('BOUNDED_ORIGINAL_JSON_REQUIRED')
    raw=path.read_bytes();after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('ORIGINAL_CHANGED_DURING_READ')
    actual=hashlib.sha256(raw).hexdigest()
    if expected is not None and actual!=expected:raise ValueError('ORIGINAL_SHA_DIFFERS')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('DUPLICATE_KEY:'+key)
            result[key]=value
        return result
    def invalid(value):raise ValueError('NONFINITE_JSON:'+value)
    return json.loads(raw,object_pairs_hook=unique,parse_constant=invalid),actual


def review(data,profile,released=None):
    issues=[];checks=0;summary={};native_failures=[]
    def check(ok,name):
        nonlocal checks
        checks+=1
        if not ok:issues.append(name)
        return bool(ok)
    def integer(x,minimum=0):return type(x) is int and x>=minimum
    def in_image(offset,key):
        return integer(offset) and any(segment['vaddr']<=offset<segment['vaddr']+segment['size']
            for segment in profile['originals'][key]['segments'])
    def tensor(t):
        return (t['descriptor']>0 and t['address']>0 and t['complete']==1 and t['called_mask']==127
            and t['null_mask']==t['truncated_mask']==0 and t['statuses']==[0]*7
            and all(integer(n) and n<=8 for n in t['counts']))
    try:
        check(data['schema']=='demoncat.fia_input_journal.observation.v1','SCHEMA')
        check(data['source']=='ACTUAL_NATIVE_RECORDS','RECORD_SOURCE_LABEL')
        check(profile['candidate']=='fia_input_journal','EXACT_BUILD_KIND')
        check(data['containers']['profile_sha256']==profile['profile_sha256']
            and data['containers']['build_sha256']==data['build_sha256'],'SAME_CONTAINER_FIA_BUILD')
        check(data['tensor_formals']==list(TENSOR_INDICES) and data['container_formals']==list(CONTAINER_INDICES),'TYPED_SLOT_LAYOUT')
        check(all(data[key] is False for key in ('C1','C2','C3','E5','qualified','slice_pointer_lineage_proven',
            'prepared_internal_tensor_lineage_proven','tensor_list_members_reread_at_gws')),'NO_INFERRED_AUTHORITY_OR_LINEAGE')
        check(data['tensor_list_members_captured_at']=='ACTUAL_CREATOR_ENTRY','CREATOR_CAPTURE_POINT')
        s=data['state'];events=data['events'];cs=data['containers']['state'];cevents=data['containers']['events']
        startup=data['startup'];identity=startup['identity'];owners=startup['original_functions'];worker=data['worker']
        check(startup['status']=='RESOLVED_BEFORE_LLM' and startup['explicit_operator_calls']==0
            and startup['explicit_device_queries']==0 and startup['install_calls']==0,'INERT_STARTUP')
        check(worker['pid']==identity['pid']==s['pid'] and worker['process_start_identity']==identity['process_start_identity']
            and identity['native_tid']==s['thread'],'SAME_WORKER_PROCESS_LIFETIME')
        check(s['installed']==s['stopped']==1 and s['retired']==0,'STOPPED_PRE_RELEASE_SNAPSHOT')
        for key in ('active_calls','pending_executors','untracked_executors','first_error','overflow','exceptions','foreign_thread_calls','getter_failures'):
            if s[key]:native_failures.append({'state_field':key,'value':s[key]})
        check(s['total_events']==s['retained_events']==len(events) and len(events)==2*s['total_calls'],'FIA_EVENT_DENOMINATOR')
        check(cs['total_events']==cs['retained_events']==len(cevents)==2*cs['total_calls'],'CONTAINER_EVENT_DENOMINATOR')
        check(cs['entries']==cs['returns']==cs['original_calls'],'CONTAINER_ORIGINAL_ONCE_COUNTERS')
        check(sum(cs['original_calls'])==cs['total_calls'],'CONTAINER_COUNTER_DENOMINATOR')
        check(not any(cs[k] for k in ('active_calls','scope_token','live_registrations','first_error','overflow',
            'abi_exceptions','getter_capture_failures','foreign_scope_thread_calls')),'CONTAINER_QUIESCENCE')
        arrays=cs['arrays']
        check(arrays['acquired_arrays']==arrays['released_arrays'] and not any(arrays[k] for k in
            ('active_captures','active_getters','live_arrays','getter_failures','getter_exceptions','capture_exceptions','ownership_errors')),
            'OWNED_GETTER_ARRAYS')
        scopes={x['identity']['scope']['token']:x for x in data['public_scopes']}
        check(len(scopes)==len(data['public_scopes']) and len({x['identity']['call_id'] for x in scopes.values()})==len(scopes),'UNIQUE_PUBLIC_WINDOWS')
        check(integer(s['window_limit'],1) and s['window_limit']<=256
            and s['accepted_windows']==s['retained_scope_tokens']==len(scopes)<=s['window_limit']
            and s['retained_scope_token_characters']==32*len(scopes),'BOUNDED_NATIVE_WINDOW_TOKENS')
        check(data.get('lifecycle_overflow',0)==0,'BOUNDED_LIFECYCLE_JOURNAL')
        if data.get('first_lifecycle_error') is not None:
            native_failures.append(dict(first_lifecycle_error=data['first_lifecycle_error']))
        for row in scopes.values():
            x=row['identity'];check(row['status']=='RETURNED' and row['original_calls']==1 and row['original_error'] is None
                and x['pid']==worker['pid'] and x['worker']==worker['worker_object_id'] and x['model']==worker['model_object_id']
                and integer(x['module'],1) and re.fullmatch('[0-9a-f]{32}',x['call_id']) is not None,'ACTUAL_PUBLIC_WINDOW')
        originals={};destroyed={};live={};cpairs={}
        for number,e in enumerate(cevents,1):
            check(e['sequence']==number,'CONTAINER_SEQUENCE')
            cpairs.setdefault(e['call'],[]).append(e)
            check(e['error']==e['status']==0,'CONTAINER_NATIVE_FAILURE')
            check(e['operation'] in (1,2,3,4) and e['stage']==3 and e['caller_kind']==1
                and e['function']==e['function_base']+e['function_offset'] and in_image(e['function_offset'],'nn')
                and e['caller']==e['caller_base']+e['caller_offset'] and in_image(e['caller_offset'],'torch_npu'),
                'CONTAINER_ORIGINAL_CALLER_IMAGE_OFFSETS')
            if e['phase']!=2:continue
            if e['operation'] in (1,3):
                check(e['handle']>0 and e['registration']==e['call'] and e['handle'] not in live,'CURRENT_CONTAINER_REGISTRATION')
                live[e['handle']]=e['registration'];originals[e['registration']]=e
            elif e['handle']:
                check(live.pop(e['handle'],None)==e['registration'],'EXACT_DESTROY_LIFETIME')
                destroyed[e['registration']]=cpairs[e['call']][0]
        check(not live,'NO_LIVE_CONTAINER')
        for pair in cpairs.values():
            check(len(pair)==2 and [x['phase'] for x in pair]==[1,2] and [x['original_calls'] for x in pair]==[0,1], 'CONTAINER_REAL_FUNCTION_ONCE')
            if len(pair)==2:
                check(all(pair[0][k]==pair[1][k] for k in ('operation','call','scope_token','forward','operator_sequence',
                    'boundary_sequence','stage','pid','thread','function','function_base','function_offset','caller','caller_base','caller_offset')),
                    'CONTAINER_CALL_IDENTITY_STABLE')
        for field,key in dict(prepare_function='opapi',execute_function='opapi',device_function='device_runtime',
                list_size_function='nn',int_size_function='nn',observer_scope_function='observer').items():
            owner=owners[field]
            check(s[field]==owner['address']==owner['base']+owner['offset'] and owner['sha256']==profile['originals'][key]['file_sha256']
                and in_image(owner['offset'],key),
                'STARTUP_ORIGINAL_BINDING:'+field)
        pairs={};last_time=0
        for number,e in enumerate(events,1):
            check(e['sequence']==number and integer(e['monotonic_ns'],1) and e['monotonic_ns']>=last_time,'FIA_ORDER');last_time=e['monotonic_ns']
            pairs.setdefault(e['call'],[]).append(e)
            if e['error'] or e['status'] or e['original_exception']:native_failures.append(dict(event=number,error=e['error'],status=e['status'],exception=e['original_exception']))
            window=scopes.get(e['scope_token']);check(window is not None,'KNOWN_PUBLIC_WINDOW')
            if window is not None:
                x=window['identity'];scope=x['scope']
                check(all(e[k]==x[k] for k in ('pid','worker','model','module','call_id'))
                    and all(e[k]==scope[k] for k in ('thread','forward','operator_sequence','boundary_sequence')),'EXACT_CURRENT_PUBLIC_SCOPE')
            check(e['pid']==s['pid'] and e['thread']==s['thread'] and e['device_status']==0 and integer(e['device']), 'DEVICE_THREAD_IDENTITY')
            check((e['owner_hi'],e['owner_lo'],e['generation'])==(s['owner_hi'],s['owner_lo'],s['generation']), 'NATIVE_OWNER')
            field='prepare_function' if e['operation']==1 else 'execute_function';owner=owners[field]
            check(e['operation'] in (1,2) and e['function']==owner['address'] and e['function_base']==owner['base']
                and e['function_offset']==owner['offset'] and e['caller']==e['caller_base']+e['caller_offset']
                and integer(e['caller'],1) and integer(e['caller_base'],1) and in_image(e['caller_offset'],'torch_npu'),
                'REAL_FUNCTION_AND_CALLER')
        prepares={};executes={};public_joins=[]
        for call,pair in pairs.items():
            if not check(len(pair)==2 and [x['phase'] for x in pair]==[1,2],'COMPLETE_NATIVE_CALL_PAIR'):continue
            b,a=pair
            check(b['original_calls']==0 and a['original_calls']==a['original_status_valid']==1
                and a['original_exception']==0,'ORIGINAL_RETURN_PRESERVED')
            check(all(b[k]==a[k] for k in ('operation','call','prepare_registration','scope_token','function','caller','call_id','pid','thread','device','formals')),'CALL_IDENTITY_STABLE')
            if b['operation']==1:
                prepares[call]=a
                check(b['prepare_registration']==call and a['executor']>0,'FRESH_PREPARE_IDENTITY')
                check(len(b['formals'])==44 and b['tensors']==a['tensors'] and len(b['tensors'])==24,'GWS_ARGUMENT_DESCRIPTORS_STABLE')
                for index,t in zip(TENSOR_INDICES,b['tensors']):
                    check(t['descriptor']==b['formals'][index] and (not t['descriptor'] or tensor(t)), 'NAMED_TENSOR_SLOT:'+str(index))
                check([x['formal'] for x in b['containers']]==list(CONTAINER_INDICES)
                    and [x['formal'] for x in a['containers']]==list(CONTAINER_INDICES),'CONTAINER_SLOT_ORDER')
                for slot,after in zip(b['containers'],a['containers']):
                    index=slot['formal'];check(index in CONTAINER_INDICES and slot['handle']==b['formals'][index] and slot['joined']==after['live_at_return']==1,'CURRENT_CONTAINER_SLOT')
                    check(all(slot[k]==after[k] for k in slot if k!='live_at_return'),'CONTAINER_SLOT_STABLE')
                    if not slot['handle']:continue
                    create=originals.get(slot['registration']);entry=cpairs.get(slot['registration'],[None])[0];destroy=destroyed.get(slot['registration'])
                    if not check(create is not None and entry is not None and destroy is not None,'COMPLETE_CONTAINER_ORIGINALS'):continue
                    check(create['handle']==slot['handle'] and slot['creator_entry']==entry['sequence'] and slot['creator_return']==create['sequence']
                        and create['operation']==slot['kind']==(1 if index in (1,2) else 3)
                        and slot['count']==slot['actual_count']==entry['count'] and slot['payload_bytes']==entry['payload_bytes']
                        and slot['payload_offset']==entry['payload_offset'] and slot['size_status']==0
                        and create['monotonic_ns']<b['monotonic_ns']<=a['monotonic_ns']<destroy['monotonic_ns']
                        and all(entry[k]==b[k] for k in ('scope_token','pid','thread','forward','operator_sequence','boundary_sequence')),'CREATOR_GWS_DESTROY_JOIN')
                    if index in (1,2):check(len(entry.get('tensor_descriptors',[]))==slot['count']
                        and all(tensor(t) for t in entry['tensor_descriptors']),'ACTUAL_LIST_MEMBERS_PRESENT')
                    else:check(len(entry.get('integer_values',[]))==slot['count'],'ACTUAL_INT_ARRAY_MEMBERS_PRESENT')
                check(all(b['formals'][i]==(x & ((1<<64)-1)) for i,x in zip(SCALAR_INDICES,b['signed_scalars']))
                    and len(b['signed_scalars'])==10 and math.isfinite(b['scale'])
                    and b['formals'][28]==struct.unpack('<Q',struct.pack('<d',b['scale']))[0]
                    and b['formals'][37]==b['softmax_lse'] and b['softmax_lse'] in (0,1)
                    and 0<b['layout_length']<16 and len(b['layout'])==b['layout_length'],'ACTUAL_TYPED_SCALARS')
                window=scopes.get(b['scope_token'])
                public=window.get('evidence',{}).get('actual_input_addresses',{}) if window else {}
                roles={'query':b['tensors'][0],'key':None,'value':None,'atten_mask':b['tensors'][2]}
                member_counts={}
                for n,role in ((0,'key'),(1,'value')):
                    slot=b['containers'][n];entry=cpairs.get(slot['registration'],[{}])[0]
                    members=entry.get('tensor_descriptors',[])
                    member_counts[role]=len(members)
                    if len(members)==1:roles[role]=members[0]
                for role,t in roles.items():
                    p=public.get(role);equal=False
                    if p and t is not None:
                        tags=data['sdk_tags'];dtype=tags['float16_tag'] if p['dtype']=='torch.float16' else tags['int8_tag'] if p['dtype']=='torch.int8' else None
                        equal=(t['address']==p['storage_address'] and t['offset']==p['storage_offset'] and t['dtype']==dtype
                            and t['format']==tags['nd_tag'] and p['object_id']>0 and p['storage_identity']>0
                            and t['shape'][:t['counts'][0]]==p['shape'] and t['strides'][:t['counts'][1]]==p['stride']
                            and math.prod(t['storage'][:t['counts'][2]])*p['element_bytes']==p['storage_bytes']
                            and p['data_address']==p['storage_address']+p['storage_offset']*p['element_bytes'])
                    public_joins.append(dict(call=call,role=role,public_record_present=p is not None,storage_geometry_equal=equal,
                        native_descriptor_present=t is not None,creator_member_count=member_counts.get(role),
                        unjoined_reason=None if equal else 'SINGLE_CREATOR_MEMBER_REQUIRED' if role in member_counts and member_counts[role]!=1
                            else 'PUBLIC_RECORD_MISSING' if not p else 'STORAGE_GEOMETRY_OR_TYPE_NOT_EQUAL',
                        capture_point='CREATOR_ENTRY' if role in ('key','value') else 'GWS_ARGUMENT_ENTRY',internal_prepared_input_proven=False))
            else:executes.setdefault(b['prepare_registration'],[]).append(pair)
        for registration,after in prepares.items():
            actual=executes.get(registration,[])
            if not check(len(actual)==1,'ONE_EXECUTE_PER_PREPARE'):continue
            b,a=actual[0]
            check(after['executor']==b['executor']==b['formals'][2] and b['workspace']==b['formals'][0]
                and b['workspace_bytes']==b['formals'][1]>=after['workspace_bytes'] and b['formals'][3]>0
                and (not after['workspace_bytes'] or b['workspace']>0) and after['monotonic_ns']<b['monotonic_ns']
                and all(after[k]==b[k] for k in ('scope_token','pid','thread','device','call_id')),'REAL_EXECUTOR_WORKSPACE_STREAM')
        check(set(executes)==set(prepares),'NO_UNKNOWN_EXECUTE')
        check(s['prepare_returns']==s['original_prepares']==len(prepares)
            and s['execute_returns']==s['original_executes']==s['paired_execute_returns']==len(executes),'REAL_PAIR_COUNTERS')
        window_counts={token:sum(x['scope_token']==token for x in prepares.values()) for token in scopes}
        check(bool(window_counts) and all(window_counts.values()),'EVERY_PUBLIC_WINDOW_HAS_NATIVE_PREPARE')
        summary.update(prepare_calls=len(prepares),execute_calls=len(executes),windows=len(scopes),
            window_prepare_counts=window_counts,
            retained_events=len(events),container_calls=len(cpairs),public_storage_geometry=public_joins,
            public_storage_geometry_all_joined=bool(public_joins) and all(x['storage_geometry_equal'] for x in public_joins))
        check(released is not None,'RELEASE_ORIGINAL_REQUIRED')
        if released is not None:
            final=released['fia'];container=released['containers']
            check(final['installed']==0 and final['retired']==final['stopped']==1 and not any(final[k] for k in
                ('active_calls','pending_executors','untracked_executors','record_capacity_bytes','retained_events',
                 'retained_scope_tokens','retained_scope_token_characters')),
                'ZERO_OWNED_FIA_RESOURCES')
            check(container['installed']==0 and container['retired']==container['stopped']==1 and not any(container[k] for k in
                ('active_calls','scope_token','live_registrations','record_capacity_bytes','payload_capacity_bytes','retained_events')),
                'ZERO_OWNED_CONTAINER_RESOURCES')
            unchanged=('owner_hi','owner_lo','generation','pid','thread','total_calls','total_events','first_error',
                'prepare_returns','execute_returns','original_prepares','original_executes','paired_execute_returns',
                'window_limit','accepted_windows')
            check(all(final[k]==s[k] for k in unchanged),'RELEASE_SAME_FIA_SESSION')
            check(all(container[k]==cs[k] for k in ('owner_hi','owner_lo','generation','owner_thread','total_calls',
                'total_events','entries','returns','original_calls','first_error','arrays')),'RELEASE_SAME_CONTAINER_SESSION')
            check(released['global_wrappers_unloaded'] is False and released['preloaded_library_remains_until_process_exit'] is True,
                'NO_GLOBAL_UNLOAD_CLAIM')
            check(released.get('lifecycle_overflow',0)==0,'RELEASE_LIFECYCLE_BUDGET')
            if released.get('first_lifecycle_error') is not None:
                native_failures.append(dict(release_first_lifecycle_error=released['first_lifecycle_error']))
    except (KeyError,TypeError,ValueError,IndexError,OverflowError,AttributeError) as error:
        issues.append('MISSING_OR_INVALID_ORIGINAL:'+repr(error))
    complete=not issues and not native_failures and summary.get('prepare_calls',0)>0
    return dict(schema='demoncat.fia_input_journal.offline_review.v1',checks=checks,issues=issues,native_failures=native_failures,
        status=('COMPLETE_ARGUMENT_JOURNAL_NO_QUALIFICATION' if summary.get('public_storage_geometry_all_joined')
            else 'COMPLETE_NATIVE_JOURNAL_PUBLIC_JOIN_INCOMPLETE') if complete else 'NO_OBSERVATION_OR_PARTIAL_FAILURE',
        observed=summary,originals_modified=False,
        Slice_ABI_known=False,slice_pointer_lineage_proven=False,prepared_internal_tensor_lineage_proven=False,
        C1=False,C2=False,C3=False,E5=False,qualification_granted=False,
        limitations=['Creator metadata is not a reread of opaque list members at execute.',
            'Equal public/GWS storage geometry does not prove the Slice operand or the internal prepared FIA tensor.',
            'A saved JSON review cannot issue a live proof. No numerical or first-difference evidence is assessed.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('observation','build_root','out'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--observation-sha256',required=True);p.add_argument('--build-sha256',required=True)
    p.add_argument('--released',type=Path);p.add_argument('--released-sha256');args=p.parse_args()
    if (args.released is None)!=(args.released_sha256 is None):raise ValueError('RELEASE_PATH_AND_PIN_REQUIRED_TOGETHER')
    if args.out.exists():raise ValueError('FRESH_REVIEW_OUTPUT_REQUIRED')
    data,digest=read(args.observation,args.observation_sha256);build,_=read(args.build_root/'build.json',args.build_sha256)
    if (build.get('candidate')!='fia_input_journal' or build.get('qualified') is not False
            or data.get('build_sha256')!=args.build_sha256):raise ValueError('EXACT_READONLY_BUILD_REQUIRED')
    profile,_=read(args.build_root/'profile.json',build['files']['profile.json'])
    canonical=dict(profile);ident=canonical.pop('profile_sha256')
    if ident!=build['profile_sha256'] or ident!=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest():
        raise ValueError('PROFILE_CANONICAL_IDENTITY_CHANGED')
    for relative,expected in build['files'].items():
        path=(args.build_root/relative).resolve(strict=True)
        if not path.is_relative_to(args.build_root.resolve(strict=True)) or sha(path)!=expected:raise ValueError('BUILD_ARTIFACT_CHANGED:'+relative)
    released,release_digest=read(args.released,args.released_sha256) if args.released else (None,None)
    result=review(data,profile,released);result.update(original_sha256=digest,released_sha256=release_digest,
        build_sha256=args.build_sha256,reviewer_sha256=sha(__file__))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
