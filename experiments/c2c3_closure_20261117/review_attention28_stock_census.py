"""Independent file review of all public TND2048 stock call scopes; never F.

Only stdlib and the reviewed pure raw-event/ELF parser are used. No injector,
framework, DSO, subprocess or device is loaded. Recorded success flags are
not proof; each actual scope, artifact and entry symbol is read again.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0, str(HERE))
import arch22_normal_bridge_evidence as E
import attention28_census_raw as RAW
from prepare_attention28_stock_census import REQUIRED

PHASES = ('observe', 'observe_repeat')
# collect_replay() sends advance(settings={'collect': True}); history records
# the actual RPC action, not the Host convenience method's Python name.
ACTIONS = ['attach', 'prepare', 'register', 'check', 'enable'] + ['advance', 'query', 'advance']*2 + [
    'finish', 'disable', 'unregister', 'detach']
SUPPLIED = {'query','key','value','atten_mask','block_table','input_layout','block_size',
    'num_heads','num_key_value_heads','scale','sparse_mode','actual_seq_lengths','actual_seq_lengths_kv'}
MECHANISMS = ('28.fia.legal_tiling', '28.fia.denominator_replay')
RAW_READER = E.WORKER+'attention_stock_launch.py'


def need(value, reason):
    if not value:raise ValueError(reason)


def valid_sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def json_file(path, limit=512*1024**2):
    path=Path(path);before=path.stat()
    need(0 < before.st_size <= limit and not path.is_symlink(), 'ORIGINAL_JSON_BOUND_OR_LINK')
    with path.open(encoding='utf-8') as stream:
        value=json.load(stream,object_pairs_hook=E.unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON')))
    after=path.stat()
    need((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'ORIGINAL_JSON_CHANGED_DURING_READ')
    return value


def source_files(root, expected):
    root=Path(root).resolve(strict=True);path=root/'source_manifest.json'
    need(valid_sha(expected) and E.sha(path) == expected, 'EXACT_FROZEN_SOURCE_MANIFEST_REQUIRED')
    manifest=json_file(path);rows=manifest['files'];files={r['path']:r for r in rows}
    need(manifest['schema'] == 'demoncat.c2c3.source.v1' and len(files) == len(rows), 'SOURCE_MANIFEST_AMBIGUOUS')
    pins={}
    for name in REQUIRED:
        path=E.child(root,name);pin=files[name]
        need(path.stat().st_size == pin['bytes'] and E.sha(path) == pin['sha256'], 'FROZEN_SOURCE_CHANGED:'+name)
        pins[name]=pin['sha256']
    return dict(root=root,pins=pins,manifest=manifest,manifest_sha256=expected)


def raw_reader(source):
    # This module contains only the previously reviewed stdlib file parser.
    # Loading it by file avoids importing the production provider/package.
    path=E.child(source['root'],RAW_READER)
    need(E.sha(path) == source['pins'][RAW_READER], 'FROZEN_STDLIB_READER_CHANGED')
    spec=importlib.util.spec_from_file_location('_attention28_independent_raw_file_reader',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Review:
    def __init__(self, root, *, recorded_root='/evidence/probe',expected_runtime=None,expected_elf=None):
        self.root=Path(root).resolve(strict=True);self.recorded=PurePosixPath(recorded_root)
        need(self.recorded.is_absolute() and '..' not in self.recorded.parts, 'RECORDED_ROOT_INVALID')
        self.checks=0;self.issues=[];self.artifacts=[];self._elf={}
        self.expected_runtime=expected_runtime;self.expected_elf=expected_elf;self.seen_elf=set()

    def check(self, value, reason, where=''):
        self.checks+=1
        if not value:self.issues.append(dict(where=where,reason=reason))
        return bool(value)

    def attempt(self, where, function):
        try:return function()
        except Exception as error:
            self.check(False,repr(error),where);return None

    def path(self, recorded):
        value=PurePosixPath(recorded)
        need(type(recorded) is str and '\\' not in recorded and '..' not in value.parts, 'RECORDED_PATH_INVALID')
        relative=value.relative_to(self.recorded) if value.is_absolute() else value
        return E.child(self.root,str(relative))

    def artifact(self, ref, *, parse=True, limit=512*1024**2):
        path=self.path(ref['path']);size=path.stat().st_size
        need(valid_sha(ref['sha256']) and E.sha(path) == ref['sha256']
             and ('bytes' not in ref or size == ref['bytes']), 'ORIGINAL_ARTIFACT_HASH_OR_SIZE_CHANGED:'+str(path))
        self.artifacts.append(dict(path=str(path),bytes=size,sha256=ref['sha256']))
        return json_file(path,limit) if parse else path


def tensor_record(value):
    need(type(value) is dict and set(value) == {'shape','dtype','sha256','finite'}, 'COMPLETE_TENSOR_HASH_SCHEMA')
    need(type(value['shape']) is list and all(type(n) is int and n >= 0 for n in value['shape'])
         and type(value['dtype']) is str and valid_sha(value['sha256']) and value['finite'] is True,
         'COMPLETE_TENSOR_HASH_INVALID')


def all_tensors(value):
    if type(value) is dict:
        if 'sha256' in value and 'shape' in value:tensor_record(value);return 1
        return sum(all_tensors(v) for v in value.values())
    if type(value) is list:return sum(all_tensors(v) for v in value)
    need(value is None or type(value) in (str,bool,int,float), 'UNSUPPORTED_COMPLETE_OUTPUT')
    return 0


def mask_sha():
    digest=hashlib.sha256()
    for row in range(2048):digest.update(bytes(row+1)+b'\1'*(2047-row))
    return digest.hexdigest()


def api_input(call, request_id):
    c=call['native_census'];api=c['api'];values=api['inputs'];meta=api['live_metadata'];scheduler=call['scheduler']
    need(set(values) == SUPPLIED and api['supplied_argument_names'] == sorted(SUPPLIED)
         and type(api['positional_count']) is int and 0 <= api['positional_count'] <= 3,
         'ACTUAL_COMPLETE_SUPPLIED_ARGUMENTS_DIFFER')
    need(api['original_callable'] == call['actual_callable']
         and api['original_callable']['type'] == 'torch._ops.OpOverloadPacket'
         and type(api['original_callable']['object_id']) is int and api['original_callable']['object_id'] > 0,
         'ORIGINAL_PUBLIC_PACKET_OBJECT_JOIN')
    defaults=api['omitted_argument_defaults']
    need(type(defaults) is dict and not set(defaults).intersection(values)
         and all(type(defaults.get(k)) is int for k in ('pre_tokens','next_tokens','inner_precise'))
         and type(defaults.get('softmax_lse_flag')) is bool and type(api['public_default_schema']) is str
         and 'npu_fused_infer_attention_score' in api['public_default_schema']
         and api['default_schema_dispatch_observed'] is False and api['actual_native_attributes_observed'] is False,
         'DECLARED_DEFAULTS_MUST_REMAIN_SEPARATE_FROM_NATIVE_ATTRIBUTES')
    expected=dict(input_layout='TND',block_size=128,block_table=None,num_heads=16,num_key_value_heads=8,
        sparse_mode=3,actual_seq_lengths=[2048],actual_seq_lengths_kv=[2048])
    need(all(values[k] == dict(kind='scalar',value=v) for k,v in expected.items())
         and values['scale']['kind'] == 'scalar' and type(values['scale']['value']) in (int,float)
         and math.isclose(values['scale']['value'],128**-.5,rel_tol=0,abs_tol=1e-12), 'ACTUAL_TND2048_SCALARS_DIFFER')
    need(scheduler['internal_request_id'] == request_id and scheduler['num_requests'] == 1
         and scheduler['num_computed_tokens'] == 0
         and scheduler['num_prompt_tokens'] == scheduler['num_scheduled_tokens'] == 2048, 'ACTUAL_FRESH_REQUEST_UNBOUND')
    need(meta['num_prefills'] == 1 and meta['num_decodes'] == 0 and meta['num_actual_tokens'] == 2048
         and meta['causal'] is True and meta['seq_lens_list'] == meta['actual_seq_lengths_q'] == [2048]
         and meta['mask_is_actual_metadata_object'] is True, 'ACTUAL_LIVE_PREFILL_METADATA_DIFFERS')
    devices=set()
    for name,shape,dtype,width in (('query',[2048,16,128],'torch.float16',2),('key',[2048,8,128],'torch.float16',2),
            ('value',[2048,8,128],'torch.float16',2),('atten_mask',[2048,2048],'torch.int8',1)):
        item=values[name];snap=item['snapshot'];m=snap['metadata'];devices.add(m['device'])
        need(item['kind'] == 'tensor' and m['shape'] == shape and m['dtype'] == dtype
             and len(m['stride']) == len(shape) and all(type(n) is int and n > 0 for n in m['stride'])
             and type(m['storage_offset']) is int and m['storage_offset'] >= 0
             and all(type(m[k]) is int and m[k] > 0 for k in ('object_id','storage_identity','storage_nbytes')),
             'ACTUAL_QKV_MASK_GEOMETRY_DIFFERS:'+name)
        need(snap['finite'] is True and snap['bytes'] == math.prod(shape)*width and valid_sha(snap['sha256'])
             and (m['storage_offset']+1+sum((n-1)*s for n,s in zip(shape,m['stride'])))*width <= m['storage_nbytes'],
             'ACTUAL_QKV_MASK_STORAGE_EXTENT_OR_HASH_DIFFERS:'+name)
        need(call['input'][name] == {k:(m[k] if k in ('shape','dtype') else snap[k])
             for k in ('shape','dtype','sha256','finite')}, 'PUBLIC_INPUT_TO_BOUNDARY_HASH_JOIN:'+name)
    need(len(devices) == 1 and next(iter(devices)).split(':')[0] == 'npu', 'ACTUAL_NPU_DEVICE_DIFFERS')
    need(values['atten_mask']['snapshot']['sha256'] == mask_sha(), 'COMPLETE_CAUSAL_MASK_HASH_DIFFERS')
    need(set(call['input']) == set(values) and all(call['input'][k] == values[k]['value']
         for k in values if values[k]['kind'] == 'scalar'), 'PUBLIC_SCALAR_INPUT_JOIN')
    need(api == c['api_inputs_after'], 'COMPLETE_ACTUAL_INPUT_AFTER_DIFFERS')
    return dict(supplied_scalars={k:v['value'] for k,v in values.items() if v['kind'] == 'scalar'},
        declared_omitted_defaults=defaults,public_default_schema=api['public_default_schema'],
        qkv_mask={k:{field:values[k]['snapshot']['metadata'][field] for field in
            ('shape','stride','storage_offset','storage_nbytes','dtype','device')} for k in ('query','key','value','atten_mask')})


def raw_scope(a, parser, rows, call, worker, owner, generation):
    c=call['native_census'];identity=c['thread_identity'];scope=c['execution_scope'];token=c['call_id']
    need(identity['source'] == 'CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID'
         and all(type(identity[k]) is int and identity[k] > 0 for k in ('pid','python_thread_ident','native_tid')),
         'EXPLICIT_THREAD_DOMAINS_REQUIRED')
    need(identity['pid'] == worker['pid'] and scope == call['native_census_scope']
         and scope['owner'] == owner and scope['generation'] == generation
         and scope['rank'] == worker['rank'] == 0 and scope['logical_device'] == worker['logical_device']
         and scope['device_type'] == 'npu'
         and all(type(scope[k]) is int and scope[k] > 0 for k in ('worker_object_id','runner_object_id','model_object_id')),
         'ACTUAL_PUBLIC_OWNER_WORKER_SCOPE_JOIN')
    for key in ('request_id','module_path','module_object_id','actual_operator','occurrence','ordinal','target_key'):
        need(c[key] == call[key], 'NATIVE_CAPTURE_PUBLIC_CALL_JOIN:'+key)
    observed=RAW.scope(a.root,rows,token,expected_pid=identity['pid'])
    scoped=[row for row in rows if row.get('call_id') == token]
    need(all((r['pid'],r['tid']) == (identity['pid'],identity['native_tid']) for r in scoped)
         and not observed['preceding_unscoped_capture_gaps'].get('capture_overflow')
         and not any(r['event'] in ('capture_overflow','unbound_function') for r in scoped)
         and all(r.get('result') == 0 for r in scoped if r['event'] in ('stream_sync','device_sync')),
         'ACTUAL_NATIVE_THREAD_CAPTURE_OR_COMPLETION_INCOMPLETE')
    profiles=[]
    for launch in observed['launches']:
        binary=launch['registered_binary'];key=launch['tiling_key'];cache_key=(binary['sha256'],key)
        name=Path(binary['path']).name;a.seen_elf.add(name)
        if a.expected_elf is not None:need(a.expected_elf.get(name) == binary['sha256'],'EXTERNALLY_PINNED_REGISTERED_ELF_DIFFERS')
        if cache_key not in a._elf:a._elf[cache_key]=RAW.entry_symbols(binary['path'],binary['sha256'],selected_keys=[key])
        symbols=a._elf[cache_key];entries=symbols.get('actual_selected_entry_symbols',[])
        need(entries and all(r['symbol_type'] == 2 and r['binding'] in (1,2) and r['section_index'] > 0
             and r['bytes'] > 0 for r in entries), 'ACTUAL_REGISTERED_ENTRY_SYMBOLS_UNOBSERVED')
        profiles.append(dict(api=launch['api'],actual_tiling_key=key,block_dim=launch['block_dim'],
            schedule_mode=launch['schedule_mode'],argument_layout=launch['argument_layout'],args_bytes=launch['args_bytes'],
            registered_binary=binary,selected_entry_symbols=entries,symbols_sha256=symbols['symbols_sha256'],
            matching_non_function_symbols=symbols['matching_non_function_symbols'],
            co_resident_internal_symbols_not_entry_reachability=symbols.get('internal_route_symbols',[]),
            native_algorithm_branch='UNRESOLVED_WITHOUT_ENTRY_CALLGRAPH_OR_MATCHING_BUILD_METADATA'))
    recorded=c.get('launches')
    if recorded is not None:
        prefix=recorded['runtime_prefix'];path=a.path(prefix['path'])
        need(path == a.root/'runtime/runtime.jsonl' and type(prefix['bytes']) is int and 0 < prefix['bytes'] <= path.stat().st_size,
             'CAPTURED_RUNTIME_PREFIX_BOUND')
        with path.open('rb') as stream:actual_prefix=stream.read(prefix['bytes'])
        need(hashlib.sha256(actual_prefix).hexdigest() == prefix['sha256'] and actual_prefix.endswith(b'\n'),
             'CAPTURED_RUNTIME_PREFIX_HASH_DIFFERS')
        need(recorded['call_id'] == token and recorded['thread_identity'] == identity
             and len(recorded['launches']) == len(observed['launches']), 'RECORDED_ACTUAL_SCOPE_DIFFERS')
        fields=('api','pid','tid','raw_line','call_id','handle','generation','tiling_key','block_dim',
                'schedule_mode','argument_layout','args_bytes','args_sha256','placeholders_bytes','placeholders_sha256')
        for old,new,profile in zip(recorded['launches'],observed['launches'],profiles):
            need(all(old[k] == new[k] for k in fields) and old['registered_binary']['sha256'] == new['registered_binary']['sha256']
                 and old['registered_binary']['bytes'] == new['registered_binary']['bytes']
                 and a.path(old['registered_binary']['path']) == Path(new['registered_binary']['path'])
                 and all(old['function_binding'][k] == new['function_binding'][k]
                    for k in ('raw_line','handle','generation','function_handle','tiling_key'))
                 and old['registered_entry_symbols']['actual_selected_entry_symbols'] ==
                    [{k:x[k] for k in ('name','symbol_type','binding','section_index','bytes','table')}
                     for x in profile['selected_entry_symbols']],
                 'RECORDED_LAUNCH_OR_ELF_DIFFERS_FROM_ORIGINAL_BYTES')
    return dict(call_id=token,request_id=call['request_id'],module_path=call['module_path'],
        original_capture_status=c['status'],original_reasons=c['reasons'],profiles=profiles,
        launches=observed['launches'],
        actual_event_counts=observed['event_counts'],native_argument_tensor_pointer_join='UNOBSERVED_NO_STOCK_ABI')


def cleanup(state):
    need(all(state[k] == 0 for k in ('inflight','leases','owned_hooks','owned_api_wrappers'))
         and not any(state[k] for k in ('enabled','native_registered','replay_pending','replay_failed')),
         'PUBLIC_CLEANUP_RESOURCE_NOT_ZERO')
    native=state['native_state']
    need(all(not native[k] for k in ('scope_active','scope_owned','environment_owned','pending_drains','pending_references'))
         and native['first_error'] is None and native['failures'] == [] and native['observer_only'] is True,
         'ACTUAL_SCOPE_DRAIN_REFERENCE_CLEANUP_INCOMPLETE')
    resource=state['serving']['resource_state']
    need(state['serving']['pending_request'] is None and state['serving']['failures'] == []
         and resource.get('failed_requests',[]) == [] and resource['failures'] == []
         and resource['pending_request'] is None and resource['needs_off'] is False, 'PUBLIC_FAILED_OR_PENDING_REQUEST_RETAINED')


def protocol(a, data):
    host=a.artifact(data['host_receipt'],limit=1024*1024**2);core=data['worker_core'];history=host['history'];refs=core['receipts']
    need(host['schema'] == 'demoncat.worker_core.host_receipt.v1' and [r['action'] for r in history] == ACTIONS
         and len(history) == len(refs), 'COMPLETE_15_ORIGINAL_RPC_ACTIONS_REQUIRED')
    for ref,row in zip(refs,history):
        need(ref['action'] == row['action'] and a.artifact(ref) == row, 'ORIGINAL_RPC_FILE_HOST_HISTORY_DIFFERS')
    worker=host['worker'];ticket=history[1]['response']['ticket'];owner=host['owner'];generation=ticket['generation']
    need(worker == data['setup']['worker'] == ticket['worker'] and ticket['owner'] == owner
         and ticket['tenant_id'] == host['tenant_id'] and ticket['mechanism'] == data['mechanism']
         and type(generation) is int and generation > 0, 'ACTUAL_HOST_TICKET_WORKER_JOIN')
    attached=history[0]['response']
    need(attached['status'] == 'ATTACHED' and all(attached['ticket'][k] == ticket[k]
         for k in ('tenant_id','attach_nonce','worker')), 'ORIGINAL_ATTACHMENT_JOIN')
    for item in history[1:-1]:
        response=item['response'];state=response['provider_state']
        need(response['status'] == ('QUERY_OK' if item['action'] == 'query' else 'ACTION_COMPLETED')
             and (item['action'] == 'query' or response.get('action') == item['action'])
             and response['ticket'] == ticket and response['recovery_required'] is False
             and state['owner'] == owner and state['generation'] == generation and state['simulation'] is False
             and all(state[k] is False for k in ('C1','C2','C3','public_qualification_granted')),
             'PUBLIC_ACTION_OWNER_OR_RECOVERY_DIFFERS:'+item['action'])
    requests=data['requests']
    for index,row in enumerate(requests):
        before,queried,collected=[r['response']['provider_state'] for r in history[5+index*3:8+index*3]]
        expected=[row['mapping']['internal_request_id'],row['phase']]
        need(before['serving']['pending_request'] == queried['serving']['pending_request'] == expected
             and before['evidence']['requests'] == [x['worker'] for x in requests[:index]]
             and queried['evidence']['requests'] == before['evidence']['requests']
             and collected['serving']['pending_request'] is None
             and collected['evidence']['requests'] == [x['worker'] for x in requests[:index+1]],
             'QUERY_PURITY_AND_ACTUAL_COLLECTED_REQUEST_JOIN')
    final=history[-4]['response']['provider_state']
    need(final == data['final_cleanup'], 'FINAL_FINISH_ORIGINAL_STATE_DIFFERS');cleanup(final)
    unregistered=history[-2]['response']['provider_state'];cleanup(unregistered)
    need(unregistered['registered'] is False and unregistered['binding_reserved'] is False
         and history[-1]['response']['status'] == 'DETACHED' and host['ticket'] is None and host['attachment'] is None
         and host['census_ticket'] is None and host['ownership_retained'] is False and core['ownership_retained'] is False,
         'PUBLIC_UNREGISTER_DETACH_NOT_CONFIRMED')
    need(data['binding_removal']['status'] == 'SERVING_DEPLOYMENT_REMOVED'
         and data['binding_removal']['mechanism'] == data['mechanism']
         and data['binding_removal']['native_scope_active'] is False, 'SAME_WORKER_BINDING_REMOVAL_UNCONFIRMED')
    health=data['post_unregister_health']
    need(health['scope'] == 'TOKENS_ONLY_NO_POST_UNREGISTER_LOGITS_OBSERVER'
         and health['output_tokens'] == data['healthy_before']
         and all(r['output_tokens'] == health['output_tokens'] for r in requests), 'SAME_WORKER_TOKEN_HEALTH_CHANGED')
    request_ids=[r['mapping']['internal_request_id'] for r in requests]+[health['mapping']['internal_request_id']]
    need(len(request_ids) == 3 and len(set(request_ids)) == 3 and all(type(x) is str and x for x in request_ids),
         'POST_UNREGISTER_ORDINARY_REQUEST_MUST_BE_NEW')
    return worker,owner,generation


def review_files(a, data, parser, *, worker, owner, generation):
    requests=data['requests'];need(len(requests) == 2 and tuple(r['phase'] for r in requests) == PHASES, 'TWO_ACTUAL_PHASES_REQUIRED')
    rows,runtime=parser.read_runtime(a.root);scopes=[];signatures=[];modules=[];ids=[];original_gaps=[];api_profiles=[];owners=[]
    if a.expected_runtime is not None:need(runtime['sha256'] == a.expected_runtime,'EXTERNALLY_PINNED_RUNTIME_DIFFERS')
    for ri,request in enumerate(requests):
        where=PHASES[ri];request_id=request['mapping']['internal_request_id'];ids.append(request_id)
        a.check(request['worker']['request_id'] == request_id and request['worker']['phase'] == where,
                'DRIVER_WORKER_REQUEST_MAPPING_DIFFERS',where)
        forwards=request['worker']['forwards'];need(len(forwards) == 1, 'ONE_PREFILL_FORWARD_REQUIRED')
        forward=forwards[0];calls=forward['calls']
        a.check(len(calls) == 28 and forward['request_id'] == request_id and forward['phase'] == where
            and forward['root_returned'] is True and forward['decode'] is False, 'COMPLETE_28_NATURAL_PREFILL_CALLS',where)
        a.attempt(where+'/outputs',lambda:need(all_tensors(forward['input']) >= 2
            and all_tensors(forward['hidden']) > 0 and all_tensors(forward['logits']) > 0, 'COMPLETE_REQUEST_HASHES_REQUIRED'))
        signatures.append({k:forward[k] for k in ('input','hidden','logits')})
        modules.append([(c.get('module_path'),c.get('module_object_id')) for c in calls])
        for ci,call in enumerate(calls):
            at=where+'/'+str(ci);c=call.get('native_census')
            a.check(call['ordinal'] == ci and call['occurrence'] == 0 and call['request_id'] == request_id
                and call['scheduler'] == forward['scheduler'] and call['original_calls'] == 1
                and call['original_returned'] is True and call['inputs_unchanged'] is True and call['native'] is None
                and call['actual_operator'] == 'torch_npu.npu_fused_infer_attention_score'
                and call['module_path'] in call['aliases'] and call['parent_path'] == call['module_path'].rpartition('.')[0],
                'ACTUAL_ORIGINAL_CALL_AND_PARENT_JOIN',at)
            a.attempt(at+'/output',lambda:need(all_tensors(call['output']) > 0, 'COMPLETE_API_OUTPUT_HASH_REQUIRED'))
            a.check(call['target_key'] == E.digest(dict(path=call['module_path'],operator=call['actual_operator'],
                occurrence=call['occurrence'],input=call['input'],decode=False)), 'RECOMPUTED_ACTUAL_CALL_KEY',at)
            if not isinstance(c,dict):a.check(False,'NATIVE_CAPTURE_ABSENT',at);continue
            owners.append({k:v for k,v in c.get('execution_scope',{}).items() if k != 'phase'})
            a.check(c.get('execution_scope',{}).get('phase') == where,'ACTUAL_NATIVE_SCOPE_PHASE_JOIN',at)
            if c.get('status') != 'OBSERVED_INTERCEPTED_STOCK_LAUNCHES':original_gaps.append(dict(where=at,reasons=c.get('reasons')))
            a.check(c.get('input_substitutions') == 0 and c.get('operator_arguments_modified') is False
                and c.get('fault_authorized') is False and c.get('native_fault_qualified') is False
                and c.get('original_calls') == 1 and c.get('original_returned') is True, 'NO_FAULT_OR_SUBSTITUTION_CALL_RECEIPT',at)
            if 'artifact' in c:a.attempt(at+'/artifact',lambda:need(a.artifact(c['artifact']) ==
                {k:v for k,v in c.items() if k != 'artifact'}, 'ACTUAL_CALL_ARTIFACT_DIFFERS'))
            else:a.check(False,'CALL_ARTIFACT_UNOBSERVED',at)
            api=a.attempt(at+'/api',lambda:api_input(call,request_id))
            if api is not None:api_profiles.append(api)
            resolved=a.attempt(at+'/raw',lambda:raw_scope(a,parser,rows,call,worker,owner,generation))
            if resolved is not None:scopes.append(resolved)
    a.check(len(set(ids)) == 2 and all(type(i) is str and i for i in ids), 'ACTUAL_DISTINCT_REQUESTS')
    a.check(modules[0] == modules[1] and len(set(modules[0])) == 28
        and len({identity for _,identity in modules[0]}) == 28, 'COMPLETE_STABLE_ACTUAL_MODULE_DENOMINATOR')
    a.check(signatures[0] == signatures[1], 'REPEATED_COMPLETE_REQUEST_INPUT_HIDDEN_LOGITS_HASHES')
    a.check(len(owners) == 56 and all(x == owners[0] for x in owners),'SAME_ACTUAL_WORKER_RUNNER_MODEL_OWNER')
    tokens=[s['call_id'] for s in scopes]
    a.check(len(tokens) == 56 and len(set(tokens)) == 56, 'ALL_56_DISTINCT_ACTUAL_SCOPES_RECONCILED')
    raw_tokens={r['call_id'] for r in rows if r.get('call_id') and r['event'] in ('begin','end','launch')}
    a.check(raw_tokens == set(tokens), 'NO_EXTRA_OR_OMITTED_OWNED_RAW_SCOPES')
    if a.expected_elf is not None:a.check(a.seen_elf == set(a.expected_elf),'EXACT_USED_REGISTERED_ELF_PIN_SET')
    profiles={E.digest(p):p for s in scopes for p in s['profiles']}
    return dict(runtime=runtime,actual_calls_per_request=[len(r['worker']['forwards'][0]['calls']) for r in requests],
        independently_resolved_scopes=len(scopes),original_unobserved_scopes=original_gaps,scopes=scopes,
        actual_launch_profiles=list(profiles.values()),api_profiles=list({E.digest(p):p for p in api_profiles}.values()),
        binding_kinds=dict(Counter(l['function_binding']['kind'] for s in scopes for l in s['launches'] if l['function_binding'])),
        actual_registered_entry_keys=sorted({p['actual_tiling_key'] for p in profiles.values()}))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','observer','out'):cli.add_argument('--'+name,type=Path,required=True)
    for name in ('result-sha256','source-manifest-sha256','runtime-sha256'):cli.add_argument('--'+name,required=True)
    cli.add_argument('--registered-elf-sha256',action='append',required=True,metavar='BASENAME=SHA256')
    cli.add_argument('--recorded-root',default='/evidence/probe');args=cli.parse_args()
    sys.dont_write_bytecode=True;sys.addaudithook(E.prohibit_runtime)
    need(not args.out.exists(), 'FRESH_REVIEW_REPORT_REQUIRED')
    elf_pins={}
    for item in args.registered_elf_sha256:
        name,separator,value=item.partition('=')
        need(separator and re.fullmatch(r'registered-\d+-\d+\.bin',name) and valid_sha(value)
             and name not in elf_pins,'EXACT_REGISTERED_ELF_PINS_REQUIRED');elf_pins[name]=value
    need(valid_sha(args.runtime_sha256),'EXACT_RUNTIME_SHA_REQUIRED')
    a=Review(args.root,recorded_root=args.recorded_root,expected_runtime=args.runtime_sha256,expected_elf=elf_pins)
    result=dict(schema='demoncat.attention28.stock_census.independent_review.v1',
        reviewer_sha256=E.sha(__file__),status='INCOMPLETE_OR_REFUSED',fault_authorized=False,C1=False,C2=False,C3=False,
        offline_raw_reader_sha256=E.sha(RAW.__file__),
        E5='NOT_ASSESSED',all_runtime_entrypoints_covered=False,stock_pod_decoded=False,
        limitations=['QKV and complete model arrays were not retained; their full hashes are source-bound receipts.',
            'Only intercepted launch scopes are reconciled; no claim covers every runtime entrypoint.',
            'Schema defaults are declarations, not observed final Native typed attributes.',
            'Registered ELF entry symbols identify the launched entry, not a decoded stock POD or selected ACLNN owner.',
            'Co-resident internal symbols do not prove entry reachability. No expected key classifies the actual branch.',
            'Post-unregister health is tokens only; no post-clean full logits are observed.'])
    def run():
        source=source_files(args.source,args.source_manifest_sha256)
        data=a.artifact(dict(path='result.json',sha256=args.result_sha256))
        need(data['mechanism'] in MECHANISMS and data['mode'] == 'ACTUAL_TND_PREFILL_STOCK_NATIVE_CENSUS_ONLY'
             and data['probe_sha256'] == source['pins'][E.EXPERIMENT+'natural_attention28_stock_census_probe.py'],
             'FROZEN_CENSUS_PROBE_OR_MECHANISM_DIFFERS')
        observer=E.observer_bundle(args.observer,source,data['observer_receipt_sha256'])
        need(data['observer_library_sha256'] == observer['receipt']['observer_library']['sha256'], 'ACTUAL_OBSERVER_LIBRARY_JOIN')
        result['provenance']=dict(source_commit=source['manifest']['base_commit'],manifest_sha256=source['manifest_sha256'],
            source_pins=source['pins'],observer_receipt_sha256=observer['sha256'],observer_library_sha256=data['observer_library_sha256'])
        a.check(not any(data[k] for k in ('forced_private_vendor','fault_authorized','public_qualification_granted','C1','C2','C3')),
                'OBSERVATION_MUST_NOT_CLAIM_FAULT_QUALIFICATION')
        result['original_status']=data.get('status');result['original_failures']={k:data[k] for k in ('error','abort_error','cleanup_errors') if k in data}
        a.check(not result['original_failures'], 'ORIGINAL_PROBE_FAILURES_PRESERVED')
        binding=a.attempt('public_protocol',lambda:protocol(a,data))
        if binding is None:
            state=data['final_cleanup'];binding=(data['setup']['worker'],state['owner'],state['generation'])
        result['census']=review_files(a,data,raw_reader(source),worker=binding[0],owner=binding[1],generation=binding[2])
    a.attempt('review',run)
    result.update(checks=a.checks,issues=a.issues,artifacts=a.artifacts)
    if not a.issues:result['status']='SOURCE_BOUND_TND_STOCK_SCOPES_RECONCILED_NO_FAULT_QUALIFICATION'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=a.checks,issues=a.issues,out=str(args.out),
        sha256=E.sha(args.out),actual_registered_entry_keys=result.get('census',{}).get('actual_registered_entry_keys'),
        independently_resolved_scopes=result.get('census',{}).get('independently_resolved_scopes'),fault_authorized=False)))
    return 0 if not a.issues else 2


if __name__ == '__main__':raise SystemExit(main())
