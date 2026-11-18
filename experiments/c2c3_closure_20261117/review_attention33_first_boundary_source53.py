"""Read-only source53 first-boundary comparison; no bridge or fault authority.

Preserve the original runtime gap and add a separately identified offline join.
No stock kernel ABI is inferred from the private adapter or its tiling key.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys

SOURCE = 'b017a43ac8cabc48b9090e9db25914739f9730b5'
MANIFEST = '8cf340e5d51bd9fbafa2437a9e9fa128ab1d178596e46e3da4f41fde9572ed36'
STOCK = 'fde2b7717c93162aeb2d4d1e3fe5b56ed1901b775d9976a574ea899e1a6b5077'
PRIVATE = '68157b6970a94498cf9a91cfd718593b6d7ddbb930cbc7f08a6a508d561a69a3'
PHASES = ('baseline_before', 'baseline_repeat', 'measured', 'off_after', 'healthy_after')
FIRST = 'model.layers.0.self_attn.attn'
WORKER = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
EXPERIMENT = 'experiments/c2c3_closure_20261117/'
SOURCE_FILES = (EXPERIMENT+'attention33_contract.py', EXPERIMENT+'attention33_bridge.py',
    EXPERIMENT+'diagnose_attention33_baselines.py', EXPERIMENT+'attention33_worker_probe.py',
    *(WORKER+name+'.py' for name in ('paged33_boundary_probe', 'paged33_tnd_probe', 'paged33_audit',
        'paged33_tnd', 'paged33_loading', 'serving_context', 'call_binding')),
    'fault_injection/native_adapters/attention28/launch_observer.cpp')
FORMATS = {'torch.float16':('e', 2), 'torch.float32':('f', 4), 'torch.float64':('d', 8),
    'torch.bfloat16':('H', 2), 'torch.int32':('i', 4), 'torch.int64':('q', 8),
    'torch.uint8':('B', 1), 'torch.int8':('b', 1), 'torch.bool':('?', 1)}
SCHEMA = 'demoncat.attention.runtime_observer.v1'


def need(value, code):
    if not value: raise ValueError(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): value.update(block)
    return value.hexdigest()


def valid_sha(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'DUPLICATE_JSON_KEY:'+key); result[key] = value
    return result


def invalid(value): raise ValueError('NONFINITE_JSON:'+value)


def read(path):
    need(Path(path).stat().st_size <= 128*1024**2, 'JSON_FILE_BOUND')
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)


def safe_path(run, recorded):
    value = PurePosixPath(recorded); prefix = PurePosixPath('/evidence/run')
    need('..' not in value.parts and value.is_relative_to(prefix), 'EVIDENCE_PATH_INVALID')
    root = Path(run).resolve(strict=True); target = root/Path(*value.relative_to(prefix).parts)
    need(not target.is_symlink() and target.resolve(strict=True).is_relative_to(root), 'EVIDENCE_PATH_ESCAPES_RUN')
    return target


def load_frozen(root):
    sys.dont_write_bytecode = True
    root = Path(root).resolve(strict=True); manifest = root/'source_manifest.json'; record = read(manifest)
    need(sha(manifest) == MANIFEST and record['base_commit'] == SOURCE, 'EXACT_SOURCE53_MANIFEST_REQUIRED')
    files = {row['path']:row for row in record['files']}; pins = {}
    need(len(files) == len(record['files']), 'DUPLICATE_SOURCE_MANIFEST_PATH')
    for name in SOURCE_FILES:
        path = root/name; pins[name] = sha(path)
        need(files[name]['sha256'] == pins[name] and files[name]['bytes'] == path.stat().st_size,
             'FROZEN_SOURCE_BYTES_CHANGED:'+name)
    modules = []
    for name, path in (('attention33_contract', EXPERIMENT+'attention33_contract.py'),
                       ('attention33_bridge', EXPERIMENT+'attention33_bridge.py'),
                       ('_source53_difference_reader', EXPERIMENT+'diagnose_attention33_baselines.py')):
        need(name not in sys.modules, 'PRELOADED_REVIEW_HELPER_REFUSED:'+name)
        spec = importlib.util.spec_from_file_location(name, root/path)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
        modules.append(module)
    return (*modules, dict(base_commit=SOURCE, manifest_sha256=sha(manifest), files_rehashed=len(pins),
        source_files_sha256=digest(pins), pins=pins))


def read_runtime(run):
    path = Path(run)/'runtime/runtime.jsonl'; before = path.stat()
    need(0 < before.st_size <= 128*1024**2, 'RUNTIME_FILE_BOUND')
    raw = path.read_bytes(); after = path.stat()
    need((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns)
        and raw.endswith(b'\n'), 'RUNTIME_NOT_A_STABLE_COMPLETE_FILE')
    rows = [json.loads(line, object_pairs_hook=unique, parse_constant=invalid) for line in raw.splitlines()]
    need(all(row.get('schema') == SCHEMA for row in rows), 'UNKNOWN_RUNTIME_SCHEMA')
    return rows, dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), records=len(rows))


def scoped_runtime(run, rows, token, *, expected_pid):
    """Resolve repeated exact binds, never discard conflicting function history."""
    need(isinstance(token, str) and re.fullmatch('[0-9a-f]{32}', token), 'NATIVE_SCOPE_TOKEN_INVALID')
    scoped = [(i,row) for i,row in enumerate(rows) if row.get('call_id') == token]
    begins = [(i,row) for i,row in scoped if row['event'] == 'begin']
    ends = [(i,row) for i,row in scoped if row['event'] == 'end']
    need(len(begins) == len(ends) == 1, 'SCOPE_BEGIN_END_AMBIGUOUS')
    begin_index, begin = begins[0]; end_index, end = ends[0]
    owner = (begin['pid'], begin['tid'])
    need(owner[0] == expected_pid and begin_index < end_index and begin['monotonic_ns'] < end['monotonic_ns'], 'SCOPE_OWNER_OR_ORDER')
    need(all((row['pid'],row['tid']) == owner and begin_index <= i <= end_index
        and begin['monotonic_ns'] <= row['monotonic_ns'] <= end['monotonic_ns'] for i,row in scoped), 'SCOPED_EVENT_OWNER_OR_ORDER')
    need(not any(row['event'] == 'unbound_function' for _,row in scoped), 'SCOPED_FUNCTION_BINDING_INCOMPLETE')
    outside_gaps = Counter(row['event'] for row in rows if row['event'] in ('capture_overflow','unbound_function')
        and row['pid'] == owner[0] and row['monotonic_ns'] <= end['monotonic_ns'])
    launches = []
    for index, row in scoped:
        if row['event'] != 'launch': continue
        need(row['result'] == 0 and row['snapshot_valid'] is True and row['observer_mutates_arguments'] is False
            and begin['monotonic_ns'] <= row['launch_begin_ns'] <= row['launch_end_ns'] <= row['monotonic_ns'] <= end['monotonic_ns'],
            'LAUNCH_FAILED_SNAPSHOT_INVALID_OR_ORDER')
        need(row['api'] in ('aclrtLaunchKernelWithHostArgs','rtKernelLaunchWithHandle','rtKernelLaunchWithHandleV2')
            and type(row['block_dim']) is int and row['block_dim'] > 0 and type(row['schedule_mode']) is int
            and row['schedule_mode'] >= 0 and str(row['tiling_key']).isdigit(), 'LAUNCH_PROFILE_NOT_OBSERVED')
        registrations = [(i,r) for i,r in enumerate(rows) if r['event'] == 'register' and r['pid'] == owner[0]
            and r['handle'] == row['handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
        need(bool(registrations), 'REGISTERED_BINARY_JOIN_MISSING')
        same = [(i,r) for i,r in registrations if r['generation'] == row['generation']]
        need(len(same) == 1, 'REGISTERED_GENERATION_AMBIGUOUS')
        reg_index, registration = same[0]
        need(max(registrations, key=lambda item:item[1]['monotonic_ns']) == same[0]
            and reg_index < index and registration['result'] == 0, 'STALE_BINARY_HANDLE_GENERATION')
        name = row['binary_file']; binary = Path(run)/'runtime'/name
        need(name == registration['binary_file'] == 'registered-'+str(owner[0])+'-'+str(row['generation'])+'.bin'
            and re.fullmatch(r'registered-\d+-\d+\.bin', name)
            and not binary.is_symlink() and binary.resolve(strict=True).parent == (Path(run)/'runtime').resolve(strict=True)
            and 0 < binary.stat().st_size == registration['binary_bytes'] <= 128*1024**2, 'REGISTERED_BINARY_FILE_CHANGED')
        binding = None
        if row['api'] == 'aclrtLaunchKernelWithHostArgs':
            need(type(row['function_handle']) is int and row['function_handle'] > 0
                and row['argument_layout'] == 'aclrt_host_placeholders', 'ACTUAL_FUNCTION_HANDLE_MISSING')
            history = [(i,r) for i,r in enumerate(rows) if r['event'] == 'function_bind' and r['pid'] == owner[0]
                and r['function_handle'] == row['function_handle'] and r['monotonic_ns'] <= row['launch_begin_ns']]
            need(bool(history), 'ACTUAL_FUNCTION_JOIN_MISSING')
            identity = (row['handle'], row['generation'], str(row['tiling_key']))
            need(all(r['result'] == 0 and (r['handle'],r['generation'],str(r['tiling_key'])) == identity
                     for _,r in history), 'CONFLICTING_FUNCTION_HANDLE_KEY_OR_GENERATION_HISTORY')
            current = [(i,r) for i,r in history if r.get('call_id') == token]
            need(len(current) == 1, 'CURRENT_SCOPE_FUNCTION_BIND_NOT_UNIQUE')
            bind_index, bound = current[0]
            newest = max(r['monotonic_ns'] for _,r in history)
            need(sum(r['monotonic_ns'] == newest for _,r in history) == 1 and bound['monotonic_ns'] == newest
                and (bound['pid'],bound['tid']) == owner and reg_index < bind_index < index
                and begin['monotonic_ns'] <= bound['monotonic_ns'] <= row['launch_begin_ns'], 'LATEST_EFFECTIVE_BIND_NOT_IN_CURRENT_SCOPE')
            binding = dict(rule='CURRENT_SCOPE_LAST_EFFECTIVE_BIND_WITH_ALL_PRIOR_FUNCTION_IDENTITIES_CONSISTENT',
                function_handle=row['function_handle'], raw_line=bind_index+1, repeated_identical_history=len(history)-1,
                current_scope_binds=1, handle=row['handle'], generation=row['generation'], tiling_key=str(row['tiling_key']))
        args = bytes.fromhex(row['args_hex']); placeholders = bytes.fromhex(row.get('placeholders_hex', ''))
        need(len(args) == row['args_size'] <= 1024**2 and len(placeholders) <= 1024**2, 'RAW_ARGUMENT_SNAPSHOT_SIZE')
        syncs = [r for i,r in scoped if i > index and r['event'] in ('device_sync','stream_sync')
            and r.get('result') == 0 and row['monotonic_ns'] <= r['monotonic_ns'] <= end['monotonic_ns']
            and (r['event'] == 'device_sync' or r.get('stream') == row['stream'])]
        need(bool(syncs), 'LAUNCH_COMPLETION_SYNC_UNOBSERVED')
        launches.append(dict(api=row['api'], pid=owner[0], tid=owner[1], raw_line=index+1,
            call_id=token, handle=row['handle'], generation=row['generation'], function_binding=binding,
            tiling_key=str(row['tiling_key']), block_dim=row['block_dim'], schedule_mode=row['schedule_mode'],
            argument_layout=row['argument_layout'], args_bytes=len(args), args_sha256=hashlib.sha256(args).hexdigest(),
            placeholders_bytes=len(placeholders), placeholders_sha256=hashlib.sha256(placeholders).hexdigest(),
            registered_binary=dict(path=str(binary), bytes=binary.stat().st_size, sha256=sha(binary)),
            kernel_argument_abi='NOT_DECODED', selected_aclnn_library_owner='NOT_OBSERVED'))
    need(0 < len(launches) <= 64, 'NO_OR_UNBOUNDED_INTERCEPTED_LAUNCHES')
    return dict(status='NEW_OFFLINE_RESOLUTION_FROM_PRESERVED_RAW_EVENTS', call_id=token,
        event_counts=dict(sorted(Counter(row['event'] for _,row in scoped).items())), launches=launches,
        preceding_unscoped_capture_gaps=dict(outside_gaps), intercepted_launch_joins_complete=True,
        original_runtime_observation_rewritten=False, stock_private_abi_assumption=False,
        selected_aclnn_owner_observed=False, all_runtime_entrypoints_covered=False)


def metadata(meta):
    return {key:meta[key] for key in ('shape','stride','storage_offset','dtype','device')}


def snapshot_signature(snapshot):
    return dict(metadata=metadata(snapshot['metadata']), bytes=snapshot['bytes'], sha256=snapshot['sha256'])


def validate_snapshot(snapshot):
    meta = snapshot['metadata']; shape = meta['shape']; dtype = meta['dtype']
    need(dtype in FORMATS and all(type(n) is int and n >= 0 for n in shape)
        and len(shape) == len(meta['stride']) and all(type(n) is int and n >= 0 for n in meta['stride'])
        and type(meta['storage_offset']) is int and meta['storage_offset'] >= 0, 'SNAPSHOT_GEOMETRY_INVALID')
    size = math.prod(shape)*FORMATS[dtype][1]
    need(snapshot['bytes'] == size <= 128*1024**2 and valid_sha(snapshot['sha256']) and snapshot['finite'] is True,
         'SNAPSHOT_BYTES_HASH_OR_FINITE_INVALID')
    need(all(type(meta[k]) is int and meta[k] > 0 for k in ('object_id','storage_identity'))
        and meta['storage_nbytes'] >= size and str(meta['device']).split(':')[0] in ('npu','cpu'), 'SNAPSHOT_STORAGE_OR_DEVICE_INVALID')
    if size:
        last = meta['storage_offset']+sum((n-1)*s for n,s in zip(shape,meta['stride']))+1
        need(last*FORMATS[dtype][1] <= meta['storage_nbytes'], 'SNAPSHOT_VIEW_OUTSIDE_STORAGE')


def raw_snapshot(run, snapshot):
    validate_snapshot(snapshot)
    path = safe_path(run, snapshot['path']); need(path.stat().st_size == snapshot['bytes'], 'RAW_SNAPSHOT_SIZE_CHANGED')
    import numpy as np
    formats = {'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8','torch.bfloat16':'<u2'}
    need(snapshot['metadata']['dtype'] in formats, 'RAW_FLOAT_SNAPSHOT_TYPE_UNADAPTED')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block); values = np.frombuffer(block, dtype=formats[snapshot['metadata']['dtype']])
            finite = (values & 0x7f80 != 0x7f80).all() if snapshot['metadata']['dtype'] == 'torch.bfloat16' else np.isfinite(values).all()
            need(bool(finite), 'RAW_SNAPSHOT_NONFINITE')
    need(h.hexdigest() == snapshot['sha256'], 'RAW_SNAPSHOT_HASH_CHANGED')
    return dict(path=str(path), shape=snapshot['metadata']['shape'], dtype=snapshot['metadata']['dtype'],
        bytes=snapshot['bytes'], sha256=h.hexdigest())


def same_snapshot(left, right):
    return all(left.get(k) == right.get(k) for k in ('metadata','version','bytes','sha256','finite'))


def first_boundary(run, phase, frozen_row, *, difference_reader):
    capture = phase['finish']['capture']; value = capture['first_decode_boundary']; bound = value['boundary']
    name = phase['phase']; prefix = '/evidence/run/'+name+'/first_boundary/'
    need(read(Path(run)/name/'first_boundary/boundary.json') == value, 'BOUNDARY_FILE_DIFFERS_FROM_RESULT')
    need(value['schema'] == 'demoncat.paged33.first_decode_boundary.v1' and value['state'] == 'FINALIZED'
        and value['execution'] == 'NATURAL_WORKER_DIAGNOSTIC' and value['errors'] == [] and value['original_calls'] == 1
        and value['input_substitutions'] == 0 and value['operator_arguments_modified'] is False
        and all(value[k] is False for k in ('fault_enabled','c2','c3')) and value['rank'] == 0, 'FIRST_BOUNDARY_SCOPE_OR_ORIGINAL_ONCE')
    need(bound['module_path'] == FIRST and bound['metadata_binding'] == 'VERIFIED'
        and all(bound[k] == frozen_row[k] for k in ('module_path','module_object_id','impl_object_id','query','key','value',
            'page_table','live_key_cache','live_value_cache','used_pages','slots','sequence_lengths','scheduler','writer')),
        'FIRST_BOUNDARY_ACTUAL_CALL_JOIN')
    inputs = value['inputs']; need(sorted(inputs) == value['supplied_argument_names'] and 0 <= value['positional_count'] <= 3,
        'COMPLETE_ACTUAL_ARGUMENT_SET')
    need({'query','key','value','atten_mask','block_table'}.issubset(inputs), 'ACTUAL_QKV_MASK_TABLE_REQUIRED')
    need(value['original_callable']['type'] == 'torch._ops.OpOverloadPacket'
        and type(value['original_callable']['object_id']) is int and value['original_callable']['object_id'] > 0
        and isinstance(value['public_default_schema'], str) and value['public_default_schema']
        and value['default_schema_dispatch_observed'] is False and isinstance(value['omitted_argument_defaults'], dict),
        'ACTUAL_PUBLIC_PACKET_AND_DEFAULT_SCHEMA')
    need(not set(value['omitted_argument_defaults']).intersection(inputs), 'SUPPLIED_AND_OMITTED_ARGUMENT_OVERLAP')
    signature = {}; tensor_names = set(); hash_only = []
    for label, item in inputs.items():
        if item['kind'] == 'tensor':
            tensor_names.add(label); before = item['snapshot']; validate_snapshot(before)
            after = value['inputs_after'][label]; validate_snapshot(after)
            need(same_snapshot(before, after), 'INPUT_CHECKPOINT_CHANGED:'+label)
            signature[label] = dict(kind='tensor', snapshot=snapshot_signature(before))
            if before['path'] is None: hash_only.append(label)
        elif item['kind'] == 'paged_cache':
            need(label in ('key','value') and item['metadata'] == bound[label], 'ACTUAL_FLATTENED_CACHE_IDENTITY')
            signature[label] = dict(kind='paged_cache', metadata=metadata(item['metadata']))
        else:
            need(item['kind'] == 'scalar', 'UNKNOWN_ACTUAL_ARGUMENT_KIND')
            digest(item['value']); signature[label] = item
    need(set(value['inputs_after']) == tensor_names, 'AFTER_INPUT_SNAPSHOT_SET_INCOMPLETE')
    need(inputs['query']['snapshot']['metadata'] == bound['query']
        and inputs['block_table']['snapshot']['metadata'] == bound['page_table'], 'Q_TABLE_ACTUAL_OBJECT_BINDING')
    query = inputs['query']['snapshot']; need(query['path'] == prefix+'query.bin', 'QUERY_BIN_PATH_CHANGED')
    query_raw = raw_snapshot(run, query)
    need(query_raw['shape'] == [1,16,128] and query_raw['dtype'] == 'torch.float16'
        and query['metadata']['device'].split(':')[0] == 'npu', 'ACTUAL_QUERY_GEOMETRY')
    table = inputs['block_table']['snapshot']; dtype = table['metadata']['dtype']; values = value['block_table_values']
    need(dtype in ('torch.int32','torch.int64') and len(values) == 1
        and [len(values),len(values[0])] == table['metadata']['shape'] == [1,80]
        and all(type(n) is int for n in values[0]), 'ACTUAL_BLOCK_TABLE_VALUES')
    raw = struct.pack('<'+FORMATS[dtype][0]*len(values[0]), *values[0])
    need(len(raw) == table['bytes'] and hashlib.sha256(raw).hexdigest() == table['sha256']
        and [values[0][:65]] == bound['used_pages'], 'BLOCK_TABLE_FULL_HASH_OR_USED_PAGES_CHANGED')
    kv = {}
    for kind in ('key','value'):
        before, after = value['logical_kv_before'][kind], value['logical_kv_after'][kind]
        live = bound['live_'+kind+'_cache']; post = capture['attention'][FIRST][kind]
        need(before['metadata'] == live and same_snapshot(before,after) and before['shape'] == [1025,8,128]
            and before['bytes'] == 2099200 and before['finite'] is True and before['version'] == inputs[kind]['version']
            and before['physical_pages'] == after['physical_pages'] == post['physical_pages'] == bound['used_pages'][0]
            and before['readback'] == after['readback'] == 'ACTUAL_VALID_LOGICAL_PAGE_ORDER'
            and before['operator_input_repacked'] is after['operator_input_repacked'] is False
            and all(before[k] == post[k] for k in ('shape','bytes','sha256'))
            and live['dtype'] == post['dtype'] == inputs[kind]['metadata']['dtype'], 'PRE_POST_LOGICAL_KV_LIFETIME_OR_HASH:'+kind)
        post_path = safe_path(run, post['path'])
        need(sha(post_path) == before['sha256'] and post_path.stat().st_size == before['bytes'], 'LOGICAL_KV_POST_FILE_HASH:'+kind)
        kv[kind] = dict(shape=before['shape'], dtype=live['dtype'], bytes=before['bytes'], sha256=before['sha256'],
            post_consumer_file=str(post_path), pre_boundary_file_preserved=False, matching_post_file_rehashed=True)
    outputs = value['outputs']; need(0 < len(outputs) == len(value['outputs_after']) <= 16
        and value['return_container_type'] in ('tuple','list','Tensor'), 'COMPLETE_IMMEDIATE_OUTPUT_SET')
    output_raw = []
    for index, (before,after) in enumerate(zip(outputs,value['outputs_after'])):
        need(before['path'] == prefix+'output-'+str(index)+'.bin' and same_snapshot(before,after), 'IMMEDIATE_OUTPUT_CHECKPOINT_CHANGED')
        output_raw.append(raw_snapshot(run,before))
    first_output = capture['attention'][FIRST]['output']
    need(all(output_raw[0][k] == first_output[k] for k in ('shape','dtype','bytes','sha256')), 'IMMEDIATE_OUTPUT_DIFFERS_FROM_LATER_RETAINED_OUTPUT')
    expected_checks = set(inputs) | {'logical_key','logical_value','impl_key_same_object','impl_value_same_object'} | {'output_'+str(i) for i in range(len(outputs))}
    need(set(value['subsequent_checks']) == expected_checks and all(item is True for item in value['subsequent_checks'].values())
        and value['all_checked_values_unchanged'] is True, 'POST_CALL_CHECKPOINT_SET_OR_RESULT')
    logical_signature = {kind:{key:row[key] for key in ('shape','dtype','bytes','sha256')} for kind,row in kv.items()}
    signature.update(logical_kv=logical_signature, block_table_values=values,
        public_default_schema=value['public_default_schema'], omitted_argument_defaults=value['omitted_argument_defaults'],
        positional_count=value['positional_count'])
    signature = json.loads(json.dumps(signature, allow_nan=False))
    return dict(query=query_raw, outputs=output_raw, input_signature=signature, input_sha256=digest(signature),
        summary=dict(phase=name, query=query_raw, logical_kv=kv, immediate_outputs=output_raw,
            complete_input_signature_sha256=digest(signature), scalar_optional_arguments_sha256=digest({k:v for k,v in inputs.items() if v['kind']=='scalar'}),
            hash_only_input_tensors={k:inputs[k]['snapshot']['sha256'] for k in hash_only},
            block_table_full_values_rehashed=True, block_table_sha256=table['sha256'],
            checkpoint_checks=len(expected_checks), original_runtime_observation=value['runtime_observation'],
            mapped_api_libraries=value.get('mapped_api_libraries'), original_boundary_json_sha256=sha(Path(run)/name/'first_boundary/boundary.json'),
            checkpoint_equality=True, intervening_write_absence_proven=False,
            scalar_after_values_independently_recomputed=False, mask_raw_bytes_rehashed=False))


def difference(reader, left, right):
    if left['bytes'] == right['bytes'] == 0:
        need(left['shape'] == right['shape'] and left['dtype'] == right['dtype']
            and left['sha256'] == right['sha256'] == hashlib.sha256(b'').hexdigest(), 'EMPTY_OUTPUT_DIFFERS')
        return dict(shape=left['shape'],dtype=left['dtype'],bytes=0,exact=True,changed_bytes=0,
            changed_elements=0,numeric_changed_elements=0,max_abs_difference=0.0,
            left_path=left['path'],right_path=right['path'],left_sha256=left['sha256'],right_sha256=right['sha256'])
    return reader.raw_difference(left,right)


def payload_comparison(left, right):
    need(set(left) == set(right) and len(left) == 87, 'COMPLETE_EXISTING_PAYLOAD_SET')
    changed = [name for name in left if any(left[name][k] != right[name][k] for k in ('shape','dtype','bytes','sha256'))]
    layers = sorted({name.rsplit('/',1)[0] for name in changed if name.endswith('/output')}, key=lambda name:int(name.split('.')[2]))
    return dict(pairs=87, exact=87-len(changed), different=len(changed), first_attention_output_difference=layers[0] if layers else None,
                root_hidden_equal='hidden' not in changed, full_logits_equal='logits' not in changed)


def private_host_controls(run, phase, calls, first_launch):
    path = Path(run)/phase/'native/host.jsonl'
    need(0 < path.stat().st_size <= 128*1024**2, 'HOST_CALLBACK_FILE_BOUND')
    hosts = [json.loads(line,object_pairs_hook=unique,parse_constant=invalid) for line in path.read_bytes().splitlines()]
    need(len(hosts) == len(calls) == 28 and len({row['token'] for row in hosts}) == 28
        and {row['token'] for row in hosts} == {row['call_id'] for row in calls}, 'PRIVATE_HOST_CALLBACK_CENSUS')
    need(all(row['schema'] == 'demoncat.paged33.fai_fd_host.v1' and row['eligible'] is True
        and row['policy_applied'] is False and row['mode'] == 'off' and row['native_s2'] == row['selected_s2'] == 512
        for row in hosts), 'PRIVATE_HOST_CONTROL_WAS_NOT_UNMODIFIED')
    first, = [row for row in hosts if row['token'] == first_launch['call_id']]
    need((first['pid'],first['tid']) == (first_launch['pid'],first_launch['tid']), 'PRIVATE_HOST_RUNTIME_OWNER_JOIN')
    return dict(path=str(path),sha256=sha(path),callbacks=28,policy_applied=0,
        first_runtime_owner_joined=True,all_112_private_abi_launches_reverified=False)


def review_arm(run, expected_sha, kind, contract, bridge, reader, source):
    run = Path(run).resolve(strict=True); result = run/'result.json'
    need(sha(result) == expected_sha, 'PINNED_RESULT_SHA256_DIFFERS:'+kind)
    document = read(result)
    need(document['schema'] == 'demoncat.paged33.first_boundary_worker_run.v1'
        and document['status'] == 'FIRST_BOUNDARY_DIAGNOSTIC_COMPLETE_REQUIRES_INDEPENDENT_REVIEW'
        and document['phase'] == kind and document['diagnostic_only'] is True and document['statistics'] is False
        and all(document[k] is False for k in ('bridge_compatible','fault_authorized','public_worker_qualified','c2','c3'))
        and [row['phase'] for row in document['phases']] == list(PHASES), 'SOURCE53_DIAGNOSTIC_SCOPE')
    execution = document['execution_contract']; pins = source['pins']
    need(execution['driver_sha256'] == pins[EXPERIMENT+'attention33_worker_probe.py']
        and execution['contract_source_sha256'] == pins[EXPERIMENT+'attention33_contract.py'], 'ACTUAL_DRIVER_SOURCE_CHANGED')
    for module in ('paged33_boundary_probe','paged33_tnd_probe','paged33_audit','serving_context','call_binding'):
        path = 'runtime/vllm_ascend/worker_extension/'+module+'.py'
        need(execution['installed_files']['demoncat'][path]['sha256'] == pins['src/demoncat/'+path], 'INSTALLED_CAPTURE_SOURCE_CHANGED:'+module)
    expected_runner = dict(contract.RUNNER, gpu_memory_utilization=execution['runner']['gpu_memory_utilization'])
    need(execution['runner'] == expected_runner and 0 < expected_runner['gpu_memory_utilization'] <= .8
        and execution['sampling'] == dict(temperature=0,max_tokens=2,ignore_eos=True)
        and execution['prompt_tokens'] == 1024, 'ACTUAL_STANDARD_WORKER_EXECUTION_CONTRACT')
    tokens = document['warm_healthy']['prompt_token_ids']
    need(len(tokens) == document['prompt']['token_count'] == 1024
        and hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest() == document['prompt']['token_sha256'], 'PROMPT_HASH')
    runtime_rows, runtime_file = read_runtime(run)
    summaries = []; boundaries = {}; payloads = {}; identities = []; request_ids = []; native_tokens = []
    for phase in document['phases']:
        name = phase['phase']; finish = phase['finish']; capture = finish['capture']; cleanup = finish['cleanup']
        need(read(run/name/'runtime_result.json') == capture, 'RUNTIME_RESULT_COPY_CHANGED')
        need(phase['response']['prompt_token_ids'] == tokens and phase['response']['outputs'] == document['warm_healthy']['outputs']
            and phase['response']['request_id'] == phase['mapping']['external_request_id']
            and phase['mapping']['internal_request_id'] == phase['expected']['actual_internal_request_id'], 'ACTUAL_REQUEST_MAPPING')
        request_ids.append(phase['mapping']['internal_request_id'])
        need(finish['closed'] is True and not finish.get('error') and not finish.get('cleanup_error')
            and cleanup['scope_cleared'] is True and cleanup['local_streams_drained'] is True
            and cleanup['owned_hooks'] == cleanup['owned_api_wrappers'] == 0 and cleanup['boundary_refs_released'] is True
            and cleanup['core_methods_replaced'] is False and not cleanup.get('failures'), 'OWNER_OR_RESOURCE_CLEANUP_INCOMPLETE')
        need(phase['setup']['candidate_count'] == 28 and phase['setup']['source'] == capture['audit']['source']
            and phase['setup']['stock_read_only'] is (kind == 'stock' or name == 'healthy_after')
            and phase['setup']['module_sha256'] == pins[WORKER+'paged33_tnd_probe.py']
            and phase['setup']['first_boundary_diagnostic'] is True, 'ACTUAL_CAPTURE_SOURCE_OR_SCOPE')
        payloads[name] = bridge.payload_rows(run,capture)
        rows = bridge.boundary_rows(capture,phase,tokens)
        boundary = first_boundary(run,phase,rows[FIRST],difference_reader=reader); boundaries[name] = boundary
        saved = capture['first_decode_boundary']; token = saved['native_call_id']; native_tokens.append(token)
        identities.append(tuple(saved[k] for k in ('pid','thread','worker_object_id','model_object_id','rank')))
        resolution = scoped_runtime(run,runtime_rows,token,expected_pid=saved['pid'])
        original_gap = saved['runtime_observation']
        resolution['original_gap_resolution'] = ('DUPLICATE_EXACT_HISTORY_RECONCILED_OFFLINE'
            if original_gap.get('status') == 'UNOBSERVED_OR_INCOMPLETE'
            and 'BOUNDARY_ACTUAL_FUNCTION_JOIN_MISSING' in original_gap.get('error','')
            and all(row['function_binding'] and row['function_binding']['repeated_identical_history'] > 0 for row in resolution['launches'])
            else 'ORIGINAL_OBSERVATION_RETAINED_WITHOUT_RECLASSIFICATION')
        if kind == 'stock' or name == 'healthy_after':
            need(capture['native_calls'] == [] and cleanup['stock_launch_scope']['scope_cleared'] is True
                and cleanup['stock_launch_scope']['local_streams_drained'] is True
                and cleanup['stock_launch_scope']['errors'] == [], 'STOCK_SCOPE_OR_NATIVE_POLICY_PRESENT')
        else:
            calls = capture['native_calls']; need(len(calls) == 28, 'PRIVATE_NATIVE_CALL_CENSUS')
            first = [row for row in calls if row['module_path'] == FIRST]
            need(len(first) == 1 and first[0]['call_id'] == token and first[0]['module_object_id'] == rows[FIRST]['module_object_id']
                and first[0]['scheduler'] == rows[FIRST]['scheduler']
                and all(first[0][k] is False for k in ('cache_repacked','api_arguments_changed','output_mutated')), 'PRIVATE_FIRST_CALL_IDENTITY')
            need(len(resolution['launches']) == 1 and all(str(first[0]['native'][k]) == str(resolution['launches'][0][k])
                for k in ('tiling_key','block_dim','schedule_mode','args_sha256'))
                and first[0]['native']['kernel_sha256'] == resolution['launches'][0]['registered_binary']['sha256'], 'PRIVATE_RECORDED_FIRST_LAUNCH_DIFFERS_FROM_RAW')
            boundary['summary']['private_host_baseline_controls'] = private_host_controls(run,name,calls,resolution['launches'][0])
        boundary['summary']['new_offline_runtime_resolution'] = resolution
        boundary['summary']['existing_complete_payloads_rehashed'] = len(payloads[name])
        summaries.append(boundary['summary'])
    need(len(set(identities)) == 1 and len(set(request_ids)) == len(set(native_tokens)) == 5, 'SAME_WORKER_OR_UNIQUE_REQUEST_SCOPE')
    need(document['healthy_after_owned_observers_removed']['outputs'] == document['warm_healthy']['outputs'], 'POST_CLEAN_HEALTHY_GENERATION')
    within = {}
    for name in PHASES[1:]:
        first, current = boundaries[PHASES[0]], boundaries[name]
        need(len(first['outputs']) == len(current['outputs']), 'REPEATED_OUTPUT_STRUCTURE_CHANGED')
        within[name] = dict(complete_input_signature_equal=first['input_signature'] == current['input_signature'],
            query=difference(reader,first['query'],current['query']),
            immediate_outputs=[difference(reader,a,b) for a,b in zip(first['outputs'],current['outputs'])],
            all_existing_payloads=payload_comparison(payloads[PHASES[0]],payloads[name]))
    return dict(document=document,boundaries=boundaries,payloads=payloads,
        summary=dict(result_path=str(result),result_sha256=sha(result),phase=kind,runtime_file=runtime_file,
            phase_summaries=summaries,within_run=within,
            repeated_inputs_and_outputs_exact=all(item['complete_input_signature_equal'] and item['query']['exact']
                and all(row['exact'] for row in item['immediate_outputs']) and item['all_existing_payloads']['different']==0 for item in within.values()),
            checkpoint_and_cleanup_records_reconciled=True, selected_aclnn_owner_observed=False))


def prohibit_runtime(event,args):
    if event in ('ctypes.dlopen','subprocess.Popen','os.system','socket.connect'): raise RuntimeError('CPU_READ_ONLY_ACTION_REFUSED:'+event)
    if event == 'import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
        raise RuntimeError('DEVICE_FRAMEWORK_IMPORT_REFUSED:'+args[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','stock-run','baseline-run'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    import numpy  # CPU byte interpretation initializes before denying future dynamic-library loads.
    sys.addaudithook(prohibit_runtime)
    result = dict(schema='demoncat.paged33.source53.first_boundary_offline_review.v1', reviewer_sha256=sha(__file__),
        bridge_pass=False,fault_authorized=False,c2=False,c3=False,e5=False,public_qualification_granted=False)
    try:
        contract, bridge, reader, source = load_frozen(args.source_root)
        stock = review_arm(args.stock_run,STOCK,'stock',contract,bridge,reader,source)
        private = review_arm(args.baseline_run,PRIVATE,'baseline',contract,bridge,reader,source)
        need(stock['document']['execution_contract'] == private['document']['execution_contract']
            and stock['document']['prompt'] == private['document']['prompt']
            and stock['document']['observer_receipt_sha256'] == private['document']['observer_receipt_sha256'], 'PAIRED_EXECUTION_INPUT_OR_OBSERVER_DIFFERS')
        cross = {}
        for name in PHASES:
            left, right = stock['boundaries'][name], private['boundaries'][name]
            need(len(left['outputs']) == len(right['outputs']), 'PAIRED_OUTPUT_STRUCTURE_CHANGED')
            cross[name] = dict(complete_input_signature_equal=left['input_signature'] == right['input_signature'],
                query=difference(reader,left['query'],right['query']),
                immediate_outputs=[difference(reader,a,b) for a,b in zip(left['outputs'],right['outputs'])],
                logical_kv_before_hash_equal=all(left['summary']['logical_kv'][k]['sha256'] == right['summary']['logical_kv'][k]['sha256'] for k in ('key','value')),
                all_existing_payloads=payload_comparison(stock['payloads'][name],private['payloads'][name]))
        source.pop('pins'); result.update(status='DIAGNOSTIC_RECONCILED_BRIDGE_REMAINS_UNAUTHORIZED',source=source,
            stock=stock['summary'],private=private['summary'],cross_run=cross,
            first_observed_difference=('IMMEDIATE_LAYER0_ORIGINAL_API_RETURN' if all(row['complete_input_signature_equal']
                and row['query']['exact'] and any(not out['exact'] for out in row['immediate_outputs']) for row in cross.values()) else 'SEE_RECONCILED_INPUT_AND_OUTPUT_DIFFERENCES'),
            mechanism_cause_proven=False,
            limits=['Original runtime_observation gaps are preserved; the same-scope function binding is a new offline resolution.',
                'Logical K/V before the call have hashes only; matching preserved post-consumer bytes are rehashed, never relabeled as pre-call files.',
                'Mask/optional tensor values have complete hashes and checkpoint metadata but no raw files; scalar after-values are source-bound check results.',
                'Snapshot equality cannot exclude intervening writes followed by restoration when version counters are unavailable.',
                'Registered binary bytes identify intercepted launches, not the selected ACLNN API library owner or every runtime entrypoint.',
                'Private healthy_after retains the private vendor process; no stock-library recovery is claimed.',
                'Raw arguments are hashed without applying the private ABI to stock; no E5/statistics or fault/C2/C3 qualification is performed.'])
    except Exception as error:
        result.update(status='DIAGNOSTIC_FAILED_WITH_EVIDENCE_GAP',error=type(error).__name__+': '+str(error)[:400])
    print(json.dumps(result,separators=(',',':'),allow_nan=False))
    return 2 if result['status']=='DIAGNOSTIC_FAILED_WITH_EVIDENCE_GAP' else 0


if __name__ == '__main__': raise SystemExit(main())
