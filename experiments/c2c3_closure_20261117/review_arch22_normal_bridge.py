"""Read preserved stock/private normal evidence; never authorize a fault.

The old FD certificate format is deliberately not produced. A successful
normal comparison is source-bound evidence for a separately reviewed next
step, not C1/C2/C3, statistics localization or a production admission token.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import arch22_normal_bridge_evidence as evidence

need = evidence.need
FIRST = 'model.layers.0.self_attn.attn'
LAYERS = [f'model.layers.{i}.self_attn.attn' for i in range(28)]
PHASES = ('baseline_before', 'baseline_repeat', 'measured', 'off_after', 'healthy_after')


class Checks:
    """Keep independent failures instead of stopping at the first bad phase."""
    def __init__(self): self.checks, self.issues = 0, []

    def run(self, path, action):
        self.checks += 1
        try: return action()
        except Exception as error:
            self.issues.append(dict(path=path, error=repr(error)))
            return None

    def require(self, path, value, reason):
        return self.run(path, lambda: need(value, reason))


def original_failures(value, path='$'):
    result = []
    if isinstance(value, dict):
        for name, item in value.items():
            # The observer's known ambiguous-history gap remains separately
            # visible in each boundary, even if the new raw join resolves it.
            if name == 'runtime_observation': continue
            here = path+'.'+name
            if name in ('error', 'errors', 'failure', 'failures', 'first_error',
                        'traceback', 'cleanup_error', 'cleanup_traceback', 'abort_error') and item:
                result.append(dict(path=here, original=item))
            elif name not in ('raw_hex', 'args_hex', 'typed'):
                result.extend(original_failures(item, here))
    elif isinstance(value, list):
        for index, item in enumerate(value): result.extend(original_failures(item, f'{path}[{index}]'))
    return result


def literal_constants(source, filename, names):
    """Read constants without importing a module that can load a DSO."""
    tree = ast.parse((source['root']/evidence.WORKER/filename).read_bytes())
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names: result[name] = ast.literal_eval(node.value)
    need(set(result) == set(names), 'FROZEN_CONSTANT_SET_DIFFERS')
    return result


def check_execution(document, kind, source, helper, bundle, observer):
    need(document['schema'] == 'demoncat.paged33.first_boundary_worker_run.v1'
        and document['phase'] == kind and document['diagnostic_only'] is True
        and document['statistics'] is False and document['normal_bridge_variant'] == evidence.VARIANT
        and document['normal_route_preference_changed'] is (kind == 'baseline')
        and all(document[k] is False for k in ('bridge_compatible', 'fault_authorized',
            'stock_private_exact_proven', 'public_worker_qualified', 'c2', 'c3')), 'NORMAL_ONLY_RUN_SCOPE')
    need(document['observer_receipt_sha256'] == observer['sha256'], 'FRESH_SHARED_OBSERVER_RECEIPT_DIFFERS')
    if kind == 'baseline':
        need(document['completion_sha256'] == bundle['completion_sha256']
            and document['schema_sha256'] == bundle['summary']['schema_sha256'], 'FRESH_NORMAL_BUILD_OR_SCHEMA_DIFFERS')
    else:
        need(document['completion_sha256'] is None and document['schema_sha256'] is None,
             'STOCK_ARM_HAS_PRIVATE_BUILD')
    execution = document['execution_contract']; pins = source['pins']
    need(execution['schema'] == 'demoncat.paged33.execution_contract.v1'
        and execution['driver_sha256'] == pins[evidence.EXPERIMENT+'attention33_worker_probe.py']
        and execution['contract_source_sha256'] == pins[evidence.EXPERIMENT+'attention33_contract.py'],
        'FROZEN_DRIVER_OR_CONTRACT_DIFFERS')
    for relative, expected in pins.items():
        if relative.startswith('src/demoncat/'):
            actual = execution['installed_files']['demoncat'][relative.removeprefix('src/demoncat/')]
        elif relative.startswith('fault_injection/'):
            actual = execution['installed_files']['fault_injection'][relative.removeprefix('fault_injection/')]
        else: continue
        need(actual['sha256'] == expected, 'INSTALLED_FROZEN_CAPTURE_SOURCE_DIFFERS:'+relative)
    expected = dict(helper.contract.RUNNER, gpu_memory_utilization=.35)
    need(execution['runner'] == expected and execution['sampling'] == dict(temperature=0, max_tokens=2, ignore_eos=True)
        and execution['prompt_tokens'] == 1024, 'STANDARD_WORKER_EXECUTION_CONTRACT_DIFFERS')
    card = execution['environment']['DCAT33_REVIEWED_PHYSICAL_CARD']
    need(card in ('2', '5') and execution['environment'] ==
         {key:helper.preparer.case_environment(int(card))[key] for key in helper.contract.ENVIRONMENT}, 'REVIEWED_RUNTIME_ENVIRONMENT_DIFFERS')
    need(execution['model_layout'] == dict(model_type='qwen3', hidden_size=1024, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128, vocab_size=151936)
        and 'config.json' in execution['model_files']
        and any(name.endswith('.safetensors') for name in execution['model_files']), 'ORIGINAL_MODEL_INVENTORY_MISSING')
    return execution


def check_cleanup(finish, private):
    cleanup = finish['cleanup']
    need(finish.get('closed') is True and not finish.get('error') and not finish.get('cleanup_error')
        and cleanup['scope_cleared'] is True and cleanup['local_streams_drained'] is True
        and cleanup['owned_hooks'] == cleanup['owned_api_wrappers'] == 0
        and cleanup['boundary_refs_released'] is True and cleanup['core_methods_replaced'] is False
        and not cleanup.get('failures'), 'OWNER_OR_RESOURCE_CLEANUP_INCOMPLETE')
    if private:
        need(cleanup['scope_active'] is cleanup['scope_owned'] is False and cleanup['environment_owned'] == 0
            and cleanup['first_error'] is None and cleanup['calls'] == 28 and cleanup['simulation'] is False,
             'PRIVATE_NATIVE_SCOPE_OR_ERROR_REMAINS')
    else:
        stock = cleanup['stock_launch_scope']
        need(stock['scope_cleared'] is True and stock['local_streams_drained'] is True and stock['errors'] == [],
             'STOCK_OBSERVER_SCOPE_REMAINS')
    return dict(owned_hooks=0, owned_api_wrappers=0, boundary_refs_released=True,
                scope_cleared=True, local_streams_drained=True)


def payloads_for_phase(run, name, capture, helper):
    prefix='/evidence/run/'+name+'/private/'
    for label,filename in (('hidden','hidden.bin'),('logits_hidden','logits-hidden.bin'),('logits','logits.bin')):
        need(capture[label]['path'] == prefix+filename, 'RAW_PAYLOAD_BELONGS_TO_DIFFERENT_PHASE:'+label)
    for index,path in enumerate(sorted(LAYERS)):
        for kind in ('output','key','value'):
            need(capture['attention'][path][kind]['path'] == prefix+f'layer-{index}-{kind}.bin',
                 'RAW_LAYER_PAYLOAD_BELONGS_TO_DIFFERENT_TARGET_OR_PHASE:'+path)
    return helper.bridge.payload_rows(run,capture)


def check_actual_inputs(saved):
    inputs = saved['inputs']; defaults = saved['omitted_argument_defaults']
    required = dict(input_layout='TND', block_size=16, sparse_mode=3, num_heads=16,
        num_key_value_heads=8, scale=128**-.5, actual_seq_lengths=[1], actual_seq_lengths_kv=[1025])
    optional = ('pse_shift dequant_scale1 quant_scale1 dequant_scale2 quant_scale2 quant_offset2 '
        'antiquant_scale antiquant_offset query_padding_size kv_padding_size key_antiquant_scale '
        'key_antiquant_offset value_antiquant_scale value_antiquant_offset key_shared_prefix '
        'value_shared_prefix actual_shared_prefix_len query_rope key_rope key_rope_antiquant_scale '
        'dequant_scale_query learnable_sink q_start_idx kv_start_idx').split()
    declared = dict(inner_precise=0, pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0,
        query_quant_mode=0, pse_type=1, **{name:None for name in optional})
    need(set(inputs) <= set(required)|set(declared)|{'query','key','value','atten_mask','block_table'},
         'UNKNOWN_ACTUAL_ARGUMENT_NOT_IGNORED')
    for name, expected in required.items():
        item=inputs.get(name)
        if name == 'scale':
            need(isinstance(item,dict) and item.get('kind') == 'scalar'
                and type(item.get('value')) in (int,float)
                and math.isclose(item['value'],expected,rel_tol=0,abs_tol=1e-12), 'ACTUAL_SCALAR_DIFFERS:'+name)
        else:
            need(item == dict(kind='scalar',value=expected), 'ACTUAL_SCALAR_DIFFERS:'+name)
    for name, expected in declared.items():
        if name in inputs:
            need(inputs[name] == dict(kind='scalar',value=expected), 'ACTUAL_OPTIONAL_DIFFERS:'+name)
        elif name in defaults:
            need(defaults[name] == expected, 'OBSERVED_PUBLIC_DEFAULT_DIFFERS:'+name)
    # pre/next are declared public defaults when omitted, never evidence that
    # a stock internal dispatcher or its typed POD was observed.
    need(all(name in inputs or name in defaults for name in ('pre_tokens','next_tokens','softmax_lse_flag')),
         'REQUIRED_OMITTED_ARGUMENT_DEFAULTS_NOT_RECORDED')
    mask = inputs['atten_mask']['snapshot']['metadata']; table = inputs['block_table']['snapshot']['metadata']
    need(mask['dtype'] == 'torch.int8' and mask['shape'] == [2048,2048] and mask['stride'] == [2048,1]
        and table['dtype'] == 'torch.int32' and table['shape'] == [1,80] and table['stride'] == [80,1],
         'ACTUAL_MASK_OR_TABLE_GEOMETRY_DIFFERS')
    need(saved['return_container_type'] == 'tuple' and len(saved['outputs']) == 2
        and saved['outputs'][1]['metadata']['shape'] == [0]
        and saved['outputs'][1]['metadata']['dtype'] == 'torch.float32', 'ACTUAL_COMPLETE_API_RETURN_STRUCTURE_DIFFERS')


def one_launch(run, rows, token, pid, helper, *, kernel_sha=None, symbols=None):
    resolution = helper.boundary.scoped_runtime(run, rows, token, expected_pid=pid)
    need(len(resolution['launches']) == 1, 'EXACT_ONE_ACTUAL_KERNEL_PER_NORMAL_CALL_REQUIRED')
    launch = resolution['launches'][0]
    need(int(launch['tiling_key']) == evidence.KEY and launch['block_dim'] == 16 and launch['schedule_mode'] == 1,
         'ACTUAL_NORMAL_ENTRY_PROFILE_DIFFERS')
    # A successful sync cannot mask another failed sync or capture overflow.
    scoped = [row for row in rows if row.get('call_id') == token]
    need(not any(row['event'] in ('capture_overflow','unbound_function') for row in scoped)
        and all(row.get('result') == 0 for row in scoped if row['event'] in ('device_sync','stream_sync')),
         'SCOPED_CAPTURE_OR_COMPLETION_FAILED')
    binary = launch['registered_binary']; digest = binary['sha256']
    need(kernel_sha is None or digest == kernel_sha, 'REGISTERED_PRIVATE_BINARY_DIFFERS_FROM_FRESH_BUILD')
    if symbols is not None:
        if digest not in symbols:
            found = helper.elf.elf_symbols(binary['path'], digest, selected_keys=[str(evidence.KEY)])
            entries = found.get('actual_selected_entry_symbols', [])
            for core in ('aic','aiv'):
                need(any(row['name'].endswith('_'+str(evidence.KEY)+'_mix_'+core)
                    and row['name'].startswith('FusedInferAttentionScore_') and row['symbol_type'] == 2
                    and row['binding'] == 1 and row['section_index'] != 0 and row['bytes'] > 0 for row in entries),
                     'ACTUAL_SELECTED_ELF_ENTRY_SYMBOL_MISSING:'+core)
            symbols[digest] = {key:found[key] for key in ('path','bytes','sha256','status',
                'symbols_are_not_abi_proof','symbols_sha256','actual_selected_keys','actual_selected_entry_symbols')}
    return resolution


def final_private_pod(raw_event, resolved, host, bundle, abi_helper):
    """Select raw bytes by the fresh entry ABI, never by expected byte search."""
    abi, schema = bundle['abi'], bundle['schema']; size = schema['size']
    args = bytes.fromhex(raw_event['args_hex'])
    need(len(args) == raw_event['args_size'] <= 1024**2 and abi['pointer_bytes'] == 8,
         'FINAL_ARGUMENT_SIZE_OR_ABI')
    if raw_event['argument_layout'] == 'aclrt_host_placeholders':
        raw = bytes.fromhex(raw_event['placeholders_hex'])
        need(raw and len(raw)%8 == 0, 'PLACEHOLDER_DESCRIPTOR_SIZE')
        pairs = list(struct.iter_unpack('<II', raw))
        need(len({address for address,_ in pairs}) == len(pairs), 'DUPLICATE_PLACEHOLDER_SLOT')
        selected = [(address,offset) for address,offset in pairs if address == abi['tiling_pointer_offset']]
        need(len(selected) == 1, 'COMPILED_TILING_SLOT_NOT_PRESENT')
        address, offset = selected[0]; others = [(a,b) for a,b in pairs if a != address]
    else:
        need(raw_event['argument_layout'] == 'rtArgsEx' and raw_event['has_tiling'] == 1
            and raw_event['no_h2d_copy'] == 0, 'UNADAPTED_INLINE_ARGUMENT_LAYOUT')
        address, offset = raw_event['tiling_addr_offset'], raw_event['tiling_data_offset']
        descriptors = bytes.fromhex(raw_event['host_input_hex'])
        need(len(descriptors) == raw_event['host_input_count']*8, 'HOST_INPUT_DESCRIPTOR_SIZE')
        others = list(struct.iter_unpack('<II', descriptors))
    need(type(offset) is int and 0 <= address == abi['tiling_pointer_offset']
        and address+8 <= offset and offset+size <= len(args), 'COMPILED_TILING_SLOT_OR_EXTENT_DIFFERS')
    need(all(a+8 <= offset and offset+size <= b < len(args) for a,b in others), 'HOST_INPUT_OVERLAPS_TILING')
    raw = args[offset:offset+size]
    need(host['token'] == resolved['call_id'] and (host['pid'],host['tid']) == (resolved['pid'],resolved['tid'])
        and host['raw_size'] == size and bytes.fromhex(host['raw_hex']) == raw
        and all(str(host[key]) == str(resolved[key]) for key in ('tiling_key','block_dim','schedule_mode')),
         'HOST_FINAL_RAW_ARGUMENT_OR_IDENTITY_DIFFERS')
    typed = abi_helper.decode_raw(raw, schema)
    padding = abi_helper.verify_public_values(typed, schema, host['typed'])
    need(set(host['typed']) == set(typed)-set(padding), 'HOST_TYPED_FIELD_SET_NOT_COMPLETE')
    expected = {'baseParams.bSize':1, 'baseParams.n2Size':8, 'baseParams.gSize':2, 'baseParams.headDim':128,
        'baseParams.headDimRope':0, 'baseParams.softmaxLseFlag':0, 'baseParams.actualSeqS1Dims':1,
        'baseParams.actualSeqS2Dims':1, 'baseParams.s1Size':1, 'innerSplitParams.s2BaseSize':512,
        'pageAttenParams.blockSize':16, 'pageAttenParams.maxBlockNumPerBatch':80,
        'maskParams.attenMaskFlag':1, 'maskParams.sparseMode':3, 'maskParams.preToken':2147483647,
        'maskParams.nextToken':2147483647, 'leftPaddingParams.qPaddingFlag':0,
        'leftPaddingParams.kvPaddingFlag':0, 'prefixParams.prefixFlag':0, 'pseParams.pseShiftFlag':0,
        'baseParams.scaleValue':struct.unpack('<f',struct.pack('<f',128**-.5))[0]}
    need(all(typed.get(key) == value for key,value in expected.items()) and 1025 <= typed['baseParams.s2Size'] <= 2048,
         'FINAL_PUBLIC_POD_NORMAL_GEOMETRY_DIFFERS')
    need(host['schema'] == 'demoncat.paged33.arch22_normal_host.v1' and host['route'] == 'FiaTilingNonQuant'
        and host['mode'] == 'arch22_normal' and host['normal_route_preference_changed'] is True
        and host['eligible'] is True and host['policy_applied'] is False
        and host['native_s2'] == host['selected_s2'] == 512, 'NATIVE_HOST_NORMAL_ONLY_ROUTE_DIFFERS')
    return dict(raw_sha256=hashlib.sha256(raw).hexdigest(), raw_size=size, args_sha256=hashlib.sha256(args).hexdigest(),
        args_size=len(args), tiling_pointer_offset=address, tiling_data_offset=offset, typed=typed,
        verified_public_serializer_padding=padding, kernel_argument_abi=abi)


def native_phase(run, phase, audit, runtime, saved, helper, bundle, checks, symbols):
    name = phase['phase']; calls = phase['finish']['capture']['native_calls']; prefix = name+'.native'
    need(len(calls) == 28 and {row['module_path'] for row in calls} == set(LAYERS)
        and len({row['call_id'] for row in calls}) == 28, 'COMPLETE_28_PRIVATE_NATIVE_CALLS_REQUIRED')
    folder = run/name/'native'; log = folder/'host.jsonl'; before = log.stat()
    need(0 < before.st_size <= 32*1024**2 and not log.is_symlink(), 'HOST_LOG_BOUND_OR_LINK')
    raw = log.read_bytes(); after = log.stat()
    need(raw.endswith(b'\n') and (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'HOST_LOG_NOT_COMPLETE_STABLE')
    hosts = [json.loads(line,object_pairs_hook=evidence.unique) for line in raw.splitlines()]
    need(len(hosts) == 28 and len({row['token'] for row in hosts}) == 28
        and {row['token'] for row in hosts} == {row['call_id'] for row in calls}, 'HOST_AND_ACTUAL_CALL_SETS_DIFFER')
    host_map = {row['token']:row for row in hosts}; summaries = []
    def single(call):
        token, path = call['call_id'], call['module_path']; row = audit[path]
        need(evidence.read(evidence.child(folder, token+'.json')) == call, 'ORIGINAL_NATIVE_RECEIPT_COPY_DIFFERS')
        need(call['host'] == host_map[token] and call['token'] == token
            and call['module_object_id'] == row['module_object_id'] and call['scheduler'] == row['scheduler']
            and call['virtual_engine'] == row['virtual_engine'] and call['original_calls'] == 1
            and call['phase'] == ('baseline' if name == 'measured' else 'off')
            and all(call[key] is False for key in ('selection_applied','cache_repacked','api_arguments_changed','output_mutated')),
             'ACTUAL_NATIVE_TO_PUBLIC_REQUEST_ORIGINAL_ONCE_JOIN_DIFFERS')
        if path == FIRST: need(token == saved['native_call_id'], 'FIRST_BOUNDARY_NATIVE_TOKEN_DIFFERS')
        resolution = one_launch(run,runtime,token,saved['pid'],helper,
            kernel_sha=bundle['completion']['kernel_binary_sha256'],symbols=symbols)
        launch = resolution['launches'][0]
        need(launch['tid'] == saved['thread'], 'NATIVE_OWNER_THREAD_DIFFERS_FROM_ACTUAL_BOUNDARY')
        decoded = final_private_pod(runtime[launch['raw_line']-1],launch,host_map[token],bundle,helper.abi)
        recorded = call['native']
        need(all(recorded[key] == value for key,value in decoded.items())
            and recorded['call_id'] == token and recorded['pid'] == launch['pid'] and recorded['tid'] == launch['tid']
            and all(str(recorded[key]) == str(launch[key]) for key in ('tiling_key','block_dim','schedule_mode'))
            and recorded['kernel_sha256'] == launch['registered_binary']['sha256'], 'STORED_NATIVE_RECEIPT_DIFFERS_FROM_RAW_REPLAY')
        return dict(module_path=path, call_id=token, original_calls=1, request_ids=row['scheduler']['request_ids'],
            raw_tiling_sha256=decoded['raw_sha256'], all_typed_fields_sha256=evidence.digest(decoded['typed']),
            typed_field_count=len(decoded['typed']), public_alignment_fields=sorted(decoded['verified_public_serializer_padding']),
            launch=launch, original_receipt_sha256=evidence.sha(folder/(token+'.json')))
    for call in calls:
        value = checks.run(prefix+'.'+call['module_path'],lambda call=call:single(call))
        if value is not None: summaries.append(value)
    return dict(host_path=str(log), host_sha256=hashlib.sha256(raw).hexdigest(), host_rows=len(hosts),
        independently_replayed=len(summaries), calls=summaries)


def check_coexistence(document, source, bundle, pid):
    constants = literal_constants(source, 'paged33_loading.py',
        ('STOCK_FILES','LOADER_HEADER_SHA','NORM_SYMBOLS','FIA_SYMBOLS','PRIVATE_LIB'))
    stock_root = '/vllm-workspace/vllm-ascend/vllm_ascend/'
    stock_path = stock_root+'_cann_ops_custom/vendors/vllm-ascend/'+constants['PRIVATE_LIB']
    private_path = '/native/stage/vendors/'+evidence.VENDOR+'/'+constants['PRIVATE_LIB']
    stock_sha = constants['STOCK_FILES']['_cann_ops_custom/vendors/vllm-ascend/'+constants['PRIVATE_LIB']]
    private_sha = bundle['artifacts'][constants['PRIVATE_LIB']]['sha256']
    rows = [document[key] for key in ('library_bootstrap','library_coexistence',
        'library_coexistence_after_standard_worker_initialization')]
    rows += [phase['finish']['cleanup']['library_coexistence'] for phase in document['phases'][:-1]]
    for index,row in enumerate(rows):
        need(row['schema'] == 'demoncat.paged33.api_coexistence.v1' and row['pid'] == pid
            and row['stock_files'] == constants['STOCK_FILES'] and row['stock_loader_source_sha256'] == constants['LOADER_HEADER_SHA']
            and row['private_global_library_path_added'] is False and row['operator_function_invocations'] == 0
            and row['model_or_norm_replaced'] is False and row['actual_fia_launch_proven'] is False
            and row['qualification'] == 'SYMBOL_BINDING_ONLY' and row['platform_ready'] is (index != 0),
             'STOCK_PRIVATE_LIBRARY_SOURCE_OR_LIFETIME_DIFFERS')
        for symbols,path,digest in ((row['stock_binding_before_private'],stock_path,stock_sha),
                (row['bindings']['stock_norm'],stock_path,stock_sha),(row['bindings']['private_fia'],private_path,private_sha)):
            expected = constants['FIA_SYMBOLS'] if path == private_path else constants['NORM_SYMBOLS']
            need(len(symbols) == len(expected) and {item['symbol'] for item in symbols} == set(expected)
                and all(item['path'] == path and item['sha256'] == digest and type(item['address']) is int
                    and item['address'] > 0 for item in symbols), 'API_LIBRARY_SYMBOL_RECEIPT_DIFFERS')
        need(row['bindings']['distinct_library_handles'] is True, 'STOCK_PRIVATE_HANDLES_ALIASED')
        expected_order = ['/native/stage/vendors/'+evidence.VENDOR]
        if index: expected_order += [stock_root+'_cann_ops_custom/vendors/vllm-ascend']
        need(row['vendor_order'] == expected_order, 'EXACT_VENDOR_SEARCH_ORDER_DIFFERS')
    return dict(receipts=len(rows), all_source_bound_symbol_owners_consistent=True,
                selected_callsite_aclnn_owner_observed=False)


def compare_pair(left, right, helper):
    a,b = left['boundary'],right['boundary']
    payloads = helper.differences.compare(left['payloads'],right['payloads'],LAYERS)
    if a is None or b is None:
        return dict(complete_first_input_signature_equal=None,query=None,immediate_outputs=None,
                    all_complete_payloads=payloads,first_boundary_incomplete=True,exact=False)
    need(len(a['outputs']) == len(b['outputs']), 'IMMEDIATE_OUTPUT_COUNT_DIFFERS')
    outputs = [helper.boundary.difference(helper.differences,x,y) for x,y in zip(a['outputs'],b['outputs'])]
    query = helper.boundary.difference(helper.differences,a['query'],b['query'])
    equal = a['input_signature'] == b['input_signature']
    return dict(complete_first_input_signature_equal=equal, query=query, immediate_outputs=outputs,
        all_complete_payloads=payloads, exact=equal and query['exact'] and all(row['exact'] for row in outputs)
            and payloads['differing_payloads'] == 0)


def review_arm(run, expected_sha, kind, source, helper, bundle, observer, checks, symbols):
    run = Path(run).resolve(strict=True); result = run/'result.json'
    need(evidence.sha(result) == expected_sha, 'PINNED_RESULT_SHA_DIFFERS:'+kind)
    document = evidence.read(result); failures = original_failures(document)
    summary = dict(result_path=str(result), result_sha256=expected_sha, original_status=document.get('status'),
        original_failures=failures, phase_summaries=[], within_run={}, qualification_authority=False)
    checks.require(kind, not failures, 'ORIGINAL_RUN_FAILURES_PRESERVED')
    checks.run(kind+'.execution', lambda:check_execution(document,kind,source,helper,bundle,observer))
    phases = document.get('phases', [])
    checks.require(kind+'.phases', [row.get('phase') for row in phases] == list(PHASES), 'FIVE_COMPLETE_ORDERED_PHASES_REQUIRED')
    need(len(phases) <= 5 and len({row['phase'] for row in phases}) == len(phases), 'PHASES_AMBIGUOUS_OR_OVERSIZED')
    tokens = document['warm_healthy']['prompt_token_ids']
    checks.require(kind+'.prompt', len(tokens) == document['prompt']['token_count'] == 1024
        and all(type(token) is int and token >= 0 for token in tokens)
        and hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest() == document['prompt']['token_sha256'],
        'ACTUAL_PROMPT_BYTES_DIFFER')
    runtime = checks.run(kind+'.runtime', lambda:helper.boundary.read_runtime(run))
    runtime_rows = runtime[0] if runtime is not None else []
    summary['runtime'] = runtime[1] if runtime is not None else None
    checks.require(kind+'.runtime_overflow', not any(row['event'] == 'capture_overflow' for row in runtime_rows),
                   'RUNTIME_CAPTURE_OVERFLOW_RETAINS_INCOMPLETE_SCOPE')
    values, identities, requests, scope_tokens, native_total = {}, [], [], [], 0
    for phase in phases:
        name = phase['phase']; path = kind+'.'+name; private = kind == 'baseline' and name != 'healthy_after'
        phase_summary = dict(phase=name); summary['phase_summaries'].append(phase_summary)
        if not name in PHASES:
            checks.require(path,False,'UNKNOWN_PHASE'); continue
        finish = phase.get('finish', {})
        phase_summary['cleanup'] = checks.run(path+'.cleanup',lambda:check_cleanup(finish,private))
        def request():
            need(phase['response']['prompt_token_ids'] == tokens and phase['response']['outputs'] == document['warm_healthy']['outputs']
                and phase['response']['request_id'] == phase['mapping']['external_request_id']
                and phase['mapping']['internal_request_id'] == phase['expected']['actual_internal_request_id'], 'ACTUAL_REQUEST_MAPPING_OR_GENERATION_DIFFERS')
            return phase['mapping']['internal_request_id']
        request_id = checks.run(path+'.request',request)
        if request_id is not None: requests.append(request_id)
        capture = finish.get('capture')
        if capture is None:
            checks.require(path+'.capture',False,'ORIGINAL_CAPTURE_MISSING'); continue
        checks.run(path+'.capture_copy',lambda:need(evidence.read(run/name/'runtime_result.json') == capture,'CAPTURE_COPY_DIFFERS'))
        checks.run(path+'.source',lambda:need(phase['setup']['candidate_count'] == 28
            and phase['setup']['source'] == capture['audit']['source'] and phase['setup']['stock_read_only'] is (not private)
            and phase['setup']['module_sha256'] == source['pins'][evidence.WORKER+'paged33_tnd_probe.py']
            and phase['setup']['core_methods_replaced'] is False and phase['setup']['first_boundary_diagnostic'] is True,
            'ACTUAL_CAPTURE_SOURCE_OR_SCOPE'))
        payloads = checks.run(path+'.87_payloads',lambda:payloads_for_phase(run,name,capture,helper))
        if payloads is not None: values[name] = dict(boundary=None,payloads=payloads)
        audit = checks.run(path+'.28_natural_boundaries',lambda:helper.bridge.boundary_rows(capture,phase,tokens))
        saved = capture.get('first_decode_boundary')
        if saved is None or audit is None:
            checks.require(path+'.first_boundary',False,'COMPLETE_FIRST_BOUNDARY_OR_NATURAL_CENSUS_MISSING'); continue
        checks.run(path+'.actual_inputs',lambda:check_actual_inputs(saved))
        boundary = checks.run(path+'.first_boundary',lambda:helper.boundary.first_boundary(run,phase,audit[FIRST],difference_reader=helper.differences))
        if boundary is not None: phase_summary['first_boundary'] = boundary['summary']
        if payloads is not None: values[name]['boundary'] = boundary
        identities.append(tuple(saved[key] for key in ('pid','thread','worker_object_id','model_object_id','rank')))
        token = saved['native_call_id']
        resolution = checks.run(path+'.first_runtime', lambda:one_launch(run,runtime_rows,token,saved['pid'],helper,
            kernel_sha=bundle['completion']['kernel_binary_sha256'] if kind == 'baseline' else None,symbols=symbols))
        phase_summary['new_offline_runtime_resolution'] = resolution
        phase_summary['original_runtime_observation'] = saved['runtime_observation']
        if private:
            native = checks.run(path+'.all_native',lambda:native_phase(run,phase,audit,runtime_rows,saved,helper,bundle,checks,symbols))
            phase_summary['native_replay'] = native
            if native is not None:
                native_total += native['independently_replayed']; scope_tokens += [row['call_id'] for row in native['calls']]
        else:
            checks.require(path+'.no_policy_calls',capture['native_calls'] == [],'NATIVE_POLICY_PRESENT_IN_STOCK_OR_POST_CLEAN_HEALTH')
            scope_tokens.append(token)
    checks.require(kind+'.identity',len(identities) == 5 and len(set(identities)) == 1
        and len(requests) == len(set(requests)) == 5, 'SAME_WORKER_OR_UNIQUE_REQUEST_SCOPE_INCOMPLETE')
    begins = [row['call_id'] for row in runtime_rows if row['event'] == 'begin']
    ends = [row['call_id'] for row in runtime_rows if row['event'] == 'end']
    expected_scopes = 113 if kind == 'baseline' else 5
    checks.require(kind+'.all_owned_runtime_scopes', len(scope_tokens) == expected_scopes
        and len(set(scope_tokens)) == expected_scopes and Counter(begins) == Counter(ends) == Counter(scope_tokens),
        'EXTRA_MISSING_OR_UNCLOSED_OWNED_RUNTIME_SCOPE')
    checks.require(kind+'.native_census',native_total == (112 if kind == 'baseline' else 0),'COMPLETE_NATIVE_NORMAL_CENSUS_NOT_REPLAYED')
    summary['private_native_replayed'] = native_total
    if kind == 'baseline' and identities:
        summary['library_coexistence'] = checks.run(kind+'.libraries',lambda:check_coexistence(document,source,bundle,identities[0][0]))
    for name in PHASES[1:]:
        if PHASES[0] in values and name in values:
            compared = checks.run(kind+'.repeat.'+name,lambda name=name:compare_pair(values[PHASES[0]],values[name],helper))
            summary['within_run'][name] = compared
            checks.require(kind+'.repeat.'+name,compared is not None and compared['exact'],'WITHIN_ARM_NORMAL_REPEAT_DIFFERS')
    checks.run(kind+'.post_clean_request',lambda:need(document['healthy_after_owned_observers_removed']['outputs'] == document['warm_healthy']['outputs']
        and document['healthy_after_owned_observers_removed']['prompt_token_ids'] == tokens,'POST_CLEAN_HEALTHY_GENERATION_DIFFERS'))
    summary['post_clean_scope'] = 'ORDINARY_REQUEST_TOKENS_TEXT_ONLY_NO_FULL_LOGITS_CAPTURE'
    need(evidence.sha(result) == expected_sha, 'RESULT_CHANGED_DURING_REVIEW')
    if runtime is not None: need(evidence.sha(run/'runtime/runtime.jsonl') == runtime[1]['sha256'], 'RUNTIME_CHANGED_DURING_REVIEW')
    return dict(document=document,values=values,summary=summary)


def review(args):
    checks = Checks(); report = dict(schema='demoncat.paged33.arch22_normal_offline_review.v1',
        status='INCOMPLETE_OR_REJECTED_NORMAL_BRIDGE_NO_F', actual_npu_calls=0,
        original_results_modified=False, fault_authorized=False, bridge_certificate=False,
        c1=False,c2=False,c3=False,E5='NOT_ASSESSED',arms={},cross_arm={},registered_elfs={})
    # Preserve original early/setup failures too, even if there is no warm
    # request, valid bundle or enough capture state to enter the detailed pass.
    for kind,path,digest in (('stock',args.stock_run,args.stock_sha256),('baseline',args.baseline_run,args.baseline_sha256)):
        def original(path=path,digest=digest):
            result = Path(path)/'result.json'
            need(evidence.sha(result) == digest,'PINNED_RESULT_SHA_DIFFERS')
            document = evidence.read(result)
            return dict(result_path=str(result),result_sha256=digest,original_status=document.get('status'),
                        original_failures=original_failures(document),qualification_authority=False)
        report['arms'][kind] = checks.run(kind+'.original',original)
    def context():
        source = evidence.frozen_source(args.source,args.source_manifest_sha256)
        need(evidence.sha(Path(__file__)) == source['pins'][evidence.EXPERIMENT+Path(__file__).name], 'REVIEWER_NOT_FROZEN_SOURCE')
        helper = evidence.helpers(source)
        bundle = evidence.build_bundle(args.build_root,source,helper,args.completion_sha256)
        observer = evidence.observer_bundle(args.observer,source,args.observer_receipt_sha256)
        return source,helper,bundle,observer
    loaded = checks.run('source_build_observer',context)
    if loaded is not None:
        source,helper,bundle,observer = loaded
        report['source'] = dict(root=str(source['root']),commit=source['manifest']['base_commit'],
            manifest_sha256=source['manifest_sha256'],required_sources=source['pins'])
        report['build'] = bundle['summary']; report['observer'] = dict(receipt_sha256=observer['sha256'],
            library_sha256=observer['receipt']['observer_library']['sha256'])
        arms = {}
        for kind,path,digest in (('stock',args.stock_run,args.stock_sha256),('baseline',args.baseline_run,args.baseline_sha256)):
            item = checks.run(kind,lambda kind=kind,path=path,digest=digest:review_arm(path,digest,kind,source,helper,bundle,observer,checks,report['registered_elfs']))
            if item is not None: arms[kind] = item; report['arms'][kind] = item['summary']
        if len(arms) == 2:
            checks.require('paired_execution',arms['stock']['document']['execution_contract'] == arms['baseline']['document']['execution_contract'],
                           'PAIRED_MODEL_SOURCE_ENVIRONMENT_OR_CAPTURE_CONTRACT_DIFFERS')
            for phase in PHASES:
                if all(phase in arm['values'] for arm in arms.values()):
                    compared = checks.run('cross.'+phase,lambda phase=phase:compare_pair(arms['stock']['values'][phase],arms['baseline']['values'][phase],helper))
                    report['cross_arm'][phase] = compared
            complete = len(report['cross_arm']) == 5 and all(row is not None for row in report['cross_arm'].values())
            checks.require('all_cross_phase_payloads',complete,'FIVE_BY_87_COMPLETE_CROSS_COMPARISONS_REQUIRED')
            if complete:
                report['complete_payload_pairs'] = sum(row['all_complete_payloads']['payload_pairs'] for row in report['cross_arm'].values())
                report['normal_payloads_exact'] = all(row['exact'] for row in report['cross_arm'].values())
                if not checks.issues:
                    report['status'] = ('SOURCE_BOUND_EXACT_NORMAL_BRIDGE_RECORDED_NO_F_AUTHORITY'
                        if report['normal_payloads_exact'] else 'NORMAL_BRIDGE_DIFFERS_REJECTED_NO_F')
    report.update(checks=checks.checks,issues=checks.issues,limitations=[
        'No stock private ABI equality assumed. Stock key, symbols and launch bytes are observed without decoding its POD.',
        'Selected ACLNN callsite owner and every possible runtime entrypoint are not independently observed.',
        'Mask is hash-only. Logical KV before the first API is hash-only; complete matching later logical KV bins are rehashed.',
        'Checkpoint equality does not exclude intervening writes. Supplied scalars and omitted public defaults remain distinguished.',
        'Native typed arrays are fully decoded from final private arguments and matched to all host fields, including alignment bytes.',
        'Model/package identities and source-bound observer receipts are reconciled, not a cryptographic attestation of execution.',
        'No C1/C2/C3, E5 localization, raw-device fault payload or generic 8x/9x qualification is granted.'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','build-root','observer','stock-run','baseline-run','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args(); sys.dont_write_bytecode = True; sys.addaudithook(evidence.prohibit_runtime)
    need(not args.out.exists(),'FRESH_REVIEW_OUTPUT_REQUIRED')
    report = review(args); args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],issues=len(report['issues']),
        path=str(args.out),sha256=evidence.sha(args.out),fault_authorized=False)))
    return 0 if report['status'] == 'SOURCE_BOUND_EXACT_NORMAL_BRIDGE_RECORDED_NO_F_AUTHORITY' else 2


if __name__ == '__main__': raise SystemExit(main())
