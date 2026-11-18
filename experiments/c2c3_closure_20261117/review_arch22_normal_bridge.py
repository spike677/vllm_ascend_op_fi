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


def initialize_cpu_byte_reader():
    """Initialize only NumPy's CPU byte operations before the no-load audit.

    The frozen readers import NumPy lazily. A cold import may initialize
    ctypes, so doing that after prohibit_runtime rejects the reader itself.
    This bounded setup neither imports a device framework nor calls a device.
    The permanent audit hook is installed immediately afterwards by main.
    """
    frameworks = ('torch', 'torch_npu', 'vllm', 'vllm_ascend')
    need(not any(name.split('.')[0] in frameworks for name in sys.modules),
         'CPU_READER_DEVICE_FRAMEWORK_ALREADY_IMPORTED')
    import numpy as np
    module = Path(np.__file__).resolve(strict=True)
    core = (sys.modules.get('numpy._core._multiarray_umath')
            or sys.modules.get('numpy.core._multiarray_umath'))
    need(core is not None and getattr(core, '__file__', None), 'CPU_NUMPY_CORE_SOURCE_MISSING')
    core_path = Path(core.__file__).resolve(strict=True)
    need(core_path.is_relative_to(module.parent) and core_path.suffix in ('.so', '.pyd')
         and type(np.__version__) is str and np.__version__, 'CPU_NUMPY_SOURCE_OR_VERSION_UNBOUND')
    half = np.frombuffer(struct.pack('<4e', 0.0, -0.0, 1.5, -2.0), dtype='<f2')
    need(half.astype(np.float64).tolist() == [0.0, -0.0, 1.5, -2.0]
         and np.signbit(half).tolist() == [False, True, False, True]
         and np.isfinite(half).all(), 'CPU_NUMPY_FP16_OR_SIGNED_ZERO_SELFTEST')
    bf = np.frombuffer(struct.pack('<4H', 0x3f80, 0xc000, 0x7f80, 0x7fc0), dtype='<u2')
    decoded = (bf.astype('<u4') << 16).view('<f4')
    need(decoded[:2].astype(np.float64).tolist() == [1.0, -2.0]
         and np.isfinite(decoded).tolist() == [True, True, False, False]
         and ((bf & 0x7f80) != 0x7f80).tolist() == [True, True, False, False],
         'CPU_NUMPY_BF16_OR_NONFINITE_SELFTEST')
    left = np.frombuffer(struct.pack('<2f', 1.0, -0.0), dtype='<f4').astype(np.float64)
    right = np.frombuffer(struct.pack('<2f', 1.5, 0.0), dtype='<f4').astype(np.float64)
    need(int(np.count_nonzero(left != right)) == 1 and float(np.abs(left-right).max(initial=0)) == 0.5,
         'CPU_NUMPY_DIFFERENCE_SELFTEST')
    need(not any(name.split('.')[0] in frameworks for name in sys.modules),
         'CPU_READER_DEVICE_FRAMEWORK_IMPORTED')
    return dict(purpose='CPU_RAW_BYTE_FINITE_AND_NUMERIC_DIFFERENCE_ONLY', version=np.__version__,
        module_path=str(module), module_sha256=evidence.sha(module),
        core_path=str(core_path), core_sha256=evidence.sha(core_path),
        initialized_before_no_dynamic_load_audit=True, dtype_selftests=['float16','bfloat16','float32','float64'],
        device_frameworks_imported=False, actual_npu_calls=0)


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
        and document['statistics'] is False and document['normal_bridge_variant'] == evidence.variant(source)
        and document['normal_route_preference_changed'] is (kind == 'baseline')
        and all(document[k] is False for k in ('bridge_compatible', 'fault_authorized',
            'stock_private_exact_proven', 'public_worker_qualified', 'c2', 'c3')), 'NORMAL_ONLY_RUN_SCOPE')
    if evidence.variant(source)==evidence.S2_VARIANT:
        need(document.get('native_target_scope')=='FIRST_LAYER_NORMAL_ONLY','S2_FIRST_LAYER_NATIVE_SCOPE_REQUIRED')
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


def check_cleanup(finish, private, *, first_layer_only=False):
    cleanup = finish['cleanup']
    need(finish.get('closed') is True and not finish.get('error') and not finish.get('cleanup_error')
        and cleanup['scope_cleared'] is True and cleanup['local_streams_drained'] is True
        and cleanup['owned_hooks'] == cleanup['owned_api_wrappers'] == 0
        and cleanup['boundary_refs_released'] is True and cleanup['core_methods_replaced'] is False
        and not cleanup.get('failures'), 'OWNER_OR_RESOURCE_CLEANUP_INCOMPLETE')
    if private:
        need(cleanup['scope_active'] is cleanup['scope_owned'] is False and cleanup['environment_owned'] == 0
            and cleanup['first_error'] is None and cleanup['calls'] == (1 if first_layer_only else 28) and cleanup['simulation'] is False,
             'PRIVATE_NATIVE_SCOPE_OR_ERROR_REMAINS')
        if first_layer_only:
            rows=cleanup['unscoped_original_calls']
            need(len(rows)==27 and {row['module_path'] for row in rows}==set(LAYERS)-{FIRST}
                and all(row['original_calls']==1 and row['returned'] is True for row in rows),
                'S2_NON_TARGET_ORIGINAL_CALL_CENSUS_INCOMPLETE')
    else:
        stock = cleanup['stock_launch_scope']
        need(stock['scope_cleared'] is True and stock['local_streams_drained'] is True and stock['errors'] == [],
             'STOCK_OBSERVER_SCOPE_REMAINS')
    if first_layer_only:need(cleanup.get('logical_binding_released') is True,'S2_LIVE_INPUT_REFERENCES_REMAIN')
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
    need(inputs['query']['snapshot']['metadata']['dtype'] == inputs['key']['metadata']['dtype']
        == inputs['value']['metadata']['dtype'] == 'torch.float16', 'ACTUAL_NONQUANT_QKV_REQUIRED')
    need(mask['dtype'] == 'torch.int8' and mask['shape'] == [2048,2048] and mask['stride'] == [2048,1]
        and table['dtype'] == 'torch.int32' and table['shape'] == [1,80] and table['stride'] == [80,1],
         'ACTUAL_MASK_OR_TABLE_GEOMETRY_DIFFERS')
    need(saved['return_container_type'] == 'tuple' and len(saved['outputs']) == 2
        and saved['outputs'][1]['metadata']['shape'] == [0]
        and saved['outputs'][1]['metadata']['dtype'] == 'torch.float32', 'ACTUAL_COMPLETE_API_RETURN_STRUCTURE_DIFFERS')
    return dict(supplied_python_token_attributes={name:inputs[name]['value'] for name in
                    ('pre_tokens','next_tokens') if name in inputs},
                omitted_python_token_defaults={name:defaults[name] for name in
                    ('pre_tokens','next_tokens') if name not in inputs},
                declaration_is_native_observation=False)


def page_relocation_evidence(saved, validated, completed_boundary):
    """Prove a per-request page map without changing the raw input signature.

    ``validated`` is produced in this process by first_boundary, which reads
    the actual query/output files and rehashes the complete later logical KV
    files against both API checkpoints. It is never a flag read from result.
    Only the 65 used page IDs and their full-table hash may be normalized.
    """
    check_actual_inputs(saved)
    inputs, bound = saved['inputs'], saved['boundary']
    # The first-boundary record is frozen before the call; its returned flag
    # is normally false. Join the final independent audit row rather than
    # rewriting that historical flag or using it as completion evidence.
    need(type(bound['consumer_returned']) is bool and completed_boundary['consumer_returned'] is True
         and all(completed_boundary[k] == bound[k] for k in ('module_path','module_object_id','impl_object_id',
             'query','key','value','page_table','live_key_cache','live_value_cache','used_pages',
             'slots','expected_slots','sequence_lengths','scheduler','writer')),
         'REPEAT_FINAL_CONSUMER_BOUNDARY_NOT_JOINED')
    table = inputs['block_table']['snapshot']; values = saved['block_table_values']
    need(type(values) is list and len(values) == 1 and type(values[0]) is list
         and len(values[0]) == 80 and all(type(n) is int for n in values[0]),
         'REPEAT_COMPLETE_TYPED_PAGE_TABLE_REQUIRED')
    raw = struct.pack('<80i', *values[0]); table_sha = hashlib.sha256(raw).hexdigest()
    need(table['bytes'] == len(raw) == 320 and table['sha256'] == table_sha
         and saved['inputs_after']['block_table']['sha256'] == table_sha
         and table['metadata'] == bound['page_table'] and table['metadata']['storage_offset'] == 0,
         'REPEAT_PAGE_TABLE_BYTES_OR_ACTUAL_VIEW_DIFFERS')
    used = values[0][:65]
    need(bound['used_pages'] == [used] and len(set(used)) == 65
         and bound['sequence_lengths'] == [1025], 'REPEAT_EFFECTIVE_PAGES_AMBIGUOUS')
    scheduler = bound['scheduler']
    need(scheduler['counts'] == [1] and scheduler['pure_decode'] is True
         and scheduler['row_token_positions'] == [1024]
         and len(scheduler['request_ids']) == 1 and type(scheduler['request_ids'][0]) is str
         and scheduler['request_ids'][0], 'REPEAT_ACTUAL_DECODE_POSITION_UNBOUND')
    slot = used[64]*16
    writer = bound['writer']
    need(all(type(n) is int for n in bound['slots'])
         and bound['slots'] == bound['expected_slots'] == writer['slots'] == [slot]
         and writer['returned'] is True
         and writer['module_path'] == bound['module_path'] == FIRST,
         'REPEAT_CURRENT_WRITER_OR_CONSUMER_SLOT_DIFFERS')
    logical = {}; capacity = []
    for kind in ('key', 'value'):
        flat, live = bound[kind], bound['live_'+kind+'_cache']
        count = live['shape'][0]
        need(type(count) is int and count >= 65 and all(0 <= n < count for n in values[0]),
             'REPEAT_PAGE_ID_OUTSIDE_ACTUAL_CACHE:'+kind)
        need(flat == inputs[kind]['metadata'] and writer[kind+'_cache'] == live
             and flat['shape'] == [count,16,1024] and live['shape'] == [count,16,8,128]
             and flat['stride'] == [16384,1024,1] and live['stride'] == [16384,1024,128,1]
             and flat['storage_offset'] == live['storage_offset'] == 0
             and type(flat['storage_identity']) is int and flat['storage_identity'] > 0
             and flat['storage_identity'] == live['storage_identity']
             and flat['storage_nbytes'] == live['storage_nbytes'] == count*16*1024*2
             and flat['dtype'] == live['dtype'] == 'torch.float16'
             and flat['device'] == live['device'] == inputs['query']['snapshot']['metadata']['device']
             and flat['device'].split(':')[0] == 'npu', 'REPEAT_ACTUAL_3D_TO_4D_STORAGE_VIEW_DIFFERS:'+kind)
        before, after = saved['logical_kv_before'][kind], saved['logical_kv_after'][kind]
        verified = validated['summary']['logical_kv'][kind]
        need(before['metadata'] == after['metadata'] == live
             and all(before[k] == after[k] for k in ('version','shape','bytes','sha256','finite'))
             and before['version'] == inputs[kind]['version']
             and before['shape'] == [1025,8,128] and before['bytes'] == 2099200 and before['finite'] is True
             and before['physical_pages'] == after['physical_pages'] == used
             and before['readback'] == after['readback'] == 'ACTUAL_VALID_LOGICAL_PAGE_ORDER'
             and before['operator_input_repacked'] is after['operator_input_repacked'] is False
             and all(verified[k] == before[k] for k in ('shape','bytes','sha256'))
             and verified['dtype'] == live['dtype'] and verified['matching_post_file_rehashed'] is True
             and verified['pre_boundary_file_preserved'] is False and verified['post_consumer_file'],
             'REPEAT_COMPLETE_LOGICAL_KV_PROOF_MISSING_OR_CHANGED:'+kind)
        logical[kind] = {k:verified[k] for k in ('shape','dtype','bytes','sha256')}
        capacity.append(count)
    need(capacity[0] == capacity[1], 'REPEAT_KV_PAGE_CAPACITIES_DIFFER')
    signature = validated['input_signature']
    need(signature['block_table_values'] == values
         and signature['block_table']['snapshot']['sha256'] == table_sha
         and signature['logical_kv'] == logical, 'REPEAT_VERIFIED_INPUT_SIGNATURE_JOIN_DIFFERS')
    normalized = json.loads(json.dumps(signature, allow_nan=False))
    normalized['block_table_values'][0][:65] = list(range(65))
    normalized['block_table']['snapshot']['sha256'] = 'PROVEN_USED_PAGE_RELOCATION_ONLY'
    return dict(normalized_signature=normalized, summary=dict(
        kind='SOURCE_BOUND_LOGICAL_PAGE_RELOCATION_CHECK', used_page_count=65, table_entries=80,
        cache_pages=capacity[0], current_position=1024, writer_consumer_slot=slot,
        used_pages=used, complete_table_sha256=table_sha, logical_kv=logical,
        raw_input_signature_sha256=validated['input_sha256'],
        normalized_input_signature_sha256=evidence.digest(normalized),
        captured_boundary_consumer_returned=bound['consumer_returned'],final_consumer_returned=True,
        full_logical_kv_post_files_rehashed=True, pre_boundary_kv_bytes_preserved=False,
        allowed_difference_fields=['block_table_values[0][0:65]', 'block_table.snapshot.sha256']))


def captured_thread_identity(saved):
    identity = saved.get('thread_identity')
    need(type(identity) is dict and set(identity) == {'python_thread_ident','native_tid','pid','source'}
         and identity.get('source') == 'CURRENT_THREAD_GET_IDENT_AND_GET_NATIVE_ID',
         'BOUNDARY_NATIVE_THREAD_IDENTITY_NOT_CAPTURED')
    need(all(type(identity.get(key)) is int and identity[key] > 0
             for key in ('python_thread_ident','native_tid','pid'))
         and type(saved.get('pid')) is int and type(saved.get('thread')) is int
         and identity['pid'] == saved.get('pid') and identity['python_thread_ident'] == saved.get('thread'),
         'BOUNDARY_THREAD_DOMAIN_OR_PID_CHANGED')
    return dict(identity)


def check_live_input(saved,audit):
    """Join the new live binding record to independently captured real bytes."""
    report=saved['logical_input_binding'];identity=report['identity'];record=report['evidence']
    thread=captured_thread_identity(saved)
    need(report['schema']=='demoncat.paged33.live_logical_input.v1'
        and report['actual_original_calls']==1 and report['complete'] is True
        and report['inflight'] is report['closed'] is report['simulation'] is report['qualification_granted'] is False
        and report['first_error'] is None and report['operator_input_repacked'] is False
        and type(identity['owner_object_id']) is int and identity['owner_object_id']>0
        and type(identity['generation']) is int and identity['generation']==1
        and all(identity[key]==thread[key] for key in ('pid','python_thread_ident','native_tid'))
        and all(identity[key]==saved[key] for key in ('worker_object_id','model_object_id'))
        and identity['module_object_id']==audit['module_object_id']
        and identity['impl_object_id']==audit['impl_object_id'], 'S2_LIVE_OWNER_OR_ACTUAL_OBJECT_JOIN_DIFFERS')
    need(record['before']==record['after'] and record['request_id']==audit['scheduler']['request_ids'][0],
         'S2_LIVE_INPUT_CHANGED_OR_REQUEST_DIFFERS')
    before=record['before']
    for key in ('query','key','value','page_table','live_key_cache','live_value_cache','used_pages',
                'slots','expected_slots','sequence_lengths','scheduler','writer'):
        need(before['boundary'][key]==audit[key],'S2_LIVE_BOUNDARY_JOIN_DIFFERS:'+key)
    for key,value in before['tensors'].items():
        captured=saved['inputs'][key]['snapshot']
        need(all(value[name]==captured[name] for name in ('metadata','version','bytes','sha256','finite')),
             'S2_LIVE_TENSOR_BYTES_JOIN_DIFFERS:'+key)
    need(set(before['tensors'])=={'query','atten_mask','block_table'}
        and before['logical_kv']==saved['logical_kv_before'],'S2_COMPLETE_LOGICAL_KV_JOIN_DIFFERS')
    need(len(record['outputs'])==len(saved['outputs'])==2
        and all(all(left[key]==right[key] for key in ('metadata','version','bytes','sha256','finite'))
            for left,right in zip(record['outputs'],saved['outputs'])), 'S2_IMMEDIATE_ACTUAL_RETURN_JOIN_DIFFERS')
    return dict(owner_object_id=identity['owner_object_id'],generation=1,
        actual_request_id=record['request_id'],complete_logical_kv_joined=True,
        original_calls=1,source_bound=True,qualification_granted=False)


def one_launch(run, rows, token, pid, helper, *, native_tid, kernel_sha=None, symbols=None,block_dim=16):
    resolution = helper.boundary.scoped_runtime(run, rows, token, expected_pid=pid)
    need(len(resolution['launches']) == 1, 'EXACT_ONE_ACTUAL_KERNEL_PER_NORMAL_CALL_REQUIRED')
    launch = resolution['launches'][0]
    need(int(launch['tiling_key']) == evidence.KEY and launch['block_dim'] == block_dim and launch['schedule_mode'] == 1,
         'ACTUAL_NORMAL_ENTRY_PROFILE_DIFFERS')
    # A successful sync cannot mask another failed sync or capture overflow.
    scoped = [row for row in rows if row.get('call_id') == token]
    need(type(pid) is int and pid > 0 and type(native_tid) is int and native_tid > 0
         and all((row['pid'],row['tid']) == (pid,native_tid) for row in scoped),
         'NATIVE_OWNER_THREAD_DIFFERS_FROM_ACTUAL_BOUNDARY')
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


def final_private_pod(raw_event, resolved, host, bundle, abi_helper,*,selected_s2=512):
    """Select raw bytes by the fresh entry ABI, never by expected byte search."""
    need(type(selected_s2) is int and selected_s2 in (512,256)
        and (selected_s2==512 or bundle.get('completion',{}).get('variant')==evidence.S2_VARIANT),
        'EXPLICIT_NEW_S2_VARIANT_REQUIRED')
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
        'baseParams.headDimRope':0, 'baseParams.softmaxLseFlag':0, 'baseParams.isLegacyIfa':0, 'baseParams.actualSeqS1Dims':1,
        'baseParams.actualSeqS2Dims':1, 'baseParams.s1Size':1, 'innerSplitParams.s2BaseSize':selected_s2,
        'pageAttenParams.blockSize':16, 'pageAttenParams.maxBlockNumPerBatch':80,
        'maskParams.attenMaskFlag':1, 'maskParams.sparseMode':3, 'maskParams.preToken':2147483647,
        # The exact built attention parser (checked by build_bundle) converts
        # nonquant RIGHT_DOWN sparse mode to INT_MAX/0 before serialization.
        # Python's omitted next_tokens default remains INT_MAX in the input
        # signature; do not equate that declaration to the effective POD.
        'maskParams.nextToken':0, 'leftPaddingParams.qPaddingFlag':0,
        'leftPaddingParams.kvPaddingFlag':0, 'prefixParams.prefixFlag':0, 'pseParams.pseShiftFlag':0,
        'baseParams.scaleValue':struct.unpack('<f',struct.pack('<f',128**-.5))[0]}
    need(all(typed.get(key) == value for key,value in expected.items()) and 1025 <= typed['baseParams.s2Size'] <= 2048,
         'FINAL_PUBLIC_POD_NORMAL_GEOMETRY_DIFFERS')
    s2=bundle.get('completion',{}).get('variant')==evidence.S2_VARIANT
    need(host['schema'] == ('demoncat.paged33.arch22_s2_host.v1' if s2 else 'demoncat.paged33.arch22_normal_host.v1')
        and host['route'] == 'FiaTilingNonQuant'
        and host['mode'] == ('arch22_s2_256' if selected_s2==256 else 'arch22_s2_off' if s2 else 'arch22_normal')
        and host['normal_route_preference_changed'] is True
        and host['eligible'] is True and host['policy_applied'] is (selected_s2==256)
        and host['native_s2'] == 512 and host['selected_s2']==selected_s2, 'NATIVE_HOST_EXPLICIT_SELECTION_DIFFERS')
    return dict(raw_sha256=hashlib.sha256(raw).hexdigest(), raw_size=size, args_sha256=hashlib.sha256(args).hexdigest(),
        args_size=len(args), tiling_pointer_offset=address, tiling_data_offset=offset, typed=typed,
        verified_public_serializer_padding=padding, kernel_argument_abi=abi)


def native_phase(run, phase, audit, runtime, saved, helper, bundle, checks, symbols):
    name = phase['phase']; calls = phase['finish']['capture']['native_calls']; prefix = name+'.native'
    identity = captured_thread_identity(saved)
    s2=bundle['completion'].get('variant')==evidence.S2_VARIANT
    paths={FIRST} if s2 else set(LAYERS)
    need(len(calls) == len(paths) and {row['module_path'] for row in calls} == paths
        and len({row['call_id'] for row in calls}) == len(paths), 'COMPLETE_SCOPED_PRIVATE_NATIVE_CALLS_REQUIRED')
    folder = run/name/'native'; log = folder/'host.jsonl'; before = log.stat()
    need(0 < before.st_size <= 32*1024**2 and not log.is_symlink(), 'HOST_LOG_BOUND_OR_LINK')
    raw = log.read_bytes(); after = log.stat()
    need(raw.endswith(b'\n') and (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), 'HOST_LOG_NOT_COMPLETE_STABLE')
    hosts = [json.loads(line,object_pairs_hook=evidence.unique) for line in raw.splitlines()]
    need(len(hosts) == len(paths) and len({row['token'] for row in hosts}) == len(paths)
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
        need(all(row[name]['dtype'] == 'torch.float16' for name in ('query','key','value')),
             'NATIVE_BOUNDARY_NONQUANT_QKV_REQUIRED')
        if path == FIRST: need(token == saved['native_call_id'], 'FIRST_BOUNDARY_NATIVE_TOKEN_DIFFERS')
        resolution = one_launch(run,runtime,token,saved['pid'],helper,
            native_tid=identity['native_tid'],kernel_sha=bundle['completion']['kernel_binary_sha256'],symbols=symbols)
        launch = resolution['launches'][0]
        decoded = final_private_pod(runtime[launch['raw_line']-1],launch,host_map[token],bundle,helper.abi)
        if s2:
            host=host_map[token];owner=call['owner_mapping'];logical=call['logical_input_binding']
            need(host['phase']==call['phase'] and host['target_path']==FIRST
                and host['actual_observer_scope_matched'] is True
                and host['owner']==owner['native_owner'] and len(host['owner'])==32
                and all(c in '0123456789abcdef' for c in host['owner'])
                and type(host['generation']) is int and host['generation']==owner['generation']==1
                and owner['python_owner_object_id']==logical['identity']['owner_object_id']
                and logical==saved['logical_input_binding'],'S2_ACTUAL_HOST_OWNER_PHASE_INPUT_JOIN_DIFFERS')
            helper.arch22_s2_build_evidence.verify_normal_geometry(decoded['typed'],host,
                bundle['s2_cpu_matrix']['normal_first_decode_rows'][host['aic']])
        recorded = call['native']
        need(all(recorded[key] == value for key,value in decoded.items())
            and recorded['call_id'] == token and recorded['pid'] == launch['pid'] and recorded['tid'] == launch['tid']
            and all(str(recorded[key]) == str(launch[key]) for key in ('tiling_key','block_dim','schedule_mode'))
            and recorded['kernel_sha256'] == launch['registered_binary']['sha256'], 'STORED_NATIVE_RECEIPT_DIFFERS_FROM_RAW_REPLAY')
        return dict(module_path=path, call_id=token, original_calls=1, request_ids=row['scheduler']['request_ids'],
            raw_tiling_sha256=decoded['raw_sha256'], all_typed_fields_sha256=evidence.digest(decoded['typed']),
            effective_native_tokens=dict(pre=decoded['typed']['maskParams.preToken'],next=decoded['typed']['maskParams.nextToken']),
            typed_field_count=len(decoded['typed']), public_alignment_fields=sorted(decoded['verified_public_serializer_padding']),
            launch=launch, original_receipt_sha256=evidence.sha(folder/(token+'.json')))
    for call in calls:
        value = checks.run(prefix+'.'+call['module_path'],lambda call=call:single(call))
        if value is not None: summaries.append(value)
    return dict(host_path=str(log), host_sha256=hashlib.sha256(raw).hexdigest(), host_rows=len(hosts),
        independently_replayed=len(summaries), calls=summaries)


def check_coexistence(document, source, bundle, pid,*,cleanup_receipts=None):
    constants = literal_constants(source, 'paged33_loading.py',
        ('STOCK_FILES','LOADER_HEADER_SHA','NORM_SYMBOLS','FIA_SYMBOLS','PRIVATE_LIB'))
    stock_root = '/vllm-workspace/vllm-ascend/vllm_ascend/'
    stock_path = stock_root+'_cann_ops_custom/vendors/vllm-ascend/'+constants['PRIVATE_LIB']
    private_path = '/native/stage/vendors/'+evidence.vendor_name(source)+'/'+constants['PRIVATE_LIB']
    stock_sha = constants['STOCK_FILES']['_cann_ops_custom/vendors/vllm-ascend/'+constants['PRIVATE_LIB']]
    private_sha = bundle['artifacts'][constants['PRIVATE_LIB']]['sha256']
    rows = [document[key] for key in ('library_bootstrap','library_coexistence',
        'library_coexistence_after_standard_worker_initialization')]
    rows += ([phase['finish']['cleanup']['library_coexistence'] for phase in document['phases'][:-1]]
             if cleanup_receipts is None else cleanup_receipts)
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
        expected_order = ['/native/stage/vendors/'+evidence.vendor_name(source)]
        if index: expected_order += [stock_root+'_cann_ops_custom/vendors/vllm-ascend']
        need(row['vendor_order'] == expected_order, 'EXACT_VENDOR_SEARCH_ORDER_DIFFERS')
    return dict(receipts=len(rows), all_source_bound_symbol_owners_consistent=True,
                selected_callsite_aclnn_owner_observed=False)


def compare_pair(left, right, helper, *, allow_page_relocation=False):
    a,b = left['boundary'],right['boundary']
    payloads = helper.differences.compare(left['payloads'],right['payloads'],LAYERS)
    if a is None or b is None:
        return dict(complete_first_input_signature_equal=None,raw_input_exact=None,
                    logical_input_equivalent=False,normal_repeat_valid=False,
                    query=None,immediate_outputs=None,all_complete_payloads=payloads,
                    first_boundary_incomplete=True,exact=False)
    need(len(a['outputs']) == len(b['outputs']), 'IMMEDIATE_OUTPUT_COUNT_DIFFERS')
    outputs = [helper.boundary.difference(helper.differences,x,y) for x,y in zip(a['outputs'],b['outputs'])]
    query = helper.boundary.difference(helper.differences,a['query'],b['query'])
    equal = a['input_signature'] == b['input_signature']
    numeric_exact = query['exact'] and all(row['exact'] for row in outputs) and payloads['differing_payloads'] == 0
    result = dict(complete_first_input_signature_equal=equal,raw_input_exact=equal,query=query,
        immediate_outputs=outputs,all_complete_payloads=payloads,exact=equal and numeric_exact,
        comparison_scope='WITHIN_ARM_NORMAL_REPEAT' if allow_page_relocation else 'STRICT_CROSS_ARM_RAW_PAIR')
    if allow_page_relocation:
        ap,bp = a.get('page_relocation'),b.get('page_relocation')
        equivalent = (ap is not None and bp is not None
                      and ap['normalized_signature'] == bp['normalized_signature'])
        result.update(logical_input_equivalent=equivalent,normal_repeat_valid=equivalent and numeric_exact,
            repeated_raw_inputs_claimed_identical=equal,
            repeat_kind=('RAW_INPUT_EXACT' if equal else 'ACTUAL_PAGE_RELOCATION') if equivalent else 'UNPROVEN',
            page_relocation=dict(left=ap['summary'] if ap is not None else None,
                right=bp['summary'] if bp is not None else None,
                changed_used_pages=sum(x != y for x,y in zip(ap['summary']['used_pages'],bp['summary']['used_pages']))
                    if ap is not None and bp is not None else None))
    return result


def review_arm(run, expected_sha, kind, source, helper, bundle, observer, checks, symbols):
    run = Path(run).resolve(strict=True); result = run/'result.json'
    need(evidence.sha(result) == expected_sha, 'PINNED_RESULT_SHA_DIFFERS:'+kind)
    document = evidence.read(result); failures = original_failures(document)
    s2=evidence.variant(source)==evidence.S2_VARIANT
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
        phase_summary['cleanup'] = checks.run(path+'.cleanup',lambda:check_cleanup(finish,private,first_layer_only=s2))
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
        phase_summary['python_token_attributes'] = checks.run(path+'.actual_inputs',lambda:check_actual_inputs(saved))
        phase_summary['thread_identity'] = checks.run(path+'.thread_identity',lambda:captured_thread_identity(saved))
        if s2:phase_summary['live_input'] = checks.run(path+'.live_input',lambda:check_live_input(saved,audit[FIRST]))
        boundary = checks.run(path+'.first_boundary',lambda:helper.boundary.first_boundary(run,phase,audit[FIRST],difference_reader=helper.differences))
        if boundary is not None:
            phase_summary['first_boundary'] = boundary['summary']
            relocation = checks.run(path+'.page_relocation',lambda:page_relocation_evidence(saved,boundary,audit[FIRST]))
            boundary['page_relocation'] = relocation
            phase_summary['page_relocation'] = relocation['summary'] if relocation is not None else None
        if payloads is not None: values[name]['boundary'] = boundary
        identity = phase_summary['thread_identity']
        if identity is not None:
            identities.append(tuple(saved[key] for key in ('pid','thread','worker_object_id','model_object_id','rank'))
                              +(identity['native_tid'],))
        token = saved['native_call_id']
        resolution = checks.run(path+'.first_runtime', lambda:one_launch(run,runtime_rows,token,saved['pid'],helper,
            native_tid=captured_thread_identity(saved)['native_tid'],
            kernel_sha=bundle['completion']['kernel_binary_sha256'] if kind == 'baseline' else None,symbols=symbols))
        phase_summary['new_offline_runtime_resolution'] = resolution
        phase_summary['original_runtime_observation'] = saved['runtime_observation']
        if private:
            if s2:
                checks.run(path+'.unscoped_call_joins',lambda:need(all(
                    row['module_object_id']==audit[row['module_path']]['module_object_id']
                    and row['request_ids']==audit[row['module_path']]['scheduler']['request_ids']
                    for row in finish['cleanup']['unscoped_original_calls']), 'S2_UNSCOPED_ACTUAL_MODULE_OR_REQUEST_DIFFERS'))
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
    expected_scopes = 5 if s2 or kind=='stock' else 113
    checks.require(kind+'.all_owned_runtime_scopes', len(scope_tokens) == expected_scopes
        and len(set(scope_tokens)) == expected_scopes and Counter(begins) == Counter(ends) == Counter(scope_tokens),
        'EXTRA_MISSING_OR_UNCLOSED_OWNED_RUNTIME_SCOPE')
    checks.require(kind+'.native_census',native_total == ((4 if s2 else 112) if kind == 'baseline' else 0),'COMPLETE_NATIVE_NORMAL_CENSUS_NOT_REPLAYED')
    summary['private_native_replayed'] = native_total
    if kind == 'baseline' and identities:
        summary['library_coexistence'] = checks.run(kind+'.libraries',lambda:check_coexistence(document,source,bundle,identities[0][0]))
    for name in PHASES[1:]:
        if PHASES[0] in values and name in values:
            compared = checks.run(kind+'.repeat.'+name,lambda name=name:compare_pair(
                values[PHASES[0]],values[name],helper,allow_page_relocation=True))
            summary['within_run'][name] = compared
            checks.require(kind+'.repeat.'+name,compared is not None and compared['normal_repeat_valid'],
                           'WITHIN_ARM_NORMAL_REPEAT_DIFFERS')
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
        source = evidence.frozen_source(args.source,args.source_manifest_sha256,
            build_variant=evidence.S2_VARIANT if getattr(args,'arch22_s2_normal',False) else evidence.VARIANT)
        need(evidence.sha(Path(__file__)) == source['pins'][evidence.EXPERIMENT+Path(__file__).name], 'REVIEWER_NOT_FROZEN_SOURCE')
        helper = evidence.helpers(source)
        bundle = evidence.build_bundle(args.build_root,source,helper,args.completion_sha256)
        observer = evidence.observer_bundle(args.observer,source,args.observer_receipt_sha256)
        return source,helper,bundle,observer
    loaded = checks.run('source_build_observer',context)
    if loaded is not None:
        source,helper,bundle,observer = loaded
        report['native_call_scope'] = (dict(module_path=FIRST,layout='TND',query_tokens=1,kv_length=1025,page_size=16,
            private_calls=4,natural_layers_per_phase=28,normal_only=True)
            if evidence.variant(source)==evidence.S2_VARIANT else dict(private_calls=112,natural_layers_per_phase=28,normal_only=True))
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
        'Cross-arm same-phase pairs require the complete raw input signature and all 87 payloads exact. No page normalization is applied.',
        'Within-arm repeats may be actual physical page relocations, not raw-identical inputs. Only independently checked used page IDs and their table hash can differ; raw equality and logical equivalence are reported separately.',
        'Checkpoint equality does not exclude intervening writes. Supplied scalars and omitted public defaults remain distinguished.',
        'Python thread identifiers and Native OS TIDs are separate captured domains. Older results without an explicit pair remain unproven.',
        'Native typed arrays are fully decoded from final private arguments and matched to all host fields, including alignment bytes.',
        'Model/package identities and source-bound observer receipts are reconciled, not a cryptographic attestation of execution.',
        'No C1/C2/C3, E5 localization, raw-device fault payload or generic 8x/9x qualification is granted.'])
    if getattr(args,'arch22_s2_normal',False):
        report['limitations'].append('The new S2-capable package is exercised only in normal/OFF mode. Its four private Native receipts cover layer0 TND Q1/KV1025/page16; 27 other layers remain unscoped original calls. The complete 28-layer outputs are still compared. This is the trusted standard Worker probe, not completion of the public Core fault entry.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','build-root','observer','stock-run','baseline-run','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-manifest-sha256','completion-sha256','observer-receipt-sha256','stock-sha256','baseline-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--arch22-s2-normal',action='store_true',help='New S2 package normal-only first-layer profile')
    args = parser.parse_args(); sys.dont_write_bytecode = True
    need(not args.out.exists(),'FRESH_REVIEW_OUTPUT_REQUIRED')
    cpu_reader = initialize_cpu_byte_reader()
    sys.addaudithook(evidence.prohibit_runtime)
    report = review(args); report['cpu_byte_reader'] = cpu_reader
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],issues=len(report['issues']),
        path=str(args.out),sha256=evidence.sha(args.out),fault_authorized=False)))
    return 0 if report['status'] == 'SOURCE_BOUND_EXACT_NORMAL_BRIDGE_RECORDED_NO_F_AUTHORITY' else 2


if __name__ == '__main__': raise SystemExit(main())
