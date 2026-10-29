"""Read-only native attention calibration; every observed API executes once.

Native enter/exit markers bracket the original call and synchronizations.
CPU hashing/serialization runs before enter and after exit, outside that interval.
Raw logical inputs are private replay material, never msprobe statistics.
"""
import enum
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

RUNNER = 'vllm_ascend.worker.model_runner_v1'
ATTENTION = 'vllm_ascend.attention.attention_v1'
RUNNER_SHA = '94d75dbeb5d23ab5b383cdc69114167392b967994bb21342cf748b733ba9a968'
ATTENTION_SHA = '80005aba42bcf61247cc0d91502a0f1338194a4b38dd82bc521f6d3c84958390'
DEBUGGER_SHA = '43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb'
APIS = ('npu_fused_infer_attention_score', 'npu_fused_infer_attention_score_v2',
        'npu_fusion_attention', '_npu_paged_attention', '_npu_paged_attention_splitfuse')
_tls = threading.local()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def root():
    return Path(os.environ['FAULT28_OUTPUT'])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def emit(event, stdout=False, **data):
    row = {'event': event, 'pid': os.getpid(), 'thread_id': threading.get_native_id(),
           'epoch_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **data}
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    with (root() / ('pid' + str(os.getpid()) + '.jsonl')).open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
    if stdout:
        print(line, flush=True)
    return row


def token_sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


def identity(tensor):
    import torch_npu
    storage = tensor.untyped_storage()
    return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
            'device': str(tensor.device), 'storage_offset': int(tensor.storage_offset()),
            'object_id': id(tensor), 'tensor_cdata': int(tensor._cdata),
            'storage_cdata': int(storage._cdata), 'storage_ptr': int(storage.data_ptr()),
            'tensor_ptr': int(tensor.data_ptr()), 'storage_nbytes': int(storage.nbytes()),
            'actual_format': int(torch_npu.get_npu_format(tensor)) if tensor.device.type == 'npu' else None}


def tensor_record(tensor, directory=None, name=None, include_hash=True, phase='after_native_api_exit'):
    import torch
    meta = identity(tensor)
    if not include_hash:
        return {'type': 'torch.Tensor', **meta}
    require(tensor.numel() * tensor.element_size() <= 128 * 1024 * 1024,
            'Refusing an unexpectedly large tensor/cache readback')
    cpu = tensor.detach().to('cpu').contiguous()
    payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {'type': 'torch.Tensor', **meta, 'logical_bytes': len(payload),
              'logical_sha256': hashlib.sha256(payload).hexdigest(),
              'raw_order': 'C-contiguous logical elements', 'byteorder': sys.byteorder,
              'readback_phase': phase, 'identity_after_readback': identity(tensor)}
    require(result['identity_after_readback'] == meta, 'Readback changed source tensor metadata')
    if not tensor.is_floating_point() and tensor.numel() <= 32768:
        result['values'] = cpu.tolist()
    if directory is not None:
        require(name is not None and re.fullmatch('[A-Za-z0-9_.-]+', name), 'Unsafe raw capture name')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (name + '.bin')
        with path.open('xb') as stream:
            stream.write(payload)
        result['private_raw_file'] = str(path.relative_to(root()))
        result['publication'] = 'PRIVATE_REPLAY_ONLY_EXCLUDE_FROM_STATISTICS_AND_GITHUB'
    return result


def describe(value, *, hashes=False, directory=None, name='arg', depth=0, phase='after_native_api_exit'):
    import torch
    require(depth < 12, 'Unexpected nested native argument structure')
    if isinstance(value, torch.Tensor):
        return tensor_record(value, directory=directory, name=name, include_hash=hashes, phase=phase)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, enum.Enum):
        return {'enum': type(value).__qualname__, 'name': value.name, 'value': value.value}
    if isinstance(value, (list, tuple)):
        require(len(value) <= 32768, 'Unexpected native argument list size')
        return [describe(v, hashes=hashes, directory=directory, name=name + '_' + str(i), depth=depth+1, phase=phase)
                for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {str(k): describe(v, hashes=hashes, directory=directory,
                                name=name + '_' + re.sub('[^A-Za-z0-9_.-]', '_', str(k)), depth=depth+1, phase=phase)
                for k, v in value.items()}
    raise RuntimeError('Unsupported actual native argument type: ' + type(value).__qualname__)


def fingerprints(value):
    """Ignore only readback phase/file annotations, retaining actual input identities and bytes."""
    if isinstance(value, dict):
        if value.get('type') == 'torch.Tensor':
            keys = ('shape', 'stride', 'dtype', 'device', 'storage_offset', 'object_id', 'tensor_cdata',
                    'storage_cdata', 'storage_ptr', 'tensor_ptr', 'storage_nbytes', 'actual_format',
                    'logical_sha256', 'logical_bytes')
            return {key: value[key] for key in keys}
        return {key: fingerprints(child) for key, child in value.items()}
    if isinstance(value, list):
        return [fingerprints(child) for child in value]
    return value


def validate_runtime(runner):
    import torch
    require(os.environ.get('VLLM_DISABLE_REQUEST_ID_RANDOMIZATION') == '1' and
            os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0', 'Synchronous stable-ID engine required')
    require(runner.model_config.enforce_eager is True and runner.model_config.dtype == torch.float16,
            'Native eager FP16 configuration required')
    pc = runner.parallel_config
    require(pc.tensor_parallel_size == pc.pipeline_parallel_size == pc.data_parallel_size == 1 and
            pc.enable_dbo is False, 'Initial calibration supports TP1 only')
    require(all(getattr(runner, k, None) == 1 for k in ('dp_size', 'pcp_size', 'dcp_size')), 'Context parallelism unsupported')
    require(runner.use_async_scheduling is False and runner.speculative_config is None and
            runner.num_spec_tokens == 0 and not runner.uses_mrope and not runner.supports_mm_inputs,
            'Unsupported asynchronous/speculative/multimodal execution')
    from vllm_ascend.ascend_config import get_ascend_config
    require(get_ascend_config().enable_flashcomm1 is False and sys.modules[RUNNER].lmhead_tp_enable() is False,
            'Sequence repartitioning unsupported')


def capture_rows(runner, scheduler, counts, prepared, control):
    """Verify the actual scheduler/runner buffers, not a requested batch size."""
    validate_runtime(runner)
    batch = runner.input_batch
    ids = list(batch.req_ids)
    mapping = control['request_id_to_prompt_index']
    require(len(ids) == batch.num_reqs and len(set(ids)) == len(ids), 'Invalid request order')
    require(set(ids) == set(mapping) == set(scheduler.num_scheduled_tokens), 'Whole input batch must share one prefill')
    ordered = [int(scheduler.num_scheduled_tokens[rid]) for rid in ids]
    require([int(v) for v in counts] == ordered, 'Prepared count order mismatch')
    total = int(scheduler.total_num_scheduled_tokens)
    require(sum(ordered) == total and isinstance(prepared, tuple) and len(prepared) == 3 and
            prepared[1] is None and int(prepared[2]) == total, 'Prepare return contract changed')
    require(not scheduler.scheduled_spec_decode_tokens and not scheduler.scheduled_encoder_inputs, 'Unexpected scheduled inputs')
    computed = [int(v) for v in batch.num_computed_tokens_cpu[:len(ids)]]
    lengths = [int(v) for v in batch.num_prompt_tokens[:len(ids)]]
    require(computed == [0] * len(ids) and ordered == lengths, 'Cached/chunked/decode inputs rejected')
    indices = [int(v) for v in runner.req_indices.np[:total]]
    positions = [int(v) for v in runner._positions_np_buf[:total]]
    require(indices == [i for i, count in enumerate(ordered) for _ in range(count)], 'Actual request-row mapping differs')
    require(positions == [j for count in ordered for j in range(count)], 'Actual token-position mapping differs')
    requests, offset = [], 0
    for slot, rid in enumerate(ids):
        prompt_index = int(mapping[rid])
        expected = control['prompt_token_ids'][prompt_index]
        actual = runner.requests[rid]
        require(actual.req_id == rid and int(actual.num_computed_tokens) == 0 and
                actual.prompt_token_ids == expected and len(expected) == lengths[slot] and
                not list(getattr(actual, 'output_token_ids', [])), 'Request identity/tokens changed')
        requests.append({'request_id': rid, 'prompt_index': prompt_index, 'batch_slot': slot,
                         'first_row': offset, 'last_row': offset + ordered[slot] - 1,
                         'count': ordered[slot], 'prompt_token_sha256': token_sha(expected)})
        offset += ordered[slot]
    logits_indices = prepared[0].detach().cpu().tolist()
    require(logits_indices == [row['last_row'] for row in requests], 'Logits are not the final prompt positions')
    target = next(row for row in requests if row['prompt_index'] == control['target_index'])
    return {'request_ids': ids, 'requests': requests, 'target_row': target['last_row'],
            'target_output_row': target['batch_slot'], 'target_position': target['count']-1,
            'total': total, 'row_request_indices': indices, 'row_token_positions': positions,
            'logits_indices': logits_indices}


def attention_metadata(meta):
    names = ('attn_state', 'num_actual_tokens', 'num_decode_tokens', 'num_prefills', 'num_decodes',
             'seq_lens_list', 'actual_seq_lengths_q', 'max_query_len', 'causal', 'model_runner_type',
             'seq_lens', 'query_start_loc', 'block_tables', 'slot_mapping', 'attn_mask')
    return {name: describe(getattr(meta, name, None)) for name in names}


def native_mapping(name, args, kwargs, state):
    import torch
    require('paged' not in name, 'Paged native attention is outside fresh-prefill replay calibration')
    values = dict(kwargs)
    for i, label in enumerate(('query', 'key', 'value')):
        if i < len(args):
            require(label not in values, 'Native argument specified twice')
            values[label] = args[i]
    layout = values.get('input_layout', args[4] if name == 'npu_fusion_attention' and len(args) > 4 else None)
    require(layout == 'TND', 'Initial exact mapping only supports observed native TND layout')
    require(values.get('block_table') is None, 'Refusing paged/native cache input')
    tensors = []
    for label in ('query', 'key', 'value'):
        value = values[label]
        if isinstance(value, (tuple, list)):
            require(len(value) == 1, 'Unsupported native multi-tensor KV list')
            value = value[0]
        require(isinstance(value, torch.Tensor) and value.dtype == torch.float16 and value.ndim == 3 and
                value.shape[0] == state['rows']['total'], 'Native Q/K/V are not fresh FP16 TND inputs')
        tensors.append(value)
    heads = values.get('num_heads', values.get('num_query_heads', args[3] if name == 'npu_fusion_attention' and len(args)>3 else None))
    require(heads is not None and int(heads) == tensors[0].shape[1], 'Native query-head mapping differs')
    require(tensors[1].shape[1:] == tensors[2].shape[1:] and
            tensors[0].shape[2] == tensors[1].shape[2] and
            tensors[0].shape[1] % tensors[1].shape[1] == 0, 'Native GQA/KV-head geometry differs')
    kv_heads = values.get('num_key_value_heads')
    require(kv_heads in (None, 0) or int(kv_heads) == tensors[1].shape[1], 'Native KV-head argument differs from its actual tensors')
    expected = [row['last_row']+1 for row in state['rows']['requests']]
    for alternatives in (('actual_seq_lengths', 'actual_seq_qlen'), ('actual_seq_lengths_kv', 'actual_seq_kvlen')):
        supplied = next((values[key] for key in alternatives if key in values), None)
        if isinstance(supplied, torch.Tensor):
            supplied = supplied.detach().cpu().tolist()
        require(supplied is not None and list(supplied) == expected, 'Native cumulative lengths do not match actual scheduler rows')
    target = next(row for row in state['rows']['requests'] if row['prompt_index'] == state['control']['target_index'])
    return tensors, target, values


def install_native_observers(state):
    import torch
    import torch_npu
    bindings = []
    for api in APIS:
        if not hasattr(torch_npu, api):
            continue
        original = getattr(torch_npu, api)
        def make_wrapper(name, old):
            @functools.wraps(old)
            def observed(*args, **kwargs):
                active = getattr(_tls, 'state', None)
                if active is not state or active.get('layer') is None or active.get('inside_native'):
                    return old(*args, **kwargs)
                layer = active['layer']
                native_index = len(active['native_calls'])
                call_id = 'pid' + str(os.getpid()) + ':r0:n' + str(native_index)
                common = {'case_id': active['control']['case_id'], 'rank': 0, 'call_id': call_id,
                          'native_index': native_index, 'layer_index': layer['index'], 'layer_name': layer['name'],
                          'api': name, 'source_sha256': {'runner': RUNNER_SHA, 'attention': ATTENTION_SHA}}
                before = {'positional_args': describe(args), 'kwargs': describe(kwargs)}
                detailed = layer['index'] in (0, active['control']['layer_index'])
                selected = layer['index'] == active['control']['layer_index']
                input_snapshot = None
                if detailed:
                    qkv, target, bound = native_mapping(name, args, kwargs, active)
                    directory = root() / 'private_replay' / 'rank0' / ('native_' + str(native_index))
                    save = directory if selected and active['control']['capture_full_inputs'] else None
                    torch_npu.npu.synchronize()
                    input_snapshot = {
                        'positional_args': describe(args, hashes=True, directory=save, name='arg', phase='before_native_api_enter'),
                        'kwargs': describe(kwargs, hashes=True, directory=save, name='kw', phase='before_native_api_enter')}
                    # Retain the same diagnostic view objects across the call. Creating
                    # fresh slices after the call would change object_id/tensor_cdata
                    # even when their storage, geometry and logical bytes are intact.
                    target_views = {label: value[target['first_row']:target['last_row']+1]
                                    for label, value in zip(('query_all', 'key_all', 'value_all'), qkv)}
                    target_views['query_last'] = qkv[0][target['last_row']]
                    target_before = {label: tensor_record(value, phase='before_native_api_enter')
                                     for label, value in target_views.items()}
                    emit('native_input_snapshot_completed', call_id=call_id, layer_index=layer['index'],
                         private_raw_saved=save is not None)
                active['inside_native'] = True
                returned = False
                try:
                    torch_npu.npu.synchronize()
                    emit('native_api_enter', stdout=True, **common)
                    with torch.profiler.record_function('fault28/native/' + call_id + '/layer' + str(layer['index'])):
                        result = old(*args, **kwargs)
                    returned = True
                    return_value = result
                finally:
                    torch_npu.npu.synchronize()
                    emit('native_api_exit', stdout=True, status='returned' if returned else 'error', **common)
                    active['inside_native'] = False
                # Diagnostics intentionally occur outside the marked native interval.
                record = {**common, 'status': 'returned', 'arguments_before_metadata': before,
                          'arguments_after': {'positional_args': describe(args), 'kwargs': describe(kwargs)},
                          'result_metadata': describe(result), 'layer_metadata': layer['metadata'],
                          'backend_head_geometry': {key: layer[key] for key in ('head_count', 'kv_head_count', 'head_size')},
                          'precision_arguments': {key: {'explicitly_supplied': key in kwargs, 'value': kwargs.get(key)}
                                                  for key in ('inner_precise', 'softmax_scale', 'scale')},
                          'precision_default_scope': 'Omitted defaults require matching native schema/source/profile; input FP16 alone does not establish accumulator precision.',
                          'diagnostic_readback': 'Layer0/selected layer full logical input fingerprints before enter and after exit; raw inputs from the pre snapshot.',
                          'target_mapping_certified': False}
                if detailed:
                    record['arguments_before'] = input_snapshot
                    record['arguments_after'] = {'positional_args': describe(args, hashes=True),
                                                 'kwargs': describe(kwargs, hashes=True)}
                    require(fingerprints(record['arguments_before']) == fingerprints(record['arguments_after']),
                            'Native attention changed input logical bytes or source metadata')
                    record['input_integrity'] = {'exact_logical_hashes_and_metadata_unchanged': True,
                                                 'private_inputs_saved_from': 'before_native_api_enter'}
                    record['target_mapping'] = {'layout': 'TND', **target,
                                                'absolute_query_position': target['count']-1,
                                                'keys_are_target_request_all_prefill_tokens': True}
                    record['target_tensors'] = {label: tensor_record(value)
                                                for label, value in target_views.items()}
                    record['target_tensors_before'] = target_before
                    require(fingerprints(target_before) == fingerprints(record['target_tensors']),
                            'Target input slice changed during native attention')
                    output = result[0] if isinstance(result, (tuple, list)) else result
                    require(isinstance(output, torch.Tensor) and output.dtype == torch.float16 and
                            output.shape[0] == active['rows']['total'], 'Unsupported actual native attention output')
                    record['output_full'] = tensor_record(output, directory=save, name='output')
                    record['output_target_all'] = tensor_record(output[target['first_row']:target['last_row']+1])
                    record['output_target_last'] = tensor_record(output[target['last_row']])
                    record['target_mapping_certified'] = True
                active['native_calls'].append(record)
                write(root() / 'native_calls' / ('call_' + str(native_index).zfill(3) + '.json'), record)
                emit('native_api_observed', call_id=call_id, layer_index=layer['index'], api=name,
                     target_mapping_certified=record['target_mapping_certified'])
                return return_value
            return observed
        wrapper = make_wrapper(api, original)
        setattr(torch_npu, api, wrapper)
        bindings.append((torch_npu, api, original, wrapper))
    require(bindings, 'No native attention APIs are available')
    return bindings


def install_model(runner):
    import torch
    from torch import distributed as dist
    import importlib
    module = importlib.import_module(ATTENTION)
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == ATTENTION_SHA, 'Attention source changed')
    require(dist.get_rank() == 0 and dist.get_world_size() == 1, 'Calibration supports native TP1 only')
    cls = module.AscendAttentionBackendImpl
    require(not getattr(cls, '_fault28_patched', False), 'Attention observer already installed')
    old = cls.forward
    signature = inspect.signature(old)
    @functools.wraps(old)
    def forward(self, *args, **kwargs):
        state = getattr(_tls, 'state', None)
        if not state or not state.get('selected'):
            return old(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs).arguments
        layer, meta = bound['layer'], bound['attn_metadata']
        name = layer.layer_name
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', name)
        require(match is not None and meta is not None, 'Actual attention layer identity is unavailable')
        index = int(match.group(1))
        require(state.get('layer') is None, 'Nested attention backend call')
        require(meta.attn_state == module.AscendAttentionState.PrefillNoCache and meta.causal and
                meta.num_decode_tokens == 0 and meta.num_decodes == 0,
                'Only native fresh causal prefill is accepted')
        expected = [row['last_row']+1 for row in state['rows']['requests']]
        require(list(meta.actual_seq_lengths_q) == expected and list(meta.seq_lens_list) ==
                [row['count'] for row in state['rows']['requests']], 'Attention sequence metadata differs from scheduler')
        state['layer'] = {'name': name, 'index': index, 'metadata': attention_metadata(meta),
                          'head_count': self.num_heads, 'kv_head_count': self.num_kv_heads, 'head_size': self.head_size}
        state['layers'].append(index)
        start = len(state['native_calls'])
        emit('attention_layer_enter', case_id=state['control']['case_id'], layer_index=index, layer_name=name,
             metadata=state['layer']['metadata'])
        try:
            value = old(self, *args, **kwargs)
            require(len(state['native_calls']) == start+1, 'Expected exactly one captured native attention call per layer')
            return value
        finally:
            emit('attention_layer_exit', case_id=state['control']['case_id'], layer_index=index,
                 native_calls=len(state['native_calls'])-start)
            state['layer'] = None
    cls.forward = forward
    cls._fault28_patched = True
    original_logits = runner.model.compute_logits
    @functools.wraps(original_logits)
    def logits(*args, **kwargs):
        value = original_logits(*args, **kwargs)
        state = getattr(_tls, 'state', None)
        if state and state.get('selected') and value is not None:
            require(not state['logits_saved'] and value.ndim == 2 and
                    value.shape[0] == len(state['rows']['requests']), 'Actual next-token logits layout/count changed')
            import torch_npu
            torch_npu.npu.synchronize()
            target = value[state['rows']['target_output_row']]
            meta = tensor_record(target, root() / 'private_replay' / 'rank0', 'target_logits')
            write(root() / 'logits.json', {**meta, 'full_shape': list(value.shape),
                  'target_output_row': state['rows']['target_output_row'],
                  'target_position': state['rows']['target_position'],
                  'target_token_sha256': state['control']['expected_target_token_sha256']})
            state['logits_saved'] = True
        return value
    runner.model.compute_logits = logits
    runner._fault28_completed = False
    runner._fault28_expected_layers = int(runner.model_config.hf_config.num_hidden_layers)
    runner._fault28_debugger = None
    if os.environ['FAULT28_MSPROBE'] == '1':
        from msprobe.pytorch import PrecisionDebugger
        require(importlib.metadata.version('mindstudio-probe') == '26.1.0', 'Unreviewed msprobe version')
        require(hashlib.sha256(Path(inspect.getsourcefile(PrecisionDebugger)).read_bytes()).hexdigest() == DEBUGGER_SHA,
                'Debugger source changed')
        config = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0],
                  'dump_path': str(root() / 'observation'), 'async_dump': False,
                  'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                                 'summary_mode': 'statistics', 'tensor_list': []}}
        write(root() / 'config.json', config)
        runner._fault28_debugger = PrecisionDebugger(config_path=str(root() / 'config.json'))
    weight_dtypes = {}
    for parameter in runner.model.parameters():
        key = str(parameter.dtype)
        weight_dtypes[key] = weight_dtypes.get(key, 0) + parameter.numel()
    emit('model_loaded', rank=0, expected_layers=runner._fault28_expected_layers,
         weight_parameter_elements_by_dtype=weight_dtypes, model_dtype=str(runner.model_config.dtype),
         runner_sha256=RUNNER_SHA, attention_sha256=ATTENTION_SHA,
         custom_opp_path_after_native_init=os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
         msprobe_enabled=runner._fault28_debugger is not None)


def patch(module):
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == RUNNER_SHA, 'Runner source changed')
    cls = module.NPUModelRunner
    old_load, old_execute, old_prepare = cls.load_model, cls.execute_model, cls._prepare_inputs
    @functools.wraps(old_load)
    def load(self, *args, **kwargs):
        value = old_load(self, *args, **kwargs)
        install_model(self)
        return value
    @functools.wraps(old_execute)
    def execute(self, *args, **kwargs):
        setting = Path(os.environ['FAULT28_CONTROL'])
        control = json.loads(setting.read_bytes()) if setting.exists() else {}
        if not control.get('active'):
            return old_execute(self, *args, **kwargs)
        require(getattr(_tls, 'state', None) is None, 'Nested model execute')
        state = {'runner': self, 'control': control, 'rows': None, 'selected': False,
                 'layers': [], 'layer': None, 'native_calls': [], 'bindings': [], 'logits_saved': False}
        _tls.state = state
        succeeded = False
        try:
            result = old_execute(self, *args, **kwargs)
            succeeded = True
            return result
        finally:
            try:
                for owner, name, original, wrapper in reversed(state['bindings']):
                    require(getattr(owner, name) is wrapper, 'Native observer binding changed during execution')
                    setattr(owner, name, original)
                if state.get('observer_started'):
                    self._fault28_debugger.stop()
                    self._fault28_debugger.step()
                if succeeded and state['selected']:
                    require(state['layers'] == list(range(self._fault28_expected_layers)), 'Actual model layers missing/repeated/reordered')
                    require(state['logits_saved'], 'Actual next-token logits were not observed')
                    selected = [row for row in state['native_calls'] if row['layer_index'] == control['layer_index']]
                    require(len(selected) == 1 and selected[0]['target_mapping_certified'], 'Target native attention capture incomplete')
                    native_files = []
                    native_api_counts = None
                    if self._fault28_debugger:
                        directory = root() / 'observation/step0/rank0'
                        for kind in ('dump', 'stack', 'construct'):
                            path = directory / (kind + '.json')
                            obj = json.loads(path.read_bytes())
                            require(bool(obj), 'Empty native msprobe record')
                            if kind == 'dump':
                                require(obj.get('task') == 'statistics' and obj.get('level') == 'mix' and obj.get('data'), 'Wrong msprobe dump mode')
                                native_api_counts = {}
                                for api in {row['api'] for row in state['native_calls']}:
                                    names = [name for name in obj['data'] if re.search(r'\.' + re.escape(api) + r'\.\d+\.forward$', name)]
                                    expected = sum(row['api'] == api for row in state['native_calls'])
                                    require(len(names) == expected, 'Native msprobe API coverage differs from actual caller ledger: ' + api)
                                    native_api_counts[api] = len(names)
                            native_files.append({'path': str(path.relative_to(root())), 'bytes': path.stat().st_size,
                                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                        require(not any(p.suffix in ('.pt', '.bin', '.npy') for p in directory.rglob('*')), 'Native statistics contains tensor payloads')
                    write(root() / 'receipt.json', {'status': 'CAPTURE_COMPLETED', 'rank': 0, 'step': 0,
                          'case_id': control['case_id'], 'native_calls': len(state['native_calls']),
                          'selected_native_call': selected[0]['call_id'], 'prefill_certified': True,
                          'logits_saved': True, 'msprobe_enabled': bool(self._fault28_debugger),
                          'native_files': native_files, 'native_api_counts': native_api_counts,
                          'private_raw_inputs_are_not_statistics': True,
                          'scope': 'Read-only calibration; not tiling modification or mechanism acceptance.'})
                    self._fault28_completed = True
                    emit('capture_completed', rank=0, case_id=control['case_id'], native_calls=len(state['native_calls']),
                         selected_native_call=selected[0]['call_id'], logits_saved=True, prefill_certified=True)
            finally:
                _tls.state = None
    @functools.wraps(old_prepare)
    def prepare(self, *args, **kwargs):
        result = old_prepare(self, *args, **kwargs)
        state = getattr(_tls, 'state', None)
        if state and state['runner'] is self:
            require(not self._fault28_completed and state['rows'] is None, 'Multiple prefills in one calibration case')
            scheduler = args[0] if args else kwargs['scheduler_output']
            counts = args[1] if len(args)>1 else kwargs['num_scheduled_tokens']
            state['rows'] = capture_rows(self, scheduler, counts, result, state['control'])
            state['selected'] = True
            write(root() / 'scheduler.json', state['rows'])
            emit('prefill_rows', rank=0, case_id=state['control']['case_id'], rows=state['rows'])
            if self._fault28_debugger:
                self._fault28_debugger.start(model=self.model, rank_id=0)
                state['observer_started'] = True
            state['bindings'] = install_native_observers(state)
        return result
    cls.load_model, cls.execute_model, cls._prepare_inputs = load, execute, prepare


class Loader:
    def __init__(self, original): self.original = original
    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, 'create_module') else None
    def exec_module(self, module):
        self.original.exec_module(module)
        patch(module)
    def __getattr__(self, name): return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == RUNNER:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec and spec.loader:
                spec.loader = Loader(spec.loader)
            return spec
        return None


def install():
    require(not sys.flags.optimize, 'Python optimized mode unsupported')
    if not any(isinstance(finder, Finder) for finder in sys.meta_path):
        require(RUNNER not in sys.modules, 'Install observer before importing the runner')
        sys.meta_path.insert(0, Finder())
