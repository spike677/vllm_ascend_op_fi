"""Stock HF Qwen3 SDPA calibration. No attention replacement or fault injection.

The official FLASH-only context is requested, but native FA coverage requires
CANN selected/launch evidence: NPU's overrideable backend is not fail-closed.
"""
import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import re
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import uuid


EXPECTED = {
    'transformers.models.qwen3.modeling_qwen3': 'fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab',
    'transformers.integrations.sdpa_attention': '87f933d1a2d8508df572da5c0748c6b24c22ff2b625796949957dcd86cc57564',
    'transformers.masking_utils': '7fd2bf34f3abc87953a500c0d53e3e633e5efba10aebcc4dc11422e5769092dd',
    'torch.nn.attention': '96afe3932505426b14f9511605f31764fa8eab8aa8b7af95fbe2d26f70e8faa6',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(values):
    return hashlib.sha256(json.dumps(list(values), separators=(',', ':')).encode()).hexdigest()


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def validate_case(case):
    require(case['batch_size'] in (1, 8) and case['prompt_length'] in (256, 512, 1024, 2048),
            'Unsupported bounded HF calibration case')
    require(case['target_index'] == 0 and case['layer_index'] == 1 and
            case.get('tensor_parallel_size', 1) == 1 and case.get('dtype', 'half') == 'half',
            'Expected target0/layer1/TP1/FP16')
    require(case.get('operator_variant', 'stock') in ('stock', 'private_vendor'), 'Unknown operator variant')
    require(type(case.get('forwards', 1)) is int and case.get('forwards', 1) in (1, 2), 'Expected one or two forwards')
    if case.get('operator_variant') == 'private_vendor':
        require(case.get('fa_mode') in ('off', 'on'), 'Private vendor requires explicit fa_mode=off/on')
    require(type(case['capture_msprobe']) is bool, 'capture_msprobe must be boolean')
    require(type(case['capture_full_inputs']) is bool, 'capture_full_inputs must be boolean')
    tokens = case['prompt_token_ids']
    require(len(tokens) == case['batch_size'] and all(isinstance(row, list) and len(row) == case['prompt_length']
            and all(type(v) is int and v >= 0 for v in row) for row in tokens), 'Malformed frozen tokens')
    require(all(row == tokens[0] for row in tokens), 'Calibration peers must duplicate the target')
    require([token_sha(row) for row in tokens] == case['prompt_token_sha256'] and
            token_sha(tokens[0]) == case['expected_target_token_sha256'] and
            case['expected_target_last_position'] == case['prompt_length'] - 1, 'Frozen token fingerprints differ')


def resolve_fa_mode_plan(case, forwards):
    requested = case.get('fa_mode_per_forward')
    private = case.get('operator_variant', 'stock') == 'private_vendor'
    explicit = requested is not None and requested != []
    if explicit:
        require(private and case.get('forwards', 1) == 2 and forwards == 2,
                'Per-forward FA mode requires private_vendor and exactly two declared/effective forwards')
        require(isinstance(requested, list) and len(requested) == 2 and
                all(mode in ('off', 'on') for mode in requested), 'Malformed per-forward FA mode plan')
        modes = list(requested)
    else:
        modes = [case['fa_mode']] * forwards if private else [None] * forwards
    return {'modes': modes, 'base_mode': case.get('fa_mode') if private else None,
            'source': 'case.fa_mode_per_forward' if explicit else 'case.fa_mode' if private else 'stock_no_private_scope',
            'scope': 'Only PrivateVendorScope.mode changes between full no-cache forwards; model/tensors/native arguments unchanged.'}


def field_differences(before, after, prefix=''):
    differences = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            path = prefix + '.' + str(key) if prefix else str(key)
            if key not in before or key not in after:
                differences.append({'field': path, 'before': before.get(key), 'after': after.get(key),
                                    'before_present': key in before, 'after_present': key in after})
            else:
                differences.extend(field_differences(before[key], after[key], path))
    elif before != after:
        differences.append({'field': prefix, 'before': before, 'after': after})
    return differences


def classify_input_differences(before, after):
    strict, normalizations = [], []
    allowed_fields = {prefix + '.' + name + '.actual_format'
                      for prefix in ('target_inputs', 'full_inputs') for name in ('query', 'key', 'value')}
    for difference in field_differences(before, after):
        if (difference['field'] in allowed_fields and difference.get('before') == 0 and
                difference.get('after') == 2 and 'before_present' not in difference):
            normalizations.append({**difference,
                'interpretation': 'Stock npu_format_cast NCHW(0) to ND(2) descriptor normalization; not a fault or numerical difference.'})
        else:
            strict.append(difference)
    return strict, normalizations


def bind_sdpa(args, kwargs):
    names = ('query', 'key', 'value', 'attn_mask', 'dropout_p', 'is_causal')
    require(len(args) <= len(names), 'Unexpected positional SDPA contract')
    values = {'attn_mask': None, 'dropout_p': 0.0, 'is_causal': False, 'scale': None, 'enable_gqa': False}
    values.update(zip(names, args))
    require(not set(names[:len(args)]).intersection(kwargs), 'Duplicate SDPA arguments')
    require(set(kwargs) <= set(names) | {'scale', 'enable_gqa'}, 'Unknown SDPA arguments')
    values.update(kwargs)
    require(all(k in values for k in names[:3]), 'Missing SDPA Q/K/V')
    return values


def statistics_config(output, forwards):
    return {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': list(range(forwards)),
            'dump_path': str(output / 'observation'), 'async_dump': False,
            'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                           'summary_mode': 'statistics', 'tensor_list': []}}


def statistics_receipt(output, forwards):
    receipt = {'schema': 'fault28.hf_sdpa.statistics_receipt.v1', 'status': 'CAPTURE_COMPLETED',
               'rank': 0, 'steps': [], 'config_sha256': sha(output / 'config.json'),
               'native_files': [], 'diagnostics_present': True,
               'diagnostic_stack_functions': ['Observer.record', 'Observer.identity'],
               'diagnostic_policy': 'Raw native dump/stack/construct are unmodified. Classify observer record/identity stack frames separately; do not discard real SDPA nodes merely because observe appears in their stack.',
               'actual_native_kernel_coverage_verified': False,
               'scope': 'Module plus functional SDPA statistics. This receipt is not native CANN FlashAttentionScore coverage or mechanism acceptance.'}
    forbidden = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
    require(not any(p.is_file() and p.suffix.lower() in forbidden for p in (output / 'observation').rglob('*')),
            'Tensor payload present in native statistics tree')
    for step in range(forwards):
        directory = output / 'observation' / ('step' + str(step)) / 'rank0'
        objects = {}
        for kind in ('dump', 'stack', 'construct'):
            path = directory / (kind + '.json')
            objects[kind] = json.loads(path.read_bytes())
            require(isinstance(objects[kind], dict) and objects[kind], 'Missing native ' + kind)
            receipt['native_files'].append({'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        dump, stack, construct = (objects[k] for k in ('dump', 'stack', 'construct'))
        require(dump.get('task') == 'statistics' and dump.get('level') == 'mix' and dump.get('data')
                and dump.get('dump_data_dir') is None, 'Wrong native statistics mode or tensor payload directory')
        data = dump['data']
        apis = [name for name in data if re.fullmatch(r'Functional\.scaled_dot_product_attention\.[0-9]+\.forward', name)]
        modules = [name for name in data if name.startswith('Module.')]
        require(len(apis) == 28 and len(modules) >= 28, 'Native functional SDPA/module coverage differs from full model')
        stack_index = {}
        for row in stack.values():
            require(isinstance(row, list) and len(row) == 2 and isinstance(row[0], list) and isinstance(row[1], list),
                    'Unexpected native stack schema')
            for name in row[0]:
                require(name not in stack_index, 'Duplicate stack mapping')
                stack_index[name] = row[1]
        require(all(stack_index.get(name) for name in apis + modules), 'Missing API/module callsite stack')
        parents = [construct.get(name) for name in apis]
        require(all(isinstance(parent, str) and parent.startswith('Module.') and parent in data for parent in parents)
                and len(set(parents)) == 28, 'SDPA calls do not have 28 distinct captured native module parents')
        diagnostic = [name for name, frames in stack_index.items()
                      if any(re.search(r'File .*hf_runtime_probe\.py, line [0-9]+, in (record|identity),', frame)
                             for frame in frames)]
        require(not set(apis).intersection(diagnostic), 'Real SDPA call misclassified as an observer readback')
        receipt['steps'].append({'step': step, 'dump_nodes': len(data), 'module_nodes': len(modules),
                                 'functional_sdpa_nodes': apis, 'functional_sdpa_module_parents': parents,
                                 'observer_diagnostic_node_count': len(diagnostic),
                                 'observer_diagnostic_nodes': diagnostic})
    return receipt


class PrivateVendorScope:
    """Scope the source-matched host library; never changes attention arguments."""
    KEYS = ('FAULT28_FA_MODE', 'FAULT28_FA_SCOPE_PID', 'FAULT28_FA_SCOPE_TID',
            'FAULT28_FA_SCOPE_LAYER', 'FAULT28_FA_SCOPE_PHASE', 'FAULT28_FA_SCOPE_FORWARD',
            'FAULT28_FA_SCOPE_TOKEN', 'FAULT28_FA_LOG_PATH')

    def __init__(self, mode, output):
        require(mode in ('off', 'on'), 'Invalid private host mode')
        self.mode, self.log_path, self.active = mode, output.resolve() / 'fa_host_events.jsonl', None

    def before(self, scope):
        require(self.active is None, 'Nested private host scope')
        if scope['layer_index'] != 1:
            return
        require(scope['prefill'] and scope['forward_index'] >= 0, 'Only explicitly labelled full prefill is scoped')
        token = f"f{scope['forward_index']}_layer1_sdpa_{uuid.uuid4().hex}"
        values = dict(zip(self.KEYS, (self.mode, str(os.getpid()), str(threading.get_native_id()), '1', 'prefill',
                                     str(scope['forward_index']), token, str(self.log_path))))
        self.active = (scope['call_id'], {key: os.environ.get(key) for key in self.KEYS})
        scope['host_scope'] = {'token': token, 'mode': self.mode, 'log_path': str(self.log_path),
                               'pid': os.getpid(), 'thread_id': threading.get_native_id(),
                               'host_callback_verified': False, 'environment_restored': False}
        os.environ.update(values)

    def after(self, scope):
        if self.active is None:
            return
        call_id, previous = self.active
        require(call_id == scope['call_id'], 'Private host scope identity changed')
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self.active = None
        require(all(os.environ.get(name) == value for name, value in previous.items()), 'Host scope restore failed')
        scope['host_scope']['environment_restored'] = True


class Observer:
    def __init__(self, torch, torch_npu, case, output, sources, before_call=None, after_call=None):
        self.torch, self.torch_npu = torch, torch_npu
        self.case, self.output, self.sources = case, output, sources
        self.active_layer = None
        self.layer_counts, self.calls, self.handles = {}, [], []
        self.functional, self.original, self.wrapper = None, None, None
        self.forward_index = None
        self.before_call, self.after_call = before_call, after_call
        require((before_call is None) == (after_call is None), 'Scope callbacks must be paired')

    def begin_forward(self, index):
        require(self.active_layer is None and len(self.calls) == index * 28, 'Prior forward is incomplete')
        self.forward_index = index

    def emit(self, event, **fields):
        row = {'event': event, 'pid': os.getpid(), 'thread_id': threading.get_native_id(),
               'epoch_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns(), **fields}
        line = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with (self.output / ('pid' + str(os.getpid()) + '.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    def identity(self, tensor):
        return {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
                'device': str(tensor.device), 'storage_offset': int(tensor.storage_offset()),
                'object_id': id(tensor), 'tensor_ptr': int(tensor.data_ptr()),
                'actual_format': int(self.torch_npu.get_npu_format(tensor))}

    def record(self, tensor, *, target=False, raw_name=None):
        if tensor is None:
            return None
        before = self.identity(tensor)
        selected = tensor[0:1] if target else tensor
        require(selected.numel() * selected.element_size() <= 128 * 1024 * 1024, 'Readback size exceeds bound')
        payload = selected.detach().to('cpu').contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes()
        row = {**before, 'hashed_slice': 'batch[0:1], all heads/positions' if target else 'entire logical tensor',
               'logical_sha256': hashlib.sha256(payload).hexdigest(), 'logical_bytes': len(payload),
               'byteorder': sys.byteorder, 'raw_order': 'C-contiguous logical elements'}
        require(self.identity(tensor) == before, 'Observation changed source metadata')
        if raw_name is not None:
            path = self.output / 'private_replay' / (raw_name + '.bin')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(payload)
            row.update(private_raw_file=str(path.relative_to(self.output)),
                       publication='PRIVATE_REPLAY_ONLY_EXCLUDE_FROM_GITHUB')
        return row

    def install(self, model, functional):
        self.functional, self.original = functional, functional.scaled_dot_product_attention
        for layer_index, layer in enumerate(model.model.layers):
            def enter(module, args, kwargs, index=layer_index):
                require(self.active_layer is None, 'Nested/reentrant attention unsupported')
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                require(hidden is not None and list(hidden.shape[:2]) ==
                        [self.case['batch_size'], self.case['prompt_length']], 'Whole prefill batch mapping differs')
                self.active_layer = index

            def leave(module, args, kwargs, returned, index=layer_index):
                self.active_layer = None

            self.handles.append(layer.self_attn.register_forward_pre_hook(enter, with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(leave, with_kwargs=True, always_call=True))

        @functools.wraps(self.original)
        def observed(*args, **kwargs):
            return self.observe(args, kwargs)

        self.wrapper = observed
        functional.scaled_dot_product_attention = observed

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.functional is not None:
            require(self.functional.scaled_dot_product_attention is self.wrapper, 'SDPA observer was replaced externally')
            self.functional.scaled_dot_product_attention = self.original

    def observe(self, args, kwargs):
        layer = self.active_layer
        require(layer is not None, 'Unexpected SDPA call outside the stock Qwen attention module')
        require(self.forward_index is not None and layer == len(self.calls) % 28 and
                self.layer_counts.get((self.forward_index, layer), 0) == 0,
                'Expected exactly one SDPA per layer, in native layer order')
        values = bind_sdpa(args, kwargs)
        q, k, v = [values[name] for name in ('query', 'key', 'value')]
        batch, sequence = self.case['batch_size'], self.case['prompt_length']
        require(list(q.shape) == [batch, 16, sequence, 128] and
                list(k.shape) == list(v.shape) and list(k.shape) in
                ([batch, 8, sequence, 128], [batch, 16, sequence, 128]), 'Unexpected BNSD/GQA mapping')
        require(all(x.dtype == self.torch.float16 and x.device.type == 'npu' for x in (q, k, v)),
                'Actual Q/K/V must be FP16 NPU tensors')
        require(values['attn_mask'] is None or values['attn_mask'].dtype == self.torch.bool,
                'Mask does not satisfy native FA source conditions')
        require(values['dropout_p'] == 0.0, 'Inference dropout must be zero')
        require(not (values['is_causal'] and values['attn_mask'] is not None), 'Invalid SDPA causal/mask combination')
        native_index = len(self.calls)
        call_id = f'pid{os.getpid()}:r0:n{native_index}'
        fields = {'case_id': self.case['case_id'], 'rank': 0, 'call_id': call_id,
                  'native_index': native_index, 'forward_index': self.forward_index, 'layer_index': layer,
                  'layer_name': f'model.layers.{layer}.self_attn',
                  'api': 'torch.nn.functional.scaled_dot_product_attention',
                  'source_sha256': self.sources}
        target_before = {name: self.record(values[name], target=True) for name in ('query', 'key', 'value')}
        mask_before = self.record(values['attn_mask'])
        full_before, raw_before = {}, {}
        if layer == self.case['layer_index']:
            full_before = {name: self.record(values[name],
                raw_name=f'forward{self.forward_index}_layer{layer}_{name}' if self.case['capture_full_inputs'] else None)
                for name in ('query', 'key', 'value', 'attn_mask')}
            if self.case['capture_full_inputs']:
                raw_before = full_before
        self.torch_npu.npu.synchronize()
        scope = {**fields, 'input_layout': 'BNSD', 'query_shape': list(q.shape), 'key_shape': list(k.shape),
                 'value_shape': list(v.shape), 'dtype': str(q.dtype), 'is_causal': values['is_causal'],
                 'mask_present': values['attn_mask'] is not None, 'prefill': True}
        entered = False
        try:
            if self.before_call is not None:
                require(self.before_call(scope) is None, 'Scope callback must not return replacement data')
            if scope.get('host_scope'):
                fields['fa_scope_token'] = scope['host_scope']['token']
            self.emit('native_api_enter', **fields)
            entered = True
            with self.torch.profiler.record_function(f'fault28_hf_sdpa_layer{layer}'):
                returned = self.original(*args, **kwargs)
            self.torch_npu.npu.synchronize()
        except BaseException:
            if entered:
                self.emit('native_api_exit', status='error', **fields)
            raise
        finally:
            if self.after_call is not None:
                require(self.after_call(scope) is None, 'Scope callback must not return replacement data')
        self.emit('native_api_exit', status='returned', **fields)
        target_after = {name: self.record(values[name], target=True) for name in ('query', 'key', 'value')}
        mask_after = self.record(values['attn_mask'])
        integrity_before = {'target_inputs': target_before, 'attention_mask': mask_before}
        integrity_after = {'target_inputs': target_after, 'attention_mask': mask_after}
        differences, format_normalizations = classify_input_differences(integrity_before, integrity_after)
        if differences:
            write(self.output / 'input_integrity_failure.json', {**fields,
                'status': 'INPUT_INTEGRITY_FAILED', 'before': integrity_before, 'after': integrity_after,
                'field_differences': differences, 'allowed_format_normalizations': format_normalizations,
                'scope': 'Exact diagnostic records preserved; no assumption that metadata changes imply numerical changes.'})
            raise RuntimeError('Target SDPA inputs/mask changed: ' + ', '.join(row['field'] for row in differences[:12]))
        full_after = {name: self.record(values[name]) for name in ('query', 'key', 'value', 'attn_mask')} if full_before else {}
        comparable_full_before = {name: None if before is None else
            {k: v for k, v in before.items() if k not in ('private_raw_file', 'publication')}
            for name, before in full_before.items()}
        full_differences, full_normalizations = classify_input_differences(
            {'full_inputs': comparable_full_before}, {'full_inputs': full_after})
        format_normalizations.extend(full_normalizations)
        if full_differences:
            write(self.output / 'input_integrity_failure.json', {**fields,
                'status': 'FULL_BATCH_INPUT_INTEGRITY_FAILED', 'before': {'full_inputs': comparable_full_before},
                'after': {'full_inputs': full_after}, 'field_differences': full_differences,
                'allowed_format_normalizations': full_normalizations})
            raise RuntimeError('Selected layer full-batch input changed: ' + ', '.join(row['field'] for row in full_differences[:12]))
        require(list(returned.shape) == list(q.shape) and returned.dtype == q.dtype, 'Unexpected native SDPA output')
        target_output = self.record(returned, target=True)
        raw_output = self.record(returned, raw_name=f'forward{self.forward_index}_layer{layer}_output') if raw_before else None
        row = {**fields, 'status': 'returned', 'entrypoint_kind': 'PyTorch SDPA; actual CANN operator requires native logs',
               'original_api_calls': 1, 'original_return_object_id': id(returned),
               'actual_layout': 'BNSD', 'target_mapping_certified': True,
               'target_mapping': {'batch_slot': 0, 'last_token_position': sequence - 1,
                                  'token_sha256': self.case['expected_target_token_sha256']},
               'scalar_arguments': {name: values[name] for name in ('dropout_p', 'is_causal', 'scale', 'enable_gqa')},
               'target_inputs_before': target_before, 'target_inputs_after': target_after,
               'attention_mask': mask_before, 'target_output': target_output,
               'input_format_normalizations': format_normalizations,
               'selected_layer_full_inputs_before': full_before, 'selected_layer_full_inputs_after': full_after,
               'selected_layer_private_inputs': raw_before, 'selected_layer_private_output': raw_output,
               'host_scope': scope.get('host_scope'),
               'native_fa_coverage_verified': False}
        path = self.output / 'native_calls' / f'call_{native_index:03d}.json'
        write(path, row)
        self.calls.append({k: row[k] for k in ('call_id', 'native_index', 'forward_index', 'layer_index', 'layer_name', 'api', 'status')} |
                          {'file': str(path.relative_to(self.output)), 'sha256': sha(path), 'target_mapping_certified': True})
        self.layer_counts[(self.forward_index, layer)] = 1
        return returned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forwards', type=int, choices=(1, 2), default=None,
                        help='Override case.forwards (default one); independent no-cache full prefill forwards')
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    validate_case(case)
    forwards = args.forwards if args.forwards is not None else case.get('forwards', 1)
    mode_plan = resolve_fa_mode_plan(case, forwards)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'fault28.hf_sdpa.calibration.v1', 'status': 'STARTING', 'case_id': case['case_id'],
              'framework': 'stock_transformers', 'operator_variant': case.get('operator_variant', 'stock'),
              'batch_size': case['batch_size'],
              'prompt_length': case['prompt_length'], 'target_index': 0, 'layer_index': 1,
              'source_case_sha256': sha(args.case_file), 'source_sha256': {'hf_runtime_probe.py': sha(__file__)},
              'checkpoint_conversion': 'Original BF16 checkpoint loaded by stock Transformers with dtype=torch.float16.',
              'scope': 'Distinct stock-HF model path from vLLM; host variant/mode explicitly recorded. Completion is not fault PASS.',
              'native_fa_coverage_verified': False, 'mechanism_accepted': False,
              'native_profile_status': 'REQUIRES_CANN_SELECTED_AND_LAUNCHED_FA_FOR_EVERY_LAYER',
              'capture_msprobe': case['capture_msprobe'], 'capture_full_inputs': case['capture_full_inputs'],
              'forward_count': forwards, 'forward_count_cli_override': args.forwards, 'fa_mode_plan': mode_plan,
              'forward_count_case': case.get('forwards', 1), 'start_epoch': time.time(),
              'native_calls': [], 'forward_results': []}
    observer = None
    debugger = None
    debugger_active = False
    observer_installed = False
    save = lambda: write(args.output / 'result.json', result)
    save()
    try:
        require(os.environ.get('ACLNN_CACHE_LIMIT') == '0', 'ACLNN_CACHE_LIMIT=0 required')
        require(not os.environ.get('LD_PRELOAD'), 'External LD_PRELOAD forbidden')
        custom = os.environ.get('ASCEND_CUSTOM_OPP_PATH')
        private_scope = None
        if case.get('operator_variant', 'stock') == 'stock':
            require(not custom, 'Stock HF probe forbids external ASCEND_CUSTOM_OPP_PATH')
        else:
            require(custom and custom == case.get('approved_custom_opp_path'), 'Private vendor path is not explicitly approved')
            manifest = Path(case['approved_vendor_manifest_file'])
            require(manifest.is_file() and sha(manifest) == case['approved_vendor_manifest_sha256'],
                    'Private vendor manifest fingerprint differs')
            result['private_vendor'] = {'approved_custom_opp_path': custom, 'manifest_file': str(manifest),
                                        'manifest_sha256': sha(manifest), 'fa_mode': case['fa_mode'],
                                        'scope_only': 'layer1/full-prefill; host independently checks B8/S2048/Q16/KV8/D128'}
            launch_path = Path(os.environ.get('FAULT28_VENDOR_LAUNCH_RECEIPT', ''))
            expected_launch = args.output.with_name(args.output.name + '.vendor_launch.json')
            require(launch_path == expected_launch and launch_path.is_file() and
                    sha(launch_path) == os.environ.get('FAULT28_VENDOR_LAUNCH_RECEIPT_SHA256'),
                    'Verified fresh-process vendor launcher receipt is missing')
            launch = json.loads(launch_path.read_bytes())
            require(launch.get('schema') == 'fault28.fa.vendor_launch.v1' and
                    launch.get('case_sha256') == result['source_case_sha256'] and
                    launch.get('probe_sha256') == sha(__file__) and
                    launch.get('verified_vendor', {}).get('manifest_sha256') == sha(manifest),
                    'Vendor launch receipt does not match this exact case/probe/manifest')
            actual_loader = {name: os.environ.get(name) for name in ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD')}
            require(all(launch['child_environment'].get(name) == value for name, value in actual_loader.items()),
                    'Fresh-process loader environment differs from verified launch receipt')
            required_library = custom + '/op_api/lib'
            require(actual_loader['LD_LIBRARY_PATH'] == required_library or
                    actual_loader['LD_LIBRARY_PATH'].startswith(required_library + ':'),
                    'Private op_api library must precede inherited library paths at process start')
            result['private_vendor'].update(launcher_receipt=str(launch_path), launcher_receipt_sha256=sha(launch_path),
                                             actual_loader_environment=actual_loader)
            private_scope = PrivateVendorScope(case['fa_mode'], args.output)
        require(all(os.environ.get(name) is None for name in PrivateVendorScope.KEYS),
                'Preexisting FA host scope variables forbidden')
        for name in ('DCAT_DQ_V2_ENABLE', 'DCAT_V2_HOOK', 'BATCH_OUTLIER_ENABLED', 'BATCH_INPUT_ENABLED', 'FAULT28_CALIBRATION'):
            require(os.environ.get(name, '0') == '0', 'Another experiment hook is enabled: ' + name)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        model_path = Path(case.get('model', '/models/Qwen3-0.6B'))
        config = json.loads((model_path / 'config.json').read_bytes())
        require(config.get('quantization_config') is None and
                (config.get('dtype') or config.get('torch_dtype')) in ('bfloat16', 'torch.bfloat16'),
                'Expected original unquantized BF16 checkpoint')
        require(config.get('num_hidden_layers') == 28 and config.get('num_attention_heads') == 16 and
                config.get('num_key_value_heads') == 8 and config.get('head_dim') == 128,
                'Unexpected Qwen3 model architecture')
        result['checkpoint_config'] = {'sha256': sha(model_path / 'config.json'), 'original_dtype': 'bfloat16'}
        from collector_stage import select_collector, verify_loaded
        collector_selection = select_collector(case)
        result['statistics_collector'] = collector_selection
        result['source_sha256']['collector_stage.py'] = sha(Path(__file__).with_name('collector_stage.py'))
        import torch
        import torch_npu
        from torch.nn import functional as functional
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        for module_name, expected in EXPECTED.items():
            path = Path(importlib.import_module(module_name).__file__)
            require(sha(path) == expected, 'Installed source changed: ' + module_name)
            result.setdefault('native_sources', {})[module_name] = {'path': str(path), 'sha256': expected}
        result['versions'] = {name: importlib.metadata.version(name) for name in ('torch', 'torch-npu', 'transformers')}
        result['environment'] = {name: os.environ.get(name) for name in
                                 ('ACLNN_CACHE_LIMIT', 'ASCEND_GLOBAL_LOG_LEVEL', 'ASCEND_SLOG_PRINT_TO_STDOUT',
                                  'TORCH_NPU_USE_COMPATIBLE_IMPL')}
        for name, expected in case['tokenizer_sources'].items():
            require(sha(model_path / name) == expected, 'Tokenizer/config fingerprint differs: ' + name)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        vocabulary = set(tokenizer.get_vocab().values())
        require(all(v in vocabulary for row in case['prompt_token_ids'] for v in row), 'Tokens outside actual vocabulary')
        torch.manual_seed(42)
        torch_npu.npu.set_device(0)
        model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=torch.float16,
                    attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().to('npu:0')
        require(model.config._attn_implementation == 'sdpa' and len(model.model.layers) == 28,
                'Stock Qwen3 SDPA configuration was not applied')
        require(all(parameter.dtype == torch.float16 for parameter in model.parameters()), 'Unexpected parameter dtype')
        observer = Observer(torch, torch_npu, case, args.output,
                            {k: v['sha256'] for k, v in result['native_sources'].items()},
                            private_scope.before if private_scope else None,
                            private_scope.after if private_scope else None)
        if case['capture_msprobe']:
            from msprobe.pytorch import PrecisionDebugger
            collector_module = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
            collector_selection['actual_loaded'] = verify_loaded(collector_selection, collector_module)
            write(args.output / 'collector_provenance.json', collector_selection)
            require(importlib.metadata.version('mindstudio-probe') == '26.1.0', 'Unreviewed msprobe version')
            debugger_source = Path(inspect.getsourcefile(PrecisionDebugger))
            require(sha(debugger_source) == '43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb',
                    'Debugger source changed')
            support_file = debugger_source.parents[1] / 'api_dump' / 'support_wrap_ops.yaml'
            require(sha(support_file) == '5fefa502efd21c9793a286bc1d06f2fdf474876da3c3d96dd474570f8c6609e5',
                    'Native API support list changed')
            statistics_cfg = statistics_config(args.output, forwards)
            if collector_selection['precision'] == 'high':
                statistics_cfg['precision'] = 'high'
            write(args.output / 'config.json', statistics_cfg)
            debugger = PrecisionDebugger(config_path=str(args.output / 'config.json'))
            result['msprobe'] = {'version': '26.1.0', 'debugger_source_sha256': sha(debugger_source),
                'support_wrap_ops_sha256': sha(support_file), 'custom_api_registration': [],
                'functional_sdpa_supported_natively': True, 'diagnostics_present': True,
                'diagnostic_stack_functions': ['Observer.record', 'Observer.identity'],
                'scope': 'start immediately before each model forward; stop/step before logits readback. In-forward observer readbacks remain in raw native trace, identified by stack.'}
        input_ids = torch.tensor(case['prompt_token_ids'], dtype=torch.long, device='npu:0')
        require(input_ids.cpu().tolist() == case['prompt_token_ids'], 'Actual input batch differs from frozen tokens')
        result['options'] = {'attn_implementation': 'sdpa', 'dtype': 'float16', 'use_cache': False,
                             'logits_to_keep': 1, 'dropout': 0, 'batch_size': case['batch_size'],
                             'full_prefill_forward_count': forwards, 'torch_compile': False,
                             'requested_backend': 'SDPBackend.FLASH_ATTENTION'}
        save()
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            flags = {name: bool(getattr(torch._C, '_get_' + name + '_sdp_enabled')()) for name in
                     ('flash', 'math', 'mem_efficient', 'cudnn', 'overrideable')}
            result['actual_sdpa_context_flags'] = flags
            require(flags['flash'] and not any(flags[n] for n in ('math', 'mem_efficient', 'cudnn')),
                    'Official FLASH-only context was not applied')
            result['context_limit'] = 'NPU overrideable implementation may ignore disabled math on fallback; verify native logs.'
            save()
            for forward_index in range(forwards):
                if private_scope is not None:
                    require(private_scope.active is None, 'Previous host scope is still active at forward boundary')
                    private_scope.mode = mode_plan['modes'][forward_index]
                if debugger is not None:
                    debugger.start(model=model, rank_id=0)
                    debugger_active = True
                if not observer_installed:
                    # Install after msprobe so self.original is its official functional API wrapper.
                    observer.install(model, functional)
                    observer_installed = True
                require(functional.scaled_dot_product_attention is observer.wrapper,
                        'Capture lifecycle replaced our native observer binding')
                observer.begin_forward(forward_index)
                try:
                    replies = model(input_ids=input_ids, attention_mask=None, use_cache=False,
                                    logits_to_keep=1, return_dict=True)
                    torch_npu.npu.synchronize()
                finally:
                    if debugger_active:
                        debugger.stop()
                        debugger_active = False
                        debugger.step()
                require(len(observer.calls) == 28 * (forward_index + 1) and
                        all(observer.layer_counts.get((forward_index, i)) == 1 for i in range(28)),
                        'Every layer must execute exactly one SDPA in each full prefill')
                require(list(replies.logits.shape) == [case['batch_size'], 1, config['vocab_size']],
                        'Unexpected last-token logits mapping')
                require(replies.past_key_values is None and input_ids.cpu().tolist() == case['prompt_token_ids'],
                        'Forward retained KV cache or changed input tokens')
                forward_row = {'forward_index': forward_index, 'fa_mode': mode_plan['modes'][forward_index], 'target_logits': observer.record(replies.logits, target=True),
                               'requests': [], 'all_requests_in_one_prefill': True}
                if case['capture_full_inputs']:
                    forward_row['private_logits'] = observer.record(replies.logits, raw_name=f'forward{forward_index}_last_token_logits')
                logits = replies.logits[:, -1].detach().float().cpu()
                for index, row in enumerate(logits):
                    scores, ids = row.log_softmax(dim=-1).topk(20)
                    forward_row['requests'].append({'prompt_index': index, 'batch_slot': index,
                        'prompt_token_sha256': case['prompt_token_sha256'][index],
                        'last_token_position': case['prompt_length'] - 1,
                        'next_token_argmax': int(row.argmax()),
                        'top20_logprobs': dict(zip(map(str, ids.tolist()), scores.tolist()))})
                result['forward_results'].append(forward_row)
                result['native_calls'] = observer.calls
                save()
        if debugger is not None:
            receipt = statistics_receipt(args.output, forwards)
            receipt['statistics_collector'] = collector_selection
            receipt['collector_provenance_sha256'] = sha(args.output / 'collector_provenance.json')
            write(args.output / 'receipt.json', receipt)
            result['msprobe_receipt_sha256'] = sha(args.output / 'receipt.json')
        observer.restore()
        result.update(status='CAPTURE_COMPLETED', native_calls=observer.calls, all_requests_in_one_prefill=True,
                      prefill_certified=True, end_epoch=time.time())
        observer = None
        save()
        print(json.dumps({'status': result['status'], 'case_id': case['case_id'], 'native_calls': 28 * forwards,
                          'framework': 'stock_transformers', 'native_fa_coverage_verified': False}), flush=True)
        return 0
    except BaseException:
        result.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), end_epoch=time.time())
        if observer is not None:
            result['native_calls'] = observer.calls
        save()
        print(result['traceback'], file=sys.stderr)
        return 1
    finally:
        if debugger_active:
            debugger.stop()
        if observer is not None:
            observer.restore()


if __name__ == '__main__':
    raise SystemExit(main())
