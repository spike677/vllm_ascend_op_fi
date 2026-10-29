"""One original native FA call on fixed captured-model inputs; not a model forward.

Controller supplies process-start vendor loader paths. Strategy labels are intent;
actual native host/profile evidence must independently establish execution.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import threading
import time
import traceback
import uuid

PROFILES = {'base': (128, 8), 'm_only': (64, 8), 'r_only': (128, 4), 'joint': (64, 4)}
AUTO_PROFILES = {'batch_m': (64, 8), 'batch_r': (128, 4), 'batch_joint': (64, 4)}
REFERENCE_QUERIES = [0, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047]
SCOPE_KEYS = tuple('FAULT28_CAUSAL_' + key for key in
                   ('PROFILE', 'SCOPE_PID', 'SCOPE_TID', 'SCOPE_PHASE', 'SCOPE_BATCH', 'SCOPE_TOKEN', 'LOG_PATH'))
ORIGINAL_CALL_SHA = '458a683a616da397bd1eeba47b7348d0e2bf78c253ea43c158f7f393bb67c0c8'
SUPPORT_SHA = '5fefa502efd21c9793a286bc1d06f2fdf474876da3c3d96dd474570f8c6609e5'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def validate_case(case):
    require(case.get('schema') == 'fault28.causal.case.v1', 'Unexpected replay case schema')
    require(case.get('operator_variant') in ('stock', 'private_vendor'), 'Explicit operator variant required')
    if case['operator_variant'] == 'stock':
        require(case.get('stock_bridge') is True and case.get('profile') == 'base' and case.get('batch_size') in (1, 8),
                'Stock path is limited to the source-capture bridge')
    else:
        require(not case.get('stock_bridge'), 'Private vendor cannot claim stock bridge')
    require(case.get('profile') in PROFILES | AUTO_PROFILES and case.get('batch_size') in (1, 8), 'Unsupported factor')
    requested = (PROFILES | AUTO_PROFILES)[case['profile']]
    if case['profile'] in AUTO_PROFILES and case['batch_size'] == 1:
        requested = (128, 8)
    require((case.get('requested_m'), case.get('requested_n_ratio')) == requested, 'Profile intent differs')
    require(type(case.get('repeat_index')) is int and case['repeat_index'] in (0, 1), 'Expected exactly two repeats')
    require(case.get('native_calls_per_process') == 1, 'Every replay case must make exactly one native call')
    require(case.get('capture_private_target_output') is True and case.get('private_last_query_comparison') is True,
            'Full target output and final-query comparison are required private evidence')
    require(type(case.get('capture_msprobe')) is bool, 'Explicit capture choice required')
    require(case.get('reference_query_indices') == REFERENCE_QUERIES, 'FP64 query boundaries must be preregistered')


def validate_vendor(case):
    require(os.environ.get('ACLNN_CACHE_LIMIT') == '0', 'Native host cache must be disabled at process start')
    require(not os.environ.get('LD_PRELOAD'), 'External LD_PRELOAD forbidden')
    if case['operator_variant'] == 'stock':
        require(not os.environ.get('ASCEND_CUSTOM_OPP_PATH') and not any(os.environ.get(k) for k in SCOPE_KEYS),
                'Stock bridge requires original library and no policy scope')
        require(not any(case.get(key) for key in ('approved_vendor_manifest_file', 'approved_vendor_manifest_sha256',
                                                 'approved_custom_opp_path')), 'Stock bridge cannot approve a private vendor')
        return {'operator_variant': 'stock', 'custom_opp_path': None, 'scope': 'Original native API bridge to captured model call'}
    path = Path(case['approved_vendor_manifest_file'])
    require(path.is_file() and sha(path) == case['approved_vendor_manifest_sha256'], 'Vendor manifest differs')
    manifest = json.loads(path.read_bytes())
    vendor = Path(case['approved_custom_opp_path'])
    require(vendor.is_absolute() and vendor.resolve(strict=True) == vendor and not vendor.is_symlink(), 'Vendor root is not direct')
    declared = Path(manifest['vendor_root'])
    declared = declared if declared.is_absolute() else Path('/audit') / declared
    require(declared == vendor and os.environ.get('ASCEND_CUSTOM_OPP_PATH') == str(vendor), 'Vendor loader path differs')
    lib = str(vendor / 'op_api/lib')
    require(os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == lib, 'Vendor op_api must lead loader path before Python starts')
    names, total = set(), 0
    for row in manifest['artifacts']:
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() not in names, 'Unsafe vendor artifact')
        names.add(rel.as_posix())
        file = vendor / rel
        require(file.is_file() and file.resolve(strict=True).is_relative_to(vendor), 'Escaping vendor file')
        require(file.stat().st_size == row['bytes'] and sha(file) == row['sha256'], 'Vendor artifact differs')
        total += row['bytes']
    actual = {p.relative_to(vendor).as_posix() for p in vendor.rglob('*') if p.is_file()}
    require(actual == names and 0 < len(names) <= 20000 and total <= 8 * 1024**3, 'Vendor inventory differs')
    return {'manifest_file': str(path), 'manifest_sha256': sha(path), 'artifact_count': len(names),
            'source_commit': manifest.get('source_commit'), 'policy_source_manifest_sha256': manifest.get('policy_source_manifest_sha256'),
            'actual_custom_opp_path': str(vendor), 'actual_first_library_path': lib}


def load_frozen_inputs(torch, case):
    path = Path(case['input_manifest_file'])
    require(path.is_file() and sha(path) == case['input_manifest_sha256'], 'Input manifest differs')
    manifest = json.loads(path.read_bytes())
    require(manifest['schema'] == 'fault28.causal.inputs.v1' and manifest['source_call_sha256'] == ORIGINAL_CALL_SHA,
            'Wrong original model capture provenance')
    require(sha(path.parent / manifest['source_call_file']) == ORIGINAL_CALL_SHA, 'Copied original call metadata differs')
    values = {}
    for name in ('query', 'key', 'value'):
        row = manifest['inputs'][name]
        raw = (path.parent / row['file']).read_bytes()
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'Captured target input differs')
        require(row['shape'] == [1, 16 if name == 'query' else 8, 2048, 128] and row['dtype'] == 'torch.float16'
                and row['byteorder'] == 'little', 'Unsupported original input geometry')
        target = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(row['shape']).clone()
        batch = target.repeat(case['batch_size'], 1, 1, 1)
        if name == 'value':
            batch = batch.transpose(1, 2).contiguous().transpose(1, 2)
        require(list(batch.stride()) == row['source_stride'], 'Replay stride differs from original captured input')
        values[name] = batch
    return manifest, values


def raw_bytes(torch, tensor):
    return tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()


def snapshot(torch, torch_npu, tensor, batch_sliced=True):
    data = raw_bytes(torch, tensor)
    row = {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
           'device': str(tensor.device), 'storage_offset': tensor.storage_offset(), 'object_id': id(tensor),
           'data_ptr': tensor.data_ptr(), 'version': tensor._version,
           'actual_format': int(torch_npu.get_npu_format(tensor)) if tensor.device.type == 'npu' else None,
           'full_sha256': digest(data), 'bytes': len(data), 'byteorder': 'little', 'raw_order': 'C-contiguous logical elements'}
    if batch_sliced:
        size = len(data) // tensor.shape[0]
        row.update(target_sha256=digest(data[:size]), per_batch_sha256=[digest(data[i * size:(i + 1) * size]) for i in range(tensor.shape[0])])
    return row


def input_integrity(before, after):
    normalizations = []
    for name in before:
        a, b = dict(before[name]), dict(after[name])
        old, new = a.pop('actual_format'), b.pop('actual_format')
        require(a == b, 'Native call changed original input bytes/identity/geometry: ' + name)
        require(old == new or name in ('query', 'key', 'value') and old == 0 and new == 2,
                'Unreviewed input format transition: ' + name)
        if old != new:
            normalizations.append({'input': name, 'before': old, 'after': new, 'logical_values_unchanged': True})
    return normalizations


def reference_queries(torch, cpu_inputs, scale):
    # Independent CPU FP64 references at preregistered causal boundaries; not a native output substitute.
    q, k, v = (cpu_inputs[name][0].double() for name in ('query', 'key', 'value'))
    effective_scale = struct.unpack('<f', struct.pack('<f', scale))[0]
    output = []
    for query_index in REFERENCE_QUERIES:
        heads = []
        for head in range(16):
            scores = torch.mv(k[head // 2, :query_index + 1], q[head, query_index]) * effective_scale
            probabilities = torch.softmax(scores, dim=0)
            heads.append(torch.matmul(probabilities, v[head // 2, :query_index + 1]))
        output.append(torch.stack(heads))
    return torch.stack(output), effective_scale


def statistics_config(output, precision):
    result = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0], 'dump_path': str(output / 'observation'),
              'async_dump': False, 'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
              'summary_mode': 'statistics', 'tensor_list': []}}
    if precision == 'high':
        result['precision'] = 'high'
    return result


def statistics_receipt(output, collector):
    root = output / 'observation'
    forbidden = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
    require(not any(p.is_file() and p.suffix.lower() in forbidden for p in root.rglob('*')), 'Tensor payload in statistics')
    rows, objects = [], {}
    for kind in ('dump', 'stack', 'construct'):
        path = root / 'step0/rank0' / (kind + '.json')
        objects[kind] = json.loads(path.read_bytes())
        rows.append({'path': path.relative_to(output).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    dump = objects['dump']
    require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('dump_data_dir') is None, 'Wrong native statistics mode')
    # Installed Const.NPU_API_TYPE_PREFIX is exactly "NPU"; do not relabel the emitted API.
    apis = [name for name in dump['data'] if re.fullmatch(r'NPU\.npu_fusion_attention_v3\.\d+\.forward', name)]
    require(len(apis) == 1 and any(name.startswith('Module.') for name in dump['data']), 'Expected one native public API plus replay module statistics')
    parent = objects['construct'].get(apis[0])
    require(isinstance(parent, str) and parent.startswith('Module.') and '.ReplayModule.forward.' in parent
            and parent in dump['data'], 'Original native API must belong to the captured ReplayModule child')
    return {'schema': 'fault28.causal.statistics_receipt.v1', 'scope': 'Original operator replay, not a model token/forward trace',
            'rank': 0, 'steps': [0], 'native_files': rows, 'native_api_nodes': apis,
            'config_sha256': sha(output / 'config.json'), 'statistics_collector': collector,
            'collector_provenance_sha256': sha(output / 'collector_provenance.json'), 'native_cann_coverage_verified': False}


class HostScope:
    def __init__(self, case, output):
        self.values = dict(zip(SCOPE_KEYS, (case['profile'], str(os.getpid()), str(threading.get_native_id()),
            'operator_replay', str(case['batch_size']), 'replay_' + uuid.uuid4().hex, str(output / 'host_events.jsonl'))))
        self.prior = {}

    def __enter__(self):
        self.prior = {key: os.environ.get(key) for key in SCOPE_KEYS}
        require(all(value is None for value in self.prior.values()), 'Inherited causal scope forbidden')
        os.environ.update(self.values)
        return self.values

    def __exit__(self, *_):
        for key, value in self.prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    validate_case(case)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'fault28.causal.replay.v1', 'status': 'STARTING', 'case_id': case['case_id'],
              'profile': case['profile'], 'operator_variant': case['operator_variant'],
              'batch_size': case['batch_size'], 'repeat_index': case['repeat_index'],
              'source_case_sha256': sha(args.case_file), 'replay_source_sha256': sha(__file__),
              'input_manifest_sha256': case['input_manifest_sha256'], 'original_api_calls': 0,
              'execution_kind': 'isolated_original_native_operator_replay', 'model_forward': False,
              'origin': 'Qwen3 captured layer1 input provenance only; this process does not execute a model layer.',
              'requested_m': case['requested_m'], 'requested_n_ratio': case['requested_n_ratio'],
              'native_profile_verified': False, 'mechanism_accepted': False, 'start_epoch': time.time()}
    debugger, active = None, False
    save = lambda: write(args.output / 'result.json', result)
    save()
    try:
        result['vendor'] = validate_vendor(case)
        if case['operator_variant'] == 'private_vendor':
            receipt_path = args.output.with_name(args.output.name + '.vendor_launch.json')
            require(os.environ.get('FAULT28_VENDOR_LAUNCH_RECEIPT') == str(receipt_path)
                    and receipt_path.is_file() and sha(receipt_path) == os.environ.get('FAULT28_VENDOR_LAUNCH_RECEIPT_SHA256'),
                    'Fresh-process vendor launcher receipt is missing')
            launch = json.loads(receipt_path.read_bytes())
            require(launch.get('schema') == 'fault28.causal.vendor_launch.v1'
                    and launch.get('case_sha256') == result['source_case_sha256']
                    and launch.get('probe_sha256') == sha(__file__)
                    and launch.get('verified_vendor', {}).get('manifest_sha256') == case['approved_vendor_manifest_sha256'],
                    'Launcher receipt differs from actual case/probe/vendor')
            require(all(launch['child_environment'].get(k) == os.environ.get(k)
                        for k in ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')),
                    'Actual process loader environment differs from verified launcher')
            result['vendor'].update(launcher_receipt=str(receipt_path), launcher_receipt_sha256=sha(receipt_path))
        from collector_stage import select_collector, verify_loaded
        collector = select_collector(case)
        result['statistics_collector'] = collector
        import torch
        import torch_npu
        require(sys.byteorder == 'little', 'Raw input requires little-endian host')
        result['versions'] = {n: importlib.metadata.version(n) for n in ('torch', 'torch-npu')}
        manifest, cpu = load_frozen_inputs(torch, case)
        reference, native_scale = reference_queries(torch, cpu, manifest['native_kwargs']['scale'])
        result['fp64_reference_contract'] = {'query_indices': REFERENCE_QUERIES, 'batch_slot': 0, 'heads': 16, 'head_dim': 128,
            'kv_group_rule': 'query_head//2', 'effective_scale_float32': native_scale, 'compute_device': 'CPU',
            'dtype': 'float64', 'causal_keys': 'keys 0 through query_index inclusive', 'not_a_native_output_proxy': True}
        torch_npu.npu.set_device(0)
        inputs = {}
        for name, value in cpu.items():
            tensor = torch.empty_strided(value.shape, value.stride(), dtype=torch.float16, device='npu:0')
            tensor.copy_(value)
            inputs[name] = tensor
        mask_cpu = torch.triu(torch.ones((2048, 2048), dtype=torch.bool), diagonal=1)
        mask = mask_cpu.to('npu:0')
        kwargs = dict(manifest['native_kwargs'])
        kwargs.update(pse=None, padding_mask=None, atten_mask=mask, prefix=None,
                      actual_seq_qlen=None, actual_seq_kvlen=None, sink=None)
        result['native_kwargs'] = {k: v for k, v in kwargs.items() if k != 'atten_mask'}
        result['inputs_before'] = {n: snapshot(torch, torch_npu, x) for n, x in inputs.items()}
        result['mask_before'] = snapshot(torch, torch_npu, mask, batch_sliced=False)
        for name, before in result['inputs_before'].items():
            row = manifest['inputs'][name]
            require(before['full_sha256'] == row['expected_full_sha256'][str(case['batch_size'])]
                    and before['per_batch_sha256'] == [row['sha256']] * case['batch_size'], 'Actual NPU input differs from frozen repeated target')
        class ReplayModule(torch.nn.Module):
            def forward(self, query, key, value):
                return torch_npu.npu_fusion_attention_v3(query, key, value, **kwargs)
        module = ReplayModule().eval()
        # msprobe 26.1 skips the root passed to start(); register the real replay as a child.
        # This holder is never called and adds no numerical forward or native API invocation.
        capture_root = torch.nn.Module()
        capture_root.add_module('replay', module)
        result['native_dispatch_schema'] = str(torch.ops.npu.npu_fusion_attention_v3.default._schema)
        if case['capture_msprobe']:
            from msprobe.pytorch import PrecisionDebugger
            processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
            collector['actual_loaded'] = verify_loaded(collector, processor)
            support = Path(inspect.getsourcefile(PrecisionDebugger)).parents[1] / 'api_dump/support_wrap_ops.yaml'
            require(sha(support) == SUPPORT_SHA and '- npu_fusion_attention_v3' in support.read_text(), 'Unreviewed public API statistics support')
            write(args.output / 'collector_provenance.json', collector)
            write(args.output / 'config.json', statistics_config(args.output, collector['precision']))
            debugger = PrecisionDebugger(config_path=str(args.output / 'config.json'))
        torch_npu.npu.synchronize()
        save()
        call = {'case_id': case['case_id'], 'api': 'torch_npu.npu_fusion_attention_v3', 'pid': os.getpid(),
                'thread_id': threading.get_native_id(), 'phase': 'operator_replay', 'step': 0,
                'batch_size': case['batch_size'], 'profile_intent': case['profile']}
        scope_manager = HostScope(case, args.output) if case['operator_variant'] == 'private_vendor' else nullcontext(None)
        with torch.no_grad(), scope_manager as scope:
            call['scope_token'] = scope['FAULT28_CAUSAL_SCOPE_TOKEN'] if scope else None
            if debugger is not None:
                debugger.start(model=capture_root, rank_id=0)
                active = True
            call.update(enter_epoch_ns=time.time_ns(), enter_monotonic_ns=time.monotonic_ns())
            print(json.dumps({'event': 'native_api_enter', **call}), flush=True)
            try:
                result['original_api_calls'] += 1
                native = module(inputs['query'], inputs['key'], inputs['value'])
                torch_npu.npu.synchronize()
                call.update(exit_epoch_ns=time.time_ns(), exit_monotonic_ns=time.monotonic_ns(), status='returned')
            finally:
                print(json.dumps({'event': 'native_api_exit', **call}), flush=True)
                if active:
                    debugger.stop()
                    active = False
                    debugger.step()
        require(all(os.environ.get(k) is None for k in SCOPE_KEYS), 'Host scope did not restore')
        result['native_call'] = call
        require(isinstance(native, (tuple, list)) and len(native) == 6, 'Unexpected original native API return contract')
        output = native[0]
        require(list(output.shape) == [case['batch_size'], 16, 2048, 128] and output.dtype == torch.float16,
                'Unexpected native output')
        result['inputs_after'] = {n: snapshot(torch, torch_npu, x) for n, x in inputs.items()}
        result['mask_after'] = snapshot(torch, torch_npu, mask, batch_sliced=False)
        result['input_format_normalizations'] = input_integrity(result['inputs_before'], result['inputs_after'])
        input_integrity({'mask': result['mask_before']}, {'mask': result['mask_after']})
        result['output'] = snapshot(torch, torch_npu, output)
        result['auxiliary_outputs'] = []
        for name, tensor in zip(('softmax_max', 'softmax_sum', 'softmax_out', 'seed', 'offset'), native[1:]):
            require(isinstance(tensor, torch.Tensor), 'Unexpected auxiliary output type')
            batch_mapping = name in ('softmax_max', 'softmax_sum') and list(tensor.shape) == [case['batch_size'], 16, 2048, 8]
            entry = {'name': name, 'batch_mapping_certified': batch_mapping,
                     **snapshot(torch, torch_npu, tensor, batch_sliced=batch_mapping)}
            values = tensor.detach().cpu().double().reshape(-1)
            if values.numel():
                finite = torch.isfinite(values)
                entry['statistics'] = {'elements': values.numel(), 'finite_count': int(finite.sum()),
                    'nan_count': int(torch.isnan(values).sum()), 'inf_count': int(torch.isinf(values).sum())}
                if bool(finite.all()):
                    entry['statistics'].update(max=values.max().item(), min=values.min().item(),
                                               mean=values.mean().item(), norm=values.norm().item())
            result['auxiliary_outputs'].append(entry)
        target = raw_bytes(torch, output[0])
        target_path = args.output / 'private_output/target_fp16.bin'
        target_path.parent.mkdir()
        target_path.write_bytes(target)
        require(digest(target) == result['output']['target_sha256'], 'Private complete target differs from output fingerprint')
        result['output']['target_private_file'] = target_path.relative_to(args.output).as_posix()
        result['private_target_output'] = {'file': target_path.relative_to(args.output).as_posix(),
            'shape': [16, 2048, 128], 'dtype': 'torch.float16', 'bytes': len(target), 'sha256': digest(target),
            'batch_slot': 0, 'byteorder': 'little', 'raw_order': 'C-contiguous logical elements',
            'publication': 'PRIVATE_CAUSAL_COMPARISON_EXCLUDE_FROM_STATISTICS_AND_GITHUB'}
        if case['operator_variant'] == 'stock':
            source_call = json.loads((Path(case['input_manifest_file']).parent / manifest['source_call_file']).read_bytes())
            original_output = source_call['target_output']['logical_sha256']
            result['stock_bridge'] = {'expected_original_target_output_sha256': original_output,
                'actual_target_output_sha256': digest(target), 'same_target_output_bytes': original_output == digest(target),
                'required_match_to_old_B8_capture': case['batch_size'] == 8,
                'scope': 'Complete batch0 output compared with old B8 model capture. Only B8 requires exact equality; B1 batch effect is observed without an equality assumption.'}
            require(case['batch_size'] != 8 or result['stock_bridge']['same_target_output_bytes'],
                    'Stock B8 original-API replay bridge differs from captured model output')
        last = output[0, :, 2047, :].detach().cpu()
        raw_last = raw_bytes(torch, last)
        last_path = args.output / 'comparison/target_last_query_fp16.bin'
        last_path.parent.mkdir()
        last_path.write_bytes(raw_last)
        result['target_last_query'] = {'file': last_path.relative_to(args.output).as_posix(), 'shape': [16, 128],
            'dtype': 'torch.float16', 'bytes': len(raw_last), 'sha256': digest(raw_last), 'batch_slot': 0,
            'query_index': 2047, 'byteorder': 'little', 'raw_order': 'C-contiguous logical elements'}
        boundary_outputs = torch.stack([output[0, :, index, :].detach().cpu() for index in REFERENCE_QUERIES])
        delta = boundary_outputs.double() - reference
        reference_path = args.output / 'comparison/reference_fp64.json'
        write(reference_path, {'schema': 'fault28.causal.boundary_query_reference.v1', 'shape': [len(REFERENCE_QUERIES), 16, 128],
                              'dtype': 'float64', 'values': reference.tolist(), 'contract': result['fp64_reference_contract']})
        result['reference_fp64'] = {'file': reference_path.relative_to(args.output).as_posix(), 'sha256': sha(reference_path),
            'max_abs_error': delta.abs().max().item(), 'l2_error': delta.norm().item(),
            'rms_error': delta.square().mean().sqrt().item(), 'reference_l2': reference.norm().item(),
            'per_query': [{'query_index': index, 'max_abs_error': delta[i].abs().max().item(),
                          'l2_error': delta[i].norm().item(), 'rms_error': delta[i].square().mean().sqrt().item()}
                          for i, index in enumerate(REFERENCE_QUERIES)]}
        if debugger is not None:
            receipt = statistics_receipt(args.output, collector)
            write(args.output / 'receipt.json', receipt)
            result['receipt_sha256'] = sha(args.output / 'receipt.json')
        result.update(status='CAPTURE_COMPLETED', input_unchanged=True, host_scope_restored=True, end_epoch=time.time())
        save()
        print(json.dumps({'status': result['status'], 'case_id': case['case_id'], 'original_api_calls': 1,
                          'native_profile_verified': False, 'mechanism_accepted': False}), flush=True)
        return 0
    except BaseException:
        result.update(status='CAPTURE_FAILED', traceback=traceback.format_exc(), end_epoch=time.time())
        save()
        print(result['traceback'], file=sys.stderr)
        return 1
    finally:
        if active:
            debugger.stop()
            debugger.step()


if __name__ == '__main__':
    raise SystemExit(main())
