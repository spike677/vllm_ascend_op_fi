"""Shared capture helpers for genuine native FA replay, with no output substitution.

Private binary fingerprints are computed outside msprobe capture. The only
allowed native input descriptor normalization is unchanged values with 0 -> 2.
"""
import hashlib
import json
from pathlib import Path
import re
import struct

REFERENCE_QUERIES = [0, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047]
ORIGINAL_CALL_SHA = '458a683a616da397bd1eeba47b7348d0e2bf78c253ea43c158f7f393bb67c0c8'
PARENT_INPUT_MANIFEST_SHA = '509123dc725e351d6b10840919c2640b327a13eecceba2ca4a888c4a094f8e62'
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


def load_frozen_inputs(torch, case):
    path = Path(case['input_manifest_file'])
    require(path.is_file() and sha(path) == case['input_manifest_sha256'], 'Input manifest differs')
    manifest = json.loads(path.read_bytes())
    require(manifest['schema'] == 'nes.captured_inputs.v1' and manifest['source_call_sha256'] == ORIGINAL_CALL_SHA,
            'Wrong original model capture provenance')
    require(sha(path.parent / manifest['source_call_file']) == ORIGINAL_CALL_SHA, 'Copied original call metadata differs')
    require(manifest['parent_manifest_sha256'] == PARENT_INPUT_MANIFEST_SHA
            and sha(path.parent / manifest['parent_manifest_file']) == PARENT_INPUT_MANIFEST_SHA,
            'Parent input snapshot differs')
    require(case['batch_size'] in (1, 2, 4, 8), 'Unsupported observation batch')
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


def statistics_config(output, steps):
    require(type(steps) is int and steps in (1, 4), 'One discovery step or four controlled steps required')
    return {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': list(range(steps)),
            'dump_path': str(output / 'observation'), 'async_dump': False, 'precision': 'high',
            'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                           'summary_mode': 'statistics', 'tensor_list': []}}


def statistics_receipt(output, collector, step_count):
    root = output / 'observation'
    forbidden = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
    require(not any(p.is_symlink() or p.is_file() and p.suffix.lower() in forbidden for p in root.rglob('*')),
            'Tensor payload or symlink in statistics')
    rows, nodes = [], []
    for step in range(step_count):
        objects = {}
        for kind in ('dump', 'stack', 'construct'):
            path = root / f'step{step}/rank0' / (kind + '.json')
            objects[kind] = json.loads(path.read_bytes())
            rows.append({'path': path.relative_to(output).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
        dump = objects['dump']
        require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('dump_data_dir') is None,
                'Unexpected native statistics mode')
        apis = [name for name in dump['data'] if re.fullmatch(r'NPU\.npu_fusion_attention_v3\.\d+\.forward', name)]
        require(len(apis) == 1, 'Each step must record exactly one original native FA API call')
        stacked_names = set()
        for stack_row in objects['stack'].values():
            require(isinstance(stack_row, list) and len(stack_row) == 2
                    and isinstance(stack_row[0], list) and isinstance(stack_row[1], list),
                    'Unexpected installed msprobe stack schema')
            stacked_names.update(stack_row[0])
        parent = objects['construct'].get(apis[0])
        require(isinstance(parent, str) and parent.startswith('Module.') and '.ReplayModule.forward.' in parent
                and parent in dump['data'] and apis[0] in stacked_names,
                'Native API must belong to the captured real ReplayModule child')
        nodes.append({'step': step, 'api': apis[0], 'parent_module': parent})
    return {'schema': 'nes.statistics_receipt.v1', 'scope': 'Real operator calls; steps are not model tokens/forwards',
            'rank': 0, 'steps': list(range(step_count)), 'native_files': rows, 'native_api_nodes': nodes,
            'config_sha256': sha(output / 'config.json'), 'statistics_collector': collector,
            'collector_provenance_sha256': sha(output / 'collector_provenance.json'),
            'native_cann_coverage_verified': False, 'mechanism_accepted': False}


def capture_outputs(torch, torch_npu, native, batch_size, step_output, reference, reference_contract):
    """Read original returns after capture stops; full tensor bytes remain private."""
    require(isinstance(native, (tuple, list)) and len(native) == 6, 'Unexpected original native API return contract')
    output = native[0]
    require(list(output.shape) == [batch_size, 16, 2048, 128] and output.dtype == torch.float16,
            'Unexpected native output')
    record = {'output': snapshot(torch, torch_npu, output), 'auxiliary_outputs': []}
    for name, tensor in zip(('softmax_max', 'softmax_sum', 'softmax_out', 'seed', 'offset'), native[1:]):
        require(isinstance(tensor, torch.Tensor), 'Unexpected auxiliary return type')
        mapped = name in ('softmax_max', 'softmax_sum') and list(tensor.shape) == [batch_size, 16, 2048, 8]
        entry = {'name': name, 'batch_mapping_certified': mapped,
                 **snapshot(torch, torch_npu, tensor, batch_sliced=mapped)}
        values = tensor.detach().cpu().double().reshape(-1)
        if values.numel():
            finite = torch.isfinite(values)
            entry['statistics'] = {'elements': values.numel(), 'finite_count': int(finite.sum()),
                                   'nan_count': int(torch.isnan(values).sum()), 'inf_count': int(torch.isinf(values).sum())}
            if bool(finite.all()):
                entry['statistics'].update(max=values.max().item(), min=values.min().item(),
                                           mean=values.mean().item(), norm=values.norm().item())
        record['auxiliary_outputs'].append(entry)
    raw = raw_bytes(torch, output[0])
    private = step_output / 'private_output/target_fp16.bin'
    private.parent.mkdir(parents=True, exist_ok=False)
    with private.open('xb') as stream:
        stream.write(raw)
    require(digest(raw) == record['output']['target_sha256'], 'Private target differs from original output fingerprint')
    record['private_target_output'] = {'file': private.relative_to(step_output).as_posix(),
        'shape': [16, 2048, 128], 'dtype': 'torch.float16', 'bytes': len(raw), 'sha256': digest(raw),
        'batch_slot': 0, 'byteorder': 'little', 'raw_order': 'C-contiguous logical elements',
        'publication': 'PRIVATE_OUTPUT_BINARY_EXCLUDE_FROM_STATISTICS_AND_GITHUB'}
    observed = torch.stack([output[0, :, index, :].detach().cpu() for index in REFERENCE_QUERIES])
    delta = observed.double() - reference
    reference_path = step_output / 'private_output/reference_fp64.json'
    write(reference_path, {'schema': 'nes.boundary_query_reference.v1',
                          'shape': [len(REFERENCE_QUERIES), 16, 128], 'dtype': 'float64',
                          'values': reference.tolist(), 'contract': reference_contract,
                          'publication': 'PRIVATE_REFERENCE_VALUES_EXCLUDE_FROM_STATISTICS_AND_GITHUB'})
    record['reference_fp64'] = {'file': reference_path.relative_to(step_output).as_posix(), 'sha256': sha(reference_path),
        'max_abs_error': delta.abs().max().item(), 'l2_error': delta.norm().item(),
        'rms_error': delta.square().mean().sqrt().item(), 'reference_l2': reference.norm().item(),
        'per_query': [{'query_index': index, 'max_abs_error': delta[i].abs().max().item(),
                      'l2_error': delta[i].norm().item(), 'rms_error': delta[i].square().mean().sqrt().item()}
                     for i, index in enumerate(REFERENCE_QUERIES)]}
    return record
