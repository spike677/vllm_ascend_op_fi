"""Model observations preserve original tensors and cover every explicit batch slot."""
import hashlib
import json
from pathlib import Path
import re


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def token_sha(values):
    return digest(json.dumps(list(values), separators=(',', ':')).encode())


def physical_device_binding(torch_npu, evidence_root):
    preflight_path = evidence_root / 'before.json'
    container_path = evidence_root / 'container_identity.json'
    preflight = json.loads(preflight_path.read_bytes())
    container = json.loads(container_path.read_bytes())
    require(preflight['selected_card'] == 5 and preflight['eligible'] is True, 'Physical card5 preflight missing')
    found = re.search(r'^\|\s*5\s+(\S+)\s*\|', preflight['npu_smi'], re.M)
    require(found is not None and found[1] == '910B4', 'Actual physical model differs from certified910B4')
    devices = [row['PathOnHost'] for row in container['devices'] if re.fullmatch(r'/dev/davinci\d+', row['PathOnHost'])]
    require(devices == ['/dev/davinci5'], 'Logical NPU does not map only to card5')
    name = str(torch_npu.npu.get_device_name(0))
    require(name, 'Observed device name unavailable')
    return {'device': 'Ascend' + found[1], 'public_api_device_name': name, 'logical_device': 0,
            'physical_card': 5, 'preflight_sha256': sha(preflight_path), 'container_identity_sha256': sha(container_path)}


def verify_fingerprint(path, schema):
    value = json.loads(Path(path).read_bytes())
    require(value['schema'] == schema, 'Wrong fingerprint schema')
    root = Path(value['container_root'])
    for relative, item in value['files'].items():
        source = root / relative
        require(source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root.resolve())
                and source.stat().st_size == item['bytes'] and sha(source) == item['sha256'], 'Model/tokenizer bytes changed: ' + relative)
    return value


def tensor_snapshot(torch, torch_npu, tensor, *, private_file=None, numeric_context=None):
    """Diagnostics remain in raw msprobe; stack labels distinguish these reads."""
    require(tensor.ndim >= 1 and tensor.numel() * tensor.element_size() <= 192 * 1024 * 1024, 'Unbounded diagnostic read')
    before = {'shape': list(tensor.shape), 'stride': list(tensor.stride()), 'dtype': str(tensor.dtype),
              'device': str(tensor.device), 'object_id': id(tensor), 'data_ptr': tensor.data_ptr(),
              'storage_offset': tensor.storage_offset(),
              'actual_format': int(torch_npu.get_npu_format(tensor)) if tensor.device.type == 'npu' else None}
    payload = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    size = len(payload) // tensor.shape[0]
    require(size * tensor.shape[0] == len(payload), 'Nonintegral batch slices')
    result = {**before, 'full_sha256': digest(payload), 'bytes': len(payload),
              'per_batch_sha256': [digest(payload[i * size:(i + 1) * size]) for i in range(tensor.shape[0])],
              'raw_order': 'C-contiguous logical elements', 'byteorder': 'little'}
    require(before['data_ptr'] == tensor.data_ptr() and before['object_id'] == id(tensor)
            and before['shape'] == list(tensor.shape) and before['stride'] == list(tensor.stride()), 'Diagnostic changed original tensor')
    if numeric_context is not None:
        from numeric_observer import observe
        result['exact_numeric_observation']=observe(payload,list(tensor.shape),numeric_context)
    if private_file is not None:
        path = Path(private_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
    return result


def inputs_unchanged(before, after):
    allowed = []
    for name, first in before.items():
        a, b = dict(first), dict(after[name])
        old, new = a.pop('actual_format'), b.pop('actual_format')
        require(a == b, 'Original SDPA changed its input bytes/identity/geometry: ' + name)
        require(old == new or old == 0 and new == 2, 'Unexpected native input descriptor transition')
        if old != new:
            allowed.append({'input': name, 'from': old, 'to': new, 'logical_bytes_unchanged': True})
    return allowed


def statistics_config(output, count):
    return {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': list(range(count)),
            'dump_path': str(output / 'observation'), 'async_dump': False, 'precision': 'high',
            'statistics': {'scope': [], 'list': [], 'data_mode': ['all'], 'summary_mode': 'statistics', 'tensor_list': []}}


def statistics_receipt(output, collector, count):
    files, rows = [], []
    directory = output / 'observation'
    forbidden = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
    require(not any(p.is_symlink() or p.is_file() and p.suffix in forbidden for p in directory.rglob('*')), 'Private tensor payload in statistics')
    for step in range(count):
        objects = {}
        for kind in ('dump', 'stack', 'construct'):
            path = directory / f'step{step}/rank0/{kind}.json'
            objects[kind] = json.loads(path.read_bytes())
            files.append({'path': path.relative_to(output).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
        dump, stack, construct = (objects[k] for k in ('dump', 'stack', 'construct'))
        require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('dump_data_dir') is None, 'Wrong statistics mode')
        apis = [name for name in dump['data'] if re.fullmatch(r'Functional\.scaled_dot_product_attention\.\d+\.forward', name)]
        require(len(apis) == 28, 'Full model must expose28 SDPA records per phase')
        parents = [construct.get(name) for name in apis]
        require(len(set(parents)) == 28 and all(isinstance(name, str) and name in dump['data'] and '.self_attn.' in name for name in parents), 'Incomplete real attention module parents')
        stack_names = {name for item in stack.values() for name in item[0]}
        require(all(name in stack_names for name in apis + parents), 'Missing original API/module stack')
        rows.append({'step': step, 'functional_sdpa_nodes': apis, 'parents': parents, 'dump_nodes': len(dump['data'])})
    return {'schema': 'nes.model_coverage.statistics_receipt.v1', 'steps': rows, 'native_files': files,
            'config_sha256': sha(output / 'config.json'), 'statistics_collector': collector,
            'collector_provenance_sha256': sha(output / 'collector_provenance.json'),
            'diagnostics_present': True,
            'diagnostic_functions': ['tensor_snapshot'], 'diagnostic_policy': 'Raw dump/stack/construct unchanged; identify diagnostic reads by exact model_helpers.py tensor_snapshot stack. Do not remove real SDPA records.',
            'native_coverage_accepted': False, 'model_steps_are_not_tokens': True}
