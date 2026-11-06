"""Fixed-geometry paged-IFA host-policy replay; real kernel, statistics only.

The small projection is a disclosed synthetic consumer, not a language model.
Full native outputs are private comparison material outside the msprobe dump.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from stock_probe import require, write, record, payload, make_logical, physical_cache, compare

SUPPORT_SHA = '5fefa502efd21c9793a286bc1d06f2fdf474876da3c3d96dd474570f8c6609e5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loaded_native_libraries():
    """Actual mapped library files, read after execution and outside capture."""
    names = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6 or not parts[5].startswith('/'):
            continue
        path = parts[5]
        if ('/audit/vendor_stage_a4/' in path or '/Ascend/' in path) and any(
                token in path.lower() for token in ('opapi', 'op_api', 'optiling', 'op_tiling', 'ophost', 'cust_op')):
            names.add(path)
    rows = []
    for name in sorted(names):
        path = Path(name)
        require(path.is_file(), 'Mapped native library disappeared: ' + name)
        rows.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    require(rows, 'No mapped native execution libraries recorded')
    return rows


@contextmanager
def host_scope(case, output, call_id):
    values = {
        'FAULT33_PA_MODE': case['mode'], 'FAULT33_PA_SCOPE_TOKEN': uuid.uuid4().hex,
        'FAULT33_PA_SCOPE_PID': str(os.getpid()), 'FAULT33_PA_SCOPE_TID': str(threading.get_native_id()),
        'FAULT33_PA_SCOPE_PHASE': 'decode', 'FAULT33_PA_CALLER_DETERMINISTIC': '1',
        'FAULT33_PA_BATCH': str(case['batch']), 'FAULT33_PA_LENGTH': str(case['kv_length']),
        'FAULT33_PA_BLOCK_SIZE': str(case['block_size']), 'FAULT33_PA_CALL_ID': call_id,
        'FAULT33_PA_LOG_PATH': str(output / 'host_events.jsonl')}
    require(not any(k.startswith('FAULT33_PA_') for k in os.environ), 'Inherited host scope forbidden')
    os.environ.update(values)
    try:
        yield values
    finally:
        for key in values:
            os.environ.pop(key, None)


def statistics_receipt(output, collector, repeats):
    root = output / 'observation'
    forbidden = {'.pt', '.pth', '.bin', '.npy', '.npz', '.safetensors'}
    require(not any(p.is_file() and p.suffix.lower() in forbidden for p in root.rglob('*')),
            'Tensor payload in statistics directory')
    files, apis = [], []
    for step in range(repeats):
        objects = {}
        for kind in ('dump', 'stack', 'construct'):
            path = root / ('step' + str(step)) / 'rank0' / (kind + '.json')
            objects[kind] = json.loads(path.read_bytes())
            files.append({'path': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size,
                          'sha256': sha(path)})
        dump = objects['dump']
        require(dump['task'] == 'statistics' and dump['level'] == 'mix' and dump.get('dump_data_dir') is None,
                'Wrong native statistics mode')
        nodes = [name for name in dump['data'] if re.fullmatch(r'NPU\.npu_incre_flash_attention\.\d+\.forward', name)]
        require(len(nodes) == 1, 'Each captured step requires one real paged IFA API')
        parent = objects['construct'].get(nodes[0])
        require(isinstance(parent, str) and parent.startswith('Module.') and parent in dump['data']
                and '.ReplayModule.forward.' in parent, 'Native IFA must belong to the captured real module')
        apis.append({'step': step, 'api': nodes[0], 'parent': parent})
    return {'schema': 'fault33.pa.statistics_receipt.v1', 'native_files': files,
            'native_api_nodes': apis, 'rank': 0, 'steps': list(range(repeats)),
            'config_sha256': sha(output / 'config.json'), 'statistics_collector': collector,
            'collector_provenance_sha256': sha(output / 'collector_provenance.json'),
            'scope': 'Isolated native operator plus synthetic projection; not model token steps'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case_file.read_bytes())
    require(case['schema'] == 'fault33.pa.replay.case.v1', 'Wrong case schema')
    require(case['mode'] in ('off', 'inner1024', 'inner512'), 'Unreviewed policy')
    require(case['batch'] == 8 and case['kv_length'] == 16384 and case['block_size'] == 128
            and case['q_heads'] == 16 and case['kv_heads'] == 8 and case['head_dim'] == 128,
            'Pinned mechanism geometry required')
    modes = case.get('mode_sequence', [case['mode']] * case['repeats'])
    require(modes == [case['mode']] * 2 or modes == ['off', 'inner1024', 'inner1024', 'off'],
            'Unregistered finite mode sequence')
    require(case['dtype'] == 'float16' and case['inner_precise'] == 0 and case['repeats'] == len(modes)
            and case['deterministic_algorithms'] is True, 'Pinned deterministic high-precision repeat plan required')
    require(len(modes) == 2 or case['capture_msprobe'] is False, 'Restoration is a diagnostic without extra statistics groups')
    out = args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    (out / 'private_output').mkdir()
    write(out / 'case.json', case)
    vendor = None
    if case['operator_variant'] == 'private_vendor':
        from vendor_exec import validate_vendor
        vendor = validate_vendor(case)
        receipt_path = out.with_name(out.name + '.vendor_launch.json')
        require(os.environ.get('FAULT33_VENDOR_LAUNCH_RECEIPT') == str(receipt_path)
                and sha(receipt_path) == os.environ.get('FAULT33_VENDOR_LAUNCH_RECEIPT_SHA256'), 'Missing launcher receipt')
        receipt = json.loads(receipt_path.read_bytes())
        require(receipt['case_sha256'] == sha(args.case_file) and receipt['probe_sha256'] == sha(__file__)
                and receipt['verified_vendor']['manifest_sha256'] == vendor['manifest_sha256'], 'Launcher fingerprint mismatch')
        require(all(receipt['child_environment'].get(k) == os.environ.get(k) for k in
                    ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')),
                'Loader environment differs from verified launch')
    else:
        require(case['operator_variant'] == 'stock' and case['mode'] == 'off'
                and not os.environ.get('ASCEND_CUSTOM_OPP_PATH'), 'Stock bridge must use the original library')
    from collector_stage import select_collector, verify_loaded
    collector = select_collector(case)
    import torch
    import torch_npu
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_device(0)
    q_cpu, k_cpu, v_cpu = make_logical(case)
    kc, table = physical_cache(k_cpu, case['block_size'])
    vc, table_v = physical_cache(v_cpu, case['block_size'])
    require(torch.equal(table, table_v), 'K/V mapping differs')
    q, k, v, blocks = [t.to('npu') for t in (q_cpu, kc, vc, table)]
    logical = {n: record(t) for n, t in [('query', q_cpu), ('key', k_cpu), ('value', v_cpu)]}
    write(out / 'logical_inputs.json', logical)
    del kc, vc
    generator = torch.Generator(device='cpu').manual_seed(case['projection_seed'])
    weight_cpu = torch.randn((2048, 256), generator=generator, dtype=torch.float32).div(16).half()
    weight = weight_cpu.to('npu')
    write(out / 'projection.json', {'kind': 'synthetic_fixed_linear_consumer', 'weight': record(weight_cpu)})
    reference = ((q_cpu.double() @ k_cpu.repeat_interleave(2, dim=1).double().transpose(-1, -2))
                 / math.sqrt(128)).softmax(-1) @ v_cpu.repeat_interleave(2, dim=1).double()
    before = {name: record(t) for name, t in [('query', q), ('key', k), ('value', v), ('block_table', blocks)]}
    kwargs = dict(block_table=blocks, actual_seq_lengths=[case['kv_length']] * case['batch'],
                  num_heads=16, num_key_value_heads=8, input_layout='BNSD',
                  scale_value=1.0 / math.sqrt(128), block_size=128, inner_precise=0)

    class ReplayModule(torch.nn.Module):
        def forward(self, query, key, value):
            attended = torch_npu.npu_incre_flash_attention(query, key, value, **kwargs)
            projected = attended.flatten(1) @ weight
            return attended, projected

    module = ReplayModule().eval()
    holder = torch.nn.Module()
    holder.add_module('replay', module)
    debugger = None
    if case['capture_msprobe']:
        from msprobe.pytorch import PrecisionDebugger
        processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
        collector['actual_loaded'] = verify_loaded(collector, processor)
        support = Path(inspect.getsourcefile(PrecisionDebugger)).parents[1] / 'api_dump/support_wrap_ops.yaml'
        require(sha(support) == SUPPORT_SHA and '- npu_incre_flash_attention' in support.read_text(), 'Unreviewed API support')
        write(out / 'collector_provenance.json', collector)
        config = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0, 1],
                  'dump_path': str(out / 'observation'), 'async_dump': False,
                  'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                                 'summary_mode': 'statistics', 'tensor_list': []}}
        if collector['precision'] == 'high':
            config['precision'] = 'high'
        write(out / 'config.json', config)
        debugger = PrecisionDebugger(config_path=str(out / 'config.json'))
    calls, outputs, projected_outputs = [], [], []
    for repeat, actual_mode in enumerate(modes):
        call_id = 'repeat' + str(repeat)
        marker = {'fault33': 'enter', 'call_id': call_id, 'pid': os.getpid(),
                  'tid': threading.get_native_id(), 'block_size': 128,
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        scope = host_scope({**case, 'mode': actual_mode}, out, call_id)
        with torch.no_grad(), scope as actual_scope:
            if debugger:
                debugger.start(model=holder, rank_id=0)
            torch.npu.synchronize()
            print(json.dumps({**marker, 'scope_token': actual_scope['FAULT33_PA_SCOPE_TOKEN']}), flush=True)
            started = time.monotonic()
            try:
                attended, projected = module(q, k, v)
                torch.npu.synchronize()
                print(json.dumps({**marker, 'fault33': 'exit', 'elapsed_seconds': time.monotonic() - started}), flush=True)
            finally:
                if debugger:
                    debugger.stop()
                    debugger.step()
        after = {name: record(t) for name, t in [('query', q), ('key', k), ('value', v), ('block_table', blocks)]}
        require(before == after, 'Actual input data or geometry changed')
        ac, pc = attended.detach().cpu().contiguous(), projected.detach().cpu().contiguous()
        require(bool(torch.isfinite(ac).all() and torch.isfinite(pc).all()), 'Nonfinite output')
        for label, tensor in [('attention', ac), ('projection', pc)]:
            (out / 'private_output' / (call_id + '_' + label + '.bin')).write_bytes(payload(tensor))
        calls.append({'call_id': call_id, 'mode': actual_mode, 'scope_token': actual_scope['FAULT33_PA_SCOPE_TOKEN'],
                      'pid': marker['pid'], 'tid': marker['tid'], 'original_api_calls': 1,
                      'inputs_before': before, 'inputs_after': after,
                      'attention': record(ac), 'synthetic_projection': record(pc),
                      'reference_comparison': compare(reference, ac)})
        outputs.append(ac)
        projected_outputs.append(pc)
    write(out / 'runtime_native_maps.json', {'pid': os.getpid(), 'libraries': loaded_native_libraries(),
          'scope': 'Actual mapped files; mapping alone does not prove which symbol executed.'})
    stats = statistics_receipt(out, collector, case['repeats']) if debugger else None
    if stats:
        write(out / 'statistics_receipt.json', stats)
    result = {'schema': 'fault33.pa.replay.result.v1', 'status': 'CAPTURE_COMPLETED',
              'case_id': case['case_id'], 'mode': case['mode'], 'original_api_calls': len(calls),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'calls': calls, 'logical_inputs': logical, 'vendor': vendor,
              'attention_repeat': compare(outputs[0], outputs[1]),
              'projection_repeat': compare(projected_outputs[0], projected_outputs[1]),
              'statistics_receipt': stats, 'software': {'torch': torch.__version__, 'torch_npu': torch_npu.__version__},
              'runtime_native_maps_sha256': sha(out / 'runtime_native_maps.json'),
              'same_process_off_restoration': compare(outputs[0], outputs[3]) if len(modes) == 4 else None,
              'same_process_on_repeat': compare(outputs[1], outputs[2]) if len(modes) == 4 else None,
              'mechanism_accepted': False, 'model_forward': False,
              'scope': 'Fixed native paging geometry; synthetic projection is a consumer, not model logits.'}
    write(out / 'result.json', result)
    print(json.dumps({'case_id': case['case_id'], 'status': result['status'],
                      'original_api_calls': len(calls), 'repeat_bitwise_equal': result['attention_repeat']['bitwise_equal']}), flush=True)


if __name__ == '__main__':
    main()
