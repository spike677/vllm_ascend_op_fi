"""Read preserved source48 stock/B evidence; emit summaries, never tensor values.

This diagnostic cannot satisfy the bridge prerequisite or authorize a fault.
It calls the frozen bridge's existing CPU validators without changing them.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


SOURCE_BASE = '78f40eab2f91bd8882b2f77ee69dd6f21b7cac09'
CHUNK = 4 * 1024**2
FORMATS = {'torch.float16': '<f2', 'torch.float32': '<f4', 'torch.bfloat16': '<u2'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            value.update(block)
    return value.hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def attempt(function):
    try:
        return dict(status='PASS', result=function())
    except Exception as error:
        return dict(status='FAILED', error=type(error).__name__ + ': ' + str(error)[:320])


def load_frozen(source):
    source = Path(source).resolve(strict=True)
    # Load only the explicitly supplied frozen source, never a deployment or
    # ambient installed package. Its original bridge is not edited or patched.
    sys.path[:0] = [str(source/'experiments/c2c3_closure_20261117'), str(source/'src'), str(source)]
    contract = importlib.import_module('attention33_contract')
    manifest = contract.read_json(source/'source_manifest.json')
    need(manifest.get('schema') == 'demoncat.c2c3.source.v1'
         and manifest.get('base_commit') == SOURCE_BASE, 'EXACT_SOURCE48_MANIFEST_REQUIRED')
    recorded = {row['path']: row['sha256'] for row in manifest['files']}
    required = ('experiments/c2c3_closure_20261117/attention33_contract.py',
        'experiments/c2c3_closure_20261117/attention33_bridge.py',
        'experiments/c2c3_closure_20261117/attention33_worker_probe.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_tnd.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_loading.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/paged33_audit.py',
        'src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py',
        'fault_injection/native_adapters/attention28/launch.py',
        'fault_injection/native_adapters/attention28/fia/fai_fd_geometry.py')
    pins = {name: file_sha(source/name) for name in required}
    need(all(recorded.get(name) == value for name, value in pins.items()), 'SOURCE48_VALIDATOR_BYTES_DIFFER')
    bridge = importlib.import_module('attention33_bridge')
    need(Path(bridge.__file__).resolve() == source/required[1]
         and Path(contract.__file__).resolve() == source/required[0], 'AMBIENT_REVIEW_MODULE_REFUSED')
    return bridge, contract, dict(base_commit=manifest['base_commit'],
        manifest_sha256=file_sha(source/'source_manifest.json'), helper_sha256=pins)


def prohibit_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'socket.connect'):
        raise RuntimeError('READ_ONLY_CPU_DIAGNOSTIC_RUNTIME_ACTION_REFUSED:' + event)
    if event == 'import' and args[0].split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        raise RuntimeError('DEVICE_FRAMEWORK_IMPORT_REFUSED:' + args[0])


def raw_difference(left, right, *, split_cache=False):
    """Read every byte again; signed zero remains a bridge difference."""
    import numpy as np
    fields = ('shape', 'dtype', 'bytes')
    need(all(left[key] == right[key] for key in fields), 'PAIR_PAYLOAD_GEOMETRY_DIFFERS')
    dtype = left['dtype']; fmt = np.dtype(FORMATS[dtype]); size = left['bytes']
    need(0 < size <= 128*1024**2 and size == math.prod(left['shape'])*fmt.itemsize,
         'PAIR_PAYLOAD_BOUND_OR_SIZE_DIFFERS')
    result = dict(shape=left['shape'], dtype=dtype, bytes=size,
        left_path=left['path'], right_path=right['path'],
        left_sha256=left['sha256'], right_sha256=right['sha256'],
        changed_bytes=0, changed_elements=0, numeric_changed_elements=0, max_abs_difference=0.0)
    hashes = [hashlib.sha256(), hashlib.sha256()]
    split = (left['shape'][0]-1)*math.prod(left['shape'][1:])*fmt.itemsize if split_cache else None
    segments = {name: dict(bytes=0, changed_elements=0, max_abs_difference=0.0,
                          _left=hashlib.sha256(), _right=hashlib.sha256())
                for name in ('prefill_rows', 'last_decode_row')} if split_cache else {}

    def values(raw):
        array = np.frombuffer(raw, dtype=fmt)
        if dtype == 'torch.bfloat16':
            array = (array.astype('<u4') << 16).view('<f4')
        need(bool(np.isfinite(array).all()), 'PAIR_NONFINITE_RAW_VALUE')
        return array.astype(np.float64)

    def stats(a, b):
        av, bv = values(a), values(b)
        bits = np.any(np.frombuffer(a, dtype='u1').reshape(-1, fmt.itemsize)
                      != np.frombuffer(b, dtype='u1').reshape(-1, fmt.itemsize), axis=1)
        return int(np.count_nonzero(bits)), int(np.count_nonzero(av != bv)), float(np.abs(av-bv).max(initial=0))

    offset = 0
    with Path(left['path']).open('rb') as aa, Path(right['path']).open('rb') as bb:
        while offset < size:
            a, b = aa.read(min(CHUNK, size-offset)), bb.read(min(CHUNK, size-offset))
            need(len(a) == len(b) == min(CHUNK, size-offset), 'PAIR_RAW_LENGTH_CHANGED')
            hashes[0].update(a); hashes[1].update(b)
            changed, numeric, maximum = stats(a, b)
            result['changed_bytes'] += int(np.count_nonzero(np.frombuffer(a, dtype='u1') != np.frombuffer(b, dtype='u1')))
            result['changed_elements'] += changed
            result['numeric_changed_elements'] += numeric
            result['max_abs_difference'] = max(result['max_abs_difference'], maximum)
            if split_cache:
                for name, start, end in (('prefill_rows', 0, split), ('last_decode_row', split, size)):
                    lo, hi = max(start, offset)-offset, min(end, offset+len(a))-offset
                    if hi <= lo:
                        continue
                    x, y = a[lo:hi], b[lo:hi]; item = segments[name]
                    count, _, maximum = stats(x, y)
                    item['bytes'] += len(x); item['changed_elements'] += count
                    item['max_abs_difference'] = max(item['max_abs_difference'], maximum)
                    item['_left'].update(x); item['_right'].update(y)
            offset += len(a)
        need(aa.read(1) == bb.read(1) == b'', 'PAIR_TRAILING_RAW_BYTES')
    need(hashes[0].hexdigest() == left['sha256'] and hashes[1].hexdigest() == right['sha256'],
         'PAIR_RAW_HASH_CHANGED_SINCE_VALIDATION')
    result['exact'] = result['changed_bytes'] == 0
    if split_cache:
        result['cache_segments'] = {name: dict(bytes=item['bytes'], changed_elements=item['changed_elements'],
            max_abs_difference=item['max_abs_difference'], left_sha256=item['_left'].hexdigest(),
            right_sha256=item['_right'].hexdigest()) for name, item in segments.items()}
    return result


def layer_index(name):
    return int(name.split('.')[2])


def compare(left, right, order):
    need(set(left) == set(right) and len(left) == 87, 'COMPLETE_87_PAYLOAD_PAIR_SET_REQUIRED')
    differences = {}
    for name in left:
        value = raw_difference(left[name], right[name], split_cache=name.endswith(('/key', '/value')))
        if not value['exact']:
            differences[name] = value
    changed = {kind: [layer_index(path) for path in order if path+'/'+kind in differences]
               for kind in ('output', 'key', 'value')}
    first = next((path for path in order if path+'/output' in differences), None)
    selected = {name: differences[name] for name in ('hidden', 'logits_hidden', 'logits') if name in differences}
    for kind in ('output', 'key', 'value'):
        path = next((path for path in order if path+'/'+kind in differences), None)
        if path is not None:
            selected[path+'/'+kind] = differences[path+'/'+kind]
    if first is not None:
        for kind in ('key', 'value'):
            name = first+'/'+kind
            # Retain exact input-cache hashes at the first differing output too.
            selected[name] = differences.get(name) or raw_difference(left[name], right[name], split_cache=True)
    before = order[:order.index(first)] if first else order
    return dict(payload_pairs=87, differing_payloads=len(differences), exact_payloads=87-len(differences),
        changed_attention_layers=changed, first_recorded_attention_output_difference=first,
        preceding_attention_outputs_exact=len(before),
        preceding_attention_cache_payloads_exact=all(path+'/'+kind not in differences for path in before for kind in ('key', 'value')),
        selected_payload_summaries=selected)


def load_run(bridge, contract, path, kind):
    path = Path(path).resolve(strict=True); report = contract.read_json(path/'result.json')
    audited = attempt(lambda: bridge.audit_run(path, kind))
    summary = dict(path=str(path), result_sha256=file_sha(path/'result.json'),
        reported_status=report.get('status'), strict_original_run_audit={k: v for k, v in audited.items() if k != 'result'})
    phases = report['phases']
    need([row['phase'] for row in phases] == list(contract.PHASES), 'FIVE_ORDERED_PHASES_REQUIRED')
    payloads, boundaries = {}, {}
    for row in phases:
        name = row['phase']; capture = row['finish']['capture']
        need(contract.read_json(path/name/'runtime_result.json') == capture, 'ORIGINAL_CAPTURE_COPY_CHANGED')
        payloads[name] = bridge.payload_rows(path, capture)
        boundaries[name] = bridge.boundary_rows(capture, row, report['warm_healthy']['prompt_token_ids'])
    return dict(run=path, report=report, result_sha256=summary['result_sha256'], payloads=payloads,
                boundaries=boundaries), summary


def input_summary(left, right, phase, order):
    a = next(row for row in left['report']['phases'] if row['phase'] == phase)
    b = next(row for row in right['report']['phases'] if row['phase'] == phase)
    semantic = ('request_count', 'token_row_count', 'counts', 'pure_decode', 'row_token_positions', 'input_ids', 'prefixes')
    geometry = ('shape', 'stride', 'storage_offset', 'dtype', 'device', 'storage_nbytes')
    scheduler_equal, query_geometry_equal = [], []
    for path in order:
        x, y = left['boundaries'][phase][path], right['boundaries'][phase][path]
        scheduler_equal.append(all(key in x['scheduler'] and key in y['scheduler']
                                   and x['scheduler'][key] == y['scheduler'][key] for key in semantic))
        query_geometry_equal.append(all(key in x['query'] and key in y['query']
                                        and x['query'][key] == y['query'][key] for key in geometry))
    return dict(prompt_token_vectors_equal=a['response']['prompt_token_ids'] == b['response']['prompt_token_ids'],
        setup_source_equal=a['setup']['source'] == b['setup']['source'],
        probe_module_sha256_equal=a['setup']['module_sha256'] == b['setup']['module_sha256'],
        returned_tokens_equal=a['response']['outputs'] == b['response']['outputs'],
        actual_decode_scheduler_equal_layers=sum(scheduler_equal), query_metadata_equal_layers=sum(query_geometry_equal),
        query_value_equality='NOT_OBSERVED_SOURCE48_STORES_METADATA_ONLY',
        intermediate_residual_norm_equality='NOT_CAPTURED', full_operator_input_equality_proven=False)


def diagnose(bridge, contract, stock_path, baseline_path, build_root):
    stock, stock_summary = load_run(bridge, contract, stock_path, 'stock')
    baseline, baseline_summary = load_run(bridge, contract, baseline_path, 'baseline')
    first = contract.PHASES[0]
    order = list(stock['boundaries'][first])
    need(order == sorted(bridge.LAYERS, key=layer_index), 'ACTUAL_LAYER_OCCURRENCE_ORDER_UNEXPECTED')
    need(all(list(run['boundaries'][phase]) == order for run in (stock, baseline) for phase in contract.PHASES),
         'ACTUAL_LAYER_OCCURRENCE_ORDER_CHANGED')
    within = {label: {phase: compare(run['payloads'][first], run['payloads'][phase], order)
                     for phase in contract.PHASES[1:]} for label, run in (('stock', stock), ('private', baseline))}
    cross = {phase: dict(compare(stock['payloads'][phase], baseline['payloads'][phase], order),
                        input_evidence=input_summary(stock, baseline, phase, order)) for phase in contract.PHASES}
    for phase in contract.PHASES[1:]:
        if not within['stock'][phase]['differing_payloads'] and not within['private'][phase]['differing_payloads']:
            cross[phase].pop('selected_payload_summaries')
            cross[phase]['payload_details_identical_to_phase'] = first
    native = attempt(lambda: bridge.audit_native(baseline, build_root))
    if native['status'] == 'PASS':
        native['result'].pop('native_calls')  # No per-launch arrays/arguments leave the machine.
    profiles = Counter()
    for phase in baseline['report']['phases']:
        for call in phase['finish']['capture']['native_calls']:
            launch = call['native']
            profiles[(launch['tiling_key'], launch['block_dim'], launch['schedule_mode'], digest(launch['typed']))] += 1
    library = baseline['report'].get('library_coexistence_after_standard_worker_initialization', {})
    bindings = library.get('bindings', {})
    library_summary = {name: [{key: row[key] for key in ('symbol', 'path', 'sha256')} for row in bindings.get(name, [])]
                       for name in ('private_fia', 'stock_norm')}
    library_summary['distinct_library_handles'] = bindings.get('distinct_library_handles')
    return dict(schema='demoncat.paged33.offline_difference_diagnostic.v1',
        status='DIAGNOSTIC_ONLY_BRIDGE_GATE_UNCHANGED', stock=stock_summary, private=baseline_summary,
        execution_contract_equal=stock['report']['execution_contract'] == baseline['report']['execution_contract'],
        execution_contract_sha256={label: digest(run['report']['execution_contract']) for label, run in (('stock', stock), ('private', baseline))},
        within_run=within, cross_run=cross, private_original_native_audit=native,
        private_recorded_launch_profiles=[dict(tiling_key=k, block_dim=b, schedule_mode=s, typed_sha256=h, count=n)
            for (k, b, s, h), n in sorted(profiles.items())],
        private_recorded_library_bindings=library_summary,
        stock_launch_profile='NOT_RECORDED_BY_SOURCE48_STOCK_DRIVER',
        stock_runtime_library_ownership='NOT_RECORDED_BY_SOURCE48_STOCK_DRIVER',
        limits=['First recorded attention output is not a first operator-level cause.',
                'K/V files are complete logical readbacks after the consumer, not pre-launch snapshots.',
                'Private healthy_after removes scoped observers but retains the private vendor process.',
                'Raw tensor arrays and per-launch arguments remain on the evidence host; only summaries are emitted.',
                'No statistics or query-value files are read; no runtime actions or NPU calls are executed.'],
        bridge_pass=False, fault_authorized=False, c2=False, c3=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'stock-run', 'baseline-run', 'build-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    import numpy  # Initialize only the CPU byte reader before blocking runtime loads.
    sys.addaudithook(prohibit_runtime)
    try:
        bridge, contract, source = load_frozen(args.source_root)
        result = diagnose(bridge, contract, args.stock_run, args.baseline_run, args.build_root)
        result.update(diagnostic_sha256=file_sha(__file__), source=source)
    except Exception as error:
        result = dict(status='DIAGNOSTIC_FAILED', error=type(error).__name__+': '+str(error)[:400],
                      bridge_pass=False, fault_authorized=False, c2=False, c3=False)
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 2 if result['status'] == 'DIAGNOSTIC_FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
