"""Independent CPU review of raw complete Worker payloads and native FD launches.

This reads preserved files only. It never imports torch, starts a container,
changes a result, computes E5 statistics, or grants public Coverage admission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

from attention33_contract import PHASES, read_json, require, sha

LAYERS = {f'model.layers.{i}.self_attn.attn' for i in range(28)}
SUCCESS = 'NATIVE_WORKER_ATTEMPT_AND_OFF_COMPLETE_REQUIRES_EVIDENCE_REVIEW'


def evidence_path(run, recorded):
    value = PurePosixPath(recorded)
    require('..' not in value.parts and value.is_relative_to('/evidence/run'), 'TND33_EVIDENCE_PATH_INVALID')
    target = Path(run)/value.relative_to('/evidence/run')
    require(target.resolve(strict=True).is_relative_to(Path(run).resolve(strict=True)), 'TND33_EVIDENCE_ESCAPES_RUN')
    return target


def raw_tensor(run, row, shape, *, cache=False):
    import numpy as np  # CPU byte interpretation only; no framework import.
    formats = {'torch.float16': '<f2', 'torch.float32': '<f4', 'torch.bfloat16': '<u2'}
    dtype = row.get('dtype')
    require(dtype in formats and row.get('shape') == shape and row.get('finite') is True,
            'TND33_COMPLETE_PAYLOAD_SHAPE_DTYPE_OR_FINITE_DIFFERS')
    size = math.prod(shape)*np.dtype(formats[dtype]).itemsize
    require(0 < size <= 128*1024**2 and row.get('bytes') == size, 'TND33_COMPLETE_PAYLOAD_SIZE_DIFFERS')
    path = evidence_path(run, row['path'])
    require(path.is_file() and path.stat().st_size == size, 'TND33_RAW_PAYLOAD_FILE_SIZE_DIFFERS')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            digest.update(block);values = np.frombuffer(block, dtype=formats[dtype])
            finite = ((values & 0x7f80) != 0x7f80).all() if dtype == 'torch.bfloat16' else np.isfinite(values).all()
            require(bool(finite), 'TND33_RAW_PAYLOAD_NONFINITE')
    require(digest.hexdigest() == row.get('sha256'), 'TND33_RAW_PAYLOAD_HASH_DIFFERS')
    if cache:
        require(row.get('readback') == 'VALID_LOGICAL_PAGE_ORDER_AFTER_CONSUMER'
            and row.get('operator_input_repacked') is False, 'TND33_CACHE_READBACK_CONTRACT_DIFFERS')
    return dict(path=str(path), shape=shape, dtype=dtype, bytes=size, sha256=digest.hexdigest())


def payload_rows(run, capture):
    require(capture.get('schema') == 'demoncat.paged33.natural_probe.v1'
        and capture.get('all_represented_tensors_finite') is True
        and capture.get('complete_tensor_readback_during_statistics') is False
        and set(capture.get('attention', {})) == LAYERS, 'TND33_COMPLETE_CAPTURE_REQUIRED')
    rows = {name: raw_tensor(run, capture[name], shape) for name, shape in
            (('hidden', [1, 1024]), ('logits_hidden', [1, 1024]), ('logits', [1, 151936]))}
    for layer in sorted(LAYERS):
        item = capture['attention'][layer]
        require(set(item) == {'output', 'key', 'value'}, 'TND33_LAYER_PAYLOAD_SET_DIFFERS')
        for kind in ('output', 'key', 'value'):
            rows[layer+'/'+kind] = raw_tensor(run, item[kind], [1, 16, 128] if kind == 'output' else [1025, 8, 128],
                                               cache=kind != 'output')
    return rows


def compare_payload(left, right):
    require(set(left) == set(right) and len(left) == 87, 'TND33_PAYLOAD_CENSUS_DIFFERS')
    compared = 0
    for name in sorted(left):
        a, b = left[name], right[name]
        require(all(a[key] == b[key] for key in ('shape', 'dtype', 'bytes', 'sha256')),
                'TND33_STOCK_PRIVATE_PAYLOAD_DIFFERS:' + name)
        # Direct complete comparison in addition to freshly recomputed hashes.
        with Path(a['path']).open('rb') as aa, Path(b['path']).open('rb') as bb:
            while True:
                x, y = aa.read(4*1024**2), bb.read(4*1024**2)
                require(x == y, 'TND33_STOCK_PRIVATE_RAW_BYTES_DIFFER:' + name)
                if not x:break
                compared += len(x)
    return compared


def boundary_rows(capture, phase, prompt):
    audit = capture['audit'];rows = audit.get('rows', [])
    require(audit.get('candidate_count') == 28 and audit.get('decode_forwards') == 1
        and audit.get('prefill_forwards') == 1 and not audit.get('errors') and len(rows) == 28
        and {r['module_path'] for r in rows} == LAYERS and audit.get('core_methods_replaced') is False,
        'TND33_NATURAL_BOUNDARY_CENSUS_DIFFERS')
    internal = phase['expected']['actual_internal_request_id']
    expected_output = phase['response']['outputs'][0]['tokens'][:1]
    result = {}
    for row in rows:
        scheduler = row['scheduler'];writer = row['writer'];layer = row['module_path']
        require(row.get('actual_operator') == 'torch_npu.npu_fused_infer_attention_score'
            and row.get('consumer_returned') is True and row.get('metadata_binding') == 'VERIFIED'
            and row.get('layout') == 'TND' and row.get('sequence_lengths') == [1025]
            and row.get('virtual_engine') == 0 and row['query']['shape'] == [1, 16, 128]
            and row['page_table']['shape'] == [1, 80]
            and scheduler.get('request_ids') == [internal] and scheduler.get('counts') == [1]
            and scheduler.get('pure_decode') is True and scheduler.get('row_token_positions') == [1024]
            and scheduler.get('input_ids') == expected_output
            and scheduler.get('prefixes') == [dict(prompt=prompt, output=expected_output)],
            'TND33_ACTUAL_REQUEST_OR_SEQUENCE_BINDING_DIFFERS')
        pages = row['used_pages']
        require(len(pages) == 1 and len(pages[0]) == len(set(pages[0])) == 65
            and all(type(page) is int and 0 <= page < 128 for page in pages[0]), 'TND33_ACTUAL_PAGE_SET_DIFFERS')
        slot = pages[0][-1]*16
        require(row.get('slots') == row.get('expected_slots') == writer.get('slots') == [slot]
            and writer.get('returned') is True and writer.get('module_path') == layer,
            'TND33_NATURAL_WRITER_SLOT_BINDING_DIFFERS')
        for kind in ('key', 'value'):
            live = row['live_'+kind+'_cache'];written = writer[kind+'_cache'];actual = row[kind]
            require(live['shape'] == [128, 16, 8, 128] and actual['shape'] == [128, 16, 1024]
                and all(live[x] == written[x] == actual[x] for x in
                    ('storage_identity', 'storage_offset', 'dtype', 'device', 'storage_nbytes'))
                and capture['attention'][layer][kind]['physical_pages'] == pages[0],
                'TND33_NATURAL_CACHE_LIFETIME_OR_LOGICAL_READBACK_DIFFERS')
        result[layer] = row
    return result


def audit_run(run, phase):
    run = Path(run).resolve(strict=True);path = run/'result.json';report = read_json(path)
    require(report.get('schema') == 'demoncat.paged33.actual_fai_worker_run.v1'
        and report.get('phase') == phase and report.get('status') ==
        ('NATURAL_STOCK_WORKER_COMPLETE' if phase == 'stock' else SUCCESS)
        and [p.get('phase') for p in report.get('phases', [])] == list(PHASES)
        and report.get('execution_contract', {}).get('schema') == 'demoncat.paged33.execution_contract.v1',
        'TND33_COMPLETE_BOUND_RUN_REQUIRED')
    require(report.get('statistics') is (phase != 'stock'), 'TND33_COLLECTOR_ARM_DIFFERS')
    tokens = report['warm_healthy']['prompt_token_ids']
    require(len(tokens) == 1024 and report['prompt']['token_count'] == 1024 and
        hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest() == report['prompt']['token_sha256'],
        'TND33_ACTUAL_PROMPT_HASH_DIFFERS')
    rows, boundaries = {}, {}
    for item in report['phases']:
        name = item['phase'];finish = item['finish'];capture = finish['capture'];cleanup = finish['cleanup']
        require(item['response']['prompt_token_ids'] == tokens and item['response']['outputs'] == report['warm_healthy']['outputs']
            and item['response']['request_id'] == item['mapping']['external_request_id']
            and finish.get('closed') is True and not finish.get('error') and not finish.get('cleanup_error')
            and cleanup.get('scope_cleared') is True and cleanup.get('local_streams_drained') is True
            and cleanup.get('owned_hooks') == cleanup.get('owned_api_wrappers') == 0
            and not cleanup.get('failures') and cleanup.get('core_methods_replaced') is False,
            'TND33_COMPLETE_REQUEST_OR_OWNED_RECOVERY_DIFFERS')
        require(read_json(run/name/'runtime_result.json') == capture, 'TND33_CAPTURE_COPY_DIFFERS_FROM_ORIGINAL')
        require(item['setup']['source'] == capture['audit']['source'] and item['setup']['candidate_count'] == 28
            and item['setup']['stock_read_only'] is (phase == 'stock' or name == 'healthy_after'),
            'TND33_ACTUAL_CAPTURE_SOURCE_DIFFERS')
        rows[name] = payload_rows(run, capture)
        boundaries[name] = boundary_rows(capture, item, tokens)
        if name != 'baseline_before':compare_payload(rows['baseline_before'], rows[name])
    require(report['healthy_after_owned_observers_removed']['outputs'] == report['warm_healthy']['outputs'],
            'TND33_POST_CLEAN_HEALTHY_GENERATION_DIFFERS')
    return dict(run=run, report=report, result_sha256=sha(path), payloads=rows, boundaries=boundaries)


def audit_native(baseline, build_root):
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd import verify_bundle
    from demoncat.runtime.vllm_ascend.worker_extension.paged33_loading import (
        FIA_SYMBOLS, NORM_SYMBOLS, PRIVATE_LIB, STOCK_FILES, STOCK_ROOT, STOCK_VENDOR, LOADER_HEADER_SHA)
    from fault_injection.native_adapters.attention28.launch import bundle_kernel_abi, verify_launch
    from fault_injection.native_adapters.attention28.fia.fai_fd_geometry import validate_fd_geometry
    build_root = Path(build_root);completion = build_root/'evidence/completion.json'
    bundle = verify_bundle(completion, sha(completion), build_root/'stage/vendors/dcat_f33_fai_fd_transformer')
    data = bundle['completion'];report = baseline['report'];run = baseline['run']
    require(report.get('completion_sha256') == sha(completion)
        and report.get('schema_sha256') == data['fai_public_schema_sha256'], 'TND33_RUN_FRESH_BUILD_DIFFERS')
    api = report['library_coexistence_after_standard_worker_initialization']
    private_path = '/native/stage/vendors/dcat_f33_fai_fd_transformer/' + PRIVATE_LIB
    stock_path = STOCK_VENDOR/PRIVATE_LIB
    require(api.get('platform_ready') is True and api.get('model_or_norm_replaced') is False
        and api.get('private_global_library_path_added') is False and api['bindings']['distinct_library_handles'] is True
        and api.get('stock_files') == STOCK_FILES and api.get('stock_loader_source_sha256') == LOADER_HEADER_SHA
        and {r['symbol'] for r in api['bindings']['private_fia']} == set(FIA_SYMBOLS)
        and {r['symbol'] for r in api['bindings']['stock_norm']} == set(NORM_SYMBOLS)
        and all(r['sha256'] == bundle['artifacts'][PRIVATE_LIB]['sha256'] and r['path'] == private_path
                for r in api['bindings']['private_fia'])
        and all(r['path'] == str(stock_path) and r['sha256'] == STOCK_FILES[stock_path.relative_to(STOCK_ROOT).as_posix()]
                for r in api['bindings']['stock_norm']),
        'TND33_ACTUAL_WORKER_API_LIBRARY_BINDING_DIFFERS')
    schema = read_json(build_root/'evidence/fai_schema/fai_abi_schema.json')
    metadata, abi = bundle_kernel_abi(bundle['vendor'], bundle['artifacts'], data['kernel_binary_sha256'])
    calls, cores, active, keys = [], set(), set(), set()
    for phase in report['phases']:
        name = phase['phase'];native_calls = phase['finish']['capture']['native_calls']
        if name == 'healthy_after':
            require(native_calls == [], 'TND33_HEALTHY_AFTER_NATIVE_SCOPE_PRESENT');continue
        require(len(native_calls) == 28 and {r['module_path'] for r in native_calls} == LAYERS,
                'TND33_NATIVE_LAYER_CENSUS_DIFFERS')
        hosts = [json.loads(line) for line in (run/name/'native/host.jsonl').read_text().splitlines()]
        require(len(hosts) == len({h['token'] for h in hosts}) == 28, 'TND33_NATIVE_HOST_CALLBACK_SET_DIFFERS')
        for row in native_calls:
            call_id = row['call_id'];host, = [h for h in hosts if h['token'] == call_id]
            boundary = baseline['boundaries'][name][row['module_path']]
            require(host.get('schema') == 'demoncat.paged33.fai_fd_host.v1' and host.get('eligible') is True
                and host.get('policy_applied') is False and host.get('mode') == 'off'
                and host.get('native_s2') == host.get('selected_s2') == 512
                and row['module_object_id'] == boundary['module_object_id'] and row['scheduler'] == boundary['scheduler']
                and row.get('cache_repacked') is False and row.get('api_arguments_changed') is False
                and row.get('output_mutated') is False, 'TND33_BASELINE_NATIVE_SCOPE_DIFFERS')
            observed = verify_launch(run/'runtime', call_id, schema=schema, host=host,
                kernel_sha256=data['kernel_binary_sha256'], kernel_metadata=metadata, kernel_abi=abi)
            require(all(observed[key] == row['native'][key] for key in ('raw_sha256', 'args_sha256',
                'tiling_key', 'block_dim', 'schedule_mode', 'typed', 'kernel_sha256', 'kernel_argument_abi')),
                'TND33_RECORDED_NATIVE_FACTS_DIFFER_FROM_RAW_REPLAY')
            require(observed['typed']['numBlocks'] == boundary['key']['shape'][0]
                and observed['typed']['maxNumBlocksPerBatch'] == boundary['page_table']['shape'][1]
                and observed['typed']['maskType'] == 1 and observed['typed']['sparseMode'] == 3,
                'TND33_FINAL_NATIVE_API_PAGE_OR_MASK_DIFFERS')
            geometry = validate_fd_geometry(observed['typed'], host, key=observed['tiling_key'],
                block_dim=observed['block_dim'], schedule_mode=observed['schedule_mode'], arm='baseline', kv_length=1025)
            cores.add(geometry['actual_aic']);active.add(geometry['actual_active_cores']);keys.add(observed['tiling_key'])
            calls.append(dict(phase=name, module_path=row['module_path'], call_id=call_id,
                raw_sha256=observed['raw_sha256'], args_sha256=observed['args_sha256'],
                actual_aic=geometry['actual_aic'], active_blocks=geometry['actual_active_cores']))
    require(len(calls) == len({r['call_id'] for r in calls}) == 112 and len(cores) == 1
        and keys == {5100000000010200103}, 'TND33_ACTUAL_PLATFORM_OR_BASE_KEY_UNPROVEN')
    return dict(native_baseline_launches=len(calls), actual_aic=sorted(cores), active_blocks=sorted(active),
                native_keys=sorted(keys), native_calls=calls, completion_sha256=sha(completion))


def review_bridge(stock_run, baseline_run, build_root):
    stock, baseline = audit_run(stock_run, 'stock'), audit_run(baseline_run, 'baseline')
    require(stock['run'] != baseline['run'] and stock['report']['execution_contract'] == baseline['report']['execution_contract']
        and stock['report']['prompt'] == baseline['report']['prompt'], 'TND33_STOCK_PRIVATE_INPUT_OR_SOURCE_DIFFERS')
    compared = 0
    for phase in PHASES:
        compared += compare_payload(stock['payloads'][phase], baseline['payloads'][phase])
        a = next(p for p in stock['report']['phases'] if p['phase'] == phase)
        b = next(p for p in baseline['report']['phases'] if p['phase'] == phase)
        require(a['setup']['source'] == b['setup']['source'] and a['setup']['module_sha256'] == b['setup']['module_sha256']
            and a['response']['outputs'] == b['response']['outputs'], 'TND33_BRIDGE_ACTUAL_SOURCE_OR_RESPONSE_DIFFERS')
    native = audit_native(baseline, build_root)
    return dict(schema='demoncat.paged33.stock_private_bridge.v1', status='PASS_EXACT_COMPLETE_STOCK_PRIVATE_BRIDGE',
        reviewer_source_sha256=sha(Path(__file__)), stock_result_sha256=stock['result_sha256'],
        baseline_result_sha256=baseline['result_sha256'], execution_contract=baseline['report']['execution_contract'],
        full_payload_file_pairs=435, complete_payload_bytes_compared=compared, raw_payloads_finite=True,
        runtime_launches_reverified=True, statistical_first_difference='NOT_READ', c2=False, c3=False, **native)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stock-run', 'baseline-run', 'build-root', 'out'):p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args();sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2]/'src')]
    require(not args.out.exists(), 'TND33_FRESH_BRIDGE_OUTPUT_REQUIRED')
    try:result = review_bridge(args.stock_run, args.baseline_run, args.build_root)
    except Exception as error:
        result = dict(schema='demoncat.paged33.stock_private_bridge.v1', status='FAILED_WITH_EVIDENCE',
            error=repr(error), reviewer_source_sha256=sha(Path(__file__)), c2=False, c3=False)
    with args.out.open('x', encoding='utf-8') as stream:json.dump(result, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'c2', 'c3')}))
    return 0 if result['status'].startswith('PASS_') else 2


if __name__ == '__main__':raise SystemExit(main())
