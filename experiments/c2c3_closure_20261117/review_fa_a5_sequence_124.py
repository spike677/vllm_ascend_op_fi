"""CPU-only review of a completed FA sequence; original payloads stay on 124."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import types


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=sha(path))


def load(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_inventory(root, reader):
    manifest = load(root / 'source_manifest.json')
    result = reader.check_files(root, manifest['files'])
    require(not result['mismatches'], 'FROZEN_SOURCE_CHANGED')
    return dict(manifest=record(root / 'source_manifest.json'), commit=manifest['base_commit'], **result)


def input_hashes():
    """Reconstruct the declared exact FP16 arithmetic without torch/NPU."""
    results = []
    for heads, factor, offset in ((16, 17, 3), (8, 29, 5), (8, 43, 7)):
        period = b''.join(struct.pack('<e', ((i * factor + offset) % 257 - 128) / 128) for i in range(257))
        count = heads * 2048 * 128
        one = period * (count // 257) + period[:2 * (count % 257)]
        digest = hashlib.sha256()
        for _ in range(8):
            digest.update(one)
        results.append(digest.hexdigest())
    digest = hashlib.sha256()
    for row in range(2048):
        digest.update(b'\0' * (row + 1) + b'\1' * (2047 - row))
    results.append(digest.hexdigest())
    return results


def audit_gate(gate):
    require(gate['cards'] == [5] and gate['idle'] is True and not gate['blockers'], 'A5_CARD_SCOPE_DIFFERS')
    stale = gate['stale_checkpoint_allocations']
    proof = gate['stale_checkpoint_review']
    require(len(stale) == 1 and proof['status'] == 'NO_LIVE_OWNER_CORROBORATED', 'A5_SINGLE_TARGET_REVIEW_REQUIRED')
    original = stale[0]['original_blocker']
    require(original['card'] == 5 and original['pod_uid'] == '33991897-e3ed-4910-aa2c-835b9fe682a0'
            and original['checkpoint_sha256'] == 'b570d0ea0d0280157a26d48c536a6aacff8cdf4fb55037f3b3882e85c1395414',
            'A5_AUTHORIZED_OWNER_OR_CHECKPOINT_DIFFERS')
    require(canonical({k: v for k, v in proof.items() if k != 'evidence_sha256'}) == proof['evidence_sha256']
            == stale[0]['release_evidence_sha256'], 'A5_GATE_PROOF_HASH_DIFFERS')
    require(proof['post_gate']['blockers'] == [original] and
            canonical(gate['oci_tasks']) == proof['oci_tasks_sha256'], 'A5_OLD_EVIDENCE_NOT_PRESERVED')
    require(all(proof[k] is False for k in ('internal_allocation_clearance_proven',
            'driver_initialization_available_proven', 'exclusive_device_reservation')), 'A5_GATE_OVERCLAIMS')
    return dict(status=proof['status'], cards=[5], original_blocker=original,
        evidence_sha256=proof['evidence_sha256'], internal_allocation_clearance_proven=False,
        driver_initialization_available_proven=False, exclusive_device_reservation=False,
        original_api_payloads_independently_replayed=False)


def full_payload(folder, payload, reference, np):
    require(payload['output'] == {'kind': 'tuple', 'length': 6}
            and set(payload) == {'output', *(f'output-{i}' for i in range(6))}, 'COMPLETE_SIX_LEAF_OUTPUT_REQUIRED')
    formats = {'torch.float16': '<f2', 'torch.float32': '<f4', 'torch.int64': '<i8'}
    baseline_payload = load(reference / 'call.json')['payload']
    output = {}
    for name in sorted(set(payload) - {'output'}):
        descriptor = payload[name]
        require(all(descriptor[key] == baseline_payload[name][key] for key in ('kind', 'shape', 'dtype', 'bytes')),
                'PAYLOAD_STRUCTURE_DIFFERS_FROM_BASELINE')
        path = folder / (name + '.bin')
        require(descriptor['kind'] == 'tensor' and descriptor['dtype'] in formats, 'UNREVIEWED_PAYLOAD_DTYPE')
        dtype = np.dtype(formats[descriptor['dtype']])
        actual = record(path)
        require(actual['bytes'] == math.prod(descriptor['shape']) * dtype.itemsize == descriptor['bytes']
                and actual['sha256'] == descriptor['sha256'], 'COMPLETE_PAYLOAD_IDENTITY_DIFFERS')
        finite, changed, changed_bytes, maximum = True, 0, 0, 0.0
        with path.open('rb') as stream, (reference / path.name).open('rb') as baseline:
            while block := stream.read(1024 * 1024):
                other = baseline.read(len(block))
                require(len(other) == len(block), 'BASELINE_PAYLOAD_LENGTH_DIFFERS')
                values, original = np.frombuffer(block, dtype=dtype), np.frombuffer(other, dtype=dtype)
                finite &= bool(np.isfinite(values).all())
                changed += int(np.count_nonzero(values != original))
                changed_bytes += int(np.count_nonzero(np.frombuffer(block, dtype='u1') != np.frombuffer(other, dtype='u1')))
                if len(values):
                    maximum = max(maximum, float(np.max(np.abs(values.astype('float64') - original.astype('float64')))))
            require(not baseline.read(1), 'BASELINE_HAS_EXTRA_BYTES')
        require(finite and descriptor['finite'] is True, 'NONFINITE_COMPLETE_PAYLOAD')
        output[name] = dict(**actual, shape=descriptor['shape'], dtype=descriptor['dtype'], finite=finite,
            byte_exact_vs_baseline=changed_bytes == 0, changed_bytes=changed_bytes,
            changed_elements=changed, max_absolute_difference=maximum)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    root = args.root
    helper_path = root / 'agent7_audit_cpu_builds_v5.py'
    require(sha(helper_path) == '6a6747921760a819e45e43ea0c289cad12620147f4377a089757884edaa0c55a', 'AUDITED_IMAGE_READER_CHANGED')
    spec = importlib.util.spec_from_file_location('reader', helper_path)
    reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    sources = {name: source_inventory(root / name, reader) for name in ('source-a33', 'source-a34')}
    # Avoid executing package __init__ code. Only these reviewed file readers
    # and the bounded public POD CPU decoder are loaded, never a runtime hook.
    source = root / 'source-a33'
    for name in ('fault_injection', 'fault_injection.native_adapters', 'fault_injection.native_adapters.attention28',
                 'fault_injection.native_adapters.attention28.fa'):
        module = types.ModuleType(name); module.__path__ = [str(source / name.replace('.', '/'))]
        sys.modules[name] = module
    from fault_injection.native_adapters.attention28.fa import launch
    from fault_injection.native_adapters.attention28.launch import observe_launch
    attempt = root / 'fa28-launch-short-a5'
    controller_path = attempt / 'controller.json'
    require(sha(controller_path) == '8302e00c12f98a5b37040163179fc196ef36c89af7a75f2442b5af8eb3a4437e', 'ORIGINAL_CONTROLLER_CHANGED')
    controller = load(controller_path); require(len(controller['cases']) == 1, 'SINGLE_CASE_REQUIRED')
    case = controller['cases'][0]
    require(case['exit_code'] == 0 and case['container_state']['OOMKilled'] is False, 'CASE_DID_NOT_COMPLETE')
    command = case['command']
    require(f'type=bind,src={source},dst=/driver,readonly' in command, 'ACTUAL_PROBE_SOURCE_MOUNT_DIFFERS')
    require(sha(root / 'source-a34/experiments/c2c3_closure_20261117/run_queue_124.py') == controller['controller_sha256']
            and sha(root / 'source-a34/experiments/c2c3_closure_20261117/queue_contract.py') == controller['contract_sha256']
            and sha(root / 'fa28-launch-short-a5-cases.json') == controller['case_manifest_sha256'], 'QUEUE_SOURCE_OR_MANIFEST_CHANGED')
    gates = {name: audit_gate(case[name]) for name in ('gate', 'post_cleanup_gate')}
    run = attempt / case['id'] / 'run'; runtime = run.parent / 'runtime'
    result_path = run / 'result.json'; result = load(result_path)
    require(sha(result_path) == 'e919244c2a615ec71413fe859fd28a4c26ecd929941a1e0a1191bb292832eea6'
            and result['actual_npu_api_calls'] == 4 and not result['failures'], 'ORIGINAL_RESULT_CHANGED')
    build = root / 'fa28-fresh-build-a1/build/run'
    bundle = launch.verify_bundle(build, result['completion_sha256'])
    image = reader.image_metadata(case['image_id'])
    sdk_checks = []
    for row in bundle['completion']['sdk_interface_files']:
        actual = reader.image_file(image, '/usr/local/Ascend/cann-9.1.0/' + row['path'])
        require(actual['sha256'] == row['sha256'] and actual['bytes'] == row['bytes'], 'ACTUAL_IMAGE_SDK_DIFFERS')
        sdk_checks.append(dict(relative_path=row['path'], sha256=actual['sha256'], bytes=actual['bytes']))
    observer_receipt = root / 'attention-preflight-a4/cann910/attention-preflight/result.json'
    require(sha(observer_receipt) == '428b3385a2d742b5c66746de4e85b4355bafcd96d4c367314a7679560858d74e', 'OBSERVER_RECEIPT_CHANGED')
    observer = load(observer_receipt)
    observer_binary = observer_receipt.parent / 'libdcat_attention_observer.so'
    require(sha(observer_binary) == observer['observer_library']['sha256'], 'OBSERVER_BINARY_CHANGED')
    manifest = load(run / 'inputs.json')
    require(manifest['batch'] == 8 and [row['shape'] for row in manifest['tensors']] ==
            [[8, 16, 2048, 128], [8, 8, 2048, 128], [8, 8, 2048, 128], [2048, 2048]],
            'FIXED_INPUT_SHAPES_DIFFER')
    require([row['sha256'] for row in manifest['tensors']] == input_hashes(), 'DECLARED_INPUT_GENERATOR_BYTES_DIFFER')
    runtime_bytes = (runtime / 'runtime.jsonl').read_bytes(); lines = runtime_bytes.splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    require(sum(row['event'] == 'launch' for row in events) == 4, 'ACTUAL_LAUNCH_COUNT_DIFFERS')
    os.environ.pop('LD_PRELOAD', None); os.environ.pop('LD_LIBRARY_PATH', None)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    import numpy as np
    rows = []
    for index, phase in enumerate(('baseline', 'baseline_repeat', 'fault', 'off')):
        call = result['calls'][index]; folder = run / f'{index}-{phase}'
        require(call['index'] == index and call['phase'] == phase and load(folder / 'call.json') == call, 'CALL_LEDGER_DIFFERS')
        observed = observe_launch(runtime, call['call_id'], kernel_sha256=bundle['completion']['kernel_binary_sha256'],
            tiling_size=bundle['schema']['size'], kernel_metadata=bundle['metadata'], kernel_abi=bundle['abi'])
        raw = bytes.fromhex(observed['raw_hex'])
        require((folder / 'final-tiling.bin').read_bytes() == raw, 'FINAL_POD_BYTES_DIFFER')
        pod = launch.decode_public_pod(raw, bundle['schema'])
        decoded = subprocess.run([bundle['decoder'], str(folder / 'final-tiling.bin')], capture_output=True, text=True, timeout=15)
        require(decoded.returncode == 0, 'CPU_DECODER_FAILED:' + decoded.stderr[-250:])
        launch.verify_compiled_decoder_document(json.loads(decoded.stdout), pod)
        require(pod == call['launch']['typed'] and launch.semantic_pod(raw, bundle['schema']) == call['launch']['semantic_pod'], 'RECORDED_POD_DIFFERS')
        for key in ('args_sha256', 'raw_sha256', 'tiling_key', 'block_dim', 'schedule_mode', 'tiling_pointer_offset', 'tiling_data_offset'):
            require(observed[key] == call['launch'][key], 'RECORDED_RUNTIME_BINDING_DIFFERS:' + key)
        end = next(i for i, event in enumerate(events) if event['call_id'] == call['call_id'] and event['event'] == 'end')
        require(hashlib.sha256(b''.join(lines[:end + 1])).hexdigest() == call['launch']['runtime_log_sha256'], 'ORIGINAL_RUNTIME_PREFIX_HASH_DIFFERS')
        expected = dict(m=64, n_ratio=4) if phase == 'fault' else dict(m=128, n_ratio=8)
        factors = dict(m=pod['coreParams.s1BaseSize'], n_ratio=pod['coreParams.nRatio'])
        require(factors == expected == call['actual_factors'] and pod['coreParams.s2BaseSize'] == 128, 'ACTUAL_STRATEGY_DIFFERS')
        if phase != 'off':
            hosts = [json.loads(line) for line in (folder / 'host.jsonl').read_text().splitlines()]
            require(len(hosts) == 1 and hosts[0]['token'] == call['call_id'] and bytes.fromhex(hosts[0]['raw_hex']) == raw, 'HOST_RAW_OR_OWNER_DIFFERS')
            launch.verify_host_projection(pod, hosts[0])
        else:
            require(not (folder / 'host.jsonl').exists() and load(folder / 'requested_plan.json') == {'plan': None, 'unscoped_off': True}, 'OFF_NOT_UNSCOPED')
        payload = full_payload(folder, call['payload'], run / '0-baseline', np)
        exact = all(row['byte_exact_vs_baseline'] for row in payload.values())
        require(exact is (phase != 'fault'), 'COMPLETE_SEQUENCE_EFFECT_OR_RECOVERY_DIFFERS')
        require(call['full_inputs_unchanged'] is True and call['original_api_return_preserved'] is True, 'EXECUTION_INPUT_ASSERTION_FAILED')
        rows.append(dict(phase=phase, call_id=call['call_id'], actual_factors=factors, s2_base=128,
            s2_group=128*factors['n_ratio'], tiling_key=observed['tiling_key'], block_dim=observed['block_dim'],
            schedule_mode=observed['schedule_mode'], raw_sha256=observed['raw_sha256'], args_sha256=observed['args_sha256'],
            public_pod_fields_compared=205, registered_kernel_sha256=observed['kernel_sha256'],
            payload=payload, complete_byte_exact_vs_baseline=exact, post_device_input_check_recorded=True))
    report = dict(schema='demoncat.agent7.fa_a5_review.v1', status='STANDALONE_FOUR_CALL_SEQUENCE_INDEPENDENTLY_REVIEWED',
        checks_passed=True, reviewer=record(__file__), controller=record(controller_path), result=record(result_path),
        source_snapshots=sources, actual_probe_source='source-a33', controller_source_snapshot_declared_by_root='source-a34',
        controller_snapshot_path_not_unique_from_shared_script_hash=True,
        source_identity_limits='Controller frozen_scripts/source_manifests arrays are empty; exact code/manifest matches were checked post-run, not upgraded to a recorded launch-time snapshot assertion.',
        build_completion=record(build / 'evidence/completion.json'), vendor_artifacts_verified=len(bundle['files']),
        actual_image_id=image['Id'], exact_sdk_header_files=sdk_checks, observer_receipt=record(observer_receipt),
        observer_library=record(observer_binary), runtime_log=record(runtime / 'runtime.jsonl'),
        registered_binary=record(runtime / 'registered-1-1.bin'), cpu_decoder=record(Path(bundle['decoder'])),
        actual_launch_count=4, actual_successful_sync_count=sum(e['event']=='device_sync' and e['result']==0 for e in events),
        input_manifest=record(run / 'inputs.json'), declared_inputs_reconstructed_on_cpu=True,
        input_limit='Each live post-call full device readback was checked by the pinned probe and recorded as true; separate post-call input byte files were not saved for offline rehash.',
        gates=gates, gate_scope_limit='source-a34 default stale review was overly broad; this historical a5 actual scope matches only the approved UID/checkpoint on card5. Future attempts require the separately reviewed explicit opt-in fix.',
        calls=rows, qualifications=dict(standalone_native_strategy_effect_and_off=True, stock_package_bridge=False,
            natural_model=False, standard_worker=False, batch_causal_chain=False, m_only_attribution=False,
            repeated_fault_stability=False, statistics_mix_e5=False, c2=False, c3=False),
        no_raw_payloads_exported=True, audit_npu_calls=0, audit_container_starts=0)
    require(sha(controller_path) == report['controller']['sha256'] and sha(result_path) == report['result']['sha256'], 'ORIGINALS_CHANGED_DURING_REVIEW')
    with (args.out / 'summary.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], checks_passed=True, actual_launch_count=4)))


if __name__ == '__main__':
    main()
