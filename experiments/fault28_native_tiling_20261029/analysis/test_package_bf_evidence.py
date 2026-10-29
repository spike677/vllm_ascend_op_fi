"""Temporary, explicitly synthetic CPU fixtures for the publication allowlist.

No NPU execution, no real acceptance, and no final artifact is created/retained.
Collector provenance bytes are copied from the real frozen stage; all run and
scientific-decision records below are test doubles solely for packager testing.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pack', HERE / 'package_bf_evidence.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def payload(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def fixture(root, stage):
    attempt = root / p.ATTEMPT
    attempt.mkdir()
    def write(name, value, raw=False):
        path = attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if raw else payload(value))
        return p.digest(path.read_bytes())
    source = {}
    for name in ('controller.py', 'env.sh', 'hf_runtime_probe.py', 'vendor_exec.py', 'collector_stage.py'):
        source[name] = write('source/' + name, b'# SYNTHETIC CPU FIXTURE; NOT EXECUTED\n', True)
    manifest = json.loads((stage / 'stage_manifest.json').read_bytes())
    manifest_sha = p.digest((stage / 'stage_manifest.json').read_bytes())
    collector = {'variant': 'task_local_fp32_high_v1', 'precision': 'high', 'modified_official_collector': True,
                 'manifest_file': manifest['stage_root'] + '/stage_manifest.json', 'manifest_sha256': manifest_sha,
                 'package_root': manifest['stage_root'] + '/packages',
                 'processor_file': manifest['stage_root'] + '/packages/' + p.PROCESSOR,
                 'original_processor_sha256': manifest['original_processor_sha256'],
                 'patched_processor_sha256': manifest['patched_processor_sha256'],
                 'source_manifest_sha256': p.digest((stage / 'source_manifest.json').read_bytes()),
                 'patch_sha256': p.digest((stage / 'collector.patch').read_bytes()),
                 'artifact_count': len(manifest['artifacts']), 'provenance': manifest['provenance'], 'policy': manifest['policy']}
    collector['actual_loaded'] = {'module': 'msprobe.core.dump.data_dump.data_processor.pytorch_processor',
                                  'path': collector['processor_file'], 'method_code_file': collector['processor_file'],
                                  'sha256': collector['patched_processor_sha256']}
    selected, completed, native_hashes, runtime_hashes = [], [], {}, {}
    for arm, mode in [('B', 'off'), ('F', 'on')]:
        case = {'capture_msprobe': True, 'capture_full_inputs': False, 'forwards': 2, 'fa_mode': mode,
                'statistics_variant': collector['variant'], 'approved_collector_manifest_file': collector['manifest_file'],
                'approved_collector_manifest_sha256': manifest_sha}
        case_name = 'setup/' + arm + '.json'
        source[case_name] = write('source/' + case_name, case)
        selected.append({'case_id': arm, 'file': case_name, 'case': case})
        config = {'task': 'statistics', 'level': 'mix', 'step': [0, 1], 'rank': [], 'precision': 'high',
                  'async_dump': False, 'statistics': {'summary_mode': 'statistics', 'tensor_list': []}}
        config_sha = write(arm + '/config.json', config)
        native, native_hashes[arm] = [], {'config': config_sha}
        for step in (0, 1):
            for kind in ('dump', 'stack', 'construct'):
                name = f'observation/step{step}/rank0/{kind}.json'
                value = {'synthetic_fixture': True}
                if kind == 'dump':
                    value = {'task': 'statistics', 'level': 'mix', 'dump_data_dir': None, 'data': {
                        'Functional.scaled_dot_product_attention.1.forward': {'output': [{'type': 'torch.Tensor',
                        'dtype': 'torch.float16', 'shape': [1], 'Max': 1., 'Min': 0., 'Mean': 0.5, 'Norm': 1.}]}}}
                h = write(arm + '/' + name, value)
                native.append({'path': name, 'bytes': len(payload(value)), 'sha256': h})
                native_hashes[arm][kind] = h
        collector_sha = write(arm + '/collector_provenance.json', collector)
        receipt = {'status': 'CAPTURE_COMPLETED', 'rank': 0, 'steps': [{'step': 0}, {'step': 1}],
                   'config_sha256': config_sha, 'native_files': native, 'statistics_collector': collector,
                   'collector_provenance_sha256': collector_sha}
        receipt_sha = write(arm + '/receipt.json', receipt)
        calls = []
        for i in range(56):
            name = f'native_calls/call_{i:03d}.json'
            details = {'call_id': f'fixture_{arm}_{i}', 'native_index': i, 'forward_index': i // 28,
                       'layer_index': i % 28, 'status': 'returned', 'selected_layer_private_inputs': {},
                       'selected_layer_private_output': None}
            h = write(arm + '/' + name, details)
            calls.append({**{k: details[k] for k in ('call_id', 'native_index', 'forward_index', 'layer_index', 'status')},
                          'file': name, 'sha256': h})
        events = [{'token': f'fixture_{arm}_{i}', 'forward': str(i), 'mode': mode, 'callback_status': 0,
                   'eligible': True, 'policy_applied': mode == 'on', 'layer': 1, 'phase': 'prefill'} for i in (0, 1)]
        write(arm + '/fa_host_events.jsonl', b''.join(json.dumps(e).encode() + b'\n' for e in events), True)
        launch = {'schema': 'fault28.fa.vendor_launch.v1', 'case_id': arm, 'case_sha256': source[case_name],
                  'probe_sha256': source['hf_runtime_probe.py'], 'launcher_sha256': source['vendor_exec.py']}
        launch_sha = write(arm + '.vendor_launch.json', launch)
        result = {'case_id': arm, 'status': 'CAPTURE_COMPLETED', 'capture_msprobe': True, 'capture_full_inputs': False,
                  'forward_count': 2, 'prefill_certified': True, 'batch_size': 8, 'prompt_length': 2048,
                  'fa_mode_plan': {'modes': [mode, mode]}, 'source_case_sha256': source[case_name],
                  'source_sha256': {'hf_runtime_probe.py': source['hf_runtime_probe.py']},
                  'private_vendor': {'launcher_receipt_sha256': launch_sha}, 'msprobe_receipt_sha256': receipt_sha,
                  'native_calls': calls, 'statistics_collector': collector}
        h = write(arm + '/result.json', result)
        runtime_hashes['/synthetic/outputs/' + p.ATTEMPT + '/' + arm + '/result.json'] = h
        completed.append({'case_id': arm, 'rc': 0, 'status': 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE', 'result_sha256': h})
        write(arm + '.controller.json', {'case_id': arm, 'rc': 0})
    write('source_manifest.json', source)
    write('selected_cases.json', selected)
    write('completion.json', {'status': 'CAPTURE_COMPLETED', 'owned_container_removed': True, 'cleanup_errors': [],
                             'device_lock_released': True, 'capture': 'none', 'probe': 'hf_runtime_probe.py',
                             'external_container_mutations': False, 'cases': completed})
    summaries = root / 'summaries'
    summaries.mkdir()
    docs = {'off_compatibility.json': {'status': 'PASS_STOCK_OFF_COMPATIBILITY'},
            'mechanism_validation.json': {'status': 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS'},
            'formal_runtime_crosscheck.json': {'status': 'PASS_RUNTIME_CROSSCHECK', 'input_sha256': runtime_hashes},
            'collector_validation.json': {'status': 'SYNTHETIC_CPU_TEST_DOUBLE_NOT_EXECUTION_PROOF'}}
    for step in (0, 1):
        first = {arm: {'name': 'Functional.scaled_dot_product_attention.1.forward', 'attention_kind': 'PYTORCH_SDPA_API'}
                 for arm in p.ARMS}
        first.update(input={'numerical_differences': []}, output={'numerical_differences': [{'synthetic': True}]})
        docs[f'independent_msprobe_step{step}.json'] = {'status': 'ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE',
            'alignment': {'complete_alignment': True}, 'first_observed_model_api_numerical_difference': first,
            'attention_boundary_coverage': {arm: {'complete_hf_sdpa_module_coverage': True,
                                                 'record_counts_by_boundary': {'PYTORCH_SDPA_API': 28}} for arm in p.ARMS},
            'input_sha256': native_hashes}
        docs[f'official_default_msprobe_step{step}.json'] = {'fault_mechanism_accepted': False,
            'first_observed_model_api_numerical_difference': {'B': {'name': 'Tensor.pow.7.forward'}}}
    acceptance = {'status': p.ACCEPTANCE, 'attempt': p.ATTEMPT, 'native_msprobe_first_difference_pass': True,
                  'mechanism_validation_pass': True, 'formal_runtime_consistency_pass': True,
                  'collector_variant': p.COLLECTOR_VARIANT, 'official_default_msprobe_exact_location_pass': False,
                  'historical_defect_reproduced': False,
                  'source_reports': [{'path': 'summaries/' + name, 'sha256': p.digest(payload(value))} for name, value in docs.items()]}
    docs['execution_acceptance.json'] = acceptance
    for name, value in docs.items():
        (summaries / name).write_bytes(payload(value))
    return attempt, [name + '=' + str(summaries / name) for name in sorted(docs)]


def main():
    stage = HERE / 'draft/packager_fixture_provenance'
    license_path = HERE / 'collector_high/LICENSE.MulanPSL-2.0'
    checks = []
    def rejects(name, function):
        try:
            function()
        except (RuntimeError, KeyError, ValueError):
            checks.append(name)
            return
        raise AssertionError('Invalid fixture accepted: ' + name)
    workspace = HERE.parents[2].resolve()
    with tempfile.TemporaryDirectory(prefix='_bf_pkg_', dir=workspace) as td:
        root = Path(td).resolve()
        assert root.is_relative_to(workspace) and root.name.startswith('_bf_pkg_')
        attempt, summaries = fixture(root, stage)
        output = root / 'artifacts/attention_tiling' / p.ARTIFACT
        archive = output.parent / (p.ARTIFACT + '.tar.gz')
        result = p.create_package(attempt, output, summaries, stage, license_path, archive)
        assert result['status'] == p.verify_package(output)['status'] == p.verify_archive(archive)['status'] == 'VERIFIED_PACKAGE_INTEGRITY'
        checks.extend(['valid_synthetic_directory_roundtrip', 'valid_synthetic_archive_roundtrip'])
        def mutate(name, value):
            path = output / name
            old = path.read_bytes()
            path.write_bytes(value)
            try:
                rejects('reject_tampered_' + name, lambda: p.verify_package(output))
            finally:
                path.write_bytes(old)
        mutate('B/observation/step0/rank0/dump.json', b'{}')
        mutate('collector/patched_pytorch_processor.py', b'# changed')
        mutate('summaries/execution_acceptance.json', payload({'status': 'PASS'}))
        extra = output / 'B/unapproved.pt';extra.write_bytes(b'fake tensor')
        rejects('reject_extra_tensor_file', lambda: p.verify_package(output));extra.unlink()
        rejects('reject_private_payload_reference', lambda: p.no_private_payload({'private_raw_file': 'private_replay/x.bin'}))
        rejects('reject_tensor_values_in_json', lambda: p.no_private_payload({'type': 'torch.Tensor', 'value': [1, 2]}))
        rejects('reject_path_traversal', lambda: p.allowed('../escape.json'))
        rejects('reject_wrong_attempt', lambda: p.create_package(root, output, summaries, stage, license_path))
        rejects('reject_missing_independent_report', lambda: p.create_package(attempt,
                    root / 'new/artifacts/attention_tiling' / p.ARTIFACT, summaries[:-1], stage, license_path))
        rejects('reject_overwrite', lambda: p.create_package(attempt, output, summaries, stage, license_path))
        malicious = root / 'malicious.tar.gz'
        with tarfile.open(malicious, 'w:gz') as tar:
            member = tarfile.TarInfo(p.ARTIFACT + '/../escape');member.size = 1;tar.addfile(member, io.BytesIO(b'x'))
        rejects('reject_archive_traversal', lambda: p.verify_archive(malicious))
        # Tamper in the supplied decision before creation, so rejection is semantic
        # rather than merely a package-manifest mismatch.
        decision = root / 'summaries/execution_acceptance.json'
        saved = decision.read_bytes();bad = json.loads(saved);bad['official_default_msprobe_exact_location_pass'] = True
        decision.write_bytes(payload(bad))
        rejects('reject_relabelled_default_collector_pass', lambda: p.create_package(attempt,
                    root / 'relabel/artifacts/attention_tiling' / p.ARTIFACT, summaries, stage, license_path))
        decision.write_bytes(saved)
        bad = json.loads(saved);bad['source_reports'][0]['sha256'] = '0' * 64;decision.write_bytes(payload(bad))
        rejects('reject_wrong_source_report_binding', lambda: p.create_package(attempt,
                    root / 'binding/artifacts/attention_tiling' / p.ARTIFACT, summaries, stage, license_path))
    print(json.dumps({'status': 'PASS_CPU_PACKAGER_CHECKS', 'checks': len(checks), 'tests': checks,
                      'scope': 'Temporary synthetic run/decision fixtures only; no real acceptance package created or retained',
                      'packager_sha256': p.digest((HERE / 'package_bf_evidence.py').read_bytes())}, indent=2))


if __name__ == '__main__':
    main()
