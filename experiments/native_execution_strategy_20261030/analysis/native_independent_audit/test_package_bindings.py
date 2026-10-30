"""Synthetic CPU-only publication-gate tests, never experiment acceptance.

Uses a complete minimal 20-case/56-call metadata fixture plus four-call canary.
Each rejection must happen before the destination directory is created.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


HERE = Path(__file__).resolve().parent
PACKAGER = HERE.parent / 'package_statistics.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def fixture(root):
    attempts = ['discovery_fixture', 'query_fixture', 'key_fixture', 'joint_fixture']
    canary = 'canary_fixture'
    files = {'replay.py': b'# original formal source\n',
             'runtime/native_execution_strategy/runtime_catalog.json': b'{"state":"experimental"}\n'}
    for name, data in files.items():
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    generation = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    private_path, selected_result = None, None
    for aindex, attempt in enumerate(attempts + [canary]):
        folder = root / 'outputs' / attempt
        for name, data in files.items():
            p = folder / 'source' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        write(folder / 'source_manifest.json', generation)
        cases = []
        for index in range(8 if aindex == 0 else 1 if attempt == canary else 4):
            cid = f'{attempt}_case{index}'
            case = folder / cid
            count = 1 if aindex == 0 else 4
            config = {'task': 'statistics', 'level': 'mix',
                      'statistics': {'summary_mode': 'statistics', 'tensor_list': []}}
            write(case / 'config.json', config)
            native = []
            for step in range(count):
                for kind in ('dump', 'stack', 'construct'):
                    relative = f'observation/step{step}/rank0/{kind}.json'
                    p = case / relative; write(p, {'synthetic_cpu_fixture': True, 'kind': kind})
                    native.append({'path': relative, 'sha256': sha(p), 'bytes': p.stat().st_size})
            receipt = {'config_sha256': sha(case / 'config.json'), 'native_files': native}
            write(case / 'receipt.json', receipt)
            result = {'status': 'CAPTURE_COMPLETED', 'original_api_calls': count,
                      'steps': [{'step': s} for s in range(count)], 'receipt_sha256': sha(case / 'receipt.json')}
            if attempt == canary:
                result['adapter_resolution'] = {'requested_version': 'validated', 'validation_state': 'validated',
                                                'resolved_version': 'cann9.1-s1s2-v1'}
            write(case / 'result.json', result)
            (folder / (cid + '.log')).write_bytes(b'synthetic CPU log; no tensors\n')
            cases.append({'case_id': cid, 'rc': 0, 'result_sha256': sha(case / 'result.json')})
            if private_path is None:
                private_path = case / 'private_output' / 'target.bin'
                private_path.parent.mkdir(); private_path.write_bytes(b'private tensor sentinel\x00\x01')
                selected_result = case / 'result.json'
        write(folder / 'completion.json', {'status': 'CAPTURE_COMPLETED', 'owned_container_removed': True,
              'device_lock_released': True, 'cleanup_errors': [], 'cases': cases})
    write(root / 'inputs/manifest.json', {'synthetic_cpu_fixture': True})
    write(root / 'native_vendor_stage_a1/vendor_manifest.json', {'synthetic_cpu_fixture': True})
    statistics = root / 'reports/statistics.json'
    write(statistics, {'group': 'all', 'planned_comparisons': 88, 'completed_comparisons': 88})
    evidence = {}
    for path in root.rglob('*'):
        if path.is_file() and canary not in path.relative_to(root).parts:
            evidence[path.relative_to(root).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    validation = root / 'reports/validation.json'
    write(validation, {'status': 'PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1',
        'requested_attempts': attempts, 'case_count': 20, 'native_call_count': 56,
        'independent_msprobe': {'summary_sha256': sha(statistics)},
        'source_generation': generation, 'evidence_files': evidence})
    return {'root': root, 'destination': root / 'publication', 'attempts': attempts,
            'canary': canary, 'validation': validation, 'statistics': statistics, 'reports': [],
            'private_path': private_path, 'selected_result': selected_result}


def invoke(module, state):
    return module.build(**{k: state[k] for k in
        ('root', 'destination', 'attempts', 'canary', 'validation', 'statistics', 'reports')})


def main():
    global PACKAGER
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--packager', type=Path, default=PACKAGER)
    args = ap.parse_args()
    PACKAGER = args.packager.resolve(strict=True)
    spec = importlib.util.spec_from_file_location('package_under_review', PACKAGER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    checks = []
    with tempfile.TemporaryDirectory(prefix='package_gate_cpu_', dir=HERE) as tmp:
        work = Path(tmp).resolve()
        assert work.is_relative_to(HERE)

        def scenario(name, mutate, should_pass=False):
            state = fixture(work / name)
            mutate(state)
            try:
                value = invoke(module, state)
                if not should_pass:
                    raise AssertionError('Expected rejection but publication succeeded')
                files = [p for p in state['destination'].rglob('*') if p.is_file()]
                assert not any(p.suffix in module.FORBIDDEN or 'private_output' in p.parts for p in files)
                assert not any(b'private tensor sentinel' in p.read_bytes() for p in files if p.suffix != '.gz')
                checks.append({'name': name, 'passed': True, 'publication_created': True,
                               'private_payload_published': False, 'files': value['files']})
            except (ValueError, OSError) as error:
                if should_pass:
                    checks.append({'name': name, 'passed': False, 'error': str(error)})
                else:
                    checks.append({'name': name, 'passed': not state['destination'].exists(),
                                   'rejected': True, 'error': str(error),
                                   'destination_created': state['destination'].exists()})

        def canary_source_advances(s):
            payload = b'# later canary source\n'
            (s['root'] / 'replay.py').write_bytes(payload)
            canary = s['root'] / 'outputs' / s['canary']
            (canary / 'source/replay.py').write_bytes(payload)
            generation = json.loads((canary / 'source_manifest.json').read_bytes())
            generation['replay.py'] = hashlib.sha256(payload).hexdigest()
            write(canary / 'source_manifest.json', generation)
        scenario('valid_bound_formal_sources_with_later_canary_and_private_exclusion', canary_source_advances, True)
        scenario('wrong_requested_attempt_rejected_before_output',
                 lambda s: s['attempts'].__setitem__(0, 'other_discovery'))
        scenario('wrong_statistics_summary_sha_rejected_before_output',
                 lambda s: s['statistics'].write_bytes(s['statistics'].read_bytes() + b' '))

        def wrong_generation(s):
            p = s['root'] / 'outputs' / s['attempts'][1] / 'source_manifest.json'
            data = json.loads(p.read_bytes()); data['unexpected.py'] = '1' * 64; write(p, data)
            # Isolate generation identity from byte-hash mismatch: a self-
            # inconsistent PASS report cannot bypass its source-generation gate.
            report = json.loads(s['validation'].read_bytes())
            report['evidence_files'][p.relative_to(s['root']).as_posix()] = {'sha256': sha(p), 'bytes': p.stat().st_size}
            write(s['validation'], report)
        scenario('wrong_formal_source_generation_rejected_before_output', wrong_generation)
        scenario('changed_private_evidence_rejected_before_output',
                 lambda s: s['private_path'].write_bytes(b'changed private sentinel'))

        def forged_non_source_fallback(s):
            p = s['selected_result']; prior = p.read_bytes()
            backup = s['root'] / 'outputs' / s['attempts'][0] / 'source' / p.relative_to(s['root'])
            backup.parent.mkdir(parents=True, exist_ok=True); backup.write_bytes(prior)
            p.write_bytes(prior + b' ')
        scenario('non_source_evidence_cannot_use_unregistered_snapshot_backup', forged_non_source_fallback)
    report = {'schema': 'nes.statistics_package_binding_cpu_tests.v1',
              'status': 'PASS_CPU_PACKAGE_BINDINGS' if all(x['passed'] for x in checks) else 'FAIL_CPU_PACKAGE_BINDINGS',
              'test_source_sha256': sha(__file__), 'packager_source_sha256': sha(PACKAGER),
              'checks': checks, 'negative_cases': 5, 'positive_cases': 1,
              'fixture_only_not_experiment_acceptance': True, 'npu_calls': 0, 'containers_started': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'failed_checks': [x['name'] for x in checks if not x['passed']]}))
    raise SystemExit(0 if report['status'].startswith('PASS_') else 1)


if __name__ == '__main__':
    main()
