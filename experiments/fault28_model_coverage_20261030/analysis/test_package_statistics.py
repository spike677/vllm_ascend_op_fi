"""CPU path, integrity and archive subset regressions; no real capture execution."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_statistics as p


def rejected(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('Expected rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix='pkg_cpu_', dir=args.output.parent) as temporary:
        root = Path(temporary).resolve()
        assert root.is_relative_to(args.output.parent.resolve())
        source = root / 'source'; source.mkdir()
        analysis = root / 'analysis'; analysis.mkdir()
        payload = source / 'config.json'; payload.write_bytes(b'{"task":"statistics"}\n')
        def check(name, fn): fn(); checks.append({'name': name, 'passed': True})
        def forbidden():
            for name in ('cases/x/private_output/step0_logits.bin', 'cases/x/cache/a.json', 'cases/x/native_logs/a.json',
                         'source/a.so', 'source/a.so.1', 'source/a.safetensors', 'source/x.pt',
                         'attempt/main_b1.log', 'attempt/main_b1.log.gz', 'source/inputs/manifest.json', '../escaped.json'):
                rejected(lambda name=name: p.public_path(name))
            assert p.public_path('cases/x/native_calls/call_0000/native_profile/native_window.log')
        check('private_weights_binary_full_logs_and_input_manifest_rejected', forbidden)
        def source_rename():
            target = source / 'secret.bin'; target.write_bytes(b'not public')
            selection = p.Selection([source])
            rejected(lambda: selection.add(target, 'reports/disguised.json', 'fixture'))
            other = source / 'private_output'; other.mkdir()
            target = other / 'metadata.json'; target.write_bytes(b'{}')
            rejected(lambda: selection.add(target, 'reports/private_renamed.json', 'fixture'))
        check('renaming_source_cannot_bypass_private_suffix_or_directory_guard', source_rename)
        def failed_capture():
            acceptance = analysis / 'failed.json'
            acceptance.write_text(json.dumps({'schema': 'nes.model_coverage.execution_acceptance.v1',
                'status': 'FAIL_OR_UNKNOWN_MODEL_EXECUTION_ACCEPTANCE', 'formal_coverage_accepted': False}))
            suite = analysis / 'suite.json'; suite.write_text('{}')
            rejected(lambda: p.collect(source, analysis, acceptance, suite))
            assert not (root / 'bundle').exists()
        check('failed_or_partial_capture_cannot_create_public_output', failed_capture)
        def changed_source():
            s = p.Selection([source]); s.add(payload, 'cases/x/config.json', 'fixture')
            item = {'attempt': str(source), 'analysis_directory': str(analysis), 'files': s.files, 'remote_only': []}
            original = payload.read_bytes(); payload.write_bytes(b'changed')
            rejected(lambda: p.export(item, root / 'changed_bundle', root / 'changed.tar.gz'))
            assert not (root / 'changed_bundle').exists() and not (root / 'changed.tar.gz').exists()
            payload.write_bytes(original)
        check('changed_selected_bytes_rejected_before_any_output_created', changed_source)
        def changed_remote():
            s = p.Selection([source]); s.add(payload, 'cases/x/config.json', 'fixture')
            original = source / 'case.log'; original.write_bytes(b'private full log')
            s.remote(original, 'fixture remote only')
            item = {'attempt': str(source), 'analysis_directory': str(analysis), 'files': s.files, 'remote_only': s.remote_only}
            original.write_bytes(b'changed full log')
            rejected(lambda: p.export(item, root / 'remote_bundle', root / 'remote.tar.gz'))
            assert not (root / 'remote_bundle').exists()
        check('changed_remote_retained_log_rejected_without_copying_it', changed_remote)
        def source_output_overlap():
            s = p.Selection([source]); s.add(payload, 'cases/x/config.json', 'fixture')
            tools = root / 'tools'; tools.mkdir()
            item = {'attempt': str(source), 'analysis_directory': str(analysis), 'files': s.files,
                    'remote_only': [], 'protected_source_roots': [str(source), str(analysis), str(tools)]}
            for index, protected in enumerate((source, analysis, tools)):
                for kind in ('archive', 'bundle'):
                    out = root / f'overlap_bundle_{index}_{kind}'
                    archive = root / f'overlap_archive_{index}_{kind}.tar.gz'
                    if kind == 'archive': archive = protected / 'should_not_exist.tar.gz'
                    else: out = protected / 'should_not_exist_bundle'
                    rejected(lambda: p.export(item, out, archive))
                    assert not out.exists() and not archive.exists()
        check('bundle_and_archive_cannot_write_attempt_analysis_or_tool_source_roots', source_output_overlap)
        def archive_subset():
            s = p.Selection([source]); s.add(payload, 'cases/x/config.json', 'fixture')
            original = source / 'case.log'; original.write_bytes(b'full log remains excluded')
            s.remote(original, 'fixture remote only')
            item = {'attempt': str(source), 'analysis_directory': str(analysis), 'files': s.files,
                    'remote_only': s.remote_only, 'public_subset_can_rerun_full_log_acceptance': False}
            out, archive = root / 'fixture_bundle', root / 'fixture.tar.gz'
            result = p.export(item, out, archive)
            assert result['full_logs_and_private_logits_remain_remote'] is True
            assert (out / 'cases/x/config.json').read_bytes() == payload.read_bytes()
            assert not (out / 'case.log').exists()
            manifest = p.read(out / 'manifest.json')
            assert manifest['remote_only'][0]['included_in_public_bundle'] is False
            assert manifest['public_subset_can_rerun_full_log_acceptance'] is False
            assert set(manifest['files']) == {'cases/x/config.json', 'README.md', '.gitattributes'}
            rejected(lambda: p.export(item, out, root / 'other.tar.gz'))
        check('small_file_archive_roundtrip_excludes_full_log_and_discloses_limit', archive_subset)
    report = {'schema': 'nes.model_statistics_package.cpu.v1', 'status': 'PASS_TARGETED_CPU_CHECKS',
              'checks': checks, 'full560_call_selection_fixture_executed': False,
              'real_capture_packaged': False, 'ssh_connections': 0, 'npu_calls': 0,
              'sources': {'package_statistics.py': p.sha(Path(p.__file__)), 'test_package_statistics.py': p.sha(__file__)}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': p.sha(args.output)}))


if __name__ == '__main__':
    main()
