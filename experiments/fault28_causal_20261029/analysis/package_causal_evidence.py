"""Copy original statistics and metadata; compress native logs; exclude tensor payloads."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


FORBIDDEN = {'.bin', '.pt', '.pth', '.npy', '.npz', '.safetensors', '.so', '.o', '.a'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def child(root, relative):
    result = (root / relative).resolve(strict=True)
    require(result.is_relative_to(root.resolve()), 'Escaping source path')
    return result


def build(root, output, attempts, collector_stage):
    root = root.resolve(strict=True)
    output = output.resolve()
    require(not output.exists() and len(set(attempts)) == len(attempts) == 6, 'Need a fresh package and six exact attempts')
    collector_stage = collector_stage.resolve(strict=True)
    traversed = [root / 'outputs' / name for name in attempts] + [root / name for name in
                 ('analysis', 'revisions', 'setup', 'inputs', 'outputs/causal_reports', 'outputs/independent_statistics')] + [collector_stage]
    require(all(not output.is_relative_to(p.resolve()) and not p.resolve().is_relative_to(output) for p in traversed),
            'Package output overlaps a source tree')
    collector_manifest_sha = '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af'
    require(sha(collector_stage / 'stage_manifest.json') == collector_manifest_sha, 'Unapproved collector stage')
    for filename in ('stage_manifest.json', 'source_manifest.json', 'collector.patch'):
        require((collector_stage / filename).is_file() and not (collector_stage / filename).is_symlink(), 'Collector provenance missing')
    collector_manifest = read(collector_stage / 'stage_manifest.json')
    processor = collector_stage / 'packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
    require(sha(processor) == collector_manifest['patched_processor_sha256']
            == '3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141', 'Patched collector source differs')
    expected = {}
    for phase, count in (('stock_bridge', 4), ('primary', 16), ('automatic', 12)):
        matrix = read(root / 'setup' / phase / 'manifest.json')
        require(len(matrix['cases']) == count, 'Incomplete frozen matrix')
        for row in matrix['cases']:
            require(row['case_id'] not in expected, 'Duplicate frozen case')
            config = child(root / 'setup' / phase, row['file'])
            require(sha(config) == row['sha256'], 'Frozen matrix config changed')
            expected[row['case_id']] = row
    plans, seen, common_sources, source_inventory = [], set(), None, None
    for name in attempts:
        attempt = child(root / 'outputs', name)
        completion = read(attempt / 'completion.json')
        require(completion['status'] == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
                and completion.get('device_lock_released') is True and completion.get('cleanup_errors') == [], 'Attempt incomplete or cleanup failed: ' + name)
        source_sha = sha(attempt / 'source_manifest.json')
        require(common_sources is None or source_sha == common_sources, 'Capture versions differ between attempts')
        common_sources = source_sha
        frozen = read(attempt / 'source_manifest.json')
        snapshot = attempt / 'source'
        require(not any(p.is_symlink() for p in snapshot.rglob('*'))
                and {p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()} == set(frozen), 'Executed source inventory differs')
        for filename, expected_sha in frozen.items():
            require(sha(child(snapshot, filename)) == expected_sha, 'Executed source snapshot changed: ' + filename)
        source_inventory = frozen
        require(0 < len(completion['cases']) <= 8, 'Invalid bounded attempt')
        for row in completion['cases']:
            case_id = row['case_id']
            require(case_id in expected and case_id not in seen and row['rc'] == 0, 'Missing, duplicated or failed case')
            seen.add(case_id)
            folder = child(attempt, case_id)
            result = read(folder / 'result.json')
            require(result['status'] == 'CAPTURE_COMPLETED' and result['case_id'] == case_id
                    and sha(folder / 'result.json') == row['result_sha256']
                    and result['source_case_sha256'] == expected[case_id]['sha256']
                    and result['replay_source_sha256'] == frozen['replay.py'], 'Case completion or frozen-config binding failed')
            receipt = read(folder / 'receipt.json')
            require(sha(folder / 'receipt.json') == result['receipt_sha256'], 'Receipt binding failed')
            config = read(folder / 'config.json')
            provenance = read(folder / 'collector_provenance.json')
            require(sha(folder / 'config.json') == receipt['config_sha256']
                    and sha(folder / 'collector_provenance.json') == receipt['collector_provenance_sha256']
                    and receipt['statistics_collector'] == provenance
                    and provenance['manifest_sha256'] == collector_manifest_sha
                    and provenance['patched_processor_sha256'] == provenance['actual_loaded']['sha256'] == collector_manifest['patched_processor_sha256']
                    and provenance['original_processor_sha256'] == collector_manifest['original_processor_sha256']
                    and provenance['actual_loaded']['path'] == provenance['actual_loaded']['method_code_file'], 'Configuration/collector receipt mismatch')
            require(config['task'] == 'statistics' and config['level'] == 'mix'
                    and config['statistics']['summary_mode'] == 'statistics' and config['statistics']['tensor_list'] == [], 'Not statistics/mix')
            require(not any(p.suffix.lower() in FORBIDDEN or p.is_symlink() for p in (folder / 'observation').rglob('*')), 'Tensor payload or symlink in statistics tree')
            require(len(receipt['native_files']) == 3 and {x['path'] for x in receipt['native_files']} ==
                    {'observation/step0/rank0/' + name + '.json' for name in ('dump', 'stack', 'construct')}, 'Receipt must bind exactly the native statistics triple')
            for native in receipt['native_files']:
                path = child(folder, native['path'])
                require(sha(path) == native['sha256'] and path.stat().st_size == native['bytes'], 'Native file differs from receipt')
            plans.append((name, case_id, receipt))
    require(seen == set(expected), 'Six successful attempts do not cover all32 frozen cases')
    acceptance = read(root / 'outputs/causal_reports/execution_acceptance.json')
    require(acceptance.get('all_32_native_captures_completed') is True and acceptance.get('release_approved') is True, 'Separate acceptance has not approved release')
    output.mkdir(parents=True, exist_ok=False)
    origins, logs = {}, []

    def copy(source, relative):
        require(source.is_file() and not source.is_symlink() and source.suffix.lower() not in FORBIDDEN, 'Excluded source: ' + str(source))
        target = output / relative
        require(not target.exists() and target.resolve().is_relative_to(output.resolve()), 'Duplicate or escaping package path')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(source) == sha(target), 'Copy changed bytes')
        origins[target.relative_to(output).as_posix()] = str(source)

    for name in attempts:
        attempt = root / 'outputs' / name
        for filename in ('completion.json', 'source_manifest.json', 'selected_cases.json', 'container_identity.json', 'launch_command.json', 'resource_policy.json', 'before.json'):
            if (attempt / filename).is_file():
                copy(attempt / filename, Path('attempts') / name / filename)
    for name, case_id, receipt in plans:
        attempt = root / 'outputs' / name
        folder = attempt / case_id
        prefix = Path('attempts') / name / case_id
        for filename in ('result.json', 'config.json', 'receipt.json', 'collector_provenance.json', 'host_events.jsonl'):
            if (folder / filename).is_file():
                copy(folder / filename, prefix / filename)
        for row in receipt['native_files']:
            copy(child(folder, row['path']), prefix / row['path'])
        for suffix in ('.controller.json', '.vendor_launch.json'):
            if (attempt / (case_id + suffix)).is_file():
                copy(attempt / (case_id + suffix), Path('attempts') / name / (case_id + suffix))
        original = attempt / (case_id + '.log')
        target = output / 'attempts' / name / (case_id + '.log.gz')
        with original.open('rb') as src, target.open('xb') as dst:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
                shutil.copyfileobj(src, compressed, 1024 * 1024)
        logs.append({'file': target.relative_to(output).as_posix(), 'original_source': str(original),
                     'original_sha256': sha(original), 'original_bytes': original.stat().st_size})
    first_source = root / 'outputs' / attempts[0] / 'source'
    for filename in sorted(source_inventory):
        copy(child(first_source, filename), Path('executed_sources') / filename)
    for directory in ('outputs/causal_reports', 'outputs/independent_statistics'):
        for path in sorted((root / directory).rglob('*.json')):
            copy(path, Path('reports') / Path(directory).name / path.relative_to(root / directory))
    failure = root / 'outputs/stock_bridge_a2'
    require(read(failure / 'completion.json')['status'] == 'FAIL', 'Failure diagnostic unexpectedly changed state')
    diagnostic_names = ['completion.json', 'source_manifest.json', 'selected_cases.json',
                        'stock_bridge_b1_r0/result.json', 'stock_bridge_b1_r0/config.json', 'stock_bridge_b1_r0/collector_provenance.json']
    diagnostic_names += ['stock_bridge_b1_r0/observation/step0/rank0/' + kind + '.json' for kind in ('dump', 'stack', 'construct')]
    for filename in diagnostic_names:
        copy(child(failure, filename), Path('diagnostics/stock_bridge_a2') / filename)
    copy(root / 'outputs/stock_bridge_a1/completion.json', Path('diagnostics/stock_bridge_a1/completion.json'))
    for path in sorted((root / 'analysis').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.md', '.json') and '__pycache__' not in path.parts:
            copy(path, Path('analysis_sources') / path.relative_to(root / 'analysis'))
    for path in sorted((root / 'revisions').rglob('*')):
        if path.is_file() and (path.suffix in ('.py', '.json', '.md', '.patch') or path.name.startswith('LICENSE')) and '__pycache__' not in path.parts:
            copy(path, Path('revisions') / path.relative_to(root / 'revisions'))
    for filename in ('stage_manifest.json', 'source_manifest.json', 'collector.patch'):
        copy(collector_stage / filename, Path('collector') / filename)
    copy(processor, Path('collector/pytorch_processor.patched.py'))
    copy(root / 'inputs/manifest.json', Path('inputs/manifest.json'))
    copy(root / 'inputs/source_call.json', Path('inputs/source_call.json'))
    (output / '.gitattributes').write_text('* -text\n', encoding='utf-8')
    (output / 'README.txt').write_text('32 original statistics/mix operator captures; six successful attempts. Native dump/stack/construct/config/receipt bytes are preserved. Full tensor binaries, weights and compiled libraries are excluded. Native logs are gzip-compressed without content changes; the manifest binds decompressed hashes. The independent statistical tools can run from metadata alone. Complete per-element factorial re-analysis requires private outputs on124 and is not reproducible from summaries alone. Sources and failure/revision reports are included. See reports/causal_reports/execution_acceptance.json for the scientific acceptance and its limits; this packager does not grant mechanism acceptance.\n', encoding='utf-8')
    files = {p.relative_to(output).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size,
             'origin': origins.get(p.relative_to(output).as_posix(), 'Derived package metadata or lossless log compression')}
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'schema': 'fault28.causal.statistics_package.v1', 'status': 'PACKAGED_NOT_A_SCIENTIFIC_ACCEPTANCE',
                'mechanism_accepted': False, 'original_capture_files_modified': False,
                'case_count': 32, 'attempts': attempts, 'files': files, 'compressed_logs': logs}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return verify(output)


def verify(root):
    manifest = read(root / 'manifest.json')
    require(manifest['schema'] == 'fault28.causal.statistics_package.v1' and manifest['case_count'] == 32
            and manifest['mechanism_accepted'] is False, 'Unexpected package contract')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'manifest.json'}, 'Package inventory differs')
    require(not any(p.is_symlink() or p.suffix.lower() in FORBIDDEN for p in root.rglob('*')), 'Forbidden payload or symlink')
    for name, row in manifest['files'].items():
        p = child(root, name)
        require(p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'Package file changed: ' + name)
    for row in manifest['compressed_logs']:
        h, count = hashlib.sha256(), 0
        with gzip.open(child(root, row['file']), 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block); count += len(block)
        require(h.hexdigest() == row['original_sha256'] and count == row['original_bytes'], 'Native log changed during compression')
    return {'status': 'PASS_PACKAGE_INTEGRITY_ONLY', 'case_count': manifest['case_count'], 'files': len(manifest['files']),
            'bytes': sum(x['bytes'] for x in manifest['files'].values()), 'manifest_sha256': sha(root / 'manifest.json'), 'mechanism_accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--attempt', action='append')
    parser.add_argument('--collector-stage', type=Path, help='Actual host stage mounted read-only at /audit/collector_high_a1')
    args = parser.parse_args()
    if args.mode == 'build':
        require(args.output is not None and args.attempt and args.collector_stage is not None, 'Build requires output, six attempts and explicit host collector stage')
        result = build(args.root, args.output, args.attempt, args.collector_stage)
    else:
        result = verify(args.root)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
