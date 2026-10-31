"""Plan/export a strict public subset after formal full-model acceptance.

Default operation is --plan-only (explicit flag required to avoid accidental
exports); build requires --build. Never runs a framework, a device, or a capture.
Whole case logs, prompt/token input manifest, private logits and binaries remain
at their original location. A public subset cannot rerun full-log acceptance.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

MODES = ['calibration', 'native', 'controlled', 'controlled', 'off']
EXPECTED_TOTALS = {'framework_calls': 560, 'eligible_prefill_calls': 280,
    'unsupported_decode_calls': 280, 'actual_strategy_switches': 112, 'switched_layer_request_pairs': 504}
FORBIDDEN_PARTS = {'private_output', 'cache', '.cache', 'native_logs', '__pycache__', '.git'}
BINARY = re.compile(r'\.(?:bin|pt|pth|npy|npz|safetensors|so(?:\..*)?|o|a|pyc|pkl|pickle)$', re.I)
HEX = re.compile('[0-9a-f]{64}')
ATTEMPT_REQUIRED = ('completion.json', 'source_manifest.json', 'selected_cases.json', 'before.json',
    'before_disk.json', 'resource_policy.json', 'container_identity.json', 'launch_command.json',
    'resource_checks.jsonl', 'disk_checks.jsonl')
CASE_REQUIRED = ('result.json', 'receipt.json', 'config.json', 'collector_provenance.json',
    'model_config.json', 'module_inventory.json', 'module_events.jsonl', 'phase_manifest.json')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def pairs(rows):
    result = {}
    for key, value in rows:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def relative_name(name):
    need(isinstance(name, str) and name and '\\' not in name and '\x00' not in name, 'Invalid relative path')
    value = PurePosixPath(name)
    need(not value.is_absolute() and value.as_posix() == name and '..' not in value.parts
         and all(part not in ('', '.') for part in value.parts), 'Escaping/noncanonical relative path')
    return value


def child(root, name):
    relative = relative_name(name)
    path = Path(root)
    for part in relative.parts:
        path = path / part
        need(not path.is_symlink(), 'Symlinked evidence is unsupported: ' + str(path))
    need(path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)) and path.is_file(),
         'Evidence missing or escapes root: ' + name)
    return path


def public_path(name):
    path = relative_name(name)
    need(not any(part.lower() in FORBIDDEN_PARTS for part in path.parts)
         and not BINARY.search(path.name), 'Private/binary payload rejected: ' + name)
    need(path.as_posix() == '.gitattributes' or path.suffix.lower() in ('.json', '.jsonl', '.py', '.sh', '.md', '.txt', '.patch', '.log'),
         'Unreviewed public suffix: ' + name)
    need(path.suffix.lower() != '.log' or path.name == 'native_window.log', 'Whole/unreviewed log rejected')
    need(path.as_posix() != 'source/inputs/manifest.json', 'Full prompt/token manifest remains private')
    return path


class Selection:
    def __init__(self, roots):
        self.roots = [Path(root).resolve(strict=True) for root in roots]
        self.files = {}
        self.remote_only = []

    def add(self, source, target, category, expected_sha=None, expected_bytes=None):
        path = Path(source)
        need(not path.is_symlink() and path.is_file(), 'Missing/symlinked source: ' + str(path))
        resolved = path.resolve(strict=True)
        allowed = [root for root in self.roots if resolved.is_relative_to(root)]
        need(allowed, 'Source outside approved attempt/analysis roots')
        source_relative = resolved.relative_to(allowed[0]).as_posix()
        child(allowed[0], source_relative)
        public_path(source_relative)
        public_path(target)
        observed = {'source': str(resolved), 'sha256': sha(path), 'bytes': path.stat().st_size, 'category': category}
        need(expected_sha is None or observed['sha256'] == expected_sha, 'Source SHA differs: ' + str(path))
        need(expected_bytes is None or observed['bytes'] == expected_bytes, 'Source bytes differ: ' + str(path))
        if target in self.files:
            need(self.files[target] == observed, 'Conflicting public target: ' + target)
        else:
            self.files[target] = observed

    def remote(self, source, reason, expected_sha=None, expected_bytes=None):
        path = Path(source)
        need(path.is_file() and not path.is_symlink(), 'Missing remote-retained evidence: ' + str(path))
        row = {'path': str(path.resolve(strict=True)), 'sha256': sha(path), 'bytes': path.stat().st_size,
               'reason': reason, 'included_in_public_bundle': False}
        need(expected_sha is None or row['sha256'] == expected_sha, 'Remote-retained SHA differs')
        need(expected_bytes is None or row['bytes'] == expected_bytes, 'Remote-retained byte count differs')
        self.remote_only.append(row)


def collect(attempt, analysis_dir, acceptance_path, suite_path):
    attempt, analysis_dir = (Path(p).resolve(strict=True) for p in (attempt, analysis_dir))
    acceptance_path, suite_path = (Path(p).resolve(strict=True) for p in (acceptance_path, suite_path))
    need(acceptance_path.is_relative_to(analysis_dir) and suite_path.is_relative_to(analysis_dir), 'Reports/suite outside selected analysis root')
    accepted, suite = read(acceptance_path), read(suite_path)
    need(accepted.get('schema') == 'nes.model_coverage.execution_acceptance.v1'
         and accepted.get('status') == 'PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE'
         and accepted.get('formal_coverage_accepted') is True and accepted.get('totals') == EXPECTED_TOTALS,
         'Formal all-model560-call acceptance is required')
    need(accepted.get('suite_input_sha256') == sha(suite_path)
         and suite.get('schema') == 'nes.model_coverage.acceptance_inputs.v1'
         and len(suite.get('cases', [])) == 2, 'Wrong/incomplete accepted suite')
    completion = read(attempt / 'completion.json')
    need(completion.get('status') == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
         and completion.get('device_lock_released') is True and completion.get('cleanup_errors') == [], 'Attempt did not complete/clean up')
    completed = {row['case_id']: row for row in completion.get('cases', [])}
    need(len(completed) == len(completion.get('cases', [])) == 2, 'Exactly two formal completed cases required')
    approved = {row['case_id']: row for row in accepted['cases']}
    need(len(approved) == len(accepted['cases']) == 2 and set(approved) == set(completed)
         and sorted(row['batch_size'] for row in approved.values()) == [1, 8], 'Acceptance/attempt B1/B8 identity differs')
    frozen = read(attempt / 'source_manifest.json')
    here = Path(__file__).resolve().parent
    selected = Selection([attempt, analysis_dir, here])
    def report(path, expected=None):
        path = Path(path).resolve(strict=True)
        need(path.is_relative_to(analysis_dir), 'Analysis reference escapes selected analysis directory')
        selected.add(path, 'reports/' + path.relative_to(analysis_dir).as_posix(), 'independent_report', expected)
        return read(path)
    report(acceptance_path)
    report(suite_path, accepted['suite_input_sha256'])
    for name in ATTEMPT_REQUIRED:
        selected.add(child(attempt, name), 'attempt/' + name, 'execution_and_resource_evidence')
    # Source files are retained exactly. The input document is a remote-only data
    # dependency, not executable source; its original manifest SHA remains visible.
    for name, expected in frozen.items():
        path = child(attempt / 'source', name)
        if name == 'inputs/manifest.json':
            selected.remote(path, 'Full prompt/token input manifest is not public', expected)
        else:
            selected.add(path, 'source/' + name, 'executed_source_snapshot', expected)
    for name, expected in accepted['analysis_source_sha256'].items():
        need(PurePosixPath(name).name == name, 'Analysis source name must be a basename')
        candidates = [attempt / 'source' / name, analysis_dir / name, here / name]
        matches = [path for path in candidates if path.is_file() and sha(path) == expected]
        need(matches, 'Accepted analysis source not available with exact SHA: ' + name)
        selected.add(matches[0], 'analysis_sources/' + name, 'accepted_analysis_source', expected)
    selected.add(Path(__file__), 'tools/package_statistics.py', 'package_tool')
    total_native, total_steps, cases = 0, 0, []
    seen = set()
    for spec in suite['cases']:
        def resolve(value):
            path = Path(value)
            return (suite_path.parent / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
        folder = resolve(spec['case_dir'])
        need(folder.parent == attempt and folder.name in approved and folder.name not in seen, 'Suite references another/repeated attempt case')
        case_id = folder.name; seen.add(case_id)
        approval, done = approved[case_id], completed[case_id]
        need(approval['frozen_bindings']['source_files'] == frozen
             and approval['completion_sha256'] == sha(attempt / 'completion.json'), 'Accepted source/completion changed')
        need(done.get('rc') == 0 and done.get('status') == 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE', 'Case failed capture')
        framework = report(resolve(spec['framework_report']), approval['framework_report_sha256'])
        native = report(resolve(spec['native_census_report']), approval['native_census_sha256'])
        statistics_path = resolve(spec['statistics_report'])
        statistics = report(statistics_path, approval['independent_msprobe_inference']['report_sha256'])
        need(framework.get('schema') == 'nes.model_coverage.framework_denominator.v1'
             and framework.get('status') == 'FRAMEWORK_DENOMINATOR_VERIFIED'
             and framework.get('framework_sdpa_calls') == 280
             and framework['case_identity'] == statistics['case_identity']
             and framework['evidence_sha256'] == statistics['evidence_sha256'], 'Independent report input identities differ')
        metadata_keys = {'result.json': 'result', 'receipt.json': 'receipt', 'config.json': 'config',
            'collector_provenance.json': 'collector', 'module_inventory.json': 'inventory',
            'module_events.jsonl': 'module_events', 'phase_manifest.json': 'phases'}
        for filename, key in metadata_keys.items():
            need(sha(child(folder, filename)) == framework['evidence_sha256'][key], 'Accepted metadata changed: ' + filename)
        need(sha(folder / 'model_config.json') == framework['model_config_sha256'], 'Actual model config changed')
        need(native.get('status') == 'PASS_NATIVE_LAUNCH_CENSUS' and native.get('case_id') == case_id
             and native['summary']['original_sdpa_windows'] == 280, 'Native census matrix differs')
        need(statistics.get('status') == 'FORMAL_COMPARISONS_COMPLETE'
             and statistics.get('run_comparison_count') == 5 and statistics.get('phase_comparison_count') == 10,
             'Independent statistics comparison matrix incomplete')
        for row in statistics['phase_reports']:
            report(child(statistics_path.parent, row['file']), row['sha256'])
        log = attempt / (case_id + '.log')
        if 'log' in spec: need(resolve(spec['log']) == log, 'Suite original whole-log path differs')
        selected.remote(log, 'Full DEBUG case log retained remotely; full-log census/reparse cannot be rerun from public subset alone',
                        native['source_log']['sha256'], native['source_log']['bytes'])
        prefix = 'cases/' + case_id + '/'
        for name in CASE_REQUIRED:
            selected.add(child(folder, name), prefix + name, 'original_capture_metadata')
        for suffix in ('.controller.json', '.vendor_launch.json', '.before_disk.json'):
            selected.add(child(attempt, case_id + suffix), 'attempt/' + case_id + suffix, 'case_execution_evidence')
        result, receipt, config, phases = (read(folder / name) for name in
            ('result.json', 'receipt.json', 'config.json', 'phase_manifest.json'))
        need(result.get('status') == 'CAPTURE_COMPLETED' and sha(folder / 'result.json') == done['result_sha256']
             and result.get('original_api_calls') == len(result.get('calls', [])) == 280
             and result.get('receipt_sha256') == sha(folder / 'receipt.json')
             and result.get('phase_manifest_sha256') == sha(folder / 'phase_manifest.json'), 'Case result/receipt/phase seal differs')
        need(config.get('task') == 'statistics' and config.get('level') == 'mix' and config.get('step') == list(range(10))
             and config.get('statistics', {}).get('summary_mode') == 'statistics'
             and config['statistics'].get('tensor_list') == [], 'Not full five-run statistics/mix')
        need(receipt['config_sha256'] == sha(folder / 'config.json')
             and receipt['collector_provenance_sha256'] == sha(folder / 'collector_provenance.json'), 'Statistics receipt seals differ')
        expected_native = {f'observation/step{s}/rank0/{kind}.json' for s in range(10) for kind in ('dump', 'stack', 'construct')}
        need(len(receipt['native_files']) == 30 and {row['path'] for row in receipt['native_files']} == expected_native,
             'Missing/duplicate native statistics triples')
        for row in receipt['native_files']:
            pieces = PurePosixPath(row['path']).parts
            key = pieces[1] + '_' + PurePosixPath(pieces[-1]).stem
            need(row['sha256'] == framework['evidence_sha256'][key], 'Accepted original statistics file differs')
            selected.add(child(folder, row['path']), prefix + row['path'], 'original_msprobe_statistics', row['sha256'], row['bytes'])
        phase_rows = phases['phases']
        need([row['step'] for row in phase_rows] == list(range(10))
             and [row['mode'] for row in phase_rows] == [mode for mode in MODES for _ in range(2)], 'Wrong five-run phase matrix')
        for row in phase_rows:
            need(row['request_mapping_sha256'] == framework['evidence_sha256'][f"step{row['step']}_request_mapping"],
                 'Accepted request mapping changed')
            selected.add(child(folder, row['request_mapping_file']), prefix + row['request_mapping_file'],
                         'original_request_mapping', row['request_mapping_sha256'])
            selected.remote(child(folder, row['private_logits_file']), 'Full per-request logits remain private',
                            row['logits']['full_sha256'], row['logits']['bytes'])
        file_set = set()
        census_calls = {row['identity']['call_id']: row for row in native['calls']}
        need(len(census_calls) == 280, 'Duplicate native census call ID')
        for row in result['calls']:
            relative = row['file']; file_set.add(relative)
            need(re.fullmatch(r'native_calls/call_\d{4}/call\.json', relative), 'Unexpected native call path')
            path = child(folder, relative); call = read(path)
            selected.add(path, prefix + relative, 'original_sdpa_call', row['sha256'])
            need(call['status'] == 'CAPTURE_COMPLETED' and call['call_id'] == row['call_id']
                 and census_calls[row['call_id']]['metadata_source']['sha256'] == row['sha256'], 'Call/census seal differs')
            if call['stage'] == 'prefill':
                rel = call['native_record_file']
                need(rel == str(PurePosixPath(relative).parent / 'native_profile/verified_record.json'), 'Native record belongs to another call')
                selected.add(child(folder, rel), prefix + rel, 'original_public_native_record', call['native_record_sha256'])
                record = read(folder / rel)
                profile = PurePosixPath(rel).parent
                for name in ('native_window.log', 'host_event.json', 'host_events.original.jsonl', 'public_pod_schema.json'):
                    item = str(profile / name)
                    selected.add(child(folder, item), prefix + item, 'original_native_window_host_or_pod')
                host_file = str(PurePosixPath(relative).parent / 'host_events.jsonl')
                need(sha(folder / host_file) == sha(folder / str(profile / 'host_events.original.jsonl')),
                     'Live public host bytes differ from accepted original snapshot')
                selected.add(child(folder, host_file), prefix + host_file, 'original_public_host')
                expected_refs = {'native_window.log', 'host_events.original.jsonl', 'public_pod_schema.json'}
                need({Path(ref['path']).name for ref in record['evidence']} == expected_refs
                     and len(record['evidence']) == 3, 'Unreviewed native evidence additions')
                for ref in record['evidence']:
                    item = str(profile / Path(ref['path']).name)
                    need(sha(folder / item) == ref['sha256'], 'Native original evidence SHA differs')
                need(sha(folder / str(profile / 'native_window.log')) == record['native_launch']['window_sha256'], 'Native window seal differs')
                need(read(folder / str(profile / 'host_event.json')) == record['host_event'], 'Public host JSON differs')
            else:
                need(call['stage'] == 'decode' and not call.get('native_record_file'), 'Unreviewed call stage/native scope')
        actual_call_set = {path.relative_to(folder).as_posix() for path in (folder / 'native_calls').glob('call_*/call.json')}
        need(file_set == actual_call_set and len(file_set) == 280, 'Extra/missing original SDPA sidecars')
        cases.append({'case_id': case_id, 'batch_size': approval['batch_size'], 'modes': MODES,
                      'statistics_steps': 10, 'sdpa_calls': 280, 'prefill_native_records': 140})
        total_native += 280; total_steps += 10
    need(seen == set(completed) and total_native == 560 and total_steps == 20, 'Incomplete selected matrix')
    return {'schema': 'nes.model_statistics_public_selection.v1', 'status': 'REVIEWED_INPUTS_SELECTED_NOT_EXPORTED',
            'attempt': str(attempt), 'analysis_directory': str(analysis_dir), 'cases': cases,
            'protected_source_roots': [str(root) for root in selected.roots],
            'acceptance_report_sha256': sha(acceptance_path), 'suite_sha256': sha(suite_path),
            'files': selected.files, 'remote_only': selected.remote_only,
            'public_private_tensor_payloads': False, 'full_case_logs_included': False,
            'public_subset_can_rerun_full_log_acceptance': False}


def export(selection, output, archive):
    output, archive = Path(output).resolve(), Path(archive).resolve()
    need(not output.exists() and not archive.exists() and archive.name.endswith('.tar.gz'), 'Fresh directory/tar.gz required')
    need(not archive.is_relative_to(output), 'Archive must not be inside its source bundle')
    roots = [Path(selection['attempt']).resolve(), Path(selection['analysis_directory']).resolve(),
             *(Path(root).resolve() for root in selection.get('protected_source_roots', []))]
    need(not any(target.is_relative_to(root) or root.is_relative_to(target)
                 for target in (output, archive) for root in roots),
         'Bundle or archive overlaps original evidence/source')
    # Full preflight before creating either output; mutable source changes still
    # cause an error during copy and can never yield a declared complete bundle.
    for relative, row in selection['files'].items():
        public_path(relative)
        path = Path(row['source'])
        need(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Selected evidence changed before export')
    for row in selection['remote_only']:
        path = Path(row['path'])
        need(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Remote-only evidence changed before export')
    output.mkdir(parents=True, exist_ok=False)
    for relative, row in selection['files'].items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(row['source']).open('rb') as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)
        need(target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], 'Original bytes changed while copying')
    readme = (
        '# 全模型 FA 策略覆盖 statistics/mix 证据子集\n\n'
        '两组 B1/B8，各含 calibration、native、controlled、controlled、off 五轮，每轮 prefill+decode；共20统计step、560次SDPA。\n'
        '原始 dump/stack/construct/config/receipt、模块/阶段/请求映射、调用JSON及native_window、公有host/POD均逐字节保留。\n'
        '这是固定模型与已认证原生策略的覆盖证据，不声称复现历史故障。数值变化、逐请求logits变化和准确率损失必须区分。\n'
        '使用已披露的任务本地FP32 high统计collector，不是未修改的官方collector。\n\n'
        '本包保留collector来源与加载SHA；完整collector修正源码/依赖和运行环境仍需对应仓库版本及环境归档，并非本包自包含。\n\n'
        '**整段DEBUG case.log、完整私有logits以及原prompt/token输入清单保留远端，没有放入本包。**\n'
        'manifest列出这些远端文件的原路径、SHA与bytes。公开native_window不能替代整日志的漏调用检查；\n'
        '本包可以校验自身原始统计/窗口及文件完整性，但不能单独重跑全日志census、原始全前缀校验或私有logits逐字节比较。\n'
        '所附独立验收报告是在完整远端证据上产生的，不能把附带报告当成这些省略数据的本地重算。\n')
    (output / 'README.md').write_text(readme, encoding='utf-8')
    (output / '.gitattributes').write_bytes(b'* -text\n')
    inventory = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            relative = path.relative_to(output).as_posix(); public_path(relative)
            inventory[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest = {**{k: v for k, v in selection.items() if k != 'files'},
                'schema': 'nes.model_statistics_public_package.v1', 'status': 'PUBLIC_SUBSET_EXPORTED',
                'files': inventory, 'origins': selection['files'],
                'archive_contents_are_ordinary_files_only': True}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w') as tar:
                for path in sorted(output.rglob('*')):
                    if not path.is_file(): continue
                    need(not path.is_symlink(), 'Symlink appeared in public bundle')
                    info = tar.gettarinfo(str(path), arcname=output.name + '/' + path.relative_to(output).as_posix())
                    info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''; info.mode = 0o644
                    with path.open('rb') as stream: tar.addfile(info, stream)
    expected = {output.name + '/' + name: row for name, row in inventory.items()}
    expected[output.name + '/manifest.json'] = {'sha256': sha(output / 'manifest.json'),
        'bytes': (output / 'manifest.json').stat().st_size}
    seen = set()
    with tarfile.open(archive, mode='r|gz') as tar:
        for member in tar:
            need(member.isfile() and member.name in expected and member.name not in seen,
                 'Archive contains unexpected, duplicate or nonregular entries')
            seen.add(member.name)
            digest, count = hashlib.sha256(), 0
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block); count += len(block)
            need(count == member.size == expected[member.name]['bytes']
                 and digest.hexdigest() == expected[member.name]['sha256'], 'Archive changed selected source bytes')
    need(seen == set(expected), 'Archive omitted selected files')
    return {'status': 'PUBLIC_SUBSET_EXPORTED', 'files': len(inventory) + 1,
            'manifest_sha256': sha(output / 'manifest.json'), 'archive_sha256': sha(archive),
            'archive_bytes': archive.stat().st_size, 'full_logs_and_private_logits_remain_remote': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--plan-only', type=Path, metavar='NEW_SELECTION_JSON')
    operation.add_argument('--build', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--selection', type=Path, help='Previously reviewed plan-only JSON; required for --build')
    args = parser.parse_args()
    if args.plan_only:
        need(not args.plan_only.exists(), 'Selection plan must be a fresh file')
    else:
        need(args.output_dir is not None and args.archive is not None and args.selection is not None,
             'Build requires output-dir, archive and previously reviewed selection plan')
        need(not args.output_dir.exists() and not args.archive.exists(), 'Build outputs must be fresh')
    selected = collect(args.attempt, args.analysis_dir, args.acceptance, args.suite)
    if args.plan_only:
        args.plan_only.parent.mkdir(parents=True, exist_ok=True)
        with args.plan_only.open('x', encoding='utf-8') as stream:
            json.dump(selected, stream, indent=2, ensure_ascii=False); stream.write('\n')
        result = {'status': selected['status'], 'selected_files': len(selected['files']),
                  'remote_only_files': len(selected['remote_only']), 'selection_sha256': sha(args.plan_only)}
    else:
        need(read(args.selection) == selected, 'Current selected evidence differs from the reviewed selection plan')
        selected['reviewed_selection_sha256'] = sha(args.selection)
        result = export(selected, args.output_dir, args.archive)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
