"""Allowlisted B/F statistics packaging and offline integrity verification.

No devices, framework imports, network, source modification, or scientific PASS.
The package contains exactly B/F native statistics; other checks are JSON summaries.
"""
import argparse
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


ATTEMPT = 'bf_statistics_high_a1'
ARTIFACT = 'fault28_native_fa_high_statistics_bf_20261029'
ACCEPTANCE = 'PASS_FAULT28_NATIVE_TILING_AND_HIGH_STATISTICS'
COLLECTOR_VARIANT = 'task_local_msprobe26_1_fp32_high'
MAX_FILE = 64 * 1024**2
MAX_TOTAL = 512 * 1024**2
SHA = re.compile('[0-9a-f]{64}')
ARMS = ('B', 'F')
SUMMARIES = {'off_compatibility.json', 'mechanism_validation.json', 'formal_runtime_crosscheck.json',
             'independent_msprobe_step0.json', 'independent_msprobe_step1.json',
             'official_default_msprobe_step0.json', 'official_default_msprobe_step1.json',
             'collector_validation.json', 'execution_acceptance.json'}
PROCESSOR = 'msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
ORIGINAL_PROCESSOR_SHA = 'e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011'
WHEEL_SHA = 'c004abe1123629ecdfb9938eee25cc8b554694d545c5123ae801c1846e4cfa68'
LICENSE_SHA = 'eb7a1d713eb919b146787629e22e4c975cb701f529a65d4d7e0fcd417558bf1c'
COLLECTOR_FILES = {'stage_manifest.json', 'source_manifest.json', 'collector.patch',
                   'original_pytorch_processor.py', 'patched_pytorch_processor.py'}
PARENT_REQUIRED = ('completion.json', 'source_manifest.json', 'selected_cases.json',
                   'B.controller.json', 'F.controller.json', 'B.vendor_launch.json', 'F.vendor_launch.json')
PARENT_OPTIONAL = ('before.json', 'resource_policy.json', 'container_identity.json')
GENERATED = {'package_summary.json', 'README.txt', 'tools/package_bf_evidence.py', '.gitattributes'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid relative package path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and
            all(part not in ('', '.', '..') for part in path.parts), 'Unsafe/noncanonical relative path')
    return path


def source_name(name):
    path = canonical(name)
    if len(path.parts) == 1:
        return re.fullmatch(r'[A-Za-z0-9_.-]+\.(py|sh)', name) is not None
    return path.parts[0] == 'setup' and path.suffix == '.json' and all(
        re.fullmatch('[A-Za-z0-9_.-]+', part) for part in path.parts[1:])


def allowed(name):
    path = canonical(name)
    if name in GENERATED or name in PARENT_REQUIRED or name in PARENT_OPTIONAL:
        return True
    if path.parts[0] == 'source':
        return source_name('/'.join(path.parts[1:]))
    if len(path.parts) == 2 and path.parts[0] == 'summaries':
        return path.parts[1] in SUMMARIES
    if len(path.parts) == 2 and path.parts[0] == 'collector':
        return path.parts[1] in COLLECTOR_FILES | {'LICENSE.MulanPSL-2.0'}
    if path.parts[0] not in ARMS:
        return False
    tail = '/'.join(path.parts[1:])
    return (tail in ('config.json', 'receipt.json', 'result.json', 'collector_provenance.json', 'fa_host_events.jsonl') or
            re.fullmatch(r'observation/step[01]/rank0/(dump|stack|construct)\.json', tail) is not None or
            re.fullmatch(r'native_calls/call_0(?:[0-4][0-9]|5[0-5])\.json', tail) is not None)


class Directory:
    def __init__(self, root):
        self.root = root.resolve(strict=True)

    def read(self, name):
        relative = canonical(name)
        path = self.root.joinpath(*relative.parts)
        require(path.is_file() and not path.is_symlink() and self.root in path.resolve(strict=True).parents,
                'Missing/escaping/nonregular evidence file: ' + name)
        require(all(not parent.is_symlink() for parent in path.parents if parent != self.root and self.root in parent.parents),
                'Symlinked evidence directory: ' + name)
        require(path.stat().st_size <= MAX_FILE, 'Oversized file: ' + name)
        payload = path.read_bytes()
        require(len(payload) <= MAX_FILE, 'File grew past bound: ' + name)
        return payload

    def files(self):
        names = set()
        for path in self.root.rglob('*'):
            require(not path.is_symlink(), 'Symlink in package tree')
            if path.is_file():
                names.add(path.relative_to(self.root).as_posix())
        return names


def read_json(reader, name):
    return json.loads(reader.read(name))


def no_private_payload(value):
    if isinstance(value, dict):
        require(not value.get('private_raw_file'), 'Raw activation reference in statistics evidence')
        if value.get('type') in ('torch.Tensor', 'numpy.ndarray'):
            require(set(value) <= {'type', 'dtype', 'shape', 'Max', 'Min', 'Mean', 'Norm', 'requires_grad'},
                    'Native tensor record contains an unapproved payload/reference field')
        for child in value.values():
            no_private_payload(child)
    elif isinstance(value, list):
        for child in value:
            no_private_payload(child)


def required_names(reader):
    manifest = read_json(reader, 'source_manifest.json')
    require(isinstance(manifest, dict) and manifest and
            all(source_name(name) and SHA.fullmatch(value) for name, value in manifest.items()),
            'Unexpected frozen source manifest')
    require({'controller.py', 'env.sh', 'hf_runtime_probe.py', 'vendor_exec.py', 'collector_stage.py'} <= set(manifest),
            'Missing required frozen runtime sources')
    names = set(PARENT_REQUIRED) | {'source/' + name for name in manifest}
    names.update('collector/' + name for name in COLLECTOR_FILES)
    names.add('collector/LICENSE.MulanPSL-2.0')
    names.update('summaries/' + name for name in SUMMARIES)
    for arm in ARMS:
        names.update(arm + '/' + name for name in ('config.json', 'receipt.json', 'result.json', 'collector_provenance.json', 'fa_host_events.jsonl'))
        names.update(f'{arm}/observation/step{step}/rank0/{kind}.json'
                     for step in (0, 1) for kind in ('dump', 'stack', 'construct'))
        names.update(f'{arm}/native_calls/call_{i:03d}.json' for i in range(56))
    require(all(allowed(name) for name in names), 'Internal allowlist inconsistency')
    return names


def validate_capture(reader):
    """Completeness/provenance checks only; never infer a numerical fault."""
    source = read_json(reader, 'source_manifest.json')
    for name, expected in source.items():
        require(digest(reader.read('source/' + name)) == expected, 'Frozen source hash differs: ' + name)
    completion = read_json(reader, 'completion.json')
    require(completion.get('status') == 'CAPTURE_COMPLETED' and completion.get('owned_container_removed') is True
            and not completion.get('cleanup_errors') and completion.get('device_lock_released') is True,
            'Attempt did not finish and clean up successfully')
    require(completion.get('capture') == 'none' and completion.get('probe') == 'hf_runtime_probe.py',
            'Unexpected formal probe/capture configuration')
    require(completion.get('external_container_mutations') is False, 'External container mutation reported')
    records = completion.get('cases', [])
    require([row.get('case_id') for row in records] == ['B', 'F'] and
            all(row.get('rc') == 0 and row.get('status') == 'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE'
                for row in records), 'Completion must contain only successful B then F')
    selected = read_json(reader, 'selected_cases.json')
    require([row.get('case_id') for row in selected] == ['B', 'F'], 'Selected cases contain a third arm or reordered B/F')
    for arm, mode in (('B', 'off'), ('F', 'on')):
        prefix = arm + '/'
        result = read_json(reader, prefix + 'result.json')
        require(result.get('case_id') == arm and result.get('status') == 'CAPTURE_COMPLETED' and
                result.get('capture_msprobe') is True and result.get('capture_full_inputs') is False and
                result.get('forward_count') == 2 and result.get('prefill_certified') is True and
                result.get('batch_size') == 8 and result.get('prompt_length') == 2048,
                'Arm is incomplete or not two-forward statistics-only: ' + arm)
        require(result.get('fa_mode_plan', {}).get('modes') == [mode, mode], 'Formal B/F mode is not fixed: ' + arm)
        case = next(row for row in selected if row['case_id'] == arm)
        require(case['case'].get('capture_msprobe') is True and case['case'].get('capture_full_inputs') is False and
                case['case'].get('forwards') == 2 and case['case'].get('fa_mode') == mode,
                'Frozen case collection/mode differs')
        require(source.get(case['file']) == result.get('source_case_sha256'), 'Case source hash differs')
        require(read_json(reader, 'source/' + case['file']) == case['case'], 'Selected case differs from frozen source')
        require(result.get('source_sha256', {}).get('hf_runtime_probe.py') == source['hf_runtime_probe.py'],
                'Probe source hash differs')
        record = next(row for row in records if row['case_id'] == arm)
        require(record.get('result_sha256') == digest(reader.read(prefix + 'result.json')), 'Controller result hash differs')
        per_case = read_json(reader, arm + '.controller.json')
        require(per_case.get('case_id') == arm and per_case.get('rc') == 0, 'Per-case process did not exit successfully')
        launch = read_json(reader, arm + '.vendor_launch.json')
        require(launch.get('schema') == 'fault28.fa.vendor_launch.v1' and launch.get('case_id') == arm and
                launch.get('case_sha256') == result['source_case_sha256'] and
                launch.get('probe_sha256') == source['hf_runtime_probe.py'] and
                launch.get('launcher_sha256') == source['vendor_exec.py'], 'Launch/source provenance differs')
        require(result.get('private_vendor', {}).get('launcher_receipt_sha256') ==
                digest(reader.read(arm + '.vendor_launch.json')), 'Launch receipt hash differs')
        config = read_json(reader, prefix + 'config.json')
        require(config.get('task') == 'statistics' and config.get('level') == 'mix' and config.get('step') == [0, 1]
                and config.get('async_dump') is False and config.get('statistics', {}).get('tensor_list') == []
                and config['statistics'].get('summary_mode') == 'statistics' and config.get('precision') == 'high',
                'Wrong task-local high statistics configuration')
        receipt = read_json(reader, prefix + 'receipt.json')
        require(receipt.get('status') == 'CAPTURE_COMPLETED' and receipt.get('rank') == 0 and
                [row.get('step') for row in receipt.get('steps', [])] == [0, 1], 'Native receipt is incomplete')
        require(receipt.get('config_sha256') == digest(reader.read(prefix + 'config.json')) and
                result.get('msprobe_receipt_sha256') == digest(reader.read(prefix + 'receipt.json')),
                'Native config/receipt hash differs')
        expected_native = {f'observation/step{step}/rank0/{kind}.json'
                           for step in (0, 1) for kind in ('dump', 'stack', 'construct')}
        native = receipt.get('native_files', [])
        require(len(native) == 6 and {row.get('path') for row in native} == expected_native,
                'Receipt must cover exactly two native steps, six JSON files')
        for row in native:
            payload = reader.read(prefix + row['path'])
            require(len(payload) == row['bytes'] and digest(payload) == row['sha256'], 'Native statistics hash differs')
            obj = json.loads(payload)
            require(isinstance(obj, dict) and obj, 'Empty native statistics object')
            if row['path'].endswith('/dump.json'):
                require(obj.get('task') == 'statistics' and obj.get('level') == 'mix' and
                        obj.get('dump_data_dir') is None and obj.get('data'), 'Native dump contains wrong mode/payload')
                no_private_payload(obj)
        calls = result.get('native_calls', [])
        require(len(calls) == 56, 'Expected 28 layers across two forwards')
        for index, call in enumerate(calls):
            expected = f'native_calls/call_{index:03d}.json'
            require(call.get('file') == expected and call.get('native_index') == index and
                    call.get('forward_index') == index // 28 and call.get('layer_index') == index % 28 and
                    call.get('status') == 'returned', 'Native call ledger is incomplete/reordered')
            payload = reader.read(prefix + expected)
            require(digest(payload) == call['sha256'], 'Native call JSON hash differs')
            details = json.loads(payload)
            no_private_payload(details)
            require(not details.get('selected_layer_private_inputs') and not details.get('selected_layer_private_output'),
                    'Private activation capture enabled in formal statistics arm')
            require(all(details.get(key) == call.get(key) for key in
                        ('call_id', 'native_index', 'forward_index', 'layer_index', 'status')), 'Native ledger/detail mismatch')
        events = [json.loads(line) for line in reader.read(prefix + 'fa_host_events.jsonl').decode('utf-8').splitlines() if line.strip()]
        require(len(events) == 2 and len({row.get('token') for row in events}) == 2 and
                [row.get('forward') for row in events] == ['0', '1'] and all(
                    row.get('mode') == mode and row.get('callback_status') == 0 and row.get('eligible') is True and
                    row.get('policy_applied') is (mode == 'on') and row.get('layer') == 1 and row.get('phase') == 'prefill'
                    for row in events), 'Formal host event coverage/control differs')
        no_private_payload(result)
    return {'arms': ['B', 'F'], 'forwards_per_arm': 2, 'native_jsons_per_arm': 6,
            'native_call_jsons_per_arm': 56, 'capture_complete_and_cleaned_up': True,
            'mechanism_accepted': False}


def validate_scientific_reports(reader):
    """Check the supplied SHA-bound decision; never manufacture its PASS."""
    acceptance = read_json(reader, 'summaries/execution_acceptance.json')
    expected = {'status': ACCEPTANCE, 'attempt': ATTEMPT,
                'native_msprobe_first_difference_pass': True, 'mechanism_validation_pass': True,
                'formal_runtime_consistency_pass': True, 'collector_variant': COLLECTOR_VARIANT,
                'official_default_msprobe_exact_location_pass': False, 'historical_defect_reproduced': False}
    require(all(acceptance.get(k) == v and type(acceptance.get(k)) is type(v) for k, v in expected.items()),
            'Final independent acceptance is absent, failed, or overclaims its scope')
    referenced = acceptance.get('source_reports', [])
    names = [row.get('path') for row in referenced]
    required = {'summaries/' + name for name in SUMMARIES - {'execution_acceptance.json'}}
    require(len(names) == len(set(names)) and required <= set(names), 'Acceptance must bind all source reports')
    for row in referenced:
        name = row['path']
        require(name in required or name in {'collector/' + x for x in COLLECTOR_FILES},
                'Acceptance references a nonallowlisted/self-referential report')
        require(isinstance(row.get('sha256'), str) and SHA.fullmatch(row['sha256']) and
                digest(reader.read(name)) == row['sha256'], 'Acceptance source report SHA differs: ' + name)
    require(read_json(reader, 'summaries/off_compatibility.json').get('status') == 'PASS_STOCK_OFF_COMPATIBILITY',
            'Stock/OFF compatibility did not pass')
    require(read_json(reader, 'summaries/mechanism_validation.json').get('status') == 'PASS_NATIVE_FA_EFFECT_DIAGNOSTICS',
            'Native mechanism/guard/recovery diagnostics did not pass')
    runtime_report = read_json(reader, 'summaries/formal_runtime_crosscheck.json')
    require(runtime_report.get('status') == 'PASS_RUNTIME_CROSSCHECK',
            'Formal runtime consistency did not pass')
    for arm in ARMS:
        suffix = '/' + ATTEMPT + '/' + arm + '/result.json'
        hashes = [value for path, value in runtime_report.get('input_sha256', {}).items() if path.endswith(suffix)]
        require(hashes == [digest(reader.read(arm + '/result.json'))], 'Runtime crosscheck refers to a different formal attempt')
    for step in (0, 1):
        report = read_json(reader, f'summaries/independent_msprobe_step{step}.json')
        first = report.get('first_observed_model_api_numerical_difference')
        require(report.get('status') == 'ALIGNED_OBSERVED_API_STATISTICAL_DIFFERENCE' and
                report.get('alignment', {}).get('complete_alignment') is True and isinstance(first, dict),
                'High statistics inference is incomplete')
        for arm in ARMS:
            require(first.get(arm, {}).get('name') == 'Functional.scaled_dot_product_attention.1.forward' and
                    first[arm].get('attention_kind') == 'PYTORCH_SDPA_API', 'High first difference is not the selected SDPA API')
            coverage = report.get('attention_boundary_coverage', {}).get(arm, {})
            require(coverage.get('complete_hf_sdpa_module_coverage') is True and
                    coverage.get('record_counts_by_boundary', {}).get('PYTORCH_SDPA_API') == 28,
                    'High native SDPA coverage is incomplete')
            hashes = report.get('input_sha256', {}).get(arm, {})
            for kind, name in [('config', 'config.json')] + [(k, f'observation/step{step}/rank0/{k}.json')
                                                            for k in ('dump', 'stack', 'construct')]:
                require(hashes.get(kind) == digest(reader.read(arm + '/' + name)), 'High inference is not bound to published native JSON')
        require(not first.get('input', {}).get('numerical_differences') and
                first.get('output', {}).get('numerical_differences'), 'High SDPA first is not an output-only observed difference')
        old = read_json(reader, f'summaries/official_default_msprobe_step{step}.json')
        old_first = old.get('first_observed_model_api_numerical_difference')
        require(isinstance(old_first, dict) and old_first.get('B', {}).get('name') == 'Tensor.pow.7.forward',
                'Original default-statistics localization failure was not retained')
        require(old.get('fault_mechanism_accepted') is False, 'Original default-statistics report was relabelled as accepted')
    return acceptance


def validate_collector(reader):
    require(digest(reader.read('collector/LICENSE.MulanPSL-2.0')) == LICENSE_SHA, 'MulanPSL-2.0 license text differs')
    stage = read_json(reader, 'collector/stage_manifest.json')
    original = read_json(reader, 'collector/source_manifest.json')
    require(stage.get('schema') == 'fault28.msprobe.collector_high.v1' and
            stage.get('variant') == 'task_local_fp32_high_v1' and stage.get('package_version') == '26.1.0' and
            stage.get('precision') == 'high' and stage.get('stage_root') == '/audit/collector_high_a1',
            'Unexpected task-local collector identity')
    require(original.get('schema') == 'fault28.msprobe.installed_source.v1', 'Missing installed-source provenance')
    require(stage.get('wheel_sha256_from_direct_url') == original.get('wheel_sha256_from_direct_url') == WHEEL_SHA and
            stage.get('wheel_archive_available') is False and original.get('wheel_archive_available') is False,
            'Wheel provenance changed or unavailable archive overclaimed')
    old = reader.read('collector/original_pytorch_processor.py')
    new = reader.read('collector/patched_pytorch_processor.py')
    require(digest(old) == stage.get('original_processor_sha256') == ORIGINAL_PROCESSOR_SHA,
            'Original official processor SHA differs')
    require(digest(new) == stage.get('patched_processor_sha256') and old != new, 'Patched processor SHA missing/differs')
    patch = ''.join(difflib.unified_diff(old.decode('utf8').splitlines(True), new.decode('utf8').splitlines(True),
                                       fromfile='original/' + PROCESSOR, tofile='task_local_high/' + PROCESSOR)).encode()
    require(reader.read('collector/collector.patch') == patch, 'Collector patch does not exactly describe published source bytes')
    source_rows, stage_rows = original.get('files', []), stage.get('artifacts', [])
    sources, artifacts = {}, {}
    for rows, output in ((source_rows, sources), (stage_rows, artifacts)):
        require(rows, 'Missing complete collector inventory')
        for row in rows:
            name = row.get('path')
            canonical(name)
            require(name not in output and type(row.get('bytes')) is int and 0 <= row['bytes'] <= MAX_FILE and
                    isinstance(row.get('sha256'), str) and SHA.fullmatch(row['sha256']), 'Invalid/duplicate collector inventory row')
            output[name] = row
    require(PROCESSOR in sources and sources[PROCESSOR]['sha256'] == ORIGINAL_PROCESSOR_SHA,
            'Original processor absent from RECORD-verified installed inventory')
    for name, row in sources.items():
        require(name.startswith('msprobe/') or name.startswith('mindstudio_probe-26.1.0.dist-info/'), 'Unexpected installed source path')
        require(row.get('record_verified') is True or
                (name == 'mindstudio_probe-26.1.0.dist-info/RECORD' and row.get('record_verified') is False),
                'Installed source not RECORD verified')
        staged = artifacts.get('packages/' + name, {})
        expected = stage['patched_processor_sha256'] if name == PROCESSOR else row['sha256']
        expected_bytes = len(new) if name == PROCESSOR else row['bytes']
        require(staged.get('sha256') == expected and staged.get('bytes') == expected_bytes,
                'Stage changes a file beyond the one reviewed processor: ' + name)
    expected_artifacts = {'packages/' + name for name in sources} | {
        'original_pytorch_processor.py', 'collector.patch', 'source_manifest.json'}
    require(set(artifacts) == expected_artifacts, 'Unexpected staged collector file')
    for public_name, stage_name in [('source_manifest.json', 'source_manifest.json'),
                                     ('collector.patch', 'collector.patch'),
                                     ('original_pytorch_processor.py', 'original_pytorch_processor.py'),
                                     ('patched_pytorch_processor.py', 'packages/' + PROCESSOR)]:
        payload = reader.read('collector/' + public_name)
        require(artifacts[stage_name]['sha256'] == digest(payload) and artifacts[stage_name]['bytes'] == len(payload),
                'Published collector file differs from frozen stage: ' + public_name)
    for arm in ARMS:
        result = read_json(reader, arm + '/result.json')
        receipt = read_json(reader, arm + '/receipt.json')
        actual = read_json(reader, arm + '/collector_provenance.json')
        require(result.get('statistics_collector') == receipt.get('statistics_collector') == actual,
                'Collector selection/import observations disagree: ' + arm)
        require(receipt.get('collector_provenance_sha256') == digest(reader.read(arm + '/collector_provenance.json')),
                'Collector provenance receipt SHA differs')
        root = stage['stage_root']
        expected = {'variant': 'task_local_fp32_high_v1', 'precision': 'high', 'modified_official_collector': True,
                    'manifest_file': root + '/stage_manifest.json',
                    'manifest_sha256': digest(reader.read('collector/stage_manifest.json')),
                    'package_root': root + '/packages', 'processor_file': root + '/packages/' + PROCESSOR,
                    'original_processor_sha256': ORIGINAL_PROCESSOR_SHA,
                    'patched_processor_sha256': digest(new),
                    'source_manifest_sha256': digest(reader.read('collector/source_manifest.json')),
                    'patch_sha256': digest(patch), 'artifact_count': len(artifacts),
                    'provenance': stage['provenance'], 'policy': stage['policy']}
        require(all(actual.get(k) == v and type(actual.get(k)) is type(v) for k, v in expected.items()),
                'Runtime collector was not the approved frozen task-local stage')
        loaded = actual.get('actual_loaded', {})
        require(loaded.get('module') == 'msprobe.core.dump.data_dump.data_processor.pytorch_processor' and
                loaded.get('path') == loaded.get('method_code_file') == expected['processor_file'] and
                loaded.get('sha256') == digest(new), 'Actual loaded collector method/module differs')
        case = next(row['case'] for row in read_json(reader, 'selected_cases.json') if row['case_id'] == arm)
        require(case.get('statistics_variant') == expected['variant'] and
                case.get('approved_collector_manifest_file') == expected['manifest_file'] and
                case.get('approved_collector_manifest_sha256') == expected['manifest_sha256'],
                'Frozen case did not approve the actual collector stage')
    return stage


def verify_reader(reader, actual_names):
    manifest = read_json(reader, 'package_manifest.json')
    require(manifest.get('schema') == 'fault28.fa.statistics_package.v1' and manifest.get('artifact') == ARTIFACT and
            manifest.get('attempt') == ATTEMPT and manifest.get('mechanism_accepted') is False,
            'Unexpected package identity/claim')
    rows = manifest.get('files', [])
    names = [row.get('path') for row in rows]
    require(len(names) == len(set(names)) and all(allowed(name) for name in names), 'Package contains nonallowlisted/duplicate files')
    require(actual_names == set(names) | {'package_manifest.json'}, 'Package has extra or missing files')
    require(required_names(reader) <= set(names) and GENERATED <= set(names), 'Required B/F evidence is missing')
    total = 0
    for row in rows:
        payload = reader.read(row['path'])
        require(len(payload) == row['bytes'] and digest(payload) == row['sha256'], 'Package hash/size mismatch: ' + row['path'])
        total += len(payload)
    require(total <= MAX_TOTAL and total == manifest.get('total_bytes'), 'Package total bytes mismatch/bound exceeded')
    require(manifest.get('packager_sha256') == digest(reader.read('tools/package_bf_evidence.py')),
            'Packager source fingerprint differs')
    validate_capture(reader)
    validate_collector(reader)
    acceptance = validate_scientific_reports(reader)
    return {'status': 'VERIFIED_PACKAGE_INTEGRITY', 'files': len(rows), 'bytes': total,
            'manifest_sha256': digest(reader.read('package_manifest.json')), 'mechanism_accepted': False,
            'independent_acceptance_status': acceptance['status']}


def verify_package(path):
    reader = Directory(path)
    return verify_reader(reader, reader.files())


def verify_archive(path):
    with tarfile.open(path, 'r:gz') as archive:
        members, names, total = {}, set(), 0
        for member in archive:
            parsed = canonical(member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and
                    parsed.parts[0] == ARTIFACT and len(parsed.parts) >= 2,
                    'Archive contains a link, special file or wrong root')
            name = '/'.join(parsed.parts[1:])
            require(name not in names and 0 <= member.size <= MAX_FILE, 'Duplicate/oversized archive member')
            require(name == 'package_manifest.json' or allowed(name), 'Nonallowlisted archive member')
            names.add(name)
            members[name] = member
            total += member.size
            require(len(names) <= 10000 and total <= MAX_TOTAL, 'Archive exceeds bounds')

        class Reader:
            def read(self, name):
                canonical(name)
                require(name in members, 'Missing archive member: ' + name)
                with archive.extractfile(members[name]) as stream:
                    payload = stream.read(MAX_FILE + 1)
                require(len(payload) <= MAX_FILE, 'Oversized archive payload')
                return payload

        return verify_reader(Reader(), names)


def create_package(attempt, output, summaries, collector_stage, collector_license, archive_path=None):
    require(attempt.name == ATTEMPT, 'Only the explicitly named formal attempt is supported')
    require(output.parts[-3:] == ('artifacts', 'attention_tiling', ARTIFACT), 'Unexpected artifact destination suffix')
    require(not output.exists(), 'Package output already exists')
    source = Directory(attempt)
    names = required_names(source)
    validation = validate_capture(source)
    names.update(name for name in PARENT_OPTIONAL if (source.root / name).is_file())
    extras = {}
    for entry in summaries:
        name, separator, raw_path = entry.partition('=')
        require(separator and name in SUMMARIES and 'summaries/' + name not in extras, 'Unsupported/duplicate summary name')
        path = Path(raw_path)
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 8 * 1024**2,
                'Invalid bounded JSON summary')
        payload = path.read_bytes()
        require(isinstance(json.loads(payload), dict), 'Summary must be a JSON object')
        no_private_payload(json.loads(payload))
        extras['summaries/' + name] = payload
    require(set(extras) == {'summaries/' + name for name in SUMMARIES}, 'Every independent high/default/provenance report is required')
    collector = Directory(collector_stage)
    for name in COLLECTOR_FILES:
        source_path = 'packages/' + PROCESSOR if name == 'patched_pytorch_processor.py' else name
        extras['collector/' + name] = collector.read(source_path)
    require(collector_license.is_file() and not collector_license.is_symlink() and collector_license.stat().st_size < 1024**2,
            'Invalid collector license source')
    extras['collector/LICENSE.MulanPSL-2.0'] = collector_license.read_bytes()

    class Overlay:
        def read(self, name):
            return extras[name] if name in extras else source.read(name)

    validate_collector(Overlay())
    acceptance = validate_scientific_reports(Overlay())
    extras['tools/package_bf_evidence.py'] = Path(__file__).read_bytes()
    extras['.gitattributes'] = b'* -text\n'
    extras['package_summary.json'] = (json.dumps({'status': 'EVIDENCE_PACKAGED_NOT_MECHANISM_ACCEPTANCE',
        'artifact': ARTIFACT, 'attempt': ATTEMPT, **validation,
        'independent_acceptance_status': acceptance['status'],
        'collector_variant': COLLECTOR_VARIANT,
        'official_default_msprobe_exact_location_pass': False,
        'historical_defect_reproduced': False,
        'additional_json_summaries': sorted(extras),
        'excluded': ['third-arm msprobe data', 'activation/tensor binaries', 'private_replay', 'cache',
                     'large stdout/native debug logs', 'compiled vendor libraries'],
        'tiling_metadata': 'JSON/JSONL metadata including tiling raw_hex is retained; it is not tensor payload.'}, indent=2) + '\n').encode()
    extras['README.txt'] = (
        'B/F native FA task-local HIGH statistics evidence. Scientific decision is the supplied independent acceptance report.\n'
        'This uses a modified task-local mindstudio-probe 26.1.0 collector, NOT an unmodified official collector.\n'
        'FP16/BF16 statistics temporaries use FP32; original tensor values/dtype/shape and model compute are preserved.\n'
        'Original default 26.1.0 statistics attempt bf_statistics_a2 did not locate the attention output exactly; its inference reports remain included.\n'
        'The historical defect has not been reproduced. Integrity validation is not an independent causal proof.\n'
        'Original wheel archive is unavailable; its SHA is provenance from installed direct_url metadata plus RECORD verification, not a fresh wheel hash.\n'
        'Collector manifests retain complete original/stage inventories; this subset contains the reviewed original/patched module and exact patch, not the full installed wheel.\n'
        'Collector copyright headers remain intact. collector/LICENSE.MulanPSL-2.0 is the SPDX v3.26.0 license-list-data original text, SHA eb7a1d713eb919b146787629e22e4c975cb701f529a65d4d7e0fcd417558bf1c.\n'
        'Offline verification: python tools/package_bf_evidence.py verify --package .\n'
        'Archive verification: python tools/package_bf_evidence.py verify --archive PATH.tar.gz\n'
        'Raw native dump/stack/construct remain unchanged. Observer diagnostics remain identifiable by stack.\n'
        'Only B and F contain msprobe data; guards/recovery may appear as bounded JSON summaries.\n').encode()
    output.mkdir(parents=True, exist_ok=False)
    rows, total = [], 0
    for name in sorted(names | set(extras)):
        require(allowed(name), 'Nonallowlisted export')
        payload = extras[name] if name in extras else source.read(name)
        total += len(payload)
        require(len(payload) <= MAX_FILE and total <= MAX_TOTAL, 'Package exceeds size bounds')
        target = output.joinpath(*canonical(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
        rows.append({'path': name, 'bytes': len(payload), 'sha256': digest(payload)})
    manifest = {'schema': 'fault28.fa.statistics_package.v1', 'artifact': ARTIFACT, 'attempt': ATTEMPT,
                'mechanism_accepted': False, 'total_bytes': total, 'files': rows,
                'packager_sha256': digest(extras['tools/package_bf_evidence.py'])}
    with (output / 'package_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    verified = verify_package(output)
    if archive_path is not None:
        require(archive_path.parent.resolve() == output.parent.resolve() and
                archive_path.name == ARTIFACT + '.tar.gz' and not archive_path.exists(), 'Unexpected/existing archive destination')
        with archive_path.open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                    for name in sorted(Directory(output).files()):
                        payload = Directory(output).read(name)
                        info = tarfile.TarInfo(ARTIFACT + '/' + name)
                        info.size, info.mode, info.mtime = len(payload), 0o644, 0
                        archive.addfile(info, io.BytesIO(payload))
        verify_archive(archive_path)
        h = hashlib.sha256()
        with archive_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(chunk)
        verified.update(archive=str(archive_path), archive_sha256=h.hexdigest(), archive_bytes=archive_path.stat().st_size)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('--attempt', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--summary', action='append', default=[], metavar='NAME=JSON_PATH')
    create.add_argument('--collector-stage', type=Path, required=True)
    create.add_argument('--collector-license', type=Path, required=True)
    create.add_argument('--archive', type=Path)
    verify = commands.add_parser('verify')
    target = verify.add_mutually_exclusive_group(required=True)
    target.add_argument('--package', type=Path)
    target.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.command == 'create':
        result = create_package(args.attempt, args.output, args.summary, args.collector_stage, args.collector_license, args.archive)
    else:
        result = verify_package(args.package) if args.package else verify_archive(args.archive)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
