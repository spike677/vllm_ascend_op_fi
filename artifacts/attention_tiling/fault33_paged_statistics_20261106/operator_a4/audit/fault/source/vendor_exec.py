"""Validate one explicitly approved private paged-IFA vendor, then exec a fresh Python.

stdlib only: no framework, driver, shared library, shell setup script or NPU is
loaded here. The controller chooses this launcher only for private_vendor cases.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time


AUDIT_ROOT = Path('/audit')
STAGE_ROOT = AUDIT_ROOT / 'vendor_stage_a4'
PROBE = AUDIT_ROOT / 'replay.py'
EVIDENCE_ROOT = Path('/evidence')
SHA_RE = re.compile('[0-9a-f]{64}')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def child_of(path, root):
    return path != root and root in path.parents


def relative_path(text):
    require(isinstance(text, str) and text and '\\' not in text and '\x00' not in text,
            'Invalid manifest path')
    path = PurePosixPath(text)
    require(not path.is_absolute() and text == path.as_posix() and
            all(part not in ('.', '..', '') for part in path.parts), 'Noncanonical/traversing manifest path')
    return Path(*path.parts)


def validate_vendor(case, audit_root=AUDIT_ROOT, stage_root=STAGE_ROOT):
    """Return an independently hashed full file inventory, without loading it."""
    require(case.get('operator_variant') == 'private_vendor', 'Launcher accepts private_vendor only')
    require(case.get('mode') in ('off', 'inner1024', 'inner512'), 'Explicit paged IFA mode required')
    expected = case.get('approved_vendor_manifest_sha256', '')
    require(SHA_RE.fullmatch(expected) is not None, 'Invalid approved manifest hash')
    stage = stage_root.resolve(strict=True)
    require(child_of(stage, audit_root.resolve(strict=True)), 'Stage is outside the readonly audit mount')
    manifest_path = Path(case['approved_vendor_manifest_file'])
    require(manifest_path.is_absolute() and child_of(manifest_path.resolve(strict=True), stage),
            'Manifest is outside the exact approved stage')
    require(manifest_path.stat().st_size <= 8 * 1024 * 1024 and sha(manifest_path) == expected,
            'Manifest size/hash mismatch')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest.get('schema') == 'fault33.pa.private_vendor.v1', 'Unsupported vendor manifest schema')
    vendor_path = audit_root / relative_path(manifest['vendor_root'])
    vendor = vendor_path.resolve(strict=True)
    require(vendor_path == vendor and child_of(vendor, stage) and vendor.is_dir(),
            'Vendor root must be a direct directory within the exact approved stage')
    require(case.get('approved_custom_opp_path') == str(vendor), 'Case vendor root mismatch')
    require(manifest.get('source_commit') == 'c77c0829f1f0f0b8f882db1d9b051a758b605472',
            'Paged IFA source commit differs from the audited release')
    require(SHA_RE.fullmatch(manifest.get('policy_source_manifest_sha256', '')) is not None,
            'Missing policy source manifest fingerprint')
    artifacts = manifest.get('artifacts')
    require(isinstance(artifacts, list) and 0 < len(artifacts) <= 20000, 'Invalid bounded artifact inventory')
    checked, names, total = [], set(), 0
    for row in artifacts:
        require(isinstance(row, dict), 'Invalid artifact row')
        relative = relative_path(row['path'])
        name = relative.as_posix()
        require(name not in names, 'Duplicate artifact path')
        names.add(name)
        path = vendor / relative
        resolved = path.resolve(strict=True)
        require(child_of(resolved, vendor) and resolved.is_file(), 'Artifact escapes vendor or is not a file')
        require(type(row.get('bytes')) is int and 0 <= row['bytes'] <= 2 * 1024**3 and
                path.stat().st_size == row['bytes'], 'Artifact byte count mismatch')
        require(SHA_RE.fullmatch(row.get('sha256', '')) is not None and sha(path) == row['sha256'],
                'Artifact hash mismatch: ' + name)
        total += row['bytes']
        require(total <= 8 * 1024**3, 'Total vendor inventory exceeds bound')
        checked.append({'path': name, 'bytes': row['bytes'], 'sha256': row['sha256'],
                        'symlink': path.is_symlink(), 'resolved_path': str(resolved)})
    actual = set()
    for current, directories, files in os.walk(vendor, followlinks=False):
        directory = Path(current)
        require(not any((directory / name).is_symlink() for name in directories),
                'Directory symlinks are unsupported in the private vendor')
        for name in files:
            path = directory / name
            require(path.is_file() and child_of(path.resolve(strict=True), vendor),
                    'Unexpected vendor entry or escaping symlink')
            actual.add(path.relative_to(vendor).as_posix())
    require(actual == names, 'Unlisted or missing private vendor files, including possible extra libraries')
    op_api = vendor / 'op_api/lib'
    require(op_api.is_dir() and not op_api.is_symlink() and
            any(row['path'].startswith('op_api/lib/') and '.so' in Path(row['path']).name for row in checked),
            'Missing verified vendor op_api shared library')
    return {'vendor_root': str(vendor), 'op_api_library_directory': str(op_api),
            'manifest_file': str(manifest_path), 'manifest_sha256': expected,
            'source_commit': manifest['source_commit'],
            'policy_source_manifest_sha256': manifest['policy_source_manifest_sha256'],
            'artifacts': checked, 'total_artifact_bytes': total}


def loader_environment(inherited, verified):
    environment = dict(inherited)
    previous = inherited.get('LD_LIBRARY_PATH')
    components = [] if previous is None else previous.split(':')
    retained = [component for component in components if component]
    environment['ASCEND_CUSTOM_OPP_PATH'] = verified['vendor_root']
    environment['LD_LIBRARY_PATH'] = ':'.join([verified['op_api_library_directory'], *retained])
    return environment, {'inherited': previous, 'effective': environment['LD_LIBRARY_PATH'],
                         'removed_empty_count': len(components) - len(retained),
                         'nonempty_inherited_priority_preserved': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    
    args = parser.parse_args()
    require(child_of(args.case_file.resolve(strict=True), (AUDIT_ROOT / 'setup').resolve(strict=True)),
            'Case must be within the readonly frozen setup directory')
    require(args.case_file.stat().st_size <= 8 * 1024 * 1024, 'Case exceeds bounded size')
    case_sha = sha(args.case_file)
    case = json.loads(args.case_file.read_bytes())
    probe = AUDIT_ROOT / ('model_probe.py' if case.get('execution_kind') == 'model_paged_adapter' else 'replay.py')
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('ASCEND_CUSTOM_OPP_PATH'),
            'Launcher requires a clean inherited vendor/preload environment')
    require(not any(name.startswith(('FAULT28_FA_', 'FAULT28_CAUSAL_', 'FAULT33_PA_')) for name in os.environ), 'Inherited FA host scope forbidden')
    verified = validate_vendor(case, AUDIT_ROOT, STAGE_ROOT)
    require(sha(args.case_file) == case_sha, 'Case changed during vendor validation')
    require(probe.is_file() and probe.resolve(strict=True) == probe, 'Expected frozen probe source is missing')
    output = args.output.resolve()
    require(child_of(output, EVIDENCE_ROOT.resolve(strict=True)) and not output.exists(),
            'Output must be a fresh directory within this attempt evidence mount')
    require(output.parent.is_dir(), 'Output parent does not exist')
    receipt_path = output.with_name(output.name + '.vendor_launch.json')
    environment, library_path_receipt = loader_environment(os.environ, verified)
    command = [sys.executable, str(probe), '--case-file', str(args.case_file.resolve()), '--output', str(output)]
    receipt = {'schema': 'fault33.pa.vendor_launch.v1', 'status': 'VERIFIED_READY_TO_EXEC',
               'epoch': time.time(), 'pid': os.getpid(), 'case_id': case['case_id'], 'case_sha256': case_sha,
               'launcher_sha256': sha(Path(__file__)), 'probe_sha256': sha(probe),
               'verified_vendor': verified, 'command': command,
               'library_path_normalization': library_path_receipt,
               'child_environment': {name: environment.get(name) for name in
                   ('ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'ACLNN_CACHE_LIMIT', 'PYTHONPATH')},
               'scope': 'Fresh process loader environment, no global environment changes and no shell setup script. This is not an execution-success receipt.'}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    environment['FAULT33_VENDOR_LAUNCH_RECEIPT'] = str(receipt_path)
    environment['FAULT33_VENDOR_LAUNCH_RECEIPT_SHA256'] = sha(receipt_path)
    print(json.dumps({'status': 'VERIFIED_READY_TO_EXEC', 'case_id': case['case_id'],
                      'artifacts': len(verified['artifacts']), 'receipt': str(receipt_path)}), flush=True)
    os.execve(sys.executable, command, environment)


if __name__ == '__main__':
    main()
