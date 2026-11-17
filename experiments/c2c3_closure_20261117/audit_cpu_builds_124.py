"""Read-only audit of recorded CPU builds; never starts containers or an NPU.

Only explicitly named host-memory/control test executables may run. SDK bytes
are read from existing image layers; unresolved/opaque paths fail closed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess


def load(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def record(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check_files(root, entries):
    if isinstance(entries, dict):
        entries = [dict(path=key, sha256=value) for key, value in entries.items()]
    root = Path(root).resolve(strict=True)
    mismatches, actual, seen = [], [], set()
    for entry in entries:
        rel = Path(entry['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:
            raise ValueError('UNSAFE_OR_DUPLICATE_MANIFEST_PATH:' + str(rel))
        seen.add(str(rel)); path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('MANIFEST_SYMLINK_ESCAPES_ROOT:' + str(rel))
        item = dict(path=str(rel), bytes=path.stat().st_size, sha256=digest(path))
        actual.append(item)
        if item['sha256'] != entry['sha256'] or ('bytes' in entry and item['bytes'] != entry['bytes']):
            mismatches.append(dict(expected=entry, actual=item))
    return dict(checked=len(actual), actual_inventory_sha256=canonical(actual), mismatches=mismatches)


def image_metadata(image_id):
    value = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image_id]))[0]
    if value['Id'] != image_id or value['Architecture'] != 'arm64' or value['Os'] != 'linux':
        raise ValueError('IMAGE_ID_OR_ARCHITECTURE_MISMATCH')
    return value


def image_file(image, absolute):
    """Inspect existing overlay2 layers; no mount/create/start/export operation."""
    driver = image['GraphDriver']
    if driver['Name'] != 'overlay2':
        raise ValueError('UNREVIEWED_IMAGE_STORAGE_DRIVER')
    rel = Path(absolute.lstrip('/'))
    if '..' in rel.parts:
        raise ValueError('UNSAFE_IMAGE_PATH')
    data = driver['Data']
    for layer in [data['UpperDir']] + data['LowerDir'].split(':'):
        root = Path(layer).resolve(strict=True); candidate = root / rel
        # A whiteout or opaque upper ancestor must not expose a lower SDK file.
        opaque = False
        for i in range(1, len(rel.parts) + 1):
            prefix = Path(*rel.parts[:i]); part = root / prefix
            if (part.parent / ('.wh.' + part.name)).exists():
                raise ValueError('IMAGE_WHITEOUT:' + absolute)
            if part.exists() and stat.S_ISCHR(part.stat().st_mode) and part.stat().st_rdev == 0:
                raise ValueError('IMAGE_WHITEOUT:' + absolute)
            if part.is_dir():
                for attr in ('trusted.overlay.opaque', 'user.overlay.opaque'):
                    try:
                        opaque |= os.getxattr(part, attr) == b'y'
                    except OSError:
                        pass
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError('IMAGE_SYMLINK_ESCAPES_LAYER:' + absolute)
            return record(resolved)
        if opaque:
            raise ValueError('IMAGE_OPAQUE_UNRESOLVED:' + absolute)
    raise FileNotFoundError(absolute)


def source_snapshot(root):
    manifest = root / 'source_manifest.json'; data = load(manifest)
    return dict(manifest=record(manifest), source_commit=data['base_commit'],
                files=check_files(root, data['files']))


def normalized_symbols(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) not in (2, 3):
            raise ValueError('UNREVIEWED_NM_LINE:' + line)
        name = parts[-1].split('@', 1)[0]
        if name in result:
            raise ValueError('AMBIGUOUS_EXPORTED_SYMBOL:' + name)
        result[name] = (parts[-2], parts[0] if len(parts) == 3 else None)
    return result


def audit26(root, rerun):
    folder, source = root / 'native26-group-v2-a1', root / 'source-a29'
    manifest, receipt = load(folder / 'build.json'), load(folder / 'build-receipt.json')
    errors = []
    files = check_files(folder, manifest['files']); sources = check_files(source, receipt['sources'])
    if files['mismatches'] or sources['mismatches']:
        errors.append('ARTIFACT_OR_SOURCE_MISMATCH')
    if digest(folder / 'build-receipt.json') != manifest['build_receipt_sha256']:
        errors.append('BUILD_RECEIPT_HASH_MISMATCH')
    if receipt['source_commit'] != load(source / 'source_manifest.json')['base_commit']:
        errors.append('SOURCE_COMMIT_MISMATCH')
    image = image_metadata(receipt['actual_image_id'])
    recorded_image = load(folder / 'image-inspect.json')
    if isinstance(recorded_image, list):
        if len(recorded_image) != 1:
            raise ValueError('EXACTLY_ONE_RECORDED_IMAGE_REQUIRED')
        recorded_image = recorded_image[0]
    if recorded_image['Id'] != image['Id'] or digest(folder / 'image-inspect.json') != receipt['image_inspect_sha256']:
        errors.append('RECORDED_IMAGE_IDENTITY_MISMATCH')
    sdk = []
    for line in (folder / 'sdk-header-sha256.txt').read_text().splitlines():
        expected, path = line.split(maxsplit=1); actual = image_file(image, path.strip())
        sdk.append(dict(image_path=path.strip(), expected=expected, actual=actual))
        if actual['sha256'] != expected:
            errors.append('SDK_HEADER_MISMATCH:' + path)
    version = image_file(image, '/usr/local/Ascend/cann-8.5.1/aarch64-linux/ascend_toolkit_install.info')
    if 'version=8.5.1' not in Path(version['path']).read_text().splitlines():
        errors.append('ACTUAL_SDK_VERSION_MISMATCH')
    command = receipt['command']
    if (receipt['actual_npu_calls'] != 0 or receipt['device_mounts'] or receipt['build_returncode'] != 0
            or 'ASCEND_VISIBLE_DEVICES=void' not in command or receipt['actual_image_id'] not in command
            or '--runtime' not in command or command[command.index('--runtime') + 1] != 'runc'):
        errors.append('DEVICE_FREE_PINNED_BUILD_CONTRACT_MISMATCH')
    legacy_absent = not any((folder / name).exists() for name in receipt['omitted_legacy_writers'])
    if not legacy_absent:
        errors.append('LEGACY_WRITER_PRESENT')
    library = folder / 'libdcat26_group_v2.so'
    symbols = subprocess.check_output(['nm', '-D', str(library)], text=True)
    expected_symbols = re.findall(r'extern "C"\s+(?:int|bool|void|const aclTensor\*)\s+(\w+)\(',
                                 (source / 'fault_injection/native_adapters/writer26_group_runtime.cpp').read_text())
    actual_symbols = {line.split()[-1] for line in symbols.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
    if set(expected_symbols) - actual_symbols or any(name in actual_symbols for name in ('dcat26q_install_v1', 'dcat26f_install_v1')):
        errors.append('EXPORTED_ABI_CONTRACT_MISMATCH')
    recorded_symbols = (folder / 'exported-symbols.txt').read_text()
    symbols_match = normalized_symbols(symbols) == normalized_symbols(recorded_symbols)
    if not symbols_match:
        errors.append('EXPORTED_SYMBOL_SEMANTICS_MISMATCH')
    dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
    header = subprocess.check_output(['readelf', '-h', str(library)], text=True)
    versions = subprocess.check_output(['readelf', '--version-info', str(library)], text=True)
    version_pattern = r'(?:GLIBCXX|GLIBC|CXXABI|GCC)_[0-9.]+'
    actual_versions = sorted(set(re.findall(version_pattern, versions)))
    recorded_versions = sorted(set(re.findall(version_pattern, recorded_symbols)))
    if actual_versions != recorded_versions:
        errors.append('ELF_SYMBOL_VERSION_REQUIREMENTS_MISMATCH')
    if 'AArch64' not in header or 'libdcat26_group_v2.so' not in dynamic:
        errors.append('ELF_ARCH_OR_SONAME_MISMATCH')
    contracts = dict(core=load(folder / 'cpu-contract.json'), sdk=load(folder / 'sdk-control.json'))
    if contracts['core'] != receipt['cpu_contract'] or contracts['sdk'] != receipt['sdk_control']:
        errors.append('CONTRACT_RECEIPT_MISMATCH')
    repeats = {}
    if rerun:
        env = dict(os.environ); env.pop('LD_PRELOAD', None); env.pop('LD_LIBRARY_PATH', None)
        env['ASCEND_VISIBLE_DEVICES'] = 'void'
        for name, filename in [('core', 'test_group_core'), ('sdk', 'test_group_control')]:
            result = subprocess.run([str(folder / filename)], env=env, capture_output=True, text=True, timeout=30)
            repeats[name] = dict(executable=record(folder / filename), returncode=result.returncode,
                                 stdout=result.stdout, stderr=result.stderr)
            verified = result.returncode == 0 and json.loads(result.stdout) == contracts[name]
            if not verified and name == 'sdk' and 'GLIBCXX_3.4.29' in result.stderr:
                # Host libstdc++ is older than the pinned SDK image. Invoke that
                # image's already-present CPU loader/libs, without a container.
                runtime = {n: image_file(image, '/usr/lib/aarch64-linux-gnu/' + n) for n in
                           ('ld-linux-aarch64.so.1', 'libstdc++.so.6', 'libgcc_s.so.1', 'libm.so.6', 'libc.so.6')}
                libraries = ':'.join(dict.fromkeys(str(Path(x['path']).parent) for x in runtime.values()))
                argv = [runtime['ld-linux-aarch64.so.1']['path'], '--library-path', libraries, str(folder / filename)]
                matched = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=30)
                repeats[name]['matched_image_cpu_runtime'] = dict(files=runtime, command=argv,
                    returncode=matched.returncode, stdout=matched.stdout, stderr=matched.stderr)
                verified = matched.returncode == 0 and json.loads(matched.stdout) == contracts[name]
            repeats[name]['verified'] = verified
            if not verified:
                errors.append('CPU_CONTRACT_REPEAT_FAILED:' + name)
    return dict(manifest=record(folder / 'build.json'), receipt=record(folder / 'build-receipt.json'),
        artifacts=files, sources=sources, image=dict(id=image['Id'], architecture=image['Architecture']),
        sdk_headers=sdk, sdk_version=version, abi_exported_functions=expected_symbols,
        symbol_semantics_match=symbols_match,
        symbol_text_bytes_equal=hashlib.sha256(symbols.encode()).hexdigest() == digest(folder / 'exported-symbols.txt'),
        elf_symbol_version_requirements=actual_versions,
        contract=contracts, contract_rerun=repeats, legacy_writers_absent=legacy_absent,
        checks_passed=not errors, errors=errors, actual_npu_calls=0, c2=False, c3=False)


def audit_fa(root):
    folder, source = root / 'fa28-fresh-build-a1', root / 'source-a28'
    evidence = folder / 'build/run/evidence'
    controller, completion = load(folder / 'controller.json'), load(evidence / 'completion.json')
    vendor = folder / 'build/run/stage/vendors/dcat_f28_fa_transformer'
    files = check_files(vendor, completion['artifacts']); imported = check_files(source, completion['frozen_imported_sources'])
    errors = []
    if files['mismatches'] or imported['mismatches']:
        errors.append('ARTIFACT_OR_IMPORTED_SOURCE_MISMATCH')
    image = image_metadata(controller['image_id']); sdk = []
    for entry in completion['sdk_interface_files']:
        actual = image_file(image, '/usr/local/Ascend/cann-9.1.0/' + entry['path'])
        sdk.append(dict(expected=entry, actual=actual))
        if actual['sha256'] != entry['sha256'] or actual['bytes'] != entry['bytes']:
            errors.append('SDK_INTERFACE_MISMATCH:' + entry['path'])
    policy = load(folder / 'patch/policy_source_manifest.json')
    patched = check_files(folder / 'official', [dict(path=x['path'], sha256=x['after_sha256'], bytes=x['bytes']) for x in policy['files']])
    if patched['mismatches'] or digest(folder / 'patch/policy_source_manifest.json') != completion['patch_receipt_sha256']:
        errors.append('OFFICIAL_SOURCE_PATCH_MISMATCH')
    official_commit = subprocess.check_output(['git', '--no-optional-locks', '-C', str(folder / 'official'), 'rev-parse', 'HEAD'], text=True).strip()
    if official_commit != completion['source_commit'] or official_commit != policy['source_commit']:
        errors.append('OFFICIAL_SOURCE_COMMIT_MISMATCH')
    if (controller['actual_npu_calls'] != 0 or controller['devices'] or controller['exit_code'] != 0
            or completion['npu_executed'] or completion['actual_launch_verified'] or completion['loading_allowed']
            or 'ASCEND_VISIBLE_DEVICES=void' not in controller['command']):
        errors.append('CPU_ONLY_ACCEPTANCE_BOUNDARY_CHANGED')
    # Every named output hash must resolve to actual new vendor or evidence bytes.
    named = {'public_pod_schema_sha256': evidence / 'fa_public_schema.json',
             'public_pod_decoder_sha256': evidence / 'fa_pod_decoder',
             'decoder_headers_sha256': evidence / 'fa_decoder_headers.json',
             'binary_summary_registration_sha256': evidence / 'binary_summary_registration.json',
             'build_tools_sha256': evidence / 'build_tools.json',
             'dependency_manifest_sha256': evidence / 'dependency_sources.json'}
    packages = list((folder / 'build/run/source/build_out').glob('*.run'))
    if len(packages) != 1:
        raise ValueError('EXACTLY_ONE_FRESH_FA_HOST_INSTALLER_REQUIRED')
    named['host_package_sha256'] = packages[0]
    named_checks = {key: dict(expected=completion[key], actual=record(path)) for key, path in named.items()}
    for key, row in named_checks.items():
        if row['expected'] != row['actual']['sha256']:
            errors.append('NAMED_ARTIFACT_MISMATCH:' + key)
    for key in ('kernel_binary_sha256', 'kernel_metadata_sha256'):
        matches = [entry for entry in completion['artifacts'] if entry['sha256'] == completion[key]]
        if not matches:
            errors.append('NAMED_VENDOR_ARTIFACT_MISSING:' + key)
        named_checks[key] = dict(expected=completion[key], matching_vendor_entries=matches)
    return dict(controller=record(folder / 'controller.json'), completion=record(evidence / 'completion.json'),
        driver_snapshot=source_snapshot(source), official_ops_commit=official_commit,
        patched_sources=patched, imported_sources=imported, vendor_artifacts=files,
        image=dict(id=image['Id'], architecture=image['Architecture']), sdk_interfaces=sdk,
        named_artifacts=named_checks, public_pod_cpu_roundtrip=completion['public_pod_cpu_roundtrip'],
        compiled_keys=completion['compiled_keys'], actual_compiled_core_type=completion['actual_compiled_core_type'],
        sdk_interface_manifest_sha256=completion['sdk_interface_manifest_sha256'],
        checks_passed=not errors, errors=errors, actual_npu_calls=0,
        actual_launch_verified=False, actual_serialized_pod_verified=False, loading_allowed=False, c2=False, c3=False)


def scope_noninheritance(root):
    # These receipts are independent, immutable actual runs, not current code's
    # declarations. Rehash the original bytes before comparing their scopes.
    expected = {
        '18.worker': ('natural18-worker-a1/natural-worker-rms-group/probe/result.json',
                      'd344b70b98103cc0e22450bebd659adc7d2c9c785c48cea11cfda8f0bd9d9dca'),
        '20.worker': ('providers20-group-a1/providers20-deepseek-c2c3/result.json',
                      'bb683594bd2aee0bad5e0d41d2c8ca1db415f4cca7817fdc721b8787decc8178'),
        '39.worker': ('providers39-a6/providers39-qwen-c2c3/result.json',
                      'aa77add1286acd563b9a080399820aa3d088c35422c87f1c24f9066c5f1f58d0')}
    receipts = {name: dict(expected=sha, actual=record(root / path)) for name, (path, sha) in expected.items()}
    old = load(root / expected['18.worker'][0])['c2']['scope']
    source = root / 'source-a29'
    paths = ['src/demoncat/runtime/vllm_ascend/worker_extension/providers/tensor_natural.py']
    paths += ['fault_injection/' + name + '.py' for name in ('rmsnorm_return', 'index22', 'tensor_observer',
              'tensor_callable', '_scoped_dispatch', 'module_faults', 'route8', 'moe_census', 'moe_restore', 'conv3d_natural')]
    paths += ['src/demoncat/runtime/vllm_ascend/worker_extension/' + name + '.py'
              for name in ('serving_tensor', 'serving_moe')]
    files = {Path(path).name: record(source / path) for path in paths}
    if len(files) != len(paths):
        raise ValueError('COLLIDING_PROVIDER_SOURCE_IDENTITY')
    new_identity = canonical({name: row['sha256'] for name, row in files.items()})
    old_coverage = record(root / 'source-a20/fault_injection/coverage_session.py')
    new_coverage = record(source / 'fault_injection/coverage_session.py')
    return dict(original_receipts=receipts, original_receipt_bytes_unchanged=all(
        row['expected'] == row['actual']['sha256'] for row in receipts.values()),
        old_worker18_scope=old, current_shared_tensor_source_sha256=new_identity,
        current_shared_tensor_files=files, old_scope_does_not_match_new_source=old['implementation_sha256'] != new_identity,
        coverage_contract=dict(old=old_coverage, current=new_coverage,
            unchanged=old_coverage['sha256'] == new_coverage['sha256']),
        report_only=True, qualification_count_modified=False, new_model_local17_qualified=False,
        new_moe8_qualified=False, read_only22_census_qualified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rerun-cpu-contracts', action='store_true')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    result = dict(schema='demoncat.independent_cpu_build_review.v1', auditor=record(__file__),
        source_a29=source_snapshot(args.root / 'source-a29'),
        native26=audit26(args.root, args.rerun_cpu_contracts), fa28=audit_fa(args.root),
        scope_noninheritance=scope_noninheritance(args.root),
        qualifications_granted=False, actual_npu_calls=0)
    result['checks_passed'] = (not result['source_a29']['files']['mismatches']
        and all(result[k]['checks_passed'] for k in ('native26', 'fa28'))
        and result['scope_noninheritance']['original_receipt_bytes_unchanged']
        and result['scope_noninheritance']['old_scope_does_not_match_new_source'])
    with (args.out / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k:dict(checks_passed=result[k]['checks_passed'], errors=result[k]['errors']) for k in ('native26', 'fa28')}))


if __name__ == '__main__':
    main()
