"""Freeze a separate v3 deployment bundle from actual CPU/build evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--cpu-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    native, output = args.native.resolve(), args.out.resolve()
    if not native.is_dir() or output.exists() or native.is_symlink():
        raise RuntimeError('VERIFIED_NATIVE_BUILD_AND_FRESH_OUTPUT_REQUIRED')
    build = json.loads((native / 'build_manifest.json').read_text())
    profile = json.loads((native / 'version_profile.json').read_text())
    cpu = json.loads(args.cpu_receipt.read_text())
    if (build['status'] != 'PASS_CPU_BUILD_ONLY' or build['actual_npu_calls'] != 0
            or build['profile_sha256'] != profile['profile_sha256']
            or cpu['status'] != 'CPU_CONTRACTS_PASSED_NOT_NPU_QUALIFICATION'
            or cpu['actual_npu_calls'] != 0 or len(cpu['checks']) != 7
            or any(row['exit_code'] != 0 for row in cpu['checks'])
            or any(profile['sources'].get(name) != digest for name, digest in cpu['files'].items())):
        raise RuntimeError('EXACT_SOURCE_CPU_AND_NATIVE_BUILD_REVIEW_REQUIRED')
    for name, digest in build['files'].items():
        path = (native / name).resolve()
        if not path.is_relative_to(native) or path.is_symlink() or sha(path) != digest:
            raise RuntimeError('BUILD_ARTIFACT_CHANGED:' + name)
    cpu_names = ('test_matmul_requests_receipt.txt', 'test_matmul_auto_receipt.txt',
                 'current_call_cpu_receipt.txt', 'pta_gate_cpu_receipt.txt',
                 'pta_gate_elf_cpu_receipt.json', 'pta_resolver_elf_cpu_receipt.json')
    for name in cpu_names:
        raw = (native / name).read_text()
        if name.endswith('.json'):
            row = json.loads(raw)
            okay = row.get('status') == 'PASS_CPU_ONLY' and row.get('actual_npu_calls') == 0
        else:
            okay = raw.startswith(('PASS ', 'PASS_CPU_ONLY ')) and 'FAIL' not in raw
        if not okay:
            raise RuntimeError('CPU_BUILD_CHECK_FAILED:' + name)
    shutil.copytree(native, output)
    shutil.copyfile(args.cpu_receipt, output / 'group_cpu_contract.json')
    def descriptor(name):
        return dict(path='/native/' + name, sha256=sha(output / name))
    review = dict(schema='demoncat.fault20.group_review.v3', validation_level='CPU_VERIFIED',
        reviewed_contract='fault20.exact_target_set.native_group_v3', implementation_version='native-group-v3',
        build_profile_sha256=build['profile_sha256'],
        required_libraries=['libnnopbase.so', 'libascendcl.so', 'libdcat20_qmm.so', 'libopapi.so',
            'libopapi_nn.so', 'libopapi_real.so', 'libopapi_nn_real.so', 'libopapi_math.so', 'libopapi_math_real.so'],
        cpu_evidence=[descriptor(name) for name in (*cpu_names, 'group_cpu_contract.json')],
        build_evidence=descriptor('build_manifest.json'), profile_evidence=descriptor('version_profile.json'))
    with (output / 'group_review.json').open('x') as stream:
        json.dump(review, stream, indent=2)
    print(json.dumps(dict(status='CPU_REVIEW_ONLY_NATIVE_C2_C3_PENDING',
        bundle=str(output), profile_sha256=build['profile_sha256'], native_sha256=sha(output / 'libdcat20_qmm.so'))))


if __name__ == '__main__':
    main()
