"""Freeze reviewed CPU/build evidence for an explicit first NPU validation.

The manifest grants no native acceptance. Actual current-call identity,
lossless storage substitution and OFF gates remain mandatory at runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--build', type=Path, required=True)
args = parser.parse_args()
root = args.build.resolve()
build = json.loads((root / 'build_manifest.json').read_text())
if build['status'] != 'PASS_CPU_BUILD_ONLY' or build['actual_npu_calls'] != 0:
    raise RuntimeError('CPU build receipt required')
for name, digest in build['files'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Build artifact changed: ' + name)
cpu_names = ['test_matmul_requests_receipt.txt', 'test_matmul_auto_receipt.txt',
             'current_call_cpu_receipt.txt', 'pta_gate_cpu_receipt.txt',
             'pta_gate_elf_cpu_receipt.json', 'pta_resolver_elf_cpu_receipt.json']
for name in cpu_names:
    raw = (root / name).read_text()
    if name.endswith('.json'):
        report = json.loads(raw)
        if report.get('status') != 'PASS_CPU_ONLY' or report.get('actual_npu_calls') != 0:
            raise RuntimeError('CPU test not passed: ' + name)
    elif not raw.startswith(('PASS ', 'PASS_CPU_ONLY ')) or 'FAIL' in raw:
        raise RuntimeError('CPU test not passed: ' + name)

def descriptor(name):
    return dict(path='/native/' + name, sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())

manifest = dict(schema='demoncat.fault20.serving_review.v1',
    reviewed_contract='fault20.current_typed_formals.single_target_lossless_off.v1',
    validation_level='CPU_VERIFIED', implementation_version='20-serving-current-call-v1',
    build_profile_sha256=build['profile_sha256'],
    required_libraries=['libnnopbase.so', 'libascendcl.so', 'libdcat20_qmm.so',
        'libopapi.so', 'libopapi_nn.so', 'libopapi_real.so', 'libopapi_nn_real.so',
        'libopapi_math.so', 'libopapi_math_real.so'],
    cpu_evidence=[descriptor(name) for name in cpu_names],
    build_evidence=descriptor('build_manifest.json'))
with (root / 'serving_review.json').open('x') as stream:
    json.dump(manifest, stream, indent=2)
print(json.dumps(dict(status='CPU_REVIEW_FROZEN_NOT_NPU_ACCEPTANCE',
    profile=build['profile_sha256'], cpu_receipts=len(cpu_names))))
