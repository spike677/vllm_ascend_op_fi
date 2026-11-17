"""Freeze/verify fresh writer26 trace dependencies; no device or library load."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'demoncat.writer26.metadata_ownership.v1'
RECEIPT = 'metadata-array-ownership.json'
CPU_RECEIPT = 'metadata-array-cpu.json'
SYMBOLS = {'libfault26_readonly_trace_abi0.so': 'dcat26_trace_metadata_arrays_v1',
           'libopapi.so': 'dcat20_metadata_arrays_v1', 'libopapi_nn.so': 'dcat20_metadata_arrays_v1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(value, reason):
    if not value:
        raise RuntimeError('WRITER26_METADATA_' + reason)


def source_contract(source):
    from fault39.native.test_metadata_ownership import inputs
    unit, sources = inputs(source, writer26=True)
    sources['fault39/native/test_metadata_ownership.py'] = sha(source / 'fault39/native/test_metadata_ownership.py')
    return hashlib.sha256(unit.encode()).hexdigest(), sources


def check_cpu(source, receipt):
    unit, sources = source_contract(source)
    cpu = json.loads(receipt.read_text())
    counts = cpu.get('counts', {})
    expected_inputs = {key: value for key, value in sources.items() if not key.endswith('/test_metadata_ownership.py')}
    _require(cpu.get('schema') == 'demoncat.metadata_arrays.cpu_contract.v1'
             and cpu.get('status') == 'CPU_CONTRACT_PASS' and cpu.get('includes_writer26') is True
             and cpu.get('compiler_returncode') == 0 and cpu.get('returncode') == 0
             and cpu.get('actual_npu_calls') == 0 and cpu.get('actual_sdk_or_numerical_acceptance') is False
             and cpu.get('generated_unit_sha256') == unit and cpu.get('production_sources') == expected_inputs,
             'FRESH_ACTUAL_PRODUCTION_CPU_CONTRACT_REQUIRED')
    _require(type(counts.get('actual_new_arrays')) is int and counts['actual_new_arrays'] > 0
             and counts['actual_new_arrays'] == counts.get('actual_delete_arrays') == counts.get('acquired_arrays')
             == counts.get('released_arrays') and counts.get('live_arrays') == 0,
             'CPU_ALLOCATION_RELEASE_COUNTS_INCOMPLETE')
    return sources


def write_receipt(source, native):
    sources = check_cpu(source, native / CPU_RECEIPT)
    images = {}
    for name, symbol in SYMBOLS.items():
        path = native / name
        exports = subprocess.run(['nm', '-D', '--defined-only', str(path)], check=True,
                                 capture_output=True, text=True).stdout
        _require(re.search(r'\bT ' + symbol + r'$', exports, re.M), 'ACTUAL_DSO_EXPORT_MISSING:' + name)
        images[name] = dict(sha256=sha(path), symbol=symbol)
    result = dict(schema=SCHEMA, sources=sources, images=images, state_version=1,
        cpu_receipt_sha256=sha(native / CPU_RECEIPT), actual_npu_calls=0,
        reset_supported=False, counter_scope='SDK_GETTER_ARRAYS_PER_LOADED_DSO',
        actual_loaded_readback_required=True, native_acceptance=False)
    with (native / RECEIPT).open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


def verify_parent(source, parent, files):
    _require(RECEIPT in files and CPU_RECEIPT in files, 'OLD_PARENT_REBUILD_REQUIRED')
    for name in (RECEIPT, CPU_RECEIPT):
        _require(sha(parent / name) == files[name], 'PARENT_EVIDENCE_BYTES_CHANGED')
    proof = json.loads((parent / RECEIPT).read_text())
    sources = check_cpu(source, parent / CPU_RECEIPT)
    _require(proof.get('schema') == SCHEMA and proof.get('sources') == sources
             and proof.get('state_version') == 1 and proof.get('reset_supported') is False
             and proof.get('actual_loaded_readback_required') is True
             and proof.get('actual_npu_calls') == 0 and proof.get('native_acceptance') is False
             and proof.get('cpu_receipt_sha256') == files[CPU_RECEIPT], 'OWNERSHIP_BUILD_CONTRACT_CHANGED')
    expected = {name: dict(sha256=files.get(name), symbol=symbol) for name, symbol in SYMBOLS.items()}
    _require(proof.get('images') == expected and all(sha(parent / name) == files.get(name) for name in SYMBOLS),
             'DEPENDENCY_IMAGE_OWNERSHIP_PROOF_MISMATCH')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    result = write_receipt(args.source.resolve(strict=True), args.native.resolve(strict=True))
    print(json.dumps(dict(status='CPU_BUILD_OWNERSHIP_PROOF_ONLY', images=result['images'], actual_npu_calls=0)))


if __name__ == '__main__':
    main()
