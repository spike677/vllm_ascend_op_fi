"""Compile actual metadata capture/log bodies against allocating CPU SDK mocks.

No CANN import, driver, container, or NPU operation occurs in this contract.
The owning root executes the compiler on its selected CPU build host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def between(text, start, end):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('UNIQUE_PRODUCTION_FUNCTION_BOUNDARIES_REQUIRED')
    return text.split(start, 1)[1].split(end, 1)[0]


def inputs(root):
    native = root / 'fault39/native'
    f20 = root / 'fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp'
    names = [f20, native / 'metadata_capture.inc', native / 'metadata_array_owner.h',
             native / 'metadata_ownership_fixture.cpp', native / 'fault39_metadata.h',
             root / 'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/fault20_buffer_api.h']
    sources = {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in names}
    text = f20.read_text(encoding='utf-8')
    captured = 'bool checked_product(' + between(text, 'bool checked_product(',
                                                'bool capture_optional_tensor_metadata(')
    appended = 'size_t append_dims(' + between(text, 'size_t append_dims(', 'void log_call(')
    logged = 'void log_tensor(' + between(text, 'void log_tensor(', 'void log_matmul_summary(')
    blocks = {'// @F20_CAPTURE@': captured, '// @F20_APPEND_DIMS@': appended,
              '// @F20_LOG@': logged, '// @F39_CAPTURE@': (native / 'metadata_capture.inc').read_text(encoding='utf-8')}
    unit = (native / 'metadata_ownership_fixture.cpp').read_text(encoding='utf-8')
    for marker, block in blocks.items():
        if unit.count(marker) != 1:
            raise ValueError('UNIQUE_FIXTURE_INSERTION_REQUIRED')
        unit = unit.replace(marker, block)
    return unit, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    unit, sources = inputs(root)
    compiler = shutil.which(args.cxx)
    receipt = dict(schema='demoncat.metadata_arrays.cpu_contract.v1', actual_npu_calls=0,
                   compiler=compiler, production_sources=sources,
                   generated_unit_sha256=sha(unit.encode()),
                   actual_sdk_or_numerical_acceptance=False,
                   status='NOT_RUN_COMPILER_UNAVAILABLE')
    if compiler:
        with tempfile.TemporaryDirectory(prefix='metadata-ownership-') as directory:
            path = Path(directory)
            source, exe = path / 'test.cpp', path / ('test.exe' if os.name == 'nt' else 'test')
            source.write_bytes(unit.encode())
            command = [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
                       '-I', str(root / 'fault39/native'), '-I', str(root / 'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1'),
                       str(source), '-o', str(exe)]
            built = subprocess.run(command, capture_output=True, text=True, timeout=120)
            receipt.update(compiler_command=command, compiler_returncode=built.returncode,
                           compiler_stdout=built.stdout, compiler_stderr=built.stderr)
            if built.returncode:
                receipt['status'] = 'FAILED_CPU_COMPILE'
            else:
                run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
                receipt.update(status='CPU_CONTRACT_PASS' if run.returncode == 0 else 'FAILED_CPU_CONTRACT',
                               returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
                if run.returncode == 0:
                    receipt['counts'] = json.loads(run.stdout)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['status'] == 'CPU_CONTRACT_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
