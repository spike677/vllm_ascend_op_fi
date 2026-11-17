"""Linux host-memory C++/ctypes contracts; no CANN calls, containers or NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Linux host compiler and host-memory fixture required')
    source, output = args.source.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = ('fault_injection/deployments/matmul.py', 'fault_injection/native_adapters/matmul_auto.h',
        'fault39/native/runtime_identity.h', 'tests/native/test_matmul_auto.cpp',
        'tests/native/test_matmul_requests.cpp', 'tests/native/check_matmul_ctypes.py',
        'tests/native/check_matmul_request_ctypes.py')
    receipt = dict(scope='HOST_MEMORY_CPU_ONLY_NO_CANN_NO_NPU', actual_npu_calls=0,
        python=sys.version, files={name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in files},
        checks=[], status='RUNNING')
    def save():
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def run(name, command, *, environment=None):
        result = subprocess.run(command, cwd=source, env=environment, capture_output=True, text=True, timeout=180)
        receipt['checks'].append(dict(name=name, command=command, exit_code=result.returncode,
                                      output=result.stdout + result.stderr))
        save()
        if result.returncode:
            raise RuntimeError(name + '_FAILED')
    try:
        flags = ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                 '-Wno-error=misleading-indentation', '-pthread']
        for name in ('test_matmul_auto', 'test_matmul_requests'):
            run(name + '_build', flags + [str(source / 'tests/native' / (name + '.cpp')), '-ldl', '-o', str(output / name)])
            run(name, [str(output / name)])
        fixture = output / 'fixture.so'
        run('fixture_build', flags + ['-shared', '-fPIC', '-DDCAT_TEST_SHARED',
            str(source / 'tests/native/test_matmul_auto.cpp'), '-ldl', '-o', str(fixture)])
        environment = dict(os.environ, LD_PRELOAD=str(fixture), PYTHONPATH=str(source),
                           PYTHONDONTWRITEBYTECODE='1')
        for name in ('check_matmul_ctypes.py', 'check_matmul_request_ctypes.py'):
            run(name, [sys.executable, str(source / 'tests/native' / name)], environment=environment)
        receipt['status'] = 'CPU_CONTRACTS_PASSED_NOT_NPU_QUALIFICATION'
    except BaseException as error:
        receipt.update(status='CPU_CONTRACTS_FAILED', error=repr(error))
    finally:
        save()
    print(json.dumps(dict(status=receipt['status'], checks=len(receipt['checks']), receipt=str(output / 'receipt.json'))))
    return 0 if receipt['status'] == 'CPU_CONTRACTS_PASSED_NOT_NPU_QUALIFICATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
