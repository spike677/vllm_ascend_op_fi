"""Real ELF resolver-order tests using owned CPU fixtures only, never CANN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


def run(*args, env=None):
    try:
        return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.output) from error


def declaration(name, path):
    raw = path.read_bytes()
    phoff = struct.unpack_from('<Q', raw, 32)[0]
    size, count = struct.unpack_from('<HH', raw, 54)
    segments = []
    for index in range(count):
        kind, flags, offset, addr, _, filesz, memsz, _ = struct.unpack_from('<IIQQQQQQ', raw, phoff + size * index)
        if kind == 1 and flags & 1:
            segments.append((addr, memsz, hashlib.sha256(raw[offset:offset + filesz] + b'\0' * (memsz - filesz)).hexdigest()))
    return ('static constexpr f39identity::Segment ' + name + '_segments[]={'
            + ','.join('{%d,%d,"%s"}' % row for row in segments) + '};\n'
            + 'static constexpr f39identity::Image ' + name + '={"'
            + hashlib.sha256(raw).hexdigest() + '",' + str(len(raw)) + ','
            + name + '_segments,' + str(len(segments)) + '};\n')


CHILD = r'''
import ctypes, importlib.util, json, os, pathlib, sys
directory, helper_path, case = sys.argv[1:]
root = pathlib.Path(directory)
ctypes.CDLL(str(root / 'support.so'), mode=os.RTLD_GLOBAL | os.RTLD_NOW)
ctypes.CDLL('libnnopbase.so', mode=os.RTLD_GLOBAL | os.RTLD_NOW)
if case != 'math-not-loaded':
    math = ctypes.CDLL('libopapi_math.so', mode=os.RTLD_NOW | os.RTLD_LOCAL)
for name in ('libopapi_nn.so', 'libopapi.so'):
    ctypes.CDLL(name, mode=os.RTLD_NOW | os.RTLD_LOCAL)
if case in ('custom-first', 'unknown-custom', 'custom-known-main'):
    ctypes.CDLL(str(root / 'custom/op_api/lib/libcust_opapi.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
spec=importlib.util.spec_from_file_location('pta_fixture_resolver', helper_path)
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
report=helper.inspect_loaded_pta_resolution()
try:
    selected=helper.validate_loaded_pta_resolution(report)
    validation='PASS'
except ValueError as error:
    selected=None
    validation=str(error)
if case in ('math-not-loaded', 'custom-first', 'unknown-custom'):
    print(json.dumps(dict(validation=validation, resolver=report)))
    raise SystemExit(0)
winner=next(row for row in report['rows'] if row['symbols'].get('CanUsePTACache'))
address=int(winner['symbols']['CanUsePTACache']['address'], 16)
can=ctypes.CFUNCTYPE(ctypes.c_bool,ctypes.c_char_p)(address)
values=[can(name.encode()) for name in ('aclnnQuantMatmulWeightNz','aclnnMatmulWeightNz','other','original_false')]
math=ctypes.CDLL('libopapi_math.so',mode=os.RTLD_NOLOAD | os.RTLD_NOW)
add=math.aclnnAdd
add.argtypes,add.restype=[ctypes.c_int],ctypes.c_int
math_value=add(7)
gates={}
for name in ('libopapi_math.so','libopapi_nn.so','libopapi.so'):
    handle=ctypes.CDLL(name,mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    try:status=handle.dcat20_pta_gate_status_v1
    except AttributeError:continue
    status.argtypes,status.restype=[],ctypes.c_char_p
    gates[name]=json.loads(status())
nn=ctypes.CDLL('libnnopbase.so',mode=os.RTLD_NOLOAD | os.RTLD_NOW)
count=nn.fixture_calls
count.argtypes,count.restype=[],ctypes.c_ulong
print(json.dumps(dict(validation=validation,resolver=report,gates=gates,values=values,
                     math_api_value=math_value,original_calls=count())))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cases = []
    with tempfile.TemporaryDirectory(prefix='dcat20-pta-resolver-') as directory:
        work = Path(directory)
        nn = work / 'nn.cpp'
        nn.write_text('''#include <cstring>
static unsigned long calls=0;
extern "C" bool CanUsePTACache(const char* api){++calls;return !api||std::strcmp(api,"original_false")!=0;}
extern "C" unsigned long fixture_calls(){return calls;}
''')
        run('g++', '-shared', '-fPIC', str(nn), '-Wl,-soname,libnnopbase.so', '-o', str(work / 'libnnopbase.so'))
        math = work / 'math.cpp'
        math.write_text('extern "C" int aclnnAdd(int value){return value+3;}\n')
        real = work / 'libopapi_math_real.so'
        run('g++', '-shared', '-fPIC', str(math), '-L' + str(work), '-Wl,--no-as-needed',
            '-l:libnnopbase.so', '-Wl,-rpath,$ORIGIN', '-Wl,-soname,libopapi_math_real.so', '-o', str(real))
        stock_math = work / 'stock-math.so'
        run('g++', '-shared', '-fPIC', str(math), '-L' + str(work), '-Wl,--no-as-needed',
            '-l:libnnopbase.so', '-Wl,-rpath,$ORIGIN', '-Wl,-soname,libopapi_math.so', '-o', str(stock_math))
        profile = work / 'fault39_build_profile.h'
        profile.write_text('#include "runtime_identity.h"\nnamespace f39profile {\n'
            'static constexpr const char* id="cpu-fixture-profile";\n'
            + declaration('nn', work / 'libnnopbase.so') + declaration('math', real) + '}\n')
        flags = ['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
                 # The shared, unchanged identity header uses compact one-line
                 # statements. Keep its existing style warning non-fatal.
                 '-Wno-error=misleading-indentation',
                 '-I' + str(work), '-I' + str(root / 'fault39/native')]
        gate = str(root / 'fault_injection/native_adapters/qmm_pta_gate.cpp')
        for name, define in (('libopapi.so', []), ('libopapi_nn.so', ['-DDEMONCAT_QMM_SPLIT'])):
            run(*flags, *define, gate, '-ldl', '-lpthread', '-Wl,-soname,' + name, '-o', str(work / name))
        math_gate = work / 'new-math.so'
        run(*flags, '-DDEMONCAT_QMM_MATH_GATE', gate, '-L' + str(work), '-Wl,--no-as-needed',
            '-l:libopapi_math_real.so', '-Wl,-rpath,$ORIGIN', '-Wl,-soname,libopapi_math.so',
            '-ldl', '-lpthread', '-o', str(math_gate))
        support = work / 'support.cpp'
        support.write_text('extern "C" const char* dcat20_qmm_profile_v1(){return "cpu-fixture-profile";}\n')
        run('g++', '-shared', '-fPIC', str(support), '-o', str(work / 'support.so'))
        custom = work / 'custom/op_api/lib/libcust_opapi.so'
        custom.parent.mkdir(parents=True)
        source = work / 'custom.cpp'
        source.write_text('extern "C" bool CanUsePTACache(const char*){return true;}\n')
        run('g++', '-shared', '-fPIC', str(source), '-o', str(custom))
        unknown_custom = work / 'unknown-custom.so'
        shutil.copyfile(custom, unknown_custom)
        source.write_text('extern "C" int custom_fixture_identity(){return 1;}\n')
        forwarding_custom = work / 'forwarding-custom.so'
        run('g++', '-shared', '-fPIC', str(source), '-L' + str(work), '-Wl,--no-as-needed',
            '-l:libopapi.so', '-o', str(forwarding_custom))
        child = work / 'child.py'
        child.write_text(CHILD)
        helper = root / 'fault_injection/deployments/pta_resolver.py'
        for case in ('old-math-dependency-bypass', 'controlled-math-first', 'disabled-pass-through',
                     'custom-first', 'unknown-custom', 'custom-known-main',
                     'math-not-loaded', 'original-math-mismatch'):
            shutil.copyfile(stock_math if case == 'old-math-dependency-bypass' else math_gate,
                            work / 'libopapi_math.so')
            env = dict(os.environ, LD_LIBRARY_PATH=str(work), DEMONCAT_F20_REQUIRE_TYPED_GWS='1')
            env.pop('ASCEND_OPP_PATH', None)
            env.pop('ASCEND_CUSTOM_OPP_PATH', None)
            shutil.copyfile(forwarding_custom if case == 'custom-known-main' else unknown_custom, custom)
            if case in ('custom-first', 'custom-known-main'):
                env['ASCEND_CUSTOM_OPP_PATH'] = str(work / 'custom')
            if case == 'disabled-pass-through':
                env['DEMONCAT_F20_REQUIRE_TYPED_GWS'] = '0'
            if case == 'original-math-mismatch':
                math.write_text('extern "C" int aclnnAdd(int value){return value+3;}\nextern "C" int identity_drift(){return 1;}\n')
                run('g++', '-shared', '-fPIC', str(math), '-L' + str(work), '-Wl,--no-as-needed',
                    '-l:libnnopbase.so', '-Wl,-rpath,$ORIGIN', '-Wl,-soname,libopapi_math_real.so', '-o', str(real))
            result = json.loads(run(sys.executable, str(child), str(work), str(helper), case, env=env))
            if case == 'old-math-dependency-bypass':
                assert result['validation'] == 'CAN_USE_PROVIDER_IS_NOT_OWNED_GATE:libopapi_math.so', result
                assert result['values'] == [True, True, True, False]
                assert all(row['ready'] and row['qmm_bypass'] == row['weight_nz_bypass'] == row['original_calls'] == 0
                           for row in result['gates'].values())
            elif case == 'controlled-math-first':
                assert result['validation'] == 'PASS', result
                assert result['values'] == [False, False, True, False]
                gate_row = result['gates']['libopapi_math.so']
                assert gate_row['ready'] and gate_row['original_math_identity_valid']
                assert gate_row['qmm_bypass'] == gate_row['weight_nz_bypass'] == 1
                assert gate_row['original_calls'] == result['original_calls'] == 2
            elif case == 'disabled-pass-through':
                assert result['validation'] == 'PASS', result
                assert result['values'] == [True, True, True, False]
                assert result['original_calls'] == 4
                assert not result['gates']['libopapi_math.so']['requested']
            elif case == 'custom-first':
                assert result['validation'] == 'WINNING_OPAPI_PROVIDER_IS_NOT_VERIFIED_GATE', result
            elif case == 'unknown-custom':
                assert result['validation'] == 'UNENUMERATED_LOADED_CUSTOM_OPAPI', result
            elif case == 'math-not-loaded':
                assert result['validation'] == 'ACTUAL_GATE_HANDLE_NOT_LOADED:libopapi_math.so', result
            elif case == 'custom-known-main':
                assert result['validation'] == 'PASS', result
                assert result['resolver']['winner_handle'] == str(custom)
                assert result['values'] == [False, False, True, False]
                gate_row = result['gates']['libopapi.so']
                assert gate_row['ready'] and gate_row['qmm_bypass'] == gate_row['weight_nz_bypass'] == 1
                assert result['gates']['libopapi_math.so']['qmm_bypass'] == 0
                assert gate_row['original_calls'] == result['original_calls'] == 2
            elif case == 'original-math-mismatch':
                assert result['validation'] == 'PASS', result
                gate_row = result['gates']['libopapi_math.so']
                assert gate_row['refused'] and not gate_row['ready'] and not gate_row['original_math_identity_valid']
                assert gate_row['qmm_bypass'] == gate_row['weight_nz_bypass'] == 0
                assert result['values'] == [True, True, True, False]
                assert result['original_calls'] == 4
            if 'math_api_value' in result:
                assert result['math_api_value'] == 10
            cases.append({'case': case, 'status': 'PASS_CPU_ONLY', 'evidence': result})
    report = {'schema': 'dcat20.pta_resolver.cpu_elf_tests.v1', 'status': 'PASS_CPU_ONLY',
              'actual_npu_calls': 0, 'vendor_libraries_loaded': False, 'cases': cases}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'status': report['status'], 'case_count': len(cases), 'actual_npu_calls': 0}))


if __name__ == '__main__':
    main()
