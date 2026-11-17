"""Compile production PTA gate against owned CPU ELF fixtures, never CANN/NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def run(*args, env=None):
    return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT, env=env)


def image(path):
    raw = path.read_bytes()
    assert raw[:6] == b'\x7fELF\x02\x01'
    phoff = struct.unpack_from('<Q', raw, 32)[0]
    size, count = struct.unpack_from('<HH', raw, 54)
    segments = []
    for index in range(count):
        typ, flags, offset, addr, _, filesz, memsz, _ = struct.unpack_from('<IIQQQQQQ', raw, phoff + size * index)
        if typ == 1 and flags & 1:
            assert flags & 4 and not flags & 2
            segments.append((addr, memsz, hashlib.sha256(raw[offset:offset + filesz] + b'\0' * (memsz - filesz)).hexdigest()))
    return hashlib.sha256(raw).hexdigest(), len(raw), segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    receipts = []
    with tempfile.TemporaryDirectory(prefix='dcat20-pta-elf-') as directory:
        work = Path(directory)
        fixture = work / 'fixture.cpp'
        fixture.write_text('''#include <cstring>
static unsigned long count=0;
extern "C" bool CanUsePTACache(const char* api){++count; return !api || std::strcmp(api,"original_false")!=0;}
extern "C" unsigned long fixture_calls(){return count;}
''')
        library = work / 'libnnopbase.so'
        run('g++', '-shared', '-fPIC', str(fixture), '-Wl,-soname,libnnopbase.so', '-o', str(library))
        digest, length, segments = image(library)
        declarations = ','.join('{%d,%d,"%s"}' % row for row in segments)
        profile = work / 'fault39_build_profile.h'
        profile.write_text('#include "runtime_identity.h"\nnamespace f39profile {\n'
            'static constexpr const char* id="cpu-fixture-profile";\n'
            'static constexpr f39identity::Segment nn_segments[]={' + declarations + '};\n'
            'static constexpr f39identity::Image nn={"' + digest + '",' + str(length) + ',nn_segments,' + str(len(segments)) + '};\n}\n')
        proxy = work / 'libopapi.so'
        compile_flags = ['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra',
                         '-I' + str(work), '-I' + str(root / 'fault39/native')]
        run(*compile_flags, str(root / 'fault_injection/native_adapters/qmm_pta_gate.cpp'), '-ldl', '-lpthread', '-o', str(proxy))
        main = work / 'main.cpp'
        main.write_text(r'''#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
static bool profile_bad=false;
extern "C" const char* dcat20_qmm_profile_v1(){return profile_bad ? "wrong-profile" : "cpu-fixture-profile";}
int main(int argc,char** argv){
 if(argc!=4)return 11;
 profile_bad=std::strcmp(argv[3],"bad-profile")==0;
 void* nn=nullptr;
 if(std::strcmp(argv[3],"missing")!=0) nn=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);
 if(!nn && std::strcmp(argv[3],"missing")!=0)return 12;
 void* proxy=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);if(!proxy){std::fprintf(stderr,"%s\n",dlerror());return 13;}
 auto status=reinterpret_cast<const char*(*)()>(dlsym(proxy,"dcat20_pta_gate_status_v1"));
 auto can=reinterpret_cast<bool(*)(const char*)>(dlsym(proxy,"CanUsePTACache"));
 if(!can || !status)return 14;
 std::puts(status());
 if(std::strcmp(argv[3],"missing")==0)return 0;
 if(std::strcmp(argv[3],"change")==0)setenv("DEMONCAT_F20_REQUIRE_TYPED_GWS","0",1);
 int values[]={can("aclnnQuantMatmulWeightNz"),can("aclnnMatmulWeightNz"),can("aclnnMatmul"),
   can("aclnnMatmulWeightNzGetWorkspaceSize"),can("original_false"),can(nullptr)};
 auto calls=reinterpret_cast<unsigned long(*)()>(dlsym(nn,"fixture_calls"));if(!calls)return 15;
 std::printf("{\"values\":[%d,%d,%d,%d,%d,%d],\"calls\":%lu}\n",values[0],values[1],values[2],values[3],values[4],values[5],calls());
 std::puts(status());return 0;
}
''')
        executable = work / 'main'
        run('g++', '-std=c++17', '-rdynamic', str(main), '-ldl', '-o', str(executable))
        for label, mode, action in [('enabled', '1', 'normal'), ('disabled', '0', 'normal'),
                                    ('unset', None, 'normal'), ('invalid', 'true', 'normal'),
                                    ('profile-mismatch', '1', 'bad-profile'), ('changed', '1', 'change'),
                                    ('missing-original', '1', 'missing')]:
            env = dict(os.environ)
            env.pop('DEMONCAT_F20_REQUIRE_TYPED_GWS', None)
            if mode is not None:
                env['DEMONCAT_F20_REQUIRE_TYPED_GWS'] = mode
            rows = [json.loads(line) for line in run(str(executable), str(library), str(proxy), action, env=env).splitlines()]
            if label == 'enabled':
                assert rows[1] == {'values': [0, 0, 1, 1, 0, 1], 'calls': 4}
                assert rows[-1]['ready'] and not rows[-1]['refused']
                assert rows[-1]['qmm_bypass'] == rows[-1]['weight_nz_bypass'] == 1
            elif label in ('disabled', 'unset'):
                assert rows[1] == {'values': [1, 1, 1, 1, 0, 1], 'calls': 6}
                assert not rows[-1]['requested'] and not rows[-1]['refused']
            else:
                assert rows[-1]['refused'] and not rows[-1]['ready']
                assert rows[-1]['qmm_bypass'] == rows[-1]['weight_nz_bypass'] == 0
                if label != 'missing-original':
                    assert rows[1] == {'values': [1, 1, 1, 1, 0, 1], 'calls': 6}
            receipts.append({'case': label, 'status': 'PASS_CPU_ONLY', 'gate': rows[-1]})
        # Replace the profiled file with a different owned ELF while keeping its
        # SONAME. A fresh process must forward normally and refuse the control.
        fixture.write_text(fixture.read_text().replace('static unsigned long count=0;', 'static unsigned long count=0;\nextern "C" int different_image(){return 3;}'))
        run('g++', '-shared', '-fPIC', str(fixture), '-Wl,-soname,libnnopbase.so', '-o', str(library))
        env = dict(os.environ, DEMONCAT_F20_REQUIRE_TYPED_GWS='1')
        rows = [json.loads(line) for line in run(str(executable), str(library), str(proxy), 'normal', env=env).splitlines()]
        assert rows[-1]['refused'] and not rows[-1]['identity_valid'] and not rows[-1]['ready']
        assert rows[1] == {'values': [1, 1, 1, 1, 0, 1], 'calls': 6}
        receipts.append({'case': 'loaded-image-mismatch', 'status': 'PASS_CPU_ONLY', 'gate': rows[-1]})
    result = {'schema': 'dcat20.pta_gate.cpu_elf_tests.v1', 'status': 'PASS_CPU_ONLY',
              'vendor_libraries_loaded': False, 'actual_npu_calls': 0, 'cases': receipts}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'status': result['status'], 'case_count': len(receipts), 'actual_npu_calls': 0}))


if __name__ == '__main__':
    main()
