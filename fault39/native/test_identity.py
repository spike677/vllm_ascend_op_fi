"""Test the production loaded-code identity check against owned CPU DSOs.

No vendor library is loaded; these are identity tests, not NPU acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('version_profile',here.parent/'tools/version_profile.py')
    profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
    def run(*args):return subprocess.check_output(list(args),text=True,stderr=subprocess.STDOUT)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        for n in (7,13):
            src=root/f'{n}.cpp';src.write_text('extern "C" int identity_fixture(int a){return a+'+str(n)+';}')
            run('g++','-shared','-fPIC',str(src),'-o',str(root/f'fixture{n}.so'))
        good=profile.image(root/'fixture7.so');bad=profile.image(root/'fixture13.so')
        declarations=[]
        for key,row in [('good',good),('bad',bad)]:
            declarations.append('const f39identity::Segment '+key+'_segments[]={'+','.join(
                '{'+str(s['vaddr'])+','+str(s['size'])+','+json.dumps(s['sha256'])+'}' for s in row['segments'])+'};')
            declarations.append('const f39identity::Image '+key+'={'+json.dumps(row['file_sha256'])+','+str(row['file_bytes'])+','+key+'_segments,'+str(len(row['segments']))+'};')
        vectors=[b'',b'abc',b'a'*55,b'a'*56,b'a'*63,b'a'*64,b'a'*65,b'a'*1000000]
        checks='\n'.join('assert(f39identity::digest(std::string('+str(len(v))+",'a').data(),"+str(len(v))+')=="'+hashlib.sha256(v).hexdigest()+'");' for v in vectors if v!=b'abc')
        source=root/'test.cpp'
        source.write_text('#include "runtime_identity.h"\n#include <cassert>\n'+ '\n'.join(declarations)+r'''
int main(int argc,char** argv){
 assert(argc==3);
 assert(f39identity::digest("abc",3)=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
 SHA_CHECKS
 assert(!f39identity::matches(nullptr,good));
 void* handle=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);assert(handle);
 void* fn=dlsym(handle,"identity_fixture");assert(fn);
 assert(f39identity::matches(fn,good));
 assert(!f39identity::matches(fn,bad));
 auto wrong_size=good;wrong_size.file_bytes++;assert(!f39identity::file_matches(argv[1],wrong_size));
 assert(!f39identity::file_matches(argv[2],good));
 dlclose(handle);assert(f39identity::matches(fn,good)); // successful identity pins mapping
}
'''.replace('SHA_CHECKS',checks))
        exe=root/'test';run('g++','-std=c++17','-I'+str(here),str(source),'-ldl','-lpthread','-o',str(exe))
        run(str(exe),str(root/'fixture7.so'),str(root/'fixture13.so'))
    result=dict(schema='fault39.identity.cpu_tests.v1',status='PASS_CPU_ONLY',actual_npu_calls=0,
        scope=['8 SHA256 vectors','matching loaded fixture','mismatched loaded fixture','null symbol',
               'on-disk size mismatch','on-disk content mismatch','mapping pin after dlclose'])
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
