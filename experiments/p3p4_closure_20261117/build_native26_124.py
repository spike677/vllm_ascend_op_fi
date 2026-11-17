"""Rebuild the additive FIA state ABI; verify every retained parent library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--parent-manifest-sha256',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=a.parent/'build.json'
if sha(manifest)!=a.parent_manifest_sha256:raise RuntimeError('PARENT_MANIFEST_MISMATCH')
parent=json.loads(manifest.read_bytes())
for name,digest in parent['files'].items():
    file=(a.parent/name).resolve()
    if not file.is_relative_to(a.parent.resolve()) or sha(file)!=digest:raise RuntimeError('PARENT_LIBRARY_MISMATCH')
if a.out.exists():raise RuntimeError('NEW_BUILD_DIRECTORY_REQUIRED')
shutil.copytree(a.parent,a.out)
toolkit='/usr/local/Ascend/cann-8.5.1/aarch64-linux'
source='fault20_fast_validation/source/fault26_fia_writer_proxy_v1/fault26_fia_writer_lossless.cpp'
flags=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared',
    '-D_GLIBCXX_USE_CXX11_ABI=0','-DDEMONCAT_NATIVE_PROVIDER_BRIDGE',
    '-I'+toolkit+'/include','-I'+toolkit+'/include/aclnn','-I'+toolkit+'/pkg_inc',
    '/source/'+source,'-Wl,-soname,libfault26_fia_writer_proxy_abi0.so','-ldl','-lpthread',
    '-o','/build/libfault26_fia_writer_proxy_abi0.so']
cmd=['docker','run','--rm','--network','none','--runtime','runc','--cpus','4','--memory','4g',
    '-e','ASCEND_VISIBLE_DEVICES=void','-v',str(a.source.resolve())+':/source:ro',
    '-v',str(a.out.resolve())+':/build','--entrypoint','g++','quay.io/ascend/vllm-ascend:v0.18.0']+flags[1:]
with (a.out/'build.log').open('wb') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
if result.returncode:raise RuntimeError('COMPILE_FAILED')
files={file.relative_to(a.out).as_posix():sha(file) for file in a.out.rglob('*.so')}
for name,digest in parent['files'].items():
    if name!='libfault26_fia_writer_proxy_abi0.so' and files[name]!=digest:raise RuntimeError('UNEXPECTED_PARENT_CHANGE')
record=dict(schema='demoncat.native26.deployment.v1',files=files)
(a.out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
(a.out/'receipt.json').write_text(json.dumps(dict(command=cmd,actual_npu_calls=0,status='COMPILED_NOT_NPU_ACCEPTANCE',
    parent_manifest_sha256=a.parent_manifest_sha256,sources={name:sha(a.source/name) for name in
        (source,'fault_injection/native_adapters/legacy26_fia_bridge.inc')}),indent=2)+'\n')
print(json.dumps(record))
