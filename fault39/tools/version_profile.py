"""Generate an exact per-build CANN profile. No major-version allowlist.

Compile/API probes and real NPU discovery/lossless acceptance remain required.
The profile is an identity binding, not proof of ABI compatibility or a digital
signature. It covers all headers and the exact libraries used by this build.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess

PRIVATE = ('_ZN2op8internal11OpExecCache3RunEPvS2_RKSt6vectorIS2_SaIS2_EE',
           '_ZN2op8internal14AddOpExecCacheEPNS0_11OpExecCacheE',
           '_ZN2op8internal13AddParamToBufEPK9aclTensor',
           '_ZN2op8internal13AddParamToBufEP9aclTensor', '_ZN2op8internal13GetLogApiInfoEv')
METADATA = ('aclGetDataType','aclGetFormat','aclGetViewOffset','aclGetViewShape',
            'aclGetStorageShape','aclGetViewStrides','aclGetRawTensorAddr')
ACL = ('aclrtGetDevice','aclrtMalloc','aclrtFree','aclrtMallocHost','aclrtFreeHost',
       'aclrtMemcpy','aclrtSynchronizeStream','aclrtGetCurrentContext')
GETWS = ('aclnnMatmul','aclnnMatmulGetWorkspaceSize')


def image(path):
    data=path.read_bytes()
    if len(data)<64 or data[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<H',data,18)[0]!=183:
        raise ValueError('ELF64 little-endian aarch64 library required: '+str(path))
    phoff=struct.unpack_from('<Q',data,32)[0]; entsize,count=struct.unpack_from('<HH',data,54)
    if entsize<56 or count>512 or phoff+entsize*count>len(data):raise ValueError('Invalid ELF program table')
    segments=[]
    for i in range(count):
        typ,flags,offset,vaddr,_,filesz,memsz,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*entsize)
        if typ==1 and flags&1:
            if not flags&4 or flags&2 or filesz>memsz or memsz>1<<30 or offset+filesz>len(data):raise ValueError('Unsupported executable segment')
            raw=data[offset:offset+filesz]+b'\0'*(memsz-filesz)
            segments.append(dict(vaddr=vaddr,size=memsz,sha256=hashlib.sha256(raw).hexdigest()))
    if not segments:raise ValueError('No executable segments')
    return dict(path=str(path.resolve()),file_sha256=hashlib.sha256(data).hexdigest(),file_bytes=len(data),segments=segments)


def exports(path):
    r=subprocess.run(['nm','-D','--defined-only',str(path)],text=True,capture_output=True,check=True)
    return {x.split()[-1].split('@')[0] for x in r.stdout.splitlines() if x.split()}


def generate(toolkit,real,split,out,abi_facts,source_opapi):
    toolkit=toolkit.resolve(strict=True)
    paths={'nn':toolkit/'lib64/libnnopbase.so','acl':toolkit/'lib64/libascendcl.so','opapi':real}
    needs={'nn':PRIVATE+METADATA,'acl':ACL,'opapi':GETWS}
    if split is not None:
        paths['split']=split;needs['split']=GETWS
    originals={'opapi':source_opapi,'nn':paths['nn'],'acl':paths['acl']}
    if split is not None:originals['split']=toolkit/'lib64/libopapi_nn.so'
    upstream={key:dict(path=str(path.resolve()),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
              for key,path in originals.items()}
    env_script=(toolkit.parent if toolkit.name=='aarch64-linux' else toolkit)/'set_env.sh'
    environment_script=dict(path=str(env_script.resolve(strict=True)),sha256=hashlib.sha256(env_script.read_bytes()).hexdigest())
    libraries={}
    for name,path in paths.items():
        missing=set(needs[name])-exports(path)
        if missing:raise ValueError('Missing '+name+' exports: '+','.join(sorted(missing)))
        libraries[name]=image(path)
    metadata={}
    for p in sorted(toolkit.glob('*install.info')):
        metadata[p.name]=dict(re.findall(r'^([^=\n]+)=(.*)$',p.read_text(errors='replace'),re.M))
    if not metadata.get('ascend_toolkit_install.info',{}).get('version'):raise ValueError('Installed toolkit version metadata required')
    headers={p.relative_to(toolkit).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for base in ('include','pkg_inc') for p in sorted((toolkit/base).rglob('*.h')) if p.is_file()}
    for p in ('include/acl/acl.h','include/aclnn/acl_meta.h','include/aclnn/opdev/op_cache.h','include/aclnnop/aclnn_matmul.h'):
        if p not in headers:raise ValueError('Required header missing: '+p)
    facts=json.loads(abi_facts.read_text())
    if (facts.get('matmul_signature')!='verified' or facts.get('private_cache_signatures')!='verified'
            or facts.get('actual_npu_calls')!=0):raise ValueError('Successful ABI compile probe required')
    native=Path(__file__).resolve().parents[1]/'native'
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native.iterdir())
             if p.is_file() and p.suffix in ('.h','.cpp','.inc','.sh','.py')}
    sources['tools/version_profile.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile=dict(schema='fault39.cann.build_profile.v1',adapter='matmul-cache-abi0-v1',
        toolkit=str(toolkit),environment_script=environment_script,proxy_layout='main_and_nn' if split else 'main_only',abi_facts=facts,sources=sources,
        installation=metadata,headers=headers,libraries=libraries,upstream_libraries=upstream,required_exports=needs,
        acceptance='BUILD_IDENTITY_ONLY_REQUIRES_REAL_DISCOVERY_AND_LOSSLESS',actual_npu_calls=0)
    serialized=json.dumps(profile,sort_keys=True,separators=(',',':')).encode()
    ident=hashlib.sha256(serialized).hexdigest();profile['profile_sha256']=ident
    out.mkdir(parents=True,exist_ok=True)
    (out/'version_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    lines=['#pragma once','#include "runtime_identity.h"','namespace f39profile {',
           'static constexpr const char* id='+json.dumps(ident)+';',
           'static constexpr const char* version='+json.dumps(metadata['ascend_toolkit_install.info']['version'])+';']
    for name,row in libraries.items():
        lines.append('static constexpr f39identity::Segment '+name+'_segments[]={'+','.join(
            '{'+str(s['vaddr'])+','+str(s['size'])+','+json.dumps(s['sha256'])+'}' for s in row['segments'])+'};')
        lines.append('static constexpr f39identity::Image '+name+'={'+json.dumps(row['file_sha256'])+','+str(row['file_bytes'])+','+name+'_segments,'+str(len(row['segments']))+'};')
    lines.append('}')
    (out/'fault39_build_profile.h').write_text('\n'.join(lines)+'\n')
    return profile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('toolkit','real','output','abi-facts','source-opapi'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--split',type=Path)
    a=p.parse_args();r=generate(a.toolkit,a.real,a.split,a.output,a.abi_facts,a.source_opapi)
    print(json.dumps({'version':r['installation']['ascend_toolkit_install.info']['version'],'profile_sha256':r['profile_sha256'],'acceptance':r['acceptance']}))


if __name__=='__main__':main()
