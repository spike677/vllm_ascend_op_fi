"""Generate an exact per-build CANN profile. No major-version allowlist.

Compile/API probes and real NPU discovery/lossless acceptance remain required.
The profile is an identity binding, not proof of ABI compatibility or a digital
signature. It covers all headers and the exact libraries used by this build.
"""
import argparse
import hashlib
import importlib.metadata
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
FAULT20_EXPORTS = (
    'aclnnQuantMatmulWeightNz', 'aclnnQuantMatmulWeightNzGetWorkspaceSize',
    'aclnnMatmulWeightNz', 'aclnnMatmulWeightNzGetWorkspaceSize',
)
AUDITED_PTA_RESOLVER_WHEEL = 'dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041'


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


def generate(toolkit,real,split,out,abi_facts,source_opapi,operator='matmul',math=None):
    adapters = {'matmul': 'matmul-cache-abi0-v1', 'fault20-isolated': 'fault20-isolated-cache-abi0-v1'}
    if operator not in adapters:
        raise ValueError('Unknown build operator')
    isolated_fault20 = operator == 'fault20-isolated'
    if isolated_fault20 and (split is None or math is None):
        raise ValueError('Isolated Fault20 requires main, nn and math opapi proxies')
    if math is not None and not isolated_fault20:
        raise ValueError('Math PTA proxy is limited to isolated Fault20')
    toolkit=toolkit.resolve(strict=True)
    paths={'nn':toolkit/'lib64/libnnopbase.so','acl':toolkit/'lib64/libascendcl.so','opapi':real}
    execute_exports = FAULT20_EXPORTS if isolated_fault20 else GETWS
    needs={'nn':PRIVATE+METADATA+(('CanUsePTACache',) if isolated_fault20 else ()),
           'acl':ACL,'opapi':execute_exports}
    if split is not None:
        paths['split']=split;needs['split']=execute_exports
    if math is not None:
        paths['math']=math;needs['math']=('aclnnAdd', 'aclnnAddGetWorkspaceSize')
    originals={'opapi':source_opapi,'nn':paths['nn'],'acl':paths['acl']}
    if split is not None:originals['split']=toolkit/'lib64/libopapi_nn.so'
    if math is not None:originals['math']=toolkit/'lib64/libopapi_math.so'
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
    version = metadata.get('ascend_toolkit_install.info', {}).get('version')
    if not version:
        raise ValueError('Installed toolkit version metadata required')
    if isolated_fault20 and version != '8.5.1':
        raise ValueError('Isolated Fault20 build is limited to CANN 8.5.1')
    headers={p.relative_to(toolkit).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for base in ('include','pkg_inc') for p in sorted((toolkit/base).rglob('*.h')) if p.is_file()}
    required_headers = ('include/acl/acl.h', 'include/aclnn/acl_meta.h', 'include/aclnn/opdev/op_cache.h',
                        'include/aclnnop/aclnn_matmul.h')
    if isolated_fault20:
        required_headers += ('include/aclnnop/aclnn_quant_matmul_weight_nz.h',)
    for p in required_headers:
        if p not in headers:raise ValueError('Required header missing: '+p)
    facts=json.loads(abi_facts.read_text())
    signatures = ('private_cache_signatures',) + (
        ('weight_nz_signature', 'qmm_signature') if isolated_fault20 else ('matmul_signature',)
    )
    if any(facts.get(key) != 'verified' for key in signatures) or facts.get('actual_npu_calls') != 0:
        raise ValueError('Successful ABI compile probe required')
    native=Path(__file__).resolve().parents[1]/'native'
    if isolated_fault20:
        root = native.parents[1]
        adapter_dir = root / 'fault_injection/native_adapters'
        fault20_sources = root / 'fault20_fast_validation/source'
        preload = fault20_sources / 'l4_buffer_substitute_allhit_v1'
        source_paths = [
            native / 'runtime_identity.h', native / 'test_identity.py',
            native / 'metadata_array_owner.h', native / 'test_metadata_ownership.py',
            native / 'metadata_ownership_fixture.cpp',
            Path(__file__).resolve(), Path(__file__).resolve().with_name('deployment_profile.py'),
            *(adapter_dir / name for name in ('build_qmm.sh', 'fault20_abi_probe.cpp',
                                             'matmul_auto.h', 'qmm_auto.inc',
                                             'qmm_current_call.h', 'qmm_current_call.inc',
                                             'qmm_pta_gate.h', 'qmm_pta_gate.cpp')),
            *(preload / name for name in ('buffer_substitute_preload.cpp', 'fault20_buffer_api.h',
                                         'extreme_transform.h', 'int_equivalent_transform.h', 'mantissa_truncate.h')),
            fault20_sources / 'l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp',
            fault20_sources / 'fault26_readonly_trace_v1/fault26_trace_shared.h',
            root / 'fault_injection/deployments/pta_resolver.py',
            root / 'fault_injection/deployments/matmul.py',
            *(root / 'tests/native' / name for name in (
                'test_qmm_current_call.cpp', 'test_qmm_pta_gate.cpp', 'test_qmm_pta_gate_elf.py',
                'test_qmm_pta_resolver_elf.py', 'test_matmul_auto.cpp', 'test_matmul_requests.cpp',
                'check_matmul_ctypes.py', 'check_matmul_request_ctypes.py')),
        ]
        sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source_paths}
    else:
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(native.iterdir())
                 if p.is_file() and p.suffix in ('.h','.cpp','.inc','.sh','.py')}
        sources['tools/version_profile.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile=dict(schema='fault39.cann.build_profile.v1',adapter=adapters[operator],
        toolkit=str(toolkit),environment_script=environment_script,
        proxy_layout='main_nn_math' if math else ('main_and_nn' if split else 'main_only'),abi_facts=facts,sources=sources,
        installation=metadata,headers=headers,libraries=libraries,upstream_libraries=upstream,required_exports=needs,
        acceptance='BUILD_IDENTITY_ONLY_REQUIRES_REAL_DISCOVERY_AND_LOSSLESS',actual_npu_calls=0)
    profile['metadata_array_ownership'] = dict(
        schema='demoncat.metadata_arrays.v1',
        state_symbol='dcat20_metadata_arrays_v1' if isolated_fault20 else 'fault39_metadata_arrays_v1',
        proxy_images=['libopapi.so'] + (['libopapi_nn.so'] if split else []),
        state_version=1, counter_scope='SDK_GETTER_ARRAYS_PER_LOADED_PROXY',
        reset_supported=False, runtime_quiescent_readback_required=True,
        numerical_or_resource_qualification=False)
    if isolated_fault20:
        distribution = importlib.metadata.distribution('torch-npu')
        torch_npu_library = Path(distribution.locate_file('torch_npu/lib/libtorch_npu.so'))
        torch_npu_sha256 = hashlib.sha256(torch_npu_library.read_bytes()).hexdigest()
        if torch_npu_sha256 != AUDITED_PTA_RESOLVER_WHEEL:
            raise ValueError('PTA_RESOLVER_WHEEL_NOT_AUDITED: ' + torch_npu_sha256)
        profile['controlled_pta_cache'] = dict(
            kind='EXPLICIT_COMPATIBILITY_ADAPTER_NOT_OFFICIAL_ENVIRONMENT_OPTION',
            environment='DEMONCAT_F20_REQUIRE_TYPED_GWS', enabled_value='1',
            targets=['aclnnQuantMatmulWeightNz', 'aclnnMatmulWeightNz'],
            original_symbol='CanUsePTACache', status_symbol='dcat20_pta_gate_status_v1',
            original_image='nn', preserves_op_exec_cache=True,
            resolver_entry='libopapi_math.so', original_math_image='math',
            original_math_anchor='aclnnAdd', requires_loaded_provider_identity=True,
            resolver_wheel=dict(distribution_version=distribution.version,
                                library_path=str(torch_npu_library.resolve()),
                                file_sha256=torch_npu_sha256,
                                exact_upstream_source_commit=None,
                                related_source_commit='b3065ed3ed174b08b7ecfcc564c0c464fb91fd13'),
            requires_status_admission=True, binary_identity_artifact='pta_gate_identity.json')
    serialized=json.dumps(profile,sort_keys=True,separators=(',',':')).encode()
    ident=hashlib.sha256(serialized).hexdigest();profile['profile_sha256']=ident
    out.mkdir(parents=True,exist_ok=True)
    (out/'version_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    lines=['#pragma once','#include "runtime_identity.h"','namespace f39profile {',
           'static constexpr const char* id='+json.dumps(ident)+';',
           'static constexpr const char* version='+json.dumps(metadata['ascend_toolkit_install.info']['version'])+';']
    if isolated_fault20:
        lines.extend(['#define DCAT20_HAS_PTA_RESOLVER_WHEEL_PROFILE 1',
            'static constexpr const char* pta_resolver_wheel_sha256='+json.dumps(torch_npu_sha256)+';',
            'static constexpr const char* pta_resolver_wheel_version='+json.dumps(distribution.version)+';'])
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
    p.add_argument('--math',type=Path)
    p.add_argument('--operator', choices=('matmul', 'fault20-isolated'), default='matmul')
    a=p.parse_args();r=generate(a.toolkit,a.real,a.split,a.output,a.abi_facts,a.source_opapi,a.operator,a.math)
    print(json.dumps({'version':r['installation']['ascend_toolkit_install.info']['version'],'profile_sha256':r['profile_sha256'],'acceptance':r['acceptance']}))


if __name__=='__main__':main()
