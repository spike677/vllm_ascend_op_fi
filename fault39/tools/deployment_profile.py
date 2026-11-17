"""Verify an immutable build or its installed libraries without loading CANN."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_profile(native):
    profile=json.loads((native/'version_profile.json').read_text())
    content=dict(profile);identity=content.pop('profile_sha256',None)
    if (profile.get('schema')!='fault39.cann.build_profile.v1' or
            hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=identity):
        raise ValueError('BUILD_PROFILE_IDENTITY_MISMATCH')
    if profile.get('proxy_layout') not in ('main_only','main_and_nn','main_nn_math'):
        raise ValueError('UNSUPPORTED_PROXY_LAYOUT')
    toolkit=PurePosixPath(profile['toolkit'])
    if not toolkit.is_absolute() or '..' in toolkit.parts:raise ValueError('INVALID_TOOLKIT_PATH')
    return profile


def artifacts(profile):
    adapter = profile.get('adapter', 'matmul-cache-abi0-v1')
    library = {
        'matmul-cache-abi0-v1': 'libfault39_native_v1.so',
        'fault20-isolated-cache-abi0-v1': 'libdcat20_qmm.so',
    }.get(adapter)
    if library is None or (adapter == 'fault20-isolated-cache-abi0-v1'
                           and profile['proxy_layout'] not in ('main_and_nn', 'main_nn_math')):
        raise ValueError('UNSUPPORTED_BUILD_ADAPTER')
    paths=[library,'libopapi.so','real/libopapi_real.so','version_profile.json','abi_facts.json','identity_tests.json']
    if profile['proxy_layout'] in ('main_and_nn','main_nn_math'):
        paths+=['libopapi_nn.so','real/libopapi_nn_real.so']
    if profile['proxy_layout']=='main_nn_math':
        if adapter != 'fault20-isolated-cache-abi0-v1':
            raise ValueError('UNSUPPORTED_MATH_PROXY_ADAPTER')
        paths+=['libopapi_math.so','real/libopapi_math_real.so']
    if adapter == 'fault20-isolated-cache-abi0-v1' and profile.get('controlled_pta_cache'):
        paths += ['pta_gate_identity.json']
    return paths


def verify_build(native):
    profile=read_profile(native)
    manifest=json.loads((native/'build_manifest.json').read_text())
    if (manifest.get('schema')!='fault39.native.build_manifest.v1' or
            manifest.get('profile_sha256')!=profile['profile_sha256'] or
            set(manifest.get('files',{}))!=set(artifacts(profile))):
        raise ValueError('BUILD_MANIFEST_MISMATCH')
    for name,expected in manifest['files'].items():
        if sha(native/name)!=expected:raise ValueError('BUILD_ARTIFACT_MISMATCH: '+name)
    for name,file in [('opapi','real/libopapi_real.so'),('split','real/libopapi_nn_real.so'),
                      ('math','real/libopapi_math_real.so')]:
        if name in profile['libraries'] and sha(native/file)!=profile['libraries'][name]['file_sha256']:
            raise ValueError('PROFILE_LIBRARY_MISMATCH: '+name)
    return profile


def verify_installed(profile,environment=False):
    for name,row in profile['upstream_libraries'].items():
        if sha(Path(row['path']))!=row['file_sha256']:
            raise ValueError('INSTALLED_LIBRARY_MISMATCH: '+name)
    script=profile['environment_script']
    if sha(Path(script['path']))!=script['sha256']:raise ValueError('TOOLKIT_ENV_SCRIPT_MISMATCH')
    if environment:
        toolkit=Path(profile['toolkit']).resolve()
        home=Path(os.environ.get('ASCEND_HOME_PATH','/missing'))
        if (home/'aarch64-linux').is_dir():home=home/'aarch64-linux'
        if home.resolve()!=toolkit:raise ValueError('SELECTED_TOOLKIT_MISMATCH')
        if Path(os.environ.get('ASCEND_OPP_PATH','/missing')).resolve()!=(toolkit.parent/'opp').resolve():
            raise ValueError('SELECTED_OPP_MISMATCH')
        for name,library in [('nn','libnnopbase.so'),('acl','libascendcl.so')]:
            first=next((Path(folder)/library for folder in os.environ.get('LD_LIBRARY_PATH','').split(':')
                        if folder and (Path(folder)/library).is_file()),None)
            if first is None or sha(first)!=profile['upstream_libraries'][name]['file_sha256']:
                raise ValueError('LIBRARY_SEARCH_ORDER_MISMATCH: '+name)
    return dict(status='INSTALLED_FILES_AND_ENV_MATCH_BUILD' if environment else 'INSTALLED_FILES_MATCH_BUILD',actual_npu_calls=0,
                profile_sha256=profile['profile_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True)
    choice=p.add_mutually_exclusive_group()
    choice.add_argument('--write-manifest',action='store_true')
    choice.add_argument('--verify-installed',action='store_true')
    p.add_argument('--verify-environment',action='store_true')
    a=p.parse_args()
    if a.write_manifest:
        profile=read_profile(a.native)
        result=dict(schema='fault39.native.build_manifest.v1',profile_sha256=profile['profile_sha256'],
                    status='PASS_CPU_BUILD_ONLY',actual_npu_calls=0,files={name:sha(a.native/name) for name in artifacts(profile)})
        with (a.native/'build_manifest.json').open('x') as f:json.dump(result,f,indent=2)
    else:
        profile=verify_build(a.native)
        result=verify_installed(profile,a.verify_environment) if a.verify_installed else dict(status='BUILD_FILES_MATCH_PROFILE',actual_npu_calls=0)
    print(json.dumps(result))


if __name__=='__main__':main()
