"""Actual Linux loader/alias CPU contract; no CANN, device, or container."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fault_injection.metadata_arrays import (LoadedMetadataArrays, LoadedWriter26MetadataArrays,
                                             WRITER26_SYMBOLS, SCHEMA)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.platform != 'linux' or not shutil.which('cc'):
        raise RuntimeError('LINUX_HOST_C_COMPILER_REQUIRED_NO_NPU_NEEDED')
    with tempfile.TemporaryDirectory(prefix='metadata-elf-') as directory:
        root = Path(directory)
        writer26 = '--writer26' in sys.argv
        core_name = 'libdcat26_group_v2.so' if writer26 else 'libfault39_native_v1.so'
        symbols = WRITER26_SYMBOLS if writer26 else {
            name: 'fault39_metadata_arrays_v1' for name in (core_name, 'libopapi.so', 'libopapi_nn.so')}
        source = root / 'proxy.c'
        code = '''#include <stdint.h>
struct State { uint32_t version, size; uint64_t values[14]; };
static uint64_t instance;
int fault39_metadata_arrays_v1(struct State *out) {
 if (!out || out->version != 1 || out->size != sizeof(*out)) return 1;
 struct State result = {1, sizeof(struct State), {0}};
 result.values[0] = (uintptr_t)&instance;
 *out=result; return 0;
}
int fault39_snapshot_v1(void) {return 0;}
'''
        for name, symbol in symbols.items():
            source.write_text(code.replace('fault39_metadata_arrays_v1', symbol))
            subprocess.run(['cc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                str(source), '-Wl,-soname,' + name, '-o', str(root / name)], check=True)
        sdk = root / 'sdk'
        sdk.mkdir()
        alias = sdk / 'libopapi_nn.so'
        os.link(root / alias.name, alias)
        profile = dict(schema='fault39.cann.build_profile.v1', adapter='matmul-cache-abi0-v1',
            proxy_layout='main_and_nn', metadata_array_ownership=dict(schema=SCHEMA, state_version=1,
                state_symbol='fault39_metadata_arrays_v1', proxy_images=['libopapi.so', 'libopapi_nn.so'],
                cached_registration_witness='fault39_gws_commit_arrays_v1',
                reset_supported=False, runtime_quiescent_readback_required=True))
        profile['profile_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True,
                                                    separators=(',', ':')).encode()).hexdigest()
        (root / 'version_profile.json').write_text(json.dumps(profile))
        manifest = dict(schema='fault39.native.build_manifest.v1', profile_sha256=profile['profile_sha256'],
            files={name: sha(root / name) for name in (core_name, 'libopapi.so',
                                                      'libopapi_nn.so', 'version_profile.json')})
        (root / 'build_manifest.json').write_text(json.dumps(manifest))
        if writer26:
            (root / 'metadata-array-ownership.json').write_text('CPU_FIXTURE_NOT_NATIVE_BUILD_PROOF')
            manifest_path = root / 'build.json'
            proof_sha = sha(root / 'metadata-array-ownership.json')
            manifest_path.write_text(json.dumps(dict(schema='demoncat.writer26.group_deployment.v2',
                files=dict({name: sha(root / name) for name in symbols}, **{'metadata-array-ownership.json': proof_sha}),
                metadata_array_ownership=dict(schema=SCHEMA, state_version=1, images=WRITER26_SYMBOLS,
                    reset_supported=False, runtime_quiescent_readback_required=True, parent_receipt_sha256=proof_sha))))
        core = ctypes.CDLL(str(root / core_name), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        main_proxy = ctypes.CDLL(str(root / 'libopapi.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        split_proxy = ctypes.CDLL(str(alias), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        trace = ctypes.CDLL(str(root / 'libfault26_readonly_trace_abi0.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL) if writer26 else None
        def attach():
            return (LoadedWriter26MetadataArrays(bundle=root, manifest_sha256=sha(root / 'build.json')) if writer26 else
                    LoadedMetadataArrays(kind='39', build_profile=profile['profile_sha256'], anchor=core.fault39_snapshot_v1))
        reader = attach()
        value = reader.read()
        assert len({row['state']['instance'] for row in value['images']}) == (4 if writer26 else 2)
        assert next(row for row in value['images'] if row['name'] == alias.name)['mapped_paths'] == [str(alias)]
        # Identical bytes in a foreign inode are not an approved alias.
        foreign = root / 'foreign'
        foreign.mkdir()
        shutil.copyfile(root / 'libopapi.so', foreign / 'libopapi.so')
        duplicate = ctypes.CDLL(str(foreign / 'libopapi.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        try:
            attach()
        except RuntimeError as error:
            assert 'UNKNOWN_PROXY_ALIAS_OR_DUPLICATE' in str(error), str(error)
        else:
            raise AssertionError('foreign same-bytes duplicate accepted')
        assert all(item._handle for item in (main_proxy, split_proxy, duplicate))
        assert not writer26 or trace._handle
        print(json.dumps(dict(status='CPU_LOADED_ALIAS_AND_DUPLICATE_CONTRACT_PASS', actual_npu_calls=0,
            actual_cann_loaded=False, same_inode_alias_accepted=True, foreign_same_bytes_inode_rejected=True,
            counter_instances_independent=True, counter_instances=4 if writer26 else 2, includes_writer26=writer26)))


if __name__ == '__main__':
    main()
