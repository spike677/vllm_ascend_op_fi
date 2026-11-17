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
from fault_injection.metadata_arrays import LoadedMetadataArrays, SCHEMA


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sys.platform != 'linux' or not shutil.which('cc'):
        raise RuntimeError('LINUX_HOST_C_COMPILER_REQUIRED_NO_NPU_NEEDED')
    with tempfile.TemporaryDirectory(prefix='metadata-elf-') as directory:
        root = Path(directory)
        source = root / 'proxy.c'
        source.write_text('''#include <stdint.h>
struct State { uint32_t version, size; uint64_t values[14]; };
static uint64_t instance;
int fault39_metadata_arrays_v1(struct State *out) {
 if (!out || out->version != 1 || out->size != sizeof(*out)) return 1;
 struct State result = {1, sizeof(struct State), {0}};
 result.values[0] = (uintptr_t)&instance;
 *out=result; return 0;
}
int fault39_snapshot_v1(void) {return 0;}
''')
        for name in ('libfault39_native_v1.so', 'libopapi.so', 'libopapi_nn.so'):
            subprocess.run(['cc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                str(source), '-Wl,-soname,' + name, '-o', str(root / name)], check=True)
        sdk = root / 'sdk'
        sdk.mkdir()
        alias = sdk / 'libopapi_nn.so'
        os.link(root / alias.name, alias)
        profile = dict(schema='fault39.cann.build_profile.v1', adapter='matmul-cache-abi0-v1',
            proxy_layout='main_and_nn', metadata_array_ownership=dict(schema=SCHEMA, state_version=1,
                state_symbol='fault39_metadata_arrays_v1', proxy_images=['libopapi.so', 'libopapi_nn.so'],
                reset_supported=False, runtime_quiescent_readback_required=True))
        profile['profile_sha256'] = hashlib.sha256(json.dumps(profile, sort_keys=True,
                                                    separators=(',', ':')).encode()).hexdigest()
        (root / 'version_profile.json').write_text(json.dumps(profile))
        manifest = dict(schema='fault39.native.build_manifest.v1', profile_sha256=profile['profile_sha256'],
            files={name: sha(root / name) for name in ('libfault39_native_v1.so', 'libopapi.so',
                                                      'libopapi_nn.so', 'version_profile.json')})
        (root / 'build_manifest.json').write_text(json.dumps(manifest))
        core = ctypes.CDLL(str(root / 'libfault39_native_v1.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        main_proxy = ctypes.CDLL(str(root / 'libopapi.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        split_proxy = ctypes.CDLL(str(alias), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        reader = LoadedMetadataArrays(kind='39', build_profile=profile['profile_sha256'], anchor=core.fault39_snapshot_v1)
        value = reader.read()
        assert len({row['state']['instance'] for row in value['images']}) == 2
        assert value['images'][1]['mapped_paths'] == [str(alias)]
        # Identical bytes in a foreign inode are not an approved alias.
        foreign = root / 'foreign'
        foreign.mkdir()
        shutil.copyfile(root / 'libopapi.so', foreign / 'libopapi.so')
        duplicate = ctypes.CDLL(str(foreign / 'libopapi.so'), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        try:
            LoadedMetadataArrays(kind='39', build_profile=profile['profile_sha256'], anchor=core.fault39_snapshot_v1)
        except RuntimeError as error:
            assert 'UNKNOWN_PROXY_ALIAS_OR_DUPLICATE' in str(error), str(error)
        else:
            raise AssertionError('foreign same-bytes duplicate accepted')
        assert all(item._handle for item in (main_proxy, split_proxy, duplicate))
        print(json.dumps(dict(status='CPU_LOADED_ALIAS_AND_DUPLICATE_CONTRACT_PASS', actual_npu_calls=0,
            actual_cann_loaded=False, same_inode_alias_accepted=True, foreign_same_bytes_inode_rejected=True,
            counter_instances_independent=True)))


if __name__ == '__main__':
    main()
