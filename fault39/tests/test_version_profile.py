"""Malformed build identity inputs must fail before a provider can be armed."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'tools/version_profile.py'
spec=importlib.util.spec_from_file_location('version_profile',path)
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)


def elf(*,machine=183,flags=5,offset=128,filesz=4,memsz=8):
    data=bytearray(132);data[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<H',data,18,machine);struct.pack_into('<Q',data,32,64)
    struct.pack_into('<HH',data,54,56,1)
    struct.pack_into('<IIQQQQQQ',data,64,1,flags,offset,0x1000,0,filesz,memsz,4096)
    data[128:]=b'abcd'
    return data


class ElfIdentityTests(unittest.TestCase):
    def inspect(self,data):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'owned.so';p.write_bytes(data);return profile.image(p)

    def test_zero_filled_segment_and_full_file_have_separate_hashes(self):
        raw=elf();r=self.inspect(raw)
        self.assertEqual(r['file_sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(r['segments'],[dict(vaddr=4096,size=8,sha256=hashlib.sha256(b'abcd\0\0\0\0').hexdigest())])

    def test_wrong_architecture_is_not_enabled_by_version_name(self):
        with self.assertRaisesRegex(ValueError,'aarch64'):self.inspect(elf(machine=62))

    def test_corrupted_or_writable_code_is_rejected(self):
        for raw in (b'',elf(flags=7),elf(filesz=5,memsz=4),elf(offset=131),elf(memsz=(1<<30)+1)):
            with self.subTest(data=bytes(raw[:16])):
                with self.assertRaises(ValueError):self.inspect(raw)

    def test_missing_executable_segments(self):
        with self.assertRaisesRegex(ValueError,'No executable'):self.inspect(elf(flags=4))

    def test_truncated_program_table(self):
        raw=elf();struct.pack_into('<H',raw,56,512)
        with self.assertRaisesRegex(ValueError,'program table'):self.inspect(raw)

    def test_isolated_fault20_build_profiles_both_distinct_operator_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            toolkit = root / 'aarch64-linux'
            libraries = toolkit / 'lib64'
            libraries.mkdir(parents=True)
            for name in ('libnnopbase.so', 'libascendcl.so', 'libopapi.so', 'libopapi_nn.so', 'libopapi_math.so'):
                (libraries/name).write_bytes(elf())
            for name in ('acl/acl.h', 'aclnn/acl_meta.h', 'aclnn/opdev/op_cache.h', 'aclnnop/aclnn_matmul.h',
                         'aclnnop/aclnn_quant_matmul_weight_nz.h'):
                path = toolkit / 'include' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture header')
            (root/'set_env.sh').write_text('# fixture script\n')
            (toolkit/'ascend_toolkit_install.info').write_text('version=8.5.1\n')
            facts = root/'abi_facts.json'
            abi = dict(weight_nz_signature='verified', qmm_signature='verified',
                       private_cache_signatures='verified', actual_npu_calls=0)
            facts.write_text(json.dumps(abi))
            exports = set(profile.PRIVATE + profile.METADATA + profile.FAULT20_EXPORTS + profile.ACL
                          + ('CanUsePTACache', 'aclnnAdd', 'aclnnAddGetWorkspaceSize'))
            wheel = root / 'libtorch_npu.so'
            wheel.write_bytes(b'CPU fixture; not a real torch_npu library')
            wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            distribution = SimpleNamespace(version='cpu-fixture', locate_file=lambda _: wheel)
            with (patch.object(profile, 'exports', return_value=exports),
                  patch.object(profile.importlib.metadata, 'distribution', return_value=distribution),
                  patch.object(profile, 'AUDITED_PTA_RESOLVER_WHEEL', wheel_digest)):
                args = (toolkit, libraries/'libopapi.so', libraries/'libopapi_nn.so', root/'build',
                        facts, libraries/'libopapi.so')
                with self.assertRaisesRegex(ValueError, 'main, nn and math'):
                    profile.generate(*args, operator='fault20-isolated')
                result = profile.generate(*args, operator='fault20-isolated', math=libraries/'libopapi_math.so')
                self.assertEqual(result['adapter'], 'fault20-isolated-cache-abi0-v1')
                for library in ('opapi', 'split'):
                    self.assertEqual(set(result['required_exports'][library]), set(profile.FAULT20_EXPORTS))
                self.assertIn('fault_injection/native_adapters/fault20_abi_probe.cpp', result['sources'])
                self.assertNotIn('runtime.cpp', result['sources'])
                self.assertEqual(result['proxy_layout'], 'main_nn_math')
                self.assertEqual(result['controlled_pta_cache']['resolver_wheel']['file_sha256'], wheel_digest)
                facts.write_text(json.dumps(dict(abi, qmm_signature='unverified')))
                with self.assertRaisesRegex(ValueError, 'ABI compile probe'):
                    profile.generate(*args, operator='fault20-isolated', math=libraries/'libopapi_math.so')


if __name__=='__main__':unittest.main()
