"""Malformed build identity inputs must fail before a provider can be armed."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

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


if __name__=='__main__':unittest.main()
