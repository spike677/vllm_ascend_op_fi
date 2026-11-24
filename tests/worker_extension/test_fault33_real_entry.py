import struct
import unittest
from tools.decode_fault33_candidate import check_binding


class EntryTests(unittest.TestCase):
    def test_real_entry_matches_header(self):
        raw=struct.pack('<10Q',0,12,3,456,1,1,0,8,0,0)
        check_binding(raw,dict(schema='dcat.launch.binding.v1',pid=12,serial=3,
                              key_source='entry',entry=456))

    def test_name_only_and_misaligned_binding_rejected(self):
        raw=struct.pack('<10Q',0,12,3,456,1,1,0,8,0,0)
        binding=dict(schema='dcat.launch.binding.v1',pid=12,serial=3,
                     key_source='entry',entry=456)
        for delta in (dict(key_source='name_only'),dict(entry=None),
                      dict(serial=4),dict(pid=13),dict(entry=457)):
            with self.assertRaisesRegex(ValueError,'REAL_ENTRY_BINDING_REQUIRED'):
                check_binding(raw,dict(binding,**delta))
