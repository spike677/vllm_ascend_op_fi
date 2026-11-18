"""Stdlib-only bounds and identity tests for original-ELF route diagnosis."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'
spec = importlib.util.spec_from_file_location('review_attention33_first_boundary_source53',
    EXPERIMENT/'review_attention33_first_boundary_source53.py')
original = importlib.util.module_from_spec(spec);sys.modules[spec.name] = original;spec.loader.exec_module(original)
spec = importlib.util.spec_from_file_location('diagnose_attention33_native_routes', EXPERIMENT/'diagnose_attention33_native_routes.py')
reader = importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def make_elf(symbol='flash_attention_infer_split_fd'):
    names = b'\0.shstrtab\0.strtab\0.symtab\0'
    strings = b'\0'+symbol.encode()+b'\0'
    symbols = bytes(24)+struct.pack('<IBBHQQ', 1, 0x12, 0, 1, 64, 128)
    blob = bytearray(64); parts=[]
    for value in (names, strings, symbols):
        parts.append((len(blob),len(value)));blob.extend(value)
    while len(blob)%8:blob.append(0)
    shoff=len(blob)
    headers=[(0,0,0,0,0,0,0,0,0,0),
        (1,3,0,0,*parts[0],0,0,1,0),
        (11,3,0,0,*parts[1],0,0,1,0),
        (19,2,0,0,*parts[2],2,1,8,24)]
    for row in headers:blob.extend(struct.pack('<IIQQQQIIQQ', *row))
    ident=b'\x7fELF\x02\x01\x01'+bytes(9)
    blob[:64]=struct.pack('<16sHHIQQQIHHHHHH', ident, 1, 183, 1, 0,0,shoff,0,64,0,0,64,4,1)
    return bytes(blob)


class RouteDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.binary=self.root/'registered-1-327.bin'

    def read(self, raw):
        self.binary.write_bytes(raw)
        return reader.elf_symbols(self.binary, hashlib.sha256(raw).hexdigest())

    def test_real_elf_structure_symbols_are_reported_without_abi_or_route_inference(self):
        result=self.read(make_elf())
        self.assertEqual(result['status'],'ELF_SYMBOLS_READ')
        self.assertEqual(result['matching_symbol_count'],1)
        self.assertEqual(result['symbols'][0]['name'],'flash_attention_infer_split_fd')
        self.assertTrue(result['symbols_are_not_abi_proof'])
        self.assertNotIn('stock_abi',result)

    def test_symbol_label_does_not_override_unchanged_registered_bytes(self):
        raw=make_elf();self.binary.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'ELF_BYTES_CHANGED'):
            reader.elf_symbols(self.binary,'0'*64)

    def test_nonelf_is_preserved_as_unparsed_not_searched_for_embedded_magic(self):
        result=self.read(b'opaque'+make_elf())
        self.assertEqual(result['status'],'NOT_ELF64_LITTLE_ENDIAN_NO_SPECULATIVE_EXTRACTION')

    def test_out_of_bounds_table_and_symbol_string_refuse(self):
        raw=bytearray(make_elf());struct.pack_into('<Q',raw,40,2**63)
        with self.assertRaisesRegex(ValueError,'SECTION_HEADER'):self.read(raw)
        raw=bytearray(make_elf());shoff=struct.unpack_from('<Q',raw,40)[0]
        symbol_offset=struct.unpack_from('<Q',raw,shoff+3*64+24)[0]
        struct.pack_into('<I',raw,symbol_offset+24,999999)
        with self.assertRaisesRegex(ValueError,'STRING_OFFSET'):self.read(raw)

    def test_metadata_key_match_without_binary_sha_is_not_a_join(self):
        (self.root/'wrong.json').write_text(json.dumps(dict(sha256='b'*64,kernelList=[{'tilingKey':'103'}])))
        result=reader.metadata_matches(self.root,'a'*64,{'103'})
        self.assertEqual(result['matches'],[])

    def test_matching_binary_metadata_keeps_missing_actual_entry_explicit(self):
        path=self.root/'actual.json';path.write_text(json.dumps(dict(sha256='a'*64,
            kernelList=[dict(tilingKey=999,kernelName='other')],parameters=[None,None])))
        result=reader.metadata_matches(self.root,'a'*64,{'103'})
        self.assertEqual(len(result['matches']),1)
        self.assertFalse(result['matches'][0]['entry_key_present'])
        self.assertEqual(result['matches'][0]['kernel_entries'],[])
        path.write_text(json.dumps(dict(sha256='a'*64,kernelList=[dict(tilingKey=103,kernelName='actual')],parameters=[])))
        result=reader.metadata_matches(self.root,'a'*64,{'103'})
        self.assertTrue(result['matches'][0]['entry_key_present'])

    def test_duplicate_metadata_keys_refuse(self):
        (self.root/'bad.json').write_text('{"sha256":"a","sha256":"b"}')
        with self.assertRaisesRegex(ValueError,'DUPLICATE_JSON_KEY'):
            reader.metadata_matches(self.root,'a'*64,{'103'})

    def test_only_named_source_files_are_read_and_missing_stock_is_not_inapplicable(self):
        path=self.root/reader.SOURCE_NAMES[0];path.parent.mkdir(parents=True)
        path.write_text('static void ComputeSeqLenAndDecodeFlagsForFAI() {\n  return;\n}\n')
        (self.root/'unrelated.bin').write_bytes(b'not source')
        result=reader.source_excerpt(self.root)
        self.assertEqual(len(result['files']),1)
        self.assertTrue(any(hit['occurrences'] for hit in result['files'][0]['hits']))
        self.assertEqual(reader.source_excerpt(None),{'status':'SOURCE_NOT_PRESENT','files':[]})
        self.assertNotIn('inapplicable',result)


if __name__ == '__main__':unittest.main()
