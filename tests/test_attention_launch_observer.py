"""CPU contracts for raw runtime evidence, never a substitute for real launch."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from fault_injection.native_adapters.attention28.launch import verify_launch, decode_raw


def fixture(root):
    call_id = 'a'*32
    binary = b'CPU fixture only, not an ELF'
    (root/'registered-41-1.bin').write_bytes(binary)
    raw = struct.pack('<II', 512, 256)
    schema = dict(size=8, fields=[dict(name='innerSplitParams.mBaseSize', offset=0, width=4, count=1, dtype='uint32_t'),
        dict(name='innerSplitParams.s2BaseSize', offset=4, width=4, count=1, dtype='uint32_t')])
    host = dict(pid=41, tid=42, token=call_id, raw_hex=raw.hex(), raw_size=8,
                tiling_key='103000000010200003', block_dim=16, schedule_mode=0,
                typed={'innerSplitParams.mBaseSize':512, 'innerSplitParams.s2BaseSize':256})
    common = dict(schema='demoncat.attention.runtime_observer.v1', pid=41, tid=42)
    records = [dict(common, event='register', monotonic_ns=1, call_id='', handle=7, generation=1,
                    binary_file='registered-41-1.bin', binary_bytes=len(binary), result=0),
        dict(common, event='begin', monotonic_ns=2, call_id=call_id),
        dict(common, event='launch', monotonic_ns=5, call_id=call_id, launch_begin_ns=3, launch_end_ns=4,
            handle=7, generation=1, binary_file='registered-41-1.bin', tiling_key=host['tiling_key'], block_dim=16,
            schedule_mode=0, stream=10, result=0, snapshot_valid=True, observer_mutates_arguments=False,
            args_hex=(bytes(16)+raw).hex(), args_size=24, has_tiling=1, no_h2d_copy=0,
            tiling_addr_offset=8, tiling_data_offset=16, host_input_count=0, host_input_hex=''),
        dict(common, event='device_sync', monotonic_ns=6, call_id=call_id, result=0),
        dict(common, event='end', monotonic_ns=7, call_id=call_id)]
    return call_id, hashlib.sha256(binary).hexdigest(), schema, host, records


class RuntimeLaunchTests(unittest.TestCase):
    def verify(self, root, fixture):
        cid, binary, schema, host, rows = fixture
        (root/'runtime.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return verify_launch(root, cid, kernel_sha256=binary, schema=schema, host=host)

    def test_final_runtime_raw_and_registered_bytes_are_both_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = fixture(root)
            result = self.verify(root, f)
            self.assertTrue(result['kernel_argument_raw_observed'])
            self.assertTrue(result['device_completion_observed'])
            self.assertFalse(result['c2'])
            self.assertEqual(result['statistics'], 'NOT_READ')
            f[4][2]['args_hex'] = (bytes(16)+struct.pack('<II',512,512)).hex()
            with self.assertRaisesRegex(RuntimeError, 'HOST_FINAL_ARGUMENT_BYTES_DIFFER'):
                self.verify(root, f)

    def test_do_not_search_for_host_bytes_at_a_different_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); f=fixture(root); launch=f[4][2]
            launch['args_hex'] = (bytes(16)+struct.pack('<II',512,512)+bytes.fromhex(f[3]['raw_hex'])).hex()
            launch['args_size'] = 32
            with self.assertRaisesRegex(RuntimeError, 'HOST_FINAL_ARGUMENT_BYTES_DIFFER'):
                self.verify(root, f)

    def test_host_logs_alone_or_ambiguous_launches_cannot_pass(self):
        for case in ('missing', 'duplicate', 'sync', 'sync_error', 'owner', 'key', 'schedule', 'handle', 'generation', 'overflow'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); f=fixture(root); rows=f[4]
                if case=='missing': del rows[2]
                if case=='duplicate': rows.insert(3,copy.deepcopy(rows[2]))
                if case=='sync': del rows[3]
                if case=='sync_error': rows[3]['result']=1
                if case=='owner': rows[2]['tid']=99
                if case=='key': rows[2]['tiling_key']='33'
                if case=='schedule': rows[2]['schedule_mode']=-1
                if case=='handle': rows[2]['handle']=8
                if case=='generation': rows[2]['generation']=2
                if case=='overflow': rows.append(dict(event='capture_overflow'))
                with self.assertRaises(RuntimeError):self.verify(root,f)

    def test_no_h2d_device_pointer_and_host_input_overlap_refuse(self):
        for case in ('no_h2d', 'host_input', 'tiling_pointer'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);f=fixture(root);launch=f[4][2]
                if case=='no_h2d':launch['no_h2d_copy']=1
                if case=='host_input':
                    launch['host_input_count']=1;launch['host_input_hex']=struct.pack('<II',0,20).hex()
                if case=='tiling_pointer':launch['tiling_addr_offset']=20
                with self.assertRaises(RuntimeError):self.verify(root,f)

    def test_old_or_changed_registered_kernel_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=fixture(root)
            (root/'registered-41-1.bin').write_bytes(b'old binary')
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_REGISTERED_BINARY_DIFFERS'):
                self.verify(root,f)

    def test_public_schema_bounds_and_overlapping_fields_fail(self):
        schema=dict(size=8,fields=[dict(name='x',offset=0,width=4,count=1,dtype='uint32_t')])
        schema['fields'].append(dict(name='y',offset=2,width=4,count=1,dtype='uint32_t'))
        with self.assertRaisesRegex(RuntimeError,'OVERLAP'):decode_raw(bytes(8),schema)
        schema['fields'][1]['offset']=8
        with self.assertRaisesRegex(RuntimeError,'OUT_OF_BOUNDS'):decode_raw(bytes(8),schema)


if __name__=='__main__':unittest.main()
