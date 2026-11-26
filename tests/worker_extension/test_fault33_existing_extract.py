import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.extract_fault33_existing import extract, FIELDS, YELLOW_SOURCE
from tools.review_launch_capture_v2 import COUNTS, HEADER


class ExistingExtractTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.plan = dict(source_sha256=YELLOW_SOURCE, calls=[])
        api = []
        for rank in (0, 1):
            pid = rank + 10
            m = dict.fromkeys(COUNTS, 0)
            m.update(schema='dcat.launch.capture.v2', pid=pid, config_valid=True,
                     finalized=True, pending_writes=0, seen=3, selected=3, written=3)
            (self.root / f'capture-{pid}.json').write_text(json.dumps(m))
            for serial, n in enumerate((128, 512, 2304)):
                name = f'launch-{pid}-{serial}.bin'
                (self.root/name).write_bytes(HEADER.pack(0x444341544c415531, pid,
                    serial, 123, 2, 32, 1, 8, 0, 8) + bytes(range(32)) + struct.pack('<II', 0, 16))
                i = len(api)
                api.append(dict(rank=rank, prompt_length=n, api='npu_fused_infer_attention_score',
                    module='full_attention', phase='decode', q_shape=[1,12,256], k_shape=[8,128,512],
                    v_shape=[8,128,512], sparse_mode=3, mask={'present':False},
                    actual_seq_lengths=[1], kv_length=n+1, block_table_valid_range=[0,7],
                    launch_relation=dict(pid=pid, serial=serial, unique=True, method='recorded_correlation')))
                self.plan['calls'].append(dict(capture=name,
                    pointers={f:f'/{i}/{f}' for f in FIELDS},
                    tiling=dict(holder_index=0, bytes=16, boundary_evidence='fixture schema')))
        self.api = self.root/'api.json'
        self.api.write_text(json.dumps(api))

    def run_extract(self):
        return extract(self.root, self.api, self.plan)

    def test_complete_fields_never_grant_qualification(self):
        result = self.run_extract()
        self.assertTrue(result['field_extraction_complete'])
        self.assertFalse(result['qualification_granted'])
        self.assertFalse(result['serializer_validated'])
        self.assertEqual(result, self.run_extract())

    def test_name_hash_not_tiling_key(self):
        path = self.root/self.plan['calls'][0]['capture']
        data = bytearray(path.read_bytes())
        struct.pack_into('<Q', data, 24, (1 << 63) + 123)
        path.write_bytes(data)
        r = self.run_extract()
        self.assertIsNone(r['rows'][0]['launch_key'])
        self.assertFalse(r['field_extraction_complete'])

    def test_missing_fields_stay_null(self):
        del self.plan['calls'][0]['pointers']['kv_length']
        r = self.run_extract()
        self.assertIsNone(r['rows'][0]['api']['kv_length'])
        self.assertFalse(r['field_extraction_complete'])

    def test_no_nearest_serial_or_default_tiling(self):
        self.plan['calls'][0]['pointers']['launch_relation'] = '/1/launch_relation'
        self.plan['calls'][0].pop('tiling')
        r = self.run_extract()['rows'][0]
        self.assertIn('UNIQUE_API_LAUNCH_RELATION_UNPROVEN', r['issues'])
        self.assertIsNone(r['tiling']['bytes'])

    def test_boundary_overlap_rejected(self):
        self.plan['calls'][0]['tiling']['bytes'] = 33
        with self.assertRaisesRegex(ValueError, 'TILING_OUTSIDE'):
            self.run_extract()

    def test_incomplete_capture_rejected(self):
        (self.root/'launch-10-2.bin').unlink()
        self.plan['calls'][2]['capture'] = 'launch-10-1.bin'
        self.assertFalse(self.run_extract()['field_extraction_complete'])

    def test_unknown_source_rejected(self):
        self.plan['source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'UNREVIEWED'):
            self.run_extract()

    def test_duplicate_shape_rank_rejected(self):
        self.plan['calls'][1] = copy.deepcopy(self.plan['calls'][0])
        self.assertFalse(self.run_extract()['field_extraction_complete'])
