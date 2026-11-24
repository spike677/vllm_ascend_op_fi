import json,struct,tempfile,unittest
from pathlib import Path
from tools.paged33_launch_inventory import inventory
from tools.paged33_control_checkpoint import validate_spans
from tools.build_rc3_preview import EXACT,VERSIONS


class R6PackagingTests(unittest.TestCase):
    def test_unknown_launch_keeps_s2_null(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'launch-1-0.bin';p.write_bytes(struct.pack('<10Q',0x444341544c415531,1,0,42,24,4,0,8,0,0)+b'abcd')
            r=inventory(d);self.assertTrue(r['capture_complete']);self.assertIsNone(r['variants'][0]['s2_grouping'])
            self.assertFalse(r['qualification_granted'])
            self.assertFalse(inventory(d,1)['capture_complete'])
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'INVALID'):inventory(d)

    def test_key_set_must_be_complete_once(self):
        validate_spans([[0,512],[512,777]],777)
        for spans in ([[0,512],[511,777]],[[1,777]],[[0,512],[513,777]],[],[[0,777.0]]):
            with self.assertRaises(ValueError):validate_spans(spans,777)

    def test_observation_sources_packaged_and_versions_bumped(self):
        self.assertIn('experiments/worker_core_v11_20261118/paged33_launch_capture.cpp',EXACT)
        self.assertIn('experiments/worker_core_v11_20261118/paged33_native_probe.py',EXACT)
        self.assertEqual(set(VERSIONS.values()),{'0.4.0rc6.post5','0.3.0rc6.post5'})

    def test_frozen_r5_reviewer_is_still_exact(self):
        import hashlib
        raw=Path('tools/_r5/review_continuous_public.py').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'deb3c5c0d600f7bcae7d0997cfd883591ece87231ed8128e486edb5d823ffadc')
