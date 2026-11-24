import json
from pathlib import Path
import tempfile
import unittest
from tools.review_launch_capture_v2 import review, HEADER, COUNTS


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = dict.fromkeys(COUNTS, 0)
        self.manifest.update(schema='dcat.launch.capture.v2', pid=12, config_valid=True,finalized=True,pending_writes=0)

    def test_unsealed_manifest_rejected(self):
        self.record(0);self.manifest.update(seen=1,selected=1,written=1,finalized=False);self.save()
        self.assertFalse(review(self.root,12)['complete'])

    def save(self):
        (self.root / 'capture-12.json').write_text(json.dumps(self.manifest))

    def record(self, serial):
        path = self.root / f'launch-12-{serial}.bin'
        path.write_bytes(HEADER.pack(0x444341544c415531, 12, serial, 123, 2, 8, 0, 16, 0, 0) + b'\0'*8)
        return path

    def test_complete_beyond_old_8192_limit(self):
        for serial in range(8193):
            self.record(serial)
        self.manifest.update(seen=8193, selected=8193, written=8193)
        self.save()
        r = review(self.root, 12)
        self.assertTrue(r['complete'])
        self.assertFalse(r['qualification_granted'])
        self.assertFalse(r['serializer_validated'])

    def test_missing_manifest_never_complete(self):
        self.record(0)
        self.assertFalse(review(self.root, 12)['complete'])

    def test_truncated_file_rejected(self):
        p=self.record(0);p.write_bytes(p.read_bytes()[:-1])
        self.manifest.update(seen=1,selected=1,written=1);self.save()
        self.assertFalse(review(self.root,12)['complete'])

    def test_drop_or_unknown_never_complete(self):
        self.record(0)
        for key in ('capacity_drop','oversize','open_error','map_drop','unknown'):
            self.manifest.update(seen=1,selected=1,written=1)
            self.manifest[key]=1;self.save()
            self.assertIn(key.upper(),review(self.root,12)['issues'])
            self.manifest[key]=0

    def test_filtered_scope_is_explicit(self):
        self.record(0);self.manifest.update(seen=4,selected=1,written=1,filtered=3);self.save()
        r=review(self.root,12)
        self.assertTrue(r['complete'])
        self.assertEqual(r['scope'],'ARMED_WINDOW_SELECTED_KEYS_ONLY')

    def test_empty_and_counter_mismatch_rejected(self):
        self.save();self.assertFalse(review(self.root,12)['complete'])
        self.record(0);self.manifest.update(seen=2,selected=1,written=1);self.save()
        self.assertIn('CENSUS_ACCOUNTING_MISMATCH',review(self.root,12)['issues'])

    def binding_record(self, source='entry', key=123, entry=123):
        p=self.record(0)
        p.write_bytes(HEADER.pack(0x444341544c415531,12,0,key,2,8,0,16,0,0)+b'\0'*8)
        Path(str(p)+'.binding.json').write_text(json.dumps(dict(
            schema='dcat.launch.binding.v1',pid=12,serial=0,key_source=source,
            entry=entry,function_name='kernel')))
        self.manifest.update(seen=1,selected=1,written=1,binding_metadata=True)
        self.save()
        return p

    def test_name_only_complete_does_not_invent_tiling_key(self):
        self.binding_record('name_only',0,None)
        r=review(self.root,12)
        self.assertTrue(r['complete'])
        self.assertIsNone(r['records'][0]['key'])
        self.assertEqual(r['name_only_records'],1)
        self.assertFalse(r['serializer_validated'])

    def test_binding_sidecar_required_for_new_capture(self):
        p=self.binding_record()
        Path(str(p)+'.binding.json').unlink()
        self.assertFalse(review(self.root,12)['complete'])

    def test_forged_entry_rejected(self):
        self.binding_record(entry=456)
        self.assertFalse(review(self.root,12)['complete'])

    def test_name_only_cannot_claim_real_key(self):
        self.binding_record('name_only',123,None)
        self.assertFalse(review(self.root,12)['complete'])
