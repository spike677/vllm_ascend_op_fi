import copy,json,struct,unittest
from pathlib import Path
from tools.decode_fault33_candidate import decode,fields,TYPES

ROOT=Path(__file__).resolve().parents[2]
KEY=5100000000010200203


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.layout=json.loads((ROOT/'docs/worker_core_v1/fault33_sdk_layout_candidate_20261124.json').read_text())
        self.meta=dict(numHeads=12,kvHeads=2,embeddingSize=256,embeddingSizeV=256,batch=1,blockSize=128,
                       maxQSeqlen=1,maxKvSeqlen=2305,sparseMode=3)
    def capture(self,meta=None):
        values=dict(self.meta if meta is None else meta,needCoreNum=10,totalSplitNodeNum=2)
        size,flat=fields(self.layout);offset=16;data=bytearray(offset+size)
        for f in flat:
            if f['path'] in values:struct.pack_into('<'+TYPES[f['type']],data,offset+f['offset'],values[f['path']])
        return struct.pack('<10Q',0x444341544c415531,1,0,KEY,10,len(data),1,8,0,8)+data+struct.pack('<II',0,offset)
    def run_decode(self,raw=None,layout=None,meta=None,**kw):
        return decode(raw or self.capture(),layout or self.layout,meta or self.meta,holder_index=kw.get('holder_index',0),allowed_keys=kw.get('allowed_keys',[KEY]))
    def test_three_synthetic_shapes_do_not_grant_actual_qualification(self):
        for length,batch in [(129,1),(513,2),(2305,1)]:
            m=dict(self.meta,maxKvSeqlen=length,batch=batch);r=self.run_decode(raw=self.capture(m),meta=m)
            self.assertTrue(r['shape_consistent']);self.assertFalse(r['serializer_validated']);self.assertFalse(r['qualification_granted'])
    def test_actual_sdk_alignment_and_nested_boundaries(self):
        size,flat=fields(self.layout);m={f['path']:f for f in flat}
        self.assertEqual((size,m['sparseMode']['offset'],m['coreInfo.startS2Idx']['offset']),(2488,144,512))
    def test_outside_holder_and_truncated_bytes_refused(self):
        with self.assertRaisesRegex(ValueError,'HEADER'):self.run_decode(raw=self.capture()[:-1])
        raw=bytearray(self.capture());struct.pack_into('<II',raw,len(raw)-8,0,len(raw))
        with self.assertRaisesRegex(ValueError,'OUTSIDE'):self.run_decode(raw=raw)
    def test_overlap_gap_and_unknown_type_refused(self):
        for change in [('offset',1),('type','mystery'),('bytes',9)]:
            layout=copy.deepcopy(self.layout);layout['classes'][0]['fields'][0][change[0]]=change[1]
            with self.assertRaises(ValueError):self.run_decode(layout=layout)
    def test_unknown_key_or_holder_refused(self):
        with self.assertRaisesRegex(ValueError,'KEY'):self.run_decode(allowed_keys=[123])
        with self.assertRaisesRegex(ValueError,'HOLDER'):self.run_decode(holder_index=1)
    def test_field_table_without_real_marker_roundtrip_refused(self):
        layout=copy.deepcopy(self.layout);layout.pop('marker_roundtrip')
        with self.assertRaisesRegex(ValueError,'MARKER_CHECK_REQUIRED'):self.run_decode(layout=layout)
    def test_metadata_mismatch_and_missing_not_filled_from_profile(self):
        r=self.run_decode(meta=dict(self.meta,sparseMode=0));self.assertFalse(r['shape_consistent'])
        self.assertEqual(r['issues'],['sparseMode'])
        m=dict(self.meta);del m['maxKvSeqlen']
        with self.assertRaisesRegex(ValueError,'MISSING'):self.run_decode(meta=m)


if __name__=='__main__':unittest.main()
