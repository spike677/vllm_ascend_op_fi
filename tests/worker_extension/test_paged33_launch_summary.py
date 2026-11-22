import tempfile,struct,unittest
from pathlib import Path
from tools.summarize_paged33_launch import summarize,FIELDS
class LaunchTests(unittest.TestCase):
    def record(self,p,serial=0,heads=12):
        args=bytearray(2960)
        d=dict(numHeads=heads,embeddingSize=256,embeddingSizeV=256,blockSize=128,kvHeads=2,batch=1,scaleValue=.0625,sparseMode=3,totalSplitNodeNum=2,needCoreNum=10,startN1=[0]*5+[1]*5,startS2=list(range(5))*2,endS2=list(range(1,6))*2,splitNum=[5,5])
        for k,(offset,fmt,n) in FIELDS.items():struct.pack_into('<'+fmt*n,args,296+offset,*([d[k]] if n==1 else d[k]))
        h=struct.pack('<10Q',0x444341544c415531,101,serial,5100000000010200203,10,2960,5,8,0,40)
        p.write_bytes(h+args+struct.pack('<II',280,296)+bytes(32))
    def test_no_pointers_exported_and_limit_is_visible(self):
        with tempfile.TemporaryDirectory() as d:
            self.record(Path(d)/'launch-101-0.bin');r=summarize(d,capture_limit=512)
            self.assertTrue(r['decoded_template_matches']);self.assertTrue(r['all_within_capture_limit'])
            self.assertNotIn('args',r);self.assertEqual(r['decode_launches'],1)
            self.assertFalse(summarize(d,capture_limit=1)['all_within_capture_limit'])
    def test_changed_fields_and_truncation_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'launch-101-0.bin';self.record(p,heads=8)
            self.assertFalse(summarize(d,capture_limit=512)['decoded_template_matches'])
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'INVALID'):summarize(d,capture_limit=512)
if __name__=='__main__':unittest.main()
