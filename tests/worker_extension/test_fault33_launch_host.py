import base64,copy,json,hashlib,unittest
from test_fault33_existing_extract import ExistingExtractTests
from tools.extract_fault33_launch_host import records,canonical

class LaunchHostTests(unittest.TestCase):
    def setUp(self):
        self.fixture=ExistingExtractTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        for c in self.fixture.plan['calls']:c.pop('tiling')
    def run_extract(self):return records(self.fixture.root,self.fixture.api,self.fixture.plan)
    def test_reassemble_whole_host_no_tiling_claim(self):
        result=self.run_extract();self.assertTrue(result[0]['n2_launch_complete'])
        self.assertFalse(result[0]['ready_for_control_recompute']);self.assertFalse(result[0]['field_extraction_complete'])
        for region in ('host','holders_opaque'):
            chunks=[r for r in result if r['kind']=='bytes' and r['row']==0 and r['region']==region]
            payload=b''.join(base64.b64decode(r['data']) for r in sorted(chunks,key=lambda r:r['offset']))
            self.assertEqual(hashlib.sha256(payload).hexdigest(),chunks[0]['region_sha256'])
        for r in result:
            self.assertLessEqual(len(canonical(r))+1,4096);v=dict(r);digest=v.pop('extract_sha256')
            self.assertEqual(hashlib.sha256(canonical(v)).hexdigest(),digest)
        self.assertEqual(result,self.run_extract())
    def test_invalid_relation_remains_blocked(self):
        self.fixture.plan['calls'][0]['pointers']['launch_relation']='/1/launch_relation'
        self.assertFalse(self.run_extract()[0]['n2_launch_complete'])
    def test_two_rows_not_six_rejected(self):
        self.fixture.plan['calls']=self.fixture.plan['calls'][:2]
        with self.assertRaisesRegex(ValueError,'THREE_SHAPES'):self.run_extract()
    def test_unknown_block_range_deferred_not_fabricated(self):
        self.fixture.plan['calls'][0]['pointers'].pop('block_table_valid_range')
        r=self.run_extract();self.assertTrue(r[0]['n2_launch_complete'])
        self.assertIsNone(r[1]['api']['block_table_valid_range']);self.assertIn('MISSING:block_table_valid_range',r[1]['deferred_issues'])
    def test_missing_core_field_blocks_transport(self):
        self.fixture.plan['calls'][0]['pointers'].pop('module')
        self.assertFalse(self.run_extract()[0]['n2_launch_complete'])
    def test_never_accept_manual_tiling(self):
        self.fixture.plan['calls'][0]['tiling']={'holder_index':0,'bytes':16}
        with self.assertRaisesRegex(ValueError,'MUST_NOT_DECLARE'):self.run_extract()
    def test_maximum_host_payload_chunked_without_truncation(self):
        from tools.review_launch_capture_v2 import HEADER
        path=self.fixture.root/self.fixture.plan['calls'][0]['capture']
        raw=path.read_bytes();header=list(HEADER.unpack_from(raw));header[5]=65536
        host=bytes(range(256))*256
        path.write_bytes(HEADER.pack(*header)+host+raw[-8:])
        result=self.run_extract();self.assertTrue(result[0]['n2_launch_complete'])
        chunks=[r for r in result if r['kind']=='bytes' and r['row']==0 and r['region']=='host']
        self.assertEqual(b''.join(base64.b64decode(r['data']) for r in chunks),host)
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(len(canonical(r))+1<=4096 for r in result))
