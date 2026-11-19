import gzip
import hashlib
import io
import unittest
from tools.measure_oci_layers import measure

class MeasureTests(unittest.TestCase):
    def case(self):
        raw=b'abc'*1000000
        packed=gzip.compress(raw)
        return raw,packed,dict(mediaType='application/vnd.oci.image.layer.v1.tar+gzip',
            size=len(packed),digest='sha256:'+hashlib.sha256(packed).hexdigest())
    def test_measures_all_bytes_and_both_digests(self):
        raw,packed,row=self.case()
        measure(io.BytesIO(packed),row,'sha256:'+hashlib.sha256(raw).hexdigest())
        self.assertEqual(row['uncompressed_bytes'],len(raw));self.assertEqual(row['status'],'VERIFIED')
    def test_bad_digest_stops(self):
        raw,packed,row=self.case();row['digest']='sha256:'+'0'*64
        with self.assertRaisesRegex(ValueError,'MISMATCH'):measure(io.BytesIO(packed),row,'sha256:'+hashlib.sha256(raw).hexdigest())
        self.assertEqual(row['status'],'FAILED_STOP')
    def test_truncated_stream_stops(self):
        raw,packed,row=self.case()
        with self.assertRaises(EOFError):measure(io.BytesIO(packed[:-8]),row,'sha256:'+hashlib.sha256(raw).hexdigest())
        self.assertEqual(row['status'],'FAILED_STOP')
