import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import cache_pinned_oci_layers as C
class CacheTests(unittest.TestCase):
    def test_ordered_ranges_and_exact_digest(self):
        data=b'abcdefghijk';row=dict(size=len(data),digest='sha256:'+hashlib.sha256(data).hexdigest());seen=[]
        def read(digest,size,start,end):seen.append((start,end));return data[start:end+1]
        with tempfile.TemporaryDirectory() as folder,patch.object(C,'RANGE',3):
            p=Path(folder)/'one.part';result=C.transfer(row,p,read)
            self.assertTrue(result['verified']);self.assertEqual(p.read_bytes(),data)
            self.assertEqual(sorted(seen),[(0,2),(3,5),(6,8),(9,10)])
            with self.assertRaises(FileExistsError):C.transfer(row,p,read)
    def test_bad_digest_retains_partial_as_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.part'
            with self.assertRaisesRegex(ValueError,'DIGEST'):C.transfer(dict(size=3,digest='sha256:'+'0'*64),p,lambda *args:b'abc')
            self.assertEqual(p.read_bytes(),b'abc')
    def test_network_error_not_retried(self):
        calls=[]
        def fail(*args):calls.append(args);raise OSError('network')
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(OSError):C.transfer(dict(size=1,digest='sha256:'+'0'*64),Path(folder)/'bad.part',fail)
        self.assertEqual(len(calls),1)
