import copy
import unittest
from tools.review_rpc_alignment import digest,review


def fixture():
    rows=[dict(rank=i,token='call-0',action='status',owner='owner-1',result={'owner':'owner-1'},result_sha256=digest({'owner':'owner-1'})) for i in (0,1)]
    return [dict(token='call-0',action='status',ranks=rows)],{i:[copy.deepcopy(rows[i])] for i in (0,1)}


class ReviewTests(unittest.TestCase):
    def test_exact(self):
        h,r=fixture();self.assertEqual(review(h,r)['rank_replies'],2)

    def test_stale_rank_even_equal_counts(self):
        h,r=fixture();h[0]['ranks'][1]['token']='previous'
        with self.assertRaisesRegex(ValueError,'STALE'):review(h,r)

    def test_owner_or_disk_mismatch(self):
        h,r=fixture();r[1][0]['owner']='old-owner'
        with self.assertRaisesRegex(ValueError,'DURABLE'):review(h,r)

    def test_digest_mismatch(self):
        h,r=fixture();h[0]['ranks'][0]['result']['owner']='other'
        with self.assertRaisesRegex(ValueError,'DIGEST'):review(h,r)

    def test_missing_or_duplicate(self):
        h,r=fixture();r[1]=[]
        with self.assertRaisesRegex(ValueError,'LENGTH'):review(h,r)
        h,r=fixture();h*=2;r={i:rows*2 for i,rows in r.items()}
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):review(h,r)
