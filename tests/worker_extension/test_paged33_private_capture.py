import tempfile,unittest
from pathlib import Path
import torch
from tools.paged33_private_capture import PrivateCapture,unpack_actual


def actual(length):
    block=128;pages=(length+block-1)//block
    a=dict(query=torch.zeros(1,4,8,dtype=torch.bfloat16),key=torch.arange((pages+1)*block*2*8).reshape(pages+1,block,2,8).bfloat16(),
           num_heads=4,num_key_value_heads=2,scale=.25,block_size=block,block_table=torch.arange(1,pages+1).reshape(1,-1),
           actual_seq_lengths_kv=[length],input_layout='TND',sparse_mode=3)
    a['value']=a['key'].clone();return a,(a['query'].clone(),None)


class CaptureTests(unittest.TestCase):
    def test_real_page_gather_and_tail(self):
        a,o=actual(129);d=unpack_actual(a,o,['r'])
        self.assertEqual(d['k'].shape,(129,2,8));self.assertTrue(torch.equal(d['k'][0],a['key'][1,0]))
        self.assertTrue(torch.equal(d['k'][-1],a['key'][2,0]));self.assertEqual(d['metadata']['physical_pages'],[1,2])
    def test_three_shapes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            c=PrivateCapture(Path(root)/'new',rank=0,lengths=[129,513,2305])
            self.assertFalse(c.finish()['complete'])
            for length in [129,513,2305,129]:
                a,o=actual(length);c.observe(a,o,['r'+str(length)])
            self.assertTrue(c.finish()['complete']);self.assertEqual(len(c.saved),3)
    def test_mask_duplicate_page_and_wrong_request_refused(self):
        a,o=actual(129);a['atten_mask']=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'MASK'):unpack_actual(a,o,['r'])
        a,o=actual(129);a['block_table'][:]=1
        with self.assertRaisesRegex(ValueError,'PAGE'):unpack_actual(a,o,['r'])
        a,o=actual(129)
        with self.assertRaisesRegex(ValueError,'REQUEST'):unpack_actual(a,o,['r','s'])
