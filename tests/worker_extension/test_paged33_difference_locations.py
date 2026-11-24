import unittest
import torch
from tools.paged33_control_checkpoint import locate_differences


class DifferenceTests(unittest.TestCase):
    def test_all_coordinates_and_bits_are_preserved(self):
        n=torch.zeros(1,2,3,dtype=torch.bfloat16);c=n.clone()
        c[0,0,1]=-0.;c[0,1,2]=2.
        r=locate_differences(c,n,[[0,8],[8,13]])
        self.assertEqual(r['count'],2);self.assertTrue(r['complete'])
        self.assertEqual([(v['query'],v['head'],v['column']) for v in r['differences']],[(0,0,1),(0,1,2)])
        self.assertEqual(r['differences'][0]['control_bits'],32768)
        self.assertIsNone(r['single_partition_attribution'])
    def test_no_differences_is_valid(self):
        n=torch.ones(1,2,3,dtype=torch.bfloat16)
        self.assertEqual(locate_differences(n,n,[[0,2]])['differences'],[])
    def test_nonfinite_and_unknown_layout_are_rejected(self):
        n=torch.zeros(1,2,3,dtype=torch.bfloat16);c=n.clone();c[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE'):locate_differences(c,n,[[0,2]])
        with self.assertRaisesRegex(ValueError,'LAYOUT'):locate_differences(n.float(),n,[[0,2]])


if __name__=='__main__':unittest.main()
