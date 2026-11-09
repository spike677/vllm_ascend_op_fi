"""Request coverage and last-position/logit sensitivity of complete comparison."""
import unittest
import numpy as np
from compare_model import compare_values


class NumericalTests(unittest.TestCase):
    def test_difference_in_last_request_is_observed(self):
        a=np.zeros((8,2,4,3),dtype=np.float16);b=a.copy();b[7,0,1,1]=1
        rows=compare_values(a,b,'attention')
        self.assertEqual([r['changed_elements'] for r in rows],[0]*7+[1])
        self.assertEqual(rows[7]['first_changed_query'],1)
        self.assertEqual(rows[7]['last_query_changed'],0)

    def test_last_query_difference_is_distinct_from_earlier_queries(self):
        a=np.zeros((1,2,4,3),dtype=np.float16);b=a.copy();b[0,1,3,0]=1
        self.assertEqual(compare_values(a,b,'attention')[0]['last_query_changed'],1)

    def test_logit_difference_with_identical_top1_is_retained(self):
        a=np.array([[[10,1,0]]],dtype=np.float16);b=a.copy();b[0,0,1]=1.125
        row=compare_values(a,b,'logits')[0]
        self.assertEqual(row['changed_elements'],1)
        self.assertEqual(row['B_top1'],row['F_top1'])

    def test_no_difference_is_a_valid_result(self):
        a=np.zeros((1,2,4,3),dtype=np.float16)
        row=compare_values(a,a.copy(),'attention')[0]
        self.assertEqual(row['changed_elements'],0)
        self.assertIsNone(row['first_changed_query'])

    def test_shape_mismatch_cannot_be_compared(self):
        with self.assertRaises(AssertionError):
            compare_values(np.zeros((1,1)),np.zeros((8,1)),'logits')


if __name__=='__main__':unittest.main()
