import unittest,torch
from experiments.worker_core_v11_20261118.diagnose_b8_nan import stats
class StatsTests(unittest.TestCase):
    def test_actual_nonfinite_and_no_change(self):
        x=torch.tensor([[1,float('nan')],[float('inf'),2]],dtype=torch.bfloat16);before=x.view(torch.uint8).clone()
        row=stats(x)[0];self.assertEqual((row['nan'],row['inf'],row['nonfinite_per_first_row']),(1,1,[1,1]))
        self.assertTrue(torch.equal(before,x.view(torch.uint8)))
    def test_empty_and_integer(self):
        row=stats((torch.empty(0,4),torch.tensor([1,3])));self.assertTrue(row[0]['finite']);self.assertEqual(row[1]['values'],[1,3])
if __name__=='__main__':unittest.main()
