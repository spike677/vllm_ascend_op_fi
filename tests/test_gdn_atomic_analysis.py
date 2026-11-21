import unittest,torch
from experiments.worker_core_v11_20261118.analyze_gdn_atomic import pair_prediction
class AtomicEvidenceTest(unittest.TestCase):
    def test_distinguishes_write_add_and_unchanged(self):
        x=torch.tensor([[.25,-.5,2.,4.]],dtype=torch.bfloat16)
        initial=torch.full_like(x,1.25)
        atomic=(x.view(torch.float32)+initial.view(torch.float32)).view(torch.bfloat16)
        add=pair_prediction(x,atomic);normal=pair_prediction(x,x);untouched=pair_prediction(x,initial)
        self.assertEqual(add['atomic_exact'],2);self.assertLess(add['overwrite_exact'],2)
        self.assertEqual(normal['overwrite_exact'],2);self.assertLess(normal['atomic_exact'],2)
        self.assertEqual(untouched['unchanged'],2);self.assertLess(untouched['atomic_exact'],2)
    def test_never_changes_input(self):
        x=torch.tensor([[1,2]],dtype=torch.bfloat16);before=x.view(torch.uint8).clone()
        pair_prediction(x,x);self.assertTrue(torch.equal(before,x.view(torch.uint8)))
