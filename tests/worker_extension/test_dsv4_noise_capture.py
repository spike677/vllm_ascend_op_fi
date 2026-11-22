import unittest,importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import torch
spec=importlib.util.spec_from_file_location('noise_capture',Path(__file__).resolve().parents[2]/'experiments/worker_core_v11_20261118/dsv4_noise_capture.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Model:
    def compute_logits(self,x):return x+1
class CaptureTests(unittest.TestCase):
    def worker(self):return SimpleNamespace(rank=0,model_runner=SimpleNamespace(model=Model(),input_batch=SimpleNamespace(req_ids=['actual'])))
    def test_actual_four_calls_cleanup_and_values(self):
        w=self.worker();m.capture(w,'start')
        x=torch.arange(6.).reshape(1,6)
        for _ in range(4):self.assertTrue(torch.equal(w.model_runner.model.compute_logits(x),x+1))
        r=m.capture(w,'stop');self.assertEqual(len(r['steps']),4);self.assertEqual(r['steps'][0]['request_ids'],['actual'])
        self.assertNotIn('compute_logits',vars(w.model_runner.model));self.assertFalse(hasattr(w,'_dcat_noise_capture'))
    def test_nonfinite_fault_only_and_abort(self):
        w=self.worker();x=torch.tensor([[float('inf'),1,2,3,4,5.]])
        m.capture(w,'start')
        with self.assertRaisesRegex(RuntimeError,'NONFINITE'):w.model_runner.model.compute_logits(x)
        m.capture(w,'abort');m.capture(w,'start',phase='fault')
        for _ in range(4):w.model_runner.model.compute_logits(x)
        r=m.capture(w,'stop');self.assertEqual(r['steps'][0]['logits'][0],'+inf');json.dumps(r,allow_nan=False)
    def test_missing_observations_cleanup(self):
        w=self.worker();m.capture(w,'start')
        with self.assertRaisesRegex(RuntimeError,'FOUR_FORWARDS'):m.capture(w,'stop')
        self.assertFalse(hasattr(w,'_dcat_noise_capture'))
    def test_foreign_owner_not_removed(self):
        w=self.worker();m.capture(w,'start');other=lambda x:x;w.model_runner.model.compute_logits=other
        with self.assertRaisesRegex(RuntimeError,'OWNER_CHANGED'):m.capture(w,'abort')
        self.assertIs(w.model_runner.model.compute_logits,other)
if __name__=='__main__':unittest.main()
