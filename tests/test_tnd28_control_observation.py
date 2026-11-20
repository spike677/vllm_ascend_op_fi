import os,unittest
from unittest.mock import patch
from experiments.worker_core_v11_20261118.tnd28_control_observation import ControlInputCapture
from demoncat.runtime.vllm_ascend.worker_extension.attention28_prepared_input import PreparedInputCapture

class ControlOwnershipTests(unittest.TestCase):
 def setUp(self):
  self.c=object.__new__(ControlInputCapture);self.c.control_env={};self.c.receipt={}
  self.enterContext(patch.dict(os.environ,{},clear=True))
 def test_normal_capture_still_rejects_control_subclass(self):
  from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus
  with self.assertRaisesRegex(Exception,'EXACT_SINGLE_CAPTURE'):StockTndPrefillCensus._validate_prepared_capture(object(),self.c,None)
 def test_foreign_policy_refused(self):
  os.environ['DCAT28_TND_MODE']='denominator_replay'
  with self.assertRaisesRegex(Exception,'FOREIGN_POLICY'):self.c._check_policy()
 def test_uncertain_native_stop_retains_control(self):
  self.c.control_env={'DCAT28_TND_MODE':'denominator_control'};os.environ.update(self.c.control_env)
  with patch.object(PreparedInputCapture,'stop',side_effect=RuntimeError('native pending')):
   with self.assertRaisesRegex(RuntimeError,'native pending'):self.c.stop()
  self.assertEqual(os.environ['DCAT28_TND_MODE'],'denominator_control');self.assertTrue(self.c.control_env)
 def test_confirmed_release_clears_only_owned_policy(self):
  self.c.control_env={'DCAT28_TND_MODE':'denominator_control'};os.environ.update(self.c.control_env);os.environ['OTHER']='kept'
  with patch.object(PreparedInputCapture,'stop'):self.c.stop()
  self.assertEqual(dict(os.environ),{'OTHER':'kept'});self.assertTrue(self.c.receipt['control_policy_released'])
if __name__=='__main__':unittest.main()
