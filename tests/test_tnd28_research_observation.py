import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from experiments.worker_core_v11_20261118 import tnd28_research_observation as R
class ResearchObservationTests(unittest.TestCase):
 def setUp(self):
  self.identity={'rank':0};self.worker=NS(model_runner=NS(model=object()),demoncat_worker_identity=lambda:{'worker':self.identity})
  self.p=NS(index=0,phases=('observe','fault'),dispatch=lambda *a,**k:{},query=lambda **k:dict(registered=False,owned_hooks=0,owned_api_wrappers=0,inflight=0,leases=0))
 def call(self,action):
  with patch.object(R,'actual',lambda w:w),patch('demoncat.runtime.vllm_ascend.worker_extension.registry.create_research_provider',return_value=self.p):return R.command(self.worker,self.identity,'a'*32,action,'req')
 def test_one_observe_then_reject(self):
  self.call('prepare');self.assertEqual(self.call('advance')['status'],'ACTION_COMPLETED');self.p.index=1
  with self.assertRaisesRegex(RuntimeError,'EXACTLY_ONE_OBSERVE'):self.call('advance')
 def test_no_fault_action(self):
  self.call('prepare')
  with self.assertRaisesRegex(RuntimeError,'ZERO_F_ACTION'):self.call('fault')
 def test_no_coexisting_public_owner(self):
  self.worker._demoncat_worker_fault_runtime=NS(_tenants={'active':1})
  with self.assertRaisesRegex(RuntimeError,'ORDINARY_RUNTIME'):self.call('prepare')
 def test_detach_refuses_live_resource(self):
  self.call('prepare');self.p.query=lambda **k:dict(registered=False,owned_hooks=1,owned_api_wrappers=0,inflight=0,leases=0)
  with self.assertRaisesRegex(RuntimeError,'RESOURCES'):self.call('detach')
  self.assertTrue(hasattr(self.worker,R.ATTRIBUTE))
 def test_clean_detach(self):
  self.call('prepare');self.assertEqual(self.call('detach')['status'],'DETACHED');self.assertFalse(hasattr(self.worker,R.ATTRIBUTE))
if __name__=='__main__':unittest.main()
