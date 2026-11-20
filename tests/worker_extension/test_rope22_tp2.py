"""CPU real-dispatch guards, not simulated collective/NPU evidence."""
from dataclasses import replace
from unittest.mock import patch
import unittest
from test_rope22_task import Rope22TaskTests
from demoncat.runtime.vllm_ascend.worker_extension import rms18_tp2 as tp
from demoncat.runtime.vllm_ascend.worker_extension.rope22_tp2 import TP2Rope22Provider

class TP2Rope22Tests(Rope22TaskTests):
 def setUp(self):
  super().setUp();self.model.processor.use_all_gather=True
  self.enterContext(patch.object(tp.TP2Rms18Request,'_require_runner',return_value={'status':'CPU_FIXTURE'}))
  self.enterContext(patch.object(tp,'snapshot_tp2_fresh_prefill',side_effect=self.snapshot))
 def prepare_rank(self,rank):
  self.worker.rank=rank;self.ok(self.client.prepare());runtime=self.worker._demoncat_worker_fault_runtime
  old=runtime._leases['22.wrong_row'].provider;identity=old.identity;members=[dict(identity,rank=r) for r in range(2)];members[rank]=identity
  c=dict(owner='worker-group-cpu',group_generation=1,members=members,members_sha256=tp.digest(members),tp=2,mechanism='22.wrong_row',injection_policy='ONE_AUTOMATIC_RANK')
  self.worker._demoncat_natural_tensor_serving_bindings['22.wrong_row']=replace(old.binding,group_context=tp.TP2GroupContext(c,identity,object()))
  self.provider=TP2Rope22Provider(self.worker,self.model,identity);self.provider.bind(*old._ticket);runtime._leases['22.wrong_row'].provider=self.provider
  for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
  return c
 def role(self,c,selected):
  p=self.provider.group_proof();self.provider.commit_role(dict(group=c,selected_rank=selected,roles={r:'injector' if r==selected else 'guard' for r in (0,1)},normal_request_id=p['normal_request_id'],request_fingerprint=p['request_fingerprint']))
 def test_guard_rank_never_changes_reader(self):
  c=self.prepare_rank(1);self.enqueue('normal');self.role(c,0)
  with patch.object(self.provider,'admit',side_effect=AssertionError('guard write')):self.enqueue('fault')
  self.assertTrue(self.provider.rows[-1]['guard_boundary_unchanged']);self.assertFalse(self.provider.fault_applied)
  self.enqueue('off');self.enqueue('health');s=self.ok(self.client.finish());self.assertEqual(s['multi_rank']['role'],'guard');self.assertEqual((s['owned_hooks'],s['leases'],s['inflight']),(0,0,0))
 def test_auto_selected_rank1_is_supported(self):
  c=self.prepare_rank(1);self.enqueue('normal');self.role(c,1);self.enqueue('fault');self.assertTrue(self.provider.fault_applied)
  self.enqueue('off');self.enqueue('health');s=self.ok(self.client.finish());self.assertEqual(s['multi_rank']['role'],'injector');self.assertTrue(s['serving']['recovered'])
if __name__=='__main__':unittest.main()
