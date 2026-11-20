"""CPU scheduler/native doubles: rank isolation and lifecycle, not NPU evidence."""
from dataclasses import replace
from unittest.mock import patch
import unittest
from test_output39_task import Output39TaskTests
from demoncat.runtime.vllm_ascend.worker_extension import output39_tp2 as T
from demoncat.runtime.vllm_ascend.worker_extension.rms18_tp2 import TP2GroupContext,digest

class TP2Output39Tests(unittest.TestCase):
    snapshot=Output39TaskTests.snapshot
    root=Output39TaskTests.root
    forward=Output39TaskTests.forward
    dispose=Output39TaskTests.dispose
    ok=Output39TaskTests.ok
    enqueue=Output39TaskTests.enqueue
    def setUp(self):
        Output39TaskTests.setUp(self);self.model.processor.use_all_gather=True
        self.enterContext(patch.object(T.TP2Output39Session,'_require_runner',return_value={'status':'CPU_DOUBLE'}))
        self.enterContext(patch.object(T,'snapshot_tp2_fresh_prefill',side_effect=self.snapshot))
    def start(self,rank,selected):
        self.worker.rank=rank;self.ok(self.client.prepare());rt=self.worker._demoncat_worker_fault_runtime
        old=rt._leases['39.output_proxy'].provider;identity=old.identity;members=[dict(identity,rank=r) for r in (0,1)]
        c=dict(owner='worker-group-cpu',group_generation=1,members=members,members_sha256=digest(members),tp=2,mechanism='39.output_proxy',injection_policy='ONE_AUTOMATIC_RANK')
        self.worker._demoncat_output39_serving_binding=replace(old.binding,group_context=TP2GroupContext(c,identity,object()))
        self.provider=T.TP2Output39Provider(self.worker,self.model,identity);self.provider.bind(*old._ticket);rt._leases['39.output_proxy'].provider=self.provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
        self.enqueue(0);p=self.provider.group_proof()
        self.provider.commit_role(dict(group=c,selected_rank=selected,roles={r:'injector' if r==selected else 'guard' for r in (0,1)},normal_request_id=p['normal_request_id'],request_fingerprint=p['request_fingerprint']))
    def finish(self):
        self.enqueue(2);self.enqueue(3)
        self.assertTrue(self.provider.complete)
        self.assertEqual(self.provider.phase,'complete')
        s=self.ok(self.client.finish())
        self.assertEqual((s['owned_hooks'],s['leases'],s['inflight'],s['replay_pending']),(0,0,0,False))
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
    def test_guard_never_opens_fault_lease(self):
        self.start(1,0);count=len(self.transport.opened)
        with patch.object(self.provider.p,'_apply_single_return',side_effect=AssertionError('guard wrote')):self.enqueue(1)
        r=self.provider.rows[-1]['report'];self.assertTrue(r['guard_boundary_unchanged']);self.assertEqual(r['applied'],0);self.assertIsNone(r['native_lease_receipt']);self.assertEqual(len(self.transport.opened),count);self.finish()
    def test_rank1_injector_recovers(self):
        self.start(1,1);self.enqueue(1);r=self.provider.rows[-1]['report'];self.assertEqual(r['applied'],1);self.assertEqual(r['effect']['changed_elements'],1);self.finish()
    def test_guard_runs_off_health_after_peer_failure(self):
        self.start(1,0);self.enqueue(1);self.assertFalse(self.provider.p._fault_attempted)
        self.ok(self.client.recover());self.assertEqual(self.provider.phase,'off');self.assertFalse(self.provider.complete)
        self.finish()
    def test_cannot_recommit_role(self):
        self.start(0,0)
        with self.assertRaisesRegex(Exception,'COMMIT_ONCE'):self.provider.commit_role(self.provider.binding.group_context.decision)
        self.enqueue(1);self.finish()

if __name__=='__main__':unittest.main()
