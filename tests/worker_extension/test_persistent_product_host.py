import unittest
from unittest.mock import patch
import test_external_persistent_runtime as runtime_fixture
from test_core_host import ExistingExecutor
from demoncat.runtime.vllm_ascend.worker_extension import external_persistent_runtime as v2
from demoncat.runtime.vllm_ascend.worker_extension.persistent_ingress import PersistentIngress,IngressRefused
from demoncat.runtime.vllm_ascend.worker_extension.persistent_product_host import PersistentProductClient


class RecoveryDouble(runtime_fixture.PersistentDouble):
    def query(self,**kw):
        state=super().query(**kw)
        return dict(state,owned_hooks=int(state['registered']),replay_pending=False,
                    recovery=dict(bytes_restored=state['leases']==0,
                                  resources_zero=state['leases']==0 and not state['registered']))


class PersistentHostTests(unittest.TestCase):
    setUp=runtime_fixture.PersistentRuntimeTests.setUp
    worker=runtime_fixture.PersistentRuntimeTests.worker
    def resolve(self,w,*a):return self.providers.setdefault(w,RecoveryDouble())
    def client(self,size=2):
        self.workers=[];self.providers={};self.canary_drift=False
        for rank in range(size):self.worker(rank)
        self.gate=PersistentIngress(validate_request=lambda r:None if r.get('tokens')==4 else 'TOKEN_RANGE',
            drain_backend=lambda **kw:dict(inflight=0,pending_submissions=0,aborts_completed=True))
        def boundary(owner,snapshot):
            self.assertFalse(snapshot['accepting']);self.assertTrue(snapshot['backend_drained'])
            return [dict(rank=r,owner=owner,inflight=0,device_readers_complete=True) for r in range(size)]
        def canary():
            self.assertFalse(self.gate.snapshot()['accepting'])
            self.assertEqual(self.gate.snapshot()['inflight'],1)
            return [dict(rank=r,all_finite=True,output={'hidden':'b' if self.canary_drift else 'a'}) for r in range(size)]
        c=PersistentProductClient(ExistingExecutor(self.workers),v2.PersistentFaultConfig('external','scale'),
            operator='trusted',ingress=self.gate,run_canary=canary,canary_request={'tokens':4},control_boundary=boundary)
        c.attach();return c

    def test_public_facade_two_owners_one_and_two_ranks(self):
        for size in (1,2):
            c=self.client(size)
            with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
                for owner in range(2):
                    if owner:
                        c=PersistentProductClient(c.executor,c.config,operator='trusted',ingress=self.gate,
                            run_canary=c.run_canary,canary_request=c.canary_request,control_boundary=c.control_boundary)
                        c.attach()
                    c.enable()
                    with self.gate.request({'tokens':4}):pass
                    self.assertTrue(c.status()['enabled'])
                    with self.assertRaisesRegex(IngressRefused,'TOKEN_RANGE'):
                        with self.gate.request({'tokens':5}):self.fail()
                    self.assertTrue(c.disable()['recovery']['restored'])
                    self.assertFalse(self.gate.snapshot()['accepting'])
                    c.detach();self.assertTrue(self.gate.snapshot()['accepting'])
                    self.assertTrue(all(not w._demoncat_worker_fault_runtime._leases for w in self.workers))
                    self.providers={}

    def test_partial_enable_rollback_all_ranks_and_gate_reopens_only_after_cleanup(self):
        c=self.client();bad=self.resolve(self.workers[1])
        def fail():raise RuntimeError('commit failure')
        bad.callbacks['enable']=fail
        with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
            with self.assertRaisesRegex(RuntimeError,'ALL_RANK_ENABLE'):c.enable()
        self.assertTrue(self.gate.snapshot()['accepting'])
        self.assertTrue(c.recovery['resources_zero']);self.assertFalse(c.recovery['restored'])
        self.assertTrue(all(not w._demoncat_worker_fault_runtime._leases for w in self.workers))

    def test_failed_rollback_keeps_ingress_closed_and_ownership(self):
        c=self.client();bad=self.resolve(self.workers[1]);bad.broken_restore=True
        def fail():raise RuntimeError('commit failure')
        bad.callbacks['enable']=fail
        with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
            with self.assertRaisesRegex(RuntimeError,'RECOVERY_REQUIRED'):c.enable()
        self.assertFalse(self.gate.snapshot()['accepting'])
        self.assertTrue(self.workers[1]._demoncat_worker_fault_runtime._leases)

    def test_canary_mismatch_does_not_release_exclusive_boundary(self):
        c=self.client()
        with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
            c.enable();self.canary_drift=True
            with self.assertRaisesRegex(RuntimeError,'RECOVERY_NOT_PROVEN'):c.disable()
        self.assertFalse(c.recovery['canary_exact']);self.assertFalse(self.gate.snapshot()['accepting'])
        with self.assertRaisesRegex(RuntimeError,'CONFIRMED_RECOVERY'):c.detach()

    def test_missing_rank_drain_cannot_enable(self):
        c=self.client();c.control_boundary=lambda *a:[]
        with self.assertRaisesRegex(RuntimeError,'DRAIN_UNPROVEN'):c.enable()
        self.assertFalse(self.gate.snapshot()['accepting'])
        self.assertFalse(self.providers)


if __name__=='__main__':unittest.main()
