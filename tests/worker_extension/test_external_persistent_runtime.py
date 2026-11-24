"""The v2 route uses the real Runtime and HostAdapter, including failures."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime
from demoncat.runtime.vllm_ascend.worker_extension import external_persistent_runtime as v2
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from test_core_host import ExistingExecutor
from test_extension import _extended_worker,ReadbackProvider


def factory(*a,**kw):raise AssertionError('registration is inert')


class PersistentDouble(ReadbackProvider):
    persistent_protocol_version=2
    region_kind='persistent_weight'
    def dispatch(self,action,settings,*,owner,generation):
        result=super().dispatch(action,settings,owner=owner,generation=generation)
        if action in ('register','check','enable'):
            self.state.update(leases=1,enabled=action=='enable')
        if action in ('disable','unregister') and getattr(self,'broken_restore',False):
            self.state['leases']=1
        return result
    def query(self,**kw):
        return dict(super().query(**kw),persistent_protocol_version=2,
                    persistent_leases=self.state['leases'],device_readers_complete=True,
                    exclusive_ingress_paused=True)


class PersistentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.workers=[];self.providers={}
        self.authority=SimpleNamespace(source='BLUE_COMBINATION',authorize_operator=lambda p,a:p=='trusted')
        self.actual={'tp':2,'config':'a'*64}
        for p in [patch.object(identity,'_current_device',return_value=0),
                  patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=self.authority),
                  patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.observe_deployment',return_value=self.actual)]:
            p.start();self.addCleanup(p.stop)
        self.spec=v2.PersistentSite('external.scale','persistent_weight',factory,v2.factory_sha256(factory),self.actual)
    def worker(self,rank):
        w=_extended_worker(rank=rank);w.load_model()
        v2.install_persistent_runtime(w,authority=self.authority,operator='trusted')
        v2.register_persistent_site(w,self.spec,authority=self.authority,operator='trusted')
        self.workers.append(w);return w
    def resolve(self,w,*args):return self.providers.setdefault(w,PersistentDouble())
    def client(self):
        return ProductClient(ExistingExecutor(self.workers),v2.PersistentFaultConfig('external','scale'),
            profile='trusted-example',operator='trusted',run_canary=lambda:None,drain=lambda:None)
    def test_two_owners_world_size_one_and_two_real_runtime(self):
        for size in (1,2):
            self.workers=[];self.providers={}
            for rank in range(size):self.worker(rank)
            with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
                for unused in range(2):
                    c=self.client();c.attach()
                    for action in ('prepare','register','check','enable'):c._all(action)
                    states=c._all('query')
                    self.assertTrue(all(x['response']['provider_state']['leases']==1 for x in states))
                    self.assertTrue(all(x['response']['provider_state']['enabled'] for x in states))
                    self.assertTrue(all(x['response']['recovery_required'] is False for x in states))
                    self.assertTrue(all(x['response']['quarantined'] is False for x in states))
                    for action in ('disable','unregister','detach'):c._all(action)
                    self.assertTrue(all(not w._demoncat_worker_fault_runtime._leases for w in self.workers))
    def test_partial_commit_failure_keeps_owner_until_recovery(self):
        self.worker(0);w=self.worker(1)
        bad=self.providers.setdefault(w,PersistentDouble())
        def fail():raise RuntimeError('device write failure')
        bad.callbacks['enable']=fail
        with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
            c=self.client();c.attach()
            for action in ('prepare','register','check'):c._all(action)
            with self.assertRaisesRegex(RuntimeError,'ALL_RANK_ENABLE'):c._all('enable')
            self.assertTrue(all(w._demoncat_worker_fault_runtime._leases for w in self.workers))
            c._all('detach')
            self.assertTrue(all(not w._demoncat_worker_fault_runtime._leases for w in self.workers))
    def test_nonzero_cleanup_cannot_release_owner(self):
        w=self.worker(0)
        with patch.object(v2,'create_persistent_provider',side_effect=self.resolve):
            c=self.client();c.attach()
            for action in ('prepare','register','check','enable'):c._all(action)
            self.providers[w].broken_restore=True
            with self.assertRaises(RuntimeError):c._all('disable')
            with self.assertRaises(RuntimeError):c._all('detach')
            self.assertTrue(w._demoncat_worker_fault_runtime._leases)
            self.providers[w].broken_restore=False;c._all('detach')
    def test_legacy_provider_delegates_without_v2_quiescence(self):
        w=self.worker(0);p=ReadbackProvider()
        with patch.object(runtime,'_resolve_provider',return_value=p):
            c=WorkerFaultClient(HostAdapter(ExistingExecutor([w])),FaultConfig(20,'buffer_substitute'))
            for a in ('prepare','register','check','enable','disable','unregister'):
                self.assertEqual(getattr(c,a)()['status'],'ACTION_COMPLETED')
            self.assertEqual(c.detach()['status'],'DETACHED')
    def test_bootstrap_after_first_rpc_and_unauthorized_refused(self):
        w=_extended_worker(rank=0);w.load_model();w.demoncat_worker_identity()
        with self.assertRaisesRegex(RuntimeError,'FIRST_RPC'):
            v2.install_persistent_runtime(w,authority=self.authority,operator='trusted')
        w=_extended_worker(rank=0);w.load_model()
        with self.assertRaisesRegex(RuntimeError,'TRUSTED'):
            v2.install_persistent_runtime(w,authority=self.authority,operator='user')
    def test_registration_and_resolution_identity_gates(self):
        w=self.worker(0)
        with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.deployment_acceptance_report',return_value={'can_prepare':False,'acceptance_permit':None}):
            with self.assertRaisesRegex(RuntimeError,'PERMIT_REQUIRED'):
                v2.create_persistent_provider(w,w.model_runner.model,{},self.spec.mechanism)
        self.actual['tp']=1
        with self.assertRaisesRegex(RuntimeError,'IDENTITY_MISMATCH'):
            v2.create_persistent_provider(w,w.model_runner.model,{},self.spec.mechanism)
    def test_shared_method_identity_and_config_not_public(self):
        self.assertIs(v2.PersistentWorkerRuntime._detach,runtime.WorkerFaultRuntime._detach)
        self.assertIs(v2.PersistentWorkerRuntime._fresh,runtime.WorkerFaultRuntime._fresh)
        with self.assertRaises(ValueError):FaultConfig.from_dict({'fault_id':'external','variant':'scale'})


if __name__=='__main__':unittest.main()
