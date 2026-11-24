import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from demoncat.runtime.vllm_ascend.worker_extension.extension import DemoncatWorkerExtension
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension import product_binding as pb
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused


class ProductRejectionRecoveryTests(unittest.TestCase):
    def client(self, size=2):
        c = ProductClient(Mock(), FaultConfig(18, 'api_half'), profile='norm',
                          operator='trusted', run_canary=Mock(), drain=Mock())
        c.members = [dict(rank=i) for i in range(size)]
        c.clients = [Mock() for _ in c.members]
        for client in c.clients:
            client.snapshot.return_value = dict(attachment={'ticket': 1}, ownership_retained=True)
            client.detach.return_value = dict(status='DETACHED', ownership_retained=False)
        c._binding = Mock(return_value=[dict(rank=i, status='PRODUCT_BINDING_ROLLED_BACK', resources_zero=True) for i in range(size)])
        return c

    def test_unknown_profile_returns_refusal_without_escaping_worker_rpc(self):
        worker = NS(model_runner=NS(model=object()), demoncat_worker_identity=lambda: {'worker': {'rank': 0}})
        with patch.object(pb, 'authority_for', return_value=NS(authorize_operator=lambda *a: True)):
            for _ in range(2):
                row = DemoncatWorkerExtension.demoncat_product_binding(worker, 'capabilities', {'profile': 'unknown', 'operator': 'trusted'})
                self.assertEqual(row['status'], 'PRODUCT_BINDING_REFUSED')
                self.assertEqual(row['reason'], 'PRODUCT_PROFILE_UNKNOWN')
                self.assertEqual(row['exit_code'], 4)
                self.assertEqual(row['rank'], 0)

    def test_unexpected_control_error_is_not_a_contract_refusal(self):
        worker = NS(demoncat_worker_identity=lambda: {'worker': {'rank': 1}})
        with patch.object(pb, 'product_binding_rpc', side_effect=RuntimeError('unexpected')):
            row = DemoncatWorkerExtension.demoncat_product_binding(worker, 'attach', {})
        self.assertEqual(row['exit_code'], 5)
        self.assertTrue(row['recovery_required'])

    def test_missing_permit_and_partial_prepare_cleanup_every_rank(self):
        for size in (1, 2):
            for stage in ('prepare', 'register', 'check', 'enable'):
                c = self.client(size)
                def fail(scope, stage=stage):
                    c.bound = True
                    raise ValueError('PRODUCT_ALL_RANK_' + stage.upper() + '_FAILED:PROVIDER_UNAVAILABLE')
                c._enable = fail
                with self.assertRaisesRegex(ValueError, 'PROVIDER_UNAVAILABLE'):
                    c.enable()
                self.assertFalse(c.bound)
                self.assertTrue(c.recovery['resources_zero'])
                for client in c.clients: client.detach.assert_called_once()
                self.assertFalse(c.recovery['restored'])  # Not a successful canary acceptance.

    def test_cleanup_failure_retains_responsibility_and_attempts_other_ranks(self):
        c = self.client()
        c.bound = True
        c.clients[0].detach.side_effect = RuntimeError('unreachable')
        with self.assertRaisesRegex(RuntimeError, 'RUNTIME_CLEANUP_INCOMPLETE'):
            c._rollback_enable()
        c.clients[1].detach.assert_called_once()
        self.assertTrue(c.bound)
        c._binding.assert_not_called()

    def test_reenable_never_tears_down_active_scope(self):
        c = self.client(); c.bound = c.enabled = True
        with self.assertRaisesRegex(ValueError, 'NEW_ATTACHMENT_REQUIRED'): c.enable()
        for client in c.clients: client.detach.assert_not_called()
        self.assertTrue(c.enabled)

    def test_binding_rollback_never_removes_other_owner(self):
        worker = NS(model_runner=NS(model=object()), _demoncat_product_profiles={
            'norm': dict(mechanism='18.api_half')})
        with patch.object(pb, 'authority_for', return_value=NS(authorize_operator=lambda *a: True)), \
             patch.object(pb, 'current_binding', return_value=NS(group={'owner': 'other'})), \
             patch.object(pb, 'remove_continuous_binding') as remove:
            with self.assertRaisesRegex(ContextRefused, 'OWNER_MISMATCH'):
                pb.product_binding_rpc(worker, 'rollback', dict(operator='trusted', profile='norm', owner='mine'))
            remove.assert_not_called()

    def test_actual_runtime_partial_failure_releases_provider_and_tenant(self):
        import os, threading
        from test_extension import _extended_worker
        from test_core_host import ExistingExecutor
        from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ContinuousBinding, ATTRIBUTE
        from demoncat.runtime.vllm_ascend.worker_extension.providers.product import ProductProvider
        class Observer:
            def __init__(self):self.live=False
            def install(self):self.live=True
            def close(self):self.live=False
            def snapshot(self):return dict(requests=[],request_count=0,owned_hooks=int(self.live),leases=0,inflight=0,failure=None)
        for size in (1,2):
            for stage in ('prepare','register','check','enable'):
                workers=[_extended_worker(rank=i) for i in range(size)]
                for w in workers:w.load_model()
                c=ProductClient(ExistingExecutor(workers),FaultConfig(18,'api_half'),profile='norm',operator='trusted',run_canary=lambda:None,drain=lambda:None)
                def binding(action,**settings):
                    rows=[]
                    for w in workers:
                        if action=='attach':
                            member=settings['group_contract']['members'][w.rank]
                            b=ContinuousBinding(w,w.model_runner.model,c.config.mechanism,((1,2),),settings['group_contract'],member,'cpu',os.getpid(),threading.get_ident())
                            b.product={'scope':{'mode':'all'},'mode':'PRODUCT_ACCEPTANCE_ONLY'};b.warmups=[dict(all_finite=True,output=['stable'])]*2
                            setattr(w,ATTRIBUTE,{c.config.mechanism:b})
                            rows.append(dict(rank=w.rank))
                        elif action=='rollback':
                            self.assertFalse(getattr(w,ATTRIBUTE)[c.config.mechanism].leases)
                            del getattr(w,ATTRIBUTE)[c.config.mechanism]
                            rows.append(dict(rank=w.rank,status='PRODUCT_BINDING_ROLLED_BACK',resources_zero=True))
                        else:raise AssertionError(action)
                    return rows
                class Failure(ProductProvider):
                    def dispatch(self,action,settings,*,owner,generation):
                        if action==stage and self.identity['rank']==size-1:
                            if action=='prepare':self.bind(owner,generation)
                            self.error='injected control failure'
                            raise RuntimeError(self.error)
                        return super().dispatch(action,settings,owner=owner,generation=generation)
                with patch.object(identity,'_current_device',return_value=0), patch.object(runtime,'_resolve_provider',side_effect=lambda m,v,w,model,i:Failure(w,model,i,m)), patch('demoncat.runtime.vllm_ascend.worker_extension.providers.product.make_automatic_observer',side_effect=lambda b:Observer()):
                    c.attach();c._binding=binding
                    c._canary=Mock(return_value=[dict(warmup_valid=True,observations=[dict(output=['stable'])]) for _ in workers])
                    with self.assertRaisesRegex(RuntimeError,'PRODUCT_ALL_RANK_'):c.enable()
                    self.assertFalse(c.bound)
                    for w in workers:
                        r=w._demoncat_worker_fault_runtime
                        self.assertFalse(r._leases);self.assertFalse(r._tenants)
                    self.assertEqual(c.detach()['status'],'PRODUCT_DETACHED')

    def test_internal_factory_error_is_exit_five_not_contract_refusal(self):
        c=self.client()
        for child in c.clients:
            child.prepare.return_value=dict(status='PROVIDER_UNAVAILABLE',response={'error_type':'KeyError'})
        with self.assertRaises(RuntimeError):c._all('prepare')
        for child in c.clients:
            child.prepare.return_value=dict(status='PROVIDER_UNAVAILABLE',response={'error_type':'RuntimeError','provider_code':'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT'})
        with self.assertRaises(ValueError):c._all('prepare')

    def test_failed_rollback_can_be_retried_via_public_disable(self):
        c=self.client();c.bound=True;c._enable_failure_pending=True
        c.clients[0].detach.side_effect=RuntimeError('temporary transport failure')
        with self.assertRaises(RuntimeError):c.disable()
        self.assertTrue(c.bound)
        c.clients[0].detach.side_effect=None
        result=c.disable()
        self.assertFalse(c.bound);self.assertTrue(result['recovery']['resources_zero'])
        self.assertEqual(c.detach()['status'],'PRODUCT_DETACHED')


if __name__ == '__main__': unittest.main()
