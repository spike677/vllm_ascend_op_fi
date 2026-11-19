"""Real CPU writer/cache/consumer + public Host protocol; no NPU claims."""
import unittest
from unittest.mock import patch
from copy import deepcopy
from test_core_kv41 import CoreKV41Tests
from demoncat.runtime.vllm_ascend.worker_extension import serving41
from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import install_blue_acceptance_permit
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report

class KV41TaskTests(unittest.TestCase):
    def ok(self,result):
        self.assertIn(result['status'], ('ACTION_COMPLETED','QUERY_OK'),repr(result)[-2000:])
        return result['response']['provider_state']
    start=CoreKV41Tests.start
    actual_request=CoreKV41Tests.actual_request
    enqueue=CoreKV41Tests.enqueue
    through=CoreKV41Tests.through

    def setUp(self):
        CoreKV41Tests.setUp(self)
        original=serving41.install_fault41_serving_binding
        self.enterContext(patch.object(serving41,'install_fault41_serving_binding',
            lambda *a,**kw: original(*a,runtime_only=True,**kw)))
        install_blue_acceptance_permit(self.worker,mechanisms=['41.tail_column'],
            attempt='explicit-cpu-contract',source_manifest_sha256='a'*64)

    def dispose(self):
        runtime=getattr(self.worker,'_demoncat_worker_fault_runtime',None)
        lease=runtime._leases.get('41.tail_column') if runtime else None
        if lease and lease.provider.health:
            lease.provider.health.active=False;lease.provider.health.current=None
            # Disposal is not a tested successful recovery.
            for handle in lease.provider.health.handles:handle.remove()
        CoreKV41Tests.dispose(self)

    def test_actual_completion_query_is_pure_and_two_owner_cycles(self):
        owners=[]
        for scope in range(2):
            if scope:
                from demoncat.runtime.vllm_ascend.worker_extension.host import WorkerFaultClient
                self.client=WorkerFaultClient(self.client.host,self.client.config)
            self.start();owners.append(self.provider._ticket[0])
            for i in range(6):
                self.enqueue(scope*10+i)
                before=(self.model.forward_calls,len(self.provider.rows),self.provider.index,self.transport.restore_attempts)
                state=self.ok(self.client.query())
                self.ok(self.client.query())
                self.assertEqual(before,(self.model.forward_calls,len(self.provider.rows),self.provider.index,self.transport.restore_attempts))
                self.assertEqual(len(state['evidence']['actual_requests']),i+1)
            self.assertIsNone(self.provider.group)
            self.assertTrue(state['serving']['complete'])
            self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
            self.assertGreater(state['last_fault_effect']['changed_elements'],0)
            self.assertEqual(state['last_fault_effect']['original_calls'],1)
            self.assertEqual([r['new_fault_hits'] for r in self.provider.rows[:5]],[0,0,0,1,0])
            self.assertEqual((state['leases'],state['owned_hooks']), (0,0))
            self.assertFalse(self.provider.p.patches)
            for action in ('finish','disable','unregister'):self.ok(getattr(self.client,action)())
            self.client.detach()
            serving41.remove_fault41_serving_binding(self.worker)
        self.assertNotEqual(*owners)

    def test_advance_without_request_query_cannot_make_completion(self):
        self.start();self.ok(self.client.advance('not-yet-run'))
        state=self.ok(self.client.query())
        self.assertEqual(state['evidence']['actual_requests'],[])
        self.assertEqual(self.model.forward_calls,0)
        self.assertFalse(self.transport.opened)
        self.ok(self.client.unregister())

    def test_missing_ordinary_binding_and_permit_do_not_admit(self):
        report=ordinary_route_report('41.tail_column',self.worker,self.model)
        self.assertFalse(report['can_prepare'])
        delattr(self.worker,'_demoncat_blue_acceptance_permit')
        result=self.client.prepare()
        self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertEqual(self.model.forward_calls,0)
        self.assertFalse(self.transport.opened)

    def test_fault_requires_actual_off_and_unhooked_health(self):
        self.start();self.through(4)
        result=self.client.unregister()
        self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertIn('41.tail_column',self.worker._demoncat_worker_fault_runtime._leases)
        self.ok(self.client.recover())
        self.enqueue(6)
        self.assertFalse(self.provider.p.installed)
        self.assertFalse(self.provider.p.patches)
        self.enqueue(7)
        self.assertTrue(self.provider.recovered)
        self.ok(self.client.unregister())

    def test_changed_health_is_not_recovery_and_retry_does_not_repeat_fault(self):
        self.start();self.through(5)
        self.model.output_shift=10000
        with self.assertRaisesRegex(RuntimeError,'DECODE_HEALTH_OUTPUT_NOT_EXACT'):self.enqueue(5)
        state=self.ok(self.client.query())
        self.assertFalse(state['evidence']['matched_off_and_health_confirmed'])
        self.model.output_shift=0
        self.ok(self.client.recover());self.enqueue(6)
        self.assertTrue(self.provider.recovered)
        self.assertEqual(sum(r.get('new_fault_hits',0) for r in self.provider.rows),1)
        self.ok(self.client.unregister())

    def test_later_root_hook_cannot_qualify_intermediate_output(self):
        self.start();self.ok(self.client.advance('late-hook'))
        hook=self.model.register_forward_hook(lambda m,a,o:o+1)
        try:
            with self.assertRaisesRegex(RuntimeError,'FINAL_ROOT_RETURN_OBSERVER_REQUIRED'):self.actual_request('late-hook')
            self.assertFalse(self.provider.rows)
            self.assertFalse(self.transport.opened)
        finally:hook.remove()
        self.ok(self.client.recover());self.ok(self.client.unregister())

if __name__=='__main__':unittest.main()
