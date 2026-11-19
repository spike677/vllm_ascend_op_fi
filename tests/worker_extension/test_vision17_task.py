"""Public ordinary image task with actual CPU model and explicit scheduler doubles."""
import unittest
from unittest.mock import patch
import test_core_vision17_group as group_fixture
from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_task import (
    install_vision17_task_binding,remove_vision17_task_binding)
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from test_core_host import ExistingExecutor

class Vision17TaskTests(unittest.TestCase):
    setUp=group_fixture.Vision17GroupTests.setUp
    ok=group_fixture.Vision17GroupTests.ok
    start=group_fixture.Vision17GroupTests.start
    request=group_fixture.Vision17GroupTests.request

    def bind(self,actual,config):
        install_vision17_task_binding(self.worker,prompt_token_ids=[1,2,3],image_sha256='a'*64,
            weights_sha256='b'*64,environment_sha256='c'*64,_simulation=True,_processor=self.model.processor)
        return {}

    def tearDown(self):
        try:
            if self.provider:
                if self.provider.fault_started and not self.provider.recovered:
                    self.ok(self.client.recover());self.model.fail=False
                    self.request();self.request()
                self.provider._cleanup()
        finally:
            self.patches.close()
            from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
            self.assertEqual(_get_current_dispatch_mode_stack(),[])
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def finish(self):
        state=self.ok(self.client.finish())
        self.ok(self.client.disable());self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'],'DETACHED')
        remove_vision17_task_binding(self.worker)
        self.provider=None
        return state

    def test_four_requests_and_second_independent_scope_no_coverage(self):
        from fault_injection.coverage_session import CoverageSession
        with patch.object(CoverageSession,'__init__',side_effect=AssertionError('ordinary only')):
            for scope in range(2):
                if scope:
                    self.client=WorkerFaultClient(HostAdapter(ExistingExecutor([self.worker]),prepare_binding=self.bind),FaultConfig(17,'spatial_padding'))
                self.start()
                for _ in range(4):self.request()
                state=self.finish();evidence=state['evidence'];rows=evidence['actual_requests']
                self.assertEqual([r['phase'] for r in rows],['normal','fault','off','health'])
                self.assertTrue(evidence['matched_off_and_health_confirmed'])
                self.assertTrue(evidence['last_fault_effect']['complete_output_changed'])
                self.assertEqual([r['fault_injections'] for r in rows],[0,1,0,0])
                self.assertFalse(state['owned_hooks'] or state['inflight'] or state['leases'])

    def test_no_actual_padding_cannot_be_rescued_by_admission(self):
        self.start();self.request()
        with patch.object(self.provider.engine,'admit',return_value=False):
            with self.assertRaises(AssertionError):self.request()
        self.assertIn('ACTUAL_FINAL_PADDING',self.provider.error)
        self.assertNotEqual(self.client.unregister()['status'],'ACTION_COMPLETED')
        self.ok(self.client.recover());self.request();self.request()
        self.assertTrue(self.provider.recovered)

    def test_cached_encoder_cannot_be_claimed_as_fault(self):
        self.start();self.request();self.request(encode=False,collect=False)
        self.assertNotEqual(self.client.collect_replay()['status'],'ACTION_COMPLETED')
        self.assertFalse(self.provider.fault_admitted)
        self.ok(self.client.recover());self.request();self.request()
        self.assertTrue(self.provider.recovered)

    def test_actual_model_failure_keeps_owner_until_real_recovery(self):
        self.start();self.request();self.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'actual CPU model failed'):self.request()
        self.assertTrue(self.provider.fault_admitted)
        self.assertNotEqual(self.client.unregister()['status'],'ACTION_COMPLETED')
        self.ok(self.client.recover());self.model.fail=False;self.request();self.request()
        self.assertTrue(self.provider.recovered)
        self.assertTrue(self.provider.first_error)

    def test_query_never_collects_and_fifth_request_refuses(self):
        self.start();self.request(collect=False)
        count=self.model.calls
        self.assertEqual(self.client.query()['response']['provider_state']['evidence']['actual_requests'],[])
        self.assertEqual(count,self.model.calls)
        self.ok(self.client.collect_replay())
        for _ in range(3):self.request()
        result=self.client.advance('fifth')
        self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertIn('VISION17_TASK_PHASES_EXHAUSTED',self.provider.error)

if __name__=='__main__':unittest.main()
