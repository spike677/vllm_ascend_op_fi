"""Actual public Host/Runtime CPU contracts; fixtures never grant NPU scope."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from test_vision17_context import Runner
from test_core_vision17 import CompleteModel
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_group import (
    Vision17GroupProvider,install_vision17_group_binding,remove_vision17_group_binding,REQUESTS)


class Vision17GroupTests(unittest.TestCase):
    def setUp(self):
        self.patches=ExitStack();self.worker=_extended_worker(rank=7);self.worker.load_model()
        if type(self) is Vision17GroupTests:
            from demoncat.runtime.vllm_ascend.worker_extension.registry import create_research_provider
            self.patches.enter_context(patch.object(runtime,'_resolve_provider',create_research_provider))
        self.model=CompleteModel().eval();self.runner=Runner(self.model);self.worker.model_runner=self.runner
        self.patches.enter_context(patch.object(identity,'_current_device',return_value=0))
        self.patches.enter_context(patch.object(torch.distributed,'is_initialized',return_value=True))
        self.patches.enter_context(patch.object(torch.distributed,'get_rank',return_value=7))
        self.client=WorkerFaultClient(HostAdapter(ExistingExecutor([self.worker]),prepare_binding=self.bind),FaultConfig(17,'spatial_padding'))
        self.provider=None;self.serial=0;self.cached=None
    def bind(self,actual,config):
        self.assertEqual(actual,self.worker.demoncat_worker_identity()['worker'])
        install_vision17_group_binding(self.worker,prompt_token_ids=[1,2,3],image_sha256='a'*64,
            weights_sha256='b'*64,environment_sha256='c'*64,first_live_validation=True,_simulation=True,
            _processor=self.model.processor,_synchronize=lambda:None)
        return {}
    def tearDown(self):
        try:
            if self.provider:
                self.provider._abort_owned()
                group=self.provider.engine.group if self.provider.engine else None
                if group and not group.recovery_report()['complete']:
                    self.ok(self.client.recover())
                    self.model.fail=False
                    self.request();self.request()
                self.provider._cleanup()
        finally:
            self.patches.close()
            self.assertEqual(_get_current_dispatch_mode_stack(),[])
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def ok(self,result):
        response=result.get('response',{})
        self.assertEqual(result['status'],'ACTION_COMPLETED',repr(dict(
            error=self.provider.error if self.provider else None,response={k:v for k,v in response.items()
                if k not in ('provider','provider_state')},
            rejection=result.get('rejection')))[:5000])
        return result['response']['provider_state']
    def start(self):
        self.ok(self.client.prepare())
        self.provider=self.worker._demoncat_worker_fault_runtime._leases['17.spatial_padding'].provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
    def request(self,*,encode=True,collect=True):
        self.serial+=1;rid='actual-mm-'+str(self.serial)
        original=next(iter(self.runner.requests.values()));request=NS(**vars(original));request.req_id=rid
        self.runner.request_id=rid;self.runner.requests={rid:request}
        self.runner.input_batch=NS(req_ids=[rid],req_id_to_index={rid:0})
        scheduled=NS(scheduled_encoder_inputs={rid:[0]} if encode else {},num_scheduled_tokens={rid:3},
            total_num_scheduled_tokens=3,scheduled_spec_decode_tokens={})
        self.ok(self.client.advance(rid))
        with torch.inference_mode():
            if encode:self.cached=self.runner._execute_mm_encoder(scheduled)
            hidden=self.model(input_ids=torch.tensor([1,2,3]),positions=torch.arange(3),inputs_embeds=self.cached)
            result=self.model.compute_logits(hidden)
        if collect:self.ok(self.client.collect_replay())
        return result
    def healthy(self):
        for _ in range(4):self.request()
    def test_actual_public_c1_c2_c3_off_and_cleanup_health(self):
        self.start()
        for _ in REQUESTS:self.request()
        state=self.ok(self.client.finish());reports=state['reports']
        self.assertTrue(reports['C1']['complete'])
        self.assertTrue(reports['C2']['contract_satisfied']);self.assertTrue(reports['C3']['contract_satisfied'])
        self.assertEqual(reports['C2']['hits'],1);self.assertEqual(reports['C3']['hits'],2)
        self.assertEqual(reports['C3']['observed_calls'],3)
        self.assertTrue(reports['C3']['live_c2_proof']['consumed'])
        self.assertFalse(state['native_qualification']);self.assertFalse(state['multi_target_verified'])
        self.assertTrue(state['simulation']);self.assertEqual(state['actual_target_count'],1)
        self.assertTrue(reports['same_worker_healthy']['actual_encoder_observed'])
        self.assertEqual([r['fault_injections'] for r in state['phase_rows']],[0,0,0,0,1,0,1,1,0,0])
        for index,row in enumerate(state['phase_rows']):
            padding=row['conv_observation']['observed_calls'][0]['actual_geometry']['padding']
            self.assertEqual(padding,[0,7,7] if index in (4,6,7) else [0,0,0])
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        remove_vision17_group_binding(self.worker)
    def test_cache_skip_cannot_qualify_healthy_bridge(self):
        self.start();self.request();self.request(encode=False,collect=False)
        result=self.client.collect_replay();self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertIsNone(self.provider.engine.candidates)
        self.assertEqual(len(self.provider.engine.rows),1)
        self.ok(self.client.recover());self.assertTrue(self.provider.failures)
    def test_fault_cache_skip_requires_real_off_and_health_before_unregister(self):
        self.start();self.healthy();self.request(encode=False,collect=False)
        self.assertNotEqual(self.client.collect_replay()['status'],'ACTION_COMPLETED')
        self.assertNotEqual(self.client.unregister()['status'],'ACTION_COMPLETED')
        self.ok(self.client.recover());self.request();self.request()
        self.assertTrue(self.provider.recovered)
        self.assertFalse(self.provider.reports['failed_attempt_recovery']['contract_satisfied'])
        self.assertTrue(self.provider.reports['failed_attempt_recovery']['recovery']['complete'])
    def test_model_failure_after_actual_fault_keeps_failure_and_recovers(self):
        self.start();self.healthy();self.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'actual CPU model failed'):self.request()
        self.ok(self.client.recover());self.model.fail=False;self.request();self.request()
        self.assertTrue(self.provider.failures)
        report=self.provider.reports['failed_attempt_recovery']
        self.assertTrue(report['recovery']['complete']);self.assertFalse(report['contract_satisfied'])
    def test_query_does_not_execute_collect_or_advance(self):
        self.start();self.request(collect=False);calls=self.model.calls
        for _ in range(3):
            result=self.client.query();self.assertEqual(result['status'],'QUERY_OK')
            self.assertEqual(result['response']['provider_state']['serving']['completed_requests'],0)
        self.assertEqual(calls,self.model.calls);self.ok(self.client.collect_replay())
    def test_unknown_first_consumer_refuses_before_qualification(self):
        self.model.visual.unknown_reader=True;self.start()
        with self.assertRaises(AssertionError):self.request()
        self.assertIsNone(self.provider.engine.candidates)
        self.ok(self.client.recover())
    def test_no_actual_padding_change_cannot_be_rescued_by_admission(self):
        self.start();self.healthy()
        # Independent observer still grants its original ticket, but the actual
        # adapter passes through. Actual lower arguments must reject this.
        with patch.object(self.provider.engine,'admit',return_value=False):
            with self.assertRaises(AssertionError):self.request()
        self.assertIn('ACTUAL_FINAL_PADDING',self.provider.error)
        self.ok(self.client.recover());self.request();self.request()
        self.assertFalse(self.provider.reports['failed_attempt_recovery']['contract_satisfied'])
    def test_missing_real_off_holds_scope_even_after_hooks_are_removed(self):
        self.start();self.healthy();self.request()
        self.assertFalse(self.provider._hooks())
        result=self.client.unregister();self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertTrue(self.provider.binding.leases)
        self.ok(self.client.recover());self.request();self.request()
    def test_settings_cannot_select_fault_before_healthy_or_replace_report(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError,'UNSUPPORTED_PROVIDER_SETTINGS'):
            self.provider.dispatch('advance',{'request_id':'x','arm':'fault','proof':{}},
                owner=self.provider._ticket[0],generation=self.provider._ticket[1])
        self.assertIsNone(self.provider.engine.candidates)
        self.ok(self.client.recover())
    def test_manifest_is_fresh_ten_request_single_target_torch_scope(self):
        from experiments.c2c3_closure_20261117.prepare_vision17_worker_group import cases
        row,=cases('/home/reviewed-source',2)
        self.assertEqual(row['review']['physical_cards'],[2])
        self.assertEqual(row['review']['expected_hits'],3)
        self.assertIn('vision17_worker_group_probe.py',row['command'])
        self.assertEqual(len(REQUESTS),10)
        self.assertIn('no multi-target qualification',row['review']['actual_group_targets'])
    def test_foreign_hook_is_not_removed_and_binding_stays_owned(self):
        self.start();self.ok(self.client.advance('actual-pending'))
        handle,callback=self.provider.session.handles[0];hooks=handle.hooks_dict_ref()
        hooks[handle.id]=lambda *args:None
        try:
            with self.assertRaisesRegex(RuntimeError,'HOOK_OWNER_CHANGED'):self.provider._abort_owned()
            self.assertTrue(self.provider.binding.leases);self.assertIn(handle.id,hooks)
        finally:hooks[handle.id]=callback
        self.ok(self.client.recover())
    def test_equal_weight_visual_replacement_cannot_borrow_old_live_target(self):
        from copy import deepcopy
        self.start();self.healthy();original=self.model.visual
        self.model.visual=deepcopy(original)
        try:
            result=self.client.advance('actual-new-model')
            self.assertNotEqual(result['status'],'ACTION_COMPLETED')
            self.assertIn('LIFETIME_CHANGED',self.provider.error)
        finally:self.model.visual=original
        self.ok(self.client.recover());self.request();self.request()


if __name__=='__main__':unittest.main()
