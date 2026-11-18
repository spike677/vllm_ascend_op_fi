"""Real Host/Runtime CPU contracts with explicit CPU-only source/identity seams."""
from contextlib import ExitStack
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

from test_vision17_context import Model,Runner
from test_serving_tensor_group import Processor
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from demoncat.runtime.vllm_ascend.worker_extension import identity,runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17 import (
    Vision17ObservationProvider,install_vision17_binding,remove_vision17_binding)


class CompleteModel(Model):
    def __init__(self):
        super().__init__();self.processor=Processor();self.processor.weight=torch.ones(2,4)
        self.calls=0;self.fail=False
    def forward(self,input_ids,positions,inputs_embeds):
        self.calls+=1
        if self.fail:raise RuntimeError('actual CPU model failed')
        return inputs_embeds+1
    def compute_logits(self,hidden):return self.processor(hidden[-1:])


class CoreVisionTests(unittest.TestCase):
    def setUp(self):
        self.patches=ExitStack();self.worker=_extended_worker(rank=7);self.worker.load_model()
        self.model=CompleteModel().eval();self.runner=Runner(self.model);self.worker.model_runner=self.runner
        self.patches.enter_context(patch.object(identity,'_current_device',return_value=0))
        self.patches.enter_context(patch.object(torch.distributed,'is_initialized',return_value=True))
        self.patches.enter_context(patch.object(torch.distributed,'get_rank',return_value=7))
        # Resolver seam chooses the reviewed optional provider, not a fake Runtime.
        self.patches.enter_context(patch.object(runtime,'_resolve_provider',
            lambda mechanism,version,worker,model,actual:Vision17ObservationProvider(worker,model,actual)))
        self.client=WorkerFaultClient(HostAdapter(ExistingExecutor([self.worker]),prepare_binding=self.bind),FaultConfig(17,'spatial_padding'))
        self.provider=None;self.serial=0;self.cached=None
    def bind(self,actual,config):
        self.assertEqual(actual,self.worker.demoncat_worker_identity()['worker'])
        install_vision17_binding(self.worker,prompt_token_ids=[1,2,3],image_sha256='a'*64,
            weights_sha256='b'*64,first_live_validation=True,_simulation=True,_processor=self.model.processor)
        return {}
    def tearDown(self):
        try:
            if self.provider is not None:self.provider._cleanup()
        finally:
            self.patches.close()
            self.assertEqual(_get_current_dispatch_mode_stack(),[])
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))
    def ok(self,result):
        self.assertEqual(result['status'],'ACTION_COMPLETED',repr(result)[:2200])
        return result['response']['provider_state']
    def start(self):
        self.ok(self.client.prepare())
        self.provider=self.worker._demoncat_worker_fault_runtime._leases['17.spatial_padding'].provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
    def request(self,*,encode=True,collect=True,swap_request=False):
        self.serial+=1;rid='real-enqueued-'+str(self.serial)
        original=next(iter(self.runner.requests.values()));original.req_id=rid
        self.runner.request_id=rid;self.runner.requests={rid:original}
        self.runner.input_batch=NS(req_ids=[rid],req_id_to_index={rid:0})
        scheduled=NS(scheduled_encoder_inputs={rid:[0]} if encode else {},num_scheduled_tokens={rid:3},
                     total_num_scheduled_tokens=3,scheduled_spec_decode_tokens={})
        self.ok(self.client.advance(rid))
        with torch.inference_mode():
            if encode:self.cached=self.runner._execute_mm_encoder(scheduled)
            if swap_request:self.runner.requests[rid]=NS(**vars(original))
            hidden=self.model(input_ids=torch.tensor([1,2,3]),positions=torch.arange(3),inputs_embeds=self.cached)
            output=self.model.compute_logits(hidden)
        if collect:self.ok(self.client.collect_replay())
        return output
    def test_two_actual_image_requests_through_existing_runtime_and_cleanup(self):
        self.start();self.assertEqual(self.model.calls,0)
        self.request();self.request()
        state=self.ok(self.client.finish());report=state['observation']
        self.assertTrue(report['two_healthy_outputs_exact']);self.assertTrue(report['two_actual_encoder_chains_observed'])
        self.assertEqual(report['observed_calls'],2);self.assertEqual(report['fault_injections'],0)
        self.assertFalse(report['c2']);self.assertFalse(report['aclnn_launch_arguments'])
        self.ok(self.client.clean())
        with self.assertRaisesRegex(RuntimeError,'OWNER_CLEAN_REQUIRED'):remove_vision17_binding(self.worker)
        self.ok(self.client.unregister());self.assertEqual(self.client.detach()['status'],'DETACHED')
        remove_vision17_binding(self.worker)
    def test_cache_reuse_is_retained_as_unobserved_not_zero_fault_success(self):
        self.start();self.request();self.request(encode=False)
        state=self.ok(self.client.finish());report=state['observation']
        self.assertTrue(report['two_healthy_outputs_exact'])
        self.assertFalse(report['two_actual_encoder_chains_observed'])
        self.assertIn('NO_ACTUAL_ENCODER_CALL',report['rows'][1]['unresolved_reason'])
        self.assertEqual(report['observed_calls'],1)
    def test_query_does_not_execute_or_collect_a_request(self):
        self.start();self.request(collect=False)
        calls=self.model.calls
        for _ in range(3):
            row=self.client.query();self.assertEqual(row['status'],'QUERY_OK')
            self.assertEqual(row['response']['provider_state']['serving']['completed_requests'],0)
        self.assertEqual(calls,self.model.calls);self.ok(self.client.collect_replay())
    def test_finish_before_actual_two_requests_is_refused_then_cleanup_retains_failure(self):
        self.start();self.request()
        result=self.client.finish();self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.ok(self.client.recover())
        self.assertTrue(self.provider.failures)
        self.ok(self.client.unregister());self.client.detach()
    def test_runtime_never_accepts_injection_settings_for_observation_provider(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError,'UNSUPPORTED_PROVIDER_SETTINGS'):
            self.provider.dispatch('advance',{'request_id':'x','arm':'fault'},
                owner=self.provider._ticket[0],generation=self.provider._ticket[1])
    def test_model_exception_cannot_report_healthy_completion(self):
        self.start();self.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'actual CPU model failed'):self.request()
        self.assertIsNotNone(self.provider.session.error)
        self.assertFalse(self.provider.session.report()['two_healthy_outputs_exact'])
        state=self.ok(self.client.recover())
        original=state['observation']
        self.assertIn('ROOT_RETURN_UNCONFIRMED',original['error'])
        self.assertIsNotNone(original['incomplete_request'])
        after=self.ok(self.client.unregister())
        self.assertEqual(original,after['observation'])
    def test_foreign_hook_replacement_retains_owner_until_real_restoration(self):
        self.start();handle,callback=self.provider.session.handles[0]
        hooks=handle.hooks_dict_ref();hooks[handle.id]=lambda *args:None
        try:
            with self.assertRaisesRegex(RuntimeError,'HOOK_OWNER_CHANGED'):self.provider._cleanup()
            self.assertTrue(self.provider.binding.leases)
            self.assertIn(handle.id,hooks)
        finally:hooks[handle.id]=callback
        self.ok(self.client.recover());self.ok(self.client.unregister())
    def test_fresh_request_without_finished_logits_cannot_be_collected(self):
        self.start();self.ok(self.client.advance('actual-unfinished'))
        result=self.client.collect_replay()
        self.assertNotEqual(result['status'],'ACTION_COMPLETED')
        self.assertEqual(self.provider.session.rows,[])
        self.ok(self.client.recover())
    def test_equal_content_new_request_object_cannot_replace_live_encoder_request(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError,'IMAGE_REQUEST_LIFETIME_CHANGED'):
            self.request(swap_request=True)
        self.assertFalse(self.provider.session.report()['two_actual_encoder_chains_observed'])
        self.ok(self.client.recover())


if __name__=='__main__':unittest.main()
