"""Actual opaque CPU MoE calls, public Host, explicit scheduler/native doubles."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
import sys
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from demoncat.runtime.vllm_ascend.worker_extension import identity, serving_tensor_binding as bindings
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
import test_serving_session_lifecycle as life
import test_serving_moe_group as moe

class MoEModel(life.CpuModel):
    def __init__(self):
        super().__init__()
        self.first=moe.Expert(91); self.last=moe.Expert(92)
        self.forward_calls=0
    def forward(self,input_ids,positions,inputs_embeds=None,intermediate_tensors=None):
        self.forward_calls+=1
        hidden=super().forward(input_ids,positions,inputs_embeds,intermediate_tensors).float()
        return self.last(self.first(hidden,hidden),hidden).to(torch.bfloat16)

class Restore8TaskTests(unittest.TestCase):
    snapshot=life.ServingSessionLifecycleTests.snapshot
    root=life.ServingSessionLifecycleTests.root
    forward=life.ServingSessionLifecycleTests.forward
    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        self.worker=_extended_worker(rank=0);self.worker.load_model()
        self.runner=self.worker.model_runner;self.model=MoEModel().eval();self.runner.model=self.model
        self.runner.vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3)))
        self.enterContext(patch.object(identity,'_current_device',return_value=0))
        self.enterContext(patch.object(bindings,'require_serial_eager'))
        self.executor=ExistingExecutor([self.worker]);self.provider=None
        self.client=self.new_client()
        self.addCleanup(self.dispose)
    def new_client(self):
        def bind(actual,config):
            bindings.install_restore8_task_binding(self.worker,prompt_token_ids=[1,2,3,4],
                weight_manifest={'cpu_fixture':True},workload_sha256='a'*64,environment_sha256='b'*64,
                _simulation=True,_synchronize=lambda:None,_trace_factory=moe.template)
            return {}
        return WorkerFaultClient(HostAdapter(self.executor,prepare_binding=bind),FaultConfig(8,'restore_offset'))
    def dispose(self):
        if self.provider and self.provider.session and self.provider.session.installed:
            self.provider.session.request_error=self.provider.session.request_error or 'CPU_TEARDOWN_ONLY'
            self.provider.session.close()
        self.assertEqual(_get_current_dispatch_mode_stack(),[])
        self.assertIsNone(sys.getprofile())
    def ok(self,reply):
        self.assertEqual(reply['status'],'ACTION_COMPLETED',repr(reply.get('response'))[:3000])
        return reply['response']['provider_state']
    def start(self):
        self.ok(self.client.prepare())
        self.provider=self.worker._demoncat_worker_fault_runtime._leases['8.restore_offset'].provider
        for action in ('register','check','enable'):self.ok(getattr(self.client,action)())
    def enqueue(self,name):
        self.actual_request_id=name
        before=self.model.forward_calls;self.ok(self.client.advance(name))
        self.assertEqual(self.model.forward_calls,before)
        with torch.inference_mode():return self.forward(self.provider.session)
    def finish(self):
        state=self.ok(self.client.finish())
        self.ok(self.client.disable());self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'],'DETACHED')
        bindings.remove_tensor_serving_binding(self.worker,'8.restore_offset')
        return state
    def test_four_requests_auto_select_one_actual_combine_off_health_and_new_scope(self):
        from fault_injection.coverage_session import CoverageSession
        with patch.object(CoverageSession,'__init__',side_effect=AssertionError('No coverage task')):
            for scope in range(2):
                if scope:self.client=self.new_client()
                self.start()
                for phase in ('normal','fault','off','health'):self.enqueue(str(scope)+'-'+phase)
                state=self.finish();rows=state['evidence']['actual_requests']
                self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
                self.assertEqual([r['phase'] for r in rows],['normal','fault','off','health'])
                self.assertEqual([len((r['mechanism'] or {}).get('events',[])) for r in rows],[1,1,0,0])
                self.assertEqual(self.provider.target['path'],'first')
                hit=rows[1]['mechanism']['events'][0]
                self.assertTrue(hit['applied'] and hit['output_changed'])
                self.assertEqual(hit['observed_original_native_calls'],1)
                self.assertTrue(hit['actual_producer_join']['actual_routed_return_same_object'])
                self.assertGreater(hit['effect']['values']['changed_elements'],0)
                self.assertEqual(rows[0]['output'],rows[2]['output']);self.assertEqual(rows[0]['output'],rows[3]['output'])
                self.assertFalse(state['owned_hooks'] or state['leases'] or state['inflight'])
                self.assertEqual(state['evidence']['current_request_effect']['phase'],'health')
                self.assertIsNone(state['evidence']['current_request_effect']['effect'])
                from experiments.worker_core_v11_20261118.restore8_single_task_probe import checks, verify_normal
                verify_normal(rows[0],cpu_contract=True)
                self.assertGreater(checks(rows)['typed_effect']['changed_elements'],0)
        self.assertEqual(self.model.forward_calls,8)
    def test_changed_probability_producer_refuses_before_remap(self):
        original=moe.Dispatcher.token_dispatch
        def changed(dispatcher,token_dispatch_input):
            result=original(dispatcher,token_dispatch_input)
            result.combine_metadata.topk_weights=result.combine_metadata.topk_weights.clone()
            return result
        self.start()
        with patch.object(moe.Dispatcher,'token_dispatch',changed):
            with self.assertRaisesRegex(RuntimeError,'ORIGINAL_TOPK_PROBABILITY_PRODUCER_IDENTITY_REQUIRED'):
                self.enqueue('bad-producer')
        self.assertFalse(self.provider.fault_applied)
        self.ok(self.client.abort());self.ok(self.client.clean());self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'],'DETACHED')

if __name__=='__main__':unittest.main()
