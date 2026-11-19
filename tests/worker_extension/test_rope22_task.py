"""Actual CPU opaque RoPE + public task lifecycle; explicit scheduler fixture."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest,sys
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from demoncat.runtime.vllm_ascend.worker_extension import identity,serving_tensor_binding as bindings
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter,WorkerFaultClient
from fault_injection.rope_census import NaturalRoPETrace
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
import test_serving_session_lifecycle as life
from rope22_fixture import Body,triton_reader,JIT

class Model(life.CpuModel):
    def __init__(self):
        super().__init__();self.body=Body();self.forward_calls=0
    def forward(self,input_ids,positions,inputs_embeds=None,intermediate_tensors=None):
        self.forward_calls+=1
        x=super().forward(input_ids,positions,inputs_embeds,intermediate_tensors).float()
        return self.body(x,positions).to(torch.bfloat16)

def template(model):
    return NaturalRoPETrace(model,triton_reader,{'body.layers.0.rope':model.body.layers[0].rope},{},
        simulation=True,outer_operator='worker_core_rope22_cpu_fixture::opaque',triton_kernel=JIT)

class Rope22TaskTests(unittest.TestCase):
    snapshot=life.ServingSessionLifecycleTests.snapshot
    root=life.ServingSessionLifecycleTests.root
    forward=life.ServingSessionLifecycleTests.forward
    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        JIT.skip_launch=JIT.replace_position=False
        self.worker=_extended_worker(rank=0);self.worker.load_model()
        self.runner=self.worker.model_runner;self.model=Model().eval();self.runner.model=self.model
        self.runner.vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3)))
        self.enterContext(patch.object(identity,'_current_device',return_value=0))
        self.enterContext(patch.object(bindings,'require_serial_eager'))
        self.executor=ExistingExecutor([self.worker]);self.provider=None
        self.client=self.new_client();self.addCleanup(self.dispose)
    def new_client(self):
        def bind(actual,config):
            bindings.install_rope22_task_binding(self.worker,prompt_token_ids=[1,2,3,4],
                weight_manifest={'cpu_fixture':True},workload_sha256='a'*64,environment_sha256='b'*64,
                _simulation=True,_synchronize=lambda:None,_trace_factory=template)
            return {}
        return WorkerFaultClient(HostAdapter(self.executor,prepare_binding=bind),FaultConfig(22,'wrong_row'))
    def dispose(self):
        if self.provider and self.provider.session and self.provider.session.installed:
            self.provider.session.request_error=self.provider.session.request_error or 'CPU_TEARDOWN_ONLY'
            self.provider.session.close()
        self.assertEqual(_get_current_dispatch_mode_stack(),[]);self.assertIsNone(sys.getprofile())
    def ok(self,reply):
        self.assertEqual(reply['status'],'ACTION_COMPLETED',repr(reply.get('response'))[:3000])
        return reply['response']['provider_state']
    def start(self):
        self.ok(self.client.prepare())
        self.provider=self.worker._demoncat_worker_fault_runtime._leases['22.wrong_row'].provider
        for name in ('register','check','enable'):self.ok(getattr(self.client,name)())
    def enqueue(self,name):
        self.actual_request_id=name
        count=self.model.forward_calls;self.ok(self.client.advance(name))
        self.assertEqual(count,self.model.forward_calls)
        with torch.inference_mode():return self.forward(self.provider.session)
    def test_four_requests_single_actual_reader_and_new_scope(self):
        from fault_injection.coverage_session import CoverageSession
        with patch.object(CoverageSession,'__init__',side_effect=AssertionError('ordinary only')):
            for scope in range(2):
                if scope:self.client=self.new_client()
                self.start()
                for phase in ('normal','fault','off','health'):self.enqueue(str(scope)+'-'+phase)
                state=self.ok(self.client.finish());rows=state['evidence']['actual_requests']
                self.assertTrue(state['evidence']['matched_off_and_health_confirmed'])
                self.assertEqual([len((r['mechanism'] or {}).get('events',[])) for r in rows],[1,1,0,0])
                trace=rows[1]['mechanism']['rope_trace']
                self.assertEqual(trace['observed_readers'],1)
                self.assertFalse(trace['structural_coverage_complete'])
                self.assertTrue(trace['complete'])
                self.assertEqual(trace['readers'][0]['independent_position_transform'],'last_from_first')
                self.assertTrue(rows[1]['logits_changed'])
                from experiments.worker_core_v11_20261118.rope22_task_probe import checks,verify_normal
                verify_normal(rows[0],cpu_contract=True)
                self.assertTrue(checks(rows)['typed_effect']['complete_qk_changed'])
                self.ok(self.client.disable());self.ok(self.client.unregister())
                self.assertEqual(self.client.detach()['status'],'DETACHED')
                bindings.remove_tensor_serving_binding(self.worker,'22.wrong_row')
        self.assertEqual(self.model.forward_calls,8)
    def test_missing_actual_kernel_launch_refuses_no_fault_claim(self):
        self.start();JIT.skip_launch=True
        try:
            with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_COMPLETE_READER'):
                self.enqueue('missing-launch')
        finally:JIT.skip_launch=False
        self.assertFalse(self.provider.fault_applied)
        self.ok(self.client.abort());self.ok(self.client.clean());self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'],'DETACHED')

if __name__=='__main__':unittest.main()


class Rope22AcceptanceCleanupTests(unittest.TestCase):
    def test_accepted_and_experimental_deployment_cleanup(self):
        from experiments.worker_core_v11_20261118 import rope22_task_probe as probe
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as gate
        for experimental in (False,True):
            w=NS(model_runner=NS(model=object()),demoncat_worker_identity=lambda:{'worker':{'rank':0}})
            if experimental:
                gate.install_blue_acceptance_permit(w,mechanisms=['22.wrong_row'],attempt='CPU-cleanup',source_manifest_sha256='a'*64)
            with self.subTest(experimental=experimental),patch.object(probe,'actual_worker',return_value=w),patch.object(bindings,'remove_tensor_serving_binding',return_value={'status':'SERVING_DEPLOYMENT_REMOVED'}):
                self.assertEqual(probe.remove_binding(w,{'rank':0})['status'],'SERVING_DEPLOYMENT_REMOVED')
                self.assertFalse(hasattr(w,'_demoncat_blue_acceptance_permit'))
    def test_binding_cleanup_failure_keeps_experimental_permission(self):
        from experiments.worker_core_v11_20261118 import rope22_task_probe as probe
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as gate
        w=NS(model_runner=NS(model=object()),demoncat_worker_identity=lambda:{'worker':{'rank':0}})
        gate.install_blue_acceptance_permit(w,mechanisms=['22.wrong_row'],attempt='CPU-cleanup',source_manifest_sha256='a'*64)
        with patch.object(probe,'actual_worker',return_value=w),patch.object(bindings,'remove_tensor_serving_binding',side_effect=RuntimeError('OWNER_LIVE')):
            with self.assertRaisesRegex(RuntimeError,'OWNER_LIVE'):probe.remove_binding(w,{'rank':0})
        self.assertTrue(hasattr(w,'_demoncat_blue_acceptance_permit'))
