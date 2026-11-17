import ctypes as C
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from fault_injection.quantization import PinnedQuantRuntime,_VERIFIED_LOAD
from fault_injection.core import UnsupportedCall
from demoncat.runtime.vllm_ascend.worker_extension.providers import resolve_provider
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import ProviderError
from demoncat.runtime.vllm_ascend.worker_extension.serving_quant import QuantServingSession


class State:
    def __init__(self,values):self.values=values
    def __call__(self,out,count):
        for i,v in enumerate(self.values):out[i]=v
        return 0


class QuantWorkerTests(unittest.TestCase):
    def test_decode_capture_cannot_start_before_actual_prefill_or_twice(self):
        s=QuantServingSession.__new__(QuantServingSession);s.capture_scope='first_decode';s.installed=True
        s.expected={'id':dict(prompt=[1,2])};s.capture_started=False;s.active=None;s.prefill_result=None
        s.decode_result=None;s.forward_count=0;s.error=None
        with self.assertRaisesRegex(RuntimeError,'COMPLETED_PREFILL'):s.begin_decode_capture()
        s.prefill_result={};s.forward_count=1
        s.runner=NS(input_batch=NS(req_ids=['id']),requests={'id':NS(prompt_token_ids=[1,2],output_token_ids=[])})
        with self.assertRaisesRegex(RuntimeError,'REAL_PREFILL_TOKEN'):s.begin_decode_capture()
        s.runner.requests['id'].output_token_ids=[3];calls=[]
        def begin():calls.append(True);s.capture_started=True
        s.begin_observation=begin;s.begin_decode_capture();self.assertEqual(calls,[True])
        with self.assertRaisesRegex(RuntimeError,'COMPLETED_PREFILL'):s.begin_decode_capture()

    def test_query_satisfies_common_worker_actual_state_contract(self):
        from demoncat.runtime.vllm_ascend.worker_extension.runtime import WorkerFaultRuntime
        p=resolve_provider('quant.batch_shared_scale','1',object(),object(),{});p.bind('owner',1)
        p.session=NS(expected=None,error=None,aborted=False,installed=False,recovered=False,
            state=lambda:dict(version=1,armed=0,pending=0,executing=0,owner_thread=0,owned=0,errors=0),report=lambda:{})
        lease=NS(provider=p,native_owner='owner',generation=1,replay_pending=False,recovery_required=False)
        state=WorkerFaultRuntime._fresh(lease)
        self.assertIsNotNone(state);self.assertTrue(WorkerFaultRuntime._inert(state))
        p.session.installed=True
        state=WorkerFaultRuntime._fresh(lease);self.assertTrue(lease.replay_pending)

    def test_hidden_resolver_is_pending_until_actual_native_counts(self):
        b=NS(resolver_checks={k:None for k in ('aclnnDynamicQuantV2GetWorkspaceSize','aclnnDynamicQuantV2','CanUsePTACache')},
            resolver_strategy='ACTUAL_TYPED_ENTRY_COUNTS_REQUIRED',finish=lambda:dict(prepare_hits=0,execute_hits=0,
                fault_executions=0,api_version=0,prepare_errors=0,execution_errors=0))
        with self.assertRaises(UnsupportedCall):PinnedQuantRuntime(b,lambda x:x,{},_load_token=_VERIFIED_LOAD)
        r=PinnedQuantRuntime(b,lambda x:x,{},_load_token=_VERIFIED_LOAD,_invocation_binding_pending=True)
        self.assertFalse(r.proves_native_receipt(dict(prepare_hits=0,execute_hits=0),fault=False))
        with self.assertRaises(UnsupportedCall):r.finish(fault=False)
        self.assertTrue(r.poisoned)

    def runtime(self,values):
        r=PinnedQuantRuntime.__new__(PinnedQuantRuntime);r.backend=NS(lib=NS(dcat_dq_state_v1=State(values)))
        return r

    def test_actual_pending_is_not_hidden(self):
        s=self.runtime([1,0,1,2,1,1,0]).native_state()
        self.assertEqual(s['pending'],1);self.assertEqual(s['executing'],2)

    def test_foreign_owner_or_wrong_abi_refused(self):
        for values in ([1,0,1,0,0,1,0],[2,0,0,0,0,0,0]):
            with self.subTest(values=values),self.assertRaises(UnsupportedCall):self.runtime(values).native_state()

    def test_missing_deployment_cannot_prepare(self):
        p=resolve_provider('quant.batch_shared_scale','1',object(),object(),{})
        with patch('fault_injection.providers.resolve_runtime',return_value=None),self.assertRaisesRegex(ProviderError,'REVIEWED'):
            p.dispatch('prepare',{},owner='owner',generation=1)

    def test_first_decode_lineage_is_required_even_with_equal_shape(self):
        import torch
        s=QuantServingSession.__new__(QuantServingSession);m=object();x=torch.ones(8,4)
        s.active={'decode':{}};s.selected=m;s.normalized=x
        with self.assertRaises(UnsupportedCall):s.context(NS(module=m,input=x.clone()))

    def test_cleanup_does_not_remove_hooks_with_pending_native_executor(self):
        s=QuantServingSession.__new__(QuantServingSession);s.collect=lambda:None
        s.collecting=False;s.quant_session=None;s.quant_api=None
        s.state=lambda:dict(armed=0,pending=1,executing=0,owned=1)
        s.handles=[NS(remove=lambda:self.fail('removed while pending'))]
        with self.assertRaisesRegex(RuntimeError,'NATIVE_RECOVERY_PENDING'):s.close()
        self.assertEqual(len(s.handles),1)


if __name__=='__main__':unittest.main()
