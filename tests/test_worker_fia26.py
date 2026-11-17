import ctypes as C
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.core import UnsupportedCall
from fault_injection.deployments.residual26 import FiaState,Residual26Runtime
from demoncat.runtime.vllm_ascend.worker_extension.providers import resolve_provider
from demoncat.runtime.vllm_ascend.worker_extension.providers._scope import ProviderError
from demoncat.runtime.vllm_ascend.worker_extension.serving26 import FiaServingSession


class StateFunction:
    def __init__(self,**values):self.values=values
    def __call__(self,pointer):
        state=C.cast(pointer,C.POINTER(FiaState)).contents
        state.runtime_ready=state.allowed=1
        for name,value in self.values.items():setattr(state,name,value)
        return 0


class FiaNativeStateTests(unittest.TestCase):
    def runtime(self,**values):
        r=Residual26Runtime.__new__(Residual26Runtime);r.fia=True
        r.lib=SimpleNamespace(dcat26f_state_v1=StateFunction(**values))
        return r

    def test_actual_pending_and_owner_are_not_inferred_idle(self):
        state=self.runtime(registered=1,owner_thread=1,pending_consumers=2,queued_executors=1).native_state()
        self.assertEqual(state['pending_consumers'],2);self.assertEqual(state['queued_executors'],1)

    def test_foreign_owner_refused(self):
        with self.assertRaisesRegex(UnsupportedCall,'Foreign'):self.runtime(registered=1,owner_thread=0).native_state()

    def test_missing_abi_refused(self):
        r=self.runtime();r.lib=object()
        with self.assertRaisesRegex(UnsupportedCall,'missing'):r.native_state()

    def test_unready_and_changed_abi_refused(self):
        for values in ({'runtime_ready':0},{'allowed':0},{'size':1},{'version':2}):
            with self.subTest(values=values),self.assertRaises(UnsupportedCall):self.runtime(**values).native_state()


class FiaWorkerContractTests(unittest.TestCase):
    def provider(self):
        p=resolve_provider('26.fia_destination','1',object(),object(),{})
        p.bind('owner',1);return p

    def test_no_factory_cannot_prepare(self):
        p=self.provider()
        with patch('fault_injection.providers.resolve_runtime',return_value=None):
            with self.assertRaisesRegex(ProviderError,'REVIEWED'):p.dispatch('prepare',{},owner='owner',generation=1)

    def test_query_reports_real_pending_to_common_quarantine(self):
        p=self.provider();p.session=SimpleNamespace(current=None,error=None,aborted=False,
            installed=True,recovered=False,state=lambda:dict(registered=1,environment_enabled=0,
                callback_active=0,active_consumer=0,prepared=0,queued_executors=1,pending_consumers=2),report=lambda:{})
        state=p.query(owner='owner',generation=1)
        self.assertEqual(state['leases'],3);self.assertTrue(state['native_registered'])
        self.assertTrue(state['replay_pending'])
        with self.assertRaisesRegex(ProviderError,'OWNER'):p.query(owner='other',generation=1)

    def test_finish_requires_off_before_close(self):
        p=self.provider();calls=[]
        p.session=SimpleNamespace(recovered=False,collect=lambda:calls.append('collect'),close=lambda:calls.append('close'))
        with self.assertRaisesRegex(ProviderError,'OFF_RECOVERY'):p.dispatch('finish',{},owner='owner',generation=1)
        self.assertEqual(calls,['collect'])

    def test_pending_native_cleanup_keeps_hooks(self):
        s=FiaServingSession.__new__(FiaServingSession)
        s.collecting=False;s.native_session=None;s.collect=lambda:None
        s.handles=[SimpleNamespace(remove=lambda:self.fail('removed before native recovery'))]
        s.state=lambda:dict(registered=0,prepared=0,queued_executors=0,pending_consumers=1,
            active_consumer=0,callback_active=0,environment_enabled=0)
        with self.assertRaisesRegex(RuntimeError,'NATIVE_RECOVERY_PENDING'):s.close()
        self.assertEqual(len(s.handles),1)

    def test_changed_lossless_output_cannot_be_accepted(self):
        s=FiaServingSession.__new__(FiaServingSession)
        s.current=dict(phase='lossless',logits_returned=True,hidden_sha256='changed',logits_sha256='same',input_sha256='same')
        s.rows=[dict(hidden_sha256='same',logits_sha256='same',input_sha256='same')]
        s.error=None;s.collecting=False;s.native_session=None
        with self.assertRaisesRegex(RuntimeError,'NOT_RESTORED'):s.collect()
        self.assertEqual(len(s.rows),1)

    def test_advance_requires_actual_fresh_id(self):
        s=FiaServingSession.__new__(FiaServingSession)
        s.collect=lambda:None;s.installed=True;s.aborted=False;s.error=None
        s.expected=s.current=None;s.rows=[];s.seen=['used']
        for value in ('used','',None):
            with self.subTest(value=value),self.assertRaisesRegex(RuntimeError,'FRESH_ENQUEUED'):
                s.expect(value)


if __name__=='__main__':unittest.main()
