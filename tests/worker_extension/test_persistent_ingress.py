import threading
import unittest
from unittest.mock import Mock
from demoncat.runtime.vllm_ascend.worker_extension.persistent_ingress import PersistentIngress,IngressRefused


class IngressTests(unittest.TestCase):
    def setUp(self):
        self.backend=Mock(return_value=dict(inflight=0,pending_submissions=0,aborts_completed=True))
        self.gate=PersistentIngress(validate_request=lambda r:None if r['tokens']<=4 else 'TOKEN_RANGE',drain_backend=self.backend)
    def test_range_is_rejected_before_submission(self):
        with self.assertRaisesRegex(IngressRefused,'TOKEN_RANGE'):
            with self.gate.request({'tokens':5}):self.fail('backend reached')
        self.assertEqual(self.gate.snapshot()['admitted'],0)
    def test_real_active_requests_must_drain_and_no_new_submission(self):
        entered=threading.Event();finish=threading.Event()
        def request():
            with self.gate.request({'tokens':4}):entered.set();finish.wait(2)
        thread=threading.Thread(target=request);thread.start();self.assertTrue(entered.wait(1))
        try:
            with self.assertRaisesRegex(RuntimeError,'TIMEOUT'):self.gate.pause_and_drain('a',timeout=.01)
            self.backend.assert_not_called()
            with self.assertRaisesRegex(IngressRefused,'PAUSED'):
                with self.gate.request({'tokens':4}):self.fail()
        finally:finish.set();thread.join(2)
        p=self.gate.pause_and_drain('a',timeout=1)
        self.assertTrue(p['backend_drained']);self.assertIsNone(p['device_readers_complete'])
        self.gate.resume('a')
        with self.gate.request({'tokens':4}):pass
        self.gate.pause_and_drain('a',timeout=1);self.gate.release('a')
    def test_http_return_does_not_prove_backend_abort_completion(self):
        with self.gate.request({'tokens':4}):pass
        self.backend.return_value=dict(inflight=1,pending_submissions=0,aborts_completed=False)
        with self.assertRaisesRegex(RuntimeError,'UNPROVEN'):self.gate.pause_and_drain('a',timeout=1)
        self.assertFalse(self.gate.snapshot()['accepting'])
        self.assertEqual(self.gate.snapshot()['owner'],'a')
        with self.assertRaises(IngressRefused):self.gate.resume('a')
    def test_other_owner_cannot_release_or_resume(self):
        self.gate.pause_and_drain('a',timeout=1)
        for fn in (self.gate.resume,self.gate.release):
            with self.assertRaises(IngressRefused):fn('b')
        with self.assertRaisesRegex(IngressRefused,'OTHER_OWNER'):self.gate.pause_and_drain('b',timeout=1)
        self.gate.release('a')


if __name__=='__main__':unittest.main()
