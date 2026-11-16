"""CPU contract tests for actual scheduler scoping; no vLLM/NPU acceptance."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from demoncat_fault39.vllm_worker import Fault39WorkerExtension,install_runner_scope,scheduler_context


class Provider:
    def __init__(self):
        self.active=None;self.calls=[];self.snapshot_calls=0;self.fail_snapshots=set()
        self.configure_calls=[]
    def snapshot(self):
        self.snapshot_calls+=1
        if self.snapshot_calls in self.fail_snapshots:
            raise RuntimeError('snapshot failed')
        return dict(mock=True,owner='',enabled=False,inflight=0,leases=0,candidates=['inventory'])
    def prepare(self):return self.snapshot()
    def configure(self,*args,**kwargs):
        self.configure_calls.append((args,kwargs));return 1
    @contextmanager
    def context(self,owner,request_ids):
        self.active=(owner,request_ids);self.calls.append(self.active)
        try:yield
        finally:self.active=None


class Capture:
    def __init__(self,fail=()):self.fail=set(fail);self.calls=[]
    def _call(self,name):
        self.calls.append(name)
        if name in self.fail:raise RuntimeError('capture '+name+' failed')
    def start(self,**kwargs):self._call('start')
    def stop(self):self._call('stop')
    def step(self):self._call('step')


class Tests(unittest.TestCase):
    def fixture(self,max_contexts=256):
        provider=Provider()
        class Runner:
            model=object()
            calls=0
            def execute_model(self,scheduler_output,intermediate_tensors=None):
                self.calls+=1
                if intermediate_tensors=='fail':raise RuntimeError('model failed')
                return 'original_result'
        runner=Runner();worker=SimpleNamespace(rank=1,model_runner=runner,
            _fault39_provider=provider,_fault39_device=0)
        install_runner_scope(worker,provider,max_contexts=max_contexts)
        worker._fault39_owner='owner'
        scheduled=SimpleNamespace(num_scheduled_tokens={'actual-id':1},total_num_scheduled_tokens=1)
        return worker,provider,runner,scheduled

    def test_mapping_rejects_missing_or_inconsistent_counts(self):
        for item in (SimpleNamespace(),SimpleNamespace(num_scheduled_tokens={'a':2},total_num_scheduled_tokens=3)):
            with self.assertRaisesRegex(RuntimeError,'UNPROVEN'):scheduler_context(item)
        result=scheduler_context(SimpleNamespace(num_scheduled_tokens={'a':2,'b':1},total_num_scheduled_tokens=3))
        self.assertEqual(result['mapping'],'SCHEDULER_REQUEST_SET_NOT_ROW_ATTRIBUTION')

    def test_real_original_executes_once_and_context_clears_after_error(self):
        provider=Provider()
        class Runner:
            def execute_model(self,scheduler_output,intermediate_tensors=None):
                self.seen=provider.active
                if intermediate_tensors=='fail':raise RuntimeError('model failed')
                return 'original_result'
        runner=Runner();worker=SimpleNamespace(rank=1,model_runner=runner)
        install_runner_scope(worker,provider)
        scheduled=SimpleNamespace(num_scheduled_tokens={'actual-scheduler-id':1},total_num_scheduled_tokens=1)
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        self.assertIsNone(runner.seen)
        worker._fault39_owner='owner'
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        self.assertEqual(runner.seen,('owner',['actual-scheduler-id']))
        with self.assertRaisesRegex(RuntimeError,'model failed'):runner.execute_model(scheduled,'fail')
        self.assertIsNone(provider.active)
        self.assertEqual(len(provider.calls),2)
        self.assertFalse(worker._fault39_contexts[-1]['completed'])

    def test_snapshot_errors_never_poison_scope_or_mask_model(self):
        for failure in (1,2):
            with self.subTest(snapshot=failure):
                worker,provider,runner,scheduled=self.fixture()
                provider.fail_snapshots={failure}
                with self.assertRaisesRegex(RuntimeError,'model failed'):
                    runner.execute_model(scheduled,'fail')
                self.assertIsNone(provider.active)
                self.assertFalse(worker._fault39_contexts[-1]['instrumentation_complete'])
                self.assertEqual(runner.execute_model(scheduled),'original_result')
                self.assertTrue(worker._fault39_contexts[-1]['instrumentation_complete'])
                self.assertEqual(runner.calls,2)

    def test_capture_failures_cleanup_independently_and_preserve_model_error(self):
        worker,provider,runner,scheduled=self.fixture()
        capture=Capture(('start','stop','step'));worker._fault39_capture=capture
        with self.assertRaisesRegex(RuntimeError,'model failed'):
            runner.execute_model(scheduled,'fail')
        self.assertEqual(capture.calls,['start','stop','step'])
        self.assertIsNone(provider.active)
        record=worker._fault39_contexts[-1]
        self.assertFalse(record['completed'])
        self.assertFalse(record['instrumentation_complete'])
        self.assertEqual([e['operation'] for e in record['instrumentation_errors']],
                         ['capture_start','capture_stop','capture_step'])
        capture.fail.clear()
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        self.assertEqual(runner.calls,2)

    def test_collector_failure_preserves_successful_model_result(self):
        worker,provider,runner,scheduled=self.fixture()
        worker._fault39_capture=Capture(('start',))
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        record=worker._fault39_contexts[-1]
        self.assertTrue(record['completed'])
        self.assertFalse(record['instrumentation_complete'])
        self.assertFalse(record['capture_started'])
        self.assertEqual(worker._fault39_capture.calls,['start','stop','step'])

    def test_context_history_is_bounded_and_candidates_are_not_repeated(self):
        worker,provider,runner,scheduled=self.fixture(max_contexts=2)
        for _ in range(5):runner.execute_model(scheduled)
        self.assertEqual([r['index'] for r in worker._fault39_contexts],[3,4])
        self.assertEqual(worker._fault39_contexts_dropped,3)
        self.assertNotIn('candidates',worker._fault39_contexts[-1]['native_before'])
        self.assertNotIn('candidates',worker._fault39_contexts[-1]['native_after'])

    def test_finish_capture_excludes_later_off_without_changing_owner_or_wrapper(self):
        worker,provider,runner,scheduled=self.fixture()
        capture=Capture();worker._fault39_capture=capture
        worker._fault39_capture_owner='owner'
        wrapper=runner.execute_model
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        self.assertEqual(capture.calls,['start','stop','step'])
        result=Fault39WorkerExtension.fault39_rpc(worker,'finish_capture',{'owner':'owner'})
        self.assertEqual(result['status'],'CAPTURE_FINISHED')
        self.assertEqual(capture.calls,['start','stop','step','stop'])
        self.assertIsNone(worker._fault39_capture)
        self.assertEqual(worker._fault39_owner,'owner')
        self.assertIs(runner.execute_model,wrapper)
        self.assertEqual(provider.configure_calls,[])
        self.assertEqual(runner.execute_model(scheduled),'original_result')
        self.assertEqual(capture.calls,['start','stop','step','stop'])
        self.assertNotIn('capture_started',worker._fault39_contexts[-1])
        self.assertTrue(worker._fault39_contexts[-1]['completed'])
        again=Fault39WorkerExtension.fault39_rpc(worker,'finish_capture',{'owner':'owner'})
        self.assertTrue(again['already_finished'])
        # Existing native-clean prerequisite and original-method restoration
        # remain owned by detach, not by finish_capture.
        original=worker._fault39_runner_binding[1]
        self.assertEqual(Fault39WorkerExtension.fault39_rpc(worker,'detach')['status'],'DETACHED')
        self.assertIs(runner.execute_model,original)

    def test_foreign_owner_cannot_finish_active_capture(self):
        worker,provider,runner,scheduled=self.fixture()
        capture=Capture();worker._fault39_capture=capture
        worker._fault39_capture_owner='owner'
        with self.assertRaisesRegex(RuntimeError,'CAPTURE_OWNER_MISMATCH'):
            Fault39WorkerExtension.fault39_rpc(worker,'finish_capture',{'owner':'foreign'})
        self.assertIs(worker._fault39_capture,capture)
        self.assertEqual(capture.calls,[])

    def test_finish_failure_is_not_reported_as_stopped(self):
        worker,provider,runner,scheduled=self.fixture()
        capture=Capture(('stop',));worker._fault39_capture=capture
        result=Fault39WorkerExtension.fault39_rpc(worker,'finish_capture',{'owner':'owner'})
        self.assertEqual(result['status'],'CAPTURE_FINISH_FAILED')
        self.assertIsNone(worker._fault39_capture)
        self.assertFalse(worker._fault39_capture_finished)
        self.assertEqual(worker._fault39_owner,'owner')

    def test_broadcast_non_target_reads_only_and_returns_explicit_status(self):
        provider=Provider();worker=SimpleNamespace(rank=1,_fault39_provider=provider)
        result=Fault39WorkerExtension.fault39_rpc(worker,'configure',{'dispatch_rank':0})
        self.assertEqual(result['status'],'RPC_NOT_TARGETED')
        self.assertEqual(result['rank'],1)
        self.assertEqual(provider.configure_calls,[])
        self.assertFalse(hasattr(worker,'_fault39_runner_binding'))

    def test_targeted_configure_still_checks_actual_rank_and_device(self):
        worker,provider,runner,scheduled=self.fixture()
        for rank,device in ((0,0),(1,2)):
            with self.assertRaisesRegex(RuntimeError,'RPC_WORKER_IDENTITY_MISMATCH'):
                Fault39WorkerExtension.fault39_rpc(worker,'configure',
                    {'dispatch_rank':1,'rank':rank,'device':device})
        self.assertEqual(provider.configure_calls,[])

    def test_detach_snapshot_does_not_reattach_and_explicit_prepare_does(self):
        worker,provider,runner,scheduled=self.fixture()
        original=worker._fault39_runner_binding[1]
        result=Fault39WorkerExtension.fault39_rpc(worker,'detach')
        self.assertEqual(result['status'],'DETACHED')
        self.assertIs(runner.execute_model,original)
        state=Fault39WorkerExtension.fault39_rpc(worker,'snapshot')
        self.assertFalse(state['runner_scope_installed'])
        self.assertIs(runner.execute_model,original)
        self.assertEqual(Fault39WorkerExtension.fault39_rpc(worker,'detach')['status'],'DETACHED')
        Fault39WorkerExtension.fault39_rpc(worker,'prepare')
        self.assertIsNot(runner.execute_model,original)
        self.assertIsNone(worker._fault39_owner)
        self.assertEqual(runner.execute_model(scheduled),'original_result')

    def test_dispatch_and_payload_types_are_not_silently_coerced(self):
        worker,provider,runner,scheduled=self.fixture()
        for payload in ([],0,{'dispatch_rank':True},{'dispatch_rank':-1}):
            with self.assertRaises(ValueError):
                Fault39WorkerExtension.fault39_rpc(worker,'snapshot',payload)


if __name__=='__main__':unittest.main()
