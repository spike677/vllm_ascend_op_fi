"""Actual CPU behavior and failure recovery; no NPU or inner-host certification."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from fault_injection import attach, UnsupportedCall
from fault_injection.catalog import resolve
from fault_injection.lifecycle import FaultConfig, ModelFaultBackend
from fault_injection.session import ModelFaultSession


class AdapterFixture:
    def __init__(self):
        self.active = False
        self.closed = False
        self.failure = None
        self.close_count = 0

    def __enter__(self):
        self.active = True
        return self

    def close(self):
        self.close_count += 1
        if self.failure:
            raise RuntimeError(self.failure)
        self.active, self.closed = False, True

    def report(self):
        return {'fixture': True, 'adapter_active': self.active}

    def assert_coverage(self):
        return self.report()


class RemoveRuntime:
    """Only installation/removal plumbing; never invokes a native operator."""
    def __init__(self):
        self.remove_fails = True
        self.installed = False
        self.removals = 0

    def install(self, *args):
        self.installed = True
        def remove():
            self.removals += 1
            if self.remove_fails:
                raise RuntimeError('native drain pending')
            self.installed = False
        return remove

    def check(self, *args): raise AssertionError('No native call in this fixture')
    def execute(self, *args): raise AssertionError('No native call in this fixture')
    def finish(self, *args): raise AssertionError('No native call in this fixture')
    def abort(self, *args): raise AssertionError('No native call in this fixture')


class LifecycleTests(unittest.TestCase):
    def test_registration_disabled_and_each_enable_is_new_generation(self):
        model = torch.nn.Sequential(torch.nn.RMSNorm(8)).eval()
        x = torch.arange(16, dtype=torch.float32).reshape(2,8)
        normal = model(x)
        backend = ModelFaultBackend(model)
        self.assertEqual(backend.prepare()['state'], 'PREPARED')
        backend.register(FaultConfig('18.api_half'))
        self.assertTrue(torch.equal(model(x), normal))
        self.assertFalse(model._forward_hooks)
        ids=[]
        for generation in (1,2,3):
            status=backend.enable();ids.append(status['session']['session_id'])
            self.assertEqual(status['generation'], generation)
            self.assertTrue(torch.equal(model(x), normal*.5))
            backend.assert_coverage()
            status=backend.disable()
            self.assertTrue(status['cleanup_confirmed'])
            self.assertTrue(torch.equal(model(x), normal))
            self.assertFalse(model._forward_hooks)
        self.assertEqual(len(set(ids)),3)
        self.assertEqual(backend.unregister()['state'],'UNREGISTERED')

    def test_actual_conv3d_baseline_fault_off_without_layer_selection(self):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.arbitrary_branch=torch.nn.Sequential(torch.nn.Conv3d(2,3,(2,16,16),stride=(2,16,16),bias=False))
                self.arbitrary_branch[0].weight.data.fill_(1)
            def forward(self,x): return self.arbitrary_branch(x)
        model=Vision().eval();x=torch.ones(2,2,2,16,16);expected=model(x)
        weights=model.arbitrary_branch[0].weight.detach().clone()
        for arm in ('baseline','fault'):
            backend=ModelFaultBackend(model)
            backend.register({'mechanism':'17.spatial_padding','arm':arm})
            backend.enable();observed=model(x);backend.assert_coverage()
            self.assertEqual(tuple(observed.shape),tuple(expected.shape))
            self.assertEqual(torch.equal(expected,observed),arm=='baseline')
            backend.clean()
            self.assertTrue(torch.equal(model(x),expected))
            self.assertTrue(torch.equal(model.arbitrary_branch[0].weight,weights))

    def test_invalid_and_unconsumed_configuration_is_rejected(self):
        for fields in ({'mechanism':True},{'mechanism':'18.api_half','arm':True},
                       {'mechanism':'18.api_half','unsupported':0},
                       {'mechanism':'18.api_half','target':'layer.0'},
                       {'mechanism':'18.api_half','ranks':[0]},
                       {'mechanism':'18.api_half','cards':[2]},
                       {'mechanism':'18.api_half','op':'rmsnorm'},
                       {'mechanism':'18.api_half','evidence_dir':False}):
            with self.subTest(fields=fields),self.assertRaises((TypeError,ValueError)):
                FaultConfig.from_dict(fields)
        for fault in ('39.matmul_range','39.output_proxy'):
            with self.assertRaisesRegex(ValueError,'separately validated'):
                FaultConfig(fault)

    def test_blocked_and_missing_provider_never_install_hooks(self):
        model=torch.nn.Linear(4,4).eval();x=torch.ones(1,4);original=model(x)
        for fault in ('8.restore_offset','22.wrong_row','25.reduce_scatter','41.tail_column','20.buffer_substitute'):
            with self.subTest(fault=fault):
                backend=ModelFaultBackend(model);backend.register(FaultConfig(fault))
                with self.assertRaises(UnsupportedCall):backend.enable()
                self.assertEqual(backend.query()['state'],'ENABLE_FAILED')
                self.assertTrue(backend.query()['cleanup_confirmed'])
                self.assertFalse(model._forward_hooks)
                self.assertTrue(torch.equal(model(x),original))
                backend.clean()

    def test_wrong_thread_control_does_not_disable_owner(self):
        model=torch.nn.RMSNorm(8).eval();x=torch.ones(1,8);normal=model(x)
        backend=ModelFaultBackend(model);backend.register(FaultConfig('18.api_half'));backend.enable()
        errors=[]
        def wrong():
            try:backend.disable()
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=wrong);thread.start();thread.join()
        self.assertEqual(len(errors),1)
        self.assertEqual(backend.query()['state'],'ENABLED')
        self.assertTrue(torch.equal(model(x),normal*.5))
        backend.clean();self.assertTrue(torch.equal(model(x),normal))

    def test_second_fault_rejected_and_its_clean_does_not_touch_first(self):
        model=torch.nn.RMSNorm(8).eval();x=torch.ones(1,8);normal=model(x)
        first,second=ModelFaultBackend(model),ModelFaultBackend(model)
        for backend in (first,second):backend.register(FaultConfig('18.api_half'))
        first.enable()
        try:
            with self.assertRaisesRegex(RuntimeError,'Another unified'):second.enable()
            second.clean()
            self.assertTrue(torch.equal(model(x),normal*.5))
        finally:first.clean()
        self.assertTrue(torch.equal(model(x),normal))

    def test_global_lease_retained_and_cleanup_retried(self):
        adapter=AdapterFixture();adapter.failure='still in flight'
        session=ModelFaultSession(adapter,resolve('18.api_half'),'fault')
        session.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError,'still in flight'):session.close()
            self.assertFalse(session.report()['closed'])
            self.assertFalse(session.report()['cleanup_completed'])
            with self.assertRaisesRegex(RuntimeError,'Another unified'):
                ModelFaultSession(AdapterFixture(),resolve('18.api_half'),'fault').__enter__()
        finally:
            adapter.failure=None;session.close()
        self.assertEqual(adapter.close_count,2)
        self.assertTrue(session.report()['cleanup_completed'])
        self.assertIn('still in flight',session.report()['session_error'])

    def test_install_failure_rollback_is_confirmed_before_release(self):
        class Partial(AdapterFixture):
            def __enter__(self):
                self.active=True
                raise RuntimeError('partial install')
        adapter=Partial();adapter.failure='cleanup unavailable'
        session=ModelFaultSession(adapter,resolve('18.api_half'),'fault')
        try:
            with self.assertRaisesRegex(RuntimeError,'partial install.*cleanup unavailable'):session.__enter__()
            self.assertTrue(session.report()['cleanup_pending'])
            with self.assertRaisesRegex(RuntimeError,'Another unified'):
                ModelFaultSession(AdapterFixture(),resolve('18.api_half'),'fault').__enter__()
        finally:adapter.failure=None;session.close()

    def test_native_provider_removal_retained_and_backend_owner_retries(self):
        model=torch.nn.Linear(4,4).eval();runtime=RemoveRuntime()
        backend=ModelFaultBackend(model,runtime=runtime)
        backend.register(FaultConfig('20.buffer_substitute'));backend.enable()
        try:
            with self.assertRaisesRegex(RuntimeError,'native drain pending'):backend.disable()
            status=backend.query()
            self.assertEqual(status['state'],'RECOVERY_FAILED')
            self.assertFalse(status['cleanup_confirmed'])
            self.assertTrue(runtime.installed)
            with self.assertRaisesRegex(UnsupportedCall,'cleanup is pending'):model(torch.ones(1,4))
            with self.assertRaises(RuntimeError):backend.enable()
        finally:
            runtime.remove_fails=False;backend.clean()
        status=backend.query()
        self.assertEqual(status['state'],'UNREGISTERED')
        self.assertFalse(runtime.installed)
        self.assertEqual(runtime.removals,2)
        self.assertIn('native drain pending',status['last_error'])
        self.assertIn('native drain pending',status['session']['session_error'])
        self.assertFalse(model._forward_hooks)

    def test_failed_cleanup_receipt_not_overwritten_on_recovery(self):
        adapter=AdapterFixture();adapter.failure='not yet restored'
        with tempfile.TemporaryDirectory() as directory:
            session=ModelFaultSession(adapter,resolve('18.api_half'),'fault',directory)
            session.__enter__()
            with self.assertRaises(RuntimeError):session.close()
            first=next(Path(directory).glob('session-*.json'));raw=first.read_bytes()
            adapter.failure=None;session.close()
            self.assertEqual(first.read_bytes(),raw)
            rows=[json.loads(p.read_text()) for p in Path(directory).glob('session-*.json')]
            self.assertEqual(len(rows),2)
            self.assertEqual({r['cleanup_completed'] for r in rows},{False,True})

    def test_model_and_cleanup_failures_both_persist_after_recovery(self):
        adapter=AdapterFixture();adapter.failure='restore also failed'
        with tempfile.TemporaryDirectory() as directory:
            session=ModelFaultSession(adapter,resolve('18.api_half'),'fault',directory)
            try:
                with self.assertRaisesRegex(RuntimeError,'model failed first.*restore also failed'):
                    with session:raise ValueError('model failed first')
            finally:
                adapter.failure=None;session.close()
            report=session.report()
            self.assertIn('model failed first',report['session_error'])
            self.assertIn('restore also failed',report['session_error'])
            self.assertEqual([x['stage'] for x in report['error_history']],['model','cleanup'])
            saved=json.loads(next(Path(directory).glob('*revision1.json')).read_text())
            self.assertEqual(saved['error_history'],report['error_history'])

    def test_report_write_failure_can_retry_without_rearming(self):
        model=torch.nn.RMSNorm(8).eval()
        with tempfile.TemporaryDirectory() as directory:
            backend=ModelFaultBackend(model);backend.register(FaultConfig('18.api_half',evidence_dir=directory));backend.enable()
            with patch.object(ModelFaultSession,'_save_report',side_effect=OSError('evidence disk error')):
                with self.assertRaises(OSError):backend.disable()
            self.assertEqual(backend.query()['state'],'DISABLED_CONFIRMED')
            self.assertTrue(backend.query()['cleanup_confirmed'])
            self.assertFalse(model._forward_hooks)
            backend.clean()
            self.assertEqual(len(list(Path(directory).glob('session-*.json'))),1)

    def test_in_forward_disable_refuses_then_owner_can_retry_after_return(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.RMSNorm(8);self.backend=None;self.error=None
            def forward(self,x):
                try:self.backend.disable()
                except RuntimeError as error:self.error=str(error)
                return self.norm(x)
        model=Model().eval();backend=ModelFaultBackend(model);model.backend=backend
        backend.register(FaultConfig('18.api_half'));backend.enable()
        try:
            model(torch.ones(1,8))
            self.assertIsNotNone(model.error)
            self.assertFalse(backend.query()['cleanup_confirmed'])
        finally:backend.clean()
        self.assertTrue(backend.query()['cleanup_confirmed'])

    def test_irrecoverable_closed_adapter_is_not_a_second_close_success(self):
        # Permanent quarantine is exercised in a child process so no subsequent
        # test can accidentally clear or bypass its retained global lease.
        code='''from fault_injection.session import ModelFaultSession
from fault_injection.catalog import resolve
class Bad:
 def __enter__(self):return self
 def report(self):return {}
 def close(self):
  self.closed=True
  raise RuntimeError('lost cleanup handles')
s=ModelFaultSession(Bad(),resolve('18.api_half'),'fault');s.__enter__()
for i in range(2):
 try:s.close()
 except RuntimeError:pass
 else:raise AssertionError('false cleanup success')
assert s.report()['quarantined'] and not s.report()['cleanup_completed']
try:ModelFaultSession(Bad(),resolve('18.api_half'),'fault').__enter__()
except RuntimeError:pass
else:raise AssertionError('admitted while cleanup unknown')
'''
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_unknown_native_installation_quarantines_process(self):
        # The provider may change state before losing its install reply. No
        # removal capability means Python hook cleanup cannot certify recovery.
        code='''import sys
import torch
from fault_injection.lifecycle import FaultConfig, ModelFaultBackend
mode=sys.argv[1]
class PartialProvider:
 def __init__(self):self.installed=False
 def install(self,*args):
  self.installed=True
  if mode=='raise':raise RuntimeError('installation reply lost')
  return object()
 def check(self,*args):raise AssertionError('No operator call allowed')
 def execute(self,*args):raise AssertionError('No operator call allowed')
 def finish(self,*args):raise AssertionError('No operator call allowed')
 def abort(self,*args):raise AssertionError('No operator call allowed')
runtime=PartialProvider();model=torch.nn.Linear(2,2).eval()
backend=ModelFaultBackend(model,runtime=runtime)
backend.register(FaultConfig('20.buffer_substitute'))
try:backend.enable()
except RuntimeError as exc:assert 'cleanup unknown' in str(exc)
else:raise AssertionError('unknown installation accepted')
assert runtime.installed
status=backend.query()
assert status['state']=='RECOVERY_FAILED' and not status['cleanup_confirmed']
assert status['session']['installation_cleanup_unknown']
for action in (backend.disable,backend.clean,backend.unregister):
 try:action()
 except RuntimeError as exc:assert 'process restart required' in str(exc)
 else:raise AssertionError('unknown native state declared restored')
 assert not backend.query()['cleanup_confirmed']
 assert backend.query()['registered']
 assert runtime.installed
other=ModelFaultBackend(torch.nn.RMSNorm(2).eval())
other.register(FaultConfig('18.api_half'))
try:other.enable()
except RuntimeError as exc:assert 'Another unified fault session' in str(exc)
else:raise AssertionError('new fault admitted while native ownership unknown')
other.clean()
assert backend.query()['state']=='RECOVERY_FAILED'
try:model(torch.ones(1,2))
except RuntimeError as exc:assert 'cleanup is pending' in str(exc)
else:raise AssertionError('model execution admitted during unknown installation')
'''
        for mode in ('raise','noncallable'):
            with self.subTest(mode=mode):
                result=subprocess.run([sys.executable,'-B','-c',code,mode],
                    cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=45)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
