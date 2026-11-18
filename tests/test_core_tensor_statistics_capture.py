"""Actual public Host/Runtime CPU protocol with an explicit fake collector.

These contracts exercise entry/return order and cleanup, not msprobe output or
NPU qualification. The production source-pinned hook verifier is separate.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_core_tensor_serving as core
from experiments.worker_core_v11_20261118 import core_tensor_statistics_capture as capture


class FakeCollector:
    def __init__(self):
        self.active = self.enabled = False
        self.rows = []
        self.observed = []
        self.pauses = 0
        self.inject_unknown = False
        self.change_arguments = False
        self.inject_root = False
        self.fail_root = False
        self.foreign = None
        self.orphan_flag = None
        self.inject_orphan_flag = False
        self.collector = SimpleNamespace(service=SimpleNamespace(hooked_modules=[]))

    def begin(self, model, phase):
        assert not self.active
        self.model = model
        assert self.collector.service.hooked_modules == []
        self.collector.service.hooked_modules.append(model)
        self.active = self.enabled = True
        self.rows.append(phase)
        for name, module in model.named_modules():
            if name or self.inject_root:
                self._install(module, name or '<root>')
        if self.inject_unknown:
            self.foreign = model.register_forward_pre_hook(lambda module, args: None)
        if self.inject_orphan_flag:
            child = next(module for name, module in model.named_modules() if name)
            child._forward_hooks_with_kwargs[2**60] = True
            self.orphan_flag = child

    def _install(self, module, name):
        parent = self
        def pre(actual, args, kwargs):
            parent.observed.append(('input', name, parent.enabled))
            if not hasattr(actual, 'msprobe_forward_hook'):
                actual.register_forward_hook(post, with_kwargs=True, prepend=True)
                actual.msprobe_forward_hook = True
            if parent.fail_root:
                raise RuntimeError('original collector failed')
            return (args, dict(kwargs)) if name == '<root>' and parent.change_arguments else (args, kwargs)
        def post(actual, args, kwargs, output):
            parent.observed.append(('output', name, parent.enabled))
            return output
        module.register_forward_pre_hook(pre, with_kwargs=True)
        module.msprobe_hook = True
        return pre, post

    def end(self, *, completed):
        assert self.active
        self.active = self.enabled = False

    @contextmanager
    def diagnostics(self):
        previous = self.enabled
        if previous:
            self.pauses += 1
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


class FakeVerifier:
    def verify(self, function, kind):
        expected = 'pre' if kind == 'forward_pre_hook' else 'post'
        allowed = [code for code in capture.codes(FakeCollector._install.__code__) if code.co_name == expected]
        capture.require(len(allowed) == 1 and function.__code__ is allowed[0], 'UNKNOWN_COLLECTOR_HOOK_CODE')
        return dict(cpu_fake=True, kind=kind, code=capture.callable_identity(function))


class PublicStatisticsCaptureTests(unittest.TestCase):
    def setUp(self):
        self.case = core.CoreTensorServingTests()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.case.start()
        self.session = self.case.provider.session
        self.collector = FakeCollector()
        self.sidecar = None
        self.addCleanup(self.cleanup_sidecar)
        root = Path(__file__).resolve().parents[1]
        self.pins = {str(root/name): capture.sha(root/name) for name in capture.RUNTIME_FILES}

    def cleanup_sidecar(self):
        if self.collector.foreign is not None:
            self.collector.foreign.remove()
        if self.collector.orphan_flag is not None:
            self.collector.orphan_flag._forward_hooks_with_kwargs.pop(2**60, None)
        if self.sidecar:
            self.sidecar.close()

    def attach(self):
        self.sidecar = capture.PublicTensorStatisticsSidecar(self.session, self.folder,
            source_pins=self.pins, _capture=self.collector, _hook_verifier=FakeVerifier())
        return self.sidecar

    def test_real_public_entry_three_windows_keep_original_protocol_and_cleanup(self):
        observer = self.attach()
        for index in range(10):
            self.case.request('statistics-actual-'+str(index))
        state = self.case.ok(self.case.client.finish())
        self.assertEqual((state['coverage']['C2']['hits'], state['coverage']['C3']['hits']), (2, 4))
        self.assertEqual(self.collector.rows, ['statistics_baseline', 'statistics_baseline_repeat', 'statistics_fault'])
        self.assertEqual([r['actual_step'] for r in observer.rows], [0, 1, 2])
        self.assertTrue(all(r['root_hooks_unchanged'] and r['owned_hook_restore_exact'] and r['completed']
                            for r in observer.rows))
        self.assertTrue(all(enabled for _, _, enabled in self.collector.observed))
        self.assertGreater(self.collector.pauses, 0)
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(self.session.statistics_capture)
        self.assertFalse(observer.report()['e5'])
        self.assertEqual((self.case.model.forward_calls, self.case.model.logits_calls), (10, 10))
        self.case.ok(self.case.client.unregister())
        self.case.assert_clean()

    def test_capture_begins_only_after_real_begin_not_queued_rpc(self):
        observer = self.attach()
        self.case.request('native-0')
        self.case.ok(self.case.client.advance('baseline-1'))
        self.assertEqual(self.collector.rows, [])
        self.assertIsNone(observer.before)
        self.case.actual('baseline-1')
        self.case.ok(self.case.client.collect_replay())
        self.assertEqual(self.collector.rows, ['statistics_baseline'])

    def test_changed_argument_objects_refused_without_mutating_core(self):
        observer = self.attach()
        self.case.request('native-0')
        self.collector.change_arguments = True
        self.collector.inject_root = True
        self.case.ok(self.case.client.advance('baseline-1'))
        with self.assertRaisesRegex(RuntimeError, 'COLLECTOR_ROOT_HOOK_INVENTORY_CHANGED'):
            self.case.actual('baseline-1')
        self.assertFalse(observer.rows[0]['completed'])
        self.assertFalse(self.collector.observed)  # The unexpected hook was never invoked.
        for key in list(self.case.model._forward_pre_hooks):
            if key not in observer.before['<root>']['hooks']['_forward_pre_hooks']:
                del self.case.model._forward_pre_hooks[key]
                self.case.model._forward_pre_hooks_with_kwargs.pop(key, None)
        self.collector.change_arguments = False

    def test_original_collector_exception_is_not_swallowed(self):
        observer = self.attach()
        self.case.request('native-0')
        self.collector.fail_root = True
        self.case.ok(self.case.client.advance('baseline-1'))
        with self.assertRaisesRegex(RuntimeError, 'original collector failed'):
            self.case.actual('baseline-1')
        self.assertFalse(observer.rows[0]['completed'])
        self.collector.fail_root = False
        self.case.recovery()

    def test_unknown_hook_refused_and_not_deleted_as_owned(self):
        observer = self.attach()
        self.case.request('native-0')
        self.collector.inject_unknown = True
        self.case.ok(self.case.client.advance('baseline-1'))
        with self.assertRaisesRegex(RuntimeError, 'COLLECTOR_ROOT_HOOK_INVENTORY_CHANGED'):
            self.case.actual('baseline-1')
        self.assertIn(self.collector.foreign.id, self.case.model._forward_pre_hooks)
        self.assertFalse(observer.rows[0]['completed'])

    def test_existing_profile_is_not_replaced(self):
        callback = lambda *args: None
        sys.setprofile(callback)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FOREIGN_PROFILE_OBSERVER_PRESENT'):
                self.attach()
            self.assertIs(sys.getprofile(), callback)
        finally:
            sys.setprofile(None)

    def test_foreign_orphan_flag_is_refused_and_not_deleted(self):
        observer = self.attach();self.case.request('native-0')
        self.collector.inject_orphan_flag = True
        self.case.ok(self.case.client.advance('baseline-1'))
        with self.assertRaisesRegex(RuntimeError, 'FOREIGN_OR_ORPHAN_COLLECTOR_HOOK_FLAG'):
            self.case.actual('baseline-1')
        self.assertIn(2**60, self.collector.orphan_flag._forward_hooks_with_kwargs)
        self.assertFalse(observer.rows[0]['completed'])

    def test_unmatched_runtime_source_refuses_before_installation(self):
        self.pins = {name: '0'*64 for name in self.pins}
        with self.assertRaisesRegex(RuntimeError, 'STATISTICS_RUNTIME_SOURCE_CHANGED'):
            self.attach()
        self.assertIsNone(sys.getprofile())
        self.assertIsNone(self.session.statistics_capture)

    def test_fault_model_exception_keeps_partial_capture_and_releases_owned_hooks(self):
        observer = self.attach()
        for index in range(4):
            self.case.request('pre-'+str(index))
        self.case.model.fail_forward = True
        self.case.ok(self.case.client.advance('failed-fault'))
        with self.assertRaisesRegex(RuntimeError, 'CPU actual root failed'):
            self.case.actual('failed-fault')
        self.case.model.fail_forward = False
        with self.assertRaisesRegex(RuntimeError, 'CORE_OWNED_COMPLETION_REQUIRES_CORE_RECOVERY'):
            observer.close()
        self.case.recovery()
        observer.close()
        self.assertEqual([r['completed'] for r in observer.rows], [True, True, False])
        self.assertIsNone(sys.getprofile())
        self.assertFalse(observer.report()['hook_snapshot_retained'])
        self.case.assert_clean()


class PinnedHookIdentityTests(unittest.TestCase):
    """Synthetic source files test code/globals/closure checking, not msprobe."""
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        utility = '''import functools
import threading
class ThreadSafe:
    _lock = threading.RLock()
    @classmethod
    def synchronized(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with cls._lock:
                return func(*args, **kwargs)
        return wrapper
'''
        processor = '''class ModuleProcessor:
    def __init__(self): self.calls = []
    def build_module_hook(self, module_name):
        @ThreadSafe.synchronized
        def forward_pre_hook(module, args, kwargs):
            self.calls.append(module_name)
            return args, kwargs
        @ThreadSafe.synchronized
        def forward_hook(module, args, kwargs, output):
            self.calls.append(module_name)
            return output
        return forward_pre_hook, forward_hook
def unknown_hook(*args): return None
'''
        self.modules, self.pins = {}, {}
        for relative, name, text in [('core/common/utils.py', 'cpu_msprobe_utils', utility),
                ('pytorch/dump/module_dump/module_processor.py', 'cpu_msprobe_processor', processor)]:
            path = self.root/relative;path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            module = ModuleType(name);module.__file__ = str(path)
            if 'processor' in name:
                module.ThreadSafe = self.modules['cpu_msprobe_utils'].ThreadSafe
            exec(compile(path.read_bytes(), str(path), 'exec'), vars(module))
            self.modules[name] = module;self.pins[relative] = capture.sha(path)
        self.enterContext(patch.dict(sys.modules, self.modules))
        self.processor = self.modules['cpu_msprobe_processor'].ModuleProcessor()
        self.pre, self.post = self.processor.build_module_hook('Module.exact.Root.forward')
        self.verifier = capture.PinnedModuleHooks(self.root, self.pins)

    def test_classmethod_synchronized_chain_and_both_exact_hook_codes(self):
        self.assertEqual(self.verifier.verify(self.pre, 'forward_pre_hook')['processor_id'], id(self.processor))
        self.assertEqual(self.verifier.verify(self.post, 'forward_hook')['processor_id'], id(self.processor))

    def test_same_source_unknown_function_cannot_become_owned_hook(self):
        with self.assertRaisesRegex(RuntimeError, 'UNKNOWN_COLLECTOR_HOOK_CODE'):
            self.verifier.verify(self.modules['cpu_msprobe_processor'].unknown_hook, 'forward_pre_hook')

    def test_changed_source_rejected_even_if_executed_code_did_not_change(self):
        path = self.root/'core/common/utils.py'
        path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'COLLECTOR_HOOK_SOURCE_BYTES_CHANGED'):
            self.verifier.verify(self.pre, 'forward_pre_hook')

    def test_wrapped_attribute_must_be_the_actual_closed_function(self):
        self.pre.__wrapped__ = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, 'COLLECTOR_DECORATOR_ORIGINAL_NOT_BOUND'):
            self.verifier.verify(self.pre, 'forward_pre_hook')

    def test_different_live_processor_not_adopted(self):
        self.verifier.verify(self.pre, 'forward_pre_hook')
        foreign, _ = self.modules['cpu_msprobe_processor'].ModuleProcessor().build_module_hook('Module.other')
        with self.assertRaisesRegex(RuntimeError, 'FOREIGN_COLLECTOR_PROCESSOR'):
            self.verifier.verify(foreign, 'forward_pre_hook')


if __name__ == '__main__':
    unittest.main()
