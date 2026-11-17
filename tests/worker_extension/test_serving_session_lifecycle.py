"""CPU-only owned-hook/lifecycle tests, never actual Worker or NPU evidence.

Real CPU torch modules exercise PyTorch's hook/always_call behavior. Scheduler,
native provider, backend, NPU-format query and reviewed deployment discovery are
explicit mocks. No mock result is a native hit or a serving qualification.
"""
from contextlib import contextmanager
import functools
import inspect
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

try:
    import torch
except (ImportError, OSError):
    torch = None

from demoncat.runtime.vllm_ascend.worker_extension import serving
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused
if __package__:
    from .test_call_binding import COLLECTOR_MODULE, collector_fixture, wrap_forward_with_hook_safety
else:
    from test_call_binding import COLLECTOR_MODULE, collector_fixture, wrap_forward_with_hook_safety


Module = torch.nn.Module if torch is not None else object


class CpuProcessor(Module):
    def __init__(self):
        super().__init__()
        self.raise_forward = False

    def forward(self, lm_head, hidden_states, embedding_bias=None):
        if self.raise_forward:
            raise RuntimeError('CPU_PROCESSOR_FORWARD_FAILED')
        return torch.nn.functional.linear(hidden_states, lm_head.weight, embedding_bias)


class CpuModel(Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.processor = CpuProcessor()
        self.raise_forward = self.nonfinite_hidden = False

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        if self.raise_forward:
            raise RuntimeError('CPU_ROOT_FORWARD_FAILED')
        output = torch.arange(input_ids.numel() * 4).reshape(-1, 4).to(torch.bfloat16)
        if self.nonfinite_hidden:
            output[-1, 0] = float('nan')
        return output


class MockNative:
    """Owner-scoped host fixture; never reads a device or invokes an operator."""
    def __init__(self):
        self.scope_owner = self.resource_owner = None
        self.enabled = False
        self.context_entries = self.context_exits = 0
        self.bindings = []
        self.disable_owners = []
        self.fail_bind = False

    @contextmanager
    def context(self, owner, request_ids):
        if self.scope_owner is not None:
            raise RuntimeError('MOCK_CONTEXT_ALREADY_OWNED')
        self.scope_owner = owner
        self.context_entries += 1
        try:
            yield
        finally:
            if self.scope_owner != owner:
                raise RuntimeError('MOCK_CONTEXT_OWNER_CHANGED')
            self.scope_owner = None
            self.context_exits += 1

    def bind_serving_inputs(self, owner, values):
        if self.scope_owner != owner or self.fail_bind:
            raise RuntimeError('MOCK_LIVE_INPUT_BIND_FAILED')
        self.bindings.append((owner, list(values)))

    def arm(self, owner):
        if self.scope_owner != owner or self.resource_owner not in (None, owner):
            raise RuntimeError('MOCK_NATIVE_FOREIGN_OWNER')
        self.resource_owner, self.enabled = owner, True

    def disable(self, owner):
        self.disable_owners.append(owner)
        if self.resource_owner not in (None, owner):
            return 'RECOVERY_FAILED'
        self.enabled = False
        return 'DISABLED_CONFIRMED'


class MockBackend:
    def __init__(self, native, owner):
        self.native, self.owner = native, owner
        self.phase = 'reference'
        self.failure = None
        self.recovering = self.recovered = self.fault_started = self.active = False
        self.drain_status = 'NOT_REQUESTED'
        self.requests = set()
        self.rows = []
        self.begin_calls = self.end_calls = 0
        self.fail_begin = self.fail_end = False

    def begin(self, request_id, input_sha256):
        self.begin_calls += 1
        self.requests.add(request_id)
        self.active = True
        self.fault_started = self.fault_started or self.phase == 'fault'
        self.native.arm(self.owner)
        if self.fail_begin:
            raise RuntimeError('MOCK_BEGIN_FAILED_AFTER_ARM')

    def end(self, digest, *, model_output_finite):
        self.end_calls += 1
        if self.fail_end:
            raise RuntimeError('MOCK_END_FAILED')
        self.active = False
        self.drain_status = self.native.disable(self.owner)
        self.rows.append(dict(phase=self.phase, model_output_sha256=digest,
                              model_output_finite=model_output_finite, evidence_kind='CPU_MOCK_ONLY'))

    def fail(self, reason):
        self.failure = str(reason)
        self.active = self.recovering = False
        self.drain_status = self.native.disable(self.owner)

    def query(self):
        return {'qualification': dict(status='CPU_MOCK_ONLY_NOT_NPU_ACCEPTANCE',
                                      failure=self.failure, recovery_verified=self.recovered)}


class MockCollector:
    def __init__(self):
        self.running = False
        self.starts = self.stops = self.steps = 0
        self.partial_start_failure = self.stop_failure = False
        self.wrap_forwards = False
        self.wrapped = []

    def start(self, **kwargs):
        self.starts += 1
        self.running = True
        if self.wrap_forwards:
            model = kwargs['model']
            for module in (model, model.processor):
                original = module.forward
                wrapped = wrap_forward_with_hook_safety(module, original)
                module.forward = wrapped
                self.wrapped.append((module, original, wrapped))
        if self.partial_start_failure:
            raise RuntimeError('MOCK_PARTIAL_COLLECTOR_START')

    def stop(self):
        self.stops += 1
        if self.stop_failure:
            raise RuntimeError('MOCK_COLLECTOR_STOP_FAILED')
        for module, original, wrapped in reversed(self.wrapped):
            if module.forward is not wrapped:
                raise RuntimeError('MOCK_COLLECTOR_WRAPPER_OWNER_CHANGED')
            module.forward = original
        self.wrapped.clear()
        self.running = False

    def step(self):
        self.steps += 1


@unittest.skipIf(torch is None, 'CPU torch is needed to exercise real hook exception semantics')
class ServingSessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory(prefix='serving-session-cpu-'))
        self.evidence_root = Path(self.temp) / 'evidence'
        self.sessions = []
        self.actual_request_id = 'cpu-request-1'
        self.enterContext(patch.dict('os.environ', {'VLLM_ENABLE_V1_MULTIPROCESSING': '0'}))
        modules = {name: ModuleType(name) for name in (
            'vllm', 'vllm.model_executor', 'vllm.model_executor.layers',
            'vllm.model_executor.layers.logits_processor', 'vllm.forward_context', 'torch_npu')}
        modules['vllm.model_executor.layers.logits_processor'].LogitsProcessor = CpuProcessor
        modules['vllm.forward_context'].get_forward_context = lambda: NS(evidence_kind='CPU_MOCK_ONLY')
        modules['torch_npu'].get_npu_format = lambda value: 2  # Deliberately mocked; every tensor is CPU.
        modules[COLLECTOR_MODULE] = collector_fixture()
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(serving, 'require_serial_eager'))
        self.enterContext(patch.object(serving, 'discover_logits_boundary', side_effect=lambda model, **kwargs: {
            'processor': model.processor, 'lm_head': model.head, 'branch_guard': None,
            'report': {'output_contract': [1, 3], 'evidence_kind': 'CPU_MOCK_ONLY'}}))
        # Preserve the constructor's absolute/relative path check while mapping
        # its fixed evidence root into this test's private temporary directory.
        self.enterContext(patch.object(serving, 'Path', side_effect=lambda value:
            self.evidence_root if str(value) == '/evidence' else Path(value)))
        self.enterContext(patch.object(serving, 'snapshot_fresh_prefill', side_effect=self.snapshot))

    def tearDown(self):
        # Fixture disposal is not a production OFF/recovery assertion.
        for session in self.sessions:
            for handle in session.handles:
                handle.remove()

    def snapshot(self, runner, model, actual, context, **kwargs):
        if kwargs['expected_internal_request_id'] != self.actual_request_id:
            raise ContextRefused('MOCK_ACTUAL_SCHEDULER_ID_MISMATCH')
        return {'status': 'CPU_MOCK_SCHEDULER_NOT_NPU_EVIDENCE',
                'internal_request_id': self.actual_request_id}

    def session(self, *, collector=None, install=True):
        model = CpuModel().eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0)
        native = MockNative()
        backend = MockBackend(native, 'cpu-owner')
        session = serving.ServingReplaySession(worker, model, backend, native, 'cpu-owner',
            prompt_token_ids=[1, 2, 3, 4],
            evidence_dir=self.evidence_root / ('session-' + str(len(self.sessions))), collector=collector)
        self.assertIs(session.root_binding.module, model)
        self.assertIs(session.logits_binding.module, model.processor)
        self.sessions.append(session)
        if install:
            session.install()
        return session

    def root(self, session):
        return session.model(input_ids=torch.tensor(session.prompt, dtype=torch.int32),
                             positions=torch.arange(len(session.prompt), dtype=torch.int64))

    def forward(self, session):
        hidden = self.root(session)
        return session.processor(session.head, hidden[-1:].contiguous())

    @staticmethod
    def thread_error(call):
        errors = []
        def run():
            try:
                call()
            except BaseException as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        return errors

    def assert_drained(self, session):
        self.assertFalse(session.native.enabled)
        self.assertFalse(session.backend.active)
        self.assertIsNone(session.native.scope_owner)
        self.assertIsNone(session.current)
        self.assertIsNone(session.context)
        self.assertFalse(session.logits_active)

    def test_constructor_then_close_without_execute_is_inert(self):
        session = self.session(install=False)
        session.close()
        self.assertFalse(session.installed)
        self.assertEqual(session.handles, [])
        self.assertEqual(session.backend.begin_calls, 0)

    def test_close_removes_only_handles_installed_by_this_owner(self):
        session = self.session(install=False)
        foreign = [session.model.register_forward_pre_hook(lambda *args: None),
                   session.model.register_forward_hook(lambda *args: None),
                   session.processor.register_forward_pre_hook(lambda *args: None),
                   session.processor.register_forward_hook(lambda *args: None)]
        session.install()
        owned_ids = {handle.id for handle in session.handles}
        session.close()
        remaining = set(session.model._forward_pre_hooks) | set(session.model._forward_hooks)
        remaining |= set(session.processor._forward_pre_hooks) | set(session.processor._forward_hooks)
        self.assertEqual(remaining, {handle.id for handle in foreign})
        self.assertTrue(owned_ids.isdisjoint(remaining))
        for handle in foreign:
            handle.remove()

    def test_partial_hook_install_rolls_back_only_owned_handles(self):
        session = self.session(install=False)
        foreign = session.model.register_forward_hook(lambda *args: None)
        with patch.object(session.processor, 'register_forward_pre_hook', side_effect=RuntimeError('MOCK_INSTALL_FAILED')):
            with self.assertRaisesRegex(RuntimeError, 'MOCK_INSTALL_FAILED'):
                session.install()
        self.assertEqual(set(session.model._forward_hooks), {foreign.id})
        self.assertFalse(session.model._forward_pre_hooks)
        self.assertEqual(session.handles, [])
        self.assertFalse(session.installed)
        foreign.remove()

    def test_partial_collector_start_is_owned_until_stopped(self):
        collector = MockCollector()
        collector.partial_start_failure = True
        session = self.session(collector=collector)
        with self.assertRaisesRegex(RuntimeError, 'MOCK_PARTIAL_COLLECTOR_START'):
            session.expect(self.actual_request_id)
        self.assertTrue(session.collector_active)
        self.assertIsNotNone(session.backend.failure)
        self.assert_drained(session)
        session.close()
        self.assertFalse(collector.running)
        self.assertEqual((collector.starts, collector.stops, collector.steps), (1, 1, 0))

    def test_collector_stop_failure_does_not_claim_hook_release(self):
        collector = MockCollector()
        session = self.session(collector=collector)
        session.expect(self.actual_request_id)
        owned = list(session.handles)
        collector.stop_failure = True
        with self.assertRaisesRegex(RuntimeError, 'MOCK_COLLECTOR_STOP_FAILED'):
            session.close()
        self.assertTrue(session.collector_active)
        self.assertEqual(session.handles, owned)
        self.assertTrue(session.installed)
        collector.stop_failure = False
        session.close()
        self.assertFalse(collector.running)
        self.assertEqual(collector.steps, 0)

    def test_unexpected_request_without_advance_fails_closed_before_arm(self):
        session = self.session()
        with self.assertRaisesRegex(ContextRefused, 'NO_EXPLICIT_ADVANCE'):
            self.root(session)
        self.assertIsNotNone(session.backend.failure)
        self.assertEqual(session.backend.begin_calls, 0)
        self.assert_drained(session)

    def test_wrong_actual_scheduler_id_never_opens_native_context(self):
        session = self.session()
        session.expect('other-request')
        with self.assertRaisesRegex(ContextRefused, 'MOCK_ACTUAL_SCHEDULER_ID_MISMATCH'):
            self.root(session)
        self.assertEqual(session.native.context_entries, 0)
        self.assert_drained(session)

    def test_reused_request_id_is_refused_before_collector_start(self):
        collector = MockCollector()
        session = self.session(collector=collector)
        session.backend.requests.add(self.actual_request_id)
        with self.assertRaisesRegex(ContextRefused, 'FRESH_ENQUEUED_ID_REQUIRED'):
            session.expect(self.actual_request_id)
        self.assertEqual(collector.starts, 0)

    def test_root_exception_runs_always_call_cleanup_without_arming(self):
        session = self.session()
        session.model.raise_forward = True
        session.expect(self.actual_request_id)
        with self.assertRaisesRegex(RuntimeError, 'CPU_ROOT_FORWARD_FAILED'):
            self.root(session)
        self.assertIsNotNone(session.backend.failure)
        self.assertEqual(session.backend.begin_calls, 0)
        self.assert_drained(session)

    def test_nonfinite_root_hidden_is_rejected_before_logits(self):
        session = self.session()
        session.model.nonfinite_hidden = True
        session.expect(self.actual_request_id)
        with self.assertRaisesRegex(ContextRefused, 'HEALTHY_HIDDEN_STATES_NOT_FINITE'):
            self.root(session)
        self.assertEqual(session.backend.begin_calls, 0)
        self.assert_drained(session)

    def test_missing_logits_can_be_cancelled_before_fault_but_not_passed(self):
        session = self.session()
        session.expect(self.actual_request_id)
        self.root(session)
        session.close()
        self.assertEqual(session.backend.failure, 'HOST_ABORT_BEFORE_FAULT')
        self.assertFalse(session.backend.recovered)
        self.assert_drained(session)

    def test_changed_head_or_query_row_is_rejected_before_native_context(self):
        for target in ('head', 'query'):
            with self.subTest(target=target):
                session = self.session()
                session.expect(self.actual_request_id)
                hidden = self.root(session)[-1:].contiguous()
                head = session.head
                if target == 'head':
                    head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
                else:
                    hidden = hidden + 1
                with self.assertRaises(ContextRefused):
                    session.processor(head, hidden)
                self.assertEqual(session.native.context_entries, 0)
                self.assert_drained(session)

    def test_bind_or_begin_failure_clears_only_its_native_context(self):
        for target in ('bind', 'begin'):
            with self.subTest(target=target):
                session = self.session()
                session.native.fail_bind = target == 'bind'
                session.backend.fail_begin = target == 'begin'
                session.expect(self.actual_request_id)
                with self.assertRaisesRegex(RuntimeError, 'MOCK_'):
                    self.forward(session)
                self.assertEqual(session.native.context_entries, 1)
                self.assertEqual(session.native.context_exits, 1)
                self.assert_drained(session)

    def test_logits_forward_or_end_failure_drains_and_requires_off_after_fault(self):
        for target in ('forward', 'end'):
            with self.subTest(target=target):
                session = self.session()
                session.backend.phase = 'fault'
                session.processor.raise_forward = target == 'forward'
                session.backend.fail_end = target == 'end'
                session.expect(self.actual_request_id)
                with self.assertRaisesRegex(RuntimeError, 'CPU_PROCESSOR|MOCK_END'):
                    self.forward(session)
                self.assert_drained(session)
                self.assertTrue(session.backend.fault_started)
                with self.assertRaisesRegex(ContextRefused, 'OFF_RECOVERY_REQUIRED'):
                    session.close()
                self.assertTrue(session.installed)

    def test_successful_cpu_forward_closes_context_and_preserves_complete_hash(self):
        collector = MockCollector()
        session = self.session(collector=collector)
        session.expect(self.actual_request_id)
        self.forward(session)
        self.assert_drained(session)
        self.assertEqual(session.backend.end_calls, 1)
        self.assertEqual(len(session.rows), 1)
        self.assertEqual(len(session.rows[0]['model_output_sha256']), 64)
        self.assertEqual(session.rows[0]['output_shape'], [1, 3])
        self.assertEqual(len(session.native.bindings[0][1]), 13)
        session.report()
        self.assertEqual((collector.stops, collector.steps), (1, 1))
        receipt = json.loads((session.folder / 'serving_receipt.json').read_text())
        self.assertEqual(receipt['qualification']['status'], 'CPU_MOCK_ONLY_NOT_NPU_ACCEPTANCE')

    def test_fault_cannot_detach_without_separate_recovery_proof(self):
        session = self.session()
        session.backend.phase = 'fault'
        session.expect(self.actual_request_id)
        self.forward(session)
        with self.assertRaisesRegex(ContextRefused, 'OFF_RECOVERY_REQUIRED'):
            session.close()
        self.assertEqual(len(session.handles), 4)
        session.backend.recovered = True  # Explicit mock of a separately verified OFF gate.
        session.close()
        self.assertFalse(session.installed)

    def test_foreign_context_or_resource_owner_is_not_cleared_by_failure(self):
        session = self.session()
        session.native.scope_owner = session.native.resource_owner = 'foreign-owner'
        session.native.enabled = True
        session.expect(self.actual_request_id)
        with self.assertRaisesRegex(RuntimeError, 'MOCK_CONTEXT_ALREADY_OWNED'):
            self.forward(session)
        self.assertEqual(session.native.scope_owner, 'foreign-owner')
        self.assertEqual(session.native.resource_owner, 'foreign-owner')
        self.assertTrue(session.native.enabled)
        self.assertEqual(session.native.context_exits, 0)
        self.assertEqual(set(session.native.disable_owners), {'cpu-owner'})
        self.assertEqual(session.backend.drain_status, 'RECOVERY_FAILED')

    def test_model_change_is_rejected_before_any_collection_cleanup(self):
        collector = MockCollector()
        session = self.session(collector=collector)
        session.expect(self.actual_request_id)
        session.runner.model = object()
        with self.assertRaisesRegex(ContextRefused, 'WORKER_MODEL_IDENTITY_CHANGED'):
            session.close()
        self.assertEqual(collector.stops, 0)
        self.assertTrue(session.installed)
        session.runner.model = session.model
        session.close()

    def test_foreign_thread_cannot_install_or_rebind_owner_thread(self):
        session = self.session(install=False)
        owner_thread = session.owner_thread
        errors = self.thread_error(session.install)
        self.assertEqual(len(errors), 1)
        self.assertIn('FOREIGN_MODEL_THREAD', str(errors[0]))
        self.assertEqual(session.owner_thread, owner_thread)
        self.assertEqual(session.handles, [])

    def test_foreign_thread_cannot_stop_step_reset_or_close_collection(self):
        for action in ('collect', 'close', 'reset_for_recovery', 'report'):
            with self.subTest(action=action):
                collector = MockCollector()
                session = self.session(collector=collector)
                session.expect(self.actual_request_id)
                session.needs_collect = session.capture_step_executed = True
                errors = self.thread_error(getattr(session, action))
                self.assertEqual(len(errors), 1)
                self.assertIn('FOREIGN_MODEL_THREAD', str(errors[0]))
                self.assertEqual((collector.stops, collector.steps), (0, 0))
                self.assertTrue(session.installed)
                self.assertEqual(session.expected, self.actual_request_id)
                session.close()

    def test_collector_variadic_wrapper_keeps_real_root_and_logits_bindings(self):
        collector = MockCollector()
        collector.wrap_forwards = True
        session = self.session(collector=collector)
        session.expect(self.actual_request_id)
        for module in (session.model, session.processor):
            self.assertEqual(str(inspect.signature(module.forward)), '(*args, **kwargs)')
        self.forward(session)
        self.assert_drained(session)
        self.assertEqual(session.rows[0]['output_shape'], [1, 3])
        self.assertEqual(session.native.context_entries, 1)
        self.assertEqual(session.model._cpu_wrapper_calls, 1)
        self.assertEqual(session.processor._cpu_wrapper_calls, 1)
        session.close()
        self.assertEqual((collector.stops, collector.steps), (1, 1))
        self.assertFalse(collector.wrapped)
        self.assertEqual(inspect.signature(session.model.forward), session.root_binding.signature)
        self.assertEqual(inspect.signature(session.processor.forward), session.logits_binding.signature)

    def test_unreviewed_forward_wrapper_fails_before_native_arm_at_either_boundary(self):
        for boundary in ('root', 'logits'):
            with self.subTest(boundary=boundary):
                session = self.session()
                session.expect(self.actual_request_id)
                module = session.model if boundary == 'root' else session.processor
                original = module.forward
                @functools.wraps(original)
                def arbitrary_wrapper(*args, **kwargs):
                    return original(*args, **kwargs)
                module.forward = arbitrary_wrapper
                with self.assertRaisesRegex(ContextRefused, 'UNREVIEWED_FORWARD_WRAPPER_CODE'):
                    self.forward(session)
                self.assertEqual(session.native.context_entries, 0)
                self.assertEqual(session.backend.begin_calls, 0)
                self.assert_drained(session)
                module.forward = original
                session.close()

    def test_partial_collector_start_with_wrappers_still_restores_owned_callables(self):
        collector = MockCollector()
        collector.wrap_forwards = collector.partial_start_failure = True
        session = self.session(collector=collector)
        with self.assertRaisesRegex(RuntimeError, 'MOCK_PARTIAL_COLLECTOR_START'):
            session.expect(self.actual_request_id)
        self.assertEqual(len(collector.wrapped), 2)
        session.close()
        self.assertFalse(collector.wrapped)
        self.assertEqual((collector.stops, collector.steps), (1, 0))
        self.assertEqual(session.native.context_entries, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
