"""Real CPU module hooks with mocked scheduler/native receipts, not NPU evidence."""
import copy
import hashlib
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

try:
    import torch
except (ImportError, OSError):
    torch = None

from fault_injection import serving_qualification as sq
from fault_injection.deployments.matmul import _REQUIRED_BUNDLE
from fault_injection.native import _LEASE, _STACK
from demoncat.runtime.vllm_ascend.worker_extension import serving20
from demoncat.runtime.vllm_ascend.worker_extension.providers import fault20
if __package__:
    from .test_serving_session_lifecycle import CpuModel, CpuProcessor, MockCollector
    from .test_serving20_qualification import CpuRuntime
    from . import test_serving_session_lifecycle as lifecycle
    from . import test_serving20_qualification as qualification
else:
    from test_serving_session_lifecycle import CpuModel, CpuProcessor, MockCollector
    from test_serving20_qualification import CpuRuntime
    import test_serving_session_lifecycle as lifecycle
    import test_serving20_qualification as qualification


class CpuServingProcessor(CpuProcessor):
    def forward(self, lm_head, hidden_states, embedding_bias=None):
        return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)


class CpuServingModel(CpuModel):
    def __init__(self):
        super().__init__()
        self.processor = CpuServingProcessor()
        self.in_forward = False
        self.head.forward = self.forbidden_head_forward
        with torch.no_grad():
            self.head.weight.fill_(0.25)

    def forbidden_head_forward(self, *args, **kwargs):
        raise AssertionError('The real output head is not called as a module')

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.in_forward = True
        try:
            return super().forward(input_ids, positions, inputs_embeds, intermediate_tensors)
        finally:
            self.in_forward = False


@unittest.skipIf(torch is None, 'CPU torch fixture unavailable')
class Serving20HookTests(unittest.TestCase):
    snapshot = lifecycle.ServingSessionLifecycleTests.snapshot
    root = lifecycle.ServingSessionLifecycleTests.root
    forward = lifecycle.ServingSessionLifecycleTests.forward

    def setUp(self):
        lifecycle.ServingSessionLifecycleTests.setUp(self)
        self.backends = []
        self.enterContext(patch.dict(os.environ, {'DEMONCAT_F20_REQUIRE_TYPED_GWS': '0'}))
        environment = dict(pid=os.getpid(), maps_status='OBSERVED', machine='CPU-fixture', python='fixture',
            packages={}, torch_cxx11_abi=False, driver={'running_module_versions': {}, 'installation': None},
            loaded_libraries=[dict(name=name, sha256=hashlib.sha256(name.encode()).hexdigest(),
                elf_build_id='CPU', status='MAPPED_FILE_VERIFIED') for name in _REQUIRED_BUNDLE])
        self.enterContext(patch.object(sq, 'inspect_environment', side_effect=lambda **kwargs: copy.deepcopy(environment)))

    def tearDown(self):
        lifecycle.ServingSessionLifecycleTests.tearDown(self)
        for backend in self.backends:
            if backend.locked:
                _LEASE.release()  # CPU fixture disposal, never an OFF assertion.
                backend.locked = False

    def make_session(self, *, start=True, collector=None, baseline_control=False):
        model = CpuServingModel().eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0)
        folder = self.evidence_root / ('serving20-' + str(len(self.sessions)))
        folder.mkdir(parents=True)
        runtime = CpuRuntime(folder)
        boundary = dict(processor_path='processor', lm_head_path='head', vocab_size=3, output_contract=[1, 3])
        backend = sq.ServingQualifiedNativeSession(model, model.processor, model.head, boundary, runtime,
            owner='CPU-owner', generation=1,
            reviewed_manifest_path=qualification.Serving20QualificationTests.manifest(self, folder),
            first_live_validation=True, evidence_dir=folder / 'qualification', baseline_control=baseline_control)
        self.backends.append(backend)
        test = self
        class CpuQuantMethod:
            def apply(self, head, hidden, bias=None):
                test.assertFalse(model.in_forward)
                test.assertIs(head, model.head)
                test.assertIs(_STACK.get()[-1][1], model.processor)
                before = len(runtime.modes)
                if not runtime.paused:
                    runtime.emit(backend.current['transaction'])
                # This transformation is an explicit CPU native fixture. It is
                # not implementation or evidence of the real device injector.
                mode = runtime.modes[-1] if len(runtime.modes) != before else 0
                return torch.nn.functional.linear(hidden / 2 if mode == 3 else hidden, head.weight, bias)
        model.head.quant_method = CpuQuantMethod()
        session = serving20.Serving20Session(worker, model, backend, runtime, 'CPU-owner',
            prompt_token_ids=[1, 2, 3, 4], evidence_dir=folder / 'runtime', collector=collector)
        session.boundary_report = boundary
        self.sessions.append(session)
        self.enterContext(patch.object(serving20, 'live_tensor', side_effect=lambda value: dict(
            serving20.tensor_metadata(value), storage_addr=value.untyped_storage().data_ptr(),
            storage_bytes=value.untyped_storage().nbytes(), element_size=value.element_size(),
            actual_format=29 if value is model.head.weight else 2)))
        if start:
            backend.start()
            session.install()
        return session

    def actual_phase(self, session, index):
        self.actual_request_id = 'cpu-real-hook-' + str(index)
        session.expect(self.actual_request_id)
        with torch.inference_mode():
            result = self.forward(session)
        session.collect()
        return result

    def test_full_cpu_model_then_actual_processor_owner_with_collector_wrapper(self):
        collector = MockCollector()
        collector.wrap_forwards = True
        session = self.make_session(collector=collector)
        outputs = [self.actual_phase(session, i) for i in range(8)]
        self.assertEqual(session.backend.query()['qualification']['status'], 'SCOPED_ATTEMPT_COMPLETED')
        self.assertTrue(all(row['transaction']['root_forward_returned'] for row in session.backend.rows))
        self.assertTrue(all(row['transaction']['processor_object_id'] == id(session.processor)
                            for row in session.backend.rows))
        self.assertTrue(torch.equal(outputs[0], outputs[-1]))
        self.assertFalse(torch.equal(outputs[0], outputs[5]))
        self.assertEqual((collector.starts, collector.stops, collector.steps), (8, 8, 8))
        session.close()
        session.backend.close()

    def test_head_weight_replacement_refuses_before_native_scope(self):
        session = self.make_session()
        session.head.weight = torch.nn.Parameter(session.head.weight.detach().clone())
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_HEAD_OR_QUANT_METHOD_CHANGED'):
            self.actual_phase(session, 0)
        self.assertEqual(session.native.original_calls, 0)
        self.assertTrue(session.native.paused)
        session.close()
        session.backend.close()

    def test_processor_without_real_root_is_not_a_qualified_transaction(self):
        session = self.make_session()
        with self.assertRaisesRegex(RuntimeError, 'LOGITS_NOT_BOUND_TO_ONE_COMPLETE_FORWARD'):
            session.processor(session.head, torch.ones(1, 4, dtype=torch.bfloat16))
        self.assertEqual(session.native.original_calls, 0)
        session.close()
        session.backend.close()

    def test_actual_provider_rpc_branch_preserves_pending_pause_and_finishes_only_after_off(self):
        session = self.make_session(start=False, baseline_control=True)
        identity = dict(rank=0, logical_device=0, pid=os.getpid(), worker_start_identity='CPU-worker',
                        model_instance_id='CPU-model')
        provider = fault20.Fault20Provider(session.worker, session.model, identity)
        ticket = dict(owner='CPU-owner', generation=1)
        provider.bind(**ticket)
        provider.runtime = session.native
        binding = fault20.Fault20ServingReplay('a' * 64, lambda *args: session)
        with (patch.object(fault20, 'qualified_binding', return_value=None),
              patch.object(fault20, 'resolve_runtime', return_value=binding),
              patch.object(provider, '_worker_facts', return_value=session.native.worker)):
            provider.dispatch('register', {}, **ticket)
        provider.dispatch('check', {}, **ticket)
        state = provider.dispatch('execute', {}, **ticket)
        self.assertTrue(state['replay_pending'])
        self.assertTrue(state['native_registered'])
        self.assertFalse(state['enabled'])
        with self.assertRaisesRegex(RuntimeError, 'SERVING_QUALIFICATION_NOT_FINISHED'):
            provider.dispatch('finish', {}, **ticket)
        for index in range(8):
            self.actual_request_id = 'cpu-rpc-request-' + str(index)
            state = provider.dispatch('advance', {'request_id': self.actual_request_id}, **ticket)
            self.assertFalse(state['enabled'])
            with torch.inference_mode():
                self.forward(session)
            state = provider.query(**ticket)
            self.assertFalse(state['enabled'])
        self.assertEqual(state['qualification']['status'], 'BASELINE_CONTROL_COMPLETED')
        state = provider.dispatch('finish', {}, **ticket)
        self.assertFalse(state['replay_pending'])
        self.assertFalse(state['native_registered'])
        provider.dispatch('unregister', {}, **ticket)


if __name__ == '__main__':
    unittest.main(verbosity=2)
