"""CPU tensor execution with explicit native mocks; never device qualification."""
import copy
import ctypes as C
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.residual26 import FiaContext, QmmContext
from fault_injection.deployments.writer26_inventory import (InventoryEvent, InventoryState, KINDS,
    NativeWriterInventory, WriterLifetimeInventory, correlate_lifetimes, tensor_matches_native)


class CpuReader(NativeWriterInventory):
    def __init__(self):
        self.library_sha256 = 'a' * 64
        self.raw = InventoryState(version=1, size=C.sizeof(InventoryState), flags=7, capacity=32768)
        self.events = []
        self.state_fn = self._state
        self.read_fn = self._read

    def _state(self, output):
        C.memmove(output, C.byref(self.raw), C.sizeof(self.raw))
        return 0

    def _read(self, index, output):
        C.memmove(output, C.byref(self.events[index]), C.sizeof(InventoryEvent))
        return 0

    def append(self, kind, *, output=None, residual=None, **kwargs):
        value = InventoryEvent(version=1, size=C.sizeof(InventoryEvent), kind=kind,
            sequence=len(self.events) + 1, thread=threading.get_native_id(), device=0, **kwargs)
        if kind in (1, 3) and not value.call:
            value.call = value.sequence
        for tensor, prefix in ((output, ''), (residual, 'residual_')):
            if tensor is None:
                continue
            setattr(value, 'residual' if prefix else 'output', tensor.untyped_storage().data_ptr())
            setattr(value, 'residual_bytes' if prefix else 'output_bytes', tensor.untyped_storage().nbytes())
            setattr(value, prefix + 'dimensions', tensor.ndim)
            getattr(value, prefix + 'shape')[:tensor.ndim] = list(tensor.shape)
            getattr(value, 'residual_strides' if prefix else 'strides')[:tensor.ndim] = list(tensor.stride())
            value.flags |= 2 if prefix else 1
        self.events.append(value)
        self.raw.total_events += 1
        self.raw.stored_events += 1
        name = KINDS[kind]
        setattr(self.raw, name, getattr(self.raw, name) + 1)
        return value


class CpuRuntime:
    fia = False
    mechanism = '26.residual_stomp'
    sha256 = 'b' * 64
    session = 1

    def __init__(self, reader):
        self.reader, self.callback = reader, None
        self.continued = []
        self.producer_sequence = self.writer_calls = self.producer_calls = 0
        self.skip_callback = self.stale_producer = False
        self.last = None

    def native_state(self):
        return dict(registered=int(self.callback is not None), prepared=0, queued_executors=0,
            pending_consumers=0, active_consumer=0, callback_active=0, environment_enabled=0)

    def install_fn(self, callback):
        assert self.callback is None
        self.callback = callback
        return 0

    def remove_fn(self, callback):
        assert self.callback is callback
        self.callback = None
        return 0

    def cont(self, token):
        assert token not in self.continued
        self.continued.append(token)
        return 0

    def producer(self, x, residual):
        self.producer_calls += 1
        self.producer_sequence += 1
        result = x + residual
        for kind in (7, 8, 9):
            self.reader.append(kind, output=result, residual=residual, producer=self.producer_sequence,
                call=self.producer_sequence)
        self.last = result
        return result * .5, torch.ones(1), result

    def writer(self, x):
        self.writer_calls += 1
        result = x.clone()  # Actual CPU output, before a mock native callback.
        self.reader.raw.all_runs += 1
        self.reader.raw.qmm_runs += 1
        independent = self.reader.append(1, output=result, residual=self.last,
            producer=self.producer_sequence + int(self.stale_producer), cache=123, registration=1)
        packet = QmmContext(abi_version=1, device=0, run_index=self.writer_calls,
            producer_sequence=self.producer_sequence, cache=123, original_output=result.data_ptr(),
            output_bytes=result.untyped_storage().nbytes(), output_m=4, output_n=4,
            residual=self.last.data_ptr(), residual_bytes=self.last.untyped_storage().nbytes(), residual_m=4, residual_k=4)
        if not self.skip_callback:
            self.callback(C.pointer(packet), independent.call)
        self.reader.append(2, output=result, residual=self.last, producer=self.producer_sequence,
            call=independent.call, cache=123, registration=1)
        return result

    def writer_fia(self, x):
        self.writer_calls += 1
        result = x.clone()
        independent = self.reader.append(3, output=result, residual=self.last, producer=self.producer_sequence)
        packet = FiaContext(version=1, size=C.sizeof(FiaContext), token=independent.call, device=0,
            output=result.data_ptr(), output_bytes=result.untyped_storage().nbytes(),
            residual=self.last.data_ptr(), residual_bytes=self.last.untyped_storage().nbytes(),
            producer_sequence=self.producer_sequence, metadata_ok=1, dimensions=2)
        self.callback(C.pointer(packet))
        for kind in (4, 5, 6):
            self.reader.append(kind, output=result, residual=self.last, producer=self.producer_sequence, call=independent.call)
        return result


class Producer(torch.nn.Module):
    def forward(self, x, residual):
        return torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual)[2]


class Writer(torch.nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def forward(self, x):
        return self.runtime.writer(x)


class CpuWriterModel(torch.nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.producer = Producer()
        self.first = Writer(runtime)
        self.second = Writer(runtime)

    def forward(self, x):
        residual = self.producer(x, torch.zeros_like(x))
        x = self.first(x)
        residual = self.producer(x, residual)
        x = self.second(x)
        return self.producer(x, residual)


class WriterInventoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reader = CpuReader()
        self.runtime = CpuRuntime(self.reader)
        self.model = CpuWriterModel(self.runtime).eval()
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.runtime.producer, create=True))
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda t: 2,
            npu=NS(memory_snapshot=lambda: []))}))
        # Explicit CPU allocator mock. Real allocator adjacency remains a separate production gate.
        self.enterContext(patch('fault_injection.deployments.writer26_inventory.allocation_pair',
            return_value=dict(evidence_kind='EXPLICIT_CPU_ALLOCATOR_MOCK_NOT_NPU')))
        self.backend = WriterLifetimeInventory(self.model, self.runtime, self.reader,
            worker_identity={'worker_id': 'CPU-only', 'rank': 0}, folder=self.folder / 'inventory')
        self.backend.install()
        self.addCleanup(self.dispose)

    def dispose(self):
        if self.backend.active:
            self.backend.fail('CPU_FIXTURE_DISPOSAL_NOT_RECOVERY')
        self.backend.close()

    def run_request(self, request_id):
        self.backend.begin(request_id, 'c' * 64)
        with torch.inference_mode():
            output = self.model(torch.ones(4, 4, dtype=torch.bfloat16))
        self.backend.end('d' * 64, model_output_finite=bool(torch.isfinite(output).all()))
        return self.backend.rows[-1]

    def test_actual_two_modules_shared_cache_remain_distinct_with_exact_once_originals(self):
        for request in ('first', 'second'):
            row = self.run_request(request)
            self.assertEqual(len(row['candidates']), 2)
            self.assertTrue(all(not c['reasons'] for c in row['candidates']))
            self.assertEqual([c['callback']['module_path'] for c in row['candidates']], ['first', 'second'])
            self.assertTrue(all(c['callback']['original_continuations'] == 1 for c in row['candidates']))
        self.backend.close()
        report = self.backend.report()
        self.assertEqual((self.runtime.writer_calls, self.runtime.producer_calls, len(self.runtime.continued)), (4, 6, 4))
        self.assertEqual((report['native_injections'], report['qualified_native_targets'], report['owned_hooks']), (0, 0, 0))
        self.assertTrue(report['healthy_repeated'])
        self.assertFalse(report['c2'] or report['c3'])

    def test_missing_callback_remains_in_independent_native_denominator(self):
        self.runtime.skip_callback = True
        row = self.run_request('missing')
        self.assertEqual(len(row['candidates']), 2)
        self.assertTrue(all('NATIVE_WRITER_HAS_NO_OWNED_PUBLIC_CALLBACK' in c['reasons'] for c in row['candidates']))
        self.assertEqual(len(self.runtime.continued), 0)

    def test_stale_producer_with_same_address_rejected_and_original_continues_once(self):
        self.runtime.stale_producer = True
        row = self.run_request('stale')
        self.assertTrue(all('CURRENT_PUBLIC_RESIDUAL_LIFETIME_UNPROVEN' in c['reasons'] for c in row['candidates']))
        self.assertEqual(len(self.runtime.continued), 2)

    def test_outside_request_callback_continues_exactly_once(self):
        self.runtime.callback(C.pointer(QmmContext(abi_version=1)), 9001)
        self.assertEqual(self.runtime.continued, [9001])
        self.assertEqual(self.backend.callbacks, [])

    def test_missing_or_failed_consumer_cannot_form_lifetime(self):
        row = self.run_request('consumer')
        native = copy.deepcopy(row['native_events'])
        native[-1]['status'] = 7
        candidates = correlate_lifetimes(native, row['callbacks'], row['producers'], self.runtime.mechanism)
        self.assertEqual(candidates[-1]['eligibility'], 'UNRESOLVED')
        self.assertIn('NEXT_ACTUAL_NATIVE_CONSUMER_UNPROVEN', candidates[-1]['reasons'])
        row['producers'][0]['errors'].append('PUBLIC_FORMAL_CHANGED')
        candidates = correlate_lifetimes(row['native_events'], row['callbacks'], row['producers'], self.runtime.mechanism)
        self.assertIn('PUBLIC_FORMAL_CHANGED', candidates[0]['reasons'])

    def test_shared_storage_wrong_view_is_not_public_native_formal(self):
        tensor = torch.ones(4, 4)
        event = self.reader.append(9, output=tensor, residual=tensor, producer=1)
        row = self.reader.read(event.sequence - 1)
        self.assertTrue(tensor_matches_native(tensor, row))
        self.assertFalse(tensor_matches_native(tensor.reshape(2, 8), row))

    def test_overflow_counter_details_and_epoch_reset_fail_closed(self):
        before = self.reader.state()
        self.reader.append(10, registration=1, cache=123)
        self.reader.raw.qmm_registration += 1
        with self.assertRaisesRegex(RuntimeError, 'DETAILS_DO_NOT_MATCH_COUNTERS'):
            self.reader.delta(before)
        self.reader.raw.qmm_registration -= 1
        self.reader.raw.overflow_events = 1
        with self.assertRaisesRegex(RuntimeError, 'OVERFLOW_OR_ERROR'):
            self.reader.state()
        self.reader.raw.overflow_events = 0
        before = self.reader.state()
        self.reader.raw.qmm_registration = 0
        with self.assertRaisesRegex(RuntimeError, 'COUNTER_RESET'):
            self.reader.delta(before)

    def test_native_abi_sizes(self):
        self.assertEqual(C.sizeof(InventoryEvent), 416)
        self.assertEqual(C.sizeof(InventoryState), 160)

    def test_fia_before_prepare_is_bound_to_actual_execution_and_next_consumer(self):
        self.backend.close()
        self.runtime.fia, self.runtime.mechanism = True, '26.fia_destination'
        self.runtime.writer = self.runtime.writer_fia
        original_install = self.runtime.install_fn
        self.runtime.install_fn = lambda callback, session: original_install(callback)
        self.backend = WriterLifetimeInventory(self.model, self.runtime, self.reader,
            worker_identity={'worker_id': 'CPU-FIA', 'rank': 0}, folder=self.folder / 'fia')
        self.backend.install()
        row = self.run_request('fia')
        self.assertEqual(len(row['candidates']), 2)
        self.assertTrue(all(not c['reasons'] and c['native']['kind_name'] == 'fia_before' for c in row['candidates']))
        self.assertEqual(len(self.runtime.continued), 2)

    def test_actual_root_then_logits_hooks_complete_before_detaching(self):
        import test_serving_session_lifecycle as lifecycle
        from demoncat.runtime.vllm_ascend.worker_extension.serving26_inventory import Serving26InventorySession
        self.backend.close()
        fixture = lifecycle.ServingSessionLifecycleTests()
        fixture.setUp()
        try:
            class ServingModel(CpuWriterModel):
                def __init__(inner, runtime):
                    super().__init__(runtime)
                    inner.processor = lifecycle.CpuProcessor()
                    inner.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
                def forward(inner, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                    return super().forward(torch.ones(4, 4, dtype=torch.bfloat16))
            model = ServingModel(self.runtime).eval()
            # The lifecycle fixture deliberately replaces torch_npu; restore the CPU allocator mock.
            sys.modules['torch_npu'].npu = NS(memory_snapshot=lambda: [])
            runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
            worker = NS(model_runner=runner, rank=0)
            self.backend = WriterLifetimeInventory(model, self.runtime, self.reader,
                worker_identity={'worker_id': 'CPU-serving', 'rank': 0}, folder=self.folder / 'serving-inventory')
            session = Serving26InventorySession(worker, model, self.backend, self.runtime, 'CPU-read-only',
                prompt_token_ids=[1, 2, 3, 4], evidence_dir=fixture.evidence_root / 'actual-hooks')
            fixture.sessions.append(session)
            session.install()
            for request in ('serving1', 'serving2'):
                fixture.actual_request_id = request
                session.expect(request)
                with torch.inference_mode():
                    fixture.forward(session)
            session.close()
            report = session.report()
            self.assertEqual(report['actual_native_writers'], [2, 2])
            self.assertEqual(report['complete_writer_lifetimes'], [2, 2])
            self.assertEqual(report['owned_root_logits_hooks'], 0)
            self.assertEqual(report['owned_module_hooks'], 0)
            self.assertTrue(report['healthy_repeated'])
        finally:
            fixture.tearDown()
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
