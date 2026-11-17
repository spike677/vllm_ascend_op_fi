import ctypes
import os
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engineering_closure.native_observer import NativeObserver, NativeScope
from engineering_closure.path_discovery import normalize_execution_path
from fault_injection.coverage import KernelLaunchBoundary
from fault_injection.deployments.dynamic_quant_inventory import (KINDS as QUANT_KINDS,
    NativeDynamicQuantInventory, QuantEvent, QuantState)
from fault_injection.runtime_discovery import _process_start_identity


def _scope(**updates):
    values = dict(process_id=os.getpid(), process_start_identity=_process_start_identity(),
                  worker_id='worker-local', rank=0, logical_device=3, request_ids=('req-1',),
                  phase='decode', dispatcher_token='dispatcher:1', request_observed=True)
    values.update(updates)
    return NativeScope(**values)


class QuantWire:
    """Native ABI fixture. Existing state/delta readers validate counters and rows."""
    def __init__(self):
        self.rows = []
        self.overflow = 0
        self.dcat_dq_inventory_state_v1 = lambda pointer: self.state(pointer)
        self.dcat_dq_inventory_read_v1 = lambda index, pointer: self.read(index, pointer)

    def state(self, pointer):
        row = ctypes.cast(pointer, ctypes.POINTER(QuantState)).contents
        row.capacity = 32768
        row.total_events = row.stored_events = len(self.rows)
        row.overflow_events = self.overflow
        for number, name in QUANT_KINDS.items():
            setattr(row, name, sum(event['kind'] == number for event in self.rows))
        return 0

    def read(self, index, pointer):
        row = ctypes.cast(pointer, ctypes.POINTER(QuantEvent)).contents
        for key, value in self.rows[index].items():
            setattr(row, key, value)
        return 0

    def call(self, thread, *, kinds=(1, 2, 3, 4), prepare_status=0):
        call = len(self.rows) + 1
        for kind in kinds:
            self.rows.append(dict(sequence=len(self.rows) + 1, call=call, kind=kind,
                                  executor=0 if kind == 1 else 420, thread=thread, device=3,
                                  status=prepare_status if kind == 2 else 0, input=11, quant=12, scale=13))


class NativeWindowTests(unittest.TestCase):
    def setUp(self):
        self.wire = QuantWire()
        device = SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 3))
        with patch.dict(sys.modules, {'torch_npu': device}):
            reader = NativeDynamicQuantInventory(SimpleNamespace(build_identity={'build_sha256': 'a' * 64},
                backend=SimpleNamespace(lib=self.wire)))
        self.observer = NativeObserver(reader, source='dynamic_quant_v2', build_sha256='a' * 64)

    def test_all_calls_and_executors_in_window_with_foreign_thread(self):
        thread = threading.get_native_id()
        with patch.dict(os.environ, {'TASK_QUEUE_ENABLE': '0'}), self.observer.capture(_scope()) as window:
            self.wire.call(thread)
            self.wire.call(thread + 1)
            self.wire.call(thread, kinds=(1, 2), prepare_status=14)
        self.assertTrue(window['complete'])
        self.assertEqual(len(window['events']), 10)
        self.assertEqual([call['native_call'] for call in window['calls']], [1, 5, 9])
        self.assertEqual([call['executor'] for call in window['calls']], [420, 420, 420])
        self.assertEqual([call['binding_status'] for call in window['calls']], ['OBSERVED', 'NOT_VISIBLE', 'OBSERVED'])
        self.assertEqual(window['calls'][1]['request_ids'], [])
        self.assertFalse(window['calls'][2]['execution_returned'])
        self.assertEqual(window['calls'][2]['native_status'], 14)

    def test_unfinished_and_overflow_refuse_complete_evidence(self):
        with self.observer.capture(_scope()) as partial:
            self.wire.call(threading.get_native_id(), kinds=(1, 2))
        self.assertFalse(partial['complete'])
        self.assertEqual(len(partial['events']), 2)
        with self.observer.capture(_scope()) as overflow:
            self.wire.overflow = 1
        self.assertFalse(overflow['complete'])
        self.assertIn('INDEPENDENT_QUANT_OVERFLOW_OR_ERROR', overflow['errors'][0])

    def test_no_native_join_from_foreign_process_or_static_source(self):
        with self.assertRaisesRegex(ValueError, 'CURRENT_PROCESS'):
            with self.observer.capture(_scope(process_id=os.getpid() + 1)):
                self.fail('foreign capture opened')
        with patch.dict(os.environ, {'TASK_QUEUE_ENABLE': '0'}), self.observer.capture(_scope()) as window:
            self.wire.call(threading.get_native_id())
        census = {'schema': 'demoncat.natural_tensor_census.v1', 'complete': True, 'events': [
            {'sequence': 1, 'path': 'attention', 'kind': 'operator', 'operator': 'npu.example.default',
             'schema': 'npu::example(Tensor a) -> Tensor', 'forward': 0, 'dispatcher_token': 'dispatcher:1'}]}
        launch = KernelLaunchBoundary('separate-observed-kernel', 0, 'd' * 64, 'e' * 64, 'launch-token', 0)
        report = normalize_execution_path(census, context=window['scope'], native_windows=[window],
                                          kernel_launches={('dispatcher:1', 'dynamic_quant_v2', 1): launch})
        observed = lambda graph: next(edge for edge in graph['edges'] if edge['target'].startswith('native:'))
        self.assertEqual(observed(report)['status'], 'OBSERVED')
        self.assertTrue(all(edge['status'] != 'OBSERVED' for edge in report['edges']
                            if edge['target'].startswith('device_kernel:')))
        foreign = dict(window['scope'], worker_id='other-worker')
        report = normalize_execution_path(census, context=foreign, native_windows=[window], source_routes={
            'npu.example.default': {'native_api': 'aclnnDynamicQuantV2', 'file': 'pinned.cpp', 'source_sha256': 'c' * 64}})
        self.assertEqual(observed(report)['status'], 'INFERRED')
