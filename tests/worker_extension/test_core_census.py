"""CPU-only contracts: real CPU dispatch, synthetic scheduler/device metadata.

These tests do not load vLLM, torch_npu, a Worker, a Native library, or a device.
They cannot qualify a model, Native target, C1/C2/C3, or NPU recovery.
"""

from copy import deepcopy
import json
import os
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.hooks import RemovableHandle

from demoncat.runtime.vllm_ascend.worker_extension.call_census import WorkerCallCensus
from engineering_closure import path_discovery
from engineering_closure.native_observer import NativeObserver
from fault_injection.runtime_discovery import _process_start_identity
from worker_extension.test_serving_context import fixture


class NaturalCPUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.unused = torch.nn.ReLU()
        self.register_buffer('value', torch.tensor([-3.0, 2.0]))
        self.calls = 0
        self.during_forward = None

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.calls += 1
        if self.during_forward is not None:
            self.during_forward()
        return self.activation(self.value + 2)


class WorkerCensusTests(unittest.TestCase):
    def setUp(self):
        runner, _, actual, context, prompt, request_id = fixture()
        self.model = NaturalCPUModel()
        runner.model = context.model_instance = self.model
        self.runner, self.actual, self.context = runner, actual, context
        self.worker = NS(model_runner=runner, rank=0, device=NS(type='npu', index=0),
                         load_model=Mock(side_effect=AssertionError('must not load a model')),
                         execute_model=Mock(side_effect=AssertionError('host owns execution')))
        self.identity = dict(pid=os.getpid(), rank=0, logical_device=0,
            process_start_identity=_process_start_identity(), worker_start_identity='CPU_ONLY_WORKER',
            model_instance_id='CPU_ONLY_MODEL')
        self.expected = dict(internal_request_id=request_id, prompt_token_ids=prompt)
        forward_context = ModuleType('vllm.forward_context')
        forward_context.get_forward_context = lambda: self.context
        modules = patch.dict(sys.modules, {'vllm.forward_context': forward_context, 'torch_npu': None})
        modules.start()
        self.addCleanup(modules.stop)
        self.sessions = []
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for census in self.sessions:
            census.cleanup(self.worker, self.identity)

    def census(self, **kwargs):
        options = dict(phase='prefill', expected=self.expected)
        options.update(kwargs)
        census = WorkerCallCensus(self.worker, self.identity, **options)
        self.sessions.append(census)
        return census

    def normal_request(self):
        return self.model(**self.actual)

    def assert_clean(self, row):
        self.assertTrue(row['resource_cleanup_confirmed'], row)
        self.assertTrue(row['releasable'], row)
        self.assertEqual(row['observer_hooks'], 0)
        self.assertFalse(row['inflight'])
        self.assertFalse(row['qualification_granted'])

    def test_attaches_only_and_captures_natural_dispatch_with_unobserved_native_edges(self):
        forbidden = Mock(side_effect=AssertionError('device API must not be queried'))
        with patch.object(torch, 'npu', NS(is_initialized=forbidden, current_device=forbidden,
                                         init=forbidden, set_device=forbidden), create=True):
            census = self.census()
            started = census.start()
            self.assertEqual(self.model.calls, 0)
            self.assertEqual(started['state'], 'OBSERVING')
            self.assertFalse(started['releasable'])
            observed = self.normal_request()
            stopped = census.stop(self.worker, self.identity)
            self.assertFalse(stopped['releasable'])
            row = census.cleanup(self.worker, self.identity)
            self.assert_clean(row)
            self.assertTrue(row['request_binding_valid'])
            self.assertEqual(row['observation_status'], 'OBSERVED')
            report = row['report']
            self.assertEqual(report['context']['model_object_identity'], id(self.model))
            self.assertEqual(report['context']['worker_id'], self.identity['worker_start_identity'])
            self.assertEqual(row['runner_object_id'], id(self.runner))
            self.assertEqual(report['forwards'][0]['request']['evidence']['request_object_id'],
                             id(self.runner.requests[self.expected['internal_request_id']]))
            self.assertTrue(any(node.get('name') == 'aten.add.Tensor' for node in report['nodes']))
            native_edges = [edge for edge in report['edges'] if edge['target'].startswith('native:')]
            self.assertTrue(native_edges)
            self.assertTrue(all(edge['status'] == 'NOT_VISIBLE' for edge in native_edges))
            self.assertTrue(any(edge['target'].startswith('device_kernel:') and
                                edge['status'] == 'NOT_VISIBLE' for edge in report['edges']))
            self.assertEqual(self.normal_request().tolist(), observed.tolist())
            self.assertEqual(len(census.query(self.worker, self.identity)['report']['forwards']), 1)
            forbidden.assert_not_called()
            self.worker.load_model.assert_not_called()
            self.worker.execute_model.assert_not_called()
            json.dumps(row, allow_nan=False)

    def test_wrong_identity_or_model_cannot_cleanup_or_release_owner(self):
        census = self.census()
        census.start()
        for key, value in [('worker_start_identity', 'foreign'), ('logical_device', 1), ('rank', 1)]:
            other = dict(self.identity, **{key: value})
            with self.subTest(key=key):
                self.assertFalse(census.query(self.worker, other)['releasable'])
                with self.assertRaisesRegex(RuntimeError, 'GENERATION_MISMATCH'):
                    census.cleanup(self.worker, other)
        self.runner.model = NaturalCPUModel()
        try:
            self.assertFalse(census.query(self.worker, self.identity)['identity_valid'])
            with self.assertRaisesRegex(RuntimeError, 'MODEL_OBJECT_MISMATCH'):
                census.cleanup(self.worker, self.identity)
        finally:
            self.runner.model = self.model
        foreign = NS(**vars(self.worker))
        with self.assertRaisesRegex(RuntimeError, 'WORKER_OBJECT_MISMATCH'):
            census.cleanup(foreign, self.identity)
        self.assertGreater(census.query(self.worker, self.identity)['observer_hooks'], 0)
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_wrong_owner_thread_cannot_stop_or_release(self):
        census = self.census()
        census.start()
        rows = []
        def other_thread():
            rows.append(census.query(self.worker, self.identity))
            try:
                census.cleanup(self.worker, self.identity)
            except RuntimeError as error:
                rows.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertFalse(rows[0]['releasable'])
        self.assertEqual(rows[1], 'CENSUS_OWNER_THREAD_MISMATCH')
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_actual_multi_request_is_failed_observation_but_proven_cleanup_can_release(self):
        census = self.census()
        census.start()
        self.runner.input_batch.req_ids.append('CPU_ONLY_FOREIGN_REQUEST')
        self.runner.input_batch.num_reqs = 2
        self.normal_request()
        pending = census.stop(self.worker, self.identity)
        self.assertFalse(pending['releasable'])
        self.assertFalse(pending['request_binding_valid'])
        self.assertEqual(pending['observation_status'], 'FAILED')
        self.assertTrue(pending['request_failures'])
        self.assertEqual(pending['report']['forwards'][0]['request_status'], 'NOT_VISIBLE')
        self.assert_clean(census.cleanup(self.worker, self.identity))
        self.assertFalse(census.query(self.worker, self.identity)['qualification_granted'])

    def test_multiple_requested_ids_refuse_before_installation(self):
        with self.assertRaisesRegex(ValueError, 'SINGLE_REQUEST_REQUIRED'):
            self.census(phase='decode', expected={'a': {'prompt': [1], 'logical_id': 1},
                                                'b': {'prompt': [2], 'logical_id': 2}})
        self.assertEqual(len(self.model._forward_pre_hooks), 0)

    def test_inflight_stop_drains_naturally_and_cleanup_cannot_release_early(self):
        census = self.census()
        census.start()
        during = []
        def stop_during_forward():
            during.append(census.stop(self.worker, self.identity))
            with self.assertRaisesRegex(RuntimeError, 'FORWARD_STILL_INFLIGHT'):
                census.cleanup(self.worker, self.identity)
        self.model.during_forward = stop_during_forward
        result = self.normal_request()
        self.model.during_forward = None
        self.assertTrue(during[0]['inflight'])
        self.assertFalse(during[0]['releasable'])
        self.assertEqual(during[0]['state'], 'DRAINING')
        # After stop a normal request is unchanged and is outside the census.
        self.assertEqual(self.normal_request().tolist(), result.tolist())
        row = census.cleanup(self.worker, self.identity)
        self.assert_clean(row)
        self.assertEqual(len(row['report']['forwards']), 1)

    def test_cleanup_failure_keeps_handles_and_error_history_until_retry(self):
        census = self.census()
        census.start()
        self.normal_request()
        with patch.object(RemovableHandle, 'remove', side_effect=RuntimeError('CPU_REMOVE_FAILED')):
            with self.assertRaisesRegex(RuntimeError, 'CPU_REMOVE_FAILED'):
                census.cleanup(self.worker, self.identity)
        failed = census.query(self.worker, self.identity)
        self.assertEqual(failed['state'], 'CLEANUP_FAILED')
        self.assertFalse(failed['releasable'])
        self.assertGreater(failed['observer_hooks'], 0)
        row = census.cleanup(self.worker, self.identity)
        self.assert_clean(row)
        self.assertEqual(row['errors'][0]['reason'], 'CPU_REMOVE_FAILED')
        self.assertEqual(row['observation_status'], 'FAILED')

    def test_noop_remove_is_not_cleanup_success(self):
        census = self.census()
        census.start()
        with patch.object(RemovableHandle, 'remove', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'HOOK_REMOVAL_UNCONFIRMED'):
                census.cleanup(self.worker, self.identity)
        row = census.query(self.worker, self.identity)
        self.assertFalse(row['releasable'])
        self.assertGreater(row['observer_hooks'], 0)
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_foreign_hook_replacement_is_retained_and_never_removed(self):
        census = self.census()
        census.start()
        key = next(iter(self.model._forward_hooks))
        original = self.model._forward_hooks[key]
        foreign = lambda *args: None
        self.model._forward_hooks[key] = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, 'OWNERSHIP_CHANGED'):
                census.cleanup(self.worker, self.identity)
            row = census.query(self.worker, self.identity)
            self.assertFalse(row['releasable'])
            self.assertTrue(row['foreign_hooks'])
            self.assertIs(self.model._forward_hooks[key], foreign)
        finally:
            self.model._forward_hooks[key] = original
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_partial_start_and_failed_cleanup_retain_original_error_and_retry_handle(self):
        census = self.census()
        owner_table = {'owner': census}
        with patch.object(self.model.activation, 'register_forward_hook',
                          side_effect=RuntimeError('CPU_INSTALL_FAILED')):
            with patch.object(RemovableHandle, 'remove', side_effect=RuntimeError('CPU_REMOVE_FAILED')):
                with self.assertRaisesRegex(RuntimeError, 'CPU_REMOVE_FAILED'):
                    census.start()
        row = census.query(self.worker, self.identity)
        self.assertIs(owner_table['owner'], census)
        self.assertFalse(row['releasable'])
        self.assertGreater(row['observer_hooks'], 0)
        self.assertEqual(row['capture_errors']['workload_error_reason'], 'CPU_INSTALL_FAILED')
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_untrusted_native_shape_refuses_without_native_loading(self):
        factory = Mock(return_value={'aten.add.Tensor': object()})
        census = self.census(native_observer_factory=factory)
        with self.assertRaisesRegex(ValueError, 'TYPED_READ_ONLY_NATIVE_OBSERVERS_REQUIRED'):
            census.start()
        factory.assert_called_once_with(self.worker)
        self.assertEqual(len(self.model._forward_pre_hooks), 0)
        self.assert_clean(census.cleanup(self.worker, self.identity))

    def test_typed_native_window_reuses_actual_dispatch_scope_and_preserves_reader_failure(self):
        reader = NS(state=Mock(return_value={}), delta=Mock(return_value=({}, [])))
        native = NativeObserver(reader, source='dynamic_quant_v2', build_sha256='a' * 64)
        with patch.dict(os.environ, {'TASK_QUEUE_ENABLE': '0'}):
            census = self.census(native_observer_factory=lambda worker: {'aten.add.Tensor': native})
            census.start()
            reader.state.assert_not_called()
            self.normal_request()
            row = census.cleanup(self.worker, self.identity)
            self.assert_clean(row)
            window, = row['report']['native_windows']
            self.assertEqual(window['scope']['request_ids'], (self.expected['internal_request_id'],))
            self.assertEqual(window['scope']['worker_id'], self.identity['worker_start_identity'])
            self.assertTrue(window['scope']['dispatcher_token'])
            self.assertTrue(row['native_observation_complete'])
            self.assertEqual(window['events'], [])
            reader.delta.side_effect = RuntimeError('CPU_READER_FAILED')
            failed = self.census(native_observer_factory=lambda worker: {'aten.add.Tensor': native})
            failed.start()
            self.normal_request()
            row = failed.cleanup(self.worker, self.identity)
            self.assert_clean(row)
            self.assertEqual(row['observation_status'], 'FAILED')
            self.assertFalse(row['native_observation_complete'])
            self.assertIn('CPU_READER_FAILED', row['report']['native_windows'][0]['errors'][0])

    def test_legacy_worker_finish_retains_owner_after_exit_failure_and_retries_real_cleanup(self):
        self.worker.demoncat_worker_identity = lambda: dict(status='WORKER_IDENTIFIED',
                                                          quarantined=False, worker=self.identity)
        with patch.object(path_discovery, '_actual_worker', side_effect=lambda worker: worker):
            path_discovery.worker_path_start(self.worker, phase='prefill', expected=self.expected)
            self.normal_request()
            with patch.object(RemovableHandle, 'remove', side_effect=RuntimeError('CPU_REMOVE_FAILED')):
                with self.assertRaisesRegex(RuntimeError, 'CPU_REMOVE_FAILED'):
                    path_discovery.worker_path_finish(self.worker)
            self.assertTrue(hasattr(self.worker, path_discovery._WORKER_CAPTURE))
            self.assertGreater(len(self.model._forward_pre_hooks), 0)
            report = path_discovery.worker_path_finish(self.worker)
            self.assertFalse(hasattr(self.worker, path_discovery._WORKER_CAPTURE))
            self.assertEqual(len(self.model._forward_pre_hooks), 0)
            self.assertTrue(report['observation_complete'])
            self.assertEqual(self.normal_request().tolist(), [0.0, 4.0])


if __name__ == '__main__':
    unittest.main()
