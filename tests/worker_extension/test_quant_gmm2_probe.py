"""Same-process probe orchestration with explicit CPU model/Native doubles."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fault_injection import quant_gmm2_native as native
from experiments.worker_core_v11_20261118 import probe_quant_gmm2 as probe
from experiments.worker_core_v11_20261118.prepare_quant_gmm2_case import adapt
from test_quant_gmm2_native import fixture


class TraceDouble:
    def __init__(self):
        self.started = False; self.installed = False; self.failures = []; self.worker_identity = None
        self.calls = []; self.fail_stop = self.fail_release = self.fail_snapshot = False
        self.value = fixture(26); self.startup = copy.deepcopy(self.value['gmm2']['startup'])
    def failure(self, action, error):
        self.failures.append(dict(sequence=len(self.failures)+1, action=action, error=repr(error)))
    def bind_worker(self, worker):
        self.calls.append('bind_worker'); self.worker_identity = copy.deepcopy(self.value['gmm2']['worker_identity'])
        return copy.deepcopy(self.worker_identity)
    def verify_worker(self, worker): self.calls.append('verify_worker')
    def begin(self): self.calls.append('begin'); self.started = self.installed = True
    def stop(self):
        self.calls.append('stop')
        if self.fail_stop: raise RuntimeError('ACTUAL_SCOPE_STILL_ACTIVE')
    def query_all(self):
        self.calls.append('query')
        return dict(containers=dict(installed=int(self.installed), active_calls=0, live_registrations=0,
            scope_token=0, record_capacity_bytes=0, payload_capacity_bytes=0,
            arrays=dict(live_arrays=0, active_captures=0, active_getters=0)),
            gmm2=dict(installed=int(self.installed), active_calls=0, pending_executors=0,
                untracked_executors=0, record_capacity_bytes=0))
    def snapshot(self):
        self.calls.append('snapshot')
        if self.fail_snapshot: raise RuntimeError('ACTUAL_CALL_STILL_ACTIVE')
        result = copy.deepcopy(self.value); result['gmm2']['failures'] = copy.deepcopy(self.failures)
        return result
    def release(self):
        self.calls.append('release')
        if self.fail_release: raise RuntimeError('PREPARED_EXECUTOR_STILL_PENDING')
        self.installed = False
        return self.query_all()


class ProbeContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.worker = SimpleNamespace()
        self.trace = TraceDouble()
        self.worker._demoncat_container_trace = self.trace
        self.addCleanup(patch.stopall)
        patch.object(probe.common, 'actual', return_value=self.worker).start()
        patch.object(probe, 'Path', side_effect=lambda name: self.root / str(name).lstrip('/')).start()
        patch.object(native, '_startup_trace', self.trace).start()
    def folder(self): return self.root/'evidence/container-trace'

    def test_query_has_no_collect_resolution_or_owner_change(self):
        session = SimpleNamespace(collect=Mock())
        self.worker._demoncat_shared_scale_inventory = session
        before = self.trace.installed
        probe.query(self.worker)
        self.assertEqual(self.trace.calls, ['verify_worker', 'query'])
        self.assertEqual(self.trace.installed, before); session.collect.assert_not_called()

    def test_explicit_collect_stops_observation_before_inventory_collection(self):
        session = SimpleNamespace(collect=Mock(side_effect=lambda: self.trace.calls.append('collect')),
                                  report=lambda: dict(complete=True))
        self.worker._demoncat_shared_scale_inventory = session
        self.assertTrue(probe.advance(self.worker, collect=True)['complete'])
        self.assertEqual(self.trace.calls, ['verify_worker', 'stop', 'collect'])

    def test_failed_stop_saves_raw_attempt_and_does_not_release(self):
        self.trace.started = self.trace.installed = True; self.trace.fail_stop = True
        first = probe.cleanup(self.worker)
        self.assertTrue(first['errors']); self.assertNotIn('release', self.trace.calls)
        self.assertIs(self.worker._demoncat_container_trace, self.trace)
        raw = (self.folder()/'cleanup-observation-001.json').read_bytes()
        saved = json.loads(raw)
        self.assertEqual(saved['gmm2']['events'], self.trace.value['gmm2']['events'])
        self.trace.fail_stop = False
        second = probe.cleanup(self.worker)
        self.assertFalse(second['errors']); self.assertEqual(second['first_failure']['action'], 'cleanup_stop')
        self.assertFalse(hasattr(self.worker, '_demoncat_container_trace'))
        self.assertEqual((self.folder()/'cleanup-observation-001.json').read_bytes(), raw)
        self.assertTrue((self.folder()/'cleanup-observation-002.json').exists())

    def test_failed_release_keeps_owner_original_and_first_error_for_retry(self):
        self.trace.started = self.trace.installed = True; self.trace.fail_release = True
        first = probe.cleanup(self.worker)
        self.assertTrue(first['errors']); self.assertIs(native.startup_trace(), self.trace)
        raw = (self.folder()/'cleanup-observation-001.json').read_bytes()
        self.trace.fail_release = False
        second = probe.cleanup(self.worker)
        self.assertFalse(second['errors']); self.assertEqual(second['first_failure']['action'], 'cleanup_release')
        self.assertEqual((self.folder()/'cleanup-observation-001.json').read_bytes(), raw)
        with self.assertRaisesRegex(RuntimeError, 'SAME_PROCESS_PRE_LLM_HANDLE_MISSING'): native.startup_trace()

    def test_unknown_active_call_retains_owner_and_never_calls_release(self):
        self.trace.started = self.trace.installed = True; self.trace.fail_snapshot = True
        result = probe.cleanup(self.worker)
        self.assertTrue(result['errors']); self.assertNotIn('release', self.trace.calls)
        self.assertIs(native.startup_trace(), self.trace)

    def test_lost_trace_reference_cannot_claim_existing_inventory_is_clean(self):
        del self.worker._demoncat_container_trace
        self.worker._demoncat_shared_scale_inventory = object()
        with patch.object(native, '_startup_trace', None):
            result = probe.cleanup(self.worker)
        self.assertEqual(result['errors'], ['STARTUP_HANDLE_MISSING_WITH_OWNED_INVENTORY'])
        self.assertTrue(hasattr(self.worker, '_demoncat_shared_scale_inventory'))

    def test_foreign_model_identity_refuses_cleanup_without_using_hooks(self):
        self.trace.verify_worker = Mock(side_effect=RuntimeError('MODEL_CHANGED'))
        self.trace.worker_identity = {}
        self.worker._demoncat_shared_scale_inventory = SimpleNamespace(close=Mock())
        result = probe.cleanup(self.worker)
        self.assertEqual(result['errors'], ["RuntimeError('MODEL_CHANGED')"])
        self.worker._demoncat_shared_scale_inventory.close.assert_not_called()
        self.assertNotIn('release', self.trace.calls)

    def test_failure_to_remove_source_observer_does_not_free_native_journal(self):
        module = 'demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory'
        self.worker._demoncat_shared_scale_inventory = object()
        self.trace.started = self.trace.installed = True
        with patch(module+'.remove_shared_scale_inventory', side_effect=RuntimeError('FOREIGN_MODE_ABOVE_OWNED')):
            result = probe.cleanup(self.worker)
        self.assertTrue(result['errors']); self.assertNotIn('release', self.trace.calls)
        self.assertTrue((self.folder()/'cleanup-observation-001.json').exists())

    def test_complete_collect_keeps_all_typed_and_container_originals(self):
        self.worker._demoncat_shared_scale_inventory = SimpleNamespace(source_rows=SimpleNamespace(
            container_trace=SimpleNamespace(report=lambda: dict(errors=[], active=0, calls=104))))
        result = probe.collect(self.worker)
        self.assertEqual(result['assessment']['status'], 'CURRENT_WEIGHTNZ_SLOTS_AND_EXECUTION_OBSERVED')
        raw = (self.folder()/'observation.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), result['sha256'])
        self.assertEqual(len(json.loads(raw)['gmm2']['events']), 104)
        with self.assertRaises(FileExistsError): probe.collect(self.worker)
        self.assertEqual((self.folder()/'observation.json').read_bytes(), raw)

    def test_scope_failure_is_saved_instead_of_hiding_behind_native_complete(self):
        self.worker._demoncat_shared_scale_inventory = SimpleNamespace(source_rows=SimpleNamespace(
            container_trace=SimpleNamespace(report=lambda: dict(errors=['CHANGED_SOURCE'], active=0))))
        result = probe.collect(self.worker)
        self.assertIn('SOURCE_SCOPE_OBSERVER_NOT_COMPLETE', result['assessment']['reasons'])
        self.assertEqual(result['assessment']['status'], 'OBSERVABILITY_REFUSED')
        self.assertFalse(result['assessment']['request_rows_proven'])

    def test_missing_original_owner_rejects_before_llm_constructor(self):
        constructor = Mock()
        argv = ['probe', '--quant-build-sha256', 'a'*64, '--trace-build-sha256', 'b'*64, '--support-sha256', 'c'*64]
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, vllm=SimpleNamespace(LLM=constructor)), \
                patch.object(probe, 'support_source', return_value={}), \
                patch.object(native, 'prepare_startup', side_effect=RuntimeError('EXACT_ORIGINAL_MISSING')):
            self.assertEqual(probe.main(), 2)
        constructor.assert_not_called()
        result = json.loads((self.root/'evidence/result.json').read_bytes())
        self.assertEqual(result['phases'], []); self.assertIn('EXACT_ORIGINAL_MISSING', result['error'])
        self.assertEqual(result['failure_noload_state']['gmm2']['installed'], 0)

    def test_one_natural_batch_with_healthy_before_after_uses_standard_worker_only(self):
        worker, trace = self.worker, self.trace
        transcript = []
        ids = ['request-'+str(n) for n in range(8)]
        mapping = [dict(internal_request_id=x, external_request_id='external-'+x) for x in ids]
        census = dict(actual_generations=1, structural_candidates=135, actual_projection_calls=135,
            actual_native_calls=135, actual_public_calls=270, row_operator_observation_complete=True,
            actual_native_prepare_calls=135, actual_native_execute_calls=135, native_public_binding_complete=True,
            native_faults=0, request_row_proven_calls=10)
        session = SimpleNamespace(expect=lambda actual_ids: transcript.append(('expect', actual_ids)),
            begin_decode_capture=lambda: transcript.append(('capture', None)), collect=lambda: transcript.append(('collect', None)),
            report=lambda: copy.deepcopy(census), source_rows=SimpleNamespace(container_trace=SimpleNamespace(
                report=lambda: dict(errors=[], active=0, calls=104))))
        def setup(*args):
            worker._demoncat_shared_scale_inventory = session
            return dict(zero_fault=True)
        def remove(worker):
            del worker._demoncat_shared_scale_inventory
            return dict(owned_hooks=0, owned_source_profile_callbacks=0, owned_row_dispatch_modes=0)
        class LLM:
            def __init__(self, **kwargs):
                transcript.append(('construct', kwargs)); self.llm_engine = self; self.output_processor = object(); self.running = False
            def collective_rpc(self, function, args, timeout):
                if isinstance(function, str): return [trace.worker_identity]
                return [function(worker, *args)]
            def get_tokenizer(self): return SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])
            def generate(self, prompts, params, use_tqdm):
                transcript.append(('healthy', None))
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])]) for _ in prompts]
            def has_unfinished_requests(self): return self.running
            def reset_prefix_cache(self, **kwargs): return True
            def enqueue(self, *args, **kwargs): self.running = True; transcript.append(('enqueue', ids)); return ids
            def step(self): return [SimpleNamespace(finished=False)]
            def wait_for_completion(self, **kwargs):
                self.running = False
                return [SimpleNamespace(request_id=row['external_request_id'], outputs=[SimpleNamespace(token_ids=[1, 2])]) for row in mapping]
        def startup(*args): transcript.append(('preflight', None)); return copy.deepcopy(trace.startup)
        argv = ['probe', '--quant-build-sha256', 'a'*64, '--trace-build-sha256', 'b'*64, '--support-sha256', 'c'*64]
        module = 'demoncat.runtime.vllm_ascend.worker_extension.'
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, vllm=SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs)), \
                patch.object(probe, 'support_source', return_value={}), patch.object(probe, 'setup', side_effect=setup), \
                patch.object(native, 'prepare_startup', side_effect=startup), \
                patch(module+'serving_context.require_host_driven_core'), \
                patch(module+'serving_context.actual_driver_request_map', return_value=mapping), \
                patch(module+'serving_shared_scale_inventory.remove_shared_scale_inventory', side_effect=remove):
            self.assertEqual(probe.main(), 0)
        self.assertEqual([name for name, _ in transcript], ['preflight', 'construct', 'healthy', 'enqueue', 'expect', 'capture', 'collect', 'healthy'])
        kwargs = transcript[1][1]
        self.assertEqual(kwargs['worker_extension_cls'], 'demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
        self.assertTrue(kwargs['enforce_eager']); self.assertEqual(kwargs['tensor_parallel_size'], 1)
        result = json.loads((self.root/'evidence/result.json').read_bytes())
        self.assertEqual(result['status'], 'ZERO_F_CURRENT_WEIGHTNZ_ENTRY_EVIDENCE')
        self.assertEqual(len(result['phases']), 1); self.assertEqual(result['phases'][0]['census'], census)
        self.assertEqual(result['healthy_before'], result['healthy_after'])
        self.assertFalse(result['cleanup']['errors']); self.assertFalse(result['c2']); self.assertFalse(result['request_rows_proven'])


class CaseContracts(unittest.TestCase):
    def test_fixed_inproc_probe_preserves_separate_native_compile_and_runtime_sources(self):
        case = dict(id='base', mounts=[], env={'VLLM_ENABLE_V1_MULTIPROCESSING': '1'},
            review=dict(source_commit='d'*40, resource_budget={}))
        value = adapt(case, trace=PurePosixPath('/root/native-sdk106'), trace_sha='a'*64, quant_sha='b'*64,
            support=PurePosixPath('/root/source-new/probe_quant_containers.py'), support_sha='c'*64, native_source_commit='e'*40)
        self.assertEqual(value['review']['source_commit'], 'd'*40)
        self.assertEqual(value['review']['native_compile_source_commit'], 'e'*40)
        self.assertEqual(value['env']['VLLM_ENABLE_V1_MULTIPROCESSING'], '0')
        self.assertEqual(value['env']['TASK_QUEUE_ENABLE'], '0')
        self.assertTrue(all(row[2] == 'ro' for row in value['mounts']))
        self.assertIn('--support-sha256 '+'c'*64, value['command'])
        self.assertEqual(value['review']['resource_budget']['expected_natural_gmm2_pairs'], 26)
        self.assertEqual(value['review']['expected_hits'], 0)
        self.assertNotIn('mixin', value['command']); self.assertNotIn('capture-integer-routes', value['command'])

    def test_new_support_path_must_be_absolute_and_safe(self):
        for value in ('relative.py', '/root/line\n.py', '/root/a,b.py'):
            with self.assertRaises(ValueError):
                adapt({}, trace='/root/trace', trace_sha='a'*64, quant_sha='b'*64, support=value,
                      support_sha='c'*64, native_source_commit='d'*40)


if __name__ == '__main__':
    unittest.main()
