"""CPU contracts only: actual Tensor identity, single calls and raw trace joins."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import paged33_boundary_probe as probe
from demoncat.runtime.vllm_ascend.worker_extension.paged33_audit import _meta
from demoncat.runtime.vllm_ascend.worker_extension.paged33_tnd_probe import TNDWorkerProbeSession


def fixture():
    key = torch.arange(3*2*2*4, dtype=torch.float16).reshape(3, 2, 2, 4)
    value = key.clone()+10
    module = NS(impl=NS(key_cache=key, value_cache=value))
    worker = NS(rank=0, model_runner=NS(model=object()))
    actual = dict(query=torch.arange(16, dtype=torch.float16).reshape(1, 4, 4),
        key=key.view(3, 2, 8), value=value.view(3, 2, 8), atten_mask=torch.ones(8, 8, dtype=torch.bool),
        block_table=torch.tensor([[2, 0, 1]], dtype=torch.int32), actual_seq_lengths=[1],
        actual_seq_lengths_kv=[5], input_layout='TND', block_size=2, sparse_mode=3,
        num_heads=4, num_key_value_heads=2, scale=0.5, quant_scale1=None)
    boundary = dict(module_path=probe.FIRST_LAYER, module_object_id=id(module), metadata_binding='VERIFIED',
        used_pages=[[2, 0, 1]], sequence_lengths=[5], scheduler=dict(request_ids=['real-cpu-fixture-request']),
        query=_meta(actual['query']), key=_meta(actual['key']), value=_meta(actual['value']), page_table=_meta(actual['block_table']))
    return worker, module, actual, boundary


class FakeObserver:
    def __init__(self):
        self.owner = threading.get_ident(); self.active = False
        self.begin_count = self.end_count = self.sync_count = 0
        self.end_error = None; self.end_rc = 0
        self.lib = NS(dcat_attention_observer_begin_v1=self.begin, dcat_attention_observer_end_v1=self.end)

    def begin(self, token):
        self.begin_count += 1
        return 0

    def end(self):
        self.end_count += 1
        if self.end_error is not None:
            raise self.end_error
        return self.end_rc

    def synchronize(self):
        assert self.active and self.owner == threading.get_ident()
        self.sync_count += 1


class BoundaryCaptureTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.worker, self.module, self.actual, self.boundary = fixture()
        self.syncs = 0
        self.capture = probe.FirstDecodeBoundaryCapture(self.worker, self.folder/'capture', self.sync, allow_cpu_test=True)
        self.original_calls = 0; self.returned = None

    def sync(self):
        self.syncs += 1

    def original(self, *args, **kwargs):
        self.original_calls += 1
        self.assertFalse(args)
        self.assertEqual(set(kwargs), set(self.actual))
        for key, value in self.actual.items():
            self.assertIs(kwargs[key], value)
        self.returned = (kwargs['query'] + 1, torch.empty(0))
        return self.returned

    def start(self, **changes):
        args = dict(name='npu_fused_infer_attention_score', original=self.original, args=(), kwargs=self.actual,
                    boundary=self.boundary, module=self.module, ordinal=0)
        args.update(changes)
        self.capture.start(**args)

    def test_original_arguments_and_return_objects_are_preserved_exactly_once(self):
        before = {key: value.clone() for key, value in self.actual.items() if isinstance(value, torch.Tensor)}
        self.start()
        result = self.capture.invoke_original(self.original, (), self.actual)
        self.assertIs(result, self.returned)
        self.assertEqual(self.original_calls, 1)
        row = self.capture.finalize()
        self.assertTrue(row['all_checked_values_unchanged'])
        self.assertEqual(row['inputs']['query']['snapshot']['sha256'], probe.sha(self.folder/'capture/query.bin'))
        self.assertEqual(row['block_table_values'], [[2, 0, 1]])
        self.assertIsNone(row['inputs']['quant_scale1']['value'])
        self.assertEqual(row['original_calls'], 1)
        self.assertFalse(row['fault_enabled'] or row['c2'] or row['c3'])
        for key, value in before.items():
            self.assertTrue(torch.equal(value, self.actual[key]))
        with self.assertRaisesRegex(RuntimeError, 'REENTRY'):
            self.capture.invoke_original(self.original, (), self.actual)
        self.capture.release()
        self.assertIsNone(self.capture.refs); self.assertIsNone(self.capture.result)

    def test_same_value_clone_cannot_replace_audited_or_invoked_argument(self):
        wrong = dict(self.actual, query=self.actual['query'].clone())
        with self.assertRaisesRegex(RuntimeError, 'CHANGED_AFTER_AUDIT'):
            self.start(kwargs=wrong)
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'ARGUMENT_IDENTITY_CHANGED'):
            self.capture.invoke_original(self.original, (), wrong)
        self.assertEqual(self.original_calls, 0)

    def test_nonfirst_layer_ordinal_and_foreign_module_refuse_before_call(self):
        for change in (dict(ordinal=1), dict(module=NS(impl=self.module.impl)),
                       dict(boundary=dict(self.boundary, module_path='model.layers.1.self_attn.attn'))):
            with self.subTest(change=list(change)), self.assertRaisesRegex(RuntimeError, 'FIRST_ACTUAL_LAYER'):
                self.start(**change)
        self.assertEqual(self.original_calls, 0)

    def test_runtime_capture_refuses_unknown_outer_wrapper_before_any_device_action(self):
        self.capture.allow_cpu_test = False
        actual_packet = object()
        def wrapper(**kwargs): return self.original(**kwargs)
        wrapper.__wrapped__ = actual_packet
        with patch.object(torch.ops, 'npu', NS(npu_fused_infer_attention_score=actual_packet)), \
                self.assertRaisesRegex(RuntimeError, 'ACTUAL_PUBLIC_PACKET_REQUIRED'):
            self.start(original=wrapper)
        self.assertEqual(self.original_calls, 0); self.assertEqual(self.syncs, 0)

    def test_immediate_output_survives_late_output_query_and_cache_mutations(self):
        self.start(); self.capture.invoke_original(self.original, (), self.actual)
        immediate = self.capture.data['outputs'][0]['sha256']
        self.returned[0].add_(4); self.actual['query'].add_(2); self.module.impl.key_cache[2, 0].add_(1)
        row = self.capture.finalize()
        self.assertEqual(probe.sha(self.folder/'capture/output-0.bin'), immediate)
        self.assertNotEqual(immediate, row['outputs_after'][0]['sha256'])
        self.assertFalse(row['subsequent_checks']['query'])
        self.assertFalse(row['subsequent_checks']['logical_key'])
        self.assertFalse(row['subsequent_checks']['output_0'])
        self.assertFalse(row['all_checked_values_unchanged'])

    def test_inference_tensor_missing_version_is_explicit_and_bytes_still_checked(self):
        with torch.inference_mode():
            self.actual['query'] = torch.ones(1, 4, 4, dtype=torch.float16)
            self.boundary['query'] = _meta(self.actual['query'])
            self.start(); self.capture.invoke_original(self.original, (), self.actual)
            self.assertIsNone(self.capture.data['inputs']['query']['snapshot']['version'])
            self.actual['query'].add_(1)
            row = self.capture.finalize()
        self.assertFalse(row['subsequent_checks']['query'])
        self.assertIn('INTERVENING_WRITES_WITHOUT_VERSION_UNOBSERVED', row['mutation_observation'])

    def test_original_failure_is_not_retried_and_drain_failure_keeps_references(self):
        failure = ValueError('original-first-error')
        def failing(**kwargs):
            self.original_calls += 1
            raise failure
        self.start(original=failing)
        with self.assertRaises(ValueError) as raised:
            self.capture.invoke_original(failing, (), self.actual)
        self.assertIs(raised.exception, failure); self.assertEqual(self.original_calls, 1)
        with self.assertRaisesRegex(RuntimeError, 'REENTRY'):
            self.capture.invoke_original(failing, (), self.actual)
        def failed_drain():
            raise RuntimeError('drain still pending')
        self.capture.synchronize = failed_drain
        with self.assertRaisesRegex(RuntimeError, 'drain still pending'):
            self.capture.release()
        self.assertIsNotNone(self.capture.refs)
        self.capture.synchronize = self.sync; self.capture.release()
        self.assertIn('original-first-error', json.loads((self.folder/'capture/boundary.json').read_text())['errors'][0]['error'])

    def test_changed_callable_is_refused_even_with_identical_return_values(self):
        self.start()
        with self.assertRaisesRegex(RuntimeError, 'ORIGINAL_CALLABLE_CHANGED'):
            self.capture.invoke_original(lambda **kwargs: (kwargs['query'] + 1, torch.empty(0)), (), self.actual)
        self.assertEqual(self.capture.calls, 0)

    def test_cache_write_then_restore_is_detected_when_version_is_available(self):
        self.start(); self.capture.invoke_original(self.original, (), self.actual)
        cache = self.module.impl.key_cache; saved = cache.clone()
        cache.add_(1); cache.copy_(saved)
        row = self.capture.finalize()
        self.assertEqual(row['logical_kv_before']['key']['sha256'], row['logical_kv_after']['key']['sha256'])
        self.assertFalse(row['subsequent_checks']['logical_key'])

    def test_foreign_thread_or_replaced_worker_model_cannot_release_references(self):
        self.start(); errors = []
        def foreign():
            try: self.capture.release()
            except RuntimeError as error: errors.append(str(error))
        thread = threading.Thread(target=foreign); thread.start(); thread.join()
        self.assertEqual(len(errors), 1); self.assertIsNotNone(self.capture.refs)
        model = self.worker.model_runner.model; self.worker.model_runner.model = object()
        with self.assertRaisesRegex(RuntimeError, 'WORKER_MODEL_OR_THREAD_CHANGED'):
            self.capture.release()
        self.worker.model_runner.model = model; self.capture.release()

    def test_stock_and_private_session_paths_capture_once_and_return_the_original(self):
        for private in (False, True):
            with self.subTest(private=private):
                worker, module, actual, boundary = fixture()
                capture = probe.FirstDecodeBoundaryCapture(worker, self.folder/str(private), lambda: None, allow_cpu_test=True)
                session = TNDWorkerProbeSession.__new__(TNDWorkerProbeSession)
                session.stack = [(probe.FIRST_LAYER, module)]; session.rows = [boundary]; session.tensor_refs = {}
                session.boundary_capture = capture
                calls = []; returned = (actual['query'] + 1, torch.empty(0))
                def original(**kwargs):
                    calls.append(kwargs); return returned
                def execute(values, row, invoke):
                    self.assertIs(values['query'], actual['query'])
                    result = invoke(); session.adapter.calls.append(dict(call_id='a'*32)); return result
                session.adapter = NS(execute=execute, calls=[], runtime_dir=self.folder) if private else None
                session.boundary_stock_scope = None if private else probe.StockLaunchScope(FakeObserver(), lambda: None, self.folder)
                with patch.object(probe, 'read_scoped_launches', return_value={'cpu_fixture': True}), \
                        patch.object(probe, 'mapped_api_libraries', return_value={'selected_api_owner_observed': False}):
                    result = session._invoke_api('npu_fused_infer_attention_score', original, (), actual, boundary)
                    self.assertIs(result, returned)
                    self.assertEqual(len(calls), 1)
                    next_row = dict(boundary, module_path='model.layers.1.self_attn.attn')
                    session.stack = [(next_row['module_path'], module)]; session.rows.append(next_row)
                    session._invoke_api('npu_fused_infer_attention_score', original, (), actual, next_row)
                self.assertEqual(len(calls), 2)
                self.assertEqual(capture.calls, 1)
                self.assertTrue(capture.finalize()['all_checked_values_unchanged'])
                capture.release()


class ScopeRecoveryTests(unittest.TestCase):
    def test_stock_refuses_foreign_native_policy_before_original_call(self):
        scope = probe.StockLaunchScope(FakeObserver(), lambda: None, Path('.')); calls = []
        with patch.dict('os.environ', {'DCAT33_TND_MODE': 's2_256'}), self.assertRaisesRegex(RuntimeError, 'FOREIGN_NATIVE_POLICY'):
            scope.invoke(lambda: calls.append(1))
        self.assertEqual(calls, [])

    def test_original_error_wins_over_end_error_and_owned_scope_can_retry(self):
        observer = FakeObserver(); scope = probe.StockLaunchScope(observer, lambda: None, Path('.'))
        observer.end_error = RuntimeError('end-failure')
        first = ValueError('original-first'); calls = []
        def original():
            calls.append(1); raise first
        with self.assertRaises(ValueError) as raised:
            scope.invoke(original)
        self.assertIs(raised.exception, first); self.assertEqual(calls, [1])
        self.assertTrue(scope.active and observer.active)
        observer.end_error = None
        state = scope.close()
        self.assertTrue(state['scope_cleared']); self.assertFalse(observer.active)
        self.assertEqual([item['stage'] for item in state['errors']], ['invoke', 'end'])
        with self.assertRaisesRegex(RuntimeError, 'ALREADY_USED'):
            scope.invoke(original)

    def test_stock_scope_sets_shared_observer_active_and_refuses_inflight_close(self):
        observer = FakeObserver(); scope = probe.StockLaunchScope(observer, lambda: None, Path('.'))
        value = object()
        def original():
            with self.assertRaisesRegex(RuntimeError, 'STILL_INFLIGHT'): scope.close()
            return value
        self.assertIs(scope.invoke(original), value)
        self.assertEqual((observer.begin_count, observer.end_count, observer.sync_count), (1, 1, 1))
        self.assertTrue(scope.close()['scope_cleared'])

    def test_native_end_emit_failure_is_failed_capture_but_confirmed_token_clear(self):
        observer = FakeObserver(); observer.end_rc = -2
        scope = probe.StockLaunchScope(observer, lambda: None, Path('.'))
        with self.assertRaisesRegex(RuntimeError, 'END_FAILED:-2'): scope.invoke(lambda: None)
        self.assertFalse(scope.active or observer.active)
        self.assertTrue(scope.close()['scope_cleared'])

    def test_uncertain_native_end_keeps_owner_until_a_confirmed_retry(self):
        observer = FakeObserver(); observer.end_rc = -1
        scope = probe.StockLaunchScope(observer, lambda: None, Path('.'))
        with self.assertRaisesRegex(RuntimeError, 'END_FAILED:-1'): scope.invoke(lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'END_FAILED:-1'): scope.close()
        self.assertTrue(scope.active and observer.active)
        observer.end_rc = 0
        self.assertTrue(scope.close()['scope_cleared'])


class StockRawTraceTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.token = 'a'*32; binary = self.folder/'registered-12-1.bin'; binary.write_bytes(b'CPU fixture registered bytes')
        def row(event, now, **fields):
            return dict(schema='demoncat.attention.runtime_observer.v1', event=event, pid=12, tid=34,
                        call_id=self.token, monotonic_ns=now, **fields)
        self.rows = [row('register', 1, handle=80, generation=1, binary_file=binary.name, binary_bytes=binary.stat().st_size, result=0),
                     row('begin', 10)]
        self.rows[0]['call_id'] = ''
        for index, key in enumerate(('123', '987654321')):
            bind = row('function_bind', 2+index, handle=80, generation=1, function_handle=90+index, tiling_key=key, result=0)
            bind['call_id'] = ''; self.rows.insert(1, bind)
            self.rows.append(row('launch', 23+index*5, api='aclrtLaunchKernelWithHostArgs', handle=80, generation=1,
                function_handle=90+index, binary_file=binary.name, result=0, observer_mutates_arguments=False,
                launch_begin_ns=20+index*5, launch_end_ns=22+index*5, args_hex='0001020304050607', args_size=8,
                snapshot_valid=True, tiling_key=key, block_dim=6+index, schedule_mode=1, argument_layout='aclrt_host_placeholders'))
        self.rows += [row('device_sync', 40, result=0), row('end', 50)]

    def read(self):
        (self.folder/'runtime.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.rows))
        return probe.read_scoped_launches(self.folder, self.token)

    def test_multiple_stock_launches_are_kept_without_private_key_or_abi_assumption(self):
        report = self.read()
        self.assertEqual(report['intercepted_launch_count'], 2)
        self.assertEqual([row['tiling_key'] for row in report['launches']], ['123', '987654321'])
        self.assertEqual(report['launches'][0]['registered_binary']['sha256'], probe.sha(self.folder/'registered-12-1.bin'))
        self.assertFalse(report['argument_abi_qualified'] or report['actual_api_library_owner_observed'])
        self.assertEqual(report['observation_gaps'], [])

    def test_foreign_owner_function_join_and_filename_cannot_be_inferred(self):
        launch = next(row for row in self.rows if row['event'] == 'launch')
        for key, value, message in (('tid', 999, 'OWNER_OR_ORDER'), ('function_handle', 999, 'FUNCTION_JOIN'),
                                    ('binary_file', '../outside', 'BINARY_NAME')):
            saved = launch[key]; launch[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, message): self.read()
            launch[key] = saved

    def test_overflow_invalid_snapshot_and_unknown_schedule_remain_explicit_gaps(self):
        launch = next(row for row in self.rows if row['event'] == 'launch')
        launch.update(snapshot_valid=False, schedule_mode=-1)
        self.rows.append(dict(event='capture_overflow', call_id=''))
        report = self.read()
        self.assertEqual(set(report['observation_gaps']), {'REGISTERED_BINARY_CAPTURE_OVERFLOW',
            'INVALID_RAW_ARGUMENT_SNAPSHOT', 'SCHEDULE_MODE_NOT_OBSERVED'})
        self.assertFalse(report['argument_abi_qualified'])


class BoundaryCaseTests(unittest.TestCase):
    def test_manifest_uses_same_no_statistics_capture_and_observer_for_two_independent_cases(self):
        folder = Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'
        with patch.object(sys, 'path', [str(folder), *sys.path]):
            import prepare_attention33_cases as preparer
            import attention33_contract as contract
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd
        from fault_injection.native_adapters.attention28 import launch
        from fault_injection.native_adapters.attention28.fia import binary_registration
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = root/'source'; files = []
        names = ['experiments/c2c3_closure_20261117/'+name for name in
            ('attention33_worker_probe.py', 'attention33_contract.py', 'attention33_bridge.py', 'prepare_attention33_cases.py')]
        names += ['src/demoncat/runtime/vllm_ascend/worker_extension/'+name for name in
                  ('paged33_boundary_probe.py', 'paged33_tnd_probe.py')]
        for name in names:
            target = source/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('# CPU fixture\n')
            files.append(dict(path=name, sha256=contract.sha(target)))
        (source/'source_manifest.json').write_text(json.dumps(dict(schema='demoncat.c2c3.source.v1', files=files, base_commit='fixture')))
        observer_source = source/'fault_injection/native_adapters/attention28/launch_observer.cpp'
        observer_source.parent.mkdir(parents=True); observer_source.write_text('// observer CPU fixture\n')
        build = root/'fresh/build/run'; (build/'evidence').mkdir(parents=True)
        (build/'evidence/completion.json').write_text('{}')
        (root/'fresh/controller.json').write_text(json.dumps(dict(status='BUILD_COMMAND_COMPLETED', actual_npu_calls=0,
            image_id='sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0')))
        observer = root/'observer/attention-preflight'; observer.mkdir(parents=True)
        library = observer/'observer.so'; library.write_bytes(b'CPU fixture')
        (observer/'result.json').write_text(json.dumps(dict(observer_compiled=True, actual_npu_calls=0,
            observer_memory_selftest=dict(exit_code=0), observer_source=dict(sha256=contract.sha(observer_source)),
            observer_library=dict(path='/output/cann851/attention-preflight/observer.so', sha256=contract.sha(library)))))
        bundle = dict(artifacts={'new.o': {}}, completion=dict(kernel_binary_sha256='new',
            variant='tnd_page16_fai_fd_s2_256', fai_public_schema_sha256='schema'))
        args = NS(source=source, installed=root/'installed', model=root/'model', build_root=build,
            observer=observer.parent, stage='baseline', card=2, timeout=1800, bridge=None,
            stock_run=None, baseline_run=None, boundary_diagnostic=True)
        with patch.object(paged33_tnd, 'verify_bundle', return_value=bundle), \
                patch.object(launch, 'bundle_kernel_abi', return_value=({}, {})), \
                patch.object(binary_registration, 'validate_metadata'), \
                patch.object(preparer, 'execution_contract', return_value={'CPU': 'fixture'}), \
                patch.object(sys, 'path', list(sys.path)):
            cases = preparer.prepare(args)
            self.assertEqual([row['id'] for row in cases], ['attention33-fd-first-boundary-stock', 'attention33-fd-first-boundary-baseline'])
            self.assertEqual(cases[0]['env'], cases[1]['env'])
            for row in cases:
                self.assertIn('--first-boundary-diagnostic', row['command'])
                self.assertIn('--observer-receipt', row['command'])
                self.assertNotIn('--statistics', row['command'])
                self.assertNotIn('--phase fault', row['command'])
                self.assertFalse(row['review']['fault_authorized'])
                self.assertIn([str(observer), '/output/cann851/attention-preflight', 'ro'], row['mounts'])
            self.assertNotIn('--vendor', cases[0]['command'])
            args.stage = 'fault'
            with self.assertRaisesRegex(RuntimeError, 'BOUNDARY_DIAGNOSTIC_NEVER_FAULT'):
                preparer.prepare(args)

    def test_driver_refuses_fault_or_statistics_before_native_setup(self):
        folder = Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'
        with patch.object(sys, 'path', [str(folder), *sys.path]):
            import attention33_worker_probe as driver
        for extra in (['--phase', 'fault'], ['--phase', 'stock', '--statistics']):
            argv = ['driver', '--experimental-qualification', '--first-boundary-diagnostic', *extra]
            with patch.object(sys, 'argv', argv), patch.object(driver, 'Path', PurePosixPath), \
                    self.assertRaisesRegex(RuntimeError, 'NEVER_AUTHORIZES_FAULT_OR_STATISTICS'):
                driver.main()


if __name__ == '__main__':
    unittest.main()
