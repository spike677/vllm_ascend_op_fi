"""Actual Host/Runtime and Torch hooks, explicit CPU Native/scheduler doubles.

These exercise the public ownership protocol, not Native or NPU qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import identity, serving26_core as core
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from test_core_host import ExistingExecutor
from test_extension import _extended_worker
from test_writer26_inventory import CpuReader, Producer, Writer
from test_writer26_pending_consumer import FailedReceiptNative
import test_writer26_fia_typed as typed
import test_writer26_public_return as public
import test_serving_session_lifecycle as life


class OwnedNative:
    """Supply the real owner-state ABI fields omitted by older CPU doubles."""
    def __init__(self, reader):
        super().__init__(reader)
        self.owner = 'EXPLICIT_CPU_INSPECTION'
        self.build = {'files': {'libfault26_readonly_trace_abi0.so': reader.library_sha256}}
        self.after_prepare_failure = False

    def state(self):
        value = super().state()
        value['core'].update(owner_hi=self.config.owner_hi, owner_lo=self.config.owner_lo,
            generation=self.config.generation, thread=self.config.thread)
        return value

    def seal(self, targets):
        assert len(targets) == 1 and not self.sealed and self.transaction is None
        assert [mode for _, mode, _ in self.executes] == ['reference', 'lossless', 'off']
        assert {key for _, _, key in self.executes} == set(targets)
        self.sealed = True

    def execute(self, *args, **kwargs):
        value = super().execute(*args, **kwargs)
        if self.after_prepare_failure:
            raise RuntimeError('EXPLICIT_CPU_AFTER_PREPARE')
        return value


class QmmNative(OwnedNative, FailedReceiptNative):
    pass


class FiaNative(OwnedNative, typed.TypedNative):
    pass


class ActualWriters(torch.nn.Module):
    def __init__(self, native, count):
        super().__init__()
        self.native = native
        self.producer = Producer()
        self.writers = torch.nn.ModuleList([public.CopiedAttention() if native.fia else Writer(native)
                                           for _ in range(count)])
        self.alias_of_writer = self.writers[-1]
        self.root_calls = 0
        self.head = torch.nn.Linear(576 if native.fia else 16, 3, bias=False, dtype=torch.bfloat16)
        self.processor = life.CpuProcessor()
        with torch.no_grad():
            self.head.weight.fill_(.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        self.root_calls += 1
        x = torch.ones(input_ids.numel(), self.head.in_features, dtype=torch.bfloat16)
        residual = torch.zeros_like(x)
        if self.native.fia:
            self.native.last, self.native.sequence = None, 0
        else:
            residual = self.producer(x, residual)
        for writer in self.writers:
            x = writer(x)
            residual = self.producer(x, residual)
        return residual


class CoreWriter26Tests(unittest.TestCase):
    snapshot = life.ServingSessionLifecycleTests.snapshot
    root = life.ServingSessionLifecycleTests.root
    forward = life.ServingSessionLifecycleTests.forward

    def setUp(self):
        life.ServingSessionLifecycleTests.setUp(self)
        self.worker = _extended_worker(rank=0)
        self.worker.load_model()
        self.runner = self.worker.model_runner
        self.runner.vllm_config = NS(model_config=NS(hf_config=NS(vocab_size=3)))
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.enterContext(patch.object(core, 'require_serial_eager'))
        self.enterContext(patch.object(core, 'Path', side_effect=lambda value:
            self.evidence_root if str(value) == '/evidence' else Path(value)))
        self.executor = ExistingExecutor([self.worker])
        self.provider = None

    def make(self, *, fia=False, count=2):
        self.reader = CpuReader()
        self.native = (FiaNative if fia else QmmNative)(self.reader)
        self.model = ActualWriters(self.native, count).eval()
        self.runner.model = self.model
        self.mechanism = self.native.mechanism
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        if fia:
            self.enterContext(patch.object(public, '_RUNTIME', self.native))
            self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(
                npu_fused_infer_attention_score=public._PACKET, get_npu_format=lambda _: 2)}))
            original = typed.descriptor
            def flat(tensor, **kwargs):
                value = original(tensor, **kwargs)
                value['storage_rank'], value['storage'] = 1, [tensor.numel()]+[0]*7
                return value
            self.enterContext(patch.object(typed, 'descriptor', flat))
        else:
            append = self.reader.append
            def event(kind, **kwargs):
                if kind == 1:
                    kwargs['cache'] = 1000 + (self.native.writer_calls-1) % count
                return append(kind, **kwargs)
            self.enterContext(patch.object(self.reader, 'append', event))
            def allocation(backend, call, row):
                row['allocation'] = dict(residual_block=[call.residual, call.output],
                    output_block=[call.output, call.output+call.output_bytes], shift=32)
            self.enterContext(patch.object(core.CoreWriter26Experiment, '_allocation', allocation))
        def native_factory(owner, generation):
            self.native.owner, self.native.config.generation = owner, generation
            return self.native
        def bind(actual, config):
            self.assertEqual(actual, self.worker.demoncat_worker_identity()['worker'])
            self.assertEqual(config.mechanism, self.mechanism)
            core.install_writer26_core_binding(self.worker, mechanism=self.mechanism,
                prompt_token_ids=list(range(1, 14 if fia else 5)), evidence_dir=self.evidence_root/'core26',
                manifest_sha256='b'*64, _simulation=True, _runtime_factory=native_factory, _reader=self.reader)
            self.assertFalse(self.native.installed or self.model.root_calls or self.model._forward_hooks)
            return {}
        self.client = WorkerFaultClient(HostAdapter(self.executor, prepare_binding=bind),
            FaultConfig(26, 'fia_destination' if fia else 'residual_stomp'))
        self.addCleanup(self.dispose)

    def dispose(self):
        if self.provider:
            self.native.after_prepare_failure = False
            if hasattr(self.native, 'drain_fails'):
                self.native.drain_fails = False
            if self.provider.backend and self.provider.backend.active:
                self.provider.backend.fail('CPU_FIXTURE_DISPOSAL')
            if self.provider.session:
                self.provider.session.close()
            self.client.unregister()
            self.client.detach()
        if getattr(self.worker, core.ATTRIBUTE, None):
            core.remove_writer26_core_binding(self.worker, self.mechanism)

    def ok(self, reply):
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', {k: v for k, v in reply['response'].items()
            if k not in ('provider_state', 'ticket')})
        return reply['response']['provider_state']

    def start(self):
        self.ok(self.client.prepare())
        self.provider = self.worker._demoncat_worker_fault_runtime._leases[self.mechanism].provider.core
        for action in ('register', 'check', 'enable'):
            self.ok(getattr(self.client, action)())

    def enqueue(self, index, *, collect=True):
        self.actual_request_id = 'actual-cpu-'+str(index)
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            output = self.forward(self.provider.session)
        if collect:
            self.ok(self.client.collect_replay())
        return output

    def test_public_qmm_complete_135_denominator_and_one_fresh_site_eight_actual_requests(self):
        self.make(count=135)
        self.complete(135, 135)

    def test_public_typed_fia_flat_storage_all_thirty_including_unproduced_first(self):
        self.make(fia=True, count=30)
        self.complete(30, 29)
        first = self.provider.backend.candidate_inventory[0]
        self.assertEqual(first['status'], 'UNSUPPORTED')
        for outcome in self.provider.backend.outcomes:
            snap = outcome['fia_typed_consumer_snapshot']
            self.assertEqual(snap['before']['original']['format'], 47)
            self.assertEqual(snap['before']['original']['storage_rank'], 1)
            self.assertEqual(snap['before_execute']['effective']['format'], 47)
            self.assertTrue(outcome['released_receipt']['released'])

    def complete(self, total, resolved):
        self.start()
        with patch.object(self.worker, 'demoncat_worker_identity', side_effect=AssertionError('recursive RPC')):
            for i in range(8):
                self.enqueue(i)
            state = self.ok(self.client.finish())
        evidence = state['evidence']
        self.assertEqual((self.model.root_calls, len(evidence['request_index'])), (8, 8))
        self.assertEqual(len(evidence['candidate_index']), total)
        self.assertEqual(len(self.provider.backend.targets), resolved)
        self.assertTrue(all(row['native_count'] == total for row in evidence['request_index']))
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference','lossless','off','fault','off'])
        self.assertTrue(evidence['single_site_c1']['evidence_complete'])
        self.assertFalse(evidence['c2'] or evidence['c3'] or evidence['public_core']['qualification_granted'])
        self.assertEqual(evidence['public_core']['owner'], self.provider._ticket[0])
        self.assertEqual((state['leases'], state['inflight'], state['serving']['owned_hooks']), (0,0,0))
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            output = self.forward(self.provider.session)
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(self.model.root_calls, 9)

    def test_query_never_collects_or_runs_model_and_arm_requires_explicit_collect(self):
        self.make()
        self.start()
        self.enqueue(0, collect=False)
        before = deepcopy(self.native.state())
        with patch.object(self.provider.session, 'collect', side_effect=AssertionError('query collected')):
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        self.assertEqual(self.native.state(), before)
        self.assertEqual(self.model.root_calls, 1)
        self.assertTrue(self.provider.session.needs_collect)
        self.ok(self.client.collect_replay())
        self.assertFalse(self.provider.session.needs_collect)
        self.ok(self.client.advance('armed-not-run'))
        self.assertEqual(self.client.finish()['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(self.model.root_calls, 1)

    def test_explicit_collection_captures_full_original_once_query_never_saves(self):
        self.make()
        self.start()
        self.enqueue(0, collect=False)
        before = self.client.query()['response']['provider_state']['evidence']['original_observations']
        self.assertEqual(before['status'], 'NOT_COLLECTED')
        state = self.ok(self.client.collect_replay())
        evidence = state['evidence']['original_observations']
        self.assertEqual(evidence['status'], 'CURRENT_COLLECTED')
        record, = evidence['originals']
        path = Path(record['path'])
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
        original = json.loads(data)
        self.assertEqual(len(original['inventory']['rows'][0]['candidates']), 2)
        self.assertEqual(len(original['serving_requests']), 1)
        with patch.object(self.provider.session, 'capture_evidence', side_effect=AssertionError('query saved')):
            self.assertEqual(self.client.query()['status'], 'QUERY_OK')
        path.write_bytes(data+b' ')
        try:
            self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
            self.assertTrue(self.client.snapshot()['ownership_retained'])
        finally:
            path.write_bytes(data)
        self.ok(self.client.unregister())

    def test_changed_live_model_is_refused_and_owner_is_kept_for_repair(self):
        self.make()
        self.start()
        self.actual_request_id = 'changed-model'
        self.ok(self.client.advance(self.actual_request_id))
        self.runner.model = ActualWriters(self.native, 2).eval()
        try:
            with self.assertRaisesRegex(RuntimeError, 'WORKER|MODEL'):
                self.root(self.provider.session)
            self.assertEqual(self.model.root_calls, 0)
            self.assertTrue(self.client.snapshot()['ownership_retained'] and self.native.installed)
        finally:
            self.runner.model = self.model
        self.ok(self.client.unregister())

    def test_foreign_last_return_hook_is_refused_before_any_forward(self):
        self.make()
        self.start()
        handle = self.model.processor.register_forward_hook(lambda module, args, output: output+1)
        try:
            with self.assertRaisesRegex(RuntimeError, 'FINAL_BOUNDARY'):
                self.enqueue(0)
            self.assertEqual(self.model.root_calls, 0)
            self.assertTrue(self.client.snapshot()['ownership_retained'])
        finally:
            handle.remove()
        self.ok(self.client.unregister())

    def test_typed_exception_after_prepare_and_failed_close_keeps_owner_until_explicit_retry(self):
        self.make(fia=True, count=3)
        self.start()
        self.enqueue(0)
        self.enqueue(1)
        self.native.after_prepare_failure = True
        with self.assertRaises((RuntimeError, AssertionError)):
            self.enqueue(2)
        failure = self.provider.backend.failure
        with patch.object(self.native, 'close', side_effect=RuntimeError('EXPLICIT_TYPED_CLOSE_FAILED')):
            failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertTrue(self.provider.binding.leases)
        self.native.after_prepare_failure = False
        state = self.ok(self.client.unregister())
        self.assertEqual(state['evidence']['first_failure'], failure)
        self.assertFalse(state['evidence']['single_site_c1']['evidence_complete'])

    def test_advance_without_collect_requires_recovery_and_cannot_repeat_fault(self):
        self.make()
        self.start()
        self.enqueue(0, collect=False)
        self.assertEqual(self.client.advance('not-yet')['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(self.client.collect_replay()['status'], 'HOST_REJECTED')
        self.assertEqual(self.model.root_calls, 1)
        self.ok(self.client.unregister())

    def test_actual_root_without_logits_stays_inflight_and_cannot_be_cleaned_early(self):
        self.make()
        self.start()
        self.actual_request_id = 'actual-partial'
        self.ok(self.client.advance(self.actual_request_id))
        with torch.inference_mode():
            hidden = self.root(self.provider.session)
        state = self.client.query()['response']['provider_state']
        self.assertEqual(state['inflight'], 1)
        failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'] and self.native.installed)
        # Only the actual remaining serving boundary can finish this request.
        with torch.inference_mode():
            self.model.processor(self.model.head, hidden[-1:].contiguous())
        self.ok(self.client.unregister())
        self.assertEqual(self.model.root_calls, 1)

    def test_collection_file_collision_preserves_original_owner_and_explicit_cleanup_uses_new_file(self):
        self.make()
        self.start()
        self.enqueue(0, collect=False)
        folder = self.provider.session.folder/'core-evidence'
        folder.mkdir()
        occupied = folder/'000-collect.json'
        occupied.write_bytes(b'prior failed capture remains unchanged')
        failed = self.client.collect_replay()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertEqual(self.client.advance('no-retry')['status'], 'HOST_REJECTED')
        state = self.ok(self.client.unregister())
        self.assertFalse(state['evidence']['single_site_c1']['evidence_complete'])
        self.assertEqual(occupied.read_bytes(), b'prior failed capture remains unchanged')
        self.assertEqual(len(state['evidence']['original_observations']['originals']), 1)

    def test_actual_rpc_foreign_owner_or_generation_never_enters_provider_or_model(self):
        self.make()
        self.start()
        ticket = self.client.snapshot()['ticket']
        for changed in ({'owner':'foreign'}, {'generation':ticket['generation']+1}):
            reply = self.worker.demoncat_fault_rpc('advance',
                dict(ticket, **changed, settings={'request_id':'foreign-request'}))
            self.assertEqual(reply['status'], 'REJECTED')
        self.assertEqual(self.model.root_calls, 0)
        self.assertFalse(self.native.executes or self.native.passes)
        self.ok(self.client.unregister())

    def test_missing_deployment_fails_before_runtime_owner(self):
        self.make()
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(26, 'residual_stomp'))
        self.assertEqual(client.prepare()['status'], 'PROVIDER_UNAVAILABLE')
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(self.native.installed or self.model.root_calls)

    def test_qmm_binding_cannot_be_presented_as_typed_fia_before_owner_creation(self):
        self.make()
        self.client.host._binding(self.worker.demoncat_worker_identity()['worker'], self.client.config)
        binding = getattr(self.worker, core.ATTRIBUTE)[self.mechanism]
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(26, 'fia_destination'))
        with patch('fault_injection.providers.resolve_runtime', return_value=binding):
            reply = client.prepare()
        self.assertEqual(reply['status'], 'PROVIDER_UNAVAILABLE')
        self.assertEqual(reply['response']['provider_code'], 'WRITER26_CORE_MECHANISM_MISMATCH')
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertFalse(self.native.installed or self.model.root_calls)
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_valid_binding_bad_prepare_settings_zero_resources_can_unregister(self):
        self.make()
        original = self.client.host._binding
        def invalid(worker, config):
            original(worker, config)
            return {'prepare': {'layer_name': 'forbidden'}}
        self.client.host._binding = invalid
        self.assertEqual(self.client.prepare()['status'], 'PROVIDER_ACTION_FAILED')
        self.provider = self.worker._demoncat_worker_fault_runtime._leases[self.mechanism].provider.core
        self.assertFalse(self.provider.reserved or self.provider.binding.leases)
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')

    def test_clean_keeps_reservation_until_unregister_foreign_missing_ticket_retains_owner(self):
        self.make()
        self.start()
        state = self.ok(self.client.clean())
        self.assertFalse(state['registered'])
        self.assertTrue(state['binding_reserved'])
        binding, ticket = self.provider.binding, self.provider._ticket
        for changed in (set(), {(ticket[0], ticket[1]+1)}, {ticket, ('foreign', 1)}):
            binding.leases.clear()
            binding.leases.update(changed)
            failed = self.client.unregister()
            self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(failed['ownership_retained'])
            self.assertEqual(binding.leases, changed)
            binding.leases.clear()
            binding.leases.add(ticket)
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')

    def test_native_generation_change_refuses_readback_and_cleanup_without_writes(self):
        self.make()
        self.start()
        self.native.config.generation += 1
        try:
            self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
            failed = self.client.unregister()
            self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
            self.assertTrue(failed['ownership_retained'] and self.native.installed)
        finally:
            self.native.config.generation -= 1
        self.ok(self.client.unregister())

    def test_clean_hook_after_actual_removal_failure_retains_owner_and_retry_succeeds(self):
        self.make()
        self.start()
        handle = self.provider.session.handles[-1]
        original = handle.remove
        def after_remove():
            original()
            raise RuntimeError('EXPLICIT_AFTER_REMOVE')
        with patch.object(handle, 'remove', after_remove):
            failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.ok(self.client.unregister())

    def test_after_actual_native_remove_error_keeps_rpc_owner_and_retry_skips_released_native(self):
        self.make()
        self.start()
        original = self.native.close
        def after_close():
            original()
            raise RuntimeError('EXPLICIT_AFTER_NATIVE_REMOVE')
        with patch.object(self.native, 'close', after_close):
            failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertFalse(self.native.installed)
        with patch.object(self.native, 'close', side_effect=AssertionError('closed a released Native owner')):
            self.ok(self.client.unregister())

    def test_late_native_callback_does_not_passthrough_or_release(self):
        self.make()
        self.start()
        backend = self.provider.backend
        self.ok(self.client.clean())
        before = (len(self.native.passes), len(self.native.executes))
        self.assertEqual(backend._callback(None), 1)
        self.assertEqual(backend._completed(None), 1)
        self.assertEqual(before, (len(self.native.passes), len(self.native.executes)))
        self.assertIn('LATE_NATIVE_CALLBACK', backend.failure)
        self.ok(self.client.unregister())

    def test_after_prepare_failure_preserves_first_error_pending_owner_and_explicit_retry(self):
        self.make()
        self.start()
        self.enqueue(0)
        self.enqueue(1)
        self.native.after_prepare_failure = True
        self.native.inject_failure = True
        with self.assertRaises(RuntimeError):
            self.enqueue(2)
        failure = self.provider.backend.failure
        self.assertIsNotNone(self.native.transaction)
        self.native.drain_fails = True
        failed = self.client.unregister()
        self.assertEqual(failed['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(failed['ownership_retained'])
        self.assertEqual(self.provider.backend.failure, failure)
        self.native.drain_fails = self.native.after_prepare_failure = False
        state = self.ok(self.client.unregister())
        self.assertEqual(state['evidence']['failure'], failure)
        self.assertFalse(state['evidence']['single_site_c1']['evidence_complete'])


if __name__ == '__main__':
    unittest.main()
