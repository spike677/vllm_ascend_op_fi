"""Real CPU model/hooks with explicit Native/scheduler doubles; no NPU proof."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.writer26_experiment import Writer26Experiment, SINGLE_SITE_STAGE, digest
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from demoncat.runtime.vllm_ascend.worker_extension import serving26_group
from test_writer26_inventory import CpuReader, Producer, Writer
from test_writer26_pending_consumer import FailedReceiptNative
import test_serving_session_lifecycle as lifecycle
from experiments.c2c3_closure_20261117 import providers26_group_probe as probe
from experiments.c2c3_closure_20261117 import providers26_prepare_group_cases as cases


class SingleNative(FailedReceiptNative):
    def __init__(self, reader):
        super().__init__(reader)
        self.owner = 'explicit-cpu-native-owner'
        self.sealed_targets = []
        self.build = {'files': {'libfault26_readonly_trace_abi0.so': reader.library_sha256}}

    def seal(self, targets):
        # Model the existing engine's exact qualified-lease set check. This
        # double is not a Native implementation or qualification receipt.
        assert len(targets) == 1 and not self.sealed
        assert [mode for _, mode, _ in self.executes] == ['reference', 'lossless', 'off']
        assert {key for _, _, key in self.executes} == set(targets)
        assert self.transaction is None
        self.sealed_targets.append(tuple(targets))
        self.sealed = True


class ManyWriters(torch.nn.Module):
    def __init__(self, native, count):
        super().__init__()
        self.producer = Producer()
        self.writers = torch.nn.ModuleList([Writer(native) for _ in range(count)])
        self.alias_of_writer = self.writers[-1]
        self.skip = None
        self.repeat_first = False
        self.root_calls = 0

    def forward(self, x):
        self.root_calls += 1
        residual = self.producer(x, torch.zeros_like(x))
        for index, writer in enumerate(self.writers):
            if index != self.skip:
                if self.repeat_first:
                    writer = self.writers[0]
                x = writer(x)
                residual = self.producer(x, residual)
        return residual


class ServingWriters(ManyWriters):
    def __init__(self, native):
        super().__init__(native, 2)
        self.head = torch.nn.Linear(16, 3, bias=False, dtype=torch.bfloat16)
        self.processor = lifecycle.CpuProcessor()
        with torch.no_grad():
            self.head.weight.fill_(.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        return super().forward(torch.ones(input_ids.numel(), 16, dtype=torch.bfloat16))


class SingleSiteTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.experiment = None

    def make(self, count=2, *, budget=8, unresolved=None, warmup_events=0):
        self.reader = CpuReader()
        for _ in range(warmup_events):
            self.reader.append(7)
        self.native = SingleNative(self.reader)
        self.model = ManyWriters(self.native, count).eval()
        self.experiment = Writer26Experiment(self.model, self.native, self.reader,
            worker_identity=dict(rank=0, worker_start_identity='CPU_ONLY', logical_device=0, model_instance_id='CPU_MODEL'),
            folder=self.folder / 'experiment', stage=SINGLE_SITE_STAGE, max_targets=budget)
        # Stable synthetic caches and allocator evidence are explicit CPU
        # fixtures. Actual production uses the independent Native event and
        # torch_npu allocator snapshot for each call, including unselected ones.
        append = self.reader.append
        def event(kind, **kwargs):
            if kind == 1:
                kwargs['cache'] = 1000 + (self.native.writer_calls - 1) % count
            return append(kind, **kwargs)
        def allocation(call, row):
            if unresolved is not None and row['module_object_identity'] == id(self.model.writers[unresolved]):
                raise RuntimeError('EXPLICIT_CPU_UNVERIFIED_ALLOCATOR')
            row['allocation'] = dict(residual_block=[call.residual, call.output],
                output_block=[call.output, call.output + call.output_bytes], shift=32)
        self.enterContext(patch.object(self.reader, 'append', event))
        self.enterContext(patch.object(self.experiment, '_allocation', allocation))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda t: 2)}))
        self.experiment.install()
        self.addCleanup(self.cleanup)
        return self.experiment

    def request(self, *, bad_hash=False):
        experiment = self.experiment
        try:
            with torch.inference_mode():
                experiment.begin('cpu-request-' + str(len(experiment.rows)), 'e' * 64)
                output = self.model(torch.ones(4, 16, dtype=torch.bfloat16))
                _, sha, finite = complete_tensor(output)
                experiment.end('0' * 64 if bad_hash else sha, model_output_finite=finite)
                return output
        except BaseException as error:
            experiment.fail(error)
            raise

    def until(self, phase):
        for _ in range(8):
            if self.experiment.phase == phase:
                return
            self.request()
        self.assertEqual(self.experiment.phase, phase)

    def cleanup(self):
        if self.experiment and self.experiment.installed:
            self.native.drain_fails = False
            if self.experiment.active:
                self.experiment.fail('CPU_FIXTURE_DISPOSAL')
            self.experiment.close()

    def test_full_135_inventory_aliases_and_one_freshly_qualified_site(self):
        experiment = self.make(135, budget=1)
        self.until('single_site_fault')
        key = experiment.single_selection['target_key']
        self.assertEqual(key, min(experiment.targets))
        self.assertEqual(len(experiment.targets), 135)
        self.assertEqual(experiment.single_selection['complete_native_count'], 135)
        self.assertEqual(experiment.single_selection['inventory_sha256'], digest(experiment.candidate_inventory))
        self.assertEqual(self.native.sealed_targets, [(key,)])
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference', 'lossless', 'off'])
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])
        self.until('complete')
        experiment.close()
        report = experiment.report()
        self.assertEqual([row['phase'] for row in experiment.rows], ['inventory_1', 'inventory_2',
            'single_site_reference', 'single_site_lossless', 'single_site_qualification_off',
            'single_site_fault', 'single_site_recovery_off', 'health'])
        self.assertEqual(report['single_site_c1']['complete_native_counts'], [135] * 8)
        self.assertEqual({target for _, _, target in self.native.executes}, {key})
        self.assertEqual(len(self.native.executes), 5)
        self.assertEqual(self.model.root_calls, 8)
        self.assertEqual(self.native.writer_calls, 135 * 8)
        self.assertEqual(len(self.native.passes), 135 * 7 - 5)  # Native uninstalled before final health.
        aliases = [row['path'] for row in report['alias_inventory'] if row['object_identity'] == id(self.model.writers[-1])]
        self.assertEqual(aliases, ['writers.134', 'alias_of_writer'])
        self.assertTrue(report['single_site_c1']['evidence_complete'])
        self.assertEqual(report['single_site_c1']['actual_fault_hits'], 1)
        self.assertTrue(report['single_site_c1']['whole_request_effect_observed'])
        self.assertFalse(report['c1'] or report['c2'] or report['c3'] or report['native_qualified_targets'])
        self.assertFalse(report['single_site_c1']['qualification_granted'])
        for row in experiment.outcomes:
            self.assertTrue(row['public_consumer_lifetime'])
            self.assertEqual(row['residual_before_sha256'], row['residual_after_sha256'])
            self.assertEqual(row['consumer_receipt']['writer_calls'], 1)
            self.assertEqual(row['consumer_receipt']['consumer_calls'], 1)
            self.assertEqual(row['released_receipt']['released'], 1)

    def test_unverified_rows_stay_visible_and_never_become_qualified(self):
        experiment = self.make(4, budget=1, unresolved=0)
        self.until('complete')
        experiment.close()
        self.assertEqual(len(experiment.candidate_inventory), 4)
        self.assertEqual(len(experiment.targets), 3)
        self.assertEqual(experiment.candidate_inventory[0]['status'], 'UNVERIFIED')
        self.assertEqual(experiment.report()['qualified_candidate_count'], 1)
        self.assertTrue(all(len(row['candidates']) == 4 for row in experiment.rows))
        self.assertFalse(experiment.report()['single_site_c1']['all_eligible_qualification'])

    def test_capacity_budget_counts_all_135_calls_not_only_the_selected_site(self):
        experiment = self.make(135, budget=1, warmup_events=28000)
        self.until('complete')
        self.assertIn('COMPLETE_TARGET_PLAN_EXCEEDS_INDEPENDENT_NATIVE_EVENT_CAPACITY', experiment.blockers)
        self.assertEqual(len(experiment.targets), 135)
        self.assertEqual(len(experiment.rows), 3)
        self.assertTrue(all(len(row['candidates']) == 135 for row in experiment.rows))
        self.assertFalse(self.native.executes or self.native.sealed)

    def test_no_resolved_site_keeps_all_rows_without_any_selection_or_fault(self):
        experiment = self.make()
        self.native.skip_callback = True
        self.until('complete')
        experiment.close()
        self.assertEqual(len(experiment.candidate_inventory), 2)
        self.assertIsNone(experiment.single_selection)
        self.assertFalse(self.native.executes or self.native.sealed)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])

    def test_repeated_site_in_first_healthy_request_cannot_qualify_as_unique(self):
        experiment = self.make()
        append = self.reader.append
        def same_cache(kind, **kwargs):
            event = append(kind, **kwargs)
            if kind == 1:
                event.cache = 1000
            return event
        with patch.object(self.reader, 'append', same_cache):
            self.model.repeat_first = True
            self.request()
            self.model.repeat_first = False
            self.request()
            self.until('complete')
        self.assertEqual(len(experiment.candidate_inventory), 2)
        self.assertTrue(all(row['status'] == 'UNVERIFIED' for row in experiment.candidate_inventory))
        self.assertFalse(experiment.targets or self.native.executes)

    def test_lossless_failure_preserves_selection_and_never_falls_back_or_faults(self):
        experiment = self.make()
        self.until('single_site_lossless')
        selection = dict(experiment.single_selection)
        with self.assertRaisesRegex(RuntimeError, 'FULL_VOCAB_LOSSLESS_OR_OFF_CHANGED'):
            self.request(bad_hash=True)
        self.assertEqual(experiment.single_selection, selection)
        self.assertFalse(self.native.sealed)
        with self.assertRaisesRegex(RuntimeError, 'FAILED_SINGLE_SITE_CANNOT_RETRY'):
            self.request()
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference', 'lossless'])

    def test_qualification_off_must_pass_again_before_fault_sealing(self):
        experiment = self.make()
        self.until('single_site_qualification_off')
        with self.assertRaisesRegex(RuntimeError, 'FULL_VOCAB_LOSSLESS_OR_OFF_CHANGED'):
            self.request(bad_hash=True)
        self.assertFalse(self.native.sealed or experiment.single_qualification)
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference', 'lossless', 'off'])

    def test_changed_selected_allocator_pair_refuses_fault_and_keeps_original_call_once(self):
        experiment = self.make()
        self.until('single_site_fault')
        selected = experiment.single_selection['target_key']
        original = experiment._allocation
        def changed(call, row):
            if row['target_key'] == selected:
                raise RuntimeError('LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT')
            original(call, row)
        with patch.object(experiment, '_allocation', changed):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE'):
                self.request()
        self.assertEqual(self.native.writer_calls, 12)
        self.assertEqual(len(self.native.executes), 3)
        self.assertIn('LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT', experiment.failure)
        self.assertEqual(len(experiment.rows[-1]['candidates']), 2)

    def test_changed_model_object_or_alias_refuses_before_native_admission(self):
        experiment = self.make()
        self.until('single_site_fault')
        calls = self.native.writer_calls
        self.model.extra_alias = self.model.writers[0]
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_ACTUAL_MODULE_ALIASES_CHANGED'):
            self.request()
        self.assertEqual(self.native.writer_calls, calls)
        self.assertEqual(len(self.native.executes), 3)

    def test_same_geometry_replaced_selected_module_is_not_the_same_site(self):
        experiment = self.make()
        self.until('single_site_fault')
        selected = experiment.targets[experiment.single_selection['target_key']]['callback']['module_object_identity']
        index = next(i for i, module in enumerate(self.model.writers) if id(module) == selected)
        calls = self.native.writer_calls
        self.model.writers[index] = Writer(self.native)
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_ACTUAL_MODULE_ALIASES_CHANGED'):
            self.request()
        self.assertEqual(self.native.writer_calls, calls)
        self.assertEqual(len(self.native.executes), 3)

    def test_native_cache_identity_change_cannot_select_another_resolved_site(self):
        experiment = self.make()
        self.until('single_site_fault')
        selected = dict(experiment.single_selection)
        append = self.reader.append
        def changed(kind, **kwargs):
            event = append(kind, **kwargs)
            if kind == 1:
                event.cache += 10000
            return event
        with patch.object(self.reader, 'append', changed):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE'):
                self.request()
        self.assertEqual(experiment.single_selection, selected)
        self.assertEqual(len(self.native.executes), 3)
        self.assertEqual(len(experiment.rows[-1]['candidates']), 2)

    def test_current_native_stride_change_is_refused_before_selected_admission(self):
        experiment = self.make()
        self.until('single_site_fault')
        append = self.reader.append
        def changed(kind, **kwargs):
            event = append(kind, **kwargs)
            if kind == 1:
                event.strides[0] += 1
            return event
        with patch.object(self.reader, 'append', changed):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE'):
                self.request()
        self.assertEqual(experiment.failure, 'SINGLE_SITE_CURRENT_OBJECT_OR_NATIVE_FORMAL_CHANGED')
        self.assertEqual(len(self.native.executes), 3)

    def test_full_denominator_loss_after_fault_is_failed_evidence_not_subset_success(self):
        experiment = self.make(4)
        self.until('single_site_fault')
        index = experiment.single_selection['inventory_index']
        self.model.skip = (index + 1) % 4
        with self.assertRaises(RuntimeError):
            self.request()
        self.assertEqual(len(experiment.rows[-1]['candidates']), 3)
        self.assertTrue(experiment.failure)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])
        self.assertFalse(self.native.transaction)

    def test_failed_consumer_drain_retry_keeps_owner_and_first_failure(self):
        experiment = self.make()
        self.until('single_site_fault')
        self.native.inject_failure = True
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request()
        token = next(iter(experiment.pending))
        owned = experiment.pending_lifetimes[token]
        first = experiment.failure
        self.native.drain_fails = True
        with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_CPU_DRAIN_FAILED'):
            experiment.close()
        self.assertIs(experiment.pending_lifetimes[token], owned)
        self.assertEqual(self.native.restore_calls, [])
        self.native.drain_fails = False
        experiment.close()
        self.assertEqual(self.native.restore_calls, [token])
        self.assertEqual(experiment.failure, first)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])

    def test_pending_changed_storage_refuses_recovery_write_then_explicit_repair_allows_cleanup(self):
        experiment = self.make()
        self.until('single_site_reference')
        self.native.inject_failure = True
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request()
        token = next(iter(experiment.pending))
        owned = experiment.pending_lifetimes[token]['residual']
        tensor, storage = owned['tensor'], owned['storage']
        with torch.inference_mode():
            tensor.set_(torch.empty_like(tensor))
        try:
            with self.assertRaisesRegex(RuntimeError, 'PENDING_OFF_POINTER_LIVENESS_UNVERIFIED_DO_NOT_WRITE'):
                experiment.close()
            self.assertEqual(self.native.restore_calls, [])
            self.assertTrue(experiment.installed and experiment.handles)
        finally:
            # CPU fixture disposal only; production never reconstructs liveness.
            with torch.inference_mode():
                tensor.set_(storage, 0, owned['shape'], owned['stride'])
        experiment.close()
        self.assertEqual(self.native.restore_calls, [token])

    def fail_fault(self):
        self.make()
        self.until('single_site_fault')
        self.native.inject_failure = True
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request()
        return next(iter(self.experiment.pending))

    def restore_failure(self, after):
        token = self.fail_fault()
        owned = self.experiment.pending_lifetimes[token]
        restore = self.native.restore
        def fail(value):
            if after:
                restore(value)
            raise RuntimeError('CPU_RESTORE_BOUNDARY_FAILURE')
        with patch.object(self.native, 'restore', fail):
            with self.assertRaisesRegex(RuntimeError, 'CPU_RESTORE_BOUNDARY_FAILURE'):
                self.experiment.close()
        self.assertIs(self.experiment.pending_lifetimes[token], owned)
        self.assertIsNotNone(self.native.transaction)
        self.experiment.close()
        self.assertEqual(self.native.restore_calls, [token] * (2 if after else 1))
        self.assertFalse(self.experiment.report()['single_site_c1']['evidence_complete'])

    def test_before_restore_failure_keeps_owner_for_explicit_cleanup_retry(self):
        self.restore_failure(False)

    def test_after_restore_failure_keeps_owner_and_repeat_restore_is_safe(self):
        self.restore_failure(True)

    def ack_failure(self, after):
        token = self.fail_fault()
        owned = self.experiment.pending_lifetimes[token]
        acknowledge = self.native.acknowledge
        def fail(value):
            if after:
                acknowledge(value)
            raise RuntimeError('CPU_ACK_BOUNDARY_FAILURE')
        with patch.object(self.native, 'acknowledge', fail):
            with self.assertRaisesRegex(RuntimeError, 'CPU_ACK_BOUNDARY_FAILURE'):
                self.experiment.close()
        self.assertIs(self.experiment.pending_lifetimes[token], owned)
        self.experiment.close()
        self.assertEqual(self.native.restore_calls, [token] * (1 if after else 2))
        self.assertFalse(self.experiment.pending_lifetimes or self.native.transaction)
        self.assertFalse(self.experiment.report()['single_site_c1']['evidence_complete'])

    def test_before_ack_failure_restores_and_releases_on_explicit_retry(self):
        self.ack_failure(False)

    def test_after_ack_failure_reads_native_release_without_second_restore_write(self):
        self.ack_failure(True)

    def test_foreign_completion_receipt_never_acknowledges_or_qualifies(self):
        experiment = self.make()
        self.until('single_site_reference')
        finished = self.native.finished
        def foreign(pointer):
            pointer.contents.owner_hi += 1
            self.assertEqual(finished(pointer), 1)
            return 0  # Explicit Native continuation fixture, not a success assertion.
        self.native.finished = foreign
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request()
        outcome = next(iter(experiment.pending.values()))
        self.assertNotIn('released_receipt', outcome)
        self.assertIsNotNone(self.native.transaction)
        self.assertFalse(experiment.single_qualification or self.native.sealed)
        self.assertEqual(experiment.failure, 'EXACT_NATIVE_WRITER_CONSUMER_RECEIPT_REQUIRED')

    def test_fault_hit_without_effect_receipt_is_failed_not_c1_complete(self):
        experiment = self.make()
        self.until('single_site_fault')
        finished = self.native.finished
        def no_effect(pointer):
            pointer.contents.changed_residual_bytes = 0
            self.assertEqual(finished(pointer), 1)
            return 0
        self.native.finished = no_effect
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CONSUMER_DID_NOT_COMPLETE_NATIVE_LEASE'):
            self.request()
        self.assertEqual(experiment.failure, 'ACTUAL_RESIDUAL_EFFECT_NOT_OBSERVED')
        self.assertEqual(len(experiment.pending_lifetimes), 1)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])

    def test_final_same_worker_health_failure_is_not_complete_even_after_native_release(self):
        experiment = self.make()
        self.until('health')
        with self.assertRaisesRegex(RuntimeError, 'SAME_WORKER_FULL_VOCAB_AFTER_NATIVE_RELEASE_CHANGED'):
            self.request(bad_hash=True)
        self.assertTrue(experiment.native_removed)
        experiment.close()
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])


class SingleSiteServingTests(unittest.TestCase):
    snapshot = lifecycle.ServingSessionLifecycleTests.snapshot
    root = lifecycle.ServingSessionLifecycleTests.root
    forward = lifecycle.ServingSessionLifecycleTests.forward

    def setUp(self):
        lifecycle.ServingSessionLifecycleTests.setUp(self)

    def make_session(self):
        reader = CpuReader()
        native = SingleNative(reader)
        model = ServingWriters(native).eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0,
            demoncat_worker_identity=lambda: dict(status='WORKER_IDENTIFIED', worker=dict(rank=0)))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', native.producer, create=True))
        self.enterContext(patch('fault_injection.deployments.writer26_group.NativeWriter26Group', return_value=native))
        self.enterContext(patch('fault_injection.deployments.writer26_inventory.NativeWriterInventory', return_value=reader))
        self.enterContext(patch.object(serving26_group, 'Path', side_effect=lambda value:
            self.evidence_root if str(value) == '/evidence' else Path(value)))
        def allocation(experiment, call, row):
            row['allocation'] = dict(residual_block=[call.residual, call.output],
                output_block=[call.output, call.output + call.output_bytes], shift=32)
        self.enterContext(patch.object(Writer26Experiment, '_allocation', allocation))
        bound = serving26_group.install_writer26_group_experiment(worker, mechanism='26.residual_stomp',
            prompt_token_ids=[1, 2, 3, 4], evidence_dir=self.evidence_root / 'single',
            manifest_sha256='a' * 64, stage=SINGLE_SITE_STAGE, max_targets=1)
        session = worker._demoncat_writer26_group_experiment
        self.sessions.append(session)
        self.addCleanup(session.close)
        self.assertEqual(bound['stage'], SINGLE_SITE_STAGE)
        return worker, session, model

    def test_standard_serving_hooks_advance_without_executing_then_eight_actual_requests(self):
        worker, session, model = self.make_session()
        for index in range(8):
            self.actual_request_id = 'actual-enqueued-cpu-' + str(index)
            session.expect(self.actual_request_id)
            self.assertEqual(model.root_calls, index)  # No RPC-driven replay.
            with torch.inference_mode():
                self.forward(session)
            session.report()
        close = session.backend.close
        def backend_closed_before_root_hooks():
            close()
            evidence = session.backend.report()['single_site_c1']
            self.assertEqual(evidence['owned_root_logits_hooks'], 4)
            self.assertFalse(evidence['evidence_complete'])
        with patch.object(session.backend, 'close', side_effect=backend_closed_before_root_hooks):
            result = serving26_group.remove_writer26_group_experiment(worker)
        self.assertTrue(result['report']['single_site_c1']['evidence_complete'])
        self.assertEqual(result['report']['full_native_writer_counts'], [2] * 8)
        self.assertEqual(len(session.rows), 8)
        self.assertFalse(hasattr(worker, '_demoncat_writer26_group_experiment'))
        self.assertFalse(result['report']['c2'] or result['report']['c3'])

    def test_later_logits_replacement_cannot_qualify_an_earlier_output(self):
        _, session, model = self.make_session()
        hook = model.processor.register_forward_hook(lambda module, args, output: output + 1)
        self.addCleanup(hook.remove)
        self.actual_request_id = 'actual-enqueued-with-late-logits-hook'
        session.expect(self.actual_request_id)
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_FINAL_LOGITS_OBSERVER_REQUIRED'):
            self.forward(session)
        self.assertFalse(session.backend.single_qualification or session.native.executes)
        self.assertTrue(session.backend.failure)
        self.assertFalse(session.backend.report()['single_site_c1']['evidence_complete'])

    def test_logits_hook_installed_during_the_real_call_is_rechecked_at_return(self):
        _, session, model = self.make_session()
        late = []
        def add_late_hook(module, args):
            handle = module.register_forward_hook(lambda module, args, output: output + 1)
            late.append(handle)
            self.addCleanup(handle.remove)
        pre = model.processor.register_forward_pre_hook(add_late_hook)
        self.addCleanup(pre.remove)
        self.actual_request_id = 'actual-enqueued-mid-call-logits-hook'
        session.expect(self.actual_request_id)
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_FINAL_LOGITS_OBSERVER_REQUIRED'):
            self.forward(session)
        self.assertEqual(len(late), 1)
        self.assertFalse(session.backend.single_qualification or session.native.executes)


class SingleSiteEntryTests(unittest.TestCase):
    def test_probe_has_fixed_eight_request_budget_independent_of_legacy_max_targets(self):
        self.assertEqual([probe.request_limit(SINGLE_SITE_STAGE, n) for n in (1, 8, 128)], [8] * 3)
        self.assertEqual(probe.request_limit('effects', 8), 44)
        self.assertEqual(probe.request_limit('coverage', 8), 9)

    def test_probe_keeps_both_mechanisms_explicit_and_same_single_site_request_limit(self):
        from fault_injection.deployments.writer26_experiment import SINGLE_SITE_SCOPES
        for mechanism, scope in SINGLE_SITE_SCOPES.items():
            self.assertEqual(probe.single_site_scope(mechanism), scope)
            self.assertEqual(probe.request_limit(SINGLE_SITE_STAGE, 128, mechanism), 8)
        with self.assertRaisesRegex(ValueError, 'EXACT_SINGLE_SITE_WRITER_MECHANISM_REQUIRED'):
            probe.request_limit(SINGLE_SITE_STAGE, 1, 'unreviewed')

    def test_case_generator_explicit_qmm_mode_and_review_contract(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bundle = folder / 'native'
        bundle.mkdir()
        (bundle / 'build.json').write_text(json.dumps(dict(schema='demoncat.writer26.group_deployment.v2',
            actual_npu_calls=0, legacy_writers_absent=True, actual_image_id='sha256:' + 'b' * 64, files={})))
        output = folder / 'cases.json'
        argv = ['cases', '--installed', '/installed-reviewed', '--native', '/native-reviewed', '--probe', '/probe-reviewed.py',
                '--source-commit', '9db90328', '--stages', SINGLE_SITE_STAGE, '--max-targets', '1', '--out', str(output)]
        with patch.object(sys, 'argv', argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cases.main()  # Default both mechanisms must not silently drop FIA.
        with patch.object(sys, 'argv', argv + ['--mechanisms', '26.residual_stomp']), \
                patch.object(cases, 'Path', side_effect=lambda value: bundle if str(value) == '/native-reviewed' else Path(value)):
            cases.main()
        generated, = json.loads(output.read_text())
        self.assertIn('--stage c1_single_effects', generated['command'])
        self.assertNotIn('--target', generated['command'])
        self.assertEqual(generated['review']['resource_budget']['planned_single_site_targets'], 1)
        self.assertFalse(generated['review']['resource_budget']['max_targets_selects_subset'])
        self.assertIn('never truncate the inventory', generated['review']['recovery_plan'])


if __name__ == '__main__':
    unittest.main()
