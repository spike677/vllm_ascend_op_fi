"""Actual CPU public FIA/Torch calls with typed Native doubles; no NPU proof."""
from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.writer26_experiment import Writer26Experiment, SINGLE_SITE_STAGE, SINGLE_SITE_SCOPES
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from demoncat.runtime.vllm_ascend.worker_extension import serving26_group
from test_writer26_inventory import CpuReader, Producer
import test_writer26_fia_typed as typed
import test_writer26_public_return as public
import test_serving_session_lifecycle as lifecycle
from experiments.c2c3_closure_20261117 import providers26_prepare_group_cases as cases


class SingleTypedNative(typed.TypedNative):
    def __init__(self, reader):
        super().__init__(reader)
        self.sealed_targets = []
        self.after_prepare_failure = False
        self.build = {'files': {'libfault26_readonly_trace_abi0.so': reader.library_sha256}}

    def seal(self, targets):
        assert len(targets) == 1 and not self.sealed
        assert [mode for _, mode, _ in self.executes] == ['reference', 'lossless', 'off']
        assert {key for _, _, key in self.executes} == set(targets)
        assert self.transaction is None
        self.sealed_targets.append(tuple(targets))
        self.sealed = True

    def execute(self, *args, **kwargs):
        value = super().execute(*args, **kwargs)
        if self.after_prepare_failure:
            raise RuntimeError('EXPLICIT_CPU_AFTER_TYPED_PREPARE')
        return value


class ActualFiaCalls(torch.nn.Module):
    def __init__(self, native, count=30, background_events=20):
        super().__init__()
        self.native, self.background_events = native, background_events
        self.producer = Producer()
        self.attentions = torch.nn.ModuleList([public.CopiedAttention() for _ in range(count)])
        self.alias_of_attention = self.attentions[-1]
        self.root_calls = 0

    def forward(self, x):
        self.root_calls += 1
        # Explicit fresh-forward Native fixture reset. The first actual FIA
        # call has no producer; it remains in the observed denominator.
        self.native.last, self.native.sequence = None, 0
        for _ in range(self.background_events):
            self.native.reader.append(10)  # Unselected trace traffic, not a FIA target.
        residual = torch.zeros_like(x)
        for attention in self.attentions:
            x = attention(x)
            residual = self.producer(x, residual)
        return residual


class ServingFiaCalls(ActualFiaCalls):
    def __init__(self, native):
        super().__init__(native, count=3)
        self.head = torch.nn.Linear(576, 3, bias=False, dtype=torch.bfloat16)
        self.processor = lifecycle.CpuProcessor()
        with torch.no_grad():
            self.head.weight.fill_(.25)

    def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
        return super().forward(torch.ones(input_ids.numel(), 576, dtype=torch.bfloat16))


class FiaSingleSiteTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.experiment = None

    def make(self, count=3, *, stage=SINGLE_SITE_STAGE, budget=1, typed_warmup=0):
        self.reader = CpuReader()
        self.native = SingleTypedNative(self.reader)
        self.native.fia_typed.rows = [dict(phase=1)] * typed_warmup  # Capacity-only CPU history; no authority.
        self.enterContext(patch.object(public, '_RUNTIME', self.native))
        self.model = ActualFiaCalls(self.native, count).eval()
        self.npu = NS(npu_fused_infer_attention_score=public._PACKET, get_npu_format=lambda _: 2)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.npu}))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        original = typed.descriptor
        def flat(tensor, **kwargs):
            value = original(tensor, **kwargs)
            value['storage_rank'], value['storage'] = 1, [tensor.numel()] + [0]*7
            return value
        self.enterContext(patch.object(typed, 'descriptor', flat))
        self.experiment = Writer26Experiment(self.model, self.native, self.reader,
            worker_identity=dict(rank=0, worker_start_identity='EXPLICIT_CPU_FIA', logical_device=0),
            folder=self.folder/'fia-single', stage=stage, max_targets=budget)
        self.addCleanup(self.cleanup)
        self.experiment.install()
        return self.experiment

    def cleanup(self):
        if self.experiment is not None and self.experiment.installed:
            if self.experiment.current is not None:
                self.experiment.fail('EXPLICIT_CPU_DISPOSAL')
            self.experiment.close()

    def request(self):
        try:
            with torch.inference_mode():
                self.experiment.begin('actual-cpu-fia-'+str(len(self.experiment.rows)), 'e'*64)
                output = self.model(torch.ones(13, 576, dtype=torch.bfloat16))
                _, sha, finite = complete_tensor(output)
                self.experiment.end(sha, model_output_finite=finite)
                return output
        except BaseException as error:
            self.experiment.fail(error)
            raise

    def until(self, phase):
        for _ in range(8):
            if self.experiment.phase == phase:
                return
            self.request()
        self.assertEqual(self.experiment.phase, phase)

    def test_flat_storage_thirty_calls_twenty_nine_resolved_one_unproduced_and_one_actual_fault(self):
        experiment = self.make(30)
        self.until('single_site_reference')
        selection = copy.deepcopy(experiment.single_selection)
        self.assertEqual((len(experiment.candidate_inventory), len(experiment.targets)), (30, 29))
        missing = experiment.candidate_inventory[0]
        self.assertEqual(missing['status'], 'UNSUPPORTED')
        self.assertEqual(missing['reasons'], ['NO_ACTUAL_PRECEDING_RESIDUAL_IN_THIS_FRESH_FORWARD'])
        self.assertNotIn(missing['target_key'], experiment.targets)
        self.assertEqual(selection['target_key'], min(experiment.targets))
        self.assertEqual(len(experiment.fia_bridge.artifacts), 60)
        self.assertEqual(len(experiment.fia_bridge.stock), 30)
        self.assertFalse(experiment.fia_bridge.references or self.native.executes)
        self.assertEqual(selection['event_capacity_plan']['remaining_complete_requests'], 6)
        self.assertEqual(selection['event_capacity_plan']['full_request_typed_events'], 60)
        self.assertEqual(selection['event_capacity_plan']['estimated_typed_events'], 365)
        self.assertEqual(selection['event_capacity_plan']['native_capacity'], 32768)
        self.until('single_site_fault')
        self.assertEqual(self.native.sealed_targets, [(selection['target_key'],)])
        self.assertEqual(experiment.single_qualification['mechanism'], '26.fia_destination')
        self.assertEqual(experiment.single_qualification['scope'], SINGLE_SITE_SCOPES['26.fia_destination'])
        self.until('complete')
        experiment.close()
        result = experiment.report()
        single = result['single_site_c1']
        self.assertEqual(single['selection'], selection)
        self.assertTrue(single['evidence_complete'] and single['typed_fia_resources_released'])
        self.assertTrue(single['whole_request_effect_observed'])
        self.assertEqual(single['complete_native_counts'], [30]*8)
        self.assertEqual(single['actual_fault_hits'], 1)
        self.assertEqual(single['selected_native_admissions'], 5)
        self.assertEqual(single['mechanism'], '26.fia_destination')
        self.assertTrue(single['typed_fia_required'])
        self.assertFalse(result['c1'] or result['c2'] or result['c3'] or single['all_eligible_qualification'])
        self.assertEqual((self.model.root_calls, self.native.writer_calls), (8, 240))
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference','lossless','off','fault','off'])
        self.assertEqual({key for _, _, key in self.native.executes}, {selection['target_key']})
        aliases = [row['path'] for row in result['alias_inventory']
                   if row['object_identity'] == id(self.model.attentions[-1])]
        self.assertEqual(aliases, ['attentions.29', 'alias_of_attention'])
        self.assertEqual(result['fia_typed_stock_bridge']['retained_host_bytes'], 0)
        self.assertEqual(len(result['fia_typed_stock_bridge']['artifacts']), 65)
        for artifact in result['fia_typed_stock_bridge']['artifacts']:
            self.assertEqual(len(Path(artifact['path']).read_bytes()), 14976)
            self.assertTrue(artifact['complete_bytes'])
        for row in experiment.rows:
            self.assertEqual(len(row['candidates']), 30)
            self.assertEqual(sum(record['phase'] == 3 for record in row['fia_typed_records']), row['admissions'])
        for outcome in experiment.outcomes:
            self.assertTrue(outcome['public_consumer_lifetime'])
            self.assertEqual(outcome['residual_before_sha256'], outcome['residual_after_sha256'])
            self.assertEqual(outcome['released_receipt']['released'], 1)
            for phase in ('before', 'after', 'before_execute'):
                snapshot = outcome['fia_typed_consumer_snapshot'][phase]
                self.assertEqual(snapshot['original']['storage_rank'], 1)
                self.assertEqual(snapshot['original']['storage'], [7488] + [0]*7)
                self.assertEqual(snapshot['effective']['format'], 47)
                self.assertEqual(snapshot['effective']['storage'], snapshot['original']['storage'])
            if outcome['mode'] != 'fault':
                self.assertTrue(outcome['fia_public_typed']['stock_equal'])

    def test_existing_all_candidate_effects_capacity_refusal_still_keeps_all_thirty(self):
        experiment = self.make(30, stage='effects', budget=128)
        self.until('complete')
        self.assertEqual((len(experiment.candidate_inventory), len(experiment.targets)), (30, 29))
        self.assertEqual([len(row['candidates']) for row in experiment.rows], [30]*3)
        self.assertIn('COMPLETE_TARGET_PLAN_EXCEEDS_INDEPENDENT_NATIVE_EVENT_CAPACITY', experiment.blockers)
        self.assertFalse(self.native.executes or self.native.sealed_targets)

    def test_single_site_typed_trace_budget_keeps_fixed_capacity_and_all_stock_rows(self):
        experiment = self.make(30, typed_warmup=32300)
        self.until('complete')
        self.assertIn('SINGLE_SITE_PLAN_EXCEEDS_COMPLETE_TYPED_FIA_EVENT_CAPACITY', experiment.blockers)
        self.assertEqual(experiment.single_selection['event_capacity_plan']['typed_capacity'], 32768)
        self.assertEqual(len(experiment.candidate_inventory), 30)
        self.assertEqual(len(experiment.fia_bridge.artifacts), 60)
        self.assertFalse(self.native.executes or self.native.sealed)

    def test_current_selected_flat_descriptor_change_stops_fault_without_fallback(self):
        experiment = self.make()
        self.until('single_site_fault')
        selection = copy.deepcopy(experiment.single_selection)
        index = selection['inventory_index']
        def change(row):
            if (self.native.writer_calls - 1) % 3 == index:
                row['original']['storage'][0] += 1
        self.native.before_mutator = change
        with self.assertRaises(RuntimeError):
            self.request()
        self.assertEqual(experiment.single_selection, selection)
        self.assertIn('FLAT_STORAGE_SPAN_CHANGED', experiment.failure)
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference','lossless','off'])
        self.assertEqual(len(experiment.rows[-1]['candidates']), 3)
        self.assertTrue(experiment.installed)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])
        with self.assertRaisesRegex(RuntimeError, 'FAILED_SINGLE_SITE_CANNOT_RETRY'):
            self.request()

    def test_new_reference_must_match_complete_stock_before_any_lossless_or_fault(self):
        experiment = self.make()
        self.until('single_site_reference')
        selection = copy.deepcopy(experiment.single_selection)
        self.native.reference_change = True
        with self.assertRaises(AssertionError):
            self.request()
        self.assertEqual(experiment.single_selection, selection)
        self.assertIn('STOCK_TO_OWNED_DESCRIPTOR_FULL_BYTES_DIFFER', experiment.failure)
        self.assertTrue(experiment.pending and experiment.pending_lifetimes and self.native.transaction)
        self.assertFalse(experiment.fia_bridge.references or self.native.sealed)
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference'])
        experiment.close()
        self.assertFalse(experiment.pending or self.native.transaction)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])

    def test_exception_after_typed_prepare_keeps_first_failure_and_actual_public_owners(self):
        experiment = self.make()
        self.until('single_site_reference')
        self.native.after_prepare_failure = True
        with self.assertRaises(AssertionError):
            self.request()
        self.assertEqual(experiment.failure, 'EXPLICIT_CPU_AFTER_TYPED_PREPARE')
        self.assertTrue(experiment.installed and experiment.pending and self.native.transaction)
        token, outcome = next(iter(experiment.pending.items()))
        self.assertEqual(outcome['native_call']['token'], token)
        self.assertIn('observed_consumer_receipt', outcome)
        self.assertTrue(experiment.pending_lifetimes[token]['writer'])
        self.assertTrue(experiment.pending_lifetimes[token]['consumer'])
        self.assertEqual([mode for _, mode, _ in self.native.executes], ['reference'])
        with patch.object(self.native, 'restore', side_effect=RuntimeError('EXPLICIT_CPU_RESTORE_BLOCKED')):
            with self.assertRaisesRegex(RuntimeError, 'RESTORE_BLOCKED'):
                experiment.close()
        self.assertTrue(experiment.installed and experiment.pending and self.native.transaction)
        experiment.close()
        self.assertEqual(experiment.failure, 'EXPLICIT_CPU_AFTER_TYPED_PREPARE')
        self.assertFalse(experiment.pending or self.native.transaction or experiment.pending_lifetimes)
        self.assertFalse(experiment.report()['single_site_c1']['evidence_complete'])

    def test_missing_typed_bridge_is_refused_before_install(self):
        self.make()
        self.experiment.close()
        self.native.fia_typed = None
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_FIA_CURRENT_TYPED_STOCK_BRIDGE_REQUIRED'):
            Writer26Experiment(self.model, self.native, self.reader, worker_identity=dict(rank=0),
                folder=self.folder/'untyped', stage=SINGLE_SITE_STAGE)
        self.assertFalse(self.native.installed or self.native.executes)


class FiaSingleCaseTests(unittest.TestCase):
    def make_case(self, profile_value):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        native = folder/'native'; native.mkdir()
        document = dict(schema='demoncat.writer26.group_deployment.v2', actual_npu_calls=0,
            legacy_writers_absent=True, actual_image_id='sha256:'+'b'*64, files={})
        if profile_value is not None:
            document['fia_typed_profile'] = profile_value
        (native/'build.json').write_text(json.dumps(document))
        result = folder/'cases.json'
        argv = ['cases', '--installed', '/installed', '--native', '/native', '--probe', '/probe.py',
            '--source-commit', '8ea79a37', '--mechanisms', '26.fia_destination', '--stages', SINGLE_SITE_STAGE,
            '--max-targets', '1', '--out', str(result)]
        with patch.object(sys, 'argv', argv), patch.object(cases, 'Path', side_effect=lambda value:
                native if str(value) == '/native' else Path(value)), redirect_stderr(io.StringIO()):
            cases.main()
        return json.loads(result.read_text())

    def test_preparer_explicit_typed_fia_case_keeps_smol_and_full_inventory_scope(self):
        from test_writer26_group_build import build
        value = build.typed_profile(dict(fia_typed_abi=typed.ABI),
            dict(status='PASS_CPU_TYPED_FIA_RUNTIME', actual_npu_calls=0, cases=18))
        row, = self.make_case(value)
        self.assertIn('--mechanism 26.fia_destination', row['command'])
        self.assertIn('typed FIA site C1', row['review']['provider'])
        self.assertNotIn('QMM', row['review']['provider'])
        self.assertEqual(row['review']['model'], '/home/fault39-integration-20261116-a1/models-smollm2-135m')
        self.assertEqual(row['review']['resource_budget']['independent_events'], 32768)
        self.assertEqual(row['review']['resource_budget']['planned_single_site_targets'], 1)
        self.assertIn('No old reference reuse', row['review']['fia_stock_gate'])
        self.assertFalse(row['review']['resource_budget']['max_targets_selects_subset'])

    def test_preparer_cannot_make_typed_single_site_case_from_legacy_manifest(self):
        with self.assertRaisesRegex(RuntimeError, 'FIA_TYPED_FRESH_BUILD_PROFILE_REQUIRED'):
            self.make_case(None)


class FiaSingleServingTests(unittest.TestCase):
    snapshot = lifecycle.ServingSessionLifecycleTests.snapshot
    root = lifecycle.ServingSessionLifecycleTests.root
    forward = lifecycle.ServingSessionLifecycleTests.forward

    def setUp(self):
        lifecycle.ServingSessionLifecycleTests.setUp(self)
        self.reader = CpuReader()
        self.native = SingleTypedNative(self.reader)
        self.model = ServingFiaCalls(self.native).eval()
        runner = NS(model=self.model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        self.worker = NS(model_runner=runner, rank=0,
            demoncat_worker_identity=lambda: dict(status='WORKER_IDENTIFIED', quarantined=False, worker=dict(rank=0)))
        self.enterContext(patch.object(public, '_RUNTIME', self.native))
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(
            npu_fused_infer_attention_score=public._PACKET, get_npu_format=lambda _: 2)}))
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        self.enterContext(patch('fault_injection.deployments.writer26_group.NativeWriter26Group', return_value=self.native))
        self.enterContext(patch('fault_injection.deployments.writer26_inventory.NativeWriterInventory', return_value=self.reader))
        self.enterContext(patch.object(serving26_group, 'Path', side_effect=lambda value:
            self.evidence_root if str(value) == '/evidence' else Path(value)))
        original = typed.descriptor
        def flat(tensor, **kwargs):
            value = original(tensor, **kwargs)
            value['storage_rank'], value['storage'] = 1, [tensor.numel()] + [0]*7
            return value
        self.enterContext(patch.object(typed, 'descriptor', flat))

    def bind(self):
        result = serving26_group.install_writer26_group_experiment(self.worker, mechanism='26.fia_destination',
            prompt_token_ids=list(range(13)), evidence_dir=self.evidence_root/'fia', manifest_sha256='a'*64,
            stage=SINGLE_SITE_STAGE, max_targets=1)
        session = self.worker._demoncat_writer26_group_experiment
        self.sessions.append(session)
        self.addCleanup(session.close)
        self.assertEqual(result['mechanism'], '26.fia_destination')
        self.assertEqual(self.model.root_calls, 0)
        return session

    def request(self, session, index):
        self.actual_request_id = 'actual-enqueued-fia-'+str(index)
        before = self.model.root_calls
        session.expect(self.actual_request_id)
        self.assertEqual(self.model.root_calls, before)
        with torch.inference_mode():
            return self.forward(session)

    def test_real_serving_binding_and_eight_actual_requests_keep_fia_identity(self):
        session = self.bind()
        for index in range(8):
            self.request(session, index)
            session.report()
        result = serving26_group.remove_writer26_group_experiment(self.worker)['report']
        self.assertEqual(self.model.root_calls, 8)
        self.assertEqual(result['full_native_writer_counts'], [3]*8)
        self.assertTrue(result['single_site_c1']['evidence_complete'])
        self.assertEqual(result['single_site_c1']['mechanism'], '26.fia_destination')
        self.assertTrue(result['single_site_c1']['typed_fia_resources_released'])
        self.assertEqual(result['single_site_c1']['actual_fault_hits'], 1)
        self.assertFalse(result['c2'] or result['c3'])

    def test_binding_rejects_legacy_fia_before_any_owner_or_forward(self):
        self.native.fia_typed = None
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_SITE_FIA_CURRENT_TYPED_STOCK_BRIDGE_REQUIRED'):
            self.bind()
        self.assertFalse(hasattr(self.worker, '_demoncat_writer26_group_experiment'))
        self.assertFalse(self.native.installed or self.native.executes)
        self.assertEqual(self.model.root_calls, 0)

    def test_after_prepare_exception_and_failed_cleanup_keep_actual_worker_owner_until_retry(self):
        session = self.bind()
        self.request(session, 0)
        self.request(session, 1)
        owner = (self.native.config.owner_hi, self.native.config.owner_lo, self.native.config.generation)
        self.native.after_prepare_failure = True
        with self.assertRaises(AssertionError):
            self.request(session, 2)
        self.assertEqual(session.backend.failure, 'EXPLICIT_CPU_AFTER_TYPED_PREPARE')
        self.assertTrue(session.backend.pending and self.native.transaction)
        with patch.object(self.native, 'restore', side_effect=RuntimeError('EXPLICIT_CPU_RESTORE_BLOCKED')):
            with self.assertRaisesRegex(RuntimeError, 'RESTORE_BLOCKED'):
                serving26_group.remove_writer26_group_experiment(self.worker)
        self.assertIs(self.worker._demoncat_writer26_group_experiment, session)
        self.assertEqual((self.native.config.owner_hi, self.native.config.owner_lo, self.native.config.generation), owner)
        self.assertTrue(session.backend.pending and self.native.transaction)
        result = serving26_group.remove_writer26_group_experiment(self.worker)['report']
        self.assertEqual(result['failure'], 'EXPLICIT_CPU_AFTER_TYPED_PREPARE')
        self.assertFalse(result['single_site_c1']['evidence_complete'])
        self.assertFalse(hasattr(self.worker, '_demoncat_writer26_group_experiment'))
        self.assertFalse(self.native.transaction or session.backend.pending_lifetimes)


if __name__ == '__main__':
    unittest.main()
