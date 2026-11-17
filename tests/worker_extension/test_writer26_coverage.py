"""Complete-set CPU coverage contracts; all Native memory/execution is mocked."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import uuid

import torch

from fault_injection.deployments.writer26_coverage import Writer26Coverage
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from test_writer26_group_experiment import CpuV2
from test_writer26_inventory import CpuReader, CpuWriterModel
from metadata_fixture import metadata_report


class CpuCoverageNative(CpuV2):
    def __init__(self, reader):
        super().__init__(reader)
        self.owner = 'cpu-writer-' + uuid.uuid4().hex
        self.actions = []

    def stop(self):
        self.actions.append('stop')

    def drain(self):
        self.actions.append('drain')
        assert self.transaction is None

    def close(self):
        self.actions.append('release')
        super().close()


class CompleteWriterCoverageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.reader = CpuReader()
        self.native = CpuCoverageNative(self.reader)
        self.model = CpuWriterModel(self.native).eval()
        self.worker = dict(rank=0, logical_device=0, worker_start_identity=self.native.owner,
                           model_instance_id=self.native.owner + ':model')
        self.enterContext(patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', self.native.producer, create=True))
        self.enterContext(patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda t: 2)}))

    def make(self, *, trigger='once', proof=None, budget=8):
        if proof:
            self.native.config.generation = proof.generation + 1
        experiment = Writer26Coverage(self.model, self.native, self.reader, worker_identity=self.worker,
            folder=self.root / ('experiment-' + uuid.uuid4().hex), max_targets=budget,
            trigger=trigger, c2_proof=proof, evidence_kind='cpu_contract')
        def allocation(call, row):
            row['allocation'] = dict(residual_block=[call.residual, call.output],
                output_block=[call.output, call.output + call.output_bytes], shift=32)
        self.enterContext(patch.object(experiment, '_allocation', allocation))
        return experiment

    def execute(self, experiment):
        experiment.install()
        with torch.inference_mode():
            for index in range(40):
                if experiment.phase == 'complete':
                    break
                self.request(experiment, index)
        self.assertEqual(experiment.phase, 'complete')
        experiment.close()
        experiment.finalize()
        return experiment.report()

    def request(self, experiment, index):
        experiment.begin('generation-%d-request-%d' % (self.native.config.generation, index), 'e' * 64)
        output = self.model(torch.ones(4, 16, dtype=torch.bfloat16))
        _, sha, finite = complete_tensor(output)
        experiment.end(sha, model_output_finite=finite)

    def qualify(self, experiment):
        experiment.install()
        with torch.inference_mode():
            for index in range(5):
                self.request(experiment, index)
        self.assertEqual(experiment.phase, 'group_fault')
        self.assertFalse(self.native.sealed)

    def abort(self, experiment, error):
        experiment.fail(str(error))
        experiment.close()
        self.assertFalse(experiment.installed or experiment.handles or experiment.output_refs or experiment.retained)
        self.assertFalse(self.native.installed or self.native.transaction)

    def test_complete_set_once_and_live_scope_every_match(self):
        first = self.make()
        report = self.execute(first)
        self.assertTrue(report['coverage']['contract_satisfied'])
        self.assertEqual((report['coverage']['planned_targets'], report['coverage']['hits']), (2, 2))
        self.assertFalse(report['c2'])  # CPU is not public Native qualification.
        self.assertEqual(self.native.actions[:5], ['stop', 'stop', 'drain', 'drain', 'release'])
        proof = first.proof
        self.assertEqual(proof.owner, self.native.owner)
        self.assertEqual([row['phase'] for row in first.rows], ['inventory_1', 'inventory_2',
            'qualification_reference_all', 'qualification_lossless_all', 'qualification_off_all',
            'group_fault', 'group_off', 'health'])
        second = self.make(trigger='every_match', proof=proof)
        report = self.execute(second)
        self.assertTrue(report['coverage']['contract_satisfied'])
        self.assertEqual((report['coverage']['observed_eligible_calls'], report['coverage']['hits']), (4, 4))
        self.assertEqual(report['coverage']['generation'], proof.generation + 1)
        self.assertFalse(report['c3'])

    def test_fia_complete_boundary_identity_and_group_recovery(self):
        self.native.mechanism, self.native.fia, self.native.config.mechanism = '26.fia_destination', True, 2
        report = self.execute(self.make())
        self.assertTrue(report['coverage']['contract_satisfied'])
        self.assertEqual(report['coverage']['hits'], 2)
        self.assertTrue(all(row['worker'][4] == 'npu' for row in report['coverage']['calls']))

    def test_no_metadata_array_activity_prevents_group_qualification(self):
        self.native.metadata_arrays = lambda: metadata_report(0, self.native.build_identity)
        experiment = self.make()
        with self.assertRaisesRegex(RuntimeError, 'WRITER_COVERAGE_OR_GROUP_RECOVERY_INCOMPLETE'):
            self.execute(experiment)
        self.assertIsNone(experiment.proof)
        self.assertFalse(experiment.report()['coverage']['contract_satisfied'])

    def test_unresolved_denominator_cannot_seal_subset(self):
        self.native.skip_callback = True
        experiment = self.make()
        report = self.execute(experiment)
        self.assertIsNone(report['coverage'])
        self.assertEqual(len(report['candidates']), 2)
        self.assertTrue(all(row['status'] == 'UNVERIFIED' for row in report['candidates']))
        self.assertFalse(self.native.executes)

    def test_budget_rejects_entire_complete_set(self):
        report = self.execute(self.make(budget=1))
        self.assertIsNone(report['coverage'])
        self.assertEqual(report['resolved_candidate_count'], 2)
        self.assertFalse(self.native.executes)

    def test_every_match_requires_same_live_issued_scope(self):
        with self.assertRaisesRegex(RuntimeError, 'MATCHING_LIVE_C2'):
            self.make(trigger='every_match')

    def test_cloned_and_relabelled_proof_rejected_at_real_group_build_and_hooks_released(self):
        first = self.make()
        self.execute(first)
        for proof in (replace(first.proof), replace(first.proof, owner='relabelled-owner')):
            experiment = self.make(trigger='every_match', proof=proof)
            self.qualify(experiment)
            with self.assertRaisesRegex(RuntimeError, 'MATCHING_C2_PROOF_REQUIRED') as raised:
                experiment.begin('forged-proof-request', 'e' * 64)
            self.assertIsNone(experiment.coverage)
            self.assertFalse(self.native.sealed)
            self.abort(experiment, raised.exception)
            self.assertFalse(experiment.report()['c3'])

    def test_unknown_lifetime_before_anomaly_keeps_denominator_and_blocks_writes(self):
        experiment = self.make()
        self.qualify(experiment)
        original_append = self.reader.append
        def changed_registration(kind, **kwargs):
            if kind == 1 and self.native.writer_calls % 2 == 1:
                kwargs['cache'] += 9000
            return original_append(kind, **kwargs)
        with patch.object(self.reader, 'append', changed_registration), torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE') as raised:
                self.request(experiment, 5)
        report = experiment.coverage.report()
        self.assertEqual(len(experiment.rows[-1]['candidates']), 2)
        self.assertEqual(report['observed_eligible_calls'], 2)
        self.assertEqual(report['hits'], 0)
        self.assertTrue(any(row['decision']['reason'] == 'TARGET_NOT_IN_DISCOVERY' for row in report['calls']))
        self.assertFalse(any(mode == 'fault' for _, mode, _ in self.native.executes))
        self.abort(experiment, raised.exception)
        self.assertFalse(experiment.coverage.report()['contract_satisfied'])

    def test_frozen_eligible_geometry_failure_cannot_become_nontarget(self):
        experiment = self.make()
        self.qualify(experiment)
        def failed_geometry(call, row):
            raise RuntimeError('CURRENT_LIVE_ALLOCATOR_PAIR_UNVERIFIED')
        with patch.object(experiment, '_allocation', failed_geometry), torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_CANDIDATE_WRITER_LIFETIME_OR_EFFECT_INCOMPLETE') as raised:
                self.request(experiment, 5)
        report = experiment.coverage.report()
        self.assertEqual((report['observed_eligible_calls'], report['hits']), (2, 0))
        self.assertTrue(all(row['eligible'] for row in report['calls']))
        self.abort(experiment, raised.exception)

    def test_overlapping_natural_lifetimes_block_whole_group(self):
        experiment = self.make()
        original_freeze = experiment._freeze
        def overlap():
            candidates = experiment.rows[1]['candidates']
            candidates[0]['next_consumer'][0]['sequence'] = candidates[1]['native']['sequence']
            original_freeze()
        with patch.object(experiment, '_freeze', overlap):
            report = self.execute(experiment)
        self.assertEqual(report['resolved_candidate_count'], 2)
        self.assertIn('NATURAL_ELIGIBLE_WRITER_LIFETIMES_OVERLAP', report['blockers'])
        self.assertIsNone(report['coverage'])
        self.assertFalse(self.native.executes)


if __name__ == '__main__':
    unittest.main()
