"""Real Core/Torch isolated requests; explicit CPU Native doubles grant no C2."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
from unittest.mock import patch
import unittest

import torch

from demoncat.runtime.vllm_ascend.worker_extension import serving26_core as core
import test_core_writer26 as c1
import test_core_writer26_coverage as c2


class CoreWriter26IsolatedTests(unittest.TestCase):
    setUp = c2.CoreWriter26CoverageTests.setUp
    dispose = c1.CoreWriter26Tests.dispose
    ok = c2.CoreWriter26CoverageTests.ok
    start = c1.CoreWriter26Tests.start
    enqueue = c1.CoreWriter26Tests.enqueue
    snapshot = c1.CoreWriter26Tests.snapshot
    root = c1.CoreWriter26Tests.root
    forward = c1.CoreWriter26Tests.forward

    def make(self, *, fia=False, count=3, max_targets=128):
        install = core.install_writer26_core_binding
        def bind(*args, **kwargs):
            return install(*args, stage=core.ISOLATED_STAGE, max_targets=max_targets, **kwargs)
        self.enterContext(patch.object(core, 'install_writer26_core_binding', bind))
        c1.CoreWriter26Tests.make(self, fia=fia, count=count)
        def seal(targets):
            self.assertFalse(self.native.sealed)
            self.assertIsNone(self.native.transaction)
            self.assertCountEqual([(mode, key) for _, mode, key in self.native.executes],
                [(mode, key) for mode in ('reference', 'lossless', 'off') for key in targets])
            self.native.sealed = True
        self.enterContext(patch.object(self.native, 'seal', seal))

    def complete(self, total, eligible):
        self.start()
        for index in range(6+2*eligible):
            self.enqueue(index)
        state = self.ok(self.client.finish())
        backend = self.provider.backend
        report = backend.report()
        coverage = report['coverage']
        self.assertTrue(coverage['contract_satisfied'], coverage['rejection_reasons'])
        self.assertEqual((coverage['planned_targets'], coverage['hits']), (eligible, eligible))
        self.assertEqual(coverage['observed_calls'], total*eligible*2)
        self.assertEqual(len(report['candidates']), total)
        self.assertTrue(all(len(row['candidates']) == total for row in report['rows']))
        self.assertEqual(len(report['isolated_pairs']), eligible)
        self.assertEqual({pair['target_key'] for pair in report['isolated_pairs']}, set(backend.targets))
        self.assertEqual(len({pair['fault_request_id'] for pair in report['isolated_pairs']}), eligible)
        self.assertEqual(len({pair['off_request_id'] for pair in report['isolated_pairs']}), eligible)
        for index, pair in enumerate(report['isolated_pairs']):
            self.assertEqual(pair['target_key'], report['isolated_plan']['target_order'][index])
            fault, off = report['rows'][5+2*index:7+2*index]
            self.assertEqual((fault['admissions'], off['admissions']), (1, 1))
            self.assertNotEqual(fault['request_id'], off['request_id'])
            self.assertEqual(off['output_sha256'], report['rows'][0]['output_sha256'])
            before = pair['pre_fault_baseline']
            self.assertEqual(before['actual_residual_before_sha256'], before['qualified_reference_residual_sha256'])
            self.assertEqual(before['earlier_faults_this_request'], 0)
        self.assertFalse(report['c2'] or report['c3'])
        self.assertEqual(len(coverage['live_c2_proof']['issued']), 1)
        self.assertTrue(report['independent_per_target_replay'])
        self.assertEqual((state['leases'], state['inflight'], state['serving']['owned_hooks']), (0, 0, 0))
        self.assertEqual(len(self.native.executes), 5*eligible)
        self.ok(self.client.unregister())
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        with torch.inference_mode():
            self.forward(self.provider.session)
        self.assertEqual(self.model.root_calls, 7+2*eligible)
        return report

    def test_qmm_each_actual_target_owns_separate_f_off_requests_and_complete_denominator(self):
        self.make()
        self.complete(3, 3)

    def test_typed_fia_every_target_separate_f_off_preserves_unproduced_call(self):
        self.make(fia=True, count=4)
        report = self.complete(4, 3)
        self.assertEqual(report['candidates'][0]['status'], 'UNSUPPORTED')
        self.assertEqual(report['event_capacity_plan']['additional_owned_execute_records'], 15)

    def test_typed_fia_complete_thirty_calls_twenty_nine_targets_sixty_four_requests(self):
        self.make(fia=True, count=30)
        report = self.complete(30, 29)
        self.assertEqual(report['coverage']['observed_calls'], 1740)
        self.assertEqual(report['isolated_plan']['group_requests'], 64)

    def test_native_is_already_recovered_while_logical_first_decision_prevents_second_f(self):
        self.make()
        self.start()
        for index in range(5):
            self.enqueue(index)
        seen = []
        def inspect(module, args):
            backend = self.provider.backend
            seen.append(dict(native=deepcopy(self.native.state()), active=backend.isolated_active,
                faults=len([row for row in self.native.executes if row[1] == 'fault'])))
        handle = self.model.writers[1].register_forward_pre_hook(inspect)
        try:
            self.enqueue(5)
        finally:
            handle.remove()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['faults'], 1)
        self.assertEqual(seen[0]['native']['core']['pending'], 0)
        self.assertIsNotNone(seen[0]['active']['execution_sha256'])
        self.assertIsNone(self.provider.backend.isolated_active)
        for index in range(6, 12):
            self.enqueue(index)
        self.ok(self.client.finish())
        self.ok(self.client.unregister())

    def test_each_delta_original_reconstructs_full_final_report_query_does_not_write(self):
        self.make(fia=True, count=4)
        self.start()
        for index in range(12):
            self.enqueue(index)
        before = list(self.provider.session.evidence_files)
        with patch.object(self.provider.session, 'capture_evidence', side_effect=AssertionError('query wrote')):
            state = self.client.query()['response']['provider_state']
        self.assertEqual(before, self.provider.session.evidence_files)
        self.assertTrue(state['evidence']['compact_rpc_projection'])
        self.assertEqual(state['native'], self.native.state())
        self.assertEqual(state['evidence']['native_state'], self.native.state())
        self.ok(self.client.finish())
        documents = []
        previous = None
        for record in self.provider.session.evidence_files:
            data = Path(record['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])
            doc = json.loads(data)
            self.assertEqual(doc['previous_original_sha256'], previous)
            previous = record['sha256']
            documents.append(doc)
        final = documents[-1]
        self.assertEqual(final['observation_kind'], 'FINAL_COMPLETE_REPORT')
        delta = documents[:-1]
        self.assertTrue(all(len(doc['inventory']['rows']) == 1 for doc in delta))
        self.assertEqual([row for doc in delta for row in doc['inventory']['rows']], final['inventory']['rows'])
        self.assertEqual([row for doc in delta for row in doc['inventory']['outcomes']], final['inventory']['outcomes'])
        calls = [row for doc in delta for row in (doc['inventory']['coverage'] or {}).get('calls', [])]
        self.assertEqual(calls, final['inventory']['coverage']['calls'])
        first = Path(before[0]['path'])
        original = first.read_bytes()
        first.write_bytes(original+b' ')
        try:
            self.assertEqual(self.client.query()['status'], 'NATIVE_STATE_UNAVAILABLE')
        finally:
            first.write_bytes(original)
        self.ok(self.client.unregister())

    def test_final_health_failure_keeps_owner_and_cleanup_retry_without_more_forward(self):
        self.make()
        self.start()
        for index in range(12):
            self.enqueue(index)
        coverage = self.provider.backend.coverage
        def failure(*args):
            raise RuntimeError('ISOLATED_HEALTH_READBACK_FAILURE')
        with patch.object(coverage, 'recovery', replace(coverage.recovery, health=failure)):
            reply = self.client.finish()
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(reply['ownership_retained'])
        self.assertIsNone(self.provider.backend.proof)
        state = self.ok(self.client.unregister())
        self.assertTrue(state['group_recovery']['complete'])
        self.assertEqual(self.model.root_calls, 12)
        self.assertEqual(len(state['cleanup_failures']), 1)

    def test_foreign_owner_and_generation_never_run_model(self):
        c1.CoreWriter26Tests.test_actual_rpc_foreign_owner_or_generation_never_enters_provider_or_model(self)

    def test_late_native_callback_preserves_first_failure_without_another_original(self):
        c1.CoreWriter26Tests.test_late_native_callback_does_not_passthrough_or_release(self)

    def test_isolated_plan_overflow_keeps_complete_target_set_no_prefix(self):
        self.make(fia=True, count=4)
        self.start()
        self.enqueue(0)
        original = self.reader.state
        def state():
            value = original()
            value['capacity'] = value['stored_events']+1
            return value
        with patch.object(self.reader, 'state', state):
            self.enqueue(1)
        self.assertEqual(len(self.provider.backend.candidate_inventory), 4)
        self.assertEqual(len(self.provider.backend.targets), 3)
        self.assertTrue(self.provider.backend.blockers)
        self.assertFalse(self.native.executes)
        self.ok(self.client.unregister())

    def fatal_child(self, scenario):
        # A failed experiment must retain its logical owner because actual OFF
        # is absent. Use a disposable CPU process instead of inventing recovery
        # receipts or clearing production owner registries in test teardown.
        value = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--fatal', scenario],
            text=True, capture_output=True, timeout=90)
        self.assertEqual(value.returncode, 0, value.stdout+'\n'+value.stderr)
        self.assertIn('CPU_EXPECTED_RETAINED_OWNER', value.stdout)

    def test_actual_root_failure_after_consumer_finishes_logical_failure_and_retains_owner(self):
        self.fatal_child('root')

    def test_actual_logits_failure_after_consumer_finishes_logical_failure_and_retains_owner(self):
        self.fatal_child('logits')

    def test_missing_actual_native_call_blocks_full_denominator_and_preserves_owner(self):
        self.fatal_child('missing_call')

    def test_changed_actual_pre_fault_residual_refuses_native_f_and_keeps_owner(self):
        self.fatal_child('residual_changed')


def fatal_scenario(scenario):
    case = CoreWriter26IsolatedTests()
    case.setUp()
    case.make()
    case.start()
    for index in range(5):
        case.enqueue(index)
    if scenario == 'root':
        selected = patch.object(case.model.writers[1], 'forward', side_effect=RuntimeError('ACTUAL_ROOT_FAILED_AFTER_CONSUMER'))
    elif scenario == 'logits':
        selected = patch.object(case.model.processor, 'raise_forward', True)
    elif scenario == 'missing_call':
        selected = patch.object(case.model.writers[2], 'forward', side_effect=lambda value: value)
    elif scenario == 'residual_changed':
        original = case.provider.backend.original_producer
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            value[-1].add_(1)
            return value
        selected = patch.object(case.provider.backend, 'original_producer', changed)
    else:
        raise AssertionError(scenario)
    with selected, case.assertRaises(RuntimeError):
        case.enqueue(5)
    backend = case.provider.backend
    case.assertIsNone(backend.isolated_active)
    case.assertEqual(backend.coverage.recovery_report()['inflight'], 0)
    case.assertEqual(case.native.state()['core']['pending'], 0)
    case.assertEqual(len([row for row in case.native.executes if row[1] == 'fault']), 0 if scenario == 'residual_changed' else 1)
    if scenario in ('root', 'logits'):
        receipts = [row['receipt'] for row in backend.coverage.report()['calls'] if row.get('receipt')]
        case.assertEqual(receipts[0]['outcome'], 'FAILED')
        case.assertTrue(all(row['outcome'] == 'PASSTHROUGH' and row['original_calls'] == 1 for row in receipts[1:]))
    first = backend.failure
    if scenario == 'root':
        case.assertIn('ACTUAL_ROOT_FAILED_AFTER_CONSUMER', first)
    elif scenario == 'logits':
        case.assertIn('CPU_PROCESSOR_FORWARD_FAILED', first)
    elif scenario == 'residual_changed':
        case.assertIn('ISOLATED_FIRST_F_MUST_START_FROM_ACTUAL_HEALTHY_RESIDUAL', first)
    for action in ('recover', 'unregister'):
        reply = getattr(case.client, action)()
        case.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        case.assertTrue(reply['ownership_retained'])
        case.assertEqual(backend.failure, first)
    case.assertIsNone(backend.proof)
    case.assertTrue(backend.coverage.recovery_report()['held_leases'])
    case.assertEqual(case.native.state()['core']['pending'], 0)
    case.assertFalse(case.native.installed or backend.handles or case.provider.session.handles)
    case.assertTrue(case.provider.binding.leases)
    case.assertTrue(case.worker._demoncat_worker_fault_runtime._leases)
    print('CPU_EXPECTED_RETAINED_OWNER:'+scenario)
    # Dispose only this child's temporary evidence directory. Its intentionally
    # retained logical reservations are not converted into recovery receipts.
    shutil.rmtree(str(case.evidence_root.parent))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fatal':
        fatal_scenario(sys.argv[2])
    else:
        unittest.main()
