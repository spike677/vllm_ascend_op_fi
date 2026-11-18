"""Offline reviewer contracts from real CPU RoPE forwards, never NPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_closure_20261117.review_natural_rope_result import Review, read_result
import test_rope_serving_group as worker_fixture


def cpu_result():
    fixture = worker_fixture.RoPEWorkerGroupTests()
    fixture.setUp()
    session = fixture.session
    requests = []
    def request(phase, arm=None):
        row, logits = fixture.host.request(phase, arm)
        request_id = row['request_id']
        requests.append(dict(phase=phase, worker=row, output_tokens=[int(logits.argmax())],
            mapping=dict(internal_request_id=request_id, external_request_id='external-' + request_id,
                request_state_object_id=len(requests) + 1, source='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR')))
    try:
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            request(phase, arm)
        qualification = session.qualify()
        group = session.provider.open_group('once', forwards=1)
        c2_open = dict(generation=group.generation, targets=len(group.group.leases), trigger='once')
        request('c2_fault', 'fault'); request('c2_off')
        c2 = session.finish_group()
        proof = session.provider.group.c2_proof()
        group = session.provider.open_group('every_match', forwards=2, c2_proof=proof)
        c3_open = dict(generation=group.generation, targets=len(group.group.leases), trigger='every_match')
        request('c3_fault_0', 'fault'); request('c3_fault_1', 'fault'); request('c3_off')
        c3 = session.finish_group()
        cleanup = session.close()
        result = dict(mechanism='22.wrong_row.fused_positions', raw_device_launch_payload_observed=False,
            probe_sha256='a' * 64, setup=dict(provenance=session.provenance),
            healthy_before=requests[0]['output_tokens'], healthy_after=requests[-1]['output_tokens'],
            qualification=qualification, requests=requests, c2_open=c2_open, c3_open=c3_open,
            c2=c2, c3=c3, cleanup=cleanup)
        return json.loads(json.dumps(result, allow_nan=False))
    finally:
        fixture.tearDown()


class RoPEResultReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = cpu_result()

    def review(self, change=None):
        result = copy.deepcopy(self.original)
        if change is not None:
            change(result)
        return Review(result, expected_targets=3, _cpu_contract=True).run()

    def codes(self, report):
        return {issue['code'] for issue in report['issues']}

    def test_real_cpu_forwards_verify_content_but_missing_live_proof_never_grants_c3(self):
        report = self.review()
        self.assertTrue(report['c2_execution_evidence_valid'], report)
        self.assertTrue(report['c3_execution_evidence_valid'], report)
        self.assertEqual(report['actual_natural_targets'], 3)
        self.assertEqual(self.codes(report), {'C3_PROOF_ISSUANCE_USE_RECEIPT_MISSING'})
        self.assertFalse(report['c2_review_passed'] or report['c3_review_passed'] or report['public_qualification_granted'])

    def test_aggregate_success_flags_do_not_supply_the_verdict(self):
        report = self.review(lambda r: (r['c2'].update(contract_satisfied=False), r['c3'].update(contract_satisfied=False)))
        self.assertTrue(report['c2_execution_evidence_valid'] and report['c3_execution_evidence_valid'], report)

    def test_reader_subset_cannot_pass_from_candidate_or_hit_counts(self):
        def change(r):
            r['requests'][4]['worker']['independent']['rope_trace']['readers'].pop()
        report = self.review(change)
        self.assertFalse(report['c2_execution_evidence_valid'])
        self.assertIn('ALL_NATURAL_READERS_DENOMINATOR_CHANGED', self.codes(report))

    def test_independent_operator_count_cannot_be_shrunk_with_reader_summary(self):
        def change(r):
            row = r['requests'][4]['worker']
            row['independent']['operator_counts'][row['calls'][0]['operator']] += 1
        report = self.review(change)
        self.assertIn('INDEPENDENT_OPERATOR_DENOMINATOR_CHANGED', self.codes(report))

    def test_driver_mapping_requires_the_actual_worker_request(self):
        report = self.review(lambda r: r['requests'][4]['mapping'].update(internal_request_id='different'))
        self.assertIn('DRIVER_REQUEST_MAPPING_MISMATCH', self.codes(report))
        self.assertFalse(report['c2_execution_evidence_valid'])

    def test_clone_in_final_triton_arguments_is_not_accepted_from_join_flags(self):
        def change(r):
            launch = r['requests'][4]['worker']['independent']['rope_trace']['triton_launches'][0]['launches'][0]
            launch['arguments']['pos_ptr']['storage_id'] += 1
        report = self.review(change)
        self.assertIn('RECORDED_FINAL_LAUNCH_ARGUMENTS_DO_NOT_JOIN_READER', self.codes(report))

    def test_non_target_change_is_not_accepted_from_applied_or_changed_flags(self):
        def change(r):
            reader = r['requests'][4]['worker']['independent']['rope_trace']['readers'][0]
            reader['non_target_output'][0]['sha256'] = 'f' * 64
        report = self.review(change)
        self.assertIn('NON_TARGET_INPUT_OR_OUTPUT_CHANGED', self.codes(report))

    def test_fault_must_change_full_output_and_exact_legal_last_position(self):
        def change(r):
            reader = r['requests'][4]['worker']['independent']['rope_trace']['readers'][0]
            reader['actual_positions'][-1] = -1
        self.assertIn('ACTUAL_LEGAL_LAST_ROW_TRANSFORM_NOT_PROVEN', self.codes(self.review(change)))
        def unchanged(r):
            row = r['requests'][4]['worker']['independent']['rope_trace']['readers'][0]
            row['complete_output'] = copy.deepcopy(r['requests'][1]['worker']['independent']['rope_trace']['readers'][0]['complete_output'])
        self.assertIn('COMPLETE_READER_EFFECT_OR_HEALTHY_REPEAT_MISMATCH', self.codes(self.review(unchanged)))

    def test_old_cache_object_lifetime_cannot_be_replaced_by_equal_bytes(self):
        def change(r):
            r['requests'][4]['worker']['independent']['rope_trace']['readers'][0]['cache_lifetime_object_id'] += 1
        self.assertIn('READER_OR_CACHE_LIFETIME_CHANGED', self.codes(self.review(change)))

    def test_execution_digest_is_recomputed_from_full_reader_outer_and_triton(self):
        report = self.review(lambda r: r['c2']['calls'][0]['receipt'].update(execution_sha256='f' * 64))
        self.assertIn('COMPLETE_INDEPENDENT_EXECUTION_OR_EFFECT_HASH_MISMATCH', self.codes(report))

    def test_receipt_owner_generation_and_exact_once_execution_are_required(self):
        report = self.review(lambda r: r['c2']['calls'][0].update(generation=999))
        self.assertIn('CALL_OWNER_GENERATION_OR_WORKER_CHANGED', self.codes(report))
        report = self.review(lambda r: r['c2'].update(generation=True))
        self.assertIn('GROUP_OWNER_GENERATION_OR_SCOPE_MISMATCH', self.codes(report))
        report = self.review(lambda r: r['c3']['calls'][0]['receipt'].update(original_calls=2))
        self.assertIn('PER_CALL_ADMISSION_OR_COMPLETION_INVALID', self.codes(report))

    def test_off_full_vocab_hash_and_all_target_resource_receipts_are_required(self):
        def change(r):
            r['requests'][5]['worker']['output']['logits']['sha256'] = 'f' * 64
        self.assertIn('HEALTHY_OR_OFF_COMPLETE_OUTPUT_CHANGED', self.codes(self.review(change)))
        report = self.review(lambda r: r['c2']['recovery']['workers'][0].update(active_leases=1))
        self.assertIn('SAME_WORKER_FULL_OFF_OR_ZERO_RESOURCES_UNPROVEN', self.codes(report))
        report = self.review(lambda r: r['c2']['recovery']['targets'].pop())
        self.assertIn('RECOVERY_TARGET_OR_WORKER_MISSING', self.codes(report))

    def test_empty_structural_evidence_and_empty_join_maps_do_not_pass_all(self):
        report = self.review(lambda r: r['requests'][4]['worker']['independent']['rope_trace']['outer_operators'][0].update(reader_argument_join={}))
        self.assertIn('ACTUAL_OUTER_READER_JOIN_FAILED', self.codes(report))

    def test_result_hash_duplicate_keys_and_nonfinite_json_fail_before_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            for raw in (b'{"same":1,"same":2}', b'{"value":NaN}'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    read_result(path, hashlib.sha256(raw).hexdigest())
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'RAW_RESULT_SHA256_MISMATCH'):
                read_result(path, '0' * 64)

    def test_cli_uses_only_standard_library_and_emits_one_compact_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            raw = json.dumps(self.original, allow_nan=False).encode()
            path.write_bytes(raw)
            script = Path(__file__).resolve().parents[1] / 'experiments/c2c3_closure_20261117/review_natural_rope_result.py'
            result = subprocess.run([sys.executable, '-I', '-B', str(script), str(path),
                '--expected-sha256', hashlib.sha256(raw).hexdigest()], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 1)
            report = json.loads(result.stdout)
            self.assertFalse(report['c2_review_passed'] or report['c3_review_passed'])
            self.assertIn('ACTUAL_STANDARD_WORKER_REQUIRED', self.codes(report))


if __name__ == '__main__':
    unittest.main()
