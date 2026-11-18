"""Independent offline review from actual CPU Host/Runtime forwards, no NPU proof."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_rope_worker_core_probe as fixture
from fault_injection.tensor_observer import NaturalTensorObserver
from experiments.c2c3_closure_20261117 import review_natural_rope_result as old
from experiments.c2c3_closure_20261117 import review_worker_core_rope_result as new


class CoreRoPEResultReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = fixture.WorkerCoreRoPEProbeTests()
        case.setUp()
        cls.addClassCleanup(case.doCleanups)
        case.case.worker.rank = 0
        def capture(*args, **kwargs):
            return NaturalTensorObserver(*args, **dict(kwargs, record_all_operators=True))
        with patch.object(torch.distributed, 'get_rank', return_value=0), patch(
                'fault_injection.rope_serving.NaturalTensorObserver', side_effect=capture):
            result = case.run_probe()
        if result['status'] != fixture.probe.SUCCESS:
            raise AssertionError(result.get('error'))
        # Server artifacts use POSIX paths; this CPU fixture may run on Windows.
        for record in result['worker_core']['receipts']:
            record['path'] = record['path'].replace('\\', '/')
        result['probe_sha256'] = hashlib.sha256(Path(fixture.probe.__file__).read_bytes()).hexdigest()
        result['public_entrypoint'] = 'WorkerFaultClient+FaultConfig(22,wrong_row)'
        cls.folder = case.out
        fixture.probe.write_json(cls.folder / 'result.json', result)
        cls.result_hash = hashlib.sha256((cls.folder / 'result.json').read_bytes()).hexdigest()
        cls.original = new.read_artifacts(cls.folder / 'result.json', cls.result_hash)[:3]

    def review(self, change=None):
        result, host, receipts = copy.deepcopy(self.original)
        if change:
            change(result, host, receipts)
        return new.CoreReview(result, host, receipts, expected_targets=3, _cpu_contract=True).run()

    @staticmethod
    def codes(report):
        return {issue['code'] for issue in report['issues']}

    def test_complete_real_public_fixture_proofs_rpc_and_tenth_health_are_audited(self):
        report = self.review()
        self.assertEqual(report['status'], 'EVIDENCE_VERIFIED', report)
        self.assertTrue(report['c2_execution_evidence_valid'] and report['c3_execution_evidence_valid'])
        self.assertTrue(report['live_c2_issuance_use_evidence_valid'])
        self.assertGreater(report['checked_conditions'], 500)
        self.assertFalse(report['c2_review_passed'] or report['c3_review_passed'])
        self.assertFalse(report['live_capability_reconstructed'] or report['public_qualification_granted'])
        self.assertEqual(report['E5'], 'NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED')

    def test_source46_constants_and_missing_proof_verdict_remain_unchanged(self):
        self.assertEqual(old.FROZEN_PROBE_SHA256,
                         '95e0e93275f5140de6607ffdd283c65370374bc5936716223997542e091e565f')
        self.assertEqual(old.FROZEN_IMPLEMENTATION_SHA256,
                         'b77ff882d7c07f163f70ab32dfe9a7d0e004e3114a2c28d70b0f771b0b604cbd')
        report = old.Review(copy.deepcopy(self.original[0]), _cpu_contract=True).run()
        self.assertIn('C3_PROOF_ISSUANCE_USE_RECEIPT_MISSING', self.codes(report))
        self.assertFalse(report['c3_review_passed'])

    def test_issued_and_consumed_exact_object_and_pid_cannot_be_asserted_by_flags(self):
        for field in ('pid', 'python_object_id'):
            with self.subTest(field=field):
                def mutate(result, *_):
                    result['c3']['live_c2_proof']['consumed']['object_binding'][field] += 1
                report = self.review(mutate)
                self.assertFalse(report['live_c2_issuance_use_evidence_valid'])
                self.assertTrue(self.codes(report) & {'LIVE_PROOF_FIELDS_DIGEST_OR_PROCESS_BINDING_CHANGED',
                    'ISSUED_AND_CONSUMED_ORIGINAL_OBJECT_RECEIPTS_DIFFER'})

    def test_proof_report_discovery_observation_recovery_hashes_are_recomputed(self):
        for field in ('report_sha256', 'discovery_sha256', 'observation_sha256', 'recovery_sha256',
                      'candidates_sha256', 'owner', 'generation', 'session_id', 'target_ids'):
            with self.subTest(field=field):
                def mutate(result, *_):
                    for receipt in (result['c2']['live_c2_proof']['issued'][0],
                                    result['c3']['live_c2_proof']['consumed']):
                        receipt['proof'][field] = ([] if field == 'target_ids' else
                            99 if field == 'generation' else 'f' * 64)
                        receipt['proof_sha256'] = new.sha(receipt['proof'])
                report = self.review(mutate)
                self.assertIn('ISSUED_C2_PROOF_NOT_BOUND_TO_ORIGINAL_FULL_REPORT', self.codes(report))

    def test_proof_scope_and_consumer_group_binding_are_exact(self):
        def scope(result, *_):
            for receipt in (result['c2']['live_c2_proof']['issued'][0], result['c3']['live_c2_proof']['consumed']):
                receipt['proof']['scope']['workload_sha256'] = 'f' * 64
                receipt['proof_sha256'] = new.sha(receipt['proof'])
        self.assertIn('ISSUED_C2_PROOF_NOT_BOUND_TO_ORIGINAL_FULL_REPORT', self.codes(self.review(scope)))
        for field in ('scope', 'owner', 'generation', 'target_ids', 'candidates_sha256', 'session_id'):
            with self.subTest(field=field):
                def mutate(result, *_):
                    result['c3']['live_c2_proof']['consumed']['consumer'][field] = None
                report = self.review(mutate)
                self.assertIn('CONSUMER_SCOPE_OWNER_GENERATION_OR_TARGETS_CHANGED', self.codes(report))
                self.assertTrue(report['c2_execution_evidence_valid'] and report['c3_execution_evidence_valid'])
                self.assertFalse(report['live_c2_issuance_use_evidence_valid'])

    def test_frozen_report_schema_and_complete_hash_do_not_drop_new_fields(self):
        def extra(result, *_):
            result['c2']['unreviewed_status'] = True
        self.assertIn('FROZEN_COVERAGE_REPORT_SCHEMA_CHANGED', self.codes(self.review(extra)))
        def flag(result, *_):
            result['c2']['contract_satisfied'] = False
        self.assertIn('ISSUED_C2_PROOF_NOT_BOUND_TO_ORIGINAL_FULL_REPORT', self.codes(self.review(flag)))
        def copies(result, *_):
            result['c2']['live_c2_proof']['issued'].append(copy.deepcopy(result['c2']['live_c2_proof']['issued'][0]))
        self.assertIn('EXACT_ONE_ISSUANCE_AND_CONSUMPTION_REQUIRED', self.codes(self.review(copies)))
        def boolean_generation(result, *_):
            for receipt in (result['c2']['live_c2_proof']['issued'][0], result['c3']['live_c2_proof']['consumed']):
                receipt['proof']['generation'] = True
                receipt['proof_sha256'] = new.sha(receipt['proof'])
        self.assertIn('LIVE_PROOF_FIELDS_DIGEST_OR_PROCESS_BINDING_CHANGED', self.codes(self.review(boolean_generation)))

    def test_full_reader_alias_bytes_and_call_checks_are_still_active(self):
        changes = [
            (lambda r: r['requests'][4]['worker']['independent']['rope_trace']['readers'].pop(),
             'ALL_NATURAL_READERS_DENOMINATOR_CHANGED'),
            (lambda r: r['requests'][4]['worker']['independent']['rope_trace']['triton_launches'][0]
                ['launches'][0]['arguments']['pos_ptr'].update(storage_id=1),
             'RECORDED_FINAL_LAUNCH_ARGUMENTS_DO_NOT_JOIN_READER'),
            (lambda r: r['c2']['calls'][0]['receipt'].update(original_calls=2),
             'PER_CALL_ADMISSION_OR_COMPLETION_INVALID'),
            (lambda r: r['requests'][4]['worker']['independent']['rope_trace']['readers'][1]
                .update(module_object_id=1), 'INDEPENDENT_CANONICAL_MODULE_OWNER_MISMATCH'),
        ]
        for change, code in changes:
            with self.subTest(code=code):
                self.assertIn(code, self.codes(self.review(lambda r, *_: change(r))))

    def test_rpc_sequence_scope_and_uncertain_failed_response_are_not_hidden(self):
        def changed_order(result, host, receipts):
            receipts[5], receipts[6] = receipts[6], receipts[5]
            host['history'] = receipts
        self.assertIn('COMPLETE_ORIGINAL_RPC_SEQUENCE_OR_WORKER_CHANGED', self.codes(self.review(changed_order)))
        for key, value in (('owner', 'foreign'), ('generation', 2), ('attach_nonce', 'foreign')):
            with self.subTest(key=key):
                def scope(result, host, receipts):
                    receipts[17]['response']['ticket'][key] = value
                    host['history'] = receipts
                self.assertIn('RPC_FAILURE_OR_SCOPE_DRIFT', self.codes(self.review(scope)))
        def uncertain(result, host, receipts):
            receipts[17]['response']['status'] = 'RPC_UNCERTAIN'
            host['history'] = receipts
        self.assertIn('RPC_FAILURE_OR_SCOPE_DRIFT', self.codes(self.review(uncertain)))

    def test_query_rows_group_open_and_proof_timing_join_the_original_receipts(self):
        def row(result, host, receipts):
            receipts[14]['response']['provider_state']['serving']['last_request']['request_id'] = 'foreign'
            host['history'] = receipts
        self.assertIn('QUERY_ROW_DIFFERS_FROM_COMPLETE_RECORDED_RETURN', self.codes(self.review(row)))
        def early(result, host, receipts):
            receipts[15]['response']['provider_state']['coverage']['C2'] = result['c2']
            host['history'] = receipts
        self.assertIn('RPC_ORIGINAL_EVIDENCE_TIMING_OR_BYTES_CHANGED', self.codes(self.review(early)))
        def opened(result, *_):
            result['c3_open']['owner'] = 'foreign'
        self.assertIn('RPC_GROUP_OPEN_SCOPE_CHANGED', self.codes(self.review(opened)))

    def test_tenth_request_uses_complete_same_worker_bytes_after_cleanup(self):
        for field, value in (('request_id', 'actual-enqueued-8'), ('cleanup_before_request', False),
                             ('fault_hooks', 1), ('worker', {}), ('output', {})):
            with self.subTest(field=field):
                def mutate(result, *_):
                    result['health_request']['worker'][field] = value
                self.assertIn('TENTH_ACTUAL_SAME_WORKER_FULL_HEALTH_REQUEST_UNPROVEN', self.codes(self.review(mutate)))
        def cleanup(result, host, receipts):
            receipts[23]['response']['provider_state']['coverage']['cleanup']['requests'].append('too-early-health')
            host['history'] = receipts
        self.assertIn('CLEANUP_BEFORE_HEALTH_OR_FINAL_SCOPE_CHANGED', self.codes(self.review(cleanup)))

    def test_raw_rpc_hash_and_missing_or_extra_files_reject_before_content_review(self):
        record = self.original[0]['worker_core']['receipts'][0]
        path = self.folder / record['path']; raw = path.read_bytes()
        try:
            path.write_bytes(raw + b' ')
            with self.assertRaisesRegex(ValueError, 'RAW_RESULT_SHA256_MISMATCH'):
                new.read_artifacts(self.folder / 'result.json', self.result_hash)
        finally:
            path.write_bytes(raw)
        extra = self.folder / 'worker_core_rpc' / 'unexpected.json'
        try:
            extra.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'FULL_HOST_HISTORY_OR_RPC_FILE_SET_CHANGED'):
                new.read_artifacts(self.folder / 'result.json', self.result_hash)
        finally:
            extra.unlink()
        with self.assertRaisesRegex(ValueError, 'UNSAFE_EVIDENCE_OR_SOURCE_PATH'):
            new.confined_file(self.folder, '../result.json')

    def test_source_manifest_pin_is_required_and_cli_imports_no_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source_manifest.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'RAW_RESULT_SHA256_MISMATCH'):
                new.verify_source(Path(folder))
            script = Path(new.__file__)
            result = subprocess.run([sys.executable, '-I', '-B', str(script), str(self.folder / 'result.json'),
                '--source', folder], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 1)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'], 'EVIDENCE_REJECTED')
            self.assertFalse(report['c2_review_passed'] or report['c3_review_passed'])

    def test_source_bytes_hash_chain_and_unmanifested_code_are_checked(self):
        root = Path(__file__).resolve().parents[1]
        paths = list(root.glob('fault_injection/*.py'))
        paths += list(root.glob('src/demoncat/runtime/vllm_ascend/worker_extension/*.py'))
        paths += list(root.glob('src/demoncat/runtime/vllm_ascend/worker_extension/providers/*.py'))
        paths += [root / new.PROBE_PATH]
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder)
            files = []
            for path in paths:
                relative = path.relative_to(root).as_posix()
                raw = path.read_bytes()
                copied = destination / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes(raw)
                files.append(dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            manifest = dict(schema='demoncat.c2c3.source.v1', base_commit=new.SOURCE_COMMIT, files=files,
                contains_model_weights=False, contains_native_binaries=False, contains_tensor_payloads=False)
            fixture.probe.write_json(destination / 'source_manifest.json', manifest)
            report = new.verify_source(destination, _cpu_contract=True)
            self.assertFalse(report['actual_frozen_source_verified'])
            self.assertEqual(report['implementation_sha256'], self.original[0]['c2']['scope']['implementation_sha256'])
            extra = destination / 'sitecustomize.py'
            extra.write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'UNMANIFESTED_OR_MISSING_FROZEN_SOURCE_FILE'):
                new.verify_source(destination, _cpu_contract=True)
            extra.unlink()
            copied = destination / new.BINDING_PATHS[0]
            copied.write_bytes(copied.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'FROZEN_SOURCE_BYTES_CHANGED'):
                new.verify_source(destination, _cpu_contract=True)


if __name__ == '__main__':
    unittest.main()
