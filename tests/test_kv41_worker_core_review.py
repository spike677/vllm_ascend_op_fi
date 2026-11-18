"""Independent saved-result audit over a real CPU Host/Runtime protocol fixture."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_postwrite_20261117 import review41_worker_core_result as review


class Core41ReceiptReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import locally: do not re-export another TestCase and repeat its tests.
        from test_kv41_worker_core_probe import CoreKV41ProbeTests
        cls.fixture = CoreKV41ProbeTests()
        cls.fixture.setUp()
        try:
            cls.fixture.drive()
            cls.document = review.read(cls.fixture.out / 'result.json')
            cls.host = review.read(cls.fixture.out / 'host_receipt.json')
        except BaseException:
            cls.fixture.doCleanups()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def evaluate(self, document=None, host=None):
        d = self.document if document is None else document
        h = self.host if host is None else host
        return review.review_evidence(d, h, h['history'], natural=False)

    def rejected(self, change, code=None):
        d = deepcopy(self.document)
        change(d)
        result = self.evaluate(d)
        self.assertFalse(result['consistent'])
        if code:
            self.assertIn(code, [v['code'] for v in result['issues']], result['issues'])
        self.assertFalse(result['public_qualification_granted'])
        return result

    def rpc_change(self, index, change, code):
        host = dict(self.host, history=list(self.host['history']))
        host['history'][index] = deepcopy(host['history'][index])
        change(host['history'][index])
        result = self.evaluate(host=host)
        self.assertFalse(result['consistent'])
        self.assertIn(code, [v['code'] for v in result['issues']], result['issues'])

    def test_actual_cpu_protocol_passes_with_separate_three_owners_and_no_qualification(self):
        result = self.evaluate()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertGreater(result['checks'], 1000)
        self.assertEqual(result['details']['rpc']['rpc_actions'], 43)
        self.assertEqual(result['details']['c2']['recomputed_hits'], 2)
        self.assertEqual(result['details']['c3']['recomputed_hits'], 4)
        owners = result['details']['rpc']
        self.assertNotEqual(owners['host_owner'], owners['runtime_owner'])
        self.assertIsInstance(owners['native_owner'], int)
        self.assertFalse(result['public_qualification_granted'])
        self.assertEqual(result['evidence_kind'], 'CPU_CONTRACT_ONLY')

    def test_original_rpc_and_host_files_rehashed_without_rewriting(self):
        d = deepcopy(self.document)
        _, _, result = review.load_artifacts(self.fixture.out, d)
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['checks'], 88)

    def test_success_headline_never_substitutes_for_missing_full_prefix(self):
        def change(d):
            d['status'] = 'STANDARD_WORKER_KV41_CORE_CONTRACT_COMPLETE'
            d['c2']['contract_satisfied'] = True
            d['c2']['completed_requests'][0]['byte_evidence'][0]['valid_kv_range_proof']['ranges'].pop()
        self.rejected(change, 'KV41_EXISTING_EVERY_COMPLETE_VALID_KV_RANGE_RESTORED')

    def test_headline_is_not_used_as_evidence(self):
        d = deepcopy(self.document)
        d['status'] = 'HEADLINE_HAS_NO_AUTHORITY'
        self.assertTrue(self.evaluate(d)['consistent'])

    def test_actual_driver_request_must_match_model_and_output(self):
        self.rejected(lambda d: d['requests'][8]['mapping'].update(internal_request_id='foreign-request'),
                      'ACTUAL_COMPLETED_FULL_ROOT_CAPTURE')
        self.rejected(lambda d: d['requests'][8]['output'].update(request_id='foreign-output'),
                      'DRIVER_OUTPUT_REQUEST_MAPPING')

    def test_same_shape_clone_storage_cannot_replace_actual_fia_view(self):
        self.rejected(lambda d: d['requests'][7]['worker']['rows'][0]['boundary']['observational_metadata']['value'].update(
            storage_identity=123), 'ACTUAL_3D_TO_4D_STORAGE_VIEW')

    def test_boundary_request_binding_cannot_be_borrowed_from_another_request(self):
        self.rejected(lambda d: d['requests'][7]['worker']['rows'][0]['boundary']['observational_metadata']['scheduler'].update(
            request_ids=['foreign']), 'BOUNDARY_ACTUAL_SCHEDULER_JOIN')

    def test_all_actual_targets_required_even_when_headline_totals_are_lowered(self):
        def change(d):
            g = d['c2']; g['planned_targets'] = g['planned_targets_hit'] = g['hits'] = 1
            g['independent_discovery'][0]['inventory']['calls'].pop()
        self.rejected(change, 'GROUP_SCOPE_AND_NO_OMITTED_FAILURES')

    def test_original_consumer_is_exactly_once(self):
        self.rejected(lambda d: d['requests'][8]['worker']['independent_inventory']['calls'][0].update(original_calls=2),
                      'INDEPENDENT_ORIGINAL_CALL_AND_WRITER')

    def test_partial_prefix_nans_and_off_changes_refuse(self):
        mutations = (
            lambda r: r['input_after'].update(active_value_sha256='a'*64),
            lambda r: r['input_before'].update(active_key_nonfinite=1),
            lambda r: r['valid_kv_range_proof'].update(bytes_per_cache=1),
        )
        for change in mutations:
            with self.subTest(change=change):
                self.rejected(lambda d: change(d['c2']['completed_requests'][-1]['byte_evidence'][0]),
                    'KV41_EXISTING_EVERY_COMPLETE_VALID_KV_RANGE_RESTORED')

    def test_wrong_span_and_broader_tail_are_not_accepted_by_shape(self):
        for field in ('relative_indices', 'allocation_element_indices'):
            with self.subTest(field=field):
                self.rejected(lambda d: d['requests'][8]['worker']['rows'][0]['boundary'][field].append(0),
                              'EXACT_LEGAL_TAIL_SPAN')

    def test_native_owner_generation_and_actual_changed_count(self):
        for key, value, code in (('owner', 7, 'ACTUAL_NATIVE_OWNER_CONSUMER_RESTORE'),
                                 ('generation', 7, 'ACTUAL_NATIVE_OWNER_CONSUMER_RESTORE'),
                                 ('changed_elements', 1, 'ACTUAL_CHANGED_SPAN')):
            with self.subTest(key=key):
                self.rejected(lambda d: d['c2']['executions'][0]['native'].update({key:value}),
                              'ORIGINAL_NATIVE_EXECUTION_ROWS')
                # Mutate the canonical byte record too: independent ABI checks
                # still refuse even when the duplicated execution row agrees.
                def change(d):
                    d['c2']['completed_requests'][1]['byte_evidence'][0]['native'][key] = value
                    d['c2']['executions'][0]['native'][key] = value
                self.rejected(change, code)

    def test_c2_exhausted_request_cannot_admit_a_second_native_call(self):
        self.rejected(lambda d: d['c2']['completed_requests'][2]['byte_evidence'][0].update(native_applied=True),
                      'ACTUAL_NATIVE_PHASE_ADMISSION')

    def test_effect_and_execution_receipts_are_recomputed(self):
        for key in ('execution_sha256', 'effect_sha256'):
            with self.subTest(key=key):
                self.rejected(lambda d: d['c3']['calls'][0]['receipt'].update({key:'a'*64}),
                              'RECOMPUTED_INDEPENDENT_TICKET_ADMISSION_EFFECT_RECEIPT')

    def test_discovery_and_observation_hashes_are_recomputed(self):
        self.rejected(lambda d: d['c2']['discovery'].update(candidates_sha256='b'*64), 'RECOMPUTED_FULL_DISCOVERY')
        self.rejected(lambda d: d['c3']['observation'].update(calls_sha256='b'*64), 'RECOMPUTED_FULL_OBSERVATION')

    def test_missing_live_proof_is_separate_from_valid_execution_and_recovery(self):
        result = self.rejected(lambda d: d['c2']['live_c2_proof'].update(issued=[]))
        self.assertTrue(result['scopes']['c2_execution_and_recovery']['consistent'])
        self.assertTrue(result['scopes']['c3_execution_and_recovery']['consistent'])
        self.assertFalse(result['scopes']['live_c2_proof']['consistent'])

    def test_c3_must_use_original_issued_object_and_complete_c2_report(self):
        self.rejected(lambda d: d['c3']['live_c2_proof']['consumed']['object_binding'].update(python_object_id=1),
                      'C3_USED_SAME_ACTUAL_C2_OBJECT_AND_SCOPE')
        self.rejected(lambda d: d['c2']['live_c2_proof']['issued'][0]['proof'].update(report_sha256='0'*64),
                      'ORIGINAL_C2_PROOF_ALL_HASHES')
        self.rejected(lambda d: d['c3']['live_c2_proof']['consumed'].update(serializable_authority=True),
                      'C3_USED_SAME_ACTUAL_C2_OBJECT_AND_SCOPE')

    def test_c3_does_not_reuse_c2_generation(self):
        self.rejected(lambda d: d['c3'].update(generation=d['c2']['generation']),
                      'C3_USED_SAME_ACTUAL_C2_OBJECT_AND_SCOPE')

    def test_public_rpc_order_identity_and_uncertain_response(self):
        self.rpc_change(6, lambda r: r.update(action='advance'), 'ORIGINAL_43_RPC_HISTORY')
        self.rpc_change(6, lambda r: r['response']['ticket'].update(owner='foreign'), 'PUBLIC_RPC_SCOPE_AND_SUCCESS')
        self.rpc_change(6, lambda r: r['response'].update(status='RPC_UNCERTAIN'), 'PUBLIC_RPC_SCOPE_AND_SUCCESS')

    def test_query_cannot_claim_future_request_completion(self):
        self.rpc_change(6, lambda r: r['response']['provider_state']['serving']['actual_requests'].append(
            deepcopy(self.document['worker_core']['serving']['actual_requests'][0])),
            'NO_FABRICATED_OR_REPLAYED_REQUEST_COMPLETION')

    def test_proof_cannot_appear_before_c2_health_and_c3_lossless(self):
        self.rpc_change(28, lambda r: r['response']['provider_state']['group']['live_c2_proof'].update(
            issued=deepcopy(self.document['c2']['live_c2_proof']['issued'])), 'LIVE_PROOF_ISSUANCE_USE_TEMPORAL_ORDER')
        self.rpc_change(30, lambda r: r['response']['provider_state']['group']['live_c2_proof'].update(
            consumed=deepcopy(self.document['c3']['live_c2_proof']['consumed'])), 'LIVE_PROOF_ISSUANCE_USE_TEMPORAL_ORDER')

    def test_released_headline_cannot_hide_owned_resources(self):
        self.rpc_change(41, lambda r: r['response']['provider_state'].update(leases=1), 'ACTUAL_PUBLIC_RESOURCES_ZERO')
        self.rpc_change(40, lambda r: r['response']['provider_state']['owned_api_state'].update(owned_links_present=1),
                        'ACTUAL_PUBLIC_RESOURCES_ZERO')
        self.rejected(lambda d: d['c2']['recovery']['targets'][0].update(drained=False), 'ACTUAL_TARGET_DRAIN_RELEASE_OFF')

    def test_detach_must_be_actual_and_eighteenth_request_same_worker(self):
        self.rpc_change(42, lambda r: r['response'].update(recoveries=[{'error':'retained'}]), 'PUBLIC_OWNER_UNREGISTER_AND_DETACH')
        self.rejected(lambda d: d['post_unregister_health']['worker'].update(worker_start_identity='restarted'),
                      'EIGHTEENTH_ORDINARY_SAME_WORKER_REQUEST')
        self.rejected(lambda d: d['post_unregister_health']['mapping'].update(
            internal_request_id=d['requests'][0]['mapping']['internal_request_id']), 'EIGHTEENTH_ORDINARY_SAME_WORKER_REQUEST')

    def test_cpu_fixture_never_becomes_natural_worker_evidence(self):
        result = review.review_evidence(self.document, self.host, self.host['history'], natural=True)
        self.assertFalse(result['consistent'])
        self.assertFalse(result['public_qualification_granted'])


class OfflineReviewIOTests(unittest.TestCase):
    def test_duplicate_nonfinite_and_overflow_json_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'raw.json'
            for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    review.read(path)

    def test_artifact_path_cannot_escape_or_use_host_drive(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in ('../other.json', '/tmp/other.json', 'C:/other.json', 'a\\b', 'a//b'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    review.under(folder, path)

    def test_import_and_reused_helper_never_import_accelerator_or_production_runtime(self):
        code = ('import sys; from experiments.c2c3_postwrite_20261117 import review41_worker_core_result as r; '
                'sys.addaudithook(r.deny_runtime); r.helper(); '
                'assert not any(n == "torch" or n.startswith(("torch_npu", "vllm", "ctypes", "demoncat.runtime")) for n in sys.modules)')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_wrong_original_sha_fails_before_source_or_native_access(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'; path.write_text('{}')
            result = subprocess.run([sys.executable, '-B', str(Path(review.__file__)), '--result', str(path),
                '--result-sha256', 'a'*64, '--source-root', folder, '--native-library', str(Path(folder)/'absent.so')],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            value = json.loads(result.stdout)
            self.assertIn('ORIGINAL_RESULT_SHA_MISMATCH', value['error'])
            self.assertFalse(value['public_qualification_granted'])


if __name__ == '__main__':
    unittest.main()
