"""CPU public-return/typed-record fixtures only; no actual Native qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_closure_20261117 import review_native26_fia_single_source62 as review


class FiaSource62ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_writer26_fia_single_site import FiaSingleSiteTests
        from fault_injection.deployments.writer26_group import packet
        cls.b = review.shared()
        cls.case = FiaSingleSiteTests(); cls.case.setUp()
        try:
            experiment = cls.case.make(30)
            native = cls.case.native
            execute, completed = native.execute, native.finished
            # The existing lifecycle fixture uses a minimal lease dictionary.
            # Supply an explicit CPU ABI envelope at the real admission point;
            # this tests serialization joins, not CANN descriptor allocation.
            def typed_execute(call, mode, target, allocation, *, typed_before):
                execute(call, mode, target, allocation, typed_before=typed_before)
                base = dict(version=2, size=248, mode=cls.b.MODE_IDS[mode], reserved=0,
                    owner_hi=1, owner_lo=2, generation=1, thread=call.thread, target=cls.b.words(target),
                    expected=packet(call), residual_allocation_end=0, output_allocation_begin=0, shift_bytes=0)
                return dict(version=3, size=912, lease=base, original_before=deepcopy(typed_before))
            def typed_completed(pointer):
                token = pointer.contents.token
                destination = native.fia_typed.snapshots[token]['before_execute']['effective']['raw']
                pointer.contents.replacement = destination
                native.receipts[token]['replacement'] = destination
                return completed(pointer)
            native.execute, native.finished = typed_execute, typed_completed
            cls.case.until('complete'); experiment.close()
            cls.inventory = experiment.report()
        except BaseException:
            cls.case.doCleanups()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.case.doCleanups()

    def evaluate(self, value=None):
        return review.typed_review(self.b, self.inventory if value is None else value,
            review.FiaArtifacts(self.b, self.case.folder, recorded_root=self.case.folder))

    def rejected(self, change, code):
        value = deepcopy(self.inventory); change(value)
        result = self.evaluate(value)
        self.assertFalse(result['consistent'])
        self.assertIn(code, [v['code'] for v in result['issues']], result['issues'])

    def test_actual_cpu_public_returns_full_bins_and_typed_joins(self):
        result = self.evaluate()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['typed_records'], 485)
        self.assertEqual(result['stock_targets'], 30)
        self.assertEqual(result['full_output_bins_rehashed'], 65)
        self.assertEqual(result['full_output_bytes_rehashed'], 65*14976)
        self.assertFalse(result['public_qualification_granted'] or result['c2'] or result['c3'] or result['e5'])

    def test_wrong_flat_storage_span_is_not_nd_allowlist_acceptance(self):
        self.rejected(lambda v: v['rows'][0]['fia_typed_records'][0]['original']['storage'].__setitem__(0, 1),
                      'EXACT_NCL_VIEW_FLAT_STORAGE_AND_STRIDES')

    def test_view_or_stride_change_in_same_shape_descriptor_refuses(self):
        self.rejected(lambda v: v['rows'][3]['fia_typed_records'][0]['effective']['strides'].__setitem__(1, 1),
                      'EXACT_NCL_VIEW_FLAT_STORAGE_AND_STRIDES')

    def test_unselected_descriptor_cannot_be_replaced(self):
        self.rejected(lambda v: v['rows'][0]['fia_typed_records'][1]['effective'].update(descriptor=3),
                      'UNSELECTED_STOCK_DESCRIPTOR_UNCHANGED')

    def test_wrong_native_call_cannot_supply_typed_record(self):
        self.rejected(lambda v: v['rows'][0]['fia_typed_records'][0].update(event_sequence=1),
                      'CURRENT_TYPED_RECORD_NATIVE_REQUEST_JOIN')

    def test_extra_unobserved_owned_execute_and_trace_loss_refuse(self):
        self.rejected(lambda v: v['rows'][1]['fia_typed_after'].update(overflow=1), 'COMPLETE_QUIESCENT_TYPED_TRACE_ABI')
        self.rejected(lambda v: v['rows'][1]['fia_typed_records'].append(deepcopy(v['rows'][1]['fia_typed_records'][0])),
                      'COMPLETE_TYPED_DENOMINATOR')

    def test_public_return_cannot_be_same_value_clone_or_foreign_request(self):
        self.rejected(lambda v: v['public_output_observer']['calls'][0].update(tensor_object_identity=1),
                      'LITERAL_PUBLIC_RETURN_OBJECT_AND_ORIGINAL_CALL')
        self.rejected(lambda v: v['public_output_observer']['calls'][0].update(request_id='foreign'),
                      'LITERAL_PUBLIC_RETURN_OBJECT_AND_ORIGINAL_CALL')

    def test_public_format_mismatch_is_explicit_and_source_bound(self):
        self.rejected(lambda v: v['public_output_observer']['calls'][0]['format_observation'].update(public_format=47),
                      'PUBLIC_ND_IS_CURRENT_TYPED_NCL_FORMAL')

    def test_literal_api_original_call_must_be_once(self):
        self.rejected(lambda v: v['public_output_observer']['calls'][0].update(original_calls=2),
                      'LITERAL_PUBLIC_RETURN_OBJECT_AND_ORIGINAL_CALL')

    def test_public_boundary_cannot_include_another_native_call(self):
        self.rejected(lambda v: v['public_output_observer']['calls'][0]['typed_before'].update(stored=1),
                      'ONLY_CURRENT_NATIVE_CALL_INSIDE_LITERAL_PUBLIC_RETURN')

    def test_consumer_snapshot_and_owned_descriptor_release_are_required(self):
        self.rejected(lambda v: v['outcomes'][0]['fia_typed_consumer_snapshot'].update(released=0),
                      'OWNED_DESCRIPTOR_RELEASE_AND_CONSUMER_TIME_SNAPSHOT')
        self.rejected(lambda v: v['outcomes'][0]['fia_public_typed']['snapshots']['before_execute']['effective'].update(raw=1),
                      'OWNED_DESCRIPTOR_RELEASE_AND_CONSUMER_TIME_SNAPSHOT')

    def test_lease_before_cannot_be_reused_from_another_call(self):
        self.rejected(lambda v: v['outcomes'][0]['lease'].update(original_before=deepcopy(v['rows'][0]['fia_typed_records'][0])),
                      'ACTUAL_TYPED_ADMISSION_ABI')

    def test_executor_and_replacement_must_join_actual_before_execute(self):
        self.rejected(lambda v: v['outcomes'][3]['consumer_receipt'].update(executor=999),
                      'NATIVE_RECEIPT_ACTUAL_TYPED_EXECUTOR_DESTINATION')
        self.rejected(lambda v: v['outcomes'][3]['consumer_receipt'].update(replacement=1),
                      'NATIVE_RECEIPT_ACTUAL_TYPED_EXECUTOR_DESTINATION')

    def test_retained_stock_or_api_owner_is_not_released(self):
        self.rejected(lambda v: v['fia_typed_stock_bridge'].update(retained_host_bytes=14976),
                      'STOCK_OWNER_ZERO_AND_ONLY_FRESH_REFERENCE')
        self.rejected(lambda v: v['public_output_observer'].update(owned_api_wrappers=1),
                      'ALL_ACTUAL_PUBLIC_FIA_RETURNS_AND_RESTORE')

    def test_stock_pairs_compare_complete_bytes_even_when_new_hash_is_consistent(self):
        value = deepcopy(self.inventory)
        row = value['rows'][1]['callbacks'][0]
        artifact = row['fia_full_output']
        original = Path(artifact['path']).read_bytes()
        changed = original[:-2] + bytes((original[-2] ^ 1, original[-1]))
        destination = self.case.folder / 'explicit-cpu-corrupted-stock.bin'
        destination.write_bytes(changed)
        artifact.update(path=str(destination), sha256=hashlib.sha256(changed).hexdigest())
        header = hashlib.sha256(json.dumps(['torch.bfloat16', [13, 9, 64]]).encode()); header.update(changed)
        artifact['tensor_sha256'] = header.hexdigest()
        result = self.evaluate(value)
        self.assertIn('TWO_STOCK_FULL_BYTES_AND_GEOMETRY_EQUAL', [v['code'] for v in result['issues']])

    def test_bin_hash_length_and_path_are_independently_verified(self):
        row = self.inventory['rows'][0]['callbacks'][0]
        for field, value, reason in (('sha256', '0'*64, 'SHA_CHANGED'), ('bytes', 1, 'LENGTH_CHANGED'),
                                      ('path', '/outside/fake.bin', 'MAPPING_REQUIRED')):
            with self.subTest(field=field):
                artifact = dict(row['fia_full_output'], **{field:value})
                with self.assertRaisesRegex(ValueError, reason):
                    review.FiaArtifacts(self.b, self.case.folder, recorded_root=self.case.folder).bytes(
                        artifact, row['fia_typed_public_records'])

    def test_partial_run_is_a_refusal_not_c1(self):
        value = deepcopy(self.inventory)
        value.update(first_failure='ACTUAL_CURRENT_DESCRIPTOR_CHANGED', outcomes=value['outcomes'][:1], rows=value['rows'][:3])
        result = review.refusal(value, {'error':'actual refusal'})
        self.assertEqual(result['completed_outcome_modes'], ['reference'])
        self.assertFalse(result['single_site_source_bound_evidence_reconciled'] or result['public_qualification_granted'])


class FiaOfflineOnlyTests(unittest.TestCase):
    def test_shared_review_and_fia_layer_import_no_framework_or_native_library(self):
        code = ('import sys; from experiments.c2c3_closure_20261117 import review_native26_fia_single_source62 as r; '
                'b=r.shared();sys.addaudithook(b.forbid_runtime); '
                'assert not any(n.split(".")[0] in ("torch","torch_npu","vllm","ctypes") for n in sys.modules)')
        result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_rejects_wrong_original_before_loading_source_or_native(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'result.json').write_text('{}')
            result = subprocess.run([sys.executable, '-B', review.__file__, '--evidence-root', root,
                '--source-root', root, '--native-root', root], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            value = json.loads(result.stdout)
            self.assertIn('ORIGINAL_SOURCE62_SHA_MISMATCH', value['error'])
            self.assertFalse(value['single_site_source_bound_evidence_reconciled'])


class FiaSharedReceiptTests(unittest.TestCase):
    """Saved-packet doubles exercise only the shared mechanism-specific checks."""
    def setUp(self):
        from test_native26_single_source54_review import Fixture
        self.b = review.shared()
        self.fixture = Fixture(self.enterContext(tempfile.TemporaryDirectory()))

    def test_fia_fault_token_may_change_but_off_must_still_recover(self):
        f = self.fixture
        f.phases[5]['generated_tokens'] = [154]
        f.observations[5]['top1'] = 154
        r = self.b.Review()
        self.b.requests(r, f.document, f.inventory, f.serving, mechanism=2)
        self.assertTrue(r.summary()['consistent'], r.summary())
        f.phases[6]['generated_tokens'] = [154]
        f.observations[6]['top1'] = 154
        r = self.b.Review()
        self.b.requests(r, f.document, f.inventory, f.serving, mechanism=2)
        self.assertIn('ACTUAL_GENERATION_AND_POST_RELEASE_HEALTH', r.errors)

    def fault_receipt(self):
        f = self.fixture
        value = f.outcomes[3]
        value['native_call'].update(mechanism=2, registration=0)
        lease = value['lease']
        lease.update(residual_allocation_end=0, output_allocation_begin=0, shift_bytes=0)
        value['lease'] = {'lease': lease}
        for name in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'):
            value[name].update(mechanism=2, registration=777, executor=888, replacement=value['native_call']['residual'])
        return value

    def outcome(self, value):
        f = self.fixture
        r = self.b.Review()
        self.b.outcome(r, value, f.rows[5]['candidates'][f.selected['index']], f.rows[5],
            f.inventory['single_site_c1']['reference_lossless_off'], 'synthetic-FIA-envelope', mechanism=2)
        return r

    def test_fia_receipt_requires_direct_destination_and_no_qmm_shift(self):
        value = self.fault_receipt()
        self.assertTrue(self.outcome(value).summary()['consistent'])
        value['lease']['lease']['shift_bytes'] = 32
        self.assertIn('FIA_DIRECT_LIVE_RESIDUAL_DESTINATION', self.outcome(value).errors)
        value['lease']['lease']['shift_bytes'] = 0
        for name in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'):
            value[name]['replacement'] += 32
        self.assertIn('FIA_ACTUAL_RESIDUAL_DESTINATION_EFFECT', self.outcome(value).errors)

    def test_fia_nonzero_effect_full_restore_and_original_receipt_are_required(self):
        value = self.fault_receipt()
        complete = deepcopy(value)
        value['consumer_receipt']['changed_residual_bytes'] = 0
        self.assertIn('FIRST_NATIVE_RECEIPT_AND_ACK_ONLY_RELEASE', self.outcome(value).errors)
        for name in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'):
            value[name]['changed_residual_bytes'] = 0
        self.assertIn('FIA_ACTUAL_RESIDUAL_DESTINATION_EFFECT', self.outcome(value).errors)
        value = complete
        for name in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'):
            value[name]['residual_restored_exact'] = 0
        self.assertIn('NATIVE_EXECUTION_CONSUMPTION_RECOVERY', self.outcome(value).errors)


if __name__ == '__main__':
    unittest.main()
