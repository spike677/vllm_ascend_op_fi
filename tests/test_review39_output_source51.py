"""Actual CPU dispatcher evidence and tampered-artifact negative contracts.

These fixtures are CPU contracts, never pretrained/NPU/Worker qualification.
The reviewer itself imports no framework and executes no workload.
"""
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from fault_injection.coverage_session import CoverageScope
from fault_injection.deployments.postwrite import PostwriteProvider
from fault_injection.deployments.postwrite_coverage import ProjectionCoverage
from tests.test_postwrite_providers import CpuTransport, Projection
from experiments.c2c3_postwrite_20261117 import review39_output_source51 as review


class ParallelProjections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Projection(), Projection(), Projection(consumer=False)])

    def forward(self, value):
        return torch.cat([layer(value) for layer in self.layers], dim=-1)


def fixture(arm='fault', c3=False, max_live=2):
    transport = CpuTransport()
    transport.sha256 = 'f'*64
    transport.lease_capacity = max_live
    transport.lease_byte_capacity = 1024**2
    provider = PostwriteProvider(ParallelProjections(), transport, '39.output_proxy')
    value = torch.ones(1, 4, dtype=torch.bfloat16)
    c1 = {'qualification': provider.qualify(value)}
    for label, phase in (('lossless', 'baseline'), ('fault', 'fault'), ('off', 'off')):
        _, c1[label] = provider.run(phase, value)
    environment = dict(native_library_sha256=transport.sha256, build_fingerprint='b'*64)
    source = {name: review.file_sha(Path(__file__).resolve().parents[1]/name) for name in review.SOURCES}
    scope = CoverageScope('39.output_proxy', 'a'*64, review.digest(environment), review.digest(source), 'cpu_contract')
    group = ProjectionCoverage(provider, scope, max_live_leases=max_live)
    result = dict(schema='demoncat.postwrite39.natural_coverage_probe.v1', arm=arm,
        mechanism='39.output_proxy', evidence_kind='natural_pretrained_model', worker_extension=False,
        public_qualification_granted=False, c1=c1, environment=environment, source_sha256=source,
        build_fingerprint='b'*64, weights=dict(weights_sha256='d'*64), model_source=dict(source_sha256='e'*64),
        discovery=dict(eligible_targets=len(group.eligible), projection_calls=len(group.call_order),
            structural_modules=len(group.module_universe), structural_noncall_inventory=group.structural_inventory))
    result['c2'] = group.run('once', value, baseline_control=arm == 'baseline')
    if c3:
        result['c3'] = group.run('every_match', value)
    _, result['final_off'] = provider.run('off', value)
    result['live_leases'] = provider.query()['leases']
    result['independent_discovery'] = group.discovery_diagnostics
    return json.loads(json.dumps(result, allow_nan=False))


class Review39Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fault = fixture(c3=True)
        cls.baseline = fixture('baseline')
        cls.refused = fixture(c3=True, max_live=1)

    def reviewed(self, document):
        return review.review_document(document, expected_targets=2, cpu_contract=True)

    def test_actual_cpu_fault_c2_and_c3_reconcile(self):
        result = self.reviewed(self.fault)
        self.assertTrue(result['consistent'], result)
        for name in ('c2', 'c3'):
            self.assertTrue(result['groups'][name]['consistent'], result['groups'][name])

    def test_actual_baseline_control_c2_reconciles(self):
        result = self.reviewed(self.baseline)
        self.assertTrue(result['consistent'], result)
        self.assertTrue(result['groups']['c2']['consistent'], result['groups']['c2'])

    def codes(self, result, group='c2'):
        return {row['code'] for row in result['groups'][group]['issues']}

    def tampered(self, mutation):
        document = deepcopy(self.fault)
        mutation(document)
        return self.reviewed(document)

    def test_control_process_has_c1_fault_but_zero_c2_faults(self):
        result = self.reviewed(self.baseline)
        self.assertTrue(result['process_contains_c1_fault'])
        self.assertEqual(result['c1_fault_invocations'], 1)
        self.assertEqual(result['groups']['c2']['recomputed_hits'], 0)
        self.assertFalse(self.baseline['c2']['contract_satisfied'])
        self.assertTrue(result['c2_evidence_reconciled'])

    def test_c3_refusal_does_not_erase_c2_or_successful_recovery(self):
        result = self.reviewed(self.refused)
        self.assertTrue(result['c2_evidence_reconciled'])
        self.assertFalse(result['groups']['c3']['consistent'])
        self.assertTrue(result['groups']['c3']['recovery']['consistent'], result['groups']['c3'])
        self.assertIn('CONCURRENCY_BUDGET_EXHAUSTED', result['groups']['c3']['stored_provider_failure'])

    def test_success_booleans_are_not_authority(self):
        document = deepcopy(self.fault)
        document['c2']['contract_satisfied'] = False
        # Exclude C3: its signed report hash must, correctly, notice this edit.
        document.pop('c3')
        result = self.reviewed(document)
        self.assertTrue(result['groups']['c2']['consistent'])
        self.assertFalse(result['groups']['c2']['contract_boolean_used_as_verdict'])

    def test_missing_target_cannot_be_replaced_by_claimed_140_totals(self):
        def change(document):
            document['c2']['execution_artifacts'].pop()
        result = self.tampered(change)
        self.assertIn('PER_TARGET_EXECUTION_DISTRIBUTION', self.codes(result))
        self.assertIn('ADMITTED_CALL_WITHOUT_EXECUTION_ARTIFACT', self.codes(result))

    def test_unsupported_actual_projection_is_not_dropped(self):
        def change(document):
            document['c2']['discovery']['observed_target_ids'].pop()
        self.assertIn('DISCOVERY_DROPPED_UNSUPPORTED_TARGETS', self.codes(self.tampered(change)))

    def test_wrong_request_owner_and_generation_rejected(self):
        for name, value in (('request_ids', ['unrelated']), ('owner', 'another'), ('generation', 22)):
            with self.subTest(name=name):
                result = self.tampered(lambda d: d['c2']['calls'][0].__setitem__(name, value))
                self.assertIn('ACTUAL_CALL_TARGET_REQUEST_OR_OWNER_JOIN', self.codes(result))

    def test_same_shape_other_module_identity_rejected(self):
        def change(document):
            document['c2']['forward_artifacts'][1]['inventory']['calls'][0]['module_object_identity'] += 1
        self.assertIn('ACTUAL_PROJECTION_IDENTITY_OR_ORDINAL', self.codes(self.tampered(change)))

    def test_duplicate_actual_call_id_rejected(self):
        def change(document):
            document['c2']['calls'][1]['call_id'] = document['c2']['calls'][0]['call_id']
        self.assertIn('ACTUAL_CALL_TARGET_REQUEST_OR_OWNER_JOIN', self.codes(self.tampered(change)))

    @staticmethod
    def rehash_effect(document, index=0):
        effect = document['c2']['execution_artifacts'][index]
        call = next(row for row in document['c2']['calls'] if row['call_id'] == effect['call_id'])
        call['receipt']['execution_sha256'] = review.digest(effect)

    def test_same_shape_consumer_clone_rejected_after_artifact_rehash(self):
        def change(document):
            effect = document['c2']['execution_artifacts'][0]
            effect['consumer_observations'][0]['operand_views'][0]['storage_identity'] += 1
            self.rehash_effect(document)
        result = self.tampered(change)
        self.assertIn('SELECTED_PRODUCER_ELEMENTS_NOT_READ_BY_CONSUMER_VIEW', self.codes(result))
        self.assertNotIn('EXECUTION_OR_EFFECT_ARTIFACT_HASH', self.codes(result))

    def test_disjoint_consumer_slice_of_same_storage_rejected(self):
        def change(document):
            view = document['c2']['execution_artifacts'][0]['consumer_observations'][0]['operand_views'][0]
            view.update(shape=[1,3], storage_offset=1, logical_bytes=6, data_ptr=view['data_ptr']+2)
            self.rehash_effect(document)
        self.assertIn('SELECTED_PRODUCER_ELEMENTS_NOT_READ_BY_CONSUMER_VIEW', self.codes(self.tampered(change)))

    def test_no_actual_consumer_cannot_claim_hit(self):
        def change(document):
            effect = document['c2']['execution_artifacts'][0]
            effect['consumer_observations'] = []; effect['consumers'] = []
            self.rehash_effect(document)
        self.assertIn('CONSUMER_CENSUS', self.codes(self.tampered(change)))

    def test_fault_effect_zero_rejected_even_when_receipt_hash_matches(self):
        def change(document):
            document['c2']['execution_artifacts'][0]['changed_elements'] = 0
            self.rehash_effect(document)
        self.assertIn('PER_TARGET_NATIVE_EFFECT_OR_RECOVERY', self.codes(self.tampered(change)))

    def test_execution_hash_mismatch_rejected(self):
        result = self.tampered(lambda d: d['c2']['execution_artifacts'][0].__setitem__('request_id', 'fake'))
        self.assertIn('EXECUTION_OR_EFFECT_ARTIFACT_HASH', self.codes(result))

    def test_complete_output_off_difference_rejected(self):
        def change(document):
            document['c2']['forward_artifacts'][-1]['output_sha256'] = '0'*64
        result = self.tampered(change)
        self.assertIn('LOSSLESS_CONTROL_OR_OFF_FULL_OUTPUT_CHANGED', self.codes(result))
        self.assertFalse(result['groups']['c2']['recovery']['consistent'])

    def test_nonzero_hooks_lease_or_inflight_cannot_release(self):
        changes = [lambda d: d['c2']['recovery'].__setitem__('inflight', 1),
                   lambda d: d['c2']['recovery']['workers'][0].__setitem__('owned_hooks', 1),
                   lambda d: d['c2']['recovery']['targets'][0].__setitem__('released', False)]
        for change in changes:
            with self.subTest(change=change):
                result = self.tampered(change)
                self.assertFalse(result['groups']['c2']['recovery']['consistent'])
                self.assertFalse(result['c2_evidence_reconciled'])

    def test_recovery_proof_hash_and_owner_are_independently_checked(self):
        for field, value in (('proof_sha256', '0'*64), ('off_output_sha256', '1'*64)):
            with self.subTest(field=field):
                result = self.tampered(lambda d: d['c2']['recovery']['targets'][0].__setitem__(field, value))
                self.assertFalse(result['groups']['c2']['recovery']['consistent'])

    def test_c1_wrong_contraction_geometry_and_output_storage_rejected(self):
        changes = [lambda d: d['c1']['fault']['actual_target']['producer_proof']['operands']['self']['shape'].__setitem__(1, 5),
                   lambda d: d['c1']['fault']['actual_target']['storage_identity'].__setitem__(1, 123)]
        for change in changes:
            with self.subTest(change=change):
                result = self.tampered(change)
                self.assertFalse(result['c1_final_off_consistent'])
                self.assertIn('C1_ORIGINAL_PRODUCER_RETURN_BINDING', {row['code'] for row in result['issues']})

    def test_c3_use_is_recorded_but_missing_issuance_is_not_recreated(self):
        proof = self.reviewed(self.fault)['c3_live_proof']
        self.assertTrue(proof['use_receipt'])
        self.assertTrue(proof['use_fields_consistent'])
        self.assertFalse(proof['issuance_receipt'])
        self.assertFalse(proof['same_live_object_receipted'])

    def test_c3_wrong_live_proof_object_or_scope_rejected(self):
        for field in ('owner', 'generation', 'report_sha256'):
            with self.subTest(field=field):
                document = deepcopy(self.fault)
                document['c3']['live_c2_proof']['consumed']['proof'][field] = 'unrelated'
                proof = self.reviewed(document)['c3_live_proof']
                self.assertFalse(proof['use_fields_consistent'])
                self.assertFalse(proof['same_live_object_receipted'])

    def test_unknown_final_worker_or_owner_rejected(self):
        result = self.tampered(lambda d: d['final_off']['actual_target'].__setitem__('owner', 99))
        self.assertFalse(result['c1_final_off_consistent'])

    def test_cli_import_is_framework_free_and_runtime_execution_is_refused(self):
        code = ('import sys; from experiments.c2c3_postwrite_20261117 import review39_output_source51 as r; '
                'assert not any(k in sys.modules for k in ("torch","torch_npu","vllm")); '
                'sys.addaudithook(r.forbid_runtime); import torch')
        process = subprocess.run([sys.executable, '-B', '-c', code], text=True, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('RUNTIME_IMPORT_REFUSED:torch', process.stderr)
        for name in ('ctypes.dlopen', 'subprocess.Popen', 'os.system', 'socket.connect'):
            with self.assertRaisesRegex(RuntimeError, 'CPU_READ_ONLY_REVIEW'):
                review.forbid_runtime(name, ())

    def test_strict_input_and_explicit_path_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'result.json'
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON_KEY'): review.read(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(ValueError, 'NONFINITE_JSON'): review.read(path)
            with self.assertRaisesRegex(ValueError, 'EXPLICIT_PATH_MAPPING_MISSING'):
                review.resolve_recorded('/unknown/file', [])
            with self.assertRaisesRegex(ValueError, 'RECORDED_PATH_TRAVERSAL'):
                review.resolve_recorded('/source/../file', [])

    def test_frozen_source_native_and_external_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source_root = root/'source'; model = root/'model'
            source_root.mkdir(); model.mkdir()
            files = []
            for name in (*review.SOURCES, *review.SUPPORTING_SOURCES):
                path = source_root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(('CPU contract source '+name).encode())
                files.append(dict(path=name, sha256=review.file_sha(path), bytes=path.stat().st_size))
            manifest = source_root/'source_manifest.json'
            manifest.write_text(json.dumps(dict(base_commit=review.SOURCE_COMMIT, files=files)))
            native = root/'libdcat_postwrite.so'; native.write_bytes(b'CPU contract: never load this file')
            (model/'model.safetensors').write_bytes(b'CPU contract weight bytes')
            (model/'config.json').write_text('{}')
            (model/'model.py').write_text('class Fixture: pass\n')
            weights = dict(format='safetensors', files={'model.safetensors': dict(bytes=(model/'model.safetensors').stat().st_size,
                sha256=review.file_sha(model/'model.safetensors'))}, index=None, config_sha256=review.file_sha(model/'config.json'),
                loader_selection='from_pretrained(use_safetensors=True, local_files_only=True)')
            model_source = dict(files={'/model/model.py':review.file_sha(model/'model.py')},
                classes=[dict(class_name='Fixture', source='/model/model.py')])
            document = deepcopy(self.fault)
            document['weights'] = dict(path='/model', manifest=weights, weights_sha256=review.digest(weights))
            document['model_source'] = dict(manifest=model_source, source_sha256=review.digest(model_source))
            document['source_sha256'] = {row['path']:row['sha256'] for row in files if row['path'] in review.SOURCES}
            document['environment']['native_library_sha256'] = review.file_sha(native)
            scope = document['c2']['scope']
            scope.update(evidence_kind='natural_pretrained_model', model_weights_sha256=review.digest(weights),
                model_source_sha256=review.digest(model_source), environment_sha256=review.digest(document['environment']),
                implementation_sha256=review.digest(document['source_sha256']), workload_sha256=review.digest(dict(
                    prompt=review.DEFAULT_PROMPT, input_sha256=document['c1']['qualification']['input_sha256'], max_length=32, use_cache=False)))
            mappings = [(PurePosixPath('/model'), model)]
            with patch.object(review, 'MANIFEST_SHA', review.file_sha(manifest)), patch.object(review, 'NATIVE_SHA', review.file_sha(native)):
                result = review.provenance(document, source_root, mappings, native)
                self.assertTrue(result['consistent'], result)
                self.assertEqual(result['external_files_rehashed'], 3)
                self.assertTrue(result['external_artifacts_all_available'])
                missing = review.provenance(document, source_root, [], native)
                self.assertTrue(missing['consistent'])
                self.assertFalse(missing['external_artifacts_all_available'])
                self.assertEqual(len(missing['external_files_unavailable']), 3)
                original = native.read_bytes(); native.write_bytes(b'changed')
                result = review.provenance(document, source_root, mappings, native)
                self.assertIn('EXACT_NATIVE_LIBRARY_BYTES', {row['code'] for row in result['issues']})
                native.write_bytes(original)
                (source_root/review.SUPPORTING_SOURCES[0]).write_bytes(b'unknown collector helper')
                (model/'model.safetensors').write_bytes(b'changed weights')
                result = review.provenance(document, source_root, mappings, native)
                codes = {row['code'] for row in result['issues']}
                self.assertIn('FROZEN_IMPLEMENTATION_BYTES', codes)
                self.assertIn('EXTERNAL_ARTIFACT_BYTES_CHANGED', codes)
                self.assertIn('WEIGHTS_FILE_SIZE_CHANGED', codes)

    def test_logical_universe_joins_distinct_process_object_ids(self):
        fault = self.reviewed(self.fault)['groups']['c2']
        baseline = self.reviewed(self.baseline)['groups']['c2']
        self.assertNotEqual(fault['target_ids'], baseline['target_ids'])
        self.assertEqual(fault['logical_universe_sha256'], baseline['logical_universe_sha256'])


if __name__ == '__main__':
    unittest.main()
