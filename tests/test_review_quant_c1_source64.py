"""CPU fixtures exercise the offline reader, never Ascend numerical behavior."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import test_quant_group_c1 as fixtures


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'experiments/c2c3_closure_20261117/review_quant_c1_source64.py'
spec = importlib.util.spec_from_file_location('offline_quant_review', SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class QuantOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = fixtures.QuantGroupC1Tests('test_existing_provider_runs_single_site_and_keeps_full_unknown_denominator')
        fixture.setUp()
        try:
            fixture.binding.dispatch_quant = True
            fixture.enterContext(patch.object(fixtures.c1, 'dynamic_quant_operator',
                return_value=fixtures.torch.ops.cpu_quant_group_c1.quant.default))
            client = fixture.host_client()
            for _ in range(7):
                fixture.phase(collect=False)
                fixture.dispatch('advance', {'collect': True})
                fixture.dispatch('query')
            cls.serving = fixture.dispatch('finish')['serving']
            cls.inventory = copy.deepcopy(fixture.session.inventory.report())
            cls.discovery = copy.deepcopy(fixture.session.discovery)
            fixture.dispatch('unregister')
            client.detach()
            cls.host = client.snapshot()
            worker = cls.host['worker']
            cls.result = dict(serving_final=cls.serving, deployment={'worker': {'worker': worker}},
                worker_after={'worker': worker, 'quarantined': False}, remove={'status': 'DEPLOYMENT_REMOVED'},
                healthy_before=[[3, 4]]*8, healthy_after=[[3, 4]]*8, phases=[])
            for row in cls.serving['rows']:
                ids = row['decode']['decode']['request_ids']
                cls.result['phases'].append(dict(phase=row['phase'], enqueued=ids,
                    mapping=[dict(internal_request_id=v, external_request_id='external-' + v,
                        request_state_object_id=1000+i) for i, v in enumerate(ids)], output_tokens=[[3, 4]]*8))
        finally:
            fixture.doCleanups()

    def audit(self, serving=None, inventory=None, discovery=None):
        return review.evidence(serving or self.serving, inventory or self.inventory,
            discovery or self.discovery, natural=False, count=2)

    def codes(self, report):
        return {row['code'] for row in report['issues']}

    def test_actual_cpu_public_protocol_and_saved_native_records_reconcile(self):
        result = self.audit()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['denominator']['parents'], 14)
        self.assertEqual(result['denominator']['retained_unresolved'], [1]*7)
        self.assertFalse(result['complete_arrays_independently_rehashed'])
        host = review.Review()
        review.host_review(host, self.result, self.host, self.serving)
        self.assertFalse(host.errors, host.summary())

    def test_success_flags_are_not_evidence(self):
        s = copy.deepcopy(self.serving)
        s.update(status='FAILED_WITH_EVIDENCE', c1=False, functional_effect_proven=False, native_qualified_targets=0)
        self.assertTrue(self.audit(serving=s)['consistent'])

    def test_natural_denominator_is_not_replaced_by_two_cpu_calls(self):
        report = review.evidence(self.serving, self.inventory, self.discovery, natural=False, count=135)
        self.assertIn('FULL_NATURAL_PARENT_DENOMINATOR', self.codes(report))

    def test_unresolved_native_candidate_cannot_be_dropped(self):
        i = copy.deepcopy(self.inventory)
        i['native_forwards'][5]['candidates'].pop()
        self.assertIn('FULL_NATURAL_PARENT_DENOMINATOR', self.codes(self.audit(inventory=i)))

    def test_unclassified_original_call_cannot_be_fabricated(self):
        rows = copy.deepcopy(self.inventory['public_calls'][0]['independent_native']['native_events'])
        rows[4]['native']['original_calls'] = 0
        a = review.Review(); review.passthrough(a, rows, 'non_target')
        self.assertIn('ACTUAL_NATIVE_ORIGINAL_CALL_AND_FORMALS', a.errors)

    def test_census_counters_and_window_gaps_cannot_hide_a_parent(self):
        for change in ('sequence', 'counter'):
            i = copy.deepcopy(self.inventory)
            f = i['native_forwards'][-1]
            if change == 'sequence': f['native_events'].pop(0)
            else: f['after']['census']['model_parents'] += 1
            with self.subTest(change=change): self.assertFalse(self.audit(inventory=i)['consistent'])

    def test_changed_automatic_target_is_not_accepted_with_resigned_identity(self):
        s = copy.deepcopy(self.serving)
        s['selected_site']['identity']['projection']['weight_object_identity'] += 1
        s['selected_site']['sha256'] = review.digest(s['selected_site']['identity'])
        self.assertIn('AUTOMATIC_SELECTION_IDENTITY_AND_PREFIXES', self.codes(self.audit(serving=s)))

    def packet_audit(self, mutate):
        p = copy.deepcopy(self.serving['receipts'][-1]); mutate(p)
        # Re-sign the JSON so tests exercise semantics beyond an artifact checksum.
        p['artifact_sha256'] = review.digest({k:v for k, v in p.items() if k != 'artifact_sha256'})
        a = review.Review()
        a.section('receipt', review.receipt, a, p, self.serving['owner'], self.serving['generation'],
            self.serving['selected_site']['sha256'], self.inventory['lifetime'], 'fault')
        return a

    def test_native_reference_effect_mutations_are_refused(self):
        def wrong_role(p): p['raw_events'][1]['binding']['role'] = 2
        def wrong_owner(p): p['lease']['owner_hi'] += 1
        def old_executor(p): p['raw_events'][-1]['native']['executor'] += 1
        def old_input(p): p['completion']['core']['reference']['plan']['input']['address'] += 64
        def no_effect(p): p['completion']['core'].update(changed_quant_bytes=0, changed_scale_bytes=0)
        def no_readback(p): p['completion']['core']['reference']['readback'] = 0
        def late_restore(p): p['completion']['core']['public_bytes_restored'] = 1
        def alias_buffers(p):
            p['completion']['core']['effect']['plan']['quant']['address'] = p['completion']['core']['reference']['plan']['quant']['address']
        def missing_helper(p): p['raw_events'].pop(1)
        def repeated_original(p): p['completion']['core']['effect']['execute_original_calls'] = 2
        def wrong_domain(p): p['completion']['core']['effect']['plan']['input']['rows'] = 8
        for mutate in (wrong_role, wrong_owner, old_executor, old_input, no_effect, no_readback,
                       late_restore, alias_buffers, missing_helper, repeated_original, wrong_domain):
            with self.subTest(mutation=mutate.__name__):
                self.assertTrue(self.packet_audit(mutate).errors)

    def test_current_proof_request_write_epoch_and_errors_refuse(self):
        for field in ('request', 'epoch', 'error', 'opaque', 'digest'):
            s = copy.deepcopy(self.serving)
            p = s['receipts'][-1]; fresh = p['admission_row_proof']
            if field == 'request': fresh['actual_rows']['request_ids'][0] = 'old-request'
            elif field == 'epoch': fresh['observed_write_epoch'] += 1
            elif field == 'error': fresh['observation_prefix']['source_errors'] = 1
            elif field == 'opaque': fresh['native_input_dispatch']['actual_input']['object_id'] += 1
            else: p['lease']['actual_row_proof'][0] += 1
            p['artifact_sha256'] = review.digest({k:v for k, v in p.items() if k != 'artifact_sha256'})
            with self.subTest(field=field):
                self.assertFalse(self.audit(serving=s)['consistent'])

    def test_consumer_same_shape_clone_is_not_same_object(self):
        i = copy.deepcopy(self.inventory)
        f = i['forwards'][5]; qmm = i['public_calls'][f['projections'][0]['api_call_indices'][1]]
        qmm['formals']['x1']['object_id'] += 1
        self.assertIn('CONSUMER_READS_ACTUAL_Q_AND_SCALE_OBJECTS', self.codes(self.audit(inventory=i)))

    def test_later_unknown_row_does_not_invalidate_closed_selected_prefix(self):
        self.assertTrue(self.audit()['consistent'])
        self.assertTrue(all(n['candidates'][1]['reasons'] for n in self.inventory['native_forwards']))

    def test_error_before_consumer_cannot_close_a_prefix(self):
        i = copy.deepcopy(self.inventory)
        i['forwards'][5]['projections'][0]['output_row_lineage']['observation_prefix']['errors'] = 1
        self.assertIn('HEALTHY_CURRENT_OBSERVATION_PREFIX', self.codes(self.audit(inventory=i)))

    def test_off_model_hash_cannot_be_replaced_by_unchanged_tokens(self):
        s = copy.deepcopy(self.serving); s['rows'][-1]['decode']['logits_sha256'] = '0'*64
        self.assertIn('FULL_HEALTHY_MODEL_OUTPUT_RESTORED', self.codes(self.audit(serving=s)))

    def test_retained_resources_or_missing_drain_refuse(self):
        for key in ('hook', 'bytes', 'sync', 'released'):
            s = copy.deepcopy(self.serving)
            if key == 'hook': s['owned_hooks'] = 1
            elif key == 'bytes': s['native_recovery'][0]['before_remove']['core']['host_bytes'] = 64
            elif key == 'sync': s['native_recovery'][0]['actual_global_device_synchronize'] = False
            else: s['receipts'][-1]['completion']['core']['released'] = 0
            with self.subTest(key=key): self.assertFalse(self.audit(serving=s)['consistent'])

    def test_uncertain_reply_foreign_ticket_and_retained_host_refuse(self):
        for key in ('uncertain', 'generation', 'retained', 'active_lease', 'missing_collect'):
            h = copy.deepcopy(self.host)
            if key == 'uncertain': h['history'][20]['response']['status'] = 'RPC_UNCERTAIN'
            elif key == 'generation': h['history'][20]['response']['ticket']['generation'] += 1
            elif key == 'retained': h['ownership_retained'] = True
            elif key == 'active_lease': h['history'][20]['response']['provider_state']['leases'] = 1
            else: h['history'].pop(20)
            a = review.Review(); a.section('host', review.host_review, a, self.result, h, self.serving)
            with self.subTest(key=key): self.assertTrue(a.errors)

    def test_forged_input_digest_is_recomputed(self):
        # Natural schema is separately asserted; a changed hash still has a specific refusal.
        i = copy.deepcopy(self.inventory); i['forwards'][-1]['input_sha256'] = '0'*64
        s = copy.deepcopy(self.serving)
        for row, f in zip(s['rows'], i['forwards']):
            mapping = row['decode']['decode']; mapping.update(counts=[1]*8, row_request_indices=list(range(8)),
                request_count=8, token_row_count=8, scheduler_capacity=8, pure_decode=True)
            f['row_mapping'] = copy.deepcopy(mapping)
        result = review.evidence(s, i, self.discovery, natural=True, count=2)
        self.assertIn('RECOMPUTED_ACTUAL_INPUT_HASH', self.codes(result))

    def test_offline_import_does_not_load_framework_injector_or_dso(self):
        program = '''import importlib.util,sys
spec=importlib.util.spec_from_file_location('reader',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert not any(n=='torch' or n.startswith(('torch.', 'torch_npu', 'demoncat.', 'fault_injection.')) for n in sys.modules)
print('STDLIB_ONLY')'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(SCRIPT)], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'STDLIB_ONLY')

    def test_cli_wrong_original_hash_reports_refusal_without_importing_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in review.FILES.values():
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
            result = subprocess.run([sys.executable, '-I', '-B', str(SCRIPT), '--evidence-root', folder,
                '--source-root', folder, '--native-root', folder], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report['source_bound_single_site_c1_receipts_reconciled'])
            self.assertEqual(report['e5'], 'NOT_ASSESSED')
            self.assertEqual(report['artifacts']['issues'][0]['code'], 'ORIGINAL_ARTIFACT_SHA256')

    def test_source_and_native_artifacts_are_rehashed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / 'source'; native = root / 'native'
            hashes = {}
            for name in review.SOURCES:
                p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixture:' + name)
                hashes[name] = review.file_sha(p)
            manifest = source / 'source_manifest.json'
            manifest.write_text(json.dumps(dict(base_commit=review.SOURCE_COMMIT,
                files=[dict(path=k, sha256=v) for k, v in hashes.items()])))
            files = {}; sources = {}
            for name in review.SOURCES[-3:]:
                p = native / Path(name).name; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((source / name).read_bytes()); files[p.name] = review.file_sha(p); sources[name] = hashes[name]
            lib = 'runtime/vendor/op_api/lib/libcust_opapi.so'; p = native / lib
            p.parent.mkdir(parents=True); p.write_bytes(b'EXPLICIT_NOT_A_LIBRARY'); files[lib] = review.file_sha(p)
            build = dict(schema='demoncat.quant.group_deployment.v1', cann='8.5.1', actual_npu_calls=0,
                legacy_c1_library_present=False, files=files, sources=sources, library=lib,
                sdk_identity={'abi_sizes': review.ABI}, actual_image_id='CPU_ONLY', source_commit='d'*40)
            build_path = native / 'build.json'; build_path.write_text(json.dumps(build))
            serving = {'native_build':dict(build_sha256=review.file_sha(build_path), sdk_abi_sizes=review.ABI,
                actual_image_id='CPU_ONLY', source_commit='d'*40, library_sha256=files[lib])}
            result = {'deployment':{'sources':[dict(path='/installed/' + (name[4:] if name.startswith('src/') else name),
                sha256=hashes[name]) for name in review.SETUP]}}
            with patch.object(review, 'SOURCE_MANIFEST', review.file_sha(manifest)), patch.object(review, 'BUILD_SHA', review.file_sha(build_path)):
                a = review.Review(); review.provenance(a, result, serving, source, native)
                self.assertFalse(a.errors, a.summary())
                (native / lib).write_bytes(b'CHANGED')
                a = review.Review(); review.provenance(a, result, serving, source, native)
                self.assertIn('NATIVE_BUILD_FILE_REHASH', a.errors)
                (source / review.SOURCES[0]).write_text('CHANGED')
                a = review.Review(); review.provenance(a, result, serving, source, native)
                self.assertIn('SOURCE_FILE_REHASH', a.errors)


if __name__ == '__main__':
    unittest.main()
