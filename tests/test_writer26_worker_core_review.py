"""Synthetic saved originals only. No framework, Native library or device."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.c2c3_closure_20261117 import review26_worker_core_result as review
from test_native26_single_source54_review import Fixture as NativeFixture, audit, counters, native_state, sha


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.native = NativeFixture(root)
        self.inv = deepcopy(self.native.inventory)
        self.inv.update(c1=False, c2=False, c3=False, recovered=True, healthy_repeated=True)
        self.inv['single_site_c1'].update(evidence_complete=True, all_eligible_qualification=False,
            serializable_authority=False, fallback_attempted=False, selected_native_admissions=5,
            actual_fault_hits=1, complete_native_counts=[3]*8)
        self.refs, self.collected, self.raw, self.history = [], [], [], []
        self.owner, self.worker = self.native.owner, self.native.worker
        self.ticket = dict(attach_nonce='fixture-attach', tenant_id='fixture-tenant', worker=self.worker,
            mechanism='26.residual_stomp', implementation_version='1', owner='public-host-owner', generation=1)
        self.attachment = {k:self.ticket[k] for k in ('attach_nonce', 'tenant_id', 'worker')}
        self.sources = {name:sha(name) for name in review.REQUIRED_SOURCES}
        self.setup = dict(worker=self.worker, binding=dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED',
            mechanism='26.residual_stomp', stage='c1_single_effects', hooks_installed=0,
            native_owner_installed=False, workers_created=0, qualification_granted=False, c2=False, c3=False),
            loaded_sources={name:dict(path='/installed/'+(name[4:] if name.startswith('src/') else name),
                sha256=self.sources[name]) for name in review.LOADED})
        self.capabilities = dict(status='CAPABILITIES', worker=self.worker,
            providers=[dict(mechanism='26.residual_stomp', implementation_version='1')])
        self.d = dict(schema='demoncat.writer26.worker_core_probe.v1', status=review.SUCCESS,
            mechanism='26.residual_stomp', stage='c1_single_effects', scope=audit.SCOPE,
            c2=False, c3=False, qualification_granted=False, previous_direct_entry_qualification_inherited=False,
            E5='UNADAPTED_COLLECTOR_BOUNDARY', manifest_sha256=audit.BUILD_SHA,
            requests=[], actual_prompt_token_ids=self.native.tokens, sources=self.sources, setup=self.setup,
            healthy_before=dict(tokens=[185], text='healthy'), worker=self.worker, ownership_retained=False,
            deployment_removed=dict(status='WRITER26_CORE_DEPLOYMENT_REMOVED'))

    def add_provenance(self):
        """Hash synthetic fixture bytes, never present these as an SDK build."""
        self.source_root, self.native_root = self.root/'source', self.root/'native'
        self.source_root.mkdir(); self.native_root.mkdir()
        compile_sources = ('fault_injection/native_adapters/writer26_group_core.h',
                           'fault_injection/native_adapters/writer26_group_runtime.cpp')
        paths = sorted(set(self.sources) | set(compile_sources))
        entries = []
        for name in paths:
            path = self.source_root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            entries.append(dict(path=name, sha256=sha(name), bytes=len(name.encode())))
        self.commit = '1'*40
        manifest = self.source_root/'source_manifest.json'
        manifest.write_text(json.dumps(dict(base_commit=self.commit, files=entries)))
        self.source_sha = review.file_sha(manifest)
        native_sources = {name:sha(name) for name in compile_sources}
        receipt = dict(schema='demoncat.writer26.group_build.v2', actual_npu_calls=0,
            device_mounts=[], sources=native_sources)
        path = self.native_root/'build-receipt.json'; path.write_text(json.dumps(receipt))
        receipt_sha = review.file_sha(path)
        files = {}
        for name in (*audit.SYMBOLS, 'metadata-array-ownership.json'):
            path = self.native_root/name; path.write_bytes(name.encode())
            files[name] = review.file_sha(path)
        build = dict(schema='demoncat.writer26.group_deployment.v2', legacy_writers_absent=True,
            actual_npu_calls=0, files=files, sources=native_sources, build_receipt_sha256=receipt_sha,
            metadata_array_ownership=dict(schema='demoncat.metadata_arrays.v1', state_version=1,
                images=audit.SYMBOLS, reset_supported=False, runtime_quiescent_readback_required=True,
                parent_receipt_sha256=files['metadata-array-ownership.json']))
        path = self.native_root/'build.json'; path.write_text(json.dumps(build))
        self.build_sha = review.file_sha(path)
        self.inv['lifetime'].update(native_inventory_sha256=files['libfault26_readonly_trace_abi0.so'],
            native_writer_sha256=files['libdcat26_group_v2.so'])
        self.inv['single_site_c1']['reference_lossless_off']['lifetime_sha256'] = audit.digest(self.inv['lifetime'])
        self.d.update(source_manifest=dict(sha256=self.source_sha, base_commit=self.commit),
            manifest_sha256=self.build_sha, metadata_startup=dict(schema='demoncat.writer26.startup_dependencies.v1',
                pid=1, manifest_sha256=self.build_sha,
                implementation_sha256=self.sources['fault_injection/deployments/writer26_loader.py'],
                model_runtime_imports_before=[], callback_installed=False, device_api_calls=0,
                global_symbol_scope_changed=False, metadata_arrays=counters(0)))
        weight = dict(format='safetensors', files={'model.safetensors':dict(bytes=12, sha256=sha('weights'))},
            index=None, config_sha256=sha('config'),
            loader_selection="vllm.LLM(load_format='safetensors', trust_remote_code=False)")
        self.d['weights'] = dict(path='/model', manifest=weight, weights_sha256=review.digest(weight))
        model = dict(files={'/external/source.py':sha('source')},
                     classes=[dict(class_name='fixture.Model', source='/external/source.py')])
        self.setup['model_source'] = dict(manifest=model, source_sha256=review.digest(model))
        self.d['metadata_startup'] = self.bind_build(self.d['metadata_startup'])

    def bind_build(self, value):
        return json.loads(json.dumps(value).replace(audit.BUILD_SHA, getattr(self, 'build_sha', audit.BUILD_SHA)))

    def save(self, name, value, *, absolute=False, **fields):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
        path.write_bytes(data)
        return dict(path='/evidence/'+name if absolute else path.name, sha256=hashlib.sha256(data).hexdigest(), **fields)

    def partial(self, count, *, cleanup=False):
        inv = deepcopy(self.inv)
        inv['rows'] = inv['rows'][:count]
        inv['outcomes'] = inv['outcomes'][:max(0, min(5, count-2))]
        inv['candidates'] = inv['candidates'] if count >= 2 else []
        inv['phase'] = review.PHASES[count] if count < 8 else 'complete'
        inv['native_state'] = (deepcopy(self.native.phases[count-1]['result']['native_state'])
            if count else native_state(self.native.owner_words, 0, 0))
        inv['metadata_arrays_latest'] = counters(count)
        inv['installed'] = count < 8
        inv['native_removed'] = count == 8
        inv['owned_hooks'] = 0 if count == 8 else 16
        inv['resolved_candidate_count'] = 2 if count >= 2 else 0
        inv['qualified_candidate_count'] = int(count >= 4)
        inv['last_native_receipt'] = inv['outcomes'][-1]['consumer_receipt'] if inv['outcomes'] else None
        inv['single_site_c1']['owned_root_logits_hooks'] = 0 if cleanup else 4
        return self.bind_build(inv)

    def serving(self, count, *, cleanup=False, expected=None, pending=False):
        return dict(installed=not cleanup, owned_hooks=0 if cleanup else 4, needs_collect=pending,
            expected=expected, current=None, root_running=False, logits_running=False, complete_requests=count)

    def state(self, count, *, cleanup=False, expected=None, pending=False, registered=True, reserved=True):
        inv = self.partial(count, cleanup=cleanup)
        evidence = review.project_inventory(inv)
        evidence['original_observations'] = dict(status=(
            'NOT_COLLECTED' if not self.refs else 'PREVIOUS_COLLECTION_CURRENT_STATE_NOT_CAPTURED'
            if pending or expected is not None else 'CURRENT_COLLECTED'), originals=deepcopy(self.refs))
        evidence['public_core'] = dict(worker_identity=self.worker, owner=self.owner, generation=1,
            native_owner=self.native.owner_words, native_generation=1, simulation=False,
            qualification_granted=False, c2=False, c3=False, owner_kind='runtime_lease_native_namespace',
            query_read_only=True, legacy_qualification_inherited=False)
        return dict(execution_boundary='native', native_loaded=True,
            native_registered=bool(inv['native_state']['core']['installed']), registered=registered,
            enabled=False, owner=self.owner, generation=1, inflight=0, leases=0,
            native=inv['native_state'], binding_reserved=reserved,
            serving=self.serving(count, cleanup=cleanup, expected=expected, pending=pending),
            evidence=evidence, replay_pending=not cleanup, replay_failed=False,
            recovery_validation_pending=not cleanup, query_read_only=True)

    def rpc(self, method, args, reply, *, action=None):
        item = dict(method=method, args=args, responses=[deepcopy(reply)])
        ref = self.save('probe/worker_core_rpc/'+str(len(self.raw)).zfill(3)+'.json', item, method=method)
        self.raw.append(item)
        if action:
            self.history.append(dict(action=action, response=deepcopy(reply)))
        self.d.setdefault('rpc_receipts', []).append(ref)
        return ref

    def action(self, name, state, settings=None):
        payload = deepcopy(self.ticket)
        if name == 'prepare':
            payload.pop('generation')
        if settings is not None:
            payload['settings'] = settings
        reply = dict(status='QUERY_OK' if name == 'query' else 'ACTION_COMPLETED', ticket=self.ticket,
            recovery_required=False, native=state, provider_state=state)
        if name != 'query':
            reply.update(action=name, provider=state)
        return self.rpc('demoncat_fault_query' if name == 'query' else 'demoncat_fault_rpc',
            [payload] if name == 'query' else [name, payload], reply, action=name)

    def capture(self, count, cleanup=False):
        action = 'cleanup' if cleanup else 'collect'
        value = dict(schema='demoncat.writer26.core_observations.v1', action=action,
            worker=self.worker, native_owner=self.owner, generation=1, simulation=False, qualification_granted=False,
            inventory=self.partial(count, cleanup=cleanup), serving_requests=deepcopy(self.native.observations[:count]),
            serving=self.serving(count, cleanup=cleanup))
        name = 'worker-core26-runtime/'+self.owner+'/runtime/core-evidence/'+str(len(self.refs)).zfill(3)+'-'+action+'.json'
        ref = self.save(name, value, absolute=True, action=action, complete_requests=count)
        ref['bytes'] = (self.root/name).stat().st_size
        self.refs.append(ref)
        self.collected.append(value)
        return ref

    def build(self):
        self.rpc('demoncat_worker_identity', [], dict(status='WORKER_IDENTIFIED', worker=self.worker, quarantined=False))
        self.rpc('demoncat_capabilities', [], self.capabilities)
        self.rpc('demoncat_attach', [dict(tenant_id=self.ticket['tenant_id'], worker=self.worker)],
                 dict(status='ATTACHED', ticket=self.attachment), action='attach')
        self.rpc('__main__.setup', [self.worker, '26.residual_stomp', self.native.tokens, self.d['manifest_sha256'], self.sources], self.setup)
        for name in ('prepare', 'register', 'check', 'enable'):
            state = self.state(0, registered=name != 'prepare', reserved=name != 'prepare')
            state['replay_pending'] = state['recovery_validation_pending'] = name == 'enable'
            self.action(name, state)
        for index in range(8):
            mapping = deepcopy(self.native.phases[index]['mapping'])
            rid = mapping['internal_request_id']
            self.action('advance', self.state(index, expected=rid), {'request_id':rid})
            pre = self.action('query', self.state(index+1, pending=True))
            ref = self.capture(index+1)
            self.action('advance', self.state(index+1), {'collect':True})
            self.d['requests'].append(dict(phase=review.PHASES[index], mapping=mapping,
                output=dict(tokens=[185], text='healthy'), pre_collect_query=pre, original_observations=ref))
        self.capture(8, cleanup=True)
        final = self.state(8, cleanup=True)
        self.d['c1'] = deepcopy(final['evidence'])
        self.action('finish', final)
        self.action('disable', self.state(8, cleanup=True))
        self.action('unregister', self.state(8, cleanup=True, registered=False, reserved=False))
        self.rpc('demoncat_worker_identity', [], dict(status='WORKER_IDENTIFIED', worker=self.worker, quarantined=False))
        self.d['post_unregister_health'] = dict(mapping=dict(internal_request_id='request-health',
            external_request_id='external-health', request_state_object_id=99,
            source='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR',
            proof_scope='DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION'),
            output=self.d['healthy_before'], worker=self.worker)
        self.rpc('demoncat_detach', [self.attachment], dict(status='DETACHED', recoveries=[]), action='detach')
        host = dict(schema='demoncat.worker_core.host_receipt.v1', status='DETACHED',
            config=dict(fault_id=26, variant='residual_stomp', implementation_version='1'), mechanism='26.residual_stomp',
            worker=self.worker, owner=self.ticket['owner'], tenant_id=self.ticket['tenant_id'],
            capability_enumeration=self.capabilities, history=self.history, response=self.history[-1]['response'],
            ownership_retained=False, attachment=None, ticket=None, census_ticket=None)
        self.d['host_receipt'] = self.save('probe/host_receipt.json', host)
        self.rpc('__main__.remove', [self.worker, '26.residual_stomp'], self.d['deployment_removed'])
        return self.d

    def review(self):
        return review.review_document(self.d, review.Originals(self.root), mechanism='26.residual_stomp', expected_count=3)

    def rewrite(self, ref, change, *, folder=''):
        name = ref['path'].removeprefix('/evidence/') if ref['path'].startswith('/evidence/') else folder+'/'+ref['path']
        path = self.root/name
        value = review.read(path); change(value)
        data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
        path.write_bytes(data)
        previous = ref['sha256']; current = hashlib.sha256(data).hexdigest()
        # Altered artifact metadata can be internally re-signed; semantic joins
        # must still refuse. Keep every affected reference consistent.
        def replace(value):
            if isinstance(value, dict):
                if value.get('sha256') == previous:
                    value['sha256'] = current
                    if 'bytes' in value:
                        value['bytes'] = len(data)
                for item in value.values():replace(item)
            elif isinstance(value, list):
                for item in value:replace(item)
        replace(self.d)
        return value


class PublicCoreReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.fixture = Fixture(self.folder.name)
        self.d = self.fixture.build()

    def test_new_public_originals_rebuild_native_effect_and_cleanup(self):
        result = self.fixture.review()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['details']['native']['natural_calls'], 24)
        self.assertEqual(result['details']['native']['fault_changed_residual_bytes'], 31)
        self.assertFalse(result['public_qualification_granted'] or result['c2'] or result['c3'] or result['e5'])

    def reject(self, code=None):
        result = self.fixture.review()
        self.assertFalse(result['consistent'], result)
        if code:
            self.assertIn(code, [v['code'] for v in result['issues']], result['issues'])

    def test_old_direct_result_or_c1_boolean_is_not_authority(self):
        self.d['schema'] = 'old-direct-entry'
        self.d['previous_direct_entry_qualification_inherited'] = True
        self.reject('NEW_PUBLIC_CORE_ATTEMPT_WITH_NO_HIDDEN_FAILURE')

    def test_missing_raw_rpc_is_not_rebuilt_from_host_history(self):
        self.d['rpc_receipts'].pop(10)
        self.reject('PUBLIC_HOST_EXACT_PROTOCOL_AND_ORIGINAL_RPC_JOIN')

    def test_duplicate_rpc_file_or_wrong_current_owner_refuses(self):
        self.d['rpc_receipts'][8] = deepcopy(self.d['rpc_receipts'][7])
        self.reject('COMPLETE_ORDERED_RPC_FILES')

    def test_missing_collect_artifact_sha_or_changed_bytes_refuse(self):
        ref = self.d['requests'][0]['original_observations']
        path = Path(self.folder.name)/ref['path'].removeprefix('/evidence/')
        path.write_bytes(path.read_bytes()+b' ')
        self.reject('SCHEMA_OR_ARTIFACT_UNAVAILABLE')

    def test_query_cannot_claim_an_uncaptured_request(self):
        self.d['requests'][0]['pre_collect_query'] = deepcopy(self.d['requests'][1]['pre_collect_query'])
        self.reject('READ_ONLY_QUERY_THEN_ONE_EXPLICIT_COLLECTION')

    def test_driver_request_or_post_unregister_worker_mismatch_refuses(self):
        self.d['post_unregister_health']['worker'] = dict(self.fixture.worker, pid=999)
        self.reject('ORDINARY_SAME_WORKER_HEALTH_AFTER_UNREGISTER')

    def test_fault_boolean_cannot_hide_nonzero_native_effect_missing(self):
        # Full artifacts and compact RPC must agree. Tampering only the claimed
        # completion summary is also a failure before any boolean can authorize.
        self.d['c1']['native_state']['core']['pending'] = 1
        self.reject('FINISH_TO_DRIVER_ORIGINAL_JOIN')

    def test_original_field_mutation_with_new_sha_does_not_bypass_complete_join(self):
        ref = self.d['requests'][2]['original_observations']
        self.fixture.rewrite(ref, lambda v: v['inventory']['outcomes'][0]['consumer_receipt'].update(consumer_calls=0))
        self.reject()

    def test_whole_native_denominator_is_required_even_single_site(self):
        result = review.review_document(self.d, review.Originals(self.folder.name), mechanism='26.residual_stomp', expected_count=2)
        self.assertFalse(result['consistent'])
        self.assertIn('REBUILD_COMPLETE_NATIVE_CANDIDATE_DENOMINATOR', [v['code'] for v in result['issues']])

    def rebuild(self):
        self.fixture.refs.clear(); self.fixture.collected.clear()
        self.fixture.raw.clear(); self.fixture.history.clear()
        self.fixture.d['rpc_receipts'] = []
        self.fixture.d['requests'] = []
        self.fixture.build()

    def change_outcome_receipts(self, **fields):
        outcome = self.fixture.inv['outcomes'][3]
        for key in ('observed_consumer_receipt', 'consumer_receipt', 'released_receipt'):
            outcome[key].update(fields)
        self.rebuild()

    def test_resigned_complete_originals_no_actual_residual_effect_refuse(self):
        self.change_outcome_receipts(changed_residual_bytes=0)
        self.reject('QMM_ACTUAL_32_BYTE_RESIDUAL_EFFECT')

    def test_resigned_complete_originals_missing_consumer_refuse(self):
        self.change_outcome_receipts(consumer_complete=0, consumer_calls=0)
        self.reject('NATIVE_EXECUTION_CONSUMPTION_RECOVERY')

    def test_resigned_complete_originals_restoration_or_wrong_generation_refuse(self):
        self.change_outcome_receipts(residual_restored_exact=0, generation=99)
        self.reject('NATIVE_RECEIPT_IDENTITY')

    def test_failed_current_public_tensor_lifetime_cannot_be_recovered_by_summary(self):
        self.fixture.inv['outcomes'][3]['public_consumer_lifetime'] = False
        self.rebuild()
        self.reject('FULL_RESIDUAL_RESTORATION_HASH')

    def test_nonfinite_fault_output_is_separate_from_actual_native_effect(self):
        self.fixture.inv['rows'][5]['output_finite'] = False
        self.fixture.native.observations[5]['model_output_finite'] = False
        self.rebuild()
        result = self.fixture.review()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertFalse(result['details']['requests']['fault_logits_finite'])
        self.assertEqual(result['details']['native']['fault_changed_residual_bytes'], 31)
        self.assertFalse(result['e5'])

    def test_unchanged_fault_logits_do_not_erase_proven_residual_write(self):
        baseline = self.fixture.inv['rows'][0]['output_sha256']
        self.fixture.inv['rows'][5]['output_sha256'] = baseline
        self.fixture.native.observations[5]['model_output_sha256'] = baseline
        self.rebuild()
        result = self.fixture.review()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertFalse(result['details']['requests']['whole_request_output_changed'])

    def test_nonfinite_off_still_refuses(self):
        self.fixture.inv['rows'][6]['output_finite'] = False
        self.fixture.native.observations[6]['model_output_finite'] = False
        self.rebuild()
        self.reject('FULL_HEALTHY_LOSSLESS_OFF_OUTPUT_EQUAL')

    def test_registered_owner_or_host_bytes_after_finish_refuse(self):
        self.fixture.native.phases[7]['result']['native_state']['core'].update(installed=1, host_bytes=32)
        self.rebuild()
        self.reject('COMPLETE_OWNED_CLEANUP_BEFORE_RELEASE')

    def test_original_path_traversal_or_wrong_mount_refuses(self):
        for name in ('../secret', '/other/secret', '/evidence/../secret', '/evidence/a\\b', '/evidence/C:secret'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                review.Originals(self.folder.name).load(dict(path=name, sha256='a'*64), absolute=True)

    def test_duplicate_keys_nonfinite_and_overflow_float_refuse(self):
        path = Path(self.folder.name)/'bad.json'
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):review.read(path)

    def test_cli_imports_only_stdlib_under_isolated_no_site_python(self):
        module = Path(review.__file__).resolve()
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(module), '--help'],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--expected-writers', result.stdout)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.fixture = Fixture(self.folder.name)
        self.fixture.add_provenance()
        self.d = self.fixture.build()

    def evaluate(self):
        f = self.fixture
        return review.provenance(self.d, f.collected[-1], f.source_root, f.native_root,
            source_commit=f.commit, source_sha=f.source_sha, build_sha=f.build_sha)

    def reject(self, code):
        result = self.evaluate()
        self.assertFalse(result['consistent'], result)
        self.assertIn(code, [v['code'] for v in result['issues']], result['issues'])

    def test_exact_source_build_and_loaded_original_join(self):
        result = self.evaluate()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertFalse(result['external_model_and_weight_files_rehashed'])

    def test_changed_installed_runtime_not_hidden_by_source_manifest(self):
        self.d['setup']['loaded_sources'][review.PREFIX+'runtime.py']['sha256'] = 'a'*64
        self.reject('ACTUAL_LOADED_SOURCE_SHA')

    def test_missing_core_module_and_source_file_rejected(self):
        self.d['setup']['loaded_sources'].pop(review.PREFIX+'serving26_core.py')
        self.reject('COMPLETE_ACTUAL_INSTALLED_MODULE_SET')

    def test_native_file_changed_and_retired_legacy_dso_refuse(self):
        f = self.fixture
        (f.native_root/'libdcat26_group_v2.so').write_bytes(b'changed')
        (f.native_root/'libfault26_qmm_output_underflow_abi0.so').write_bytes(b'legacy')
        self.reject('NATIVE_FILE_BYTES')

    def test_pre_model_bootstrap_cannot_claim_operator_work(self):
        self.d['metadata_startup']['device_api_calls'] = 1
        self.reject('PRE_MODEL_DEPENDENCY_BOOTSTRAP')

    def test_changed_weight_receipt_digest_or_loader_refuses(self):
        self.d['weights']['manifest']['loader_selection'] = 'constructed-model'
        self.reject('FULL_LOCAL_PRETRAINED_WEIGHT_RECEIPT')

    def test_nonstandard_unique_index_is_recorded_without_renaming_weights(self):
        value = self.d['weights']['manifest']
        value['files'] = {'quant_model_weight_w8a8_dynamic-01.safetensors':dict(bytes=12, sha256=sha('shard'))}
        value['index'] = dict(file='quant_model_weight_w8a8_dynamic.safetensors.index.json', sha256=sha('index'))
        self.d['weights']['weights_sha256'] = review.digest(value)
        result = self.evaluate()
        self.assertTrue(result['consistent'], result['issues'])

    def test_complete_cli_runs_without_site_or_framework_and_preserves_all_files(self):
        f = self.fixture
        f.save('probe/result.json', self.d)
        result_sha = review.file_sha(f.root/'probe/result.json')
        before = {str(p.relative_to(f.root)):review.file_sha(p) for p in f.root.rglob('*') if p.is_file()}
        command = [sys.executable, '-I', '-S', '-B', str(Path(review.__file__).resolve()),
            '--evidence-root', str(f.root), '--result-sha256', result_sha,
            '--source-root', str(f.source_root), '--source-commit', f.commit,
            '--source-manifest-sha256', f.source_sha, '--native-root', str(f.native_root),
            '--native-build-sha256', f.build_sha, '--mechanism', '26.residual_stomp', '--expected-writers', '3']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload['consistent'] and payload['read_only'])
        self.assertFalse(payload['public_qualification_granted'] or payload['c2'] or payload['c3'] or payload['e5'])
        self.assertEqual(before, {str(p.relative_to(f.root)):review.file_sha(p) for p in f.root.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
