"""Synthetic saved JSON / allocator files only. No framework or Native loading."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.c2c3_closure_20261117 import review_native26_single_source54 as audit


def sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def event(kind, sequence, *, producer, output, residual, call=0, registration=0, cache=0):
    return dict(version=1, size=416, kind=kind, kind_name=audit.KINDS[kind], flags=7, device=0, status=0,
        sequence=sequence, thread=1, call=call, registration=registration, executor=0, cache=cache,
        producer=producer, output=output, output_bytes=64, residual=residual, residual_bytes=64,
        rows=2, columns=16, residual_rows=2, residual_columns=16, dtype=27, format=2,
        dimensions=2, residual_dimensions=2, shape=[2,16,0,0,0,0,0,0], strides=[16,1,0,0,0,0,0,0],
        residual_shape=[2,16,0,0,0,0,0,0], residual_strides=[16,1,0,0,0,0,0,0])


def tensor(address, object_id):
    return dict(object_id=object_id, storage_identity=address+1000000, storage_offset=0,
        shape=[2,16], stride=[16,1], dtype='torch.bfloat16', device='npu:0',
        storage_addr=address, storage_bytes=64, element_size=2, actual_format=2)


def initial_trace():
    return dict(version=1, size=160, flags=7, reserved=0, capacity=32768, total_events=0, stored_events=0,
        overflow_events=0, errors=0, all_runs=0, qmm_runs=0, non_target_runs=0, **{k:0 for k in audit.KINDS.values()})


def counters(step):
    images = []
    for index, (name, symbol) in enumerate(audit.SYMBOLS.items()):
        state = {key:0 for key in audit.COUNTERS}
        state.update(version=1, size=120, instance=index+1, capture_calls=10+step,
            getter_calls=30+step*3, acquired_arrays=30+step*3, released_arrays=30+step*3,
            peak_live_arrays=3, revision=60+step*6)
        images.append(dict(name=name, path='/native/'+name, sha256=sha(name), symbol=symbol,
            mapped_paths=['/native/'+name], symbol_providers=['/native/'+name], state=state))
    return dict(schema='demoncat.metadata_arrays.v1', source='ACTUAL_LOADED_PROXY_COUNTERS', pid=1,
        build_profile_sha256=audit.BUILD_SHA, build_manifest_sha256=audit.BUILD_SHA,
        images=images, scope='SDK_GETTER_ARRAYS_ONLY', reset_supported=False)


def native_state(owner, callbacks, admissions, *, removed=False, sealed=False):
    core = dict(version=2, size=136, mechanism=1, reserved=0, error=0, reserved2=0,
        installed=int(not removed), stopped=int(removed), sealed=int(sealed), owner_hi=owner[0], owner_lo=owner[1],
        generation=1, thread=1, targets=int(admissions > 0 and not removed), calls=admissions,
        pending=0, executing=0, device_bytes=0, host_bytes=128 if admissions and not removed else 0, descriptor_pending=0)
    return dict(version=2, size=240, core=core, callback_active=0, completion_active=0, prepared_fia=0,
        fia_executor=0, descriptor=0, callback_calls=callbacks, passthrough_calls=callbacks-admissions,
        admitted_calls=admissions, completion_calls=admissions, rejected_calls=0, foreign_calls=0,
        last_rejection=0, reserved=0)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.aliases = [dict(path='<root>', object_identity=1)]
        self.aliases += [dict(path='writer'+str(i), object_identity=100+i) for i in range(3)]
        self.aliases += [dict(path='norm'+str(i), object_identity=1000+i) for i in range(4)]
        self.worker = dict(pid=1, rank=0, logical_device=0, configured_logical_device=0, observed_logical_device=0,
            process_start_identity='fixture-process', worker_start_identity='fixture-worker', model_instance_id='fixture-model')
        self.lifetime = dict(worker=self.worker, pid=1, native_thread=1, process_start_identity='fixture-process',
            model_object_identity=1, native_inventory_sha256=audit.TRACE_SHA, native_writer_sha256=audit.WRITER_SHA)
        self.owner = 'cpu-only-synthetic-owner'
        self.owner_words = audit.words(sha(self.owner))[:2]
        self.inputs = dict(input_ids=dict(dtype='torch.int32', shape=[3], stride=[1], storage_offset=0,
                                         object_id=55, storage_identity=66, device='npu:0'),
                           positions=dict(dtype='torch.int64', shape=[3], stride=[1], storage_offset=0,
                                          object_id=77, storage_identity=88, device='npu:0'))
        self.tokens = [11,22,33]
        self.input_sha = audit.input_hash(self.tokens, [0,1,2], self.inputs)
        self.rows, self.phases, self.observations, self.outcomes = [], [], [], []
        self.before, self.callback_count, self.producer_count = initial_trace(), 0, 0
        self.selected = None
        for index in range(8):
            self.add_row(index)
            if index == 1:
                frozen = audit.freeze(self.rows[0]['candidates'], self.rows[1]['candidates'])
                resolved = [v for v in frozen if v['status'] == 'OBSERVED_NOT_QUALIFIED']
                self.selected = min(resolved, key=lambda v:v['target_key'])
        frozen = audit.freeze(self.rows[0]['candidates'], self.rows[1]['candidates'])
        selection = dict(policy='MIN_ACTUAL_HEALTHY_RESOLVED_TARGET_KEY_NO_FALLBACK', target_key=self.selected['target_key'],
            inventory_index=self.selected['index'], complete_native_count=3, complete_resolved_count=2,
            inventory_sha256=audit.digest(frozen), aliases_sha256=audit.digest(self.aliases), planned_target_count=1,
            legacy_complete_plan_max_targets=1, max_targets_is_selector=False)
        qualification = dict(target_key=self.selected['target_key'], request_ids=[o['request_id'] for o in self.outcomes[:3]],
            outcomes_sha256=audit.digest(self.outcomes[:3]), owner=self.owner, native_owner=self.owner_words,
            generation=1, lifetime_sha256=audit.digest(self.lifetime), build=audit.BUILD_SHA,
            baseline_output_sha256=sha('healthy'), scope=audit.SCOPE, serializable_authority=False, qualification_granted=False)
        self.inventory = dict(schema='demoncat.writer26.candidate_experiment.v2', mechanism='26.residual_stomp', stage='c1_single_effects',
            phase='complete', failure=None, first_failure=None, failure_events=[], last_native_receipt=self.outcomes[-1]['observed_consumer_receipt'],
            blockers=[], scope=audit.SCOPE, lifetime=self.lifetime, build=audit.BUILD_SHA, rows=self.rows, candidates=frozen,
            metadata_arrays_initial=counters(0), metadata_arrays_latest=counters(8), resolved_candidate_count=2, qualified_candidate_count=1,
            outcomes=self.outcomes, native_state=native_state(self.owner_words, 21, 5, removed=True, sealed=True),
            owned_hooks=0, pending_output_tensors=0, pending_residual_tensors=0, pending_per_call_leases=[], pending_outcomes=[],
            pending_public_storage_owners=0, public_output_observer=None, owned_public_api_wrappers=0,
            installed=False, native_removed=True, alias_inventory=self.aliases,
            single_site_c1=dict(selection=selection, reference_lossless_off=qualification, owned_root_logits_hooks=0,
                evidence_complete=False, qualification_granted=False))
        self.document = dict(stage='c1_single_effects', mechanism='26.residual_stomp', scope=audit.SCOPE,
            identity=dict(worker=self.worker), actual_prompt_token_ids=self.tokens, phases=self.phases,
            healthy_before_tokens=[185], healthy_after_tokens=[185], single_site_c1=self.inventory['single_site_c1'],
            sessions=dict(candidate=dict(trigger='once', phases=self.phases,
                setup=dict(binding=dict(worker=self.worker, build=audit.BUILD_SHA)),
                remove=dict(report=dict(native_state=self.inventory['native_state'], completed_requests=8,
                    full_native_writer_counts=[3]*8, phase='complete', owned_root_logits_hooks=0,
                    owned_module_hooks=0, owned_public_api_wrappers=0, installed=False,
                    single_site_c1=self.inventory['single_site_c1'])))))
        self.serving = dict(boundary=dict(output_contract=[1,256], vocab_size=256), requests=self.observations,
                            qualification=self.inventory, native_phases=self.rows)

    def allocator(self, call, nonadjacent):
        blocks = [dict(size=64, state='active_allocated')]
        if nonadjacent:
            blocks.append(dict(size=64, state='inactive'))
        blocks.append(dict(size=64, state='active_allocated'))
        snapshot = [dict(address=call['residual'], blocks=blocks)]
        relative = 'writer-v2/candidate-experiment/allocator-'+str(call['independent_call'])+'.json'
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot))
        return dict(path='/evidence/'+relative, sha256=audit.file_sha(path), complete_snapshot_sha256=audit.digest(snapshot)), snapshot

    def add_row(self, index):
        rid, phase = 'actual-request-'+str(index), audit.PHASES[index]
        events, callbacks, producers = [], [], []
        def emit(kind, **kwargs):
            e = event(kind, self.before['stored_events']+len(events)+1, **kwargs)
            events.append(e)
            return e
        def producer(j):
            sequence = index*4+j+1
            output = (j+1)*10000
            residual = j*10000 if j else 8000
            parts = [emit(k, producer=sequence, output=output, residual=residual) for k in (7,8,9)]
            producers.append(dict(module_object_identity=1000+j, module_path='norm'+str(j), request_id=rid,
                original_calls=1, errors=[], native_events=parts, native_producer=sequence,
                output=tensor(output, index*100000+1000+j), consumed=tensor(residual, index*100000+1000+j-1),
                consumed_retained_producer=sequence-1 if j else None))
        producer(0)
        for j in range(3):
            residual, output = (j+1)*10000, (j+1)*10000+(128 if j == 2 else 64)
            native = emit(1, producer=index*4+j+1, output=output, residual=residual,
                call=index*100+j+1, registration=500+j, cache=600+j)
            emit(2, producer=index*4+j+1, output=output, residual=residual,
                call=index*100+j+1, registration=500+j, cache=600+j)
            producer(j+1)
            if index == 7:
                continue
            call = dict(version=2, size=144, mechanism=1, dimensions=2, device=0, dtype=27, format=2, reserved=0,
                token=index*100+j+1, thread=1, independent_call=native['call'], registration=500+j, cache=600+j,
                producer_sequence=native['producer'], output=output, output_bytes=64, residual=residual,
                residual_bytes=64, shape=[2,16,0,0])
            reference, snapshot = self.allocator(call, j == 2)
            alloc, nonadjacent = audit.allocation(snapshot, call)
            callback = dict(native_packet=call, original_continuations=1, claimed=0,
                errors=['LIVE_ALLOCATOR_BLOCKS_PROVEN_NOT_ADJACENT'] if j == 2 else [], independent_call=native['call'],
                module_object_identity=100+j, module_path='writer'+str(j), method=dict(path='/external/source.py',
                    source_sha256=sha('source'), code_sha256=sha('code'+str(j)), callable_identity=10000+j,
                    module_class='fixture.Writer'), actual_request_id=rid, token=call['token'], independent_event=native,
                producer_tensor_object_identity=producers[j]['output']['object_id'], allocator_segments=reference,
                writer_output_retained=dict(tensor_object_identity=200000+index*100+j, storage_address=output,
                    bytes=64, shape=[2,16], stride=[16,1], dtype='torch.bfloat16'), public_consumer_lifetime=True,
                public_consumer_binding=dict(producer_tensor_object_identity=producers[j]['output']['object_id'],
                    consumer_tensor_object_identity=producers[j+1]['consumed']['object_id'], residual_storage_identity=residual+1000000,
                    writer_tensor_object_identity=200000+index*100+j, same_residual_storage=True,
                    writer_retained_through_actual_consumer=True))
            if alloc:
                callback.update(allocation=alloc, current_candidate_geometry_eligible=True)
            else:
                callback['proven_nonadjacent'] = nonadjacent
            callback['target_key'] = audit.target_key(callback)
            if audit.MODES[index] and callback['target_key'] == self.selected['target_key']:
                callback.update(claimed=1, single_site_current_binding_verified=True)
                self.outcomes.append(self.make_outcome(callback, rid, audit.MODES[index]))
            callbacks.append(callback)
        after = copy.deepcopy(self.before)
        for e in events:
            after[e['kind_name']] += 1
        after.update(total_events=after['total_events']+len(events), stored_events=after['stored_events']+len(events),
            qmm_runs=after['qmm_runs']+3, non_target_runs=after['non_target_runs']+4, all_runs=after['all_runs']+7)
        mode = audit.MODES[index]
        row = dict(phase=phase, request_id=rid, input_sha256=self.input_sha, native_before=self.before,
            callback_start=self.callback_count, producer_start=self.producer_count,
            plan=dict(phase=phase, mode=mode, target=self.selected['target_key'] if mode else None),
            admissions=int(bool(mode)), completed=int(bool(mode)), native_after=after, native_events=events,
            callbacks=callbacks, producers=producers, output_sha256=sha('fault' if mode == 'fault' else 'healthy'),
            output_finite=True, metadata_arrays=counters(index+1))
        row['candidates'] = audit.correlate(row)
        row['complete_inventory'] = dict(native_writer_count=3, native_events_sha256=audit.digest(events),
            candidates_sha256=audit.digest(row['candidates']), actual_aliases_sha256=audit.digest(self.aliases))
        self.rows.append(row)
        admissions_before = max(0, min(index-2, 5))
        mapping = dict(internal_request_id=rid, external_request_id='external-'+str(index), request_state_object_id=index+1,
            source='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR', proof_scope='DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION')
        self.phases.append(dict(session='candidate', index=index, phase=phase, mapping=mapping, generated_tokens=[185],
            advance=dict(native_state=native_state(self.owner_words, self.callback_count, admissions_before, sealed=index >= 5)),
            result=dict(native_state=native_state(self.owner_words, self.callback_count+len(callbacks),
                admissions_before+int(bool(mode)), removed=index == 7, sealed=index >= 4))))
        snapshot = dict(internal_request_id=rid, request_object_id=index+1, num_requests=1, num_prompt_tokens=3,
            num_computed_tokens=0, num_scheduled_tokens=3, token_ids=self.tokens, positions=[0,1,2],
            token_ids_sha256=hashlib.sha256(json.dumps(self.tokens, separators=(',', ':')).encode()).hexdigest(),
            input_tensor_metadata=self.inputs, device_input_bytes_equality='COMPLETE_DEVICE_READBACK_VERIFIED',
            attention_metadata_keys=['actual.attention'])
        self.observations.append(dict(snapshot=snapshot, input_sha256=self.input_sha, phase=phase,
            actual_internal_request_id=rid, root_returned=True, model_output_sha256=row['output_sha256'],
            model_output_finite=True, output_shape=[1,256], top1=185))
        self.before = after
        self.callback_count += len(callbacks)
        self.producer_count += len(producers)

    def make_outcome(self, callback, rid, mode):
        call, alloc = callback['native_packet'], callback['allocation']
        target, mode_id = callback['target_key'], audit.MODE_IDS[mode]
        lease = dict(version=2, size=248, mode=mode_id, reserved=0, owner_hi=self.owner_words[0], owner_lo=self.owner_words[1],
            generation=1, thread=1, target=audit.words(target), expected=call,
            residual_allocation_end=alloc['residual_block'][1], output_allocation_begin=alloc['output_block'][0], shift_bytes=alloc['shift'])
        receipt = dict(version=2, size=296, mode=mode_id, mechanism=1, error=0, writer_status=0, consumer_status=0, reserved=0,
            owner_hi=self.owner_words[0], owner_lo=self.owner_words[1], generation=1, target=audit.words(target),
            **{k:call[k] for k in ('token', 'independent_call', 'registration', 'cache', 'producer_sequence',
                'output', 'output_bytes', 'residual', 'residual_bytes')},
            replacement=999999 if mode == 'lossless' else call['output']-32 if mode == 'fault' else call['output'],
            stream=23, executor=0, writer_calls=1, consumer_calls=1, changed_residual_bytes=31 if mode == 'fault' else 0,
            writer_complete=1, consumer_complete=1, output_readback_exact=1, padding_readback_exact=1,
            residual_unchanged=int(mode != 'fault'), residual_restored_exact=1, reference_equal=int(mode != 'fault'),
            lossless_qualified=int(mode != 'reference'), off_exact=int(mode == 'off'), drained=0, released=0)
        return dict(mode=mode, target=target, native_call=call, residual_before_sha256=sha('residual'), request_id=rid,
            lease=lease, observed_consumer_receipt=copy.deepcopy(receipt), consumer_receipt=receipt,
            released_receipt=dict(receipt, released=1), residual_after_sha256=sha('residual'),
            writer_output_sha256=sha('writer'), public_consumer_lifetime=True)

    def run(self):
        return audit.review(self.document, self.inventory, self.serving, audit.Artifacts(self.root),
            expected_count=3, expected_aliases=8, expected_selection=None, expected_resolved=2)


class SingleSource54ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.fixture = Fixture(self.temp)

    def rejected(self, code):
        result = self.fixture.run()
        self.assertFalse(result['consistent'])
        self.assertIn(code, {e['code'] for e in result['issues']}, result['issues'])
        self.assertFalse(result['single_site_receipt_chain_reconciled'])
        self.assertFalse(result['public_qualification_granted'] or result['c2'] or result['c3'] or result['e5'])

    def provenance_fixture(self):
        source, native = Path(self.temp)/'source', Path(self.temp)/'native'
        source.mkdir(); native.mkdir()
        hashes = {}
        for name in audit.SOURCES:
            path = source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# explicit synthetic source: '+name+'\n')
            hashes[name] = audit.file_sha(path)
        manifest = dict(base_commit=audit.SOURCE_COMMIT, files=[dict(path=name, sha256=value,
            bytes=(source/name).stat().st_size) for name, value in hashes.items()])
        (source/'source_manifest.json').write_text(json.dumps(manifest))
        native_files = {}
        for name in (*audit.SYMBOLS, 'metadata-array-ownership.json'):
            (native/name).write_text('explicit synthetic bytes: '+name)
            native_files[name] = audit.file_sha(native/name)
        compile_inputs = {name:hashes[name] for name in audit.SOURCES[8:10]}
        receipt = dict(schema='demoncat.writer26.group_build.v2', actual_npu_calls=0,
                       device_mounts=[], sources=compile_inputs)
        (native/'build-receipt.json').write_text(json.dumps(receipt))
        build = dict(schema='demoncat.writer26.group_deployment.v2', legacy_writers_absent=True,
            actual_npu_calls=0, files=native_files, sources=compile_inputs,
            build_receipt_sha256=audit.file_sha(native/'build-receipt.json'),
            metadata_array_ownership=dict(schema='demoncat.metadata_arrays.v1', state_version=1,
                images=audit.SYMBOLS, reset_supported=False, runtime_quiescent_readback_required=True,
                parent_receipt_sha256=native_files['metadata-array-ownership.json']))
        (native/'build.json').write_text(json.dumps(build))
        build_sha = audit.file_sha(native/'build.json')
        self.enterContext(patch.multiple(audit, BUILD_SHA=build_sha,
            SOURCE_MANIFEST=audit.file_sha(source/'source_manifest.json'),
            TRACE_SHA=native_files['libfault26_readonly_trace_abi0.so'], WRITER_SHA=native_files['libdcat26_group_v2.so']))
        inventory, document = copy.deepcopy(self.fixture.inventory), copy.deepcopy(self.fixture.document)
        inventory['build'] = build_sha
        inventory['lifetime']['native_inventory_sha256'] = native_files['libfault26_readonly_trace_abi0.so']
        inventory['lifetime']['native_writer_sha256'] = native_files['libdcat26_group_v2.so']
        for value in (inventory['metadata_arrays_initial'], inventory['metadata_arrays_latest']):
            value['build_profile_sha256'] = value['build_manifest_sha256'] = build_sha
            for image in value['images']:
                image['sha256'] = native_files[image['name']]
        document['sessions']['candidate']['setup']['sources'] = [dict(
            path='/installed/'+(name[4:] if name.startswith('src/') else name), sha256=hashes[name]) for name in audit.SETUP_SOURCES]
        document['metadata_startup'] = dict(schema='demoncat.writer26.startup_dependencies.v1', pid=1,
            manifest_sha256=build_sha, implementation_sha256=hashes[audit.SOURCES[5]],
            model_runtime_imports_before=[], callback_installed=False, device_api_calls=0,
            global_symbol_scope_changed=False, metadata_arrays=copy.deepcopy(inventory['metadata_arrays_initial']))
        return document, inventory, source, native

    def test_positive_full_eight_request_reconciliation_without_qualification_boolean(self):
        result = self.fixture.run()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['inventory']['natural_calls'], 24)
        self.assertEqual(result['inventory']['healthy_classification'], {'OBSERVED_NOT_QUALIFIED':2, 'UNSUPPORTED':1})
        self.assertEqual(result['effect']['fault_changed_residual_bytes'], [31])
        self.assertEqual(result['effect']['allocator_artifacts_rehashed'], 21)
        self.assertTrue(result['single_site_receipt_chain_reconciled'])
        self.assertFalse(result['public_qualification_granted'] or result['complete_tensor_bytes_rehashed'])

    def test_report_true_does_not_hide_native_missing_call(self):
        self.fixture.inventory['single_site_c1']['evidence_complete'] = True
        self.fixture.rows[2]['native_events'].pop(3)
        self.rejected('NATIVE_EVENT_DENOMINATOR')

    def test_candidate_denominator_truncation_is_not_single_site_selection(self):
        self.fixture.rows[1]['candidates'].pop()
        self.rejected('COMPLETE_CANDIDATE_RECONSTRUCTION')

    def test_selected_index_must_be_min_complete_resolved_key(self):
        self.fixture.inventory['single_site_c1']['selection']['target_key'] = 'f'*64
        self.rejected('MIN_COMPLETE_HEALTHY_KEY_SELECTION')

    def test_unsupported_requires_saved_nonadjacent_live_blocks(self):
        self.fixture.rows[0]['callbacks'][2]['proven_nonadjacent'][0][0][1] += 1
        self.rejected('UNSUPPORTED_NONADJACENCY_NOT_PROVEN')

    def test_recorded_hash_not_enough_when_allocator_file_changes(self):
        path = next(Path(self.temp).rglob('allocator-1.json'))
        path.write_text('[]')
        self.rejected('SCHEMA_OR_ARTIFACT_UNAVAILABLE')

    def test_padding_and_shift_recomputed(self):
        self.fixture.rows[2]['callbacks'][self.fixture.selected['index']]['allocation']['padding'] += 1
        self.rejected('ACTUAL_ALLOCATOR_ADJACENCY')

    def test_native_event_gap_refuses(self):
        self.fixture.rows[3]['native_events'][2]['sequence'] += 1
        self.rejected('NATIVE_EVENT_IDENTITY_OR_GAP')

    def test_native_counter_overflow_refuses(self):
        self.fixture.rows[6]['native_after']['overflow_events'] = 1
        self.rejected('COMPLETE_NATIVE_COUNTERS')

    def test_between_window_non_target_calls_preserved_without_false_denominator_failure(self):
        # Sampling outside the captured root/logits interval may execute other
        # global operations. Those do not add an unobserved QMM writer.
        for row in self.fixture.rows[1:]:
            row['native_before'] = copy.deepcopy(row['native_before'])
            row['native_after'] = copy.deepcopy(row['native_after'])
            for state in (row['native_before'], row['native_after']):
                state['all_runs'] += 7
                state['non_target_runs'] += 7
        result = self.fixture.run()
        self.assertTrue(result['consistent'], result['issues'])
        self.assertEqual(result['inventory']['outside_request_non_target_runs'][0]['non_target_runs'], 7)

    def test_between_window_hidden_qmm_cannot_be_counted_as_other_sampling(self):
        self.fixture.rows[1]['native_before'] = copy.deepcopy(self.fixture.rows[1]['native_before'])
        self.fixture.rows[1]['native_before']['qmm_runs'] += 1
        self.rejected('BETWEEN_REQUEST_TARGETED_TRACE_ACTIVITY')

    def test_native_integer_packet_does_not_accept_boolean_success_field(self):
        self.fixture.outcomes[0]['consumer_receipt']['writer_calls'] = True
        self.rejected('NATIVE_RECEIPT_IDENTITY')

    def test_request_mapping_cannot_be_inferred_from_label(self):
        self.fixture.phases[4]['mapping']['internal_request_id'] = 'foreign'
        self.rejected('REAL_DRIVER_SCHEDULER_REQUEST_JOIN')

    def test_repeated_request_id_refuses(self):
        self.fixture.phases[4]['mapping']['internal_request_id'] = self.fixture.phases[3]['mapping']['internal_request_id']
        self.rejected('REAL_DRIVER_SCHEDULER_REQUEST_JOIN')

    def test_wrong_input_bytes_digest_refuses(self):
        self.fixture.rows[3]['input_sha256'] = sha('foreign-input')
        self.rejected('COMPLETE_INPUT_DIGEST_RECOMPUTATION')

    def test_same_address_different_storage_identity_refuses(self):
        cb = self.fixture.rows[3]['callbacks'][self.fixture.selected['index']]
        cb['public_consumer_binding']['residual_storage_identity'] += 1
        self.rejected('PUBLIC_PRODUCER_WRITER_CONSUMER_OBJECT_STORAGE_JOIN')

    def test_same_storage_foreign_producer_tensor_refuses(self):
        cb = self.fixture.rows[3]['callbacks'][self.fixture.selected['index']]
        cb['public_consumer_binding']['producer_tensor_object_identity'] += 1
        self.rejected('PUBLIC_PRODUCER_WRITER_CONSUMER_OBJECT_STORAGE_JOIN')

    def test_writer_tensor_replaced_before_actual_consumer_refuses(self):
        cb = self.fixture.rows[3]['callbacks'][self.fixture.selected['index']]
        cb['public_consumer_binding']['writer_tensor_object_identity'] += 1
        self.rejected('PUBLIC_PRODUCER_WRITER_CONSUMER_OBJECT_STORAGE_JOIN')

    def test_wrong_actual_call_token_refuses(self):
        self.fixture.outcomes[3]['native_call']['token'] += 1
        self.rejected('NATIVE_RECEIPT_IDENTITY')

    def test_multiple_original_calls_refuse(self):
        self.fixture.rows[2]['callbacks'][self.fixture.selected['index']]['original_continuations'] = 2
        self.rejected('ORIGINAL_WRITER_NOT_EXACTLY_ONCE')

    def test_foreign_owner_receipt_refuses(self):
        self.fixture.outcomes[0]['consumer_receipt']['owner_hi'] += 1
        self.rejected('NATIVE_RECEIPT_IDENTITY')

    def test_foreign_generation_lease_refuses(self):
        self.fixture.outcomes[1]['lease']['generation'] += 1
        self.rejected('LEASE_OWNER_GENERATION_ACTUAL_FORMALS')

    def test_missing_consumer_refuses(self):
        self.fixture.outcomes[3]['consumer_receipt']['consumer_calls'] = 0
        self.rejected('NATIVE_EXECUTION_CONSUMPTION_RECOVERY')

    def test_fault_no_effect_refuses(self):
        self.fixture.outcomes[3]['consumer_receipt']['changed_residual_bytes'] = 0
        self.rejected('QMM_ACTUAL_32_BYTE_RESIDUAL_EFFECT')

    def test_fault_outside_32_byte_span_refuses(self):
        self.fixture.outcomes[3]['consumer_receipt']['changed_residual_bytes'] = 33
        self.rejected('QMM_ACTUAL_32_BYTE_RESIDUAL_EFFECT')

    def test_residual_full_hash_not_restored_refuses(self):
        self.fixture.outcomes[3]['residual_after_sha256'] = sha('not-restored')
        self.rejected('FULL_RESIDUAL_RESTORATION_HASH')

    def test_failure_receipt_cannot_be_overwritten_by_success(self):
        self.fixture.outcomes[2]['observed_consumer_receipt']['error'] = -11
        self.rejected('FIRST_NATIVE_RECEIPT_AND_ACK_ONLY_RELEASE')

    def test_failed_ack_cannot_release_owner(self):
        self.fixture.outcomes[4]['released_receipt']['released'] = 0
        self.rejected('FIRST_NATIVE_RECEIPT_AND_ACK_ONLY_RELEASE')

    def test_runtime_rejected_callback_refuses(self):
        self.fixture.phases[5]['result']['native_state']['rejected_calls'] = 1
        self.rejected('NATIVE_QUIESCENT_NO_UNCERTAIN_REPLIES')

    def test_native_leak_refuses_even_when_installed_false(self):
        self.fixture.inventory['native_state']['core']['host_bytes'] = 64
        self.rejected('NATIVE_HOST_RESOURCES_NOT_ZERO')

    def test_retained_public_owner_refuses(self):
        self.fixture.inventory['pending_public_storage_owners'] = 1
        self.rejected('PENDING_PUBLIC_OWNERS_OR_LEASES_RETAINED')

    def test_remaining_python_hook_refuses(self):
        self.fixture.inventory['single_site_c1']['owned_root_logits_hooks'] = 1
        self.rejected('FINAL_ALL_OWNED_NATIVE_AND_PYTHON_RESOURCES_ZERO')

    def test_metadata_array_live_allocation_refuses(self):
        self.fixture.inventory['metadata_arrays_latest']['images'][0]['state']['live_arrays'] = 1
        self.rejected('METADATA_ARRAY_LEAK_OR_FAILURE')

    def test_missing_metadata_image_refuses(self):
        self.fixture.inventory['metadata_arrays_latest']['images'].pop()
        self.rejected('COMPLETE_FOUR_METADATA_DSOS')

    def test_metadata_instance_change_refuses(self):
        self.fixture.rows[3]['metadata_arrays']['images'][0]['state']['instance'] = 99
        self.rejected('METADATA_DSO_OR_INSTANCE_CHANGED')

    def test_metadata_counter_reset_refuses(self):
        self.fixture.rows[3]['metadata_arrays']['images'][0]['state']['revision'] = 1
        self.rejected('METADATA_ARRAY_COUNTER_RESET')

    def test_off_full_logits_must_match_baseline_even_when_token_unchanged(self):
        self.fixture.rows[6]['output_sha256'] = sha('changed-off')
        self.fixture.observations[6]['model_output_sha256'] = sha('changed-off')
        self.rejected('FULL_LOGITS_EFFECT_OR_HEALTHY_OFF_MISMATCH')

    def test_fault_full_logits_difference_must_be_observed(self):
        self.fixture.rows[5]['output_sha256'] = sha('healthy')
        self.fixture.observations[5]['model_output_sha256'] = sha('healthy')
        self.rejected('FULL_LOGITS_EFFECT_OR_HEALTHY_OFF_MISMATCH')

    def test_no_native_callback_in_post_remove_health(self):
        self.fixture.rows[7]['callbacks'] = [copy.deepcopy(self.fixture.rows[6]['callbacks'][0])]
        self.rejected('NATIVE_CALLBACK_REMOVED_BEFORE_HEALTH')

    def test_duplicate_json_keys_refuse(self):
        path = Path(self.temp)/'duplicate.json'
        path.write_text('{"error":1,"error":0}')
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_JSON_KEY'):
            audit.read(path)

    def test_allocator_path_cannot_escape_explicit_evidence_mapping(self):
        reference = dict(path='/evidence/../outside.json', sha256='a'*64, complete_snapshot_sha256='b'*64)
        with self.assertRaisesRegex(ValueError, 'TRAVERSAL'):
            audit.Artifacts(self.temp).allocator(reference)

    def test_no_framework_or_native_import_needed(self):
        import ast
        tree = ast.parse(Path(audit.__file__).read_text())
        forbidden = {'torch','torch_npu','vllm','vllm_ascend','ctypes','subprocess','socket'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(forbidden.intersection(n.name.split('.')[0] for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], forbidden)
        for event_name, args in [('ctypes.dlopen', ('library.so',)), ('socket.connect', ())]:
            with self.assertRaises(RuntimeError):
                audit.forbid_runtime(event_name, args)

    def test_provenance_rehashes_source_native_and_separate_build_receipt(self):
        report = audit.provenance(*self.provenance_fixture())
        self.assertTrue(report['consistent'], report['issues'])
        self.assertEqual(report['native_files_rehashed'], 6)
        self.assertEqual(report['native_compile_sources_rehashed'], 2)

    def test_installed_source_receipt_must_match_frozen_source(self):
        document, inventory, source, native = self.provenance_fixture()
        document['sessions']['candidate']['setup']['sources'][0]['sha256'] = sha('wrong')
        report = audit.provenance(document, inventory, source, native)
        self.assertIn('ACTUAL_INSTALLED_SOURCE_JOIN', {e['code'] for e in report['issues']})

    def test_native_library_bytes_must_match_bundle(self):
        document, inventory, source, native = self.provenance_fixture()
        (native/'libopapi_nn.so').write_text('changed')
        report = audit.provenance(document, inventory, source, native)
        self.assertIn('NATIVE_BUILD_FILE_BYTES', {e['code'] for e in report['issues']})

    def test_compile_source_cannot_be_replaced_by_current_unrelated_cpp(self):
        document, inventory, source, native = self.provenance_fixture()
        (source/audit.SOURCES[8]).write_text('changed')
        report = audit.provenance(document, inventory, source, native)
        self.assertIn('ACTUAL_NATIVE_COMPILE_SOURCE_BYTES', {e['code'] for e in report['issues']})

    def test_build_receipt_is_rehashed_even_when_not_in_files_map(self):
        document, inventory, source, native = self.provenance_fixture()
        path = native/'build-receipt.json'
        path.write_text(path.read_text()+'\n')
        report = audit.provenance(document, inventory, source, native)
        self.assertIn('NATIVE_BUILD_RECEIPT_BYTES_AND_SOURCES', {e['code'] for e in report['issues']})

    def test_cli_wrong_original_pin_emits_compact_refusal_without_runtime(self):
        (Path(self.temp)/'result.json').write_text('{}')
        output = io.StringIO()
        with redirect_stdout(output), patch.object(audit.sys, 'addaudithook'):
            code = audit.main(['--evidence-root', self.temp, '--source-root', self.temp, '--native-root', self.temp])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'EVIDENCE_GAP_OR_CONTRADICTION')
        self.assertIn('ORIGINAL_SOURCE54_SHA_MISMATCH:result', report['error'])
        self.assertLess(len(output.getvalue()), 3000)


if __name__ == '__main__':
    unittest.main()
