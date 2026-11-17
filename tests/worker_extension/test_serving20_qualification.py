"""Explicit CPU mocks for Fault20's serving qualification, never native evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

try:
    import torch
except (ImportError, OSError):
    torch = None

from fault_injection import serving_qualification as sq
from fault_injection.deployments.matmul import CtypesMatmulRuntime, _REQUIRED_BUNDLE
from fault_injection.native import NativeCall, _LEASE
if __package__:
    from .test_serving_session_lifecycle import CpuModel
else:
    from test_serving_session_lifecycle import CpuModel


class CpuRuntime:
    """No ctypes, device, or serving engine: only explicit callback/receipt fixtures."""
    operator = 'aclnnMatmulWeightNz'
    mechanism = '20.buffer_substitute'
    build_profile = 'a' * 64

    def __init__(self, directory):
        self.worker = dict(pid=os.getpid(), tp=1, rank=0, logical_device=0,
            soc='CPU_FIXTURE_NOT_ASCEND', execution_mode='eager', worker_id='CPU-worker', model_instance_id='CPU-model')
        self.directory = directory
        self.callback = self.remove_owner = None
        self.paused = False
        self.serial = self.original_calls = 0
        self.modes = []
        self.packet_edit = self.receipt_edit = None
        self.pause_failure = self.partial_install_failure = False

    def snapshot(self):
        return {self.operator: dict(native_registered=self.callback is not None,
            native_owned=self.callback is not None, enabled=self.callback is not None and not self.paused,
            paused=self.paused, inflight=0, leases=0, recovery_required=False, runtime_matched=True,
            request_controlled=self.callback is not None)}

    def configure_request_lifecycle(self, *, owner, generation):
        self.remove_owner = (owner, generation)

    def install_observer(self, callback, candidates, mechanism):
        assert self.callback is None and self.remove_owner and mechanism == self.mechanism
        self.callback, self.processor, self.paused = callback, candidates[0][1], True
        if self.partial_install_failure:
            raise RuntimeError('MOCK_UNKNOWN_PARTIAL_INSTALL')
        def remove():
            assert self.paused
            self.callback = None
        return remove

    def resume(self):
        assert self.callback is not None and self.paused
        self.paused = False

    def pause(self):
        if self.pause_failure:
            raise RuntimeError('MOCK_PAUSE_FAILED')
        self.paused = True

    def execute_scoped(self, call, original, mode):
        self.modes.append(mode)
        result = original()
        formals = ['self'] if self.operator == 'aclnnMatmulWeightNz' else ['x1']
        if mode == 1:
            formals = []
        receipt = dict(mode=mode, original_calls=1, native_status=0, replacements=len(formals),
            resources_released=True, temporary_readback_verified=bool(formals), replaced_formals=formals,
            device_copy=mode == 2, changed_slots=int(mode == 3),
            input_sha256_before=['b' * 64, 'c' * 64], input_sha256_after=['b' * 64, 'c' * 64],
            output_sha256=('f' if mode == 3 else 'd') * 64)
        if self.receipt_edit:
            self.receipt_edit(receipt)
        return result, receipt

    def emit(self, transaction):
        assert not self.paused
        self.serial += 1
        hidden, weight = transaction['hidden'], transaction['weight']
        input_name, weight_name = ('self', 'mat2') if self.operator == 'aclnnMatmulWeightNz' else ('x1', 'x2')
        def formal(name, slot, tensor, *, transpose=False):
            return dict(formal=name, present=True, run_slot=slot, storage_bytes=tensor['storage_bytes'],
                storage_addr=tensor['storage_addr'], run_address=tensor['storage_addr'],
                view_offset=tensor['storage_offset'], view_shape=list(reversed(tensor['shape'])) if transpose else tensor['shape'],
                strides=list(reversed(tensor['stride'])) if transpose else tensor['stride'],
                element_size=tensor['element_size'], format=tensor['actual_format'])
        left = formal(input_name, 0, hidden)
        if input_name == 'x1':
            left.update(storage_addr=999, run_address=999, storage_bytes=4, element_size=1, view_offset=0)
        right = formal(weight_name, 1, weight, transpose=True)
        out = dict(formal='out', present=True, run_slot=2, storage_addr=333, run_address=333,
                   storage_bytes=6, view_shape=[1, weight['shape'][0]])
        formals = [left, right, out]
        if input_name == 'x1':
            formals.append(dict(formal='bias', present=False, run_slot=-1, run_address=0))
        packet = dict(version=1, operator=self.operator, boundary='op_exec_cache_run',
            native_signature='CPU_SIGNATURE_NOT_NATIVE', cache_identity='CPU-cache', cache_lifecycle='1',
            candidate_count=1, build_profile_sha256=self.build_profile, runtime_identity='MATCHED_BUILD',
            layout=dict(dtype='BF16' if input_name == 'self' else 'INT8', format='ND',
                        shape=left['view_shape'], stride=left['strides'], offset=left['view_offset']),
            slots=[dict(formal=row['formal'], slot=row['run_slot'], storage_bytes=row['storage_bytes'])
                   for row in formals if row['present']],
            current_typed_call=dict(source='typed_gws_execute_scope', executor_bound=True,
                formal_params_current=True, cached_layout_used=False, gws_token=self.serial, formals=formals))
        if self.packet_edit:
            self.packet_edit(packet)
        call_id = 'cpu-call-' + str(self.serial)
        proofs = CtypesMatmulRuntime._proofs(self, call_id, self.processor, packet, self.directory)
        call = NativeCall(self.processor, call_id, self.operator, proofs=proofs)
        def original():
            self.original_calls += 1
            return 0
        return self.callback(call, original)


@unittest.skipIf(torch is None, 'CPU torch fixture unavailable')
class Serving20QualificationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory(prefix='serving20-cpu-')))
        self.sessions = []
        self.environment = dict(pid=os.getpid(), maps_status='OBSERVED', machine='CPU-fixture', python='fixture',
            packages={}, torch_cxx11_abi=False, driver={'running_module_versions': {}, 'installation': None},
            loaded_libraries=[dict(name=name, sha256=hashlib.sha256(name.encode()).hexdigest(),
                elf_build_id='CPU', status='MAPPED_FILE_VERIFIED') for name in _REQUIRED_BUNDLE])
        self.enterContext(patch.object(sq, 'inspect_environment', side_effect=lambda **kwargs: copy.deepcopy(self.environment)))
        self.enterContext(patch.dict(os.environ, {'DEMONCAT_F20_REQUIRE_TYPED_GWS': '0'}))

    def tearDown(self):
        # Fixture disposal only; deliberately not a production OFF certificate.
        for session in self.sessions:
            if session.locked:
                _LEASE.release()
                session.locked = False

    def manifest(self, folder):
        evidence = folder / 'cpu-fixture-not-acceptance.json'
        evidence.write_text('{"evidence_kind":"CPU_MOCK_ONLY"}')
        record = dict(path=str(evidence), sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
        path = folder / 'review.json'
        path.write_text(json.dumps(dict(schema='demoncat.fault20.serving_review.v1',
            reviewed_contract=sq.REVIEW_CONTRACT, validation_level='CPU_VERIFIED',
            implementation_version='CPU_ONLY_FIXTURE', build_profile_sha256='a' * 64,
            required_libraries=sorted(_REQUIRED_BUNDLE), cpu_evidence=[record], build_evidence=record)))
        return path

    def session(self, *, baseline_control=False, start=True, first_live_validation=True, operator=None):
        folder = self.folder / str(len(self.sessions))
        folder.mkdir()
        model = CpuModel().eval()
        runtime = CpuRuntime(folder)
        if operator:
            runtime.operator = operator
        session = sq.ServingQualifiedNativeSession(model, model.processor, model.head,
            dict(processor_path='processor', vocab_size=3), runtime, owner='CPU-owner', generation=1,
            reviewed_manifest_path=self.manifest(folder), first_live_validation=first_live_validation,
            evidence_dir=folder / 'profile', baseline_control=baseline_control)
        self.sessions.append(session)
        if start:
            session.start()
        return session

    def transaction(self, session):
        hidden = dict(storage_addr=111, storage_bytes=8, storage_offset=0, shape=[1, 4], stride=[4, 1],
                      element_size=2, actual_format=2, dtype='torch.bfloat16')
        quantized = session.runtime.operator == 'aclnnQuantMatmulWeightNz'
        weight = dict(storage_addr=222, storage_bytes=12 if quantized else 24, storage_offset=0,
                      shape=[3, 4], stride=[4, 1], element_size=1 if quantized else 2, actual_format=29,
                      dtype='torch.int8' if quantized else 'torch.bfloat16')
        return dict(root_forward_returned=True, processor_object_id=id(session.processor),
                    head_argument_object_id=id(session.head), hidden_sha256='1' * 64,
                    weight_sha256='2' * 64, hidden=hidden, weight=weight, evidence='CPU_MOCK_ONLY')

    def phase(self, session, *, calls=1, output=None, finite=True, transaction=None, request_id=None):
        transaction = self.transaction(session) if transaction is None else transaction
        stage = session.phase
        request_id = request_id or 'cpu-request-' + str(len(session.requests))
        try:
            with torch.inference_mode(), session.processor_scope(session.processor):
                session.begin(request_id, '0' * 64, transaction=transaction)
                if stage != 'disabled_off':
                    for _ in range(calls):
                        session.runtime.emit(transaction)
                if output is None:
                    output = ('9' if stage == 'fault' and not session.baseline_control else '8') * 64
                session.end(output, model_output_finite=finite)
        except BaseException as error:
            if not session.failure:
                session.fail(str(error))
            raise

    def through(self, session, phase):
        while session.phase != phase:
            self.phase(session)

    def test_complete_eight_stage_gate_keeps_processor_identity_and_cpu_profile_level(self):
        session = self.session()
        self.through(session, 'complete')
        report = session.query()['qualification']
        self.assertEqual(report['status'], 'SCOPED_ATTEMPT_COMPLETED')
        self.assertEqual(report['completed_phases'], list(sq.PHASES))
        self.assertEqual(session.runtime.modes, [1, 2, 1, 3, 1])
        self.assertEqual([row['native_calls'] for row in session.rows], [1] * 7 + [0])
        self.assertEqual(session.selection['object_identity'], id(session.processor))
        self.assertNotEqual(session.selection['object_identity'], id(session.head))
        self.assertEqual(session.profile.validation_level, 'CPU_VERIFIED')
        self.assertFalse(session.runtime.snapshot()[session.runtime.operator]['enabled'])
        self.assertTrue(session.runtime.snapshot()[session.runtime.operator]['native_registered'])
        session.close()
        self.assertIsNone(session.runtime.callback)

    def test_baseline_control_keeps_phases_but_never_claims_injection(self):
        session = self.session(baseline_control=True)
        self.through(session, 'complete')
        self.assertEqual(session.runtime.modes, [1, 2, 1, 1, 1])
        self.assertFalse(session.fault_started)
        self.assertEqual(session.query()['qualification']['status'], 'BASELINE_CONTROL_COMPLETED')
        self.assertFalse(session.query()['qualification']['model_output_changed'])
        session.close()

    def test_baseline_control_fault_stage_still_requires_healthy_complete_output(self):
        session = self.session(baseline_control=True)
        self.through(session, 'fault')
        with self.assertRaisesRegex(sq.UnsupportedCall, 'HEALTHY_COMPLETE_MODEL_OUTPUT_CHANGED'):
            self.phase(session, output='9' * 64)
        self.assertFalse(session.fault_started)

    def test_missing_first_live_optin_refuses_before_native_install(self):
        with self.assertRaisesRegex(sq.UnsupportedCall, 'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED'):
            self.session(first_live_validation=False)

    def test_environment_refusal_is_recorded_and_consumes_attempt_before_install(self):
        session = self.session(start=False)
        self.environment['maps_status'] = 'UNAVAILABLE'
        with self.assertRaisesRegex(sq.UnsupportedCall, 'LOADED_WORKER_ENVIRONMENT_UNPROVEN'):
            session.start()
        self.assertIn('LOADED_WORKER_ENVIRONMENT_UNPROVEN', session.query()['qualification']['failure'])
        self.assertEqual(session.query()['qualification']['environment_refusal']['maps_status'], 'UNAVAILABLE')
        self.assertEqual(json.loads((session.directory / 'environment_refusal.json').read_text())[
            'actual_environment']['maps_status'], 'UNAVAILABLE')
        self.assertIsNone(session.runtime.callback)
        self.assertFalse(session.locked)
        with self.assertRaisesRegex(sq.UnsupportedCall, 'SINGLE_SERVING_ATTEMPT_REQUIRED'):
            session.start()
        session.close()

    def test_unverified_extra_mapping_is_not_silently_excluded_from_environment(self):
        session = self.session(start=False)
        self.environment['loaded_libraries'].append(dict(name='libtorch_npu.so',
            path='/image/torch_npu/lib/libtorch_npu.so', status='MAPPED_FILE_IDENTITY_CHANGED',
            sha256=None, elf_build_id=None))
        with self.assertRaisesRegex(sq.UnsupportedCall, 'LOADED_WORKER_ENVIRONMENT_UNPROVEN'):
            session.start()
        diagnostic = session.query()['qualification']['environment_refusal']
        self.assertEqual(diagnostic['missing_required_libraries'], [])
        self.assertEqual(diagnostic['unverified_libraries'], [dict(name='libtorch_npu.so',
            path='/image/torch_npu/lib/libtorch_npu.so', status='MAPPED_FILE_IDENTITY_CHANGED')])
        self.assertIsNone(session.runtime.callback)
        self.assertFalse(session.locked)
        session.close()

    def test_manifest_cannot_claim_native_validation_or_change_reviewed_bytes(self):
        for field, value in (('validation_level', 'NATIVE_VERIFIED'), ('build_profile_sha256', 'f' * 64)):
            with self.subTest(field=field):
                folder = self.folder / field
                folder.mkdir()
                path = self.manifest(folder)
                manifest = json.loads(path.read_text())
                manifest[field] = value
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(sq.UnsupportedCall, 'REVIEWED_BUILD_CONTRACT_MISMATCH'):
                    sq.load_reviewed_manifest(path, build_profile='a' * 64, first_live_validation=True)
        folder = self.folder / 'tamper'
        folder.mkdir()
        path = self.manifest(folder)
        (folder / 'cpu-fixture-not-acceptance.json').write_text('changed')
        with self.assertRaisesRegex(sq.UnsupportedCall, 'REVIEW_EVIDENCE_HASH_MISMATCH'):
            sq.load_reviewed_manifest(path, build_profile='a' * 64, first_live_validation=True)

    def test_two_native_calls_in_one_processor_scope_refuse_before_second_original(self):
        session = self.session()
        with self.assertRaisesRegex(sq.UnsupportedCall, 'NATURAL_PROCESSOR_NATIVE_TARGET_NOT_UNIQUE'):
            self.phase(session, calls=2)
        self.assertEqual(session.runtime.original_calls, 1)
        self.assertFalse(session.runtime.modes)
        session.close()

    def test_no_native_call_is_not_accepted_as_discovery(self):
        session = self.session()
        with self.assertRaisesRegex(sq.UnsupportedCall, 'EXACT_PHASE_NATIVE_CALL_COUNT_REQUIRED'):
            self.phase(session, calls=0)
        session.close()

    def test_changed_cache_or_candidate_multiplicity_prevents_profile_admission(self):
        for key, value in (('cache_lifecycle', '2'), ('cache_identity', 'other'), ('candidate_count', 2)):
            with self.subTest(key=key):
                session = self.session()
                self.phase(session)
                session.runtime.packet_edit = lambda p, k=key, v=value: p.update({k: v})
                with self.assertRaisesRegex(sq.UnsupportedCall, 'AUTOMATIC_SERVING_TARGET_REFUSED'):
                    self.phase(session)
                self.assertIsNone(session.selection)
                session.close()

    def test_actual_storage_or_shape_mismatch_refuses_without_calling_native(self):
        for formal, field, value in (('mat2', 'storage_addr', 444), ('self', 'view_offset', 1),
                                     ('out', 'view_shape', [1, 99]), ('mat2', 'strides', [4, 1])):
            with self.subTest(formal=formal, field=field):
                session = self.session()
                def edit(packet):
                    row = next(r for r in packet['current_typed_call']['formals'] if r['formal'] == formal)
                    row[field] = value
                session.runtime.packet_edit = edit
                with self.assertRaises(sq.UnsupportedCall):
                    self.phase(session)
                self.assertEqual(session.runtime.original_calls, 0)
                session.close()

    def test_qmm_intermediate_can_have_different_storage_but_head_must_match(self):
        session = self.session(operator='aclnnQuantMatmulWeightNz')
        self.through(session, 'complete')
        current = session.events[0]['current_typed_call']['formals']
        self.assertEqual(next(r for r in current if r['formal'] == 'x1')['storage_addr'], 999)
        self.assertEqual(session.rows[0]['transaction']['hidden']['storage_addr'], 111)
        self.assertEqual(session.runtime.modes, [1, 2, 1, 3, 1])
        session.close()

    def test_stale_typed_formals_are_not_profile_evidence(self):
        session = self.session()
        session.runtime.packet_edit = lambda p: p['current_typed_call'].update(cached_layout_used=True)
        with self.assertRaisesRegex(sq.UnsupportedCall, 'CURRENT_TYPED_FORMALS_REQUIRED'):
            self.phase(session)
        self.assertEqual(session.runtime.original_calls, 0)
        self.assertIn('CURRENT_TYPED_FORMALS_REQUIRED', session.last_native_failure)
        session.fail('SECONDARY_MODULE_OUTPUT_NONE')
        self.assertTrue(session.failure.startswith('UnsupportedCall: CURRENT_TYPED_FORMALS_REQUIRED'))
        self.assertIn('SECONDARY_MODULE_OUTPUT_NONE', session.failure)
        session.close()

    def test_reference_lossless_and_qualification_off_must_preserve_native_bytes(self):
        session = self.session()
        self.through(session, 'lossless')
        session.runtime.receipt_edit = lambda r: r.update(output_sha256='e' * 64)
        with self.assertRaisesRegex(sq.UnsupportedCall, 'HEALTHY_NATIVE_OUTPUT_BYTES_CHANGED'):
            self.phase(session)
        self.assertFalse(session.fault_started)
        self.assertNotIn(3, session.runtime.modes)
        session.close()

    def test_fault_input_mismatch_consumes_attempt_and_requires_two_fresh_offs(self):
        session = self.session()
        self.through(session, 'fault')
        session.runtime.receipt_edit = lambda r: r.update(input_sha256_before=['e' * 64, 'c' * 64],
                                                          input_sha256_after=['e' * 64, 'c' * 64])
        with self.assertRaisesRegex(sq.UnsupportedCall, 'QUALIFIED_NATIVE_INPUT_BYTES_CHANGED'):
            self.phase(session)
        self.assertTrue(session.attempted)
        with self.assertRaisesRegex(sq.UnsupportedCall, 'OFF_RECOVERY_REQUIRED_BEFORE_CLEAN'):
            session.close()
        session.runtime.receipt_edit = None
        session.request_recovery()
        self.through(session, 'complete')
        self.assertEqual(session.runtime.modes.count(3), 1)
        self.assertEqual(session.query()['qualification']['status'], 'RECOVERED_ATTEMPT_FAILED')
        session.close()

    def test_complete_model_hash_and_finiteness_are_checked_in_healthy_phases(self):
        for changes in ({'output': 'e' * 64}, {'finite': False}):
            session = self.session()
            self.phase(session)
            with self.assertRaises(sq.UnsupportedCall):
                self.phase(session, **changes)
            session.close()

    def test_reused_request_or_changed_hidden_bytes_cannot_resume_native(self):
        for change in ('request', 'hidden'):
            with self.subTest(change=change):
                session = self.session()
                self.phase(session)
                tx = self.transaction(session)
                if change == 'hidden':
                    tx['hidden_sha256'] = 'e' * 64
                with self.assertRaises(sq.UnsupportedCall):
                    self.phase(session, transaction=tx, request_id='cpu-request-0' if change == 'request' else None)
                self.assertTrue(session.runtime.paused)
                self.assertEqual(session.runtime.original_calls, 1)
                session.close()

    def test_changed_loaded_libraries_refuse_before_next_native_resume(self):
        session = self.session()
        self.phase(session)
        self.environment['loaded_libraries'][0]['sha256'] = 'f' * 64
        with self.assertRaisesRegex(sq.UnsupportedCall, 'LOADED_ENVIRONMENT_CHANGED'):
            self.phase(session)
        self.assertTrue(session.runtime.paused)
        session.close()

    def test_unknown_partial_install_does_not_release_ownership(self):
        session = self.session(start=False)
        session.runtime.partial_install_failure = True
        with self.assertRaisesRegex(RuntimeError, 'MOCK_UNKNOWN_PARTIAL_INSTALL'):
            session.start()
        with self.assertRaisesRegex(sq.UnsupportedCall, 'NATIVE_INSTALLATION_RECOVERY_UNKNOWN'):
            session.close()
        self.assertTrue(session.locked)

    def test_foreign_thread_cannot_start_or_resume_session(self):
        session = self.session(start=False)
        errors = []
        def foreign():
            try:
                session.start()
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=foreign)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn('FOREIGN_MODEL_THREAD', str(errors[0]))
        self.assertIsNone(session.runtime.callback)
        session.close()

    def test_pause_failure_cannot_be_reported_as_recovered(self):
        session = self.session()
        self.through(session, 'fault')
        session.runtime.pause_failure = True
        with self.assertRaisesRegex(RuntimeError, 'MOCK_PAUSE_FAILED'):
            self.phase(session)
        self.assertFalse(session.recovered)
        self.assertEqual(session.drain_status, 'RECOVERY_PENDING')
        with self.assertRaisesRegex(RuntimeError, 'MOCK_PAUSE_FAILED'):
            session.request_recovery()
        self.assertFalse(session.recovery_attempted)


if __name__ == '__main__':
    unittest.main(verbosity=2)
