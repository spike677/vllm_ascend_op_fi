"""CPU state-machine contracts only; mock IDs/hashes are not NPU evidence."""
import unittest

from demoncat_fault39.backend import WorkerEndpoint
from demoncat_fault39.schema import Fault39Config
from demoncat_fault39.serving_qualification import PHASES, ServingQualification
from test_host_qualification import ReplayProvider


class ServingMock(ReplayProvider):
    def resume_discovery(self, owner, generation):
        self._check_owner(owner, generation)
        if self._enabled or self._mode != 'discover':
            raise RuntimeError('RESUME_NOT_DISABLED_DISCOVERY')
        self._enabled = True

    def request(self, request_id, owner):
        super().request(request_id, owner)
        if self._mode in ('discover', '') or not self.observation:
            return
        self.observation.update(original_storage_sha256_before=['a'*64, 'b'*64],
                                original_storage_sha256_after=['a'*64, 'b'*64],
                                serving_live_inputs_verified=True,
                                output_storage_sha256=('d' if self._mode == 'inject' else 'c')*64)
        if self._mode == 'observer':
            for tensor in self.observation['proof']['tensors']:
                tensor['run_address'] = tensor['original_address']

    def snapshot(self):
        result = super().snapshot()
        if self._mode == 'observer':
            result['mode'] = 4
        return result


class ServingQualificationTests(unittest.TestCase):
    def setUp(self):
        self.native = ServingMock()
        self.identity = 'actual-mock-worker-load'
        self.q = ServingQualification({0: WorkerEndpoint(0, 0, self.native)}, owner='test',
                                      build_profile_sha256='e'*64, identity=lambda: self.identity)
        self.q.prepare()
        self.q.register(Fault39Config())
        self.q.start()
        self.sequence = 0

    def one(self, phase=None, *, output=None, corrupt=None):
        phase = phase or self.q.phase
        self.sequence += 1
        request_id = 'cpu-contract-' + str(self.sequence)
        self.q.begin(request_id, 'f'*64)
        self.native.request(request_id, self.q.owner)
        if corrupt:
            corrupt()
        self.q.end(output or ('d' if phase == 'fault' else 'c')*64, model_output_finite=True)
        self.assertFalse(self.native.snapshot()['enabled'])

    def test_nine_real_request_slots_one_fault_and_disabled_recovery(self):
        for phase in PHASES:
            self.one(phase)
        report = self.q.query()['qualification']
        self.assertEqual(report['completed_phases'], list(PHASES))
        self.assertEqual(report['status'], 'SCOPED_ATTEMPT_COMPLETED')
        self.assertTrue(report['recovery_verified'])
        self.assertTrue(report['model_output_changed'])
        self.assertEqual(self.native.configures.count(('inject', 3)), 1)
        self.q.unregister()
        with self.assertRaisesRegex(RuntimeError, 'SINGLE_UNIQUE'):
            self.q.start()

    def test_ambiguous_target_does_not_choose_a_layer(self):
        self.native._candidates[0]['candidate_count'] = 2
        with self.assertRaisesRegex(RuntimeError, 'AMBIGUOUS_OR_MISSING'):
            self.one()
        self.assertFalse(self.native.snapshot()['enabled'])
        self.assertNotIn(('inject', 3), self.native.configures)

    def test_changed_input_and_reused_real_id_refused(self):
        self.one()
        for request_id, digest, reason in [('cpu-contract-1', 'f'*64, 'FRESH_ACTUAL'),
                                          ('other', 'b'*64, 'INPUT_CHANGED')]:
            with self.assertRaisesRegex(RuntimeError, reason):
                self.q.begin(request_id, digest)
        self.assertFalse(self.native.snapshot()['enabled'])

    def test_missing_full_output_drains_active_discovery(self):
        self.q.begin('id1', 'f'*64)
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_MODEL_OUTPUT'):
            self.q.end(None, model_output_finite=True)
        self.assertFalse(self.native.snapshot()['enabled'])
        self.assertIsNotNone(self.q.failure)

    def test_changed_original_storage_refuses_before_fault(self):
        self.one()
        self.one()
        def corrupt():
            self.native.observation['original_storage_sha256_after'][0] = 'b'*64
        with self.assertRaisesRegex(RuntimeError, 'STORAGE_OR_RUN'):
            self.one(corrupt=corrupt)
        self.assertFalse(self.native.snapshot()['enabled'])
        self.assertNotIn(('inject', 3), self.native.configures)

    def test_failed_off_retains_owner(self):
        for phase in PHASES[:7]:
            self.one(phase)
        with self.assertRaisesRegex(RuntimeError, 'HEALTHY_COMPLETE_MODEL'):
            self.one(output='b'*64)
        with self.assertRaisesRegex(RuntimeError, 'OFF_RECOVERY_REQUIRED'):
            self.q.unregister()
        self.assertEqual(self.native.snapshot()['owner'], self.q.owner)

    def test_worker_replacement_never_arms(self):
        self.identity = 'new-model'
        with self.assertRaisesRegex(RuntimeError, 'WORKER_BUILD_OR_OWNER'):
            self.q.begin('id1', 'f'*64)
        self.assertEqual(self.native.configures, [])

    def test_inherited_enable_cannot_skip_qualification_or_reuse_attempt(self):
        for _ in range(2):
            self.one()
        with self.assertRaisesRegex(RuntimeError, 'HOST_DRIVEN'):
            self.q.enable()
        self.assertNotIn(('inject', 3), self.native.configures)

    def test_failed_off_can_only_run_explicit_recovery_and_stays_failed(self):
        for phase in PHASES[:7]:
            self.one(phase)
        with self.assertRaises(RuntimeError):
            self.one(output='b'*64)
        self.q.request_recovery()
        self.one('off_observed')
        self.one('off')
        report = self.q.query()['qualification']
        self.assertEqual(report['status'], 'RECOVERED_ATTEMPT_FAILED')
        self.assertTrue(report['recovery_verified'])
        self.assertEqual(self.native.configures.count(('inject', 3)), 1)
        self.q.unregister()

    def test_stable_nonfinite_baseline_is_not_healthy(self):
        self.q.begin('id1', 'f'*64)
        with self.assertRaisesRegex(RuntimeError, 'NOT_FINITE'):
            self.q.end('a'*64, model_output_finite=False)
        self.assertFalse(self.native.snapshot()['enabled'])

    def test_matched_phase_baseline_is_readonly_and_never_claims_injection(self):
        self.q = ServingQualification({0: WorkerEndpoint(0, 0, self.native)}, owner='baseline',
            build_profile_sha256='e'*64, identity=lambda: self.identity, baseline_control=True)
        self.q.prepare(); self.q.register(Fault39Config()); self.q.start()
        for phase in PHASES:
            self.one(phase, output='c'*64)
        report = self.q.query()['qualification']
        self.assertEqual(report['status'], 'BASELINE_CONTROL_COMPLETED')
        self.assertFalse(report['fault_started'])
        self.assertFalse(report['model_output_changed'])
        self.assertNotIn(('inject', 3), self.native.configures)
        self.assertEqual(self.q.rows[6]['native_verdict'], 'READONLY_COMPLETE_STORAGE_VERIFIED')
        self.q.unregister()

    def test_baseline_control_requires_unchanged_finite_comparison_phase(self):
        self.q = ServingQualification({0: WorkerEndpoint(0, 0, self.native)}, owner='baseline',
            build_profile_sha256='e'*64, identity=lambda: self.identity, baseline_control=True)
        self.q.prepare(); self.q.register(Fault39Config()); self.q.start()
        for phase in PHASES[:6]:
            self.one(phase, output='c'*64)
        with self.assertRaisesRegex(RuntimeError, 'HEALTHY_COMPLETE_MODEL'):
            self.one('fault', output='d'*64)
        self.assertFalse(self.native.snapshot()['enabled'])
        self.assertNotIn(('inject', 3), self.native.configures)


if __name__ == '__main__':
    unittest.main()
