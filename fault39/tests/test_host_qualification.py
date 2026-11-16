"""CPU contract tests for the host gate, without loading a vendor library."""
from contextlib import contextmanager
import json
import unittest

from demoncat_fault39.backend import WorkerEndpoint
from demoncat_fault39.contracts import DocumentContractAdapter
from demoncat_fault39.host_qualification import QualifiedFault39Backend, ReplayObservation
from demoncat_fault39.inspector import summarize_values
from demoncat_fault39.schema import Fault39Config
from demoncat_fault39.testing import MockDemoncatHost, MockNativeProvider


class ReplayProvider(MockNativeProvider):
    def __init__(self):
        super().__init__(candidates=[
            dict(signature="test", candidate_count=1, M=1, K=2, N=2,
                 rank=0, device=0, observations=0),
        ])
        self.run_calls = 0
        self.configures = []
        self.observation = {}
        self.signature = ""

    def configure(self, owner, config, rank, device, mode="inject", target_signature="", input_mask=3):
        generation = super().configure(owner, config, rank, device, mode, target_signature, input_mask)
        self.configures.append((mode, input_mask))
        self.signature = target_signature
        self.observation = {}
        return generation

    def request(self, request_id, owner):
        self.run_calls += 1
        with self.context(owner, [request_id]):
            generation = self.begin_call()
            if generation is None:
                return
            mask = self.configures[-1][1]
            self.complete_call(generation)
            self._status = "COMPLETED"
            injected = self.configures[-1][0] == "inject"
            self.observation = dict(
                generation=generation, hit=1, owner=owner,
                proof=dict(api="aclnnMatmul", generation=generation, hit=1,
                           formal_order_proven=True, slots=[0, 1, 2], input_mask=mask,
                           tensors=[
                               dict(original_address=10, run_address=11 if mask & 1 else 10, shape=[1, 2]),
                               dict(original_address=20, run_address=21 if mask & 2 else 20, shape=[2, 2]),
                               dict(original_address=30, run_address=30, shape=[1, 2]),
                           ]),
                actual_request_ids_json=json.dumps([request_id]),
                pre_sync_status=0, copy_status=0, run_status=0, post_sync_status=0,
                free_status=0, output_source="post_run_device_readback",
                changed=[1, 1] if injected else [0, 0], unchanged=[1, 3] if injected else [2, 4],
                original_self=summarize_values([1, 2]),
                temporary_self=summarize_values([1, -2] if injected else [1, 2]),
                original_mat2=summarize_values([1, 2, 3, 4]),
                temporary_mat2=summarize_values([1, 2, 3, -4] if injected else [1, 2, 3, 4]),
                output=summarize_values([-5, 10] if injected else [7, 10]),
            )

    def snapshot(self):
        state = super().snapshot()
        mode = {"discover": 1, "inject": 2, "lossless": 3}.get(state["mode"], 0)
        return dict(state, abi_version=1, runtime_loaded=True, runtime_identity="MATCHED_BUILD",
                    build_profile_sha256="e" * 64, worker_rank=0, mode=mode,
                    target_signature=self.signature, hits=self._hits, interposed_run_calls=self.run_calls,
                    typed_metadata_calls=self.run_calls, last_error=0, last_observation=self.observation)


class HostQualificationTest(unittest.TestCase):
    def setUp(self):
        self.provider = ReplayProvider()
        self.worker = {0: WorkerEndpoint(0, 0, self.provider)}
        self.reserved = False
        self.identity = "worker-1/model-load-1"
        self.changed_phase = None
        self.phases = []
        self.host = MockDemoncatHost()
        self.adapter = DocumentContractAdapter(self.host, qualified=True)
        self.adapter.install()
        self.factory = self.host.bindings["fault39.native.matmul"][1]
        self.backend = self.factory(
            self.worker, build_profile_sha256="e" * 64, reserve_worker=self.reserve,
            worker_identity=lambda: self.identity, replay=self.replay,
        )
        self.backend.prepare()
        self.backend.register(Fault39Config())

    @contextmanager
    def reserve(self):
        self.assertFalse(self.reserved)
        self.reserved = True
        try:
            yield
        finally:
            self.reserved = False

    def replay(self, phase):
        self.assertTrue(self.reserved)
        self.phases.append(phase)
        request = f"request-{len(self.phases)}"
        self.provider.request(request, self.backend.owner)
        storage = ("a" * 64, "b" * 64)
        output = "d" if phase in ("fault", self.changed_phase) else "c"
        return ReplayObservation(request, "e" * 64, output * 64, storage, storage)

    def test_disarmed_registration_and_single_attempt_off(self):
        self.assertFalse(self.provider.snapshot()["enabled"])
        self.assertIsInstance(self.backend, QualifiedFault39Backend)
        result = self.backend.enable()
        self.assertEqual(self.phases, ["discovery_1", "discovery_2", "lossless_self",
                                      "lossless_mat2", "lossless_dual", "fault", "off"])
        self.assertEqual(self.provider.configures,
                         [("discover", 3), ("lossless", 1), ("lossless", 2), ("lossless", 3), ("inject", 3)])
        self.assertTrue(result["qualification"]["recovery_verified"])
        self.assertTrue(result["qualification"]["model_output_changed"])
        self.assertFalse(result["workers"][0]["enabled"])
        with self.assertRaisesRegex(RuntimeError, "SINGLE_REGISTERED_ATTEMPT_REQUIRED"):
            self.backend.enable()
        with self.assertRaisesRegex(RuntimeError, "OWNED_BACKEND_CLEAN_REQUIRED"):
            self.adapter.uninstall()
        self.backend.unregister()
        self.adapter.uninstall()
        with self.assertRaisesRegex(RuntimeError, "FAULT39_BINDING_NOT_ACTIVE"):
            self.backend.register(Fault39Config())

    def test_failed_lossless_never_arms_fault(self):
        self.changed_phase = "lossless_mat2"
        with self.assertRaisesRegex(RuntimeError, "REPLAY_BASELINE_MISMATCH"):
            self.backend.enable()
        self.assertFalse(self.provider.snapshot()["enabled"])
        self.assertNotIn(("inject", 3), self.provider.configures)

    def test_worker_replacement_refused_before_discovery(self):
        self.identity = "worker-2/model-load-2"
        with self.assertRaisesRegex(RuntimeError, "LIVE_WORKER_BUILD_OR_OWNER_CHANGED"):
            self.backend.enable()
        self.assertEqual(self.provider.configures, [])

    def test_failed_off_preserves_owner_for_recovery(self):
        self.changed_phase = "off"
        with self.assertRaisesRegex(RuntimeError, "REPLAY_BASELINE_MISMATCH"):
            self.backend.enable()
        self.assertFalse(self.provider.snapshot()["enabled"])
        with self.assertRaisesRegex(RuntimeError, "OFF_RECOVERY_REQUIRED_BEFORE_CLEAN"):
            self.backend.unregister()
        self.assertEqual(self.provider.snapshot()["owner"], self.backend.owner)
        self.changed_phase = None
        self.backend.disable()
        self.assertTrue(self.backend.query()["qualification"]["recovery_verified"])
        self.backend.unregister()
