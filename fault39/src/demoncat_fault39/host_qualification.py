"""Opt-in single-call gate for a host using the existing Fault39 ABI v1 backend."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import re

from .backend import NativeBackend
from .inspector import Fault39OutputInspector
from .schema import Fault39Config


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class ReplayObservation:
    """Readbacks from one fresh request through the reserved Worker.

    The host captures the actual internal scheduler ID and complete hashes.
    The two storage hashes are for the original self and mat2 before/after Run.
    """
    request_id: str
    input_sha256: str
    model_output_sha256: str
    storage_before: tuple[str, str]
    storage_after: tuple[str, str]


class QualifiedFault39Backend(NativeBackend):
    """Single Worker binding, isolated by the real host for each replay sequence.

    reserve_worker is a host context manager factory that prevents unrelated
    requests and reconfiguration on the dedicated Worker while held.
    worker_identity reads an immutable token for the actual Worker/model load.
    The host supplies replay(phase), including actual scheduler and data readback.
    """

    def __init__(self, workers, *, build_profile_sha256, reserve_worker, worker_identity,
                 replay, owner=None, binding_active=None):
        super().__init__(workers, owner=owner)
        if len(self.workers) != 1 or not is_sha256(build_profile_sha256):
            raise ValueError("MATCHED_SINGLE_WORKER_BUILD_REQUIRED")
        if not all(callable(item) for item in (reserve_worker, worker_identity, replay)):
            raise TypeError("HOST_EXCLUSIVE_WORKER_AND_REPLAY_REQUIRED")
        self._endpoint = next(iter(self.workers.values()))
        self._profile = build_profile_sha256
        self._reserve_worker = reserve_worker
        self._worker_identity = worker_identity
        self._replay = replay
        self._binding_active = binding_active
        self._identity = None
        self._reserved = False
        self._attempted = self._fault_started = False
        self._baseline = None
        self._requests = set()
        self._qualification = {"status": "PENDING", "recovery_verified": False}

    @contextmanager
    def _scope(self):
        # NativeBackend can call disable during configure rollback. Keep the
        # already held Worker reservation across that nested operation.
        if self._reserved:
            yield
            return
        with self._reserve_worker():
            self._reserved = True
            try:
                yield
            finally:
                self._reserved = False

    def prepare(self):
        with self._lock, self._scope():
            self._check_binding()
            identity = self._worker_identity()
            if not isinstance(identity, str) or not identity:
                raise RuntimeError("LIVE_WORKER_MODEL_IDENTITY_REQUIRED")
            if self._identity is not None and self._identity != identity:
                raise RuntimeError("WORKER_MODEL_IDENTITY_CHANGED")
            self._identity = identity
            self._snapshot()
            return super().prepare()

    def register(self, config):
        with self._lock:
            self._check_binding()
            if not isinstance(config, Fault39Config):
                raise TypeError("Fault39Config required")
            if (config.target_policy != "unique" or config.max_hits != 1
                    or config.ranks != (self._endpoint.rank,)):
                raise ValueError("UNIQUE_SINGLE_HIT_REQUIRED")
            super().register(config)
            self._attempted = self._fault_started = False
            self._baseline = None
            self._requests.clear()
            self._qualification = {"status": "PENDING", "recovery_verified": False}
            return self.query()

    def _check_binding(self):
        if self._binding_active is not None and not self._binding_active():
            raise RuntimeError("FAULT39_BINDING_NOT_ACTIVE")

    def _snapshot(self):
        current = self._endpoint.provider.snapshot()
        if (self._identity is None or self._worker_identity() != self._identity
                or type(current.get("abi_version")) is not int or current["abi_version"] != 1
                or current.get("runtime_loaded") is not True
                or current.get("runtime_identity") != "MATCHED_BUILD"
                or current.get("build_profile_sha256") != self._profile
                or current.get("worker_rank") != self._endpoint.rank
                or current.get("owner") not in ("", self.owner)):
            raise RuntimeError("LIVE_WORKER_BUILD_OR_OWNER_CHANGED")
        return current

    def _workload(self, phase):
        observation = self._replay(phase)
        if not isinstance(observation, ReplayObservation):
            raise RuntimeError("COMPLETE_REPLAY_OBSERVATION_REQUIRED")
        if (not isinstance(observation.request_id, str) or not observation.request_id
                or observation.request_id in self._requests
                or not is_sha256(observation.input_sha256)
                or not is_sha256(observation.model_output_sha256)
                or type(observation.storage_before) is not tuple or type(observation.storage_after) is not tuple
                or len(observation.storage_before) != 2
                or observation.storage_before != observation.storage_after
                or not all(is_sha256(value) for value in observation.storage_before)):
            raise RuntimeError("REQUEST_OR_ORIGINAL_STORAGE_PROOF_INVALID")
        self._requests.add(observation.request_id)
        if self._baseline is not None and (
            observation.input_sha256 != self._baseline.input_sha256
            or observation.storage_before != self._baseline.storage_before
            or (phase != "fault"
                and observation.model_output_sha256 != self._baseline.model_output_sha256)
        ):
            raise RuntimeError("REPLAY_BASELINE_MISMATCH: " + phase)
        return observation

    def _observe_run(self, phase, mask, mode):
        before = self._snapshot()
        replay = self._workload(phase)
        native = self._snapshot()
        observation = native.get("last_observation", {})
        proof = observation.get("proof", {})
        result = Fault39OutputInspector().inspect(native)
        allowed = {"REAL_OPERATOR_HIT_VERIFIED"}
        if phase == "fault":
            allowed.update(("MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED",
                            "MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED"))
        if (native.get("owner") != self.owner
                or native.get("generation") != self.generations[self._endpoint.rank]
                or native.get("rank") != self._endpoint.rank or native.get("device") != self._endpoint.device
                or native.get("target_signature") != self.targets[self._endpoint.rank]
                or native.get("mode") != mode or native.get("hits") != 1
                or native.get("interposed_run_calls", 0) <= before.get("interposed_run_calls", 0)
                or proof.get("api") != "aclnnMatmul" or proof.get("input_mask") != mask
                or json.loads(observation.get("actual_request_ids_json", "[]")) != [replay.request_id]
                or result["verdict"] not in allowed
                or (phase != "fault" and observation.get("changed") != [0, 0])
                or (phase == "fault" and not any(observation.get("changed", ())))):
            raise RuntimeError("NATIVE_REPLAY_EVIDENCE_INCOMPLETE: " + phase)
        if phase == "fault":
            self._qualification["model_output_changed"] = (
                replay.model_output_sha256 != self._baseline.model_output_sha256)
        return result["verdict"]

    def _drain(self, timeout_ms):
        drained = super().disable(timeout_ms)
        if drained["status"] != "DISABLED_CONFIRMED":
            raise RuntimeError(drained["status"])
        return drained

    def _recover(self, timeout_ms):
        drained = self._drain(timeout_ms)
        if self._fault_started and not self._qualification["recovery_verified"]:
            before = self._snapshot()
            self._workload("off")
            after = self._snapshot()
            if (after.get("enabled") is not False or after.get("owner") != self.owner
                    or after.get("generation") != before.get("generation")
                    or after.get("hits") != before.get("hits")
                    or after.get("inflight") != 0 or after.get("leases") != 0):
                raise RuntimeError("OFF_RECOVERY_PROOF_INCOMPLETE")
            self._qualification["recovery_verified"] = True
        return drained

    def enable(self, timeout_ms=10000):
        with self._lock:
            self._check_binding()
            if self.config is None or self._attempted:
                raise RuntimeError("SINGLE_REGISTERED_ATTEMPT_REQUIRED")
            self._attempted = True
            with self._scope():
                self._snapshot()
                try:
                    self.discover()
                    self._baseline = self._workload("discovery_1")
                    self.observe_inventory()
                    self._workload("discovery_2")
                    self.observe_inventory()
                    self._drain(timeout_ms)
                    for mask, phase in ((1, "lossless_self"), (2, "lossless_mat2"), (3, "lossless_dual")):
                        self._snapshot()
                        self._arm("lossless", input_mask=mask)
                        self._observe_run(phase, mask, mode=3)
                        self._drain(timeout_ms)
                    self._snapshot()
                    self._fault_started = True
                    self._arm("inject")
                    self._qualification["hit_verdict"] = self._observe_run("fault", 3, mode=2)
                except BaseException:
                    self._qualification["status"] = "ATTEMPT_FAILED"
                    raise
                finally:
                    try:
                        self._recover(timeout_ms)
                    except Exception:
                        self._qualification["status"] = "RECOVERY_PENDING"
                        raise
                self._qualification["status"] = "SCOPED_ATTEMPT_COMPLETED"
                return self.query()

    def disable(self, timeout_ms=10000):
        with self._lock, self._scope():
            return self._recover(timeout_ms)

    def clean(self, timeout_ms=10000):
        with self._lock, self._scope():
            if self._fault_started and not self._qualification["recovery_verified"]:
                raise RuntimeError("OFF_RECOVERY_REQUIRED_BEFORE_CLEAN")
            return super().clean(timeout_ms)

    def query(self):
        result = super().query()
        result["qualification"] = dict(self._qualification)
        return result
