"""CPU-only protocol doubles. Nothing here proves CANN execution or preload."""
from contextlib import contextmanager
from copy import deepcopy
import threading
import time
from .provider import NativeError


class MockNativeProvider:
    evidence_kind = "CPU_CONTRACT_MOCK"

    def __init__(self, rank=0, device=0, candidates=None):
        self.rank, self.device = rank, device
        self._condition = threading.Condition(threading.RLock())
        self._context = threading.local()
        self._owner = ""
        self._generation = 0
        self._enabled = False
        self._inflight = self._leases = self._hits = self._candidates_seen = 0
        self._mode = ""
        self._config = None
        self._status = "DISABLED"
        self._recovery_failed = False
        self._selected_rank = self._selected_device = -1
        self._candidates = deepcopy(candidates) if candidates is not None else [
            {"signature": f"mock:matmul:r{rank}:d{device}", "candidate_count": 1,
             "M": 1, "K": 12800, "N": 2, "rank": rank, "device": device, "observations": 0}]
        self.other_fault_owners = {}
        self.fail_next_configure = False

    def prepare(self):
        return self.snapshot()

    def bind_worker_rank(self, rank):
        if rank != self.rank:
            raise NativeError(-1, "bind_worker_rank")

    def configure(self, owner, config, rank, device, mode="inject", target_signature="", input_mask=3):
        with self._condition:
            if self.fail_next_configure:
                self.fail_next_configure = False
                raise NativeError(-3, "configure")
            if self._owner and self._owner != owner:
                raise NativeError(-4, "configure")
            if self._enabled or self._inflight or self._leases:
                raise NativeError(-3, "configure")
            if input_mask not in (1, 2, 3) or (mode != "lossless" and input_mask != 3):
                raise NativeError(-1, "configure")
            if mode != "discover":
                matches = [c for c in self._candidates if c["signature"] == target_signature]
                if not matches:
                    raise NativeError(-7, "configure")
                if len(matches) != 1 or matches[0]["candidate_count"] != 1:
                    raise NativeError(-6, "configure")
            self._owner, self._config = owner, config
            self._selected_rank, self._selected_device = rank, device
            self._generation += 1
            self._enabled = True
            self._hits = 0
            self._mode = mode
            if mode == "discover":
                for candidate in self._candidates:
                    candidate["observations"] = 0
            self._status = "DISCOVERING" if mode == "discover" else "ARMED"
            return self._generation

    def snapshot(self):
        with self._condition:
            return {"schema_version": 1, "owner": self._owner, "generation": self._generation,
                    "enabled": self._enabled, "inflight": self._inflight, "leases": self._leases,
                    "status": self._status, "mode": self._mode, "rank": self.rank, "device": self.device,
                    "candidates": deepcopy(self._candidates), "counters": {"hit_count": self._hits, "candidate_count": self._candidates_seen},
                    "runtime_loaded": False, "interposed_run_calls": 0, "typed_metadata_calls": 0,
                    "identity_source": "mock", "provider_evidence_kind": self.evidence_kind,
                    "other_fault_owners": deepcopy(self.other_fault_owners)}

    def _check_owner(self, owner, generation):
        if self._owner != owner:
            raise NativeError(-4, "stop")
        if self._generation != generation:
            raise NativeError(-5, "stop")

    def disable(self, owner, generation, timeout_ms=10000):
        with self._condition:
            self._check_owner(owner, generation)
            self._enabled = False
            deadline = time.monotonic() + timeout_ms / 1000
            while self._inflight or self._leases:
                if self._recovery_failed:
                    self._status = "RECOVERY_FAILED"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._status = "DISABLE_PENDING"
                    break
                self._condition.wait(remaining)
            else:
                self._status = "DISABLED_CONFIRMED"
            return {"status": self._status, "owner": owner, "generation": generation,
                    "native_code": 0 if self._status == "DISABLED_CONFIRMED" else -9 if self._recovery_failed else -10,
                    "snapshot": self.snapshot()}

    def clean(self, owner, generation, timeout_ms=10000):
        with self._condition:
            result = self.disable(owner, generation, timeout_ms)
            if result["status"] == "DISABLED_CONFIRMED":
                self._owner = ""
                self._config = None
                result["snapshot"] = self.snapshot()
            return result

    @contextmanager
    def context(self, owner, request_ids):
        if getattr(self._context, "active", None) is not None:
            raise RuntimeError("NESTED_REQUEST_CONTEXT_UNSUPPORTED")
        self._context.active = (owner, tuple(request_ids))
        try:
            yield
        finally:
            self._context.active = None

    def begin_call(self, rank=None, device=None):
        with self._condition:
            if not self._enabled:
                return None
            if self.rank != self._selected_rank or self.device != self._selected_device:
                return None
            if rank is not None and rank != self.rank or device is not None and device != self.device:
                return None
            self._candidates_seen += 1
            if self._mode == "discover":
                for candidate in self._candidates:
                    candidate["observations"] += 1
                return None
            context = getattr(self._context, "active", None)
            if context is None or context[0] != self._owner or not context[1]:
                self._status = "UNPROVEN_REQUEST_MAPPING"
                return None
            if self._hits >= self._config.max_hits:
                return None
            self._hits += 1
            self._inflight += 1
            self._leases += 2
            if self._hits >= self._config.max_hits:
                self._enabled = False
            return self._generation

    def complete_call(self, generation, recovery_failed=False):
        with self._condition:
            if generation != self._generation or self._inflight <= 0:
                raise RuntimeError("INVALID_CALL_LEASE")
            self._inflight -= 1
            if recovery_failed:
                self._recovery_failed = True
                self._status = "RECOVERY_FAILED"
            else:
                self._leases -= 2
            self._condition.notify_all()


class MockDemoncatHost:
    """Named mock host: deliberately cannot claim actual framework installation."""
    api = "demoncat_fault39.host_contract.v1"

    def __init__(self):
        self.catalog = {}
        self.bindings = {}

    def catalog_register_primitive(self, owner, definition):
        key = definition["primitive"]
        if key in self.catalog:
            raise RuntimeError("PRIMITIVE_ALREADY_REGISTERED")
        self.catalog[key] = (owner, deepcopy(definition))

    def catalog_unregister_primitive(self, owner, primitive):
        if self.catalog[primitive][0] != owner:
            raise RuntimeError("OWNER_MISMATCH")
        del self.catalog[primitive]

    def binding_register(self, owner, definition, factory):
        key = definition["binding"]
        if key in self.bindings:
            raise RuntimeError("BINDING_ALREADY_REGISTERED")
        self.bindings[key] = (owner, factory)

    def binding_unregister(self, owner, binding):
        if self.bindings[binding][0] != owner:
            raise RuntimeError("OWNER_MISMATCH")
        del self.bindings[binding]
